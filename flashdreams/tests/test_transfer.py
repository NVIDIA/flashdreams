# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CPU tests for Mooncake-backed tensor-bundle transfer."""

from __future__ import annotations

import ctypes

import pytest
import torch

from flashdreams.infra.transfer import (
    MooncakeTensorTransport,
    NixlTensorTransport,
    RegisteredTensorPool,
    describe_tensor_bundle,
)

pytestmark = pytest.mark.ci_cpu


class _FakeTransferEngine:
    def __init__(self) -> None:
        self.registered: set[int] = set()
        self.registration_calls = 0
        self._next_batch_id = 1

    def initialize(
        self,
        hostname: str,
        metadata: str,
        protocol: str,
        device_name: str,
    ) -> int:
        assert hostname == "encoder"
        assert metadata == "P2PHANDSHAKE"
        assert protocol == "rdma"
        assert device_name == "mlx5_0"
        return 0

    def get_rpc_port(self) -> int:
        return 12345

    def register_memory(self, address: int, length: int) -> int:
        assert length > 0
        self.registered.add(address)
        self.registration_calls += 1
        return 0

    def unregister_memory(self, address: int) -> int:
        self.registered.remove(address)
        return 0

    def batch_transfer_sync_write(
        self,
        session_id: str,
        sources: list[int],
        destinations: list[int],
        lengths: list[int],
    ) -> int:
        assert session_id == "encoder:12345"
        for source, destination, length in zip(
            sources, destinations, lengths, strict=True
        ):
            ctypes.memmove(destination, source, length)
        return 0

    def batch_transfer_async_write(
        self,
        session_id: str,
        sources: list[int],
        destinations: list[int],
        lengths: list[int],
    ) -> int:
        assert session_id == "encoder:12345"
        for source, destination, length in zip(
            sources, destinations, lengths, strict=True
        ):
            ctypes.memmove(destination, source, length)
        batch_id = self._next_batch_id
        self._next_batch_id += 1
        return batch_id

    def get_batch_transfer_status(self, batch_ids: list[int]) -> int:
        assert batch_ids
        return 0


class _FakeNixlAgent:
    agents: dict[str, "_FakeNixlAgent"] = {}

    def __init__(self, name: str, config: object) -> None:
        del config
        self.name = name
        self.registrations: dict[int, torch.Tensor] = {}
        self.remote_agents: set[str] = set()
        self.handles: dict[int, tuple[list[torch.Tensor], list[torch.Tensor]]] = {}
        self._next_handle = 1
        self.agents[name] = self

    def register_memory(self, tensor: torch.Tensor) -> tuple[int]:
        self.registrations[tensor.data_ptr()] = tensor
        return (tensor.data_ptr(),)

    def deregister_memory(self, registration: tuple[int]) -> None:
        self.registrations.pop(registration[0])

    def get_xfer_descs(self, tensors: list[torch.Tensor]) -> list[torch.Tensor]:
        return tensors

    def get_serialized_descs(self, tensors: list[torch.Tensor]) -> list[int]:
        return [tensor.data_ptr() for tensor in tensors]

    def deserialize_descs(self, addresses: list[int]) -> list[torch.Tensor]:
        tensors: list[torch.Tensor] = []
        for agent in self.agents.values():
            for address in addresses:
                if address in agent.registrations:
                    tensors.append(agent.registrations[address])
        return tensors

    def get_agent_metadata(self) -> bytes:
        return self.name.encode()

    def add_remote_agent(self, metadata: bytes) -> str:
        name = metadata.decode()
        self.remote_agents.add(name)
        return name

    def remove_remote_agent(self, name: str) -> None:
        self.remote_agents.remove(name)

    def initialize_xfer(
        self,
        operation: str,
        local: list[torch.Tensor],
        remote: list[torch.Tensor],
        remote_agent: str,
    ) -> int:
        assert operation == "WRITE"
        assert remote_agent in self.remote_agents
        handle = self._next_handle
        self._next_handle += 1
        self.handles[handle] = (local, remote)
        return handle

    def transfer(self, handle: int) -> str:
        local, remote = self.handles[handle]
        for source, destination in zip(local, remote, strict=True):
            destination.copy_(source)
        return "IN_PROGRESS"

    def check_xfer_state(self, handle: int) -> str:
        assert handle in self.handles
        return "DONE"

    def release_xfer_handle(self, handle: int) -> None:
        self.handles.pop(handle)


def _transport() -> MooncakeTensorTransport:
    return MooncakeTensorTransport(
        hostname="encoder",
        device_name="mlx5_0",
        engine_factory=_FakeTransferEngine,
    )


def test_mooncake_transfer_copies_bundle_and_reports_bandwidth() -> None:
    sender = _transport()
    receiver = _transport()
    source = {
        "context": torch.arange(8, dtype=torch.float32),
        "mask": torch.ones(2, dtype=torch.int64),
    }
    destination = receiver.allocate(
        describe_tensor_bundle(source),
        device=torch.device("cpu"),
    )
    stats = sender.send(source, receiver.make_ticket(destination))

    torch.testing.assert_close(destination["context"], source["context"])
    torch.testing.assert_close(destination["mask"], source["mask"])
    assert stats.backend == "mooncake-rdma"
    assert stats.payload_bytes == 48
    assert stats.transfer_ms >= 0.0
    assert stats.registration_ms >= 0.0
    assert stats.bandwidth_gbps > 0.0
    assert stats.asynchronous
    sender.unregister(source)
    receiver.unregister(destination)
    sender.close()
    receiver.close()


def test_transfer_rejects_noncontiguous_and_mismatched_bundles() -> None:
    transport = _transport()
    with pytest.raises(ValueError, match="must be contiguous"):
        describe_tensor_bundle({"x": torch.ones(2, 3).T})

    receiver = transport.allocate(
        describe_tensor_bundle({"x": torch.ones(2)}),
        device=torch.device("cpu"),
    )
    ticket = transport.make_ticket(receiver)
    with pytest.raises(ValueError, match="does not match"):
        transport.send({"x": torch.ones(3)}, ticket)
    transport.close()


def test_registered_pool_reuses_receiver_registration() -> None:
    transport = _transport()
    pool = RegisteredTensorPool(transport, max_buffers_per_bucket=1)
    descriptors = describe_tensor_bundle({"x": torch.ones(4)})

    first = pool.acquire(descriptors, device=torch.device("cpu"))
    registration_calls = transport.engine.registration_calls
    pool.release(first)
    second = pool.acquire(descriptors, device=torch.device("cpu"))

    assert second is first
    assert transport.engine.registration_calls == registration_calls
    pool.release(second)
    pool.close()
    transport.close()


def test_nixl_transfer_uses_serialized_receiver_descriptors() -> None:
    _FakeNixlAgent.agents.clear()
    sender = NixlTensorTransport(
        agent_name="sender",
        agent_factory=_FakeNixlAgent,
        config_factory=lambda *_: object(),
    )
    receiver = NixlTensorTransport(
        agent_name="receiver",
        agent_factory=_FakeNixlAgent,
        config_factory=lambda *_: object(),
    )
    source = {"x": torch.arange(8, dtype=torch.float32)}
    destination = receiver.allocate(
        describe_tensor_bundle(source),
        device=torch.device("cpu"),
    )

    stats = sender.send(source, receiver.make_ticket(destination))

    torch.testing.assert_close(destination["x"], source["x"])
    assert stats.backend == "nixl"
    assert stats.asynchronous
    sender.close()
    receiver.close()
