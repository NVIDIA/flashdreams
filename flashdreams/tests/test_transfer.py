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
    describe_tensor_bundle,
)

pytestmark = pytest.mark.ci_cpu


class _FakeTransferEngine:
    def __init__(self) -> None:
        self.registered: set[int] = set()

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
