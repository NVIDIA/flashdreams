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

"""Registered GPU tensor transfer through Mooncake or NIXL."""

from __future__ import annotations

import importlib
import socket
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

TensorBundle = dict[str, Tensor]
"""Flat, name-keyed tensor payload transferred between pipeline stages."""


@dataclass(frozen=True, kw_only=True)
class TensorDescriptor:
    """Shape and storage metadata for one transferred tensor."""

    name: str
    """Stable field name inside the tensor bundle."""

    shape: tuple[int, ...]
    """Tensor shape."""

    dtype: torch.dtype
    """Tensor element type."""

    nbytes: int
    """Contiguous payload size in bytes."""


@dataclass(frozen=True, kw_only=True)
class TensorTransferTicket:
    """Receiver-owned registered memory advertised to a sender."""

    session_id: str
    """Backend peer or agent identifier."""

    descriptors: tuple[TensorDescriptor, ...]
    """Expected bundle layout in transfer order."""

    addresses: tuple[int, ...] = ()
    """Registered receiver addresses used by address-based transports."""

    metadata: bytes | None = None
    """Serialized receiver-agent metadata used by NIXL."""

    remote_descriptors: bytes | None = None
    """Serialized receiver transfer descriptors used by NIXL."""


@dataclass(kw_only=True)
class TensorTransferHandle:
    """One submitted tensor transfer that may still be in flight."""

    backend: str
    """Selected transfer backend."""

    payload_bytes: int
    """Total tensor bytes in the operation."""

    registration_ms: float
    """Sender memory-registration time for previously unseen allocations."""

    submit_ms: float
    """Host time spent submitting the transfer."""

    submitted_at: float
    """Monotonic timestamp immediately before backend submission."""

    opaque_handle: Any = None
    """Backend-specific asynchronous operation handle."""

    source_bundle: Mapping[str, Tensor] | None = None
    """Strong reference preserving source storage until completion."""

    completed_stats: TransferStats | None = None
    """Immediate result when the backend used a synchronous fallback."""


@dataclass(frozen=True, kw_only=True)
class TransferStats:
    """Observed metrics for one tensor-bundle transfer."""

    backend: str
    """Selected transfer backend and protocol."""

    payload_bytes: int
    """Total tensor bytes transferred."""

    registration_ms: float
    """Sender memory-registration time for previously unseen allocations."""

    transfer_ms: float
    """Submission-to-completion-observation wall time.

    For asynchronous operations this includes any useful work performed before
    the caller invokes ``wait``; it is an in-flight residency window, not an
    isolated copy-time measurement.
    """

    bandwidth_gbps: float
    """Effective decimal GB/s computed from payload bytes and transfer time."""

    submit_ms: float = 0.0
    """Host time spent submitting the operation."""

    wait_ms: float = 0.0
    """Host time spent waiting after submission."""

    asynchronous: bool = False
    """Whether the backend used its asynchronous operation API."""


@dataclass(frozen=True, kw_only=True)
class PooledTensorBuffer:
    """One reusable registered receiver bundle and its transfer ticket."""

    bundle: TensorBundle
    """Registered receiver tensors."""

    ticket: TensorTransferTicket
    """Stable ticket advertising the registered tensors."""

    bucket: tuple[tuple[TensorDescriptor, ...], str]
    """Pool bucket key used when returning this buffer."""


def describe_tensor_bundle(
    bundle: Mapping[str, Tensor],
) -> tuple[TensorDescriptor, ...]:
    """Describe a contiguous tensor bundle in stable insertion order."""
    descriptors: list[TensorDescriptor] = []
    for name, tensor in bundle.items():
        if not tensor.is_contiguous():
            raise ValueError(f"Tensor bundle field {name!r} must be contiguous.")
        descriptors.append(
            TensorDescriptor(
                name=name,
                shape=tuple(tensor.shape),
                dtype=tensor.dtype,
                nbytes=tensor.numel() * tensor.element_size(),
            )
        )
    return tuple(descriptors)


class MooncakeTensorTransport:
    """Move tensor bundles directly between registered GPU buffers."""

    def __init__(
        self,
        *,
        hostname: str | None = None,
        device_name: str | None = None,
        protocol: str = "rdma",
        engine_factory: Callable[[], Any] | None = None,
    ) -> None:
        """Initialize a Mooncake endpoint.

        Args:
            hostname: Routable address advertised to peers. Defaults to the
                local hostname.
            device_name: Optional RDMA device selection passed to Mooncake.
            protocol: Mooncake transport protocol. Production disaggregation
                should use ``"rdma"``.
            engine_factory: Test hook returning a Transfer Engine-compatible
                object. ``None`` imports ``mooncake.engine.TransferEngine``.

        Raises:
            ImportError: Mooncake's Python package is unavailable.
            RuntimeError: The transfer endpoint fails to initialize.
        """
        if engine_factory is None:
            try:
                from mooncake.engine import TransferEngine
            except ImportError as error:
                raise ImportError(
                    "Mooncake transfer support requires the CUDA 13 package: "
                    "pip install mooncake-transfer-engine-cuda13."
                ) from error
            engine_factory = TransferEngine

        self.hostname = hostname or socket.gethostname()
        self.protocol = protocol
        self.engine = engine_factory()
        result = self.engine.initialize(
            self.hostname,
            "P2PHANDSHAKE",
            protocol,
            device_name or "",
        )
        if result != 0:
            raise RuntimeError(
                f"Mooncake Transfer Engine initialization failed with status {result}."
            )
        self.session_id = f"{self.hostname}:{self.engine.get_rpc_port()}"
        self._registered_addresses: set[int] = set()
        self._registered_tensors: dict[int, Tensor] = {}

    @property
    def backend(self) -> str:
        """Return the backend label recorded in benchmark output."""
        return f"mooncake-{self.protocol}"

    def allocate(
        self,
        descriptors: tuple[TensorDescriptor, ...],
        *,
        device: torch.device,
    ) -> TensorBundle:
        """Allocate and register receiver VRAM for a described bundle."""
        bundle: TensorBundle = {}
        for descriptor in descriptors:
            tensor = torch.empty(
                descriptor.shape,
                dtype=descriptor.dtype,
                device=device,
            ).contiguous()
            actual_nbytes = tensor.numel() * tensor.element_size()
            if actual_nbytes != descriptor.nbytes:
                raise ValueError(
                    f"Descriptor size mismatch for {descriptor.name!r}: expected "
                    f"{descriptor.nbytes}, allocated {actual_nbytes}."
                )
            result = self.engine.register_memory(tensor.data_ptr(), actual_nbytes)
            if result != 0:
                raise RuntimeError(
                    f"Mooncake failed to register {descriptor.name!r} "
                    f"({actual_nbytes} bytes), status={result}."
                )
            self._registered_addresses.add(tensor.data_ptr())
            self._registered_tensors[tensor.data_ptr()] = tensor
            bundle[descriptor.name] = tensor
        return bundle

    def make_ticket(self, bundle: Mapping[str, Tensor]) -> TensorTransferTicket:
        """Advertise registered receiver buffers to a sender."""
        descriptors = describe_tensor_bundle(bundle)
        missing = [
            descriptor.name
            for descriptor in descriptors
            if bundle[descriptor.name].data_ptr() not in self._registered_addresses
        ]
        if missing:
            raise ValueError(f"Receiver buffers are not registered: {missing}.")
        return TensorTransferTicket(
            session_id=self.session_id,
            descriptors=descriptors,
            addresses=tuple(bundle[item.name].data_ptr() for item in descriptors),
        )

    def register(self, bundle: Mapping[str, Tensor]) -> float:
        """Register unseen bundle allocations and return elapsed milliseconds."""
        started = time.perf_counter()
        for descriptor in describe_tensor_bundle(bundle):
            tensor = bundle[descriptor.name]
            address = tensor.data_ptr()
            if address in self._registered_addresses:
                continue
            result = self.engine.register_memory(address, descriptor.nbytes)
            if result != 0:
                raise RuntimeError(
                    f"Mooncake failed to register {descriptor.name!r} "
                    f"({descriptor.nbytes} bytes), status={result}."
                )
            self._registered_addresses.add(address)
            self._registered_tensors[address] = tensor
        return (time.perf_counter() - started) * 1000.0

    def unregister(self, bundle: Mapping[str, Tensor]) -> None:
        """Unregister bundle allocations before their tensors are released."""
        for tensor in bundle.values():
            address = tensor.data_ptr()
            if address not in self._registered_addresses:
                continue
            result = self.engine.unregister_memory(address)
            if result != 0:
                raise RuntimeError(
                    f"Mooncake failed to unregister address {address}, status={result}."
                )
            self._registered_addresses.remove(address)
            self._registered_tensors.pop(address, None)

    @staticmethod
    def _wait_until_source_ready(bundle: Mapping[str, Tensor]) -> None:
        """Wait only for the CUDA producer stream instead of the whole device."""
        cuda_tensors = [tensor for tensor in bundle.values() if tensor.is_cuda]
        if not cuda_tensors:
            return
        devices = {tensor.device for tensor in cuda_tensors}
        if len(devices) != 1:
            raise ValueError("A tensor bundle must reside on one CUDA device.")
        event = torch.cuda.Event()
        event.record(torch.cuda.current_stream(cuda_tensors[0].device))
        event.synchronize()

    def send_async(
        self,
        bundle: Mapping[str, Tensor],
        ticket: TensorTransferTicket,
    ) -> TensorTransferHandle:
        """Submit a tensor-bundle write and return without waiting for RDMA."""
        descriptors = describe_tensor_bundle(bundle)
        if descriptors != ticket.descriptors:
            raise ValueError(
                "Sender bundle layout does not match the receiver transfer ticket."
            )
        if not ticket.addresses:
            raise ValueError("Mooncake transfer tickets require receiver addresses.")

        registration_ms = self.register(bundle)
        self._wait_until_source_ready(bundle)
        sources = [bundle[item.name].data_ptr() for item in descriptors]
        lengths = [item.nbytes for item in descriptors]
        payload_bytes = sum(lengths)
        submitted_at = time.perf_counter()

        async_write = getattr(self.engine, "batch_transfer_async_write", None)
        if callable(async_write):
            batch_id = async_write(
                ticket.session_id,
                sources,
                list(ticket.addresses),
                lengths,
            )
            submit_ms = (time.perf_counter() - submitted_at) * 1000.0
            if not isinstance(batch_id, int) or batch_id <= 0:
                raise RuntimeError(
                    f"Mooncake async tensor transfer submission failed: {batch_id}."
                )
            return TensorTransferHandle(
                backend=self.backend,
                payload_bytes=payload_bytes,
                registration_ms=registration_ms,
                submit_ms=submit_ms,
                submitted_at=submitted_at,
                opaque_handle=batch_id,
                source_bundle=bundle,
            )

        result = self.engine.batch_transfer_sync_write(
            ticket.session_id,
            sources,
            list(ticket.addresses),
            lengths,
        )
        elapsed_s = time.perf_counter() - submitted_at
        if result != 0:
            raise RuntimeError(f"Mooncake tensor transfer failed with status {result}.")
        stats = TransferStats(
            backend=self.backend,
            payload_bytes=payload_bytes,
            registration_ms=registration_ms,
            transfer_ms=elapsed_s * 1000.0,
            bandwidth_gbps=(
                payload_bytes / elapsed_s / 1e9 if elapsed_s > 0.0 else float("inf")
            ),
            submit_ms=elapsed_s * 1000.0,
            asynchronous=False,
        )
        return TensorTransferHandle(
            backend=self.backend,
            payload_bytes=payload_bytes,
            registration_ms=registration_ms,
            submit_ms=stats.submit_ms,
            submitted_at=submitted_at,
            source_bundle=bundle,
            completed_stats=stats,
        )

    def wait(self, handle: TensorTransferHandle) -> TransferStats:
        """Wait for a submitted Mooncake transfer and return its metrics."""
        if handle.completed_stats is not None:
            return handle.completed_stats
        status = self.engine.get_batch_transfer_status([handle.opaque_handle])
        wait_finished = time.perf_counter()
        if status != 0:
            raise RuntimeError(
                f"Mooncake async tensor transfer failed with status {status}."
            )
        elapsed_s = wait_finished - handle.submitted_at
        wait_ms = max(0.0, elapsed_s * 1000.0 - handle.submit_ms)
        return TransferStats(
            backend=self.backend,
            payload_bytes=handle.payload_bytes,
            registration_ms=handle.registration_ms,
            transfer_ms=elapsed_s * 1000.0,
            bandwidth_gbps=(
                handle.payload_bytes / elapsed_s / 1e9
                if elapsed_s > 0.0
                else float("inf")
            ),
            submit_ms=handle.submit_ms,
            wait_ms=wait_ms,
            asynchronous=True,
        )

    def send(
        self,
        bundle: Mapping[str, Tensor],
        ticket: TensorTransferTicket,
    ) -> TransferStats:
        """Write a tensor bundle and wait for completion."""
        return self.wait(self.send_async(bundle, ticket))

    def close(self) -> None:
        """Unregister every receiver allocation owned by this endpoint."""
        for address in tuple(self._registered_addresses):
            result = self.engine.unregister_memory(address)
            if result != 0:
                raise RuntimeError(
                    f"Mooncake failed to unregister address {address}, status={result}."
                )
            self._registered_addresses.remove(address)
            self._registered_tensors.pop(address, None)


class NixlTensorTransport:
    """Move registered tensor bundles through the NIXL Python API."""

    def __init__(
        self,
        *,
        agent_name: str | None = None,
        agent_factory: Callable[[str, Any], Any] | None = None,
        config_factory: Callable[..., Any] | None = None,
    ) -> None:
        """Initialize a NIXL transfer agent.

        Args:
            agent_name: Unique agent name. Defaults to hostname plus process ID.
            agent_factory: Test hook compatible with ``nixl_agent``.
            config_factory: Test hook compatible with ``nixl_agent_config``.

        Raises:
            ImportError: The NIXL Python package is unavailable.
        """
        if agent_factory is None or config_factory is None:
            try:
                nixl_module = importlib.import_module("nixl")
            except ImportError as error:
                raise ImportError(
                    "NIXL transfer support requires the optional package: "
                    "pip install nixl."
                ) from error
            if not hasattr(nixl_module, "nixl_agent"):
                nixl_module = importlib.import_module("nixl._api")
            agent_factory = agent_factory or getattr(nixl_module, "nixl_agent")
            config_factory = config_factory or getattr(
                nixl_module,
                "nixl_agent_config",
            )

        import os

        self.session_id = agent_name or f"{socket.gethostname()}-{os.getpid()}"
        self.agent = agent_factory(
            self.session_id,
            config_factory(True, True, 0),
        )
        self._registrations: dict[int, Any] = {}
        self._registered_tensors: dict[int, Tensor] = {}
        self._remote_agents: set[str] = set()

    @property
    def backend(self) -> str:
        """Return the backend label recorded in benchmark output."""
        return "nixl"

    def allocate(
        self,
        descriptors: tuple[TensorDescriptor, ...],
        *,
        device: torch.device,
    ) -> TensorBundle:
        """Allocate and register receiver memory for a described bundle."""
        bundle = {
            descriptor.name: torch.empty(
                descriptor.shape,
                dtype=descriptor.dtype,
                device=device,
            ).contiguous()
            for descriptor in descriptors
        }
        self.register(bundle)
        return bundle

    def register(self, bundle: Mapping[str, Tensor]) -> float:
        """Register unseen bundle allocations and return elapsed milliseconds."""
        started = time.perf_counter()
        for tensor in bundle.values():
            address = tensor.data_ptr()
            if address in self._registrations:
                continue
            registration = self.agent.register_memory(tensor)
            if not registration:
                raise RuntimeError(
                    f"NIXL failed to register tensor at address {address}."
                )
            self._registrations[address] = registration
            self._registered_tensors[address] = tensor
        return (time.perf_counter() - started) * 1000.0

    def make_ticket(self, bundle: Mapping[str, Tensor]) -> TensorTransferTicket:
        """Serialize receiver metadata and tensor descriptors for a sender."""
        descriptors = describe_tensor_bundle(bundle)
        missing = [
            item.name
            for item in descriptors
            if bundle[item.name].data_ptr() not in self._registrations
        ]
        if missing:
            raise ValueError(f"Receiver buffers are not registered: {missing}.")
        remote = self.agent.get_xfer_descs([bundle[item.name] for item in descriptors])
        if not remote:
            raise RuntimeError("NIXL failed to create receiver transfer descriptors.")
        return TensorTransferTicket(
            session_id=self.session_id,
            descriptors=descriptors,
            metadata=self.agent.get_agent_metadata(),
            remote_descriptors=self.agent.get_serialized_descs(remote),
        )

    def send_async(
        self,
        bundle: Mapping[str, Tensor],
        ticket: TensorTransferTicket,
    ) -> TensorTransferHandle:
        """Submit a NIXL write and return its asynchronous handle."""
        descriptors = describe_tensor_bundle(bundle)
        if descriptors != ticket.descriptors:
            raise ValueError(
                "Sender bundle layout does not match the receiver transfer ticket."
            )
        if ticket.metadata is None or ticket.remote_descriptors is None:
            raise ValueError("NIXL tickets require agent metadata and descriptors.")
        registration_ms = self.register(bundle)
        MooncakeTensorTransport._wait_until_source_ready(bundle)
        if ticket.session_id not in self._remote_agents:
            loaded_name = self.agent.add_remote_agent(ticket.metadata)
            if loaded_name != ticket.session_id:
                raise RuntimeError(
                    f"NIXL ticket names {ticket.session_id!r}, metadata loaded "
                    f"{loaded_name!r}."
                )
            self._remote_agents.add(loaded_name)

        local = self.agent.get_xfer_descs([bundle[item.name] for item in descriptors])
        remote = self.agent.deserialize_descs(ticket.remote_descriptors)
        xfer = self.agent.initialize_xfer(
            "WRITE",
            local,
            remote,
            ticket.session_id,
        )
        submitted_at = time.perf_counter()
        state = self.agent.transfer(xfer)
        submit_ms = (time.perf_counter() - submitted_at) * 1000.0
        if state == "ERR":
            self.agent.release_xfer_handle(xfer)
            raise RuntimeError("NIXL tensor transfer submission failed.")
        return TensorTransferHandle(
            backend=self.backend,
            payload_bytes=sum(item.nbytes for item in descriptors),
            registration_ms=registration_ms,
            submit_ms=submit_ms,
            submitted_at=submitted_at,
            opaque_handle=xfer,
            source_bundle=bundle,
        )

    def wait(
        self,
        handle: TensorTransferHandle,
        *,
        timeout_s: float = 60.0,
    ) -> TransferStats:
        """Wait for a submitted NIXL transfer and return its metrics."""
        deadline = time.monotonic() + timeout_s
        while True:
            state = self.agent.check_xfer_state(handle.opaque_handle)
            if state == "DONE":
                break
            if state == "ERR":
                self.agent.release_xfer_handle(handle.opaque_handle)
                raise RuntimeError("NIXL tensor transfer failed.")
            if time.monotonic() >= deadline:
                self.agent.release_xfer_handle(handle.opaque_handle)
                raise TimeoutError(
                    f"NIXL tensor transfer exceeded {timeout_s:.1f} seconds."
                )
            time.sleep(0)
        finished = time.perf_counter()
        self.agent.release_xfer_handle(handle.opaque_handle)
        elapsed_s = finished - handle.submitted_at
        return TransferStats(
            backend=self.backend,
            payload_bytes=handle.payload_bytes,
            registration_ms=handle.registration_ms,
            transfer_ms=elapsed_s * 1000.0,
            bandwidth_gbps=(
                handle.payload_bytes / elapsed_s / 1e9
                if elapsed_s > 0.0
                else float("inf")
            ),
            submit_ms=handle.submit_ms,
            wait_ms=max(0.0, elapsed_s * 1000.0 - handle.submit_ms),
            asynchronous=True,
        )

    def send(
        self,
        bundle: Mapping[str, Tensor],
        ticket: TensorTransferTicket,
    ) -> TransferStats:
        """Write a tensor bundle and wait for completion."""
        return self.wait(self.send_async(bundle, ticket))

    def unregister(self, bundle: Mapping[str, Tensor]) -> None:
        """Deregister bundle allocations before their tensors are released."""
        for tensor in bundle.values():
            address = tensor.data_ptr()
            registration = self._registrations.pop(address, None)
            if registration is None:
                continue
            self.agent.deregister_memory(registration)
            self._registered_tensors.pop(address, None)

    def close(self) -> None:
        """Release remote metadata and every local memory registration."""
        for remote_agent in tuple(self._remote_agents):
            self.agent.remove_remote_agent(remote_agent)
            self._remote_agents.remove(remote_agent)
        for address, registration in tuple(self._registrations.items()):
            self.agent.deregister_memory(registration)
            self._registrations.pop(address)
            self._registered_tensors.pop(address, None)


class RegisteredTensorPool:
    """Reuse fixed-shape registered receiver buffers across transfers."""

    def __init__(self, transport: Any, *, max_buffers_per_bucket: int = 2) -> None:
        """Initialize a registered buffer pool.

        Args:
            transport: Tensor transport providing ``allocate`` and ``make_ticket``.
            max_buffers_per_bucket: Maximum simultaneous leases for one shape bucket.
        """
        if max_buffers_per_bucket < 1:
            raise ValueError("max_buffers_per_bucket must be positive.")
        self.transport = transport
        self.max_buffers_per_bucket = max_buffers_per_bucket
        self._available: dict[
            tuple[tuple[TensorDescriptor, ...], str], list[PooledTensorBuffer]
        ] = {}
        self._allocated: dict[
            tuple[tuple[TensorDescriptor, ...], str], list[PooledTensorBuffer]
        ] = {}
        self._leased: set[int] = set()
        self._lock = threading.Lock()

    def acquire(
        self,
        descriptors: tuple[TensorDescriptor, ...],
        *,
        device: torch.device,
    ) -> PooledTensorBuffer:
        """Lease one registered buffer from a fixed-shape bucket."""
        bucket = (descriptors, str(device))
        with self._lock:
            available = self._available.setdefault(bucket, [])
            if available:
                lease = available.pop()
            else:
                allocated = self._allocated.setdefault(bucket, [])
                if len(allocated) >= self.max_buffers_per_bucket:
                    raise RuntimeError(
                        "Registered tensor bucket is exhausted; release a lease "
                        "or increase max_buffers_per_bucket."
                    )
                bundle = self.transport.allocate(descriptors, device=device)
                lease = PooledTensorBuffer(
                    bundle=bundle,
                    ticket=self.transport.make_ticket(bundle),
                    bucket=bucket,
                )
                allocated.append(lease)
            self._leased.add(id(lease))
            return lease

    def release(self, lease: PooledTensorBuffer) -> None:
        """Return one registered buffer lease to its shape bucket."""
        with self._lock:
            if id(lease) not in self._leased:
                raise ValueError("Registered tensor buffer is not currently leased.")
            self._leased.remove(id(lease))
            self._available[lease.bucket].append(lease)

    def close(self) -> None:
        """Unregister every pooled allocation after all leases return."""
        with self._lock:
            if self._leased:
                raise RuntimeError("Cannot close a pool with active buffer leases.")
            for leases in self._allocated.values():
                for lease in leases:
                    self.transport.unregister(lease.bundle)
            self._available.clear()
            self._allocated.clear()
