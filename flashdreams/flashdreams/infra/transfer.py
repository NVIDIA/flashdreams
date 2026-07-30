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

"""GPU tensor-bundle transfer through the Mooncake Transfer Engine."""

from __future__ import annotations

import socket
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
    """Receiver-owned VRAM addresses advertised to a sender."""

    session_id: str
    """Mooncake peer session identifier."""

    descriptors: tuple[TensorDescriptor, ...]
    """Expected bundle layout in transfer order."""

    addresses: tuple[int, ...]
    """Registered receiver addresses matching ``descriptors``."""


@dataclass(frozen=True, kw_only=True)
class TransferStats:
    """Observed metrics for one synchronous bundle transfer."""

    backend: str
    """Selected transfer backend and protocol."""

    payload_bytes: int
    """Total tensor bytes transferred."""

    registration_ms: float
    """Sender memory-registration time for previously unseen allocations."""

    transfer_ms: float
    """Synchronous transfer wall time, including engine submission overhead."""

    bandwidth_gbps: float
    """Effective decimal GB/s computed from payload bytes and transfer time."""


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

    def send(
        self,
        bundle: Mapping[str, Tensor],
        ticket: TensorTransferTicket,
    ) -> TransferStats:
        """Synchronously write a tensor bundle into receiver-owned VRAM."""
        descriptors = describe_tensor_bundle(bundle)
        if descriptors != ticket.descriptors:
            raise ValueError(
                "Sender bundle layout does not match the receiver transfer ticket."
            )
        registration_ms = self.register(bundle)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        started = time.perf_counter()
        result = self.engine.batch_transfer_sync_write(
            ticket.session_id,
            [bundle[item.name].data_ptr() for item in descriptors],
            list(ticket.addresses),
            [item.nbytes for item in descriptors],
        )
        elapsed_s = time.perf_counter() - started
        if result != 0:
            raise RuntimeError(f"Mooncake tensor transfer failed with status {result}.")
        payload_bytes = sum(item.nbytes for item in descriptors)
        bandwidth_gbps = (
            payload_bytes / elapsed_s / 1e9 if elapsed_s > 0.0 else float("inf")
        )
        return TransferStats(
            backend=self.backend,
            payload_bytes=payload_bytes,
            registration_ms=registration_ms,
            transfer_ms=elapsed_s * 1000.0,
            bandwidth_gbps=bandwidth_gbps,
        )

    def close(self) -> None:
        """Unregister every receiver allocation owned by this endpoint."""
        for address in tuple(self._registered_addresses):
            result = self.engine.unregister_memory(address)
            if result != 0:
                raise RuntimeError(
                    f"Mooncake failed to unregister address {address}, status={result}."
                )
            self._registered_addresses.remove(address)
