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

"""Protocol-neutral serving API contracts."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class SessionStatus(str, Enum):
    """Lifecycle state of a serving session."""

    ALLOCATING = "allocating"
    READY = "ready"
    CLOSING = "closing"
    CLOSED = "closed"
    FAILED = "failed"
    EXPIRED = "expired"


@dataclass(frozen=True, slots=True)
class ResourceRequest:
    """Compute resources reserved together for one model worker."""

    gpu_count: int = 1
    """Number of GPUs in the worker's gang allocation."""

    cpu_count: int = 1
    """Minimum number of host CPU cores."""

    memory_gb: float = 0.0
    """Minimum host memory in GiB; zero leaves placement unconstrained."""

    gpu_memory_gb: float = 0.0
    """Minimum memory per GPU in GiB; zero leaves placement unconstrained."""

    placement: str = "single-node"
    """Placement constraint understood by the worker scheduler."""


@dataclass(frozen=True, slots=True)
class ModelCapabilities:
    """Input, output, transport, and concurrency capabilities of a model."""

    inputs: tuple[str, ...]
    """Accepted logical input modalities."""

    outputs: tuple[str, ...]
    """Produced logical output modalities."""

    transports: tuple[str, ...] = ("websocket", "webrtc", "grpc")
    """Supported transport adapters."""

    sessions_per_worker: int = 1
    """Maximum simultaneously live session caches on one worker."""


@dataclass(frozen=True, slots=True)
class ModelDescriptor:
    """Public description and placement requirements of a model variant."""

    id: str
    """Stable model slug accepted by ``POST /v1/sessions``."""

    capabilities: ModelCapabilities
    """Modalities, transports, and worker concurrency."""

    resources: ResourceRequest = field(default_factory=ResourceRequest)
    """Resources reserved as one indivisible worker allocation."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Model-specific discovery metadata."""

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable model description."""
        return asdict(self)


@dataclass(frozen=True, slots=True)
class SessionCreateRequest:
    """Request used to allocate a session cache on a model worker."""

    model: str
    """Model slug from ``GET /v1/models``."""

    parameters: dict[str, Any] = field(default_factory=dict)
    """Model-specific session initialization parameters."""

    lease_seconds: float | None = None
    """Idle lease duration; ``None`` uses the server default."""

    routing_hint: str | None = None
    """Opaque placement hint for external schedulers such as Dynamo."""


@dataclass(frozen=True, slots=True)
class SessionSnapshot:
    """Externally visible state of one serving session."""

    id: str
    """Opaque session identifier."""

    model: str
    """Model slug used by the session."""

    status: SessionStatus
    """Current session lifecycle state."""

    sequence_number: int
    """Next client step sequence number."""

    lease_expires_at: str
    """UTC lease expiry in ISO 8601 form."""

    worker_id: str | None = None
    """Assigned worker identifier; ``None`` while placement is pending."""

    error: str | None = None
    """Initialization or execution failure; ``None`` when healthy."""

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable session snapshot."""
        value = asdict(self)
        value["status"] = self.status.value
        return value


@dataclass(frozen=True, slots=True)
class StreamInput:
    """One ordered inference step submitted by a client."""

    sequence_number: int
    """Expected next sequence number for the session."""

    input: dict[str, Any]
    """Model-specific multimodal step payload."""


@dataclass(frozen=True, slots=True)
class StreamOutput:
    """One event emitted while processing an inference step."""

    type: str
    """Stable event kind used for client dispatch."""

    output: dict[str, Any] = field(default_factory=dict)
    """Model-specific output payload."""

    final: bool = False
    """Whether this event completes the submitted step."""

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable stream event."""
        return asdict(self)
