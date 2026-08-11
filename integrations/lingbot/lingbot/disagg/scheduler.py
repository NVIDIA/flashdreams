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

"""Session-affine scheduling policies for LingBot disaggregated workers."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import Enum


class ServiceClass(str, Enum):
    """Deployment pool selected for an interactive session."""

    LATENCY = "latency"
    THROUGHPUT = "throughput"


@dataclass(frozen=True, kw_only=True)
class SessionRequest:
    """Placement constraints supplied when a session opens."""

    session_id: str
    """Stable interactive-session identifier."""

    shape: tuple[int, ...]
    """Latent or pixel shape used for fixed-shape compatibility."""

    cp_size: int
    """Required context-parallel group size."""

    service_class: ServiceClass = ServiceClass.THROUGHPUT
    """Latency or aggregate-throughput deployment pool."""

    preferred_rack: str | None = None
    """Optional rack locality hint."""

    preferred_nic: str | None = None
    """Optional NIC or fabric locality hint."""


@dataclass(frozen=True, kw_only=True)
class WorkerSnapshot:
    """Current DiT worker-group state used for one placement decision."""

    worker_id: str
    """Stable worker or CP-group identifier."""

    pool: str
    """Deployment pool, such as ``aggregated-cp8`` or ``io-plus-7-dit``."""

    queue_depth: int
    """Queued autoregressive chunks."""

    predicted_chunk_ms: float
    """Expected service time for one compatible chunk."""

    free_hbm_gib: float
    """Currently available device memory."""

    supported_shapes: frozenset[tuple[int, ...]]
    """Shapes for which this worker has compiled kernels and buffers."""

    supported_cp_sizes: frozenset[int]
    """Context-parallel group sizes hosted by this worker."""

    resident_sessions: frozenset[str] = frozenset()
    """Sessions whose autoregressive KV state is already resident."""

    rack: str | None = None
    """Rack locality, when known."""

    nic: str | None = None
    """GPU-local NIC or fabric rail, when known."""

    rdma_capable: bool = True
    """Whether the selected path is verified as RDMA rather than TCP."""


@dataclass(frozen=True, kw_only=True)
class ScheduledMicrobatch:
    """Compatible independent sessions that may share one DiT launch."""

    worker_id: str
    shape: tuple[int, ...]
    cp_size: int
    session_ids: tuple[str, ...]


class SessionAwareScheduler:
    """Place once, preserve cache affinity, and reject non-RDMA fallbacks."""

    def __init__(
        self,
        *,
        min_free_hbm_gib: float = 0.0,
        require_rdma: bool = True,
    ) -> None:
        self.min_free_hbm_gib = min_free_hbm_gib
        self.require_rdma = require_rdma
        self._placements: dict[str, str] = {}

    @staticmethod
    def pool_for(service_class: ServiceClass) -> str:
        """Return the recommended fixed deployment pool."""
        if service_class is ServiceClass.LATENCY:
            return "aggregated-cp8"
        return "io-plus-7-dit"

    def assign(
        self,
        request: SessionRequest,
        workers: Sequence[WorkerSnapshot],
    ) -> str:
        """Return a sticky compatible worker placement for ``request``."""
        existing = self._placements.get(request.session_id)
        if existing is not None:
            if any(worker.worker_id == existing for worker in workers):
                return existing
            raise RuntimeError(
                f"Session {request.session_id!r} lost resident worker {existing!r}; "
                "restore its cache before rerouting."
            )

        target_pool = self.pool_for(request.service_class)
        compatible = [
            worker
            for worker in workers
            if worker.pool == target_pool
            and request.shape in worker.supported_shapes
            and request.cp_size in worker.supported_cp_sizes
            and worker.free_hbm_gib >= self.min_free_hbm_gib
            and (worker.rdma_capable or not self.require_rdma)
        ]
        if not compatible:
            raise RuntimeError(
                f"No compatible {target_pool!r} worker for shape={request.shape}, "
                f"CP{request.cp_size}; TCP fallback is disabled={self.require_rdma}."
            )

        def score(worker: WorkerSnapshot) -> tuple[float, int, int, str]:
            locality_penalty = 0
            if request.preferred_rack is not None:
                locality_penalty += worker.rack != request.preferred_rack
            if request.preferred_nic is not None:
                locality_penalty += worker.nic != request.preferred_nic
            predicted_wait_ms = (worker.queue_depth + 1) * worker.predicted_chunk_ms
            residency_penalty = request.session_id not in worker.resident_sessions
            return (
                predicted_wait_ms,
                locality_penalty,
                residency_penalty,
                worker.worker_id,
            )

        selected = min(compatible, key=score)
        self._placements[request.session_id] = selected.worker_id
        return selected.worker_id

    def release(self, session_id: str) -> None:
        """Forget placement after all stage caches have been destroyed."""
        self._placements.pop(session_id, None)


def build_microbatches(
    assignments: Iterable[tuple[SessionRequest, str]],
    *,
    max_batch_size: int,
) -> tuple[ScheduledMicrobatch, ...]:
    """Group independent compatible sessions for a future fused DiT launch.

    The scheduler preserves session identity; the model runtime must still
    implement batched cache gather/scatter before these groups can share one
    kernel launch.
    """
    if max_batch_size < 1:
        raise ValueError(f"max_batch_size must be positive, got {max_batch_size}.")
    groups: dict[tuple[str, tuple[int, ...], int], list[str]] = defaultdict(list)
    for request, worker_id in assignments:
        groups[(worker_id, request.shape, request.cp_size)].append(request.session_id)

    batches: list[ScheduledMicrobatch] = []
    for (worker_id, shape, cp_size), session_ids in sorted(groups.items()):
        for start in range(0, len(session_ids), max_batch_size):
            batches.append(
                ScheduledMicrobatch(
                    worker_id=worker_id,
                    shape=shape,
                    cp_size=cp_size,
                    session_ids=tuple(session_ids[start : start + max_batch_size]),
                )
            )
    return tuple(batches)
