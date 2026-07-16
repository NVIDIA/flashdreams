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

"""Model-worker and worker-scheduler abstractions for serving."""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Callable, Iterable, Mapping
from contextlib import suppress
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from flashdreams.serving.api import ModelDescriptor, StreamInput, StreamOutput


class ServingCapacityError(RuntimeError):
    """Raised when no worker has capacity for a new session."""


class ModelWorker(ABC):
    """Own model weights and isolate mutable inference state by session."""

    @property
    @abstractmethod
    def descriptor(self) -> ModelDescriptor:
        """Return the model variant hosted by this worker."""

    @abstractmethod
    async def start(self) -> None:
        """Load model weights and initialize worker-global resources."""

    @abstractmethod
    async def create_session(
        self, session_id: str, parameters: Mapping[str, Any]
    ) -> None:
        """Allocate and initialize mutable state for one session."""

    @abstractmethod
    def stream(
        self, session_id: str, request: StreamInput
    ) -> AsyncIterator[StreamOutput]:
        """Stream output events for one ordered session step."""

    @abstractmethod
    async def close_session(self, session_id: str) -> None:
        """Release cache and other mutable state owned by one session."""

    @abstractmethod
    async def close(self) -> None:
        """Release model weights and worker-global resources."""

    async def create_webrtc_answer(
        self, session_id: str, offer: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        """Create a WebRTC answer for a session when the model supports media tracks.

        Raises:
            NotImplementedError: The worker does not implement WebRTC negotiation.
        """
        del session_id, offer
        raise NotImplementedError("This model worker does not implement WebRTC offers.")


WorkerFactory = Callable[[str], ModelWorker]
"""Factory receiving a worker ID and returning an unloaded worker."""


@dataclass(slots=True)
class WorkerLease:
    """Scheduler lease binding a session to a model worker."""

    worker_id: str
    """Stable worker identifier."""

    worker: ModelWorker
    """In-process worker endpoint used by the local scheduler."""

    routing_token: str | None = None
    """Opaque request-plane address for a remote scheduler."""

    transfer_metadata: dict[str, Any] = field(default_factory=dict)
    """Opaque cache-transfer metadata for disaggregated execution."""


class WorkerScheduler(ABC):
    """Allocate model-worker capacity without exposing placement to transports."""

    @abstractmethod
    async def preload(self, models: Iterable[str]) -> None:
        """Start one idle worker for each requested model."""

    @abstractmethod
    async def acquire(
        self, model: str, session_id: str, routing_hint: str | None = None
    ) -> WorkerLease:
        """Reserve worker capacity for a session."""

    @abstractmethod
    async def release(self, lease: WorkerLease, session_id: str) -> None:
        """Return a session's worker capacity to the scheduler."""

    @abstractmethod
    async def close(self) -> None:
        """Close every worker managed by the scheduler."""


@dataclass(slots=True)
class _WorkerSlot:
    """Local worker and the session IDs consuming its capacity."""

    worker: ModelWorker
    session_ids: set[str] = field(default_factory=set)


class LocalWorkerScheduler(WorkerScheduler):
    """Lazily start local workers and share them up to model capacity."""

    def __init__(
        self,
        factories: Mapping[str, WorkerFactory],
        *,
        max_workers_per_model: int = 1,
    ) -> None:
        """Initialize a scheduler from model-slug worker factories.

        Args:
            factories: Worker factories keyed by public model slug.
            max_workers_per_model: Maximum lazy worker replicas per model.
        """
        if max_workers_per_model < 1:
            raise ValueError("max_workers_per_model must be at least one.")
        self._factories = dict(factories)
        self._max_workers_per_model = max_workers_per_model
        self._workers: dict[str, list[_WorkerSlot]] = {model: [] for model in factories}
        self._lock = asyncio.Lock()

    async def preload(self, models: Iterable[str]) -> None:
        """Start one idle local worker for each requested model."""
        async with self._lock:
            for model in models:
                if model not in self._factories:
                    raise KeyError(f"Unknown model {model!r}.")
                if not self._workers[model]:
                    self._workers[model].append(await self._start_worker(model))

    async def acquire(
        self, model: str, session_id: str, routing_hint: str | None = None
    ) -> WorkerLease:
        """Reserve the least-loaded compatible worker, starting one if needed."""
        del routing_hint
        async with self._lock:
            if model not in self._factories:
                raise KeyError(f"Unknown model {model!r}.")
            slots = self._workers[model]
            available = [
                slot
                for slot in slots
                if len(slot.session_ids)
                < slot.worker.descriptor.capabilities.sessions_per_worker
            ]
            if available:
                slot = min(available, key=lambda item: len(item.session_ids))
            elif len(slots) < self._max_workers_per_model:
                slot = await self._start_worker(model)
                slots.append(slot)
            else:
                raise ServingCapacityError(
                    f"All workers for model {model!r} are at session capacity."
                )
            slot.session_ids.add(session_id)
            worker_id = getattr(slot.worker, "worker_id", f"local-{id(slot.worker):x}")
            return WorkerLease(worker_id=str(worker_id), worker=slot.worker)

    async def _start_worker(self, model: str) -> _WorkerSlot:
        worker_id = f"{model}-{uuid4().hex[:12]}"
        worker = self._factories[model](worker_id)
        try:
            await worker.start()
        except Exception:
            with suppress(Exception):
                await worker.close()
            raise
        return _WorkerSlot(worker=worker)

    async def release(self, lease: WorkerLease, session_id: str) -> None:
        """Return one local worker slot to the scheduler."""
        async with self._lock:
            for slots in self._workers.values():
                for slot in slots:
                    if slot.worker is lease.worker:
                        slot.session_ids.discard(session_id)
                        return

    async def close(self) -> None:
        """Close all local workers and clear scheduler state."""
        async with self._lock:
            workers = [
                slot.worker for slots in self._workers.values() for slot in slots
            ]
            for slots in self._workers.values():
                slots.clear()
        await asyncio.gather(*(worker.close() for worker in workers))
