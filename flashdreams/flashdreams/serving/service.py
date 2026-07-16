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

"""Shared model discovery and session lifecycle service."""

from __future__ import annotations

import asyncio
from contextlib import suppress
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import uuid4

from flashdreams.serving.api import (
    ModelDescriptor,
    SessionCreateRequest,
    SessionSnapshot,
    SessionStatus,
    StreamInput,
    StreamOutput,
)
from flashdreams.serving.backend import WorkerLease, WorkerScheduler


class SessionNotFoundError(KeyError):
    """Raised when a session ID is unknown or already closed."""


class SessionSequenceError(ValueError):
    """Raised when a client submits a stale or future sequence number."""


class SessionStateError(RuntimeError):
    """Raised when an operation is incompatible with session state."""


@dataclass(slots=True)
class _Session:
    """Mutable server-side session record."""

    id: str
    model: str
    status: SessionStatus
    sequence_number: int
    lease_expires_at: datetime
    lease_seconds: float
    worker_lease: WorkerLease | None = None
    error: str | None = None
    step_lock: asyncio.Lock | None = None


class SessionService:
    """Coordinate model discovery, worker placement, and ordered session steps."""

    def __init__(
        self,
        models: Mapping[str, ModelDescriptor],
        scheduler: WorkerScheduler,
        *,
        default_lease_seconds: float = 300.0,
    ) -> None:
        """Initialize the protocol-neutral serving service.

        Args:
            models: Public model descriptors keyed by slug.
            scheduler: Worker capacity and placement provider.
            default_lease_seconds: Idle lifetime assigned to new sessions.
        """
        if default_lease_seconds <= 0:
            raise ValueError("default_lease_seconds must be positive.")
        self._models = dict(models)
        self._scheduler = scheduler
        self._default_lease_seconds = default_lease_seconds
        self._sessions: dict[str, _Session] = {}
        self._lock = asyncio.Lock()

    def list_models(self) -> list[ModelDescriptor]:
        """Return available model variants in stable slug order."""
        return [self._models[key] for key in sorted(self._models)]

    async def preload_models(self) -> None:
        """Start one idle worker for every exposed model variant."""
        await self._scheduler.preload(sorted(self._models))

    async def create_session(self, request: SessionCreateRequest) -> SessionSnapshot:
        """Allocate worker capacity and initialize one session cache."""
        if request.model not in self._models:
            raise KeyError(f"Unknown model {request.model!r}.")
        lease_seconds = (
            self._default_lease_seconds
            if request.lease_seconds is None
            else request.lease_seconds
        )
        if lease_seconds <= 0:
            raise ValueError("lease_seconds must be positive.")
        now = datetime.now(timezone.utc)
        session = _Session(
            id=uuid4().hex,
            model=request.model,
            status=SessionStatus.ALLOCATING,
            sequence_number=0,
            lease_expires_at=now + timedelta(seconds=lease_seconds),
            lease_seconds=lease_seconds,
            step_lock=asyncio.Lock(),
        )
        async with self._lock:
            self._sessions[session.id] = session
        try:
            lease = await self._scheduler.acquire(
                request.model, session.id, request.routing_hint
            )
            session.worker_lease = lease
            await lease.worker.create_session(session.id, request.parameters)
            session.status = SessionStatus.READY
        except Exception as exc:
            session.status = SessionStatus.FAILED
            session.error = str(exc)
            if session.worker_lease is not None:
                with suppress(Exception):
                    await session.worker_lease.worker.close_session(session.id)
                await self._scheduler.release(session.worker_lease, session.id)
            raise
        return self._snapshot(session)

    async def get_session(self, session_id: str) -> SessionSnapshot:
        """Return readiness, sequence number, placement, and lease expiry."""
        session = await self._require_session(session_id)
        await self._expire_if_needed(session)
        if session.status is SessionStatus.EXPIRED:
            raise SessionNotFoundError(session_id)
        return self._snapshot(session)

    async def stream(
        self, session_id: str, request: StreamInput
    ) -> AsyncIterator[StreamOutput]:
        """Execute one ordered step and stream every worker output event."""
        session = await self._require_session(session_id)
        await self._expire_if_needed(session)
        if session.status is not SessionStatus.READY:
            raise SessionStateError(
                f"Session {session_id!r} is {session.status.value}, not ready."
            )
        assert session.step_lock is not None
        async with session.step_lock:
            if request.sequence_number != session.sequence_number:
                raise SessionSequenceError(
                    f"Expected sequence_number {session.sequence_number}, "
                    f"got {request.sequence_number}."
                )
            assert session.worker_lease is not None
            async for output in session.worker_lease.worker.stream(session.id, request):
                yield output
            session.sequence_number += 1
            session.lease_expires_at = datetime.now(timezone.utc) + timedelta(
                seconds=session.lease_seconds
            )

    async def create_webrtc_answer(
        self, session_id: str, offer: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        """Delegate WebRTC negotiation to the session's assigned worker."""
        session = await self._require_session(session_id)
        await self._expire_if_needed(session)
        if session.status is not SessionStatus.READY:
            raise SessionStateError(f"Session {session_id!r} is not ready.")
        assert session.worker_lease is not None
        return await session.worker_lease.worker.create_webrtc_answer(session.id, offer)

    async def close_session(
        self, session_id: str, *, expired: bool = False
    ) -> SessionSnapshot:
        """Close a session and release its cache and worker capacity."""
        session = await self._require_session(session_id)
        if session.status in {SessionStatus.CLOSED, SessionStatus.EXPIRED}:
            return self._snapshot(session)
        assert session.step_lock is not None
        async with session.step_lock:
            if session.status in {SessionStatus.CLOSED, SessionStatus.EXPIRED}:
                return self._snapshot(session)
            session.status = SessionStatus.CLOSING
            if session.worker_lease is not None:
                try:
                    await session.worker_lease.worker.close_session(session.id)
                finally:
                    await self._scheduler.release(session.worker_lease, session.id)
            session.status = SessionStatus.EXPIRED if expired else SessionStatus.CLOSED
            async with self._lock:
                self._sessions.pop(session_id, None)
        return self._snapshot(session)

    async def expire_sessions(self) -> int:
        """Close every expired session and return the number reclaimed."""
        async with self._lock:
            sessions = list(self._sessions.values())
        expired = [session for session in sessions if self._is_expired(session)]
        for session in expired:
            with suppress(SessionNotFoundError):
                await self.close_session(session.id, expired=True)
        return len(expired)

    async def close(self) -> None:
        """Close all sessions and workers managed by the service."""
        async with self._lock:
            session_ids = list(self._sessions)
        for session_id in session_ids:
            await self.close_session(session_id)
        await self._scheduler.close()

    async def _require_session(self, session_id: str) -> _Session:
        async with self._lock:
            session = self._sessions.get(session_id)
        if session is None:
            raise SessionNotFoundError(session_id)
        return session

    async def _expire_if_needed(self, session: _Session) -> None:
        if self._is_expired(session):
            await self.close_session(session.id, expired=True)

    @staticmethod
    def _is_expired(session: _Session) -> bool:
        return (
            session.status is SessionStatus.READY
            and datetime.now(timezone.utc) >= session.lease_expires_at
        )

    @staticmethod
    def _snapshot(session: _Session) -> SessionSnapshot:
        return SessionSnapshot(
            id=session.id,
            model=session.model,
            status=session.status,
            sequence_number=session.sequence_number,
            lease_expires_at=session.lease_expires_at.isoformat(),
            worker_id=(
                None if session.worker_lease is None else session.worker_lease.worker_id
            ),
            error=session.error,
        )
