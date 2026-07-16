# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping
from typing import Any

import pytest

from flashdreams.serving.api import (
    ModelCapabilities,
    ModelDescriptor,
    SessionCreateRequest,
    StreamInput,
    StreamOutput,
)
from flashdreams.serving.backend import (
    LocalWorkerScheduler,
    ModelWorker,
    ServingCapacityError,
)
from flashdreams.serving.service import SessionSequenceError, SessionService
from flashdreams.scripts import serve as serve_cli

pytestmark = pytest.mark.ci_cpu


class _FakeWorker(ModelWorker):
    def __init__(self, worker_id: str, *, capacity: int = 2) -> None:
        self.worker_id = worker_id
        self._descriptor = ModelDescriptor(
            id="fake-video",
            capabilities=ModelCapabilities(
                inputs=("text",), outputs=("video",), sessions_per_worker=capacity
            ),
        )
        self.started = False
        self.sessions: set[str] = set()

    @property
    def descriptor(self) -> ModelDescriptor:
        return self._descriptor

    async def start(self) -> None:
        self.started = True

    async def create_session(
        self, session_id: str, parameters: Mapping[str, Any]
    ) -> None:
        del parameters
        self.sessions.add(session_id)

    async def stream(
        self, session_id: str, request: StreamInput
    ) -> AsyncIterator[StreamOutput]:
        assert session_id in self.sessions
        yield StreamOutput(
            type="step.completed",
            output={"prompt": request.input["prompt"]},
            final=True,
        )

    async def close_session(self, session_id: str) -> None:
        self.sessions.discard(session_id)

    async def close(self) -> None:
        self.sessions.clear()


def _service(*, capacity: int = 2, max_workers: int = 1) -> SessionService:
    descriptor = _FakeWorker("descriptor", capacity=capacity).descriptor
    scheduler = LocalWorkerScheduler(
        {"fake-video": lambda worker_id: _FakeWorker(worker_id, capacity=capacity)},
        max_workers_per_model=max_workers,
    )
    return SessionService({"fake-video": descriptor}, scheduler)


@pytest.mark.asyncio
async def test_multiple_sessions_share_one_worker() -> None:
    service = _service(capacity=2)
    first = await service.create_session(SessionCreateRequest(model="fake-video"))
    second = await service.create_session(SessionCreateRequest(model="fake-video"))

    assert first.worker_id == second.worker_id
    assert first.id != second.id

    outputs = [
        output
        async for output in service.stream(
            first.id,
            StreamInput(sequence_number=0, input={"prompt": "hello"}),
        )
    ]
    assert outputs[0].output == {"prompt": "hello"}
    assert (await service.get_session(first.id)).sequence_number == 1
    assert (await service.get_session(second.id)).sequence_number == 0
    await service.close()


@pytest.mark.asyncio
async def test_capacity_can_scale_to_another_worker() -> None:
    service = _service(capacity=1, max_workers=2)
    first = await service.create_session(SessionCreateRequest(model="fake-video"))
    second = await service.create_session(SessionCreateRequest(model="fake-video"))

    assert first.worker_id != second.worker_id
    with pytest.raises(ServingCapacityError):
        await service.create_session(SessionCreateRequest(model="fake-video"))
    await service.close()


@pytest.mark.asyncio
async def test_sequence_numbers_are_enforced_per_session() -> None:
    service = _service()
    session = await service.create_session(SessionCreateRequest(model="fake-video"))

    with pytest.raises(SessionSequenceError, match="Expected sequence_number 0"):
        async for _ in service.stream(
            session.id,
            StreamInput(sequence_number=1, input={"prompt": "late"}),
        ):
            pass
    await service.close()


@pytest.mark.asyncio
async def test_preload_starts_worker_without_consuming_session_capacity() -> None:
    workers: list[_FakeWorker] = []

    def factory(worker_id: str) -> _FakeWorker:
        worker = _FakeWorker(worker_id, capacity=1)
        workers.append(worker)
        return worker

    descriptor = _FakeWorker("descriptor", capacity=1).descriptor
    scheduler = LocalWorkerScheduler({"fake-video": factory})
    service = SessionService({"fake-video": descriptor}, scheduler)

    await service.preload_models()
    assert len(workers) == 1
    assert workers[0].started
    assert not workers[0].sessions

    session = await service.create_session(SessionCreateRequest(model="fake-video"))
    assert len(workers) == 1
    assert session.worker_id == workers[0].worker_id
    await service.close()


def test_serve_parser_accepts_eager_load() -> None:
    args = serve_cli.build_parser().parse_args(["fake-video", "--eager-load"])
    assert args.eager_load


def test_serve_entrypoint_handles_keyboard_interrupt(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(serve_cli, "discover_serve_configs", lambda: {"fake": object()})
    monkeypatch.setattr(serve_cli, "make_transport", lambda args: object())

    def interrupt(coroutine: Any) -> None:
        coroutine.close()
        raise KeyboardInterrupt

    monkeypatch.setattr(serve_cli.asyncio, "run", interrupt)
    serve_cli.entrypoint(["fake"])

    assert "FlashDreams server stopped." in capsys.readouterr().out
