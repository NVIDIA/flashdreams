# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from flashdreams.serving.webrtc import manager as manager_module
from flashdreams.serving.webrtc.controls import WSAD_SUPPORTED_KEYS
from flashdreams.serving.webrtc.manager import (
    BaseWebRTCSessionManager,
    ManagedWebRTCSession,
    WebRTCStepResult,
)
from flashdreams.serving.webrtc.server import SessionBusyError

pytestmark = pytest.mark.ci_cpu


def _runtime_config() -> SimpleNamespace:
    return SimpleNamespace(
        video_width=8,
        video_height=4,
        warmup_chunks=0,
        warmup_timeout_s=1.0,
    )


class _FakeVideoTrack:
    fps = 30

    def __init__(self) -> None:
        self.closed = False

    async def enqueue_chunk(self, chunk: Any) -> int:
        del chunk
        return 1

    def qsize(self) -> int:
        return 0

    async def close(self) -> None:
        self.closed = True


class _FakePeerConnection:
    def __init__(self) -> None:
        self.closed = False

    async def close(self) -> None:
        self.closed = True


class _FakeChannel:
    def __init__(self) -> None:
        self.messages: list[str] = []

    def send(self, message: str) -> None:
        self.messages.append(message)


class _FakeResampler:
    dt = 0.0
    next_chunk_start_v = 0.0

    def sample_chunk(
        self, num_frames: int
    ) -> tuple[list[tuple[float, float, frozenset[str]]], list[float]]:
        assert num_frames == 1
        return [(0.0, 0.0, frozenset({"w"}))], [0.0]


class _CountingVideoTrack(_FakeVideoTrack):
    async def enqueue_chunk(self, chunk: Any) -> int:
        return int(chunk.shape[0])


class _BaseTestManager(BaseWebRTCSessionManager):
    def _model_name(self) -> str:
        return "fake-model"


def _make_manager(
    manager_cls: type[BaseWebRTCSessionManager], runtime: Any
) -> BaseWebRTCSessionManager:
    return manager_cls(
        runtime=runtime,
        runtime_config=_runtime_config(),
        fps=30,
    )


def test_runtime_frame_timing_hooks_default_to_legacy_methods() -> None:
    class _LegacyRuntime:
        def peek_next_chunk_num_frames(self) -> int:
            return 2

        def peek_steady_chunk_num_frames(self) -> int:
            return 3

    runtime = _LegacyRuntime()
    manager = _make_manager(_BaseTestManager, runtime)

    assert manager._runtime_input_fps(runtime) == pytest.approx(30.0)
    assert manager._runtime_next_input_num_frames(runtime) == 2
    assert manager._runtime_steady_output_num_frames(runtime) == 3


def test_runtime_frame_timing_hooks_can_split_input_and_output() -> None:
    class _SplitRuntime:
        def peek_input_fps(self) -> float:
            return 6.0

        def peek_next_input_num_frames(self) -> int:
            return 4

        def peek_steady_output_num_frames(self) -> int:
            return 16

    runtime = _SplitRuntime()
    manager = _make_manager(_BaseTestManager, runtime)
    resampler = manager._make_resampler_at_fps(
        start_v=0.0,
        fps=manager._runtime_input_fps(runtime),
    )

    assert resampler.dt == pytest.approx(1.0 / 6.0)
    assert manager._runtime_next_input_num_frames(runtime) == 4
    assert manager._runtime_steady_output_num_frames(runtime) == 16


def _managed_session(
    runtime: Any,
) -> tuple[ManagedWebRTCSession, _FakeVideoTrack, _FakePeerConnection, _FakeChannel]:
    video_track = _FakeVideoTrack()
    peer = _FakePeerConnection()
    channel = _FakeChannel()
    first_action = asyncio.Event()
    first_action.set()
    managed = ManagedWebRTCSession(
        runtime=runtime,
        video_track=video_track,  # ty:ignore[invalid-argument-type]
        peer_connection=peer,
        resampler=_FakeResampler(),  # ty:ignore[invalid-argument-type]
        control_channel=channel,
        first_action_received=first_action,
    )
    return managed, video_track, peer, channel


@pytest.mark.asyncio
async def test_generation_worker_closes_session_when_flag_set() -> None:
    class _ClosingRuntime:
        def __init__(self) -> None:
            self.generate_calls = 0

        def peek_next_chunk_num_frames(self) -> int:
            return 1

        async def generate_chunk(
            self, *, segments: Any, frame_times: Any
        ) -> WebRTCStepResult:
            del segments, frame_times
            self.generate_calls += 1
            raise RuntimeError("boom")

    class _ClosingManager(_BaseTestManager):
        _close_session_on_generation_error = True

    runtime = _ClosingRuntime()
    manager = _make_manager(_ClosingManager, runtime)
    managed, video_track, peer, channel = _managed_session(runtime)
    manager._active_session = managed

    task = asyncio.create_task(manager._generation_worker(managed_session=managed))
    managed.generation_task = task
    await asyncio.wait_for(task, timeout=5.0)

    assert runtime.generate_calls == 1
    assert not manager.has_active_session()
    assert managed.closed
    assert video_track.closed
    assert peer.closed
    assert len(channel.messages) == 1


@pytest.mark.asyncio
async def test_generation_worker_retries_on_error_when_flag_unset() -> None:
    class _RetryRuntime:
        def __init__(self) -> None:
            self.generate_calls = 0
            self.managed_session: ManagedWebRTCSession | None = None

        def peek_next_chunk_num_frames(self) -> int:
            return 1

        async def generate_chunk(
            self, *, segments: Any, frame_times: Any
        ) -> WebRTCStepResult:
            del segments, frame_times
            self.generate_calls += 1
            # Stop the loop after the second attempt without tearing down.
            if self.generate_calls >= 2 and self.managed_session is not None:
                self.managed_session.closed = True
            raise RuntimeError("boom")

    runtime = _RetryRuntime()
    manager = _make_manager(_BaseTestManager, runtime)  # flag defaults to False
    managed, video_track, peer, channel = _managed_session(runtime)
    runtime.managed_session = managed
    manager._active_session = managed

    task = asyncio.create_task(manager._generation_worker(managed_session=managed))
    managed.generation_task = task
    await asyncio.wait_for(task, timeout=5.0)

    # Retried instead of bailing after the first error.
    assert runtime.generate_calls == 2
    # The worker reported both errors but never tore the transport down.
    assert len(channel.messages) == 2
    assert not video_track.closed
    assert not peer.closed


@pytest.mark.asyncio
async def test_chunk_done_payload_includes_model_and_extra() -> None:
    class _OneChunkRuntime:
        def __init__(self) -> None:
            self.managed_session: ManagedWebRTCSession | None = None

        def peek_next_chunk_num_frames(self) -> int:
            return 1

        async def generate_chunk(
            self, *, segments: Any, frame_times: Any
        ) -> WebRTCStepResult:
            del segments, frame_times
            if self.managed_session is not None:
                self.managed_session.closed = True
            return WebRTCStepResult(
                chunk_index=0,
                num_frames=1,
                video_chunk=torch.zeros((1, 1, 1, 3, 2, 2), dtype=torch.uint8),
                stats=None,
            )

    class _ExtraManager(_BaseTestManager):
        def _chunk_done_extra(self) -> dict[str, Any]:
            return {"stream": "rgb"}

    runtime = _OneChunkRuntime()
    manager = _make_manager(_ExtraManager, runtime)
    managed, _video_track, _peer, channel = _managed_session(runtime)
    runtime.managed_session = managed
    manager._active_session = managed

    task = asyncio.create_task(manager._generation_worker(managed_session=managed))
    managed.generation_task = task
    await asyncio.wait_for(task, timeout=5.0)

    chunk_done = [
        json.loads(m)
        for m in channel.messages
        if json.loads(m).get("type") == "chunk_done"
    ]
    assert len(chunk_done) == 1
    payload = chunk_done[0]
    assert payload["model"] == "fake-model"
    assert payload["stream"] == "rgb"
    assert payload["resolution"] == {"width": 8, "height": 4}


@pytest.mark.asyncio
async def test_generation_worker_uses_split_input_and_output_frame_counts() -> None:
    class _SplitResampler:
        dt = 0.0
        next_chunk_start_v = 0.0

        def __init__(self) -> None:
            self.sampled_num_frames: list[int] = []

        def sample_chunk(
            self, num_frames: int
        ) -> tuple[list[tuple[float, float, frozenset[str]]], list[float]]:
            self.sampled_num_frames.append(num_frames)
            return (
                [(0.0, 0.0, frozenset({"w"}))],
                [float(index) for index in range(num_frames)],
            )

    class _SplitRuntime:
        def __init__(self) -> None:
            self.managed_session: ManagedWebRTCSession | None = None
            self.frame_times: list[float] | None = None

        def peek_input_fps(self) -> float:
            return 6.0

        def peek_next_input_num_frames(self) -> int:
            return 2

        async def generate_chunk(
            self, *, segments: Any, frame_times: list[float]
        ) -> WebRTCStepResult:
            del segments
            self.frame_times = frame_times
            if self.managed_session is not None:
                self.managed_session.closed = True
            return WebRTCStepResult(
                chunk_index=0,
                num_frames=5,
                video_chunk=torch.zeros((5, 1, 1, 3, 2, 2), dtype=torch.uint8),
                stats=None,
            )

    runtime = _SplitRuntime()
    manager = _make_manager(_BaseTestManager, runtime)
    managed, _video_track, _peer, channel = _managed_session(runtime)
    resampler = _SplitResampler()
    managed.video_track = _CountingVideoTrack()  # ty:ignore[invalid-assignment]
    managed.resampler = resampler  # ty:ignore[invalid-assignment]
    runtime.managed_session = managed
    manager._active_session = managed

    task = asyncio.create_task(manager._generation_worker(managed_session=managed))
    managed.generation_task = task
    await asyncio.wait_for(task, timeout=5.0)

    assert resampler.sampled_num_frames == [2]
    assert runtime.frame_times == [0.0, 1.0]
    chunk_done = [
        json.loads(message)
        for message in channel.messages
        if json.loads(message).get("type") == "chunk_done"
    ]
    assert len(chunk_done) == 1
    assert chunk_done[0]["num_frames"] == 5
    assert chunk_done[0]["enqueued_frames"] == 5
    assert chunk_done[0]["play_ms"] == pytest.approx(5 * 1000 / 30, abs=0.05)


@pytest.mark.asyncio
async def test_generation_worker_logs_periodic_perf_stats(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    perf_logs: list[tuple[str, tuple[Any, ...]]] = []

    def _record_info(message: str, *args: Any, **_kwargs: Any) -> None:
        if message.startswith("WebRTC perf"):
            perf_logs.append((message, args))

    class _StatsRuntime:
        def __init__(self) -> None:
            self.managed_session: ManagedWebRTCSession | None = None
            self.chunk_index = 0

        def peek_next_chunk_num_frames(self) -> int:
            return 1

        async def generate_chunk(
            self, *, segments: Any, frame_times: Any
        ) -> WebRTCStepResult:
            del segments, frame_times
            chunk_index = self.chunk_index
            self.chunk_index += 1
            if chunk_index >= 2 and self.managed_session is not None:
                self.managed_session.closed = True
            return WebRTCStepResult(
                chunk_index=chunk_index,
                num_frames=4,
                video_chunk=torch.zeros((4, 1, 1, 3, 2, 2), dtype=torch.uint8),
                stats={
                    "model_step_s": 0.02,
                    "denoise_s": 0.01,
                    "decode_s": 0.004,
                    "pixel_post_s": 0.003,
                    "gpu_to_cpu_copy_s": 0.002,
                    "compile_denoise_active": 1.0,
                    "compile_denoise_start_step": 3.0,
                    "cache_frames": 13.0,
                    "cache_tokens": 512.0,
                },
            )

    class _FrequentLogManager(_BaseTestManager):
        _perf_log_interval_chunks = 2

    monkeypatch.setattr(manager_module.logger, "info", _record_info)
    runtime = _StatsRuntime()
    manager = _make_manager(_FrequentLogManager, runtime)
    managed, _video_track, _peer, _channel = _managed_session(runtime)
    runtime.managed_session = managed
    manager._active_session = managed

    task = asyncio.create_task(manager._generation_worker(managed_session=managed))
    managed.generation_task = task
    await asyncio.wait_for(task, timeout=5.0)

    assert [args[0] for _message, args in perf_logs] == [0, 2]
    assert "compile_active" in perf_logs[0][0]
    assert "pixel_post_ms" in perf_logs[0][0]
    assert "copy_ms" in perf_logs[0][0]
    assert perf_logs[0][1][-2:] == (13, 512)


@pytest.mark.asyncio
async def test_create_answer_raises_busy_with_subclass_message() -> None:
    class _BusyManager(_BaseTestManager):
        _busy_message = "custom busy message"

    manager = _make_manager(_BusyManager, runtime=SimpleNamespace())
    manager._runtime_ready = True
    manager._warmup_complete = True
    existing, *_ = _managed_session(runtime=SimpleNamespace())
    manager._active_session = existing

    with pytest.raises(SessionBusyError, match="custom busy message"):
        await manager.create_answer(offer_sdp="x", offer_type="offer")


def test_make_resampler_honors_supported_keys() -> None:
    class _WsadManager(_BaseTestManager):
        _resampler_supported_keys = WSAD_SUPPORTED_KEYS

    wsad = _make_manager(_WsadManager, runtime=SimpleNamespace())._make_resampler(
        start_v=1.0
    )
    wsad.on_edge(arrival_t=0.5, event="keydown", key="q")
    wsad_segments, _ = wsad.sample_chunk(num_frames=1)
    # 'q' is not a WSAD driving key, so it is rejected and never held.
    assert wsad_segments[0][2] == frozenset()

    default = _make_manager(
        _BaseTestManager, runtime=SimpleNamespace()
    )._make_resampler(start_v=1.0)
    default.on_edge(arrival_t=0.5, event="keydown", key="q")
    default_segments, _ = default.sample_chunk(num_frames=1)
    # The default key set (used by Lingbot) accepts 'q'.
    assert default_segments[0][2] == frozenset({"q"})


@pytest.mark.asyncio
async def test_step_action_starts_generation_without_key_edge() -> None:
    class _RecordingResampler(_FakeResampler):
        def __init__(self) -> None:
            self.edges: list[tuple[float, str, str]] = []

        def on_edge(self, *, arrival_t: float, event: str, key: str) -> None:
            self.edges.append((arrival_t, event, key))

    runtime = SimpleNamespace()
    manager = _make_manager(_BaseTestManager, runtime)
    managed, _video_track, _peer, _channel = _managed_session(runtime)
    managed.first_action_received.clear()
    resampler = _RecordingResampler()
    managed.resampler = resampler  # ty:ignore[invalid-assignment]

    await manager._handle_datachannel_message(
        managed_session=managed,
        raw_message=json.dumps({"type": "action", "action": {"event": "step"}}),
    )

    assert managed.first_action_received.is_set()
    assert len(managed.pending_action_arrivals) == 1
    assert resampler.edges == []
