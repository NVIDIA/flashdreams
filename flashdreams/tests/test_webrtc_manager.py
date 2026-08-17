# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch

from flashdreams.demo.outputs import WebRTCOutputSink
from flashdreams.runtime import (
    DRIVER_COMMAND,
    DRIVING_SUPPORTED_KEYS,
    InferenceInput,
    InputCanonicalizer,
    KeyboardToDriverCommand,
    ScriptedModality,
    StepRequest,
    StepRequirements,
    StepResult,
    UserInputEvent,
    UserInputs,
    UserInputSchema,
)
from flashdreams.runtime.demo import (
    ModelInputProvider,
    ModelWarmupPlan,
    PreparedScenario,
    ResamplerRealtimeClock,
    RunResult,
    RuntimeHost,
    WarmupSessionInputs,
    WebRTCErrorPolicy,
)
from flashdreams.runtime.keyboard import WSAD_SUPPORTED_KEYS
from flashdreams.serving.webrtc import manager as manager_module
from flashdreams.serving.webrtc import services as services_module
from flashdreams.serving.webrtc.encoders import ChunkDeliveryResult
from flashdreams.serving.webrtc.manager import (
    BaseWebRTCSessionManager,
    ManagedWebRTCSession,
)
from flashdreams.serving.webrtc.server import SessionBusyError
from flashdreams.serving.webrtc.services import (
    WEBRTC_SKIPPED_INPUTS_METADATA_KEY,
    WEBRTC_SKIPPED_WINDOW_METADATA_KEY,
    WEBRTC_USER_INPUT_SCHEMA,
    ThreadSafeWebRTCOutputBridge,
    WebRTCActivationPolicy,
    WebRTCInputSource,
    WebRTCManagerLifecycle,
    WebRTCTransportService,
)

pytestmark = pytest.mark.ci_cpu


def _runtime_config() -> SimpleNamespace:
    return SimpleNamespace(
        video_width=8,
        video_height=4,
        warmup_chunks=0,
        warmup_timeout_s=1.0,
    )


def _shared_scenario(
    *converters: Any,
    source_schema: UserInputSchema = WEBRTC_USER_INPUT_SCHEMA,
) -> PreparedScenario:
    return PreparedScenario(
        initial_inputs=InferenceInput(),
        source_schema=source_schema,
        canonicalizer=InputCanonicalizer(converters),
    )


def _step_request(step_index: int = 0, input_frame_count: int = 1) -> StepRequest:
    return StepRequest(
        step_index=step_index,
        metadata={"input_frame_count": input_frame_count},
    )


class _FakeVideoTrack:
    fps = 30

    def __init__(self) -> None:
        self.closed = False

    async def enqueue_result(self, result: StepResult) -> int:
        del result
        return 1

    def qsize(self) -> int:
        return 0

    async def close(self) -> None:
        self.closed = True


class _FakeVideoEncoder:
    """``VideoEncoder``-shaped stub for ``ManagedWebRTCSession`` construction
    and the base manager's generation-worker path. ``deliver_chunk``
    delegates to the paired track's ``enqueue_result`` so the manager
    tests that drive one result end-to-end see the frames land."""

    fps = 30
    backend = "fake"
    prefers_codec: str | None = None

    async def deliver_chunk(
        self,
        result: StepResult,
        track: Any,
        *,
        force_keyframe: bool = False,
    ) -> ChunkDeliveryResult:
        del force_keyframe
        enqueued = await track.enqueue_result(result)
        return ChunkDeliveryResult(
            backend=self.backend,
            num_frames=enqueued,
            num_keyframes=0,
            encode_ms=0.1,
        )

    def prepare_chunk_payload(
        self,
        result: StepResult,
        track: Any,
    ) -> StepResult:
        del track
        return result

    async def deliver_prepared_chunk(
        self,
        payload: object,
        track: Any,
        *,
        force_keyframe: bool = False,
    ) -> ChunkDeliveryResult:
        if not isinstance(payload, StepResult):
            raise TypeError("fake payload must be StepResult")
        return await self.deliver_chunk(
            payload,
            track,
            force_keyframe=force_keyframe,
        )

    def close(self) -> None:
        return


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

    def reset(self, *, start_v: float) -> None:
        self.next_chunk_start_v = start_v

    def sample_chunk(self, num_frames: int) -> list[float]:
        start = self.next_chunk_start_v
        frame_times = [start + index * self.dt for index in range(num_frames)]
        self.next_chunk_start_v = start + num_frames * self.dt
        return frame_times


class _RecordingLegacySegmentResampler:
    dt = 0.0
    next_chunk_start_v = 0.0

    def __init__(self) -> None:
        self.edges: list[tuple[float, str, str]] = []

    def reset(self, *, start_v: float) -> None:
        self.next_chunk_start_v = start_v
        self.edges.clear()

    def on_edge(self, *, arrival_t: float, event: str, key: str) -> None:
        self.edges.append((arrival_t, event, key))

    def sample_chunk(
        self, num_frames: int
    ) -> tuple[list[tuple[float, float, frozenset[str]]], list[float]]:
        assert num_frames == 1
        return [(0.0, 0.0, frozenset({"w"}))], [0.0]


class _SharedResampler:
    def __init__(self, *, start_v: float = 0.0, dt: float = 0.001) -> None:
        self.next_chunk_start_v = start_v
        self.dt = dt

    def reset(self, *, start_v: float) -> None:
        self.next_chunk_start_v = start_v

    def sample_chunk(self, num_frames: int) -> list[float]:
        start = self.next_chunk_start_v
        end = start + num_frames * self.dt
        self.next_chunk_start_v = end
        return [end]


class _CountingVideoTrack(_FakeVideoTrack):
    async def enqueue_result(self, result: StepResult) -> int:
        return result.frame_count


class _BaseTestManager(BaseWebRTCSessionManager):
    pass


class _WOnlyTestManager(_BaseTestManager):
    _resampler_supported_keys = frozenset({"w"})


class _RecordingWarmupSession:
    def __init__(self, runtime: _RecordingWarmupRuntime) -> None:
        self.runtime = runtime

    def next_step_request(self) -> StepRequest:
        return _step_request(step_index=len(self.runtime.warmup_step_inputs))

    def step(self, inputs: InferenceInput) -> StepResult:
        self.runtime.warmup_step_inputs.append(inputs)
        return StepResult(
            step_index=len(self.runtime.warmup_step_inputs) - 1,
            output=object(),
        )

    def reset(self, inputs: InferenceInput | None = None) -> None:
        del inputs

    def close(self) -> None:
        self.runtime.warmup_sessions_closed += 1


class _RecordingWarmupRuntime:
    def __init__(self) -> None:
        self.preload_count = 0
        self.warmup_initial_inputs: list[InferenceInput] = []
        self.warmup_step_inputs: list[InferenceInput] = []
        self.warmup_sessions_closed = 0

    def preload(self) -> None:
        self.preload_count += 1

    def start_session(self, inputs: InferenceInput) -> _RecordingWarmupSession:
        self.warmup_initial_inputs.append(inputs)
        return _RecordingWarmupSession(self)

    def close(self) -> None:
        return


class _SpyRuntimeHost(RuntimeHost):
    def __init__(self, runtime: _RecordingWarmupRuntime) -> None:
        super().__init__(runtime)
        self.warmup_calls: list[ModelWarmupPlan] = []

    def warmup(self, plan: ModelWarmupPlan | None = None) -> None:
        resolved_plan = plan or ModelWarmupPlan()
        self.warmup_calls.append(resolved_plan)
        super().warmup(resolved_plan)


def _make_manager(
    manager_cls: type[BaseWebRTCSessionManager], runtime: Any, **kwargs: Any
) -> BaseWebRTCSessionManager:
    return manager_cls(
        runtime=runtime,
        runtime_config=_runtime_config(),
        fps=30,
        identity="fake-model",
        **kwargs,
    )


async def _close_test_manager(manager: BaseWebRTCSessionManager[Any, Any]) -> None:
    """Close a manager without routing synchronous host teardown through a thread."""
    await manager.close_active_session()
    if manager._shared_context is not None:
        await manager._shared_context.close_async()
        manager._shared_context = None
    if manager._shared_host is not None:
        manager._shared_host.close()
        manager._shared_host = None
    await manager.shutdown()


@pytest.mark.asyncio
async def test_nonempty_model_warmup_plan_runs_once_for_server_lifetime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _RecordingWarmupRuntime()
    host = _SpyRuntimeHost(runtime)
    plan = ModelWarmupPlan(
        sessions=(
            WarmupSessionInputs(
                initial_input=InferenceInput(metadata={"session": 0}),
                step_inputs=(InferenceInput(step={"block": 0}),),
            ),
            WarmupSessionInputs(
                initial_input=InferenceInput(metadata={"session": 1}),
                step_inputs=(
                    InferenceInput(step={"block": 1}),
                    InferenceInput(step={"block": 6}),
                ),
            ),
        ),
    )
    config = _runtime_config()
    config.warmup_chunks = 5
    manager = _BaseTestManager(
        runtime=runtime,
        runtime_config=config,
        fps=30,
        identity="fake-model",
        model_warmup_plan=plan,
        shared_host=host,
    )
    loopback_chunks: list[int] = []
    offer_warmup_counts: list[int] = []

    async def run_inline(function: Any, /, *args: Any, **kwargs: Any) -> Any:
        return function(*args, **kwargs)

    async def record_loopback(*, num_chunks: int) -> None:
        loopback_chunks.append(num_chunks)

    async def fake_answer(**kwargs: Any) -> dict[str, str]:
        del kwargs
        offer_warmup_counts.append(len(host.warmup_calls))
        return {"sdp": "answer", "type": "answer"}

    monkeypatch.setattr(manager_module.asyncio, "to_thread", run_inline)
    monkeypatch.setattr(manager, "_run_loopback_warmup_session", record_loopback)
    monkeypatch.setattr(
        manager,
        "_create_answer_with_runtime_ready_locked",
        fake_answer,
    )

    try:
        await manager.preload_runtime()
        context = manager._shared_run_context(asyncio.get_running_loop())
        await manager.create_answer(offer_sdp="first", offer_type="offer")
        await manager.create_answer(offer_sdp="later", offer_type="offer")
        await manager.preload_runtime()

        assert context.model_warmup_plan is plan
        assert host.warmup_calls == [plan]
        assert len(runtime.warmup_initial_inputs) == len(plan.sessions)
        assert len(runtime.warmup_step_inputs) == 3
        assert runtime.warmup_sessions_closed == len(plan.sessions)
        assert runtime.preload_count == 1
        assert loopback_chunks == [0]
        assert offer_warmup_counts == [1, 1]
    finally:
        await _close_test_manager(manager)


@pytest.mark.asyncio
async def test_empty_model_warmup_plan_keeps_loopback_model_steps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _RecordingWarmupRuntime()
    host = _SpyRuntimeHost(runtime)
    plan = ModelWarmupPlan()
    config = _runtime_config()
    config.warmup_chunks = 4
    manager = _BaseTestManager(
        runtime=runtime,
        runtime_config=config,
        fps=30,
        identity="fake-model",
        model_warmup_plan=plan,
        shared_host=host,
    )
    loopback_chunks: list[int] = []

    async def run_inline(function: Any, /, *args: Any, **kwargs: Any) -> Any:
        return function(*args, **kwargs)

    async def record_loopback(*, num_chunks: int) -> None:
        loopback_chunks.append(num_chunks)

    monkeypatch.setattr(manager_module.asyncio, "to_thread", run_inline)
    monkeypatch.setattr(manager, "_run_loopback_warmup_session", record_loopback)

    try:
        await manager.preload_runtime()
        context = manager._shared_run_context(asyncio.get_running_loop())
        await manager.preload_runtime()

        assert context.model_warmup_plan is plan
        assert host.warmup_calls == []
        assert runtime.warmup_initial_inputs == []
        assert runtime.warmup_step_inputs == []
        assert runtime.preload_count == 1
        assert loopback_chunks == [4]
    finally:
        await _close_test_manager(manager)


def test_drives_inference_session_detects_session_runtime() -> None:
    class _SessionRuntime:
        async def start_inference_session(self) -> object:
            return object()

    assert BaseWebRTCSessionManager._drives_inference_session(_SessionRuntime())
    assert not BaseWebRTCSessionManager._drives_inference_session(object())


def test_runtime_frame_timing_contract() -> None:
    class _Runtime:
        def peek_input_fps(self) -> float:
            return 30.0

        def next_step_request(self) -> StepRequest:
            return _step_request(input_frame_count=2)

        def peek_steady_output_num_frames(self) -> int:
            return 3

    runtime = _Runtime()
    manager = _make_manager(_BaseTestManager, runtime)

    assert manager._runtime_input_fps(runtime) == pytest.approx(30.0)
    assert manager._runtime_next_step_request(runtime) == (
        _step_request(input_frame_count=2),
        2,
    )
    assert manager._runtime_steady_output_num_frames(runtime) == 3


def test_runtime_frame_timing_hooks_can_split_input_and_output() -> None:
    class _SplitRuntime:
        def peek_input_fps(self) -> float:
            return 6.0

        def next_step_request(self) -> StepRequest:
            return _step_request(input_frame_count=4)

        def peek_steady_output_num_frames(self) -> int:
            return 16

    runtime = _SplitRuntime()
    manager = _make_manager(_BaseTestManager, runtime)
    resampler = manager._make_resampler_at_fps(
        start_v=0.0,
        fps=manager._runtime_input_fps(runtime),
    )

    assert resampler.dt == pytest.approx(1.0 / 6.0)
    assert manager._runtime_next_step_request(runtime) == (
        _step_request(input_frame_count=4),
        4,
    )
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
        video_encoder=_FakeVideoEncoder(),  # ty:ignore[invalid-argument-type]
        peer_connection=peer,
        resampler=_FakeResampler(),  # ty:ignore[invalid-argument-type]
        control_channel=channel,
        first_action_received=first_action,
    )
    return managed, video_track, peer, channel


def test_record_user_event_rejects_full_queue(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(manager_module, "_MAX_SESSION_USER_EVENTS", 2)
    runtime = object()
    manager = _make_manager(_BaseTestManager, runtime)
    managed, _video_track, _peer, _channel = _managed_session(runtime)

    for index in range(2):
        manager._record_user_event(
            managed_session=managed,
            timestamp_s=float(index),
            event_type="key_down",
            payload={"key": "w"},
        )

    with pytest.raises(RuntimeError, match="Too many queued WebRTC user events"):
        manager._record_user_event(
            managed_session=managed,
            timestamp_s=2.0,
            event_type="key_down",
            payload={"key": "w"},
        )

    assert len(managed.user_events) == 2


def test_record_user_event_keeps_release_when_queue_full(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(manager_module, "_MAX_SESSION_USER_EVENTS", 1)
    runtime = object()
    manager = _make_manager(_BaseTestManager, runtime)
    managed, _video_track, _peer, _channel = _managed_session(runtime)
    manager._record_user_event(
        managed_session=managed,
        timestamp_s=0.0,
        event_type="key_down",
        payload={"key": "ArrowUp"},
    )

    manager._record_user_event(
        managed_session=managed,
        timestamp_s=0.1,
        event_type="key_up",
        payload={"key": "w"},
    )

    assert len(managed.user_events) == 1
    assert managed.user_events[0].event_type == "key_up"
    assert managed.user_events[0].payload["key"] == "w"


def test_record_user_event_does_not_evict_unrelated_event_for_release(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(manager_module, "_MAX_SESSION_USER_EVENTS", 1)
    runtime = object()
    manager = _make_manager(_BaseTestManager, runtime)
    managed, _video_track, _peer, _channel = _managed_session(runtime)
    manager._record_user_event(
        managed_session=managed,
        timestamp_s=0.0,
        event_type="text_event",
        payload={"event_id": "storm"},
    )

    manager._record_user_event(
        managed_session=managed,
        timestamp_s=0.1,
        event_type="key_up",
        payload={"key": "w"},
    )
    manager._record_user_event(
        managed_session=managed,
        timestamp_s=0.2,
        event_type="key_up",
        payload={"key": "w"},
    )

    assert [
        (event.event_type, dict(event.payload)) for event in managed.user_events
    ] == [("text_event", {"event_id": "storm"})]
    assert len(managed.user_events) == 1
    assert set(managed.coalesced_release_events) == {"w"}
    assert managed.coalesced_release_events["w"].timestamp_s == pytest.approx(0.2)
    assert [
        (event.event_type, dict(event.payload))
        for event in manager._pending_user_events(managed)
    ] == [
        ("text_event", {"event_id": "storm"}),
        ("key_up", {"key": "w"}),
    ]


def test_record_user_event_ignores_unsupported_key_events(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(manager_module, "_MAX_SESSION_USER_EVENTS", 1)
    runtime = object()
    manager = _make_manager(_WOnlyTestManager, runtime)
    managed, _video_track, _peer, _channel = _managed_session(runtime)
    manager._record_user_event(
        managed_session=managed,
        timestamp_s=0.0,
        event_type="text_event",
        payload={"event_id": "storm"},
    )

    manager._record_user_event(
        managed_session=managed,
        timestamp_s=0.1,
        event_type="key_up",
        payload={"key": "z"},
    )
    manager._record_user_event(
        managed_session=managed,
        timestamp_s=0.2,
        event_type="key_down",
        payload={"key": "z"},
    )

    assert [
        (event.event_type, dict(event.payload)) for event in managed.user_events
    ] == [("text_event", {"event_id": "storm"})]


def test_catch_up_input_clock_advances_session_input_state() -> None:
    class _RecordingCanonicalizer:
        def __init__(self) -> None:
            self.windows: list[tuple[float, float]] = []
            self.event_batches: list[list[tuple[float, str]]] = []

        def canonicalize(
            self,
            user_inputs: Any,
            *,
            window: Any,
            source_schema: Any,
        ) -> object:
            del source_schema
            self.windows.append((window.start_s, window.end_s))
            self.event_batches.append(
                [(event.timestamp_s, event.event_type) for event in user_inputs.events]
            )
            return object()

    canonicalizer = _RecordingCanonicalizer()
    runtime = SimpleNamespace(
        input_canonicalizer=canonicalizer,
        input_source_schema=object(),
    )
    manager = _make_manager(_BaseTestManager, runtime)
    managed, _video_track, _peer, _channel = _managed_session(runtime)
    managed.inference_session = object()
    managed.resampler.next_chunk_start_v = 0.0
    manager._record_user_event(
        managed_session=managed,
        timestamp_s=0.5,
        event_type="key_up",
        payload={"key": "w"},
    )
    manager._record_user_event(
        managed_session=managed,
        timestamp_s=2.0,
        event_type="key_down",
        payload={"key": "w"},
    )

    manager._catch_up_input_clock(
        managed_session=managed,
        now=3.0,
        chunk_duration=1.0,
    )

    assert managed.resampler.next_chunk_start_v == pytest.approx(2.0)
    assert managed.session_input_state_advanced
    assert canonicalizer.windows == [(0.0, 2.0)]
    assert canonicalizer.event_batches == [[(0.5, "key_up"), (2.0, "key_down")]]
    assert [(event.timestamp_s, event.event_type) for event in managed.user_events] == [
        (pytest.approx(2.0), "key_down")
    ]


def test_catch_up_input_clock_snaps_legacy_path_without_canonicalizer() -> None:
    runtime = object()
    manager = _make_manager(_BaseTestManager, runtime)
    managed, _video_track, _peer, _channel = _managed_session(runtime)
    managed.resampler.next_chunk_start_v = 0.0

    manager._catch_up_input_clock(
        managed_session=managed,
        now=3.0,
        chunk_duration=1.0,
    )

    assert managed.resampler.next_chunk_start_v == pytest.approx(2.0)


def test_legacy_provider_advances_skipped_webrtc_input_state() -> None:
    class _RecordingCanonicalizer:
        def __init__(self) -> None:
            self.windows: list[tuple[float, float]] = []
            self.event_batches: list[list[str]] = []

        def canonicalize(
            self,
            user_inputs: UserInputs,
            *,
            window: Any,
            source_schema: Any,
        ) -> object:
            del source_schema
            self.windows.append((window.start_s, window.end_s))
            self.event_batches.append(
                [event.event_type for event in user_inputs.events]
            )
            return object()

    class _RecordingMapping:
        def __init__(self) -> None:
            self.inference_inputs: list[InferenceInput] = []

        def map_step_inputs(
            self,
            *,
            canonical_inputs: object,
            inference_input: InferenceInput,
            request: StepRequest,
        ) -> InferenceInput:
            del canonical_inputs
            self.inference_inputs.append(inference_input)
            return InferenceInput(step={"mapped_step": request.step_index})

    mapping = _RecordingMapping()
    runtime = SimpleNamespace(
        start_inference_session=lambda: object(),
        input_canonicalizer=_RecordingCanonicalizer(),
        input_source_schema=object(),
        input_mapping=mapping,
    )
    provider = manager_module._LegacyWebRTCModelInputProvider(runtime=runtime)
    skipped_inputs = UserInputs(
        events=(
            UserInputEvent(
                timestamp_s=0.5,
                event_type="key_down",
                payload={"key": "w"},
            ),
        )
    )
    current_inputs = UserInputs(
        events=(
            UserInputEvent(
                timestamp_s=2.5,
                event_type="key_up",
                payload={"key": "w"},
            ),
        )
    )

    prepared = provider.prepare_step(
        request=StepRequirements(step_index=0, input_frame_count=1),
        user_window=manager_module.UserInputWindow(
            start_s=2.0,
            end_s=3.0,
            frame_times=(2.25, 2.75),
            inputs=current_inputs,
            metadata={
                manager_module._LEGACY_SPARSE_KEY_SEGMENTS_METADATA_KEY: (
                    (2.0, 3.0, frozenset({"w"})),
                ),
                WEBRTC_SKIPPED_INPUTS_METADATA_KEY: skipped_inputs,
                WEBRTC_SKIPPED_WINDOW_METADATA_KEY: (0.0, 2.0),
            },
        ),
    )

    assert prepared.inference_input == InferenceInput(step={"mapped_step": 0})
    assert runtime.input_canonicalizer.windows == [(0.0, 2.0), (2.0, 3.0)]
    assert runtime.input_canonicalizer.event_batches == [["key_down"], ["key_up"]]
    assert mapping.inference_inputs[0].metadata["frame_times"] == (2.25, 2.75)
    assert mapping.inference_inputs[0].metadata["window_start_s"] == 2.0
    assert mapping.inference_inputs[0].metadata["window_end_s"] == 3.0
    assert mapping.inference_inputs[0].metadata[
        manager_module._LEGACY_SPARSE_KEY_SEGMENTS_METADATA_KEY
    ] == ((2.0, 3.0, frozenset({"w"})),)


@pytest.mark.asyncio
async def test_action_keydown_reports_error_when_user_event_queue_full(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(manager_module, "_MAX_SESSION_USER_EVENTS", 1)
    runtime = object()
    manager = _make_manager(_BaseTestManager, runtime)
    managed, _video_track, _peer, channel = _managed_session(runtime)
    managed.inference_session = object()
    managed.first_action_received.clear()
    manager._record_user_event(
        managed_session=managed,
        timestamp_s=0.0,
        event_type="key_down",
        payload={"key": "w"},
    )

    await manager._handle_datachannel_message(
        managed_session=managed,
        raw_message='{"type":"action","action":{"event":"keydown","key":"w"}}',
    )

    assert len(managed.user_events) == 1
    assert not managed.first_action_received.is_set()
    assert [json.loads(message) for message in channel.messages] == [
        {
            "type": "error",
            "message": (
                "Too many queued WebRTC user events; wait for inference to catch up."
            ),
        }
    ]


@pytest.mark.asyncio
async def test_action_keyup_updates_state_when_user_event_queue_full(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(manager_module, "_MAX_SESSION_USER_EVENTS", 1)
    runtime = object()
    manager = _make_manager(_BaseTestManager, runtime)
    managed, _video_track, _peer, channel = _managed_session(runtime)
    managed.inference_session = object()
    managed.first_action_received.clear()
    resampler = _RecordingLegacySegmentResampler()
    managed.legacy_segment_resampler = resampler
    manager._record_user_event(
        managed_session=managed,
        timestamp_s=0.0,
        event_type="key_down",
        payload={"key": "w"},
    )

    await manager._handle_datachannel_message(
        managed_session=managed,
        raw_message='{"type":"action","action":{"event":"keyup","key":"w"}}',
    )

    assert [
        (event.event_type, dict(event.payload)) for event in managed.user_events
    ] == [("key_up", {"key": "w"})]
    assert managed.first_action_received.is_set()
    assert len(managed.pending_action_arrivals) == 1
    assert len(resampler.edges) == 1
    assert resampler.edges[0][1:] == ("keyup", "w")
    assert channel.messages == []


@pytest.mark.asyncio
async def test_session_event_message_validates_records_and_activates() -> None:
    class _Runtime:
        def __init__(self) -> None:
            self.validate_calls: list[tuple[str, dict[str, Any]]] = []

        def validate_user_event(
            self, *, event_type: str, payload: dict[str, Any]
        ) -> dict[str, Any]:
            self.validate_calls.append((event_type, dict(payload)))
            return {
                "event_id": payload["event_id"],
                "state": payload["state"],
                "validated": True,
            }

    runtime = _Runtime()
    manager = _make_manager(_BaseTestManager, runtime)
    managed, _video_track, _peer, channel = _managed_session(runtime)
    managed.inference_session = object()
    managed.first_action_received.clear()

    await manager._handle_datachannel_message(
        managed_session=managed,
        raw_message=json.dumps(
            {"type": "event", "event_id": "storm", "state": "trigger"}
        ),
    )

    assert runtime.validate_calls == [
        ("text_event", {"event_id": "storm", "state": "trigger"})
    ]
    assert [
        (event.event_type, dict(event.payload)) for event in managed.user_events
    ] == [
        (
            "text_event",
            {"event_id": "storm", "state": "trigger", "validated": True},
        )
    ]
    assert managed.first_action_received.is_set()
    assert [json.loads(message) for message in channel.messages] == [
        {
            "type": "event_ack",
            "event_id": "storm",
            "state": "trigger",
            "active_event_id": "storm",
        }
    ]


@pytest.mark.asyncio
async def test_generation_worker_closes_session_when_flag_set() -> None:
    class _ClosingRuntime:
        def __init__(self) -> None:
            self.generate_calls = 0

        def next_step_request(self) -> StepRequest:
            return _step_request(step_index=self.generate_calls)

        async def step(
            self, *, request: StepRequest, segments: Any, frame_times: Any
        ) -> StepResult:
            del request, segments, frame_times
            self.generate_calls += 1
            raise RuntimeError("boom")

    runtime = _ClosingRuntime()
    manager = _make_manager(
        _BaseTestManager,
        runtime,
        fatal_generation_errors=True,
    )
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

        def next_step_request(self) -> StepRequest:
            return _step_request(step_index=self.generate_calls)

        async def step(
            self, *, request: StepRequest, segments: Any, frame_times: Any
        ) -> StepResult:
            del request, segments, frame_times
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
async def test_generation_worker_closes_completed_inference_session_without_retry() -> (
    None
):
    class _CompletedSession:
        def __init__(self) -> None:
            self.calls = 0

        def next_step_request(self) -> StepRequest | None:
            self.calls += 1
            return None

        def step(self, inputs: Any) -> StepResult:
            del inputs
            raise AssertionError("completed sessions must not be stepped")

    class _SessionRuntime:
        def __init__(self) -> None:
            self.session = _CompletedSession()

        def next_step_request(self) -> StepRequest:
            return _step_request()

    runtime = _SessionRuntime()
    manager = _make_manager(_BaseTestManager, runtime)
    managed, video_track, peer, channel = _managed_session(runtime)
    managed.inference_session = runtime.session
    resampler = cast(_FakeResampler, managed.resampler)
    resampler.dt = 1.0 / 30.0
    resampler.next_chunk_start_v = asyncio.get_running_loop().time()
    manager._active_session = managed

    task = asyncio.create_task(manager._generation_worker(managed_session=managed))
    managed.generation_task = task
    await asyncio.wait_for(task, timeout=5.0)

    assert runtime.session.calls == 1
    assert not manager.has_active_session()
    assert managed.closed
    assert video_track.closed
    assert peer.closed
    assert channel.messages == []


@pytest.mark.asyncio
async def test_chunk_done_payload_includes_model_and_extra() -> None:
    class _OneChunkRuntime:
        def __init__(self) -> None:
            self.managed_session: ManagedWebRTCSession | None = None

        def next_step_request(self) -> StepRequest:
            return _step_request()

        async def step(
            self, *, request: StepRequest, segments: Any, frame_times: Any
        ) -> StepResult:
            del request, segments, frame_times
            if self.managed_session is not None:
                self.managed_session.closed = True
            return StepResult.from_video_chunk(
                step_index=0,
                video_chunk=torch.zeros((1, 1, 1, 3, 2, 2), dtype=torch.uint8),
                layout="bvtchw",
                metadata={"stream": "rgb"},
            )

    runtime = _OneChunkRuntime()
    manager = _make_manager(_BaseTestManager, runtime)
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

        def next_step_request(self) -> StepRequest:
            return _step_request(input_frame_count=2)

        async def step(
            self,
            *,
            request: StepRequest,
            segments: Any,
            frame_times: list[float],
        ) -> StepResult:
            del request, segments
            self.frame_times = frame_times
            if self.managed_session is not None:
                self.managed_session.closed = True
            return StepResult.from_video_chunk(
                step_index=0,
                video_chunk=torch.zeros((5, 3, 2, 2), dtype=torch.uint8),
                layout="tchw",
            )

    runtime = _SplitRuntime()
    manager = _make_manager(_BaseTestManager, runtime)
    managed, _video_track, _peer, channel = _managed_session(runtime)
    resampler = _SplitResampler()
    managed.video_track = _CountingVideoTrack()  # ty:ignore[invalid-assignment]
    managed.legacy_segment_resampler = resampler
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

        def next_step_request(self) -> StepRequest:
            return _step_request(step_index=self.chunk_index)

        async def step(
            self, *, request: StepRequest, segments: Any, frame_times: Any
        ) -> StepResult:
            del request, segments, frame_times
            chunk_index = self.chunk_index
            self.chunk_index += 1
            if chunk_index >= 2 and self.managed_session is not None:
                self.managed_session.closed = True
            return StepResult.from_video_chunk(
                step_index=chunk_index,
                video_chunk=torch.zeros((4, 3, 2, 2), dtype=torch.uint8),
                layout="tchw",
                metrics={
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
async def test_realtime_driver_session_uses_shared_step_pipeline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipeline_calls = 0
    original_pipeline = manager_module.StepPipeline

    class _RecordingPipeline(original_pipeline):
        def execute_step(
            self,
            *,
            request: StepRequirements,
            user_window: Any,
            provider: Any,
            session: Any,
            output: Any,
            metrics: Any,
        ) -> Any:
            nonlocal pipeline_calls
            pipeline_calls += 1
            return original_pipeline.execute_step(
                self,
                request=request,
                user_window=user_window,
                provider=provider,
                session=session,
                output=output,
                metrics=metrics,
            )

    class _SharedRuntime:
        def __init__(self) -> None:
            self.step_requests = 0
            self.step_calls: list[tuple[int, list[Any], list[float]]] = []

        async def reset_for_new_session(self, session_input: Any = None) -> None:
            del session_input

        def next_step_request(self) -> StepRequest | None:
            if self.step_requests > 0:
                return None
            self.step_requests += 1
            return _step_request(step_index=0, input_frame_count=1)

        async def step(
            self,
            *,
            request: StepRequest,
            segments: list[Any],
            frame_times: list[float],
        ) -> StepResult:
            self.step_calls.append((request.step_index, segments, frame_times))
            return StepResult(step_index=request.step_index, output="ok", frame_count=1)

        def peek_input_fps(self) -> float:
            return 30.0

        def peek_steady_output_num_frames(self) -> int:
            return 1

    monkeypatch.setattr(manager_module, "StepPipeline", _RecordingPipeline)
    runtime = _SharedRuntime()
    manager = _make_manager(_BaseTestManager, runtime)
    context = manager._shared_run_context(asyncio.get_running_loop())
    reservation = context.admission.try_reserve()
    assert reservation is not None
    resampler = _SharedResampler(start_v=asyncio.get_running_loop().time())
    input_source = WebRTCInputSource(
        resampler=resampler,
        legacy_segment_resampler=_RecordingLegacySegmentResampler(),
        legacy_segments_metadata_key=(
            manager_module._LEGACY_SPARSE_KEY_SEGMENTS_METADATA_KEY
        ),
    )
    input_source.handle_browser_payload(
        {"type": "action", "action": {"event": "step"}},
        timestamp_s=asyncio.get_running_loop().time(),
    )
    managed, video_track, peer, channel = _managed_session(runtime)
    managed.resampler = resampler  # ty:ignore[invalid-assignment]
    managed.input_source = input_source
    managed.transport = WebRTCTransportService(loop=asyncio.get_running_loop())
    managed.reservation = reservation
    manager._active_session = managed

    managed.generation_task = asyncio.create_task(
        manager._run_realtime_driver_session(
            managed_session=managed,
            context=context,
            session_input=None,
        )
    )
    await asyncio.wait_for(managed.generation_task, timeout=5.0)

    assert pipeline_calls == 1
    assert runtime.step_calls
    assert runtime.step_calls[0][0] == 0
    assert not manager.has_active_session()
    assert video_track.closed
    assert peer.closed
    chunk_done = [
        json.loads(message)
        for message in channel.messages
        if json.loads(message).get("type") == "chunk_done"
    ]
    assert len(chunk_done) == 1
    assert chunk_done[0]["model"] == "fake-model"


@pytest.mark.asyncio
async def test_realtime_driver_session_reports_non_completed_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_run_demo_session_async(**kwargs: Any) -> RunResult:
        del kwargs
        return RunResult(
            status="not_activated",
            reason="transport closed before first step",
        )

    monkeypatch.setattr(
        manager_module,
        "run_demo_session_async",
        fake_run_demo_session_async,
    )
    runtime = SimpleNamespace()
    manager = _make_manager(_BaseTestManager, runtime)
    context = manager._shared_run_context(asyncio.get_running_loop())
    reservation = context.admission.try_reserve()
    assert reservation is not None
    managed, _video_track, _peer, channel = _managed_session(runtime)
    managed.reservation = reservation
    manager._active_session = managed

    await manager._run_realtime_driver_session(
        managed_session=managed,
        context=context,
        session_input=None,
    )

    assert json.loads(channel.messages[0]) == {
        "type": "error",
        "message": "transport closed before first step",
    }
    assert not manager.has_active_session()


@pytest.mark.asyncio
async def test_base_manager_accepts_sequential_completed_peers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_count = 0

    async def fake_run_demo_session_async(**kwargs: Any) -> RunResult:
        nonlocal run_count
        run_count += 1
        reservation = kwargs["reservation"]
        assert reservation is not None
        reservation.release()
        return RunResult(status="completed")

    monkeypatch.setattr(
        manager_module,
        "run_demo_session_async",
        fake_run_demo_session_async,
    )
    runtime = SimpleNamespace()
    manager = _make_manager(_BaseTestManager, runtime)
    context = manager._shared_run_context(asyncio.get_running_loop())
    peers: list[_FakePeerConnection] = []

    for _ in range(2):
        reservation = context.admission.try_reserve()
        assert reservation is not None
        managed, track, peer, _channel = _managed_session(runtime)
        managed.reservation = reservation
        manager._active_session = managed

        await manager._run_realtime_driver_session(
            managed_session=managed,
            context=context,
            session_input=None,
        )

        assert track.closed
        assert peer.closed
        assert not manager.has_active_session()
        peers.append(peer)

    assert run_count == 2
    assert all(peer.closed for peer in peers)
    await _close_test_manager(manager)


@pytest.mark.asyncio
async def test_keep_connection_starts_next_generation_before_track_close(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_count = 0
    runtime = SimpleNamespace()
    manager = _make_manager(
        _BaseTestManager,
        runtime,
        keep_connection_after_completed=True,
    )
    context = manager._shared_run_context(asyncio.get_running_loop())
    reservation = context.admission.try_reserve()
    assert reservation is not None
    managed, track, peer, channel = _managed_session(runtime)
    input_source = WebRTCInputSource(resampler=managed.resampler)
    transport = WebRTCTransportService(loop=asyncio.get_running_loop())
    managed.input_source = input_source
    managed.transport = transport
    managed.reservation = reservation
    manager._active_session = managed

    async def fake_run_demo_session_async(**kwargs: Any) -> RunResult:
        nonlocal run_count
        run_count += 1
        if run_count == 1:
            first_reservation = kwargs["reservation"]
            assert first_reservation is not None
            first_reservation.release()
            generation_edges = kwargs["run_mode"].create_session_edges(
                context=kwargs["context"],
                spec=kwargs["spec"],
                scenario=kwargs["scenario"],
                provider=cast(ModelInputProvider, SimpleNamespace()),
                adapter=kwargs["adapter"],
            )
            result = generation_edges.close_result()
            assert transport.is_active()

            async def activate_next_generation() -> None:
                await asyncio.sleep(0)
                input_source.activation_signal.set()

            asyncio.create_task(activate_next_generation())
            return result
        assert not track.closed
        assert not peer.closed
        transport.disconnect("transport closed")
        return RunResult(status="not_activated", reason="transport closed")

    monkeypatch.setattr(
        manager_module,
        "run_demo_session_async",
        fake_run_demo_session_async,
    )

    await manager._run_realtime_driver_session(
        managed_session=managed,
        context=context,
        session_input=None,
    )

    assert run_count == 2
    assert any(
        json.loads(message).get("type") == "generation_complete"
        for message in channel.messages
    )
    assert track.closed
    assert peer.closed
    await _close_test_manager(manager)


@pytest.mark.asyncio
async def test_next_generation_wait_cancels_children_with_parent() -> None:
    class _TrackingSignal:
        def __init__(self) -> None:
            self.started = asyncio.Event()
            self.cancelled = asyncio.Event()
            self._event = asyncio.Event()

        async def wait(self) -> None:
            self.started.set()
            try:
                await self._event.wait()
            except asyncio.CancelledError:
                self.cancelled.set()
                raise

    activation_signal = _TrackingSignal()
    closed_signal = _TrackingSignal()
    input_source = cast(
        WebRTCInputSource,
        SimpleNamespace(activation_signal=activation_signal),
    )
    transport = cast(
        WebRTCTransportService,
        SimpleNamespace(
            closed_signal=closed_signal,
            is_active=lambda: True,
        ),
    )
    parent = asyncio.create_task(
        manager_module._wait_for_next_generation_or_disconnect(
            input_source=input_source,
            transport=transport,
        )
    )
    await asyncio.wait_for(
        asyncio.gather(
            activation_signal.started.wait(),
            closed_signal.started.wait(),
        ),
        timeout=1.0,
    )

    parent.cancel()
    with pytest.raises(asyncio.CancelledError):
        await parent

    assert activation_signal.cancelled.is_set()
    assert closed_signal.cancelled.is_set()


@pytest.mark.asyncio
async def test_create_answer_raises_busy_with_subclass_message() -> None:
    manager = _make_manager(
        _BaseTestManager,
        runtime=SimpleNamespace(),
        busy_message="custom busy message",
    )
    manager._runtime_ready = True
    manager._warmup_complete = True
    existing, *_ = _managed_session(runtime=SimpleNamespace())
    manager._active_session = existing

    with pytest.raises(SessionBusyError, match="custom busy message"):
        await manager.create_answer(offer_sdp="x", offer_type="offer")


def test_supported_key_payload_honors_configured_keys() -> None:
    wsad_manager = _make_manager(
        _BaseTestManager,
        runtime=SimpleNamespace(),
        supported_control_keys=WSAD_SUPPORTED_KEYS,
    )
    assert not wsad_manager._supports_key_payload({"key": "q"})
    assert wsad_manager._supports_key_payload({"key": "ArrowUp"})

    default_manager = _make_manager(_BaseTestManager, runtime=SimpleNamespace())
    assert default_manager._supports_key_payload({"key": "q"})


@pytest.mark.asyncio
async def test_step_action_starts_generation_without_key_edge() -> None:
    runtime = SimpleNamespace()
    manager = _make_manager(_BaseTestManager, runtime)
    managed, _video_track, _peer, _channel = _managed_session(runtime)
    managed.first_action_received.clear()
    resampler = _RecordingLegacySegmentResampler()
    managed.legacy_segment_resampler = resampler

    await manager._handle_datachannel_message(
        managed_session=managed,
        raw_message=json.dumps({"type": "action", "action": {"event": "step"}}),
    )

    assert managed.first_action_received.is_set()
    assert len(managed.pending_action_arrivals) == 1
    assert resampler.edges == []


def test_resolve_video_encoder_defaults_to_software_when_runtime_lacks_encoder() -> (
    None
):
    """Runtimes that do not expose ``video_encoder`` must still get a
    working session — the base manager falls back to a session-scope
    :class:`DefaultRTCEncoder` rather than raising ``AttributeError``
    from ``_create_answer_with_runtime_ready_locked``."""

    from flashdreams.serving.webrtc.encoders import DefaultRTCEncoder

    class _RuntimeWithoutEncoder:
        """Deliberately no ``video_encoder`` attribute."""

    manager = _make_manager(_BaseTestManager, _RuntimeWithoutEncoder())
    encoder = manager._resolve_video_encoder()

    assert isinstance(encoder, DefaultRTCEncoder)
    assert encoder.fps == 30


def test_resolve_video_encoder_uses_runtime_encoder_when_present() -> None:
    """Runtimes that own their encoder (e.g. omnidreams) have their
    choice honoured — the base manager does not second-guess."""
    provided = _FakeVideoEncoder()

    class _RuntimeWithEncoder:
        video_encoder = provided

    manager = _make_manager(_BaseTestManager, _RuntimeWithEncoder())
    assert manager._resolve_video_encoder() is provided


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("keep_connection", "close_track"),
    ((False, True), (True, False)),
)
async def test_shared_session_edges_use_base_webrtc_services(
    keep_connection: bool,
    close_track: bool,
) -> None:
    runtime = SimpleNamespace()
    manager = _make_manager(
        _BaseTestManager,
        runtime,
        keep_connection_after_completed=keep_connection,
        activate_without_input=True,
    )
    context = manager._shared_run_context(asyncio.get_running_loop())
    managed, _track, _peer, _channel = _managed_session(runtime)
    managed.input_source = WebRTCInputSource(resampler=managed.resampler)
    managed.transport = WebRTCTransportService(loop=asyncio.get_running_loop())
    manager._active_session = managed
    edge_factory = manager_module._ManagedWebRTCSessionEdgeFactory(
        manager=manager,
        managed_session=managed,
        loop=asyncio.get_running_loop(),
    )

    edges = edge_factory.create_session_edges(
        context=context,
        spec=SimpleNamespace(),
        scenario=cast(PreparedScenario, SimpleNamespace()),
        provider=cast(ModelInputProvider, SimpleNamespace()),
        adapter=SimpleNamespace(),
    )
    output_sink = cast(WebRTCOutputSink, edges.output_sink)
    bridge = output_sink._bridge

    assert edges.transport is not managed.transport
    assert isinstance(edges.transport, manager_module._GenerationTransportView)
    assert edges.transport.peer_transport is managed.transport
    assert isinstance(edges.error_policy, WebRTCErrorPolicy)
    assert isinstance(edges.clock, ResamplerRealtimeClock)
    assert isinstance(edges.activation, WebRTCActivationPolicy)
    assert edges.activation.activate_without_input
    assert isinstance(bridge, ThreadSafeWebRTCOutputBridge)
    assert bridge._close_track is close_track

    result = edges.close_result()
    assert result.status == "completed"
    assert managed.transport.is_active()
    assert not managed.peer_connection.closed
    await _close_test_manager(manager)


@pytest.mark.asyncio
async def test_application_liveness_policy_survives_old_ten_second_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lifecycle = WebRTCManagerLifecycle(
        busy_message="busy",
        client_liveness_timeout_s=30.0,
    )
    loop = asyncio.get_running_loop()
    stopped = False
    close_calls = 0

    async def stop_after_first_check(delay_s: float) -> None:
        nonlocal stopped
        assert delay_s == pytest.approx(1.0)
        stopped = True

    async def close_session() -> None:
        nonlocal close_calls
        close_calls += 1

    monkeypatch.setattr(services_module.asyncio, "sleep", stop_after_first_check)
    await lifecycle.watch_client_liveness(
        is_closed=lambda: stopped,
        last_message_at=lambda: loop.time() - 11.0,
        close_session=close_session,
    )

    assert close_calls == 0


def test_base_manager_advertises_feedable_driver_keys() -> None:
    manager = _make_manager(
        _BaseTestManager,
        SimpleNamespace(),
        shared_scenario=_shared_scenario(KeyboardToDriverCommand()),
    )

    assert manager.browser_ui_config() == {
        "accepted_keys": [
            "a",
            "d",
            "down",
            "left",
            "right",
            "s",
            "space",
            "up",
            "w",
        ]
    }
    assert manager._effective_supported_control_keys() == DRIVING_SUPPORTED_KEYS


def test_base_manager_omits_keys_without_feedable_advertisement() -> None:
    no_scenario = _make_manager(_BaseTestManager, SimpleNamespace())
    no_converter = _make_manager(
        _BaseTestManager,
        SimpleNamespace(),
        shared_scenario=_shared_scenario(),
    )
    unfeedable = _make_manager(
        _BaseTestManager,
        SimpleNamespace(),
        shared_scenario=_shared_scenario(
            KeyboardToDriverCommand(),
            source_schema=UserInputSchema(),
        ),
    )
    no_advertisement = _make_manager(
        _BaseTestManager,
        SimpleNamespace(),
        shared_scenario=_shared_scenario(
            ScriptedModality(modality=DRIVER_COMMAND, timeline=()),
        ),
    )

    for manager in (no_scenario, no_converter, unfeedable, no_advertisement):
        assert manager.browser_ui_config() == {"accepted_keys": []}
    assert no_advertisement._effective_supported_control_keys() is None
    assert no_advertisement._supports_key_payload({"key": "q"})


def test_explicit_supported_keys_override_converter_metadata() -> None:
    manager = _make_manager(
        _BaseTestManager,
        SimpleNamespace(),
        shared_scenario=_shared_scenario(KeyboardToDriverCommand()),
        supported_control_keys=frozenset({"q"}),
    )

    assert manager._effective_supported_control_keys() == frozenset({"q"})
    assert manager._supports_key_payload({"key": "q"})
    assert not manager._supports_key_payload({"key": "space"})


def test_legacy_supported_keys_override_converter_metadata() -> None:
    manager = _make_manager(
        _WOnlyTestManager,
        SimpleNamespace(),
        shared_scenario=_shared_scenario(KeyboardToDriverCommand()),
    )

    assert manager._effective_supported_control_keys() == frozenset({"w"})
    assert manager._supports_key_payload({"key": "w"})
    assert not manager._supports_key_payload({"key": "space"})


@pytest.mark.asyncio
async def test_shared_key_filter_runs_before_activation() -> None:
    runtime = SimpleNamespace()
    manager = _make_manager(
        _BaseTestManager,
        runtime,
        shared_scenario=_shared_scenario(KeyboardToDriverCommand()),
    )
    managed, _track, _peer, channel = _managed_session(runtime)
    managed.first_action_received.clear()
    input_source = WebRTCInputSource(resampler=managed.resampler)
    managed.input_source = input_source

    await manager._handle_datachannel_message(
        managed_session=managed,
        raw_message=json.dumps(
            {"type": "action", "action": {"event": "keydown", "key": "q"}}
        ),
    )

    assert not managed.first_action_received.is_set()
    assert not input_source.activation_signal.is_set()
    assert tuple(input_source._events) == ()
    assert channel.messages == []

    await manager._handle_datachannel_message(
        managed_session=managed,
        raw_message=json.dumps(
            {
                "type": "action",
                "action": {"event": "keydown", "key": "space"},
            }
        ),
    )

    assert managed.first_action_received.is_set()
    assert input_source.activation_signal.is_set()
    assert [event.payload for event in input_source._events] == [{"key": "space"}]


@pytest.mark.asyncio
async def test_legacy_key_filter_runs_before_activation() -> None:
    runtime = SimpleNamespace()
    manager = _make_manager(_WOnlyTestManager, runtime)
    managed, _track, _peer, channel = _managed_session(runtime)
    managed.first_action_received.clear()
    resampler = _RecordingLegacySegmentResampler()
    managed.legacy_segment_resampler = resampler

    await manager._handle_datachannel_message(
        managed_session=managed,
        raw_message=json.dumps(
            {"type": "action", "action": {"event": "keydown", "key": "q"}}
        ),
    )

    assert not managed.first_action_received.is_set()
    assert resampler.edges == []
    assert len(managed.pending_action_arrivals) == 0
    assert channel.messages == []
