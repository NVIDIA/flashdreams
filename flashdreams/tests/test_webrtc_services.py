# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import json
import threading
from collections.abc import Mapping, Sequence
from typing import Any

import pytest

from flashdreams.runtime import (
    CanonicalInputSchema,
    IdentityInputMapping,
    InferenceConfig,
    InferenceInput,
    InferenceInputSchema,
    InferenceRuntime,
    InferenceSession,
    InputMapping,
    KeyboardState,
    StepRequirements,
    StepResult,
    UserInputs,
)
from flashdreams.runtime.demo import (
    DemoAdapter,
    DemoSpec,
    InMemorySessionMetricsRecorder,
    ModelInputProvider,
    ModelWarmupPlan,
    OutputDecision,
    PreparedScenario,
    ProviderCapabilities,
    RealtimeSessionDriver,
    RunContext,
    RunResult,
    RuntimeHost,
    SessionEdges,
    SessionInfo,
    StepPipeline,
    UserInputWindow,
    WebRTCOutputSpec,
    run_demo_session_async,
)
from flashdreams.runtime.demo.timing import ResamplerRealtimeClock
from flashdreams.serving.webrtc.server import SessionBusyError
from flashdreams.serving.webrtc.services import (
    WEBRTC_SKIPPED_INPUTS_METADATA_KEY,
    AsyncioBlockingPreparationService,
    ThreadSafeWebRTCOutputBridge,
    WebRTCActivationPolicy,
    WebRTCInputOverflowError,
    WebRTCInputSource,
    WebRTCOfferRequest,
    WebRTCOutputSink,
    WebRTCRunMode,
    WebRTCSessionOfferHandler,
    WebRTCTransportService,
)

pytestmark = pytest.mark.ci_cpu


@pytest.mark.asyncio
async def test_webrtc_offer_handler_calls_shared_session_helper() -> None:
    spec = _webrtc_spec()
    adapter = _FakeAdapter()
    mode = WebRTCRunMode(edge_factory=_FinishedEdgeFactory())
    context = mode.create_run_context(
        spec=spec,
        adapter=adapter,
        host=RuntimeHost(_UnusedRuntime()),
        model_warmup_plan=ModelWarmupPlan(),
    )
    answerer = _RecordingAnswerer()
    helper_calls: list[DemoSpec] = []
    handler = WebRTCSessionOfferHandler(
        context=context,
        spec=spec,
        adapter=adapter,
        run_mode=mode,
        answerer=answerer,
        session_helper=lambda **kwargs: _record_completed_helper(
            helper_calls,
            **kwargs,
        ),
    )

    answer = await handler.handle_offer(offer_sdp="v=0\r\n", offer_type="offer")

    assert answer == {"sdp": "answer-sdp", "type": "answer"}
    assert helper_calls == [spec]
    assert answerer.offers == [WebRTCOfferRequest(sdp="v=0\r\n", type="offer")]


@pytest.mark.asyncio
async def test_webrtc_busy_rejects_before_prepare_provider_or_answer() -> None:
    spec = _webrtc_spec()
    adapter = _FakeAdapter()
    mode = WebRTCRunMode(edge_factory=_FinishedEdgeFactory())
    context = RunContext(
        host=RuntimeHost(_UnusedRuntime()),
        run_metrics=InMemorySessionMetricsRecorder(),
        admission=_BusyAdmission(),
    )
    answerer = _RecordingAnswerer()
    handler = WebRTCSessionOfferHandler(
        context=context,
        spec=spec,
        adapter=adapter,
        run_mode=mode,
        answerer=answerer,
    )

    with pytest.raises(SessionBusyError):
        await handler.handle_offer(offer_sdp="v=0\r\n", offer_type="offer")

    assert adapter.prepare_thread_id is None
    assert adapter.providers == []
    assert answerer.offers == []


@pytest.mark.asyncio
async def test_webrtc_scenario_prepare_runs_off_event_loop_thread() -> None:
    loop_thread_id = threading.get_ident()
    spec = _webrtc_spec()
    adapter = _FakeAdapter()

    result = await AsyncioBlockingPreparationService().run(
        adapter.prepare_scenario,
        spec,
    )

    assert isinstance(result, PreparedScenario)
    assert adapter.prepare_thread_id is not None
    assert adapter.prepare_thread_id != loop_thread_id


@pytest.mark.asyncio
async def test_webrtc_input_source_emits_typed_user_inputs() -> None:
    resampler = _FakeResampler(dt=0.1, start_v=0.0)
    source = WebRTCInputSource(resampler=resampler)
    source.handle_browser_message(
        json.dumps({"type": "action", "action": {"event": "keydown", "key": "w"}}),
        timestamp_s=0.05,
    )
    source.handle_browser_message(
        json.dumps({"type": "event", "event_id": "prompt-1", "state": "trigger"}),
        timestamp_s=0.06,
    )
    clock = ResamplerRealtimeClock(
        resampler=resampler,
        now_fn=lambda: 0.2,
        sleep_fn=_record_sleep,
    )

    result = await source.next_realtime_window(
        request=StepRequirements(step_index=0, input_frame_count=2),
        clock=clock,
    )

    assert source.activation_signal.is_set()
    assert result.window.start_s == pytest.approx(0.0)
    assert result.window.end_s == pytest.approx(0.2)
    assert result.window.frame_times == pytest.approx((0.1, 0.2))
    assert result.window.metadata == {}
    assert [event.event_type for event in result.window.inputs.events] == [
        "key_down",
        "text_event",
    ]
    assert result.window.inputs.events[0].payload == {"key": "w"}
    assert result.window.inputs.events[1].payload == {
        "event_id": "prompt-1",
        "state": "trigger",
    }


@pytest.mark.parametrize("event", ["keydown", "keyup"])
@pytest.mark.parametrize("key", [0, 1, False, True, None, [], {}, "", " "])
def test_webrtc_input_source_rejects_non_string_or_empty_keys(
    event: str, key: object
) -> None:
    source = WebRTCInputSource(resampler=_FakeResampler(dt=0.1, start_v=0.0))

    result = source.handle_browser_message(
        json.dumps({"type": "action", "action": {"event": event, "key": key}}),
        timestamp_s=0.05,
    )

    assert result.kind == "error"
    assert not source.activation_signal.is_set()
    assert source.activation_timestamp_s is None
    assert not source._events


@pytest.mark.parametrize(
    "event_payload",
    [
        {"event_id": []},
        {"event_id": {}},
        {"event_id": None},
        {"id": 0},
        {"event_id": "storm", "state": None},
        {"event_id": "storm", "state": []},
        {"event_id": "storm", "state": True},
        {"event_id": [], "state": "clear"},
    ],
)
def test_webrtc_input_source_rejects_malformed_events_before_activation(
    event_payload: dict[str, object],
) -> None:
    source = WebRTCInputSource(resampler=_FakeResampler(dt=0.1, start_v=0.0))

    result = source.handle_browser_message(
        json.dumps({"type": "event", **event_payload}), timestamp_s=0.05
    )

    assert result.kind == "error"
    assert result.error
    assert not source.activation_signal.is_set()
    assert source.activation_timestamp_s is None
    assert not source._events
    assert not source._event_sizes
    assert source._pending_event_bytes == 0


@pytest.mark.parametrize(
    ("event_payload", "event_id", "state"),
    [
        ({"event_id": " storm "}, "storm", "trigger"),
        ({"id": "storm", "state": " TRIGGER "}, "storm", "trigger"),
        ({"event_id": "storm", "state": " "}, "storm", "trigger"),
        ({"event_id": "storm", "state": "custom"}, "storm", "custom"),
        ({"state": "clear"}, None, "clear"),
        ({"event_id": None, "state": "release"}, None, "release"),
        ({"id": None, "state": "off"}, None, "off"),
        ({"event_id": "", "state": "none"}, None, "none"),
    ],
)
def test_webrtc_input_source_preserves_event_defaults_alias_and_clears(
    event_payload: dict[str, object], event_id: str | None, state: str
) -> None:
    source = WebRTCInputSource(resampler=_FakeResampler(dt=0.1, start_v=0.0))

    result = source.handle_browser_payload(
        {"type": "event", **event_payload}, timestamp_s=0.05
    )

    assert result.kind == "event" and result.activated
    assert source.activation_signal.is_set()
    assert len(source._events) == 1
    assert source._events[0].payload == {"event_id": event_id, "state": state}
    assert source._events[0].source_event_id == event_id


@pytest.mark.asyncio
async def test_webrtc_activation_anchors_to_first_input_event() -> None:
    resampler = _FakeResampler(dt=0.1, start_v=0.0)
    source = WebRTCInputSource(resampler=resampler)
    transport = WebRTCTransportService()
    clock = ResamplerRealtimeClock(
        resampler=resampler,
        now_fn=lambda: 0.2,
        sleep_fn=_record_sleep,
    )
    source.handle_browser_message(
        json.dumps({"type": "action", "action": {"event": "keydown", "key": "w"}}),
        timestamp_s=0.05,
    )

    result = await WebRTCActivationPolicy(
        input_source=source,
        transport=transport,
    ).wait_until_active(clock)

    assert result.activated
    assert source.activation_timestamp_s == pytest.approx(0.05)
    assert resampler.next_chunk_start_v == pytest.approx(0.05)


@pytest.mark.asyncio
async def test_webrtc_queue_reserves_releases_and_clears_then_recovers() -> None:
    resampler = _FakeResampler(dt=0.1, start_v=0.0)
    source = WebRTCInputSource(
        resampler=resampler,
        max_pending_events=4,
        reserved_recovery_events=2,
    )
    keyboard = KeyboardState()
    keyboard.apply_event(event="keydown", key="w")
    for key in ("a", "d"):
        result = source.handle_browser_payload(
            {"type": "action", "event": "keydown", "key": key}, timestamp_s=0.01
        )
        assert result.kind == "action"
    rejected = source.handle_browser_payload(
        {"type": "action", "event": "keydown", "key": "w"}, timestamp_s=0.02
    )
    assert rejected.kind == "error"
    assert len(source._events) == 2
    assert (
        source.handle_browser_payload(
            {"type": "action", "event": "keyup", "key": "w"}, timestamp_s=0.03
        ).kind
        == "action"
    )
    assert (
        source.handle_browser_payload(
            {"type": "event", "state": "clear"}, timestamp_s=0.04
        ).kind
        == "event"
    )

    window = await source.next_realtime_window(
        request=StepRequirements(step_index=0, input_frame_count=1),
        clock=ResamplerRealtimeClock(resampler=resampler, now_fn=lambda: 0.1),
    )
    for event in window.window.inputs.events:
        if event.event_type in {"key_down", "key_up"}:
            keyboard.apply_event(
                event="keydown" if event.event_type == "key_down" else "keyup",
                key=event.payload["key"],
            )
    assert "w" not in keyboard.snapshot()
    assert window.window.inputs.events[-1].payload == {
        "event_id": None,
        "state": "clear",
    }
    assert not source._events
    assert not source._event_sizes
    assert source._pending_event_bytes == 0
    assert (
        source.handle_browser_payload(
            {"type": "action", "event": "keydown", "key": "w"}, timestamp_s=0.11
        ).kind
        == "action"
    )


def test_webrtc_queue_byte_budget_counts_utf8_and_reserves_recovery() -> None:
    source = WebRTCInputSource(
        resampler=_FakeResampler(dt=0.1, start_v=0.0),
        max_pending_bytes=256,
        reserved_recovery_bytes=64,
    )
    event_id = "é" * 32
    assert (
        source.handle_browser_payload(
            {"type": "event", "event_id": event_id}, timestamp_s=0.01
        ).kind
        == "event"
    )
    expected_bytes = len(
        json.dumps(
            ["text_event", {"event_id": event_id, "state": "trigger"}, event_id],
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    assert source._pending_event_bytes == expected_bytes
    assert (
        source.handle_browser_payload(
            {"type": "event", "event_id": "another"}, timestamp_s=0.02
        ).kind
        == "error"
    )
    assert source._pending_event_bytes == expected_bytes
    assert (
        source.handle_browser_payload(
            {"type": "action", "event": "keyup", "key": "w"}, timestamp_s=0.03
        ).kind
        == "action"
    )
    assert source._pending_event_bytes <= source.max_pending_bytes
    for _ in range(10):
        try:
            source.handle_browser_payload(
                {"type": "event", "state": "clear"}, timestamp_s=0.04
            )
        except WebRTCInputOverflowError:
            break
    else:
        pytest.fail("Recovery input must not exceed the byte budget")
    assert source._pending_event_bytes <= source.max_pending_bytes
    assert len(source._events) < source.max_pending_events
    source.reset(start_v=0.1)
    assert source._pending_event_bytes == 0
    assert not source._events
    assert not source._event_sizes
    assert not source.activation_signal.is_set()


def test_webrtc_stalled_model_cannot_accumulate_unbounded_events() -> None:
    source = WebRTCInputSource(resampler=_FakeResampler(dt=0.1, start_v=0.0))
    accepted = 0
    for _ in range(10000):
        result = source.handle_browser_payload(
            {"type": "action", "event": "keydown", "key": "w"}, timestamp_s=0.01
        )
        accepted += result.kind == "action"
    assert accepted == source.max_pending_events - source.reserved_recovery_events
    assert len(source._events) == accepted
    assert len(source._event_sizes) == accepted
    assert source._pending_event_bytes <= source.max_pending_bytes


@pytest.mark.asyncio
async def test_webrtc_expired_input_is_not_replayed_on_model_resume() -> None:
    resampler = _FakeResampler(dt=0.1, start_v=0.0)
    source = WebRTCInputSource(resampler=resampler, max_pending_age_s=1.0)
    source.record_user_event(
        timestamp_s=0.01, event_type="key_down", payload={"key": "w"}
    )
    with pytest.raises(WebRTCInputOverflowError, match="backlog expired"):
        await source.next_realtime_window(
            request=StepRequirements(step_index=0, input_frame_count=1),
            clock=ResamplerRealtimeClock(resampler=resampler, now_fn=lambda: 1.1),
        )
    assert resampler.next_chunk_start_v == 0.0


def test_webrtc_pending_age_checks_out_of_order_input() -> None:
    source = WebRTCInputSource(
        resampler=_FakeResampler(dt=0.1, start_v=0.0), max_pending_age_s=1.0
    )
    for timestamp_s in (1.0, 0.4):
        source.record_user_event(
            timestamp_s=timestamp_s, event_type="key_down", payload={"key": "w"}
        )
    with pytest.raises(WebRTCInputOverflowError, match="backlog expired"):
        source.check_pending_events(now_s=1.5)


@pytest.mark.asyncio
async def test_webrtc_input_at_window_boundary_remains_pending() -> None:
    resampler = _FakeResampler(dt=0.1, start_v=0.0)
    source = WebRTCInputSource(resampler=resampler)
    source.record_user_event(timestamp_s=0.1, event_type="key_up", payload={"key": "w"})
    result = await source.next_realtime_window(
        request=StepRequirements(step_index=0, input_frame_count=1),
        clock=ResamplerRealtimeClock(resampler=resampler, now_fn=lambda: 0.1),
    )
    assert not result.window.inputs.events
    assert len(source._events) == 1
    assert source._pending_event_bytes > 0


@pytest.mark.asyncio
async def test_webrtc_catch_up_preserves_skipped_press_and_current_release() -> None:
    resampler = _FakeResampler(dt=0.1, start_v=0.0)
    source = WebRTCInputSource(resampler=resampler)
    source.record_user_event(
        timestamp_s=0.05, event_type="key_down", payload={"key": "w"}
    )
    source.record_user_event(
        timestamp_s=0.45, event_type="key_up", payload={"key": "w"}
    )

    result = await source.next_realtime_window(
        request=StepRequirements(step_index=0, input_frame_count=1),
        clock=ResamplerRealtimeClock(resampler=resampler, now_fn=lambda: 0.5),
    )

    skipped = result.window.metadata[WEBRTC_SKIPPED_INPUTS_METADATA_KEY]
    assert isinstance(skipped, UserInputs)
    assert [event.event_type for event in skipped.events] == ["key_down"]
    assert [event.event_type for event in result.window.inputs.events] == ["key_up"]
    assert not source._events
    assert source._pending_event_bytes == 0


@pytest.mark.parametrize(
    "limits",
    [
        {"max_pending_events": 32},
        {"reserved_recovery_events": 0},
        {"reserved_recovery_events": -1},
        {"max_pending_events": True},
        {"max_pending_bytes": 65536},
        {"reserved_recovery_bytes": 0},
        {"max_pending_age_s": 0},
        {"max_pending_age_s": float("nan")},
        {"max_pending_age_s": float("inf")},
    ],
)
def test_webrtc_input_budgets_must_leave_bounded_recovery_capacity(
    limits: dict[str, Any],
) -> None:
    with pytest.raises(ValueError):
        WebRTCInputSource(resampler=_FakeResampler(dt=0.1, start_v=0.0), **limits)


@pytest.mark.asyncio
async def test_webrtc_activation_can_start_without_input() -> None:
    resampler = _FakeResampler(dt=0.1, start_v=0.0)
    source = WebRTCInputSource(resampler=resampler)
    transport = WebRTCTransportService()
    clock = ResamplerRealtimeClock(
        resampler=resampler,
        now_fn=lambda: 0.2,
        sleep_fn=_record_sleep,
    )

    result = await WebRTCActivationPolicy(
        input_source=source,
        transport=transport,
        activate_without_input=True,
    ).wait_until_active(clock)

    assert result.activated
    assert source.activation_timestamp_s is None
    assert resampler.next_chunk_start_v == pytest.approx(0.2)


@pytest.mark.asyncio
async def test_webrtc_activation_without_input_still_honors_disconnect() -> None:
    resampler = _FakeResampler(dt=0.1, start_v=0.0)
    source = WebRTCInputSource(resampler=resampler)
    transport = WebRTCTransportService()
    transport.disconnect("browser disconnected")
    clock = ResamplerRealtimeClock(
        resampler=resampler,
        now_fn=lambda: 0.2,
        sleep_fn=_record_sleep,
    )

    result = await WebRTCActivationPolicy(
        input_source=source,
        transport=transport,
        activate_without_input=True,
    ).wait_until_active(clock)

    assert not result.activated
    assert result.reason == "browser disconnected"
    assert resampler.next_chunk_start_v == pytest.approx(0.0)


@pytest.mark.asyncio
async def test_webrtc_output_sink_returns_presentation_backpressure() -> None:
    loop = asyncio.get_running_loop()
    encoder = _BlockingEncoder()
    track = _FakeVideoTrack(queue_depth=3)
    deliveries: list[object] = []
    bridge = ThreadSafeWebRTCOutputBridge(
        loop=loop,
        video_encoder=encoder,
        video_track=track,
        on_delivery=deliveries.append,
    )
    sink = WebRTCOutputSink(bridge=bridge)
    sink.open(SessionInfo())
    step_result = StepResult(step_index=0, frame_count=1)

    decision = sink.write(step_result)

    assert isinstance(decision, OutputDecision)
    assert not decision.dropped
    assert decision.backpressure_s == pytest.approx(0.1)
    assert encoder.prepared_payloads == [step_result.step_index]
    await asyncio.wait_for(encoder.started.wait(), timeout=1.0)
    assert not encoder.release.is_set()
    assert bridge.pending_count == 1

    encoder.release.set()
    await asyncio.wait_for(encoder.done.wait(), timeout=1.0)
    await asyncio.sleep(0)

    assert deliveries == ["delivered"]
    assert encoder.delivered_payloads == [{"step_index": step_result.step_index}]
    assert bridge.pending_count == 0
    sink.close()


@pytest.mark.asyncio
async def test_webrtc_output_bridge_prepares_payload_before_async_delivery() -> None:
    loop = asyncio.get_running_loop()
    encoder = _BlockingEncoder()
    track = _FakeVideoTrack()
    bridge = ThreadSafeWebRTCOutputBridge(
        loop=loop,
        video_encoder=encoder,
        video_track=track,
    )
    sink = WebRTCOutputSink(bridge=bridge)
    sink.open(SessionInfo())

    decision = sink.write(StepResult(step_index=7, frame_count=1))

    assert not decision.dropped
    assert encoder.prepared_payloads == [7]
    assert encoder.delivered_payloads == []

    encoder.release.set()
    await asyncio.wait_for(encoder.done.wait(), timeout=1.0)
    await asyncio.sleep(0)

    assert encoder.delivered_payloads == [{"step_index": 7}]
    sink.close()


@pytest.mark.asyncio
async def test_webrtc_output_bridge_replaces_full_pending_queue() -> None:
    loop = asyncio.get_running_loop()
    encoder = _BlockingEncoder()
    track = _FakeVideoTrack()
    bridge = ThreadSafeWebRTCOutputBridge(
        loop=loop,
        video_encoder=encoder,
        video_track=track,
        max_pending_chunks=1,
    )
    sink = WebRTCOutputSink(bridge=bridge)
    sink.open(SessionInfo())

    first = sink.write(StepResult(step_index=0, frame_count=1))
    await asyncio.wait_for(encoder.started.wait(), timeout=1.0)
    second = sink.write(StepResult(step_index=1, frame_count=1))

    assert not first.dropped
    assert not second.dropped
    assert encoder.prepared_payloads == [0, 1]
    for _ in range(10):
        if track.flush_count:
            break
        await asyncio.sleep(0)
    assert track.flush_count == 1

    encoder.release.set()
    await asyncio.wait_for(encoder.done.wait(), timeout=1.0)
    sink.close()


@pytest.mark.asyncio
async def test_webrtc_output_bridge_flushes_full_track_queue_before_delivery() -> None:
    loop = asyncio.get_running_loop()
    encoder = _BlockingEncoder()
    track = _FakeVideoTrack(queue_depth=2)
    bridge = ThreadSafeWebRTCOutputBridge(
        loop=loop,
        video_encoder=encoder,
        video_track=track,
    )
    sink = WebRTCOutputSink(bridge=bridge)
    sink.open(SessionInfo())

    decision = sink.write(StepResult(step_index=0, frame_count=2))

    assert not decision.dropped
    await asyncio.wait_for(encoder.started.wait(), timeout=1.0)
    assert track.flush_count == 1

    encoder.release.set()
    await asyncio.wait_for(encoder.done.wait(), timeout=1.0)
    sink.close()


@pytest.mark.asyncio
async def test_webrtc_output_bridge_generation_reset_cancels_stale_delivery() -> None:
    loop = asyncio.get_running_loop()
    encoder = _BlockingEncoder()
    track = _FakeVideoTrack()
    deliveries: list[object] = []
    chunk_deliveries: list[int] = []
    bridge = ThreadSafeWebRTCOutputBridge(
        loop=loop,
        video_encoder=encoder,
        video_track=track,
        on_delivery=deliveries.append,
        on_chunk_delivery=lambda chunk: chunk_deliveries.append(chunk.step_index),
    )
    sink = WebRTCOutputSink(bridge=bridge)
    sink.open(SessionInfo())

    first = sink.write(StepResult(step_index=0, frame_count=1))
    await asyncio.wait_for(encoder.started.wait(), timeout=1.0)
    sink.begin_generation(1)
    for _ in range(10):
        if track.flush_count:
            break
        await asyncio.sleep(0)
    second = sink.write(StepResult(step_index=1, frame_count=1))

    assert not first.dropped
    assert not second.dropped
    assert track.flush_count == 1

    encoder.release.set()
    await asyncio.wait_for(encoder.done.wait(), timeout=1.0)
    await asyncio.sleep(0)

    assert deliveries == ["delivered"]
    assert chunk_deliveries == [1]
    sink.close()


@pytest.mark.asyncio
async def test_disconnect_closes_transport_and_releases_reservation_once() -> None:
    spec = _webrtc_spec()
    adapter = _FakeAdapter()
    transport_closed: list[str | None] = []
    transport = WebRTCTransportService(on_close=transport_closed.append)
    mode = WebRTCRunMode(
        edge_factory=_DisconnectedEdgeFactory(transport=transport),
        driver=RealtimeSessionDriver(cleanup_timeout_s=1.0),
    )
    context = mode.create_run_context(
        spec=spec,
        adapter=adapter,
        host=RuntimeHost(_SessionRuntime()),
        model_warmup_plan=ModelWarmupPlan(),
    )
    reservation = context.admission.try_reserve()
    assert reservation is not None
    transport.disconnect("browser disconnect")

    result = await _record_async_helper(
        [],
        context=context,
        spec=spec,
        scenario=adapter.prepare_scenario(spec),
        adapter=adapter,
        run_mode=mode,
        pipeline=StepPipeline(),
        reservation=reservation,
    )
    transport.close("cleanup close")

    assert result.status == "not_activated"
    assert result.reason == "browser disconnect"
    assert transport.close_count == 1
    assert transport_closed == ["browser disconnect"]
    assert reservation.release_count == 1  # ty:ignore[unresolved-attribute]
    assert adapter.providers[0].close_count == 1


def test_webrtc_run_mode_objects_are_control_rank_only() -> None:
    spec = _webrtc_spec()
    adapter = _FakeAdapter()
    edge_factory = _FinishedEdgeFactory()
    mode = WebRTCRunMode(edge_factory=edge_factory)
    worker_context = mode.create_run_context(
        spec=spec,
        adapter=adapter,
        host=RuntimeHost(_UnusedRuntime(), is_control_rank=False),
        model_warmup_plan=ModelWarmupPlan(),
    )

    assert worker_context.services == {}
    assert worker_context.admission.try_reserve() is None
    with pytest.raises(RuntimeError, match="control-rank only"):
        mode.create_session_edges(
            context=worker_context,
            spec=spec,
            scenario=adapter.prepare_scenario(spec),
            provider=_FakeProvider(),
            adapter=adapter,
        )

    control_context = mode.create_run_context(
        spec=spec,
        adapter=adapter,
        host=RuntimeHost(_UnusedRuntime(), is_control_rank=True),
        model_warmup_plan=ModelWarmupPlan(),
    )
    assert set(control_context.services) == {"blocking_preparation"}
    assert control_context.admission.try_reserve() is not None


def _webrtc_spec() -> DemoSpec:
    return DemoSpec(
        model_id="fake-demo",
        input_mode="keyboard-driving",
        output=WebRTCOutputSpec(port=8081),
    )


async def _record_sleep(delay_s: float) -> None:
    del delay_s


async def _record_async_helper(
    calls: list[DemoSpec],
    **kwargs: Any,
) -> RunResult:
    calls.append(kwargs["spec"])
    return await run_demo_session_async(**kwargs)


async def _record_completed_helper(
    calls: list[DemoSpec],
    **kwargs: Any,
) -> RunResult:
    calls.append(kwargs["spec"])
    return RunResult(status="completed")


class _FakeAdapter:
    model_id = "fake-demo"
    inference_input_schema = InferenceInputSchema()
    canonical_input_schema = CanonicalInputSchema()

    def __init__(self) -> None:
        self.prepare_thread_id: int | None = None
        self.providers: list[_FakeProvider] = []

    def supported_input_modes(self) -> tuple[str, ...]:
        return ("keyboard-driving",)

    def supported_output_modes(self) -> tuple[str, ...]:
        return ("webrtc",)

    def default_input_mapping(self) -> InputMapping:
        return IdentityInputMapping()

    def validate_config(self, config: InferenceConfig) -> None:
        if config.model_id != self.model_id:
            raise ValueError(f"Unsupported model_id={config.model_id!r}.")

    def create_runtime(self, config: InferenceConfig) -> InferenceRuntime:
        self.validate_config(config)
        return _SessionRuntime()

    def prepare_scenario(self, spec: DemoSpec) -> PreparedScenario:
        self.prepare_thread_id = threading.get_ident()
        assert spec.model_id == self.model_id
        return PreparedScenario(initial_inputs=InferenceInput())

    def create_model_input_provider(
        self,
        spec: DemoSpec,
        scenario: PreparedScenario,
    ) -> "_FakeProvider":
        del spec, scenario
        provider = _FakeProvider()
        self.providers.append(provider)
        return provider


class _FakeProvider:
    capabilities = ProviderCapabilities(
        supports_realtime_clock=True,
        supports_reset=True,
        deterministic_given_inputs=False,
    )

    def __init__(self) -> None:
        self.close_count = 0

    def prepare_initial_input(self) -> InferenceInput:
        return InferenceInput()

    def prepare_step(
        self,
        *,
        request: StepRequirements,
        user_window: UserInputWindow,
    ) -> Any:
        del request, user_window
        raise AssertionError("disconnected tests must stop before step prep")

    def reset(self, inputs: InferenceInput | None = None) -> None:
        del inputs

    def close(self) -> None:
        self.close_count += 1


class _SessionRuntime:
    def start_session(self, inputs: InferenceInput) -> InferenceSession:
        del inputs
        return _NeverSteppedSession()

    def close(self) -> None:
        return


class _UnusedRuntime:
    def start_session(self, inputs: InferenceInput) -> InferenceSession:
        del inputs
        raise AssertionError("runtime should not be used")

    def close(self) -> None:
        return


class _NeverSteppedSession:
    def next_step_requirements(self) -> StepRequirements | None:
        return StepRequirements(step_index=0, input_frame_count=1)

    def next_step_request(self) -> None:
        return None

    def step(self, inputs: InferenceInput) -> StepResult:
        del inputs
        raise AssertionError("disconnected tests must stop before stepping")

    def reset(self, inputs: InferenceInput | None = None) -> None:
        del inputs

    def close(self) -> None:
        return

    def session_info(self) -> SessionInfo:
        return SessionInfo()


class _FinishedEdgeFactory:
    def __init__(self) -> None:
        self.edges: list[SessionEdges] = []

    def create_session_edges(
        self,
        *,
        context: RunContext,
        spec: DemoSpec,
        scenario: PreparedScenario,
        provider: ModelInputProvider,
        adapter: DemoAdapter,
    ) -> SessionEdges:
        del spec, scenario, provider, adapter
        edges = SessionEdges(
            input_source=_FinishedRealtimeInputSource(),
            output_sink=_RecordingOutputSink(),
            cleanup_tasks=context.cleanup_tasks,
            metrics=InMemorySessionMetricsRecorder(),
            transport=WebRTCTransportService(),
            clock=_InstantClock(),
            activation=_AlreadyActive(),
        )
        self.edges.append(edges)
        return edges


class _DisconnectedEdgeFactory:
    def __init__(self, *, transport: WebRTCTransportService) -> None:
        self.transport = transport

    def create_session_edges(
        self,
        *,
        context: RunContext,
        spec: DemoSpec,
        scenario: PreparedScenario,
        provider: ModelInputProvider,
        adapter: DemoAdapter,
    ) -> SessionEdges:
        del spec, scenario, provider, adapter
        resampler = _FakeResampler(dt=0.1, start_v=0.0)
        source = WebRTCInputSource(resampler=resampler)
        return SessionEdges(
            input_source=source,
            output_sink=_RecordingOutputSink(),
            cleanup_tasks=context.cleanup_tasks,
            metrics=InMemorySessionMetricsRecorder(),
            transport=self.transport,
            clock=ResamplerRealtimeClock(
                resampler=resampler,
                now_fn=lambda: 0.0,
                sleep_fn=_record_sleep,
            ),
            activation=WebRTCActivationPolicy(
                input_source=source,
                transport=self.transport,
            ),
        )


class _FinishedRealtimeInputSource:
    is_finite = False
    is_deterministic = False
    user_input_schema = _FakeProvider.capabilities.user_input_schema

    def is_finished(self) -> bool:
        return True

    async def next_realtime_window(
        self,
        *,
        request: StepRequirements,
        clock: Any,
    ) -> Any:
        del request, clock
        raise AssertionError("finished async driver should not request windows")


class _RecordingOutputSink:
    produces_artifacts = False

    def __init__(self) -> None:
        self.close_count = 0

    def open(self, session_info: SessionInfo) -> None:
        del session_info

    def begin_generation(self, generation: int) -> None:
        del generation

    def write(self, result: StepResult) -> OutputDecision:
        del result
        return OutputDecision()

    def close(self) -> Sequence[Any]:
        self.close_count += 1
        return ()


class _AlreadyActive:
    timeout_s = None

    async def wait_until_active(self, clock: Any) -> Any:
        del clock
        return type("Activation", (), {"activated": True, "reason": None})()


class _InstantClock:
    is_realtime = True
    is_deterministic = False

    def now(self) -> float:
        return 0.0

    def anchor(self, wall_time_s: float) -> None:
        del wall_time_s

    async def wait_until_window_end(self, end_s: float) -> None:
        del end_s

    async def apply_backpressure(self, requested_s: float) -> None:
        del requested_s

    def catch_up(
        self,
        *,
        request: StepRequirements,
        max_lag_s: float,
        policy: str,
    ) -> Any:
        del request, max_lag_s, policy
        return type("CatchUp", (), {"skipped_s": 0.0})()


class _BusyAdmission:
    def try_reserve(self) -> None:
        return None


class _RecordingAnswerer:
    def __init__(self) -> None:
        self.offers: list[WebRTCOfferRequest] = []

    async def create_answer(
        self,
        *,
        offer: WebRTCOfferRequest,
        session_task: asyncio.Task[RunResult],
    ) -> Mapping[str, str]:
        self.offers.append(offer)
        result = await session_task
        assert result.status == "completed"
        return {"sdp": "answer-sdp", "type": "answer"}


class _FakeResampler:
    def __init__(self, *, dt: float, start_v: float) -> None:
        self.dt = dt
        self.next_chunk_start_v = start_v

    def reset(self, *, start_v: float) -> None:
        self.next_chunk_start_v = start_v

    def sample_chunk(
        self,
        num_frames: int,
    ) -> tuple[float, ...]:
        start = self.next_chunk_start_v
        frame_times = tuple(
            start + (index + 1) * self.dt for index in range(num_frames)
        )
        end = start + num_frames * self.dt
        self.next_chunk_start_v = end
        return frame_times


class _BlockingEncoder:
    fps = 30

    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.done = asyncio.Event()
        self.prepared_payloads: list[int] = []
        self.delivered_payloads: list[object] = []

    def prepare_chunk_payload(
        self,
        result: StepResult,
        track: Any,
    ) -> object:
        del track
        self.prepared_payloads.append(result.step_index)
        return {"step_index": result.step_index}

    async def deliver_prepared_chunk(
        self,
        payload: object,
        track: Any,
        *,
        force_keyframe: bool = False,
    ) -> str:
        del track, force_keyframe
        self.delivered_payloads.append(payload)
        self.started.set()
        await self.release.wait()
        self.done.set()
        return "delivered"

    async def deliver_chunk(
        self,
        result: StepResult,
        track: Any,
        *,
        force_keyframe: bool = False,
    ) -> str:
        return await self.deliver_prepared_chunk(
            self.prepare_chunk_payload(result, track),
            track,
            force_keyframe=force_keyframe,
        )


class _FakeVideoTrack:
    fps = 30

    def __init__(self, *, queue_depth: int = 0) -> None:
        self.queue_depth = queue_depth
        self.flush_count = 0

    def qsize(self) -> int:
        return self.queue_depth

    async def flush(self) -> None:
        self.flush_count += 1
        self.queue_depth = 0
