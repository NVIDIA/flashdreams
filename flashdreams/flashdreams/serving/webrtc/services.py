# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Session-edge services for the shared WebRTC demo run mode.

These classes are the Phase 12 decomposition layer: they translate WebRTC
transport facts into the shared demo runtime contracts without owning model
execution. The production manager still uses its legacy execution hook until
the realtime-driver adoption phase.
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import json
import math
import threading
from collections import deque
from collections.abc import Callable, Coroutine, Mapping, MutableSet, Sequence
from concurrent.futures import Future
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

from flashdreams.runtime import (
    StepRequirements,
    StepResult,
    UserInputCapability,
    UserInputEvent,
    UserInputs,
    UserInputSchema,
)
from flashdreams.runtime._utils import freeze_mapping
from flashdreams.runtime.demo import (
    AsyncSessionDriver,
    DemoAdapter,
    DemoSpec,
    InMemorySessionMetricsRecorder,
    ModelInputProvider,
    ModelWarmupPlan,
    OutputDecision,
    PreparedScenario,
    RealtimeSessionDriver,
    RealtimeWindowResult,
    RunContext,
    RunModeCapabilities,
    RunResult,
    RuntimeHost,
    SessionEdges,
    SessionInfo,
    SingleSessionAdmissionPolicy,
    StepPipeline,
    UserInputWindow,
    WebRTCErrorPolicy,
    input_frame_count_from_request,
    run_demo_session_async,
)
from flashdreams.runtime.demo.timing import (
    ActivationResult,
    CatchUpPolicy,
    DeterministicClock,
    RealtimeClock,
)

from .messages import (
    MESSAGE_TYPE_ACTION,
    MESSAGE_TYPE_DISCONNECT,
    MESSAGE_TYPE_EVENT,
    MESSAGE_TYPE_HEARTBEAT,
)
from .server import SessionBusyError

WebRTCMessageKind = Literal[
    "action",
    "disconnect",
    "event",
    "heartbeat",
    "error",
]

WebRTCDropPolicy = Literal["none", "drop_newest", "drop_oldest"]

_CLEAR_EVENT_STATES = frozenset({"clear", "release", "off", "none"})
WEBRTC_SKIPPED_INPUTS_METADATA_KEY = "webrtc_skipped_inputs"
WEBRTC_SKIPPED_WINDOW_METADATA_KEY = "webrtc_skipped_window"


@dataclass(frozen=True, kw_only=True, slots=True)
class WebRTCMessageResult:
    """Result of translating one browser data-channel message."""

    kind: WebRTCMessageKind
    activated: bool = False
    error: str | None = None


@dataclass(frozen=True, kw_only=True, slots=True)
class WebRTCOfferRequest:
    """Browser SDP offer passed to the shared offer/session handler."""

    sdp: str
    type: str

    def __post_init__(self) -> None:
        if not self.sdp.strip():
            raise ValueError("WebRTCOfferRequest.sdp must be non-empty.")
        if not self.type.strip():
            raise ValueError("WebRTCOfferRequest.type must be non-empty.")


@dataclass(frozen=True, kw_only=True, slots=True)
class WebRTCOutputBridgeDecision:
    """Immediate delivery decision from a nonblocking WebRTC output bridge."""

    accepted: bool = True
    should_stop: bool = False
    dropped: bool = False
    drop_policy: WebRTCDropPolicy = "none"
    backpressure_s: float = 0.0
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.drop_policy not in {"none", "drop_newest", "drop_oldest"}:
            raise ValueError(f"Unsupported drop_policy={self.drop_policy!r}.")
        if not math.isfinite(self.backpressure_s) or self.backpressure_s < 0.0:
            raise ValueError(
                "WebRTCOutputBridgeDecision.backpressure_s must be finite and >= 0."
            )
        object.__setattr__(self, "metadata", freeze_mapping(self.metadata))


@dataclass(frozen=True, kw_only=True, slots=True)
class WebRTCChunkDelivery:
    """Completed WebRTC chunk delivery plus the model chunk summary."""

    delivery: object
    step_index: int
    frame_count: int
    generation: int
    force_keyframe: bool
    metadata: Mapping[str, object] = field(default_factory=dict)
    metrics: Mapping[str, float | int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", freeze_mapping(self.metadata))
        object.__setattr__(self, "metrics", freeze_mapping(self.metrics))


@runtime_checkable
class WebRTCOfferAnswerer(Protocol):
    """Creates an SDP answer after the shared session task has been scheduled."""

    async def create_answer(
        self,
        *,
        offer: WebRTCOfferRequest,
        session_task: asyncio.Task[RunResult],
    ) -> Mapping[str, str]: ...


@runtime_checkable
class BlockingPreparationService(Protocol):
    """Runs blocking scenario preparation outside the aiohttp event loop."""

    async def run(
        self,
        func: Callable[..., object],
        *args: object,
        **kwargs: object,
    ) -> object: ...


@runtime_checkable
class WebRTCOutputBridge(Protocol):
    """Thread-safe bridge from model-worker output writes to WebRTC delivery."""

    def begin_generation(self, generation: int) -> None: ...

    def submit_chunk(
        self,
        result: StepResult,
        *,
        generation: int,
        force_keyframe: bool = False,
    ) -> WebRTCOutputBridgeDecision: ...

    def close(self) -> None: ...


@runtime_checkable
class WebRTCSessionEdgeFactory(Protocol):
    """Builds per-peer shared session edges on the WebRTC control rank."""

    def create_session_edges(
        self,
        *,
        context: RunContext,
        spec: DemoSpec,
        scenario: PreparedScenario,
        provider: ModelInputProvider,
        adapter: DemoAdapter,
    ) -> SessionEdges: ...


class AsyncioBlockingPreparationService:
    """Default blocking-prep service backed by ``asyncio.to_thread``."""

    async def run(
        self,
        func: Callable[..., object],
        *args: object,
        **kwargs: object,
    ) -> object:
        return await asyncio.to_thread(func, *args, **kwargs)


class WebRTCTransportService:
    """Idempotent per-peer transport lifecycle for realtime session edges."""

    def __init__(
        self,
        *,
        loop: asyncio.AbstractEventLoop | None = None,
        on_close: Callable[[str | None], None] | None = None,
    ) -> None:
        self._loop = loop
        self._on_close = on_close
        self._closed_signal = _ThreadSafeActivationSignal(loop=loop)
        self._lock = threading.Lock()
        self._closed = False
        self._close_reason: str | None = None
        self._close_count = 0
        self.last_client_message_at: float | None = None

    @property
    def close_count(self) -> int:
        """Number of effective transport closes, after idempotency."""

        return self._close_count

    @property
    def close_reason(self) -> str | None:
        return self._close_reason

    @property
    def closed_signal(self) -> "_ThreadSafeActivationSignal":
        return self._closed_signal

    def mark_client_message(self, timestamp_s: float) -> None:
        if not math.isfinite(timestamp_s) or timestamp_s < 0.0:
            raise ValueError("timestamp_s must be finite and >= 0.")
        self.last_client_message_at = float(timestamp_s)

    def is_active(self) -> bool:
        return not self._closed

    def disconnect(self, reason: str = "client disconnected") -> None:
        self.close(reason=reason)

    def close(self, reason: str | None = None) -> None:
        callback: Callable[[str | None], None] | None = None
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._close_reason = reason
            self._close_count += 1
            callback = self._on_close
        self._closed_signal.set()
        if callback is not None:
            callback(reason)


@dataclass(slots=True)
class WebRTCActivationPolicy:
    """Activate on the first browser action/event or stop on disconnect."""

    input_source: "WebRTCInputSource"
    transport: WebRTCTransportService
    timeout_s: float | None = None
    timeout_reason: str = "activation timed out"
    anchor_clock: bool = True

    def __post_init__(self) -> None:
        if self.timeout_s is not None and self.timeout_s <= 0.0:
            raise ValueError("timeout_s must be > 0 when set.")
        if not self.timeout_reason.strip():
            raise ValueError("timeout_reason must be non-empty.")

    async def wait_until_active(
        self,
        clock: RealtimeClock | DeterministicClock,
    ) -> ActivationResult:
        if self.input_source.activation_signal.is_set():
            self._anchor(clock)
            return ActivationResult(activated=True)
        if not self.transport.is_active():
            return ActivationResult(
                activated=False,
                reason=self.transport.close_reason or "transport closed",
            )

        activation_task = asyncio.create_task(
            self.input_source.activation_signal.wait()
        )
        closed_task = asyncio.create_task(self.transport.closed_signal.wait())
        try:
            done, pending = await asyncio.wait(
                {activation_task, closed_task},
                timeout=self.timeout_s,
                return_when=asyncio.FIRST_COMPLETED,
            )
            if not done:
                return ActivationResult(
                    activated=False,
                    reason=self.timeout_reason,
                )
            for task in done:
                task.result()
            if not self.transport.is_active():
                return ActivationResult(
                    activated=False,
                    reason=self.transport.close_reason or "transport closed",
                )
            self._anchor(clock)
            return ActivationResult(activated=True)
        finally:
            for task in (activation_task, closed_task):
                if not task.done():
                    task.cancel()
            await asyncio.gather(
                activation_task,
                closed_task,
                return_exceptions=True,
            )

    def _anchor(self, clock: RealtimeClock | DeterministicClock) -> None:
        if not self.anchor_clock or not clock.is_realtime:
            return
        now = getattr(clock, "now", None)
        anchor = getattr(clock, "anchor", None)
        if callable(now) and callable(anchor):
            anchor(now())


@dataclass(slots=True)
class WebRTCInputSource:
    """Realtime source fed by browser data-channel events."""

    resampler: Any
    legacy_segment_resampler: Any | None = None
    legacy_segments_metadata_key: str | None = None
    max_lag_s: float | None = None
    catch_up_policy: CatchUpPolicy = "fold"
    user_input_schema: UserInputSchema = field(
        default_factory=lambda: WEBRTC_USER_INPUT_SCHEMA
    )
    is_finite: bool = False
    is_deterministic: bool = False
    _activation_signal: "_ThreadSafeActivationSignal" = field(
        default_factory=lambda: _ThreadSafeActivationSignal(),
        init=False,
        repr=False,
    )
    _events: deque[UserInputEvent] = field(
        default_factory=deque,
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        if self.max_lag_s is not None and (
            not math.isfinite(self.max_lag_s) or self.max_lag_s < 0.0
        ):
            raise ValueError("max_lag_s must be finite and >= 0.")
        if self.catch_up_policy != "fold":
            raise NotImplementedError(
                f"Catch-up policy {self.catch_up_policy!r} has no WebRTC analog yet."
            )

    @property
    def activation_signal(self) -> "_ThreadSafeActivationSignal":
        return self._activation_signal

    def is_finished(self) -> bool:
        return False

    def reset(self, *, start_v: float) -> None:
        self.resampler.reset(start_v=start_v)
        if self.legacy_segment_resampler is not None:
            self.legacy_segment_resampler.reset(start_v=start_v)
        self._events.clear()
        self._activation_signal.clear()

    def handle_browser_message(
        self,
        raw_message: object,
        *,
        timestamp_s: float,
    ) -> WebRTCMessageResult:
        """Translate one browser data-channel message into typed user inputs."""

        if not isinstance(raw_message, str):
            return WebRTCMessageResult(kind="error", error="Expected text payload.")
        try:
            payload = json.loads(raw_message)
        except json.JSONDecodeError:
            return WebRTCMessageResult(kind="error", error="Invalid JSON payload.")
        if not isinstance(payload, dict):
            return WebRTCMessageResult(
                kind="error",
                error="Payload must be a JSON object.",
            )
        return self.handle_browser_payload(payload, timestamp_s=timestamp_s)

    def handle_browser_payload(
        self,
        payload: Mapping[str, object],
        *,
        timestamp_s: float,
    ) -> WebRTCMessageResult:
        message_type = str(payload.get("type", "")).strip().lower()
        if message_type == MESSAGE_TYPE_HEARTBEAT:
            return WebRTCMessageResult(kind="heartbeat")
        if message_type == MESSAGE_TYPE_DISCONNECT:
            return WebRTCMessageResult(kind="disconnect")
        if message_type == MESSAGE_TYPE_EVENT:
            return self._record_text_event(payload, timestamp_s=timestamp_s)
        if message_type == MESSAGE_TYPE_ACTION:
            action_payload = payload.get("action", payload)
            if not isinstance(action_payload, Mapping):
                return WebRTCMessageResult(
                    kind="error",
                    error="'action' must be an object.",
                )
            return self._record_action(
                {str(key): value for key, value in action_payload.items()},
                timestamp_s=timestamp_s,
            )
        return WebRTCMessageResult(
            kind="error",
            error=(
                "Unsupported message type, expected "
                "'action', 'event', 'heartbeat', or 'disconnect'."
            ),
        )

    def record_user_event(
        self,
        *,
        timestamp_s: float,
        event_type: str,
        payload: Mapping[str, object],
        source_event_id: str | None = None,
        activate: bool = True,
    ) -> None:
        event = UserInputEvent(
            timestamp_s=timestamp_s,
            event_type=event_type,
            payload=dict(payload),
            source="webrtc",
            source_event_id=source_event_id,
        )
        self.user_input_schema.validate_event(event)
        self._events.append(event)
        if activate:
            self._activation_signal.set()

    async def next_realtime_window(
        self,
        *,
        request: StepRequirements,
        clock: RealtimeClock,
    ) -> RealtimeWindowResult:
        input_frame_count = input_frame_count_from_request(request)
        chunk_duration_s = input_frame_count * float(self.resampler.dt)
        if chunk_duration_s <= 0.0:
            raise ValueError("Realtime resampler dt must produce a positive window.")

        window_end_s = float(self.resampler.next_chunk_start_v) + chunk_duration_s
        await clock.wait_until_window_end(window_end_s)
        pre_catch_up_start_s = float(self.resampler.next_chunk_start_v)
        catch_up = clock.catch_up(
            request=request,
            max_lag_s=self.max_lag_s
            if self.max_lag_s is not None
            else chunk_duration_s,
            policy=self.catch_up_policy,
        )
        start_s = float(self.resampler.next_chunk_start_v)
        frame_times = tuple(self.resampler.sample_chunk(input_frame_count))
        end_s = float(self.resampler.next_chunk_start_v)
        metadata: dict[str, object] = {}
        legacy_resampler = self.legacy_segment_resampler
        legacy_metadata_key = self.legacy_segments_metadata_key
        if legacy_resampler is not None and legacy_metadata_key is not None:
            legacy_resampler.next_chunk_start_v = start_s
            segments, legacy_frame_times = legacy_resampler.sample_chunk(
                input_frame_count
            )
            metadata[legacy_metadata_key] = tuple(segments)
            frame_times = tuple(legacy_frame_times)
        if start_s > pre_catch_up_start_s:
            metadata[WEBRTC_SKIPPED_INPUTS_METADATA_KEY] = UserInputs(
                events=self._events_for_window(pre_catch_up_start_s, start_s)
            )
            metadata[WEBRTC_SKIPPED_WINDOW_METADATA_KEY] = (
                pre_catch_up_start_s,
                start_s,
            )
        window = RealtimeWindowResult(
            window=_user_input_window(
                start_s=start_s,
                end_s=end_s,
                frame_times=tuple(frame_times),
                inputs=UserInputs(events=self._events_for_window(start_s, end_s)),
                metadata=metadata,
            ),
            catch_up=catch_up,
        )
        self._prune_events(before_s=start_s)
        return window

    def _record_action(
        self,
        payload: Mapping[str, object],
        *,
        timestamp_s: float,
    ) -> WebRTCMessageResult:
        event = str(payload.get("event", "")).strip().lower()
        if event == "step":
            self._activation_signal.set()
            return WebRTCMessageResult(kind="action", activated=True)
        if event not in {"keydown", "keyup"}:
            return WebRTCMessageResult(
                kind="error",
                error=f"Unsupported event={event!r}; expected 'keydown' or 'keyup'.",
            )
        key = str(payload.get("key", "")).strip()
        if not key:
            return WebRTCMessageResult(
                kind="error",
                error="Action payload must include non-empty 'key'.",
            )
        if self.legacy_segment_resampler is not None:
            self.legacy_segment_resampler.on_edge(
                arrival_t=timestamp_s,
                event=event,
                key=key,
            )
        self.record_user_event(
            timestamp_s=timestamp_s,
            event_type="key_down" if event == "keydown" else "key_up",
            payload={"key": key},
        )
        return WebRTCMessageResult(kind="action", activated=True)

    def _record_text_event(
        self,
        payload: Mapping[str, object],
        *,
        timestamp_s: float,
    ) -> WebRTCMessageResult:
        state = str(payload.get("state", "trigger")).strip().lower() or "trigger"
        event_id = str(payload.get("event_id", payload.get("id", ""))).strip()
        clears = state in _CLEAR_EVENT_STATES
        if not event_id and not clears:
            return WebRTCMessageResult(
                kind="error",
                error=(
                    "Event payload must include non-empty 'event_id' unless state "
                    "clears the active event."
                ),
            )
        active_event_id = None if clears else event_id
        self.record_user_event(
            timestamp_s=timestamp_s,
            event_type="text_event",
            payload={"event_id": active_event_id, "state": state},
            source_event_id=active_event_id,
        )
        return WebRTCMessageResult(kind="event", activated=True)

    def _events_for_window(
        self,
        start_s: float,
        end_s: float,
    ) -> tuple[UserInputEvent, ...]:
        return tuple(
            sorted(
                (
                    event
                    for event in self._events
                    if start_s <= event.timestamp_s < end_s
                ),
                key=lambda event: event.timestamp_s,
            )
        )

    def _prune_events(self, *, before_s: float) -> None:
        self._events = deque(
            event for event in self._events if event.timestamp_s >= before_s
        )


class WebRTCOutputSink:
    """Output sink that schedules WebRTC media delivery without blocking."""

    produces_artifacts = False

    def __init__(self, *, bridge: WebRTCOutputBridge) -> None:
        self._bridge = bridge
        self._opened = False
        self._closed = True
        self._bridge_closed = False
        self._generation = 0
        self._force_keyframe = False
        self.session_info: SessionInfo | None = None

    def open(self, session_info: SessionInfo) -> None:
        self.session_info = session_info
        self._opened = True
        self._closed = False
        self._generation = 0
        self._force_keyframe = True
        self._bridge.begin_generation(0)

    def begin_generation(self, generation: int) -> None:
        if generation < 0:
            raise ValueError("generation must be >= 0.")
        self._generation = generation
        self._force_keyframe = True
        self._bridge.begin_generation(generation)

    def write(self, result: StepResult) -> OutputDecision:
        if not self._opened or self._closed:
            raise RuntimeError("Cannot write to a closed output sink.")
        decision = self._bridge.submit_chunk(
            result,
            generation=self._generation,
            force_keyframe=self._force_keyframe,
        )
        self._force_keyframe = False
        return OutputDecision(
            should_stop=decision.should_stop,
            dropped=decision.dropped,
            drop_policy=decision.drop_policy,
            backpressure_s=decision.backpressure_s,
            metadata=decision.metadata,
        )

    def close(self) -> Sequence[Any]:
        if self._bridge_closed:
            return ()
        self._closed = True
        self._opened = False
        self._bridge.close()
        self._bridge_closed = True
        return ()


class ThreadSafeWebRTCOutputBridge:
    """Schedule async encoder delivery from any thread without blocking writes."""

    def __init__(
        self,
        *,
        loop: asyncio.AbstractEventLoop,
        video_encoder: Any,
        video_track: Any,
        max_pending_chunks: int = 2,
        close_track: bool = True,
        on_delivery: Callable[[object], None] | None = None,
        on_chunk_delivery: Callable[[WebRTCChunkDelivery], None] | None = None,
        on_error: Callable[[BaseException], None] | None = None,
    ) -> None:
        if max_pending_chunks <= 0:
            raise ValueError("max_pending_chunks must be > 0.")
        self._loop = loop
        self._video_encoder = video_encoder
        self._video_track = video_track
        self._max_pending_chunks = max_pending_chunks
        self._close_track = close_track
        self._on_delivery = on_delivery
        self._on_chunk_delivery = on_chunk_delivery
        self._on_error = on_error
        self._pending: dict[Future[WebRTCChunkDelivery], int] = {}
        self._delivery_lock = asyncio.Lock()
        self._lock = threading.Lock()
        self._closed = False
        self._generation = 0

    @property
    def pending_count(self) -> int:
        with self._lock:
            return len(self._pending)

    def begin_generation(self, generation: int) -> None:
        if generation < 0:
            raise ValueError("generation must be >= 0.")
        with self._lock:
            if self._closed or generation <= self._generation:
                return
            self._generation = generation
            stale = tuple(
                future
                for future, future_generation in self._pending.items()
                if future_generation < generation
            )
        for future in stale:
            future.cancel()
        self._schedule_track_flush()

    def submit_chunk(
        self,
        result: StepResult,
        *,
        generation: int,
        force_keyframe: bool = False,
    ) -> WebRTCOutputBridgeDecision:
        prepare = getattr(self._video_encoder, "prepare_chunk_payload", None)
        deliver = getattr(self._video_encoder, "deliver_prepared_chunk", None)
        if not callable(prepare) or not callable(deliver):
            raise TypeError(
                "ThreadSafeWebRTCOutputBridge requires a video encoder with "
                "prepare_chunk_payload(...) and deliver_prepared_chunk(...)."
            )
        with self._lock:
            if self._closed:
                return WebRTCOutputBridgeDecision(
                    accepted=False,
                    should_stop=True,
                    dropped=True,
                    drop_policy="drop_newest",
                    metadata={"reason": "closed"},
                )
            if generation < self._generation:
                return WebRTCOutputBridgeDecision(
                    accepted=False,
                    dropped=True,
                    drop_policy="drop_newest",
                    metadata={"reason": "stale generation"},
                )
            if len(self._pending) >= self._max_pending_chunks:
                stale = self._pop_pending_locked()
            else:
                stale = ()
        if stale:
            self._cancel_stale_deliveries(stale)
            self._schedule_track_flush()
        payload = prepare(result, self._video_track)
        chunk = WebRTCChunkDelivery(
            delivery=None,
            step_index=result.step_index,
            frame_count=result.frame_count,
            generation=generation,
            force_keyframe=force_keyframe,
            metadata=result.metadata,
            metrics=result.metrics,
        )
        with self._lock:
            if self._closed:
                return WebRTCOutputBridgeDecision(
                    accepted=False,
                    should_stop=True,
                    dropped=True,
                    drop_policy="drop_newest",
                    metadata={"reason": "closed"},
                )
            if generation < self._generation:
                return WebRTCOutputBridgeDecision(
                    accepted=False,
                    dropped=True,
                    drop_policy="drop_newest",
                    metadata={"reason": "stale generation"},
                )
            if len(self._pending) >= self._max_pending_chunks:
                stale = self._pop_pending_locked()
            else:
                stale = ()
        if stale:
            self._cancel_stale_deliveries(stale)
            self._schedule_track_flush()
        with self._lock:
            if self._closed:
                return WebRTCOutputBridgeDecision(
                    accepted=False,
                    should_stop=True,
                    dropped=True,
                    drop_policy="drop_newest",
                    metadata={"reason": "closed"},
                )
            if generation < self._generation:
                return WebRTCOutputBridgeDecision(
                    accepted=False,
                    dropped=True,
                    drop_policy="drop_newest",
                    metadata={"reason": "stale generation"},
                )
            future = asyncio.run_coroutine_threadsafe(
                self._deliver(
                    payload,
                    chunk=chunk,
                    generation=generation,
                    force_keyframe=force_keyframe,
                ),
                self._loop,
            )
            self._pending[future] = generation
            future.add_done_callback(self._on_done)

        return WebRTCOutputBridgeDecision(
            accepted=True,
            backpressure_s=self._track_backpressure_s(),
        )

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            pending = tuple(self._pending)
        for future in pending:
            future.cancel()
        if self._close_track:
            self._schedule_track_close()

    async def _deliver(
        self,
        payload: object,
        *,
        chunk: WebRTCChunkDelivery,
        generation: int,
        force_keyframe: bool,
    ) -> WebRTCChunkDelivery:
        with self._lock:
            if self._closed or generation < self._generation:
                raise asyncio.CancelledError
        async with self._delivery_lock:
            with self._lock:
                if self._closed or generation < self._generation:
                    raise asyncio.CancelledError
            await self._flush_full_track_queue(frame_count=chunk.frame_count)
            delivery = await self._video_encoder.deliver_prepared_chunk(
                payload,
                self._video_track,
                force_keyframe=force_keyframe,
            )
        with self._lock:
            if self._closed or generation < self._generation:
                stale_after_delivery = True
            else:
                stale_after_delivery = False
        if stale_after_delivery:
            self._schedule_track_flush()
            raise asyncio.CancelledError
        return WebRTCChunkDelivery(
            delivery=delivery,
            step_index=chunk.step_index,
            frame_count=chunk.frame_count,
            generation=chunk.generation,
            force_keyframe=chunk.force_keyframe,
            metadata=chunk.metadata,
            metrics=chunk.metrics,
        )

    def _on_done(self, future: Future[WebRTCChunkDelivery]) -> None:
        with self._lock:
            self._pending.pop(future, None)
        if future.cancelled():
            return
        try:
            result = future.result()
        except BaseException as exc:
            if self._on_error is not None:
                self._on_error(exc)
            return
        if self._on_delivery is not None:
            self._on_delivery(result.delivery)
        if self._on_chunk_delivery is not None:
            self._on_chunk_delivery(result)

    def _pop_pending_locked(self) -> tuple[Future[WebRTCChunkDelivery], ...]:
        pending = tuple(self._pending)
        self._pending.clear()
        return pending

    @staticmethod
    def _cancel_stale_deliveries(
        futures: Sequence[Future[WebRTCChunkDelivery]],
    ) -> None:
        for future in futures:
            future.cancel()

    def _track_backpressure_s(self) -> float:
        qsize = getattr(self._video_track, "qsize", None)
        fps = getattr(self._video_track, "fps", None) or getattr(
            self._video_encoder,
            "fps",
            None,
        )
        if not callable(qsize) or fps is None:
            return 0.0
        try:
            queue_depth = int(qsize())
            frames_per_second = float(fps)
        except (TypeError, ValueError):
            return 0.0
        if frames_per_second <= 0.0:
            return 0.0
        return max(0.0, queue_depth / frames_per_second)

    async def _flush_full_track_queue(self, *, frame_count: int) -> None:
        if frame_count <= 0:
            return
        qsize = getattr(self._video_track, "qsize", None)
        flush = getattr(self._video_track, "flush", None)
        if not callable(qsize) or not callable(flush):
            return
        try:
            queue_depth = int(qsize())
        except (TypeError, ValueError):
            return
        if queue_depth < frame_count:
            return

        # WebRTC is interactive: a full track queue means a whole generated
        # chunk is stale relative to the latest input window. Drop that queued
        # media before enqueueing the current chunk so visual latency stays
        # bounded instead of preserving every frame.
        result = flush()
        if inspect.isawaitable(result):
            await result

    def _schedule_track_close(self) -> None:
        close = getattr(self._video_track, "close", None)
        if not callable(close):
            return
        try:
            result = close()
            if inspect.isawaitable(result):
                asyncio.run_coroutine_threadsafe(result, self._loop)
        except BaseException as exc:
            if self._on_error is not None:
                self._on_error(exc)

    def _schedule_track_flush(self) -> None:
        flush = getattr(self._video_track, "flush", None)
        if not callable(flush):
            return
        try:
            result = flush()
            if inspect.isawaitable(result):
                asyncio.run_coroutine_threadsafe(result, self._loop)
        except BaseException as exc:
            if self._on_error is not None:
                self._on_error(exc)


class WebRTCRunMode:
    """Shared realtime run mode that delegates peer-specific edges to WebRTC."""

    name = "webrtc"
    capabilities = RunModeCapabilities(
        realtime=True,
        supports_backpressure=True,
        supports_interactive_events=True,
    )

    def __init__(
        self,
        *,
        edge_factory: WebRTCSessionEdgeFactory,
        blocking_preparation: BlockingPreparationService | None = None,
        driver: AsyncSessionDriver | None = None,
        error_policy: WebRTCErrorPolicy | None = None,
    ) -> None:
        self._edge_factory = edge_factory
        self._blocking_preparation = (
            blocking_preparation or AsyncioBlockingPreparationService()
        )
        self._driver = driver or RealtimeSessionDriver()
        self._error_policy = error_policy or WebRTCErrorPolicy()

    @property
    def blocking_preparation(self) -> BlockingPreparationService:
        return self._blocking_preparation

    @property
    def error_policy(self) -> WebRTCErrorPolicy:
        return self._error_policy

    def validate_run(self, *, spec: DemoSpec, adapter: DemoAdapter) -> None:
        del adapter
        if spec.output.mode != "webrtc":
            raise ValueError("WebRTCRunMode requires WebRTC output.")

    def validate_session(
        self,
        *,
        spec: DemoSpec,
        scenario: PreparedScenario,
        adapter: DemoAdapter,
        provider: ModelInputProvider,
    ) -> None:
        del spec, scenario, adapter
        if not provider.capabilities.supports_realtime_clock:
            raise ValueError("WebRTC providers must support realtime clocks.")

    def create_run_context(
        self,
        *,
        spec: DemoSpec,
        adapter: DemoAdapter,
        host: RuntimeHost,
        model_warmup_plan: ModelWarmupPlan,
    ) -> RunContext:
        del spec, adapter
        services: dict[str, object] = {}
        if host.is_control_rank:
            services["blocking_preparation"] = self._blocking_preparation
        return RunContext(
            host=host,
            run_metrics=InMemorySessionMetricsRecorder(),
            admission=SingleSessionAdmissionPolicy(
                health_check=lambda: host.is_control_rank and host.is_healthy
            ),
            model_warmup_plan=model_warmup_plan,
            services=services,
        )

    def create_session_edges(
        self,
        *,
        context: RunContext,
        spec: DemoSpec,
        scenario: PreparedScenario,
        provider: ModelInputProvider,
        adapter: DemoAdapter,
    ) -> SessionEdges:
        if not context.host.is_control_rank:
            raise RuntimeError("WebRTC session edges are control-rank only.")
        edges = self._edge_factory.create_session_edges(
            context=context,
            spec=spec,
            scenario=scenario,
            provider=provider,
            adapter=adapter,
        )
        if not isinstance(edges, SessionEdges):
            raise TypeError(
                "WebRTC edge factory must return SessionEdges, "
                f"got {type(edges).__name__}."
            )
        return edges

    def select_driver(self) -> AsyncSessionDriver:
        return self._driver


class WebRTCSessionOfferHandler:
    """Reserve, prepare, and launch one WebRTC session before SDP negotiation."""

    def __init__(
        self,
        *,
        context: RunContext,
        spec: DemoSpec,
        adapter: DemoAdapter,
        run_mode: WebRTCRunMode,
        answerer: WebRTCOfferAnswerer,
        pipeline: StepPipeline | None = None,
        session_helper: Callable[..., Coroutine[Any, Any, RunResult]] | None = None,
        busy_message: str = "Another WebRTC session is already active.",
        session_tasks: MutableSet[asyncio.Task[RunResult]] | None = None,
    ) -> None:
        self._context = context
        self._spec = spec
        self._adapter = adapter
        self._run_mode = run_mode
        self._answerer = answerer
        self._pipeline = pipeline or StepPipeline()
        self._session_helper = session_helper or run_demo_session_async
        self._busy_message = busy_message
        self._session_tasks = session_tasks if session_tasks is not None else set()

    async def handle_offer(
        self,
        *,
        offer_sdp: str,
        offer_type: str,
    ) -> Mapping[str, str]:
        if not self._context.host.is_control_rank:
            raise RuntimeError("WebRTC offers are handled only on the control rank.")
        reservation = self._context.admission.try_reserve()
        if reservation is None:
            raise SessionBusyError(self._busy_message)

        task: asyncio.Task[RunResult] | None = None
        try:
            scenario = await self._run_blocking_prepare(self._spec)
            task = asyncio.create_task(
                self._session_helper(
                    context=self._context,
                    spec=self._spec,
                    scenario=scenario,
                    adapter=self._adapter,
                    run_mode=self._run_mode,
                    pipeline=self._pipeline,
                    reservation=reservation,
                )
            )
            self._track_task(task)
            answer = await self._answerer.create_answer(
                offer=WebRTCOfferRequest(sdp=offer_sdp, type=offer_type),
                session_task=task,
            )
            return dict(answer)
        except Exception:
            if task is not None:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
            reservation.release()
            raise

    async def _run_blocking_prepare(self, spec: DemoSpec) -> PreparedScenario:
        service = self._run_mode.blocking_preparation
        result = await service.run(self._adapter.prepare_scenario, spec)
        if not isinstance(result, PreparedScenario):
            raise TypeError(
                "DemoAdapter.prepare_scenario must return PreparedScenario, "
                f"got {type(result).__name__}."
            )
        return result

    def _track_task(self, task: asyncio.Task[RunResult]) -> None:
        self._session_tasks.add(task)
        task.add_done_callback(self._discard_task)

    def _discard_task(self, task: asyncio.Task[RunResult]) -> None:
        self._session_tasks.discard(task)
        if task.cancelled():
            return
        with contextlib.suppress(Exception):
            task.exception()


WEBRTC_USER_INPUT_SCHEMA = UserInputSchema(
    capabilities=(
        UserInputCapability(
            event_type="key_down",
            input_modality="keyboard",
            payload_fields=frozenset({"key"}),
        ),
        UserInputCapability(
            event_type="key_up",
            input_modality="keyboard",
            payload_fields=frozenset({"key"}),
        ),
        UserInputCapability(
            event_type="text_event",
            input_modality="text",
            payload_fields=frozenset({"event_id", "state"}),
        ),
    ),
    description="browser WebRTC data-channel events",
)


def _user_input_window(
    *,
    start_s: float,
    end_s: float,
    frame_times: Sequence[float],
    inputs: UserInputs,
    metadata: Mapping[str, object],
) -> UserInputWindow:
    return UserInputWindow(
        start_s=start_s,
        end_s=end_s,
        frame_times=frame_times,
        inputs=inputs,
        metadata=metadata,
    )


class _ThreadSafeActivationSignal:
    def __init__(self, *, loop: asyncio.AbstractEventLoop | None = None) -> None:
        self._loop = loop
        self._event = asyncio.Event()

    def is_set(self) -> bool:
        return self._event.is_set()

    async def wait(self) -> object:
        return await self._event.wait()

    def set(self) -> None:
        if self._event.is_set():
            return
        if self._loop is not None and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._event.set)
            return
        self._event.set()

    def clear(self) -> None:
        self._event.clear()


__all__ = [
    "AsyncioBlockingPreparationService",
    "BlockingPreparationService",
    "ThreadSafeWebRTCOutputBridge",
    "WEBRTC_USER_INPUT_SCHEMA",
    "WEBRTC_SKIPPED_INPUTS_METADATA_KEY",
    "WEBRTC_SKIPPED_WINDOW_METADATA_KEY",
    "WebRTCActivationPolicy",
    "WebRTCInputSource",
    "WebRTCMessageResult",
    "WebRTCOfferAnswerer",
    "WebRTCOfferRequest",
    "WebRTCOutputBridge",
    "WebRTCOutputBridgeDecision",
    "WebRTCChunkDelivery",
    "WebRTCOutputSink",
    "WebRTCRunMode",
    "WebRTCSessionEdgeFactory",
    "WebRTCSessionOfferHandler",
    "WebRTCTransportService",
]
