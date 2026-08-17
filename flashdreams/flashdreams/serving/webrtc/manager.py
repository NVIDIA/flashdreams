# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""WebRTC session lifecycle and control-message orchestration."""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import json
import time
from collections import deque
from collections.abc import Callable, Mapping
from collections.abc import Set as AbstractSet
from dataclasses import dataclass, field, replace
from typing import Any, Generic, TypeVar, cast

from aiortc import (
    RTCConfiguration,
    RTCPeerConnection,
    RTCRtpSender,
    RTCSessionDescription,
)
from loguru import logger

from flashdreams.runtime.canonical import DeviceConverterSchema
from flashdreams.runtime.demo import (
    DemoSpec,
    InMemorySessionMetricsRecorder,
    ModelInputProvider,
    ModelWarmupPlan,
    PreparedScenario,
    PreparedStep,
    ProviderCapabilities,
    RealtimeEventResampler,
    ResamplerRealtimeClock,
    RunContext,
    RuntimeHost,
    SessionEdges,
    SessionInfo,
    SingleSessionAdmissionPolicy,
    StepPipeline,
    UserInputWindow,
    WebRTCErrorPolicy,
    WebRTCOutputSpec,
    run_demo_session_async,
)
from flashdreams.runtime.inputs import (
    CanonicalInputSchema,
    InferenceInput,
    InferenceInputSchema,
    TimeWindow,
    UserInputEvent,
    UserInputs,
    UserInputSchema,
)
from flashdreams.runtime.keyboard import DEFAULT_SUPPORTED_KEYS, normalize_key
from flashdreams.runtime.mapping import InputMapping
from flashdreams.runtime.types import StepRequest, StepResult
from flashdreams.serving.webrtc.encoders import (
    DefaultRTCEncoder,
    EncoderBackend,
    VideoEncoder,
    select_encoder,
)
from flashdreams.serving.webrtc.media import BufferedVideoTrack, NVENCVideoTrack
from flashdreams.serving.webrtc.messages import (
    MESSAGE_TYPE_ACTION,
    MESSAGE_TYPE_DISCONNECT,
    MESSAGE_TYPE_EVENT,
    MESSAGE_TYPE_HEARTBEAT,
    make_chunk_done_payload,
    make_error_payload,
    make_event_ack_payload,
)
from flashdreams.serving.webrtc.runtime import (
    WebRTCControlSignal,
    WebRTCRuntimeConfig,
)
from flashdreams.serving.webrtc.server import SessionBusyError
from flashdreams.serving.webrtc.services import (
    WEBRTC_SKIPPED_INPUTS_METADATA_KEY,
    WEBRTC_SKIPPED_WINDOW_METADATA_KEY,
    WEBRTC_USER_INPUT_SCHEMA,
    ThreadSafeWebRTCOutputBridge,
    WebRTCActivationPolicy,
    WebRTCChunkDelivery,
    WebRTCInputSource,
    WebRTCManagerLifecycle,
    WebRTCOutputSink,
    WebRTCRunMode,
    WebRTCTransportService,
)
from flashdreams.serving.webrtc.warmup import (
    run_loopback_warmup_session,
    wait_for_ice_gathering_complete,
)

__all__ = [
    "BaseWebRTCSessionManager",
    "ManagedWebRTCSession",
    "WebRTCControlSignal",
    "StepResult",
]

# Close the active session if no client heartbeat/control message arrives
# within this many seconds. Browsers sends periodic heartbeats.
DEFAULT_CLIENT_LIVENESS_TIMEOUT_S = 10.0

# How often the liveness watchdog wakes to re-check the elapsed-since-last-message.
_CLIENT_LIVENESS_CHECK_INTERVAL_S = 1.0
_DEFAULT_PERF_LOG_INTERVAL_CHUNKS = 5
_MAX_SESSION_USER_EVENTS = 1024
"""Maximum unconsumed raw events kept for an ``InferenceSession`` step."""
_RELEASE_USER_EVENT_TYPES = frozenset({"key_up"})
_KEY_USER_EVENT_TYPES = frozenset({"key_down", "key_up"})
_SESSION_INPUT_KEY = "webrtc_session_input"
_STEP_REQUEST_KEY = "webrtc_step_request"
_SEGMENTS_KEY = "webrtc_segments"
_FRAME_TIMES_KEY = "webrtc_frame_times"
_LEGACY_SPARSE_KEY_SEGMENTS_METADATA_KEY = "sparse_key_segments"

_RuntimeT = TypeVar("_RuntimeT")
_RuntimeConfigT = TypeVar("_RuntimeConfigT", bound=WebRTCRuntimeConfig)


class _InferenceSessionExhausted(RuntimeError):
    """Raised when an ``InferenceSession`` reports normal completion."""


def _summarize_sdp_candidates(sdp: str) -> str:
    candidates = [
        line.removeprefix("a=candidate:")
        for line in sdp.splitlines()
        if line.startswith("a=candidate:")
    ]
    if not candidates:
        return "0 candidates"

    protocols: dict[str, int] = {}
    addresses: set[str] = set()
    endpoints: list[str] = []
    for candidate in candidates:
        parts = candidate.split()
        if len(parts) >= 5:
            protocols[parts[2].lower()] = protocols.get(parts[2].lower(), 0) + 1
            addresses.add(parts[4])
        if len(parts) >= 6:
            endpoints.append(f"{parts[2].lower()}://{parts[4]}:{parts[5]}")
    protocol_summary = ",".join(
        f"{key}={value}" for key, value in sorted(protocols.items())
    )
    address_summary = ",".join(sorted(addresses)[:8])
    if len(addresses) > 8:
        address_summary += f",+{len(addresses) - 8} more"
    endpoint_summary = ",".join(endpoints[:12])
    if len(endpoints) > 12:
        endpoint_summary += f",+{len(endpoints) - 12} more"
    return (
        f"{len(candidates)} candidates protocols=[{protocol_summary}] "
        f"addresses=[{address_summary}] endpoints=[{endpoint_summary}]"
    )


def _stat_float(
    stats: Mapping[str, float | int], name: str, default: float = 0.0
) -> float:
    value = stats.get(name)
    if value is None:
        return default
    return float(value)


def _stat_ms(
    stats: Mapping[str, float | int], name: str, default_ms: float = 0.0
) -> float:
    return _stat_float(stats, name, default_ms / 1e3) * 1e3


def _stat_int(stats: Mapping[str, float | int], name: str) -> int:
    return int(round(_stat_float(stats, name)))


def _runtime_drives_inference_session(runtime: Any) -> bool:
    return callable(getattr(runtime, "start_inference_session", None))


def _run_on_event_loop(loop: asyncio.AbstractEventLoop, awaitable: Any) -> Any:
    """Run one legacy async WebRTC runtime call from a RuntimeHost worker."""
    return asyncio.run_coroutine_threadsafe(awaitable, loop).result()


def _step_request_from_requirements(
    request: Any,
    *,
    window: TimeWindow,
) -> StepRequest:
    metadata = dict(getattr(request, "metadata", {}))
    metadata["input_frame_count"] = request.input_frame_count
    steady_output_frame_count = getattr(request, "steady_output_frame_count", None)
    if steady_output_frame_count is not None:
        metadata["steady_output_frame_count"] = steady_output_frame_count
    return StepRequest(
        step_index=request.step_index,
        inference_input_schema=getattr(request, "inference_input_schema", None),
        user_input_window=window,
        metadata=metadata,
    )


def _encoder_backend_from_config(value: object) -> EncoderBackend:
    backend = str(value)
    if backend not in {"auto", "default", "nvenc"}:
        raise ValueError(
            f"encoder_backend must be 'auto', 'default', or 'nvenc', got {backend!r}."
        )
    return cast(EncoderBackend, backend)


def _gpu_id_from_device_spec(device_spec: str) -> int:
    if not device_spec.startswith("cuda"):
        return 0
    _prefix, separator, index = device_spec.partition(":")
    if not separator or not index:
        return 0
    try:
        return int(index)
    except ValueError:
        return 0


class _LegacyWebRTCRuntimeAdapter:
    """Shared compatibility adapter from old async WebRTC runtimes to RuntimeHost."""

    def __init__(self, *, runtime: Any, loop: asyncio.AbstractEventLoop) -> None:
        self._runtime = runtime
        self._loop = loop

    def reset_for_new_session(self, session_input: Any = None) -> None:
        _run_on_event_loop(
            self._loop,
            self._runtime.reset_for_new_session(session_input=session_input),
        )

    def start_session(self, inputs: InferenceInput) -> "_LegacyWebRTCSessionAdapter":
        del inputs
        inference_session = None
        if _runtime_drives_inference_session(self._runtime):
            inference_session = _run_on_event_loop(
                self._loop,
                self._runtime.start_inference_session(),
            )
        return _LegacyWebRTCSessionAdapter(
            runtime=self._runtime,
            inference_session=inference_session,
            loop=self._loop,
        )

    def close(self) -> None:
        # The underlying async runtime is owned by BaseWebRTCSessionManager and
        # closed from shutdown(); RuntimeHost only owns this adapter's worker.
        return


class _LegacyWebRTCSessionAdapter:
    """RuntimeHost-facing session view over a legacy WebRTC runtime/session."""

    def __init__(
        self,
        *,
        runtime: Any,
        inference_session: Any | None,
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        self._runtime = runtime
        self._inference_session = inference_session
        self._loop = loop

    def session_info(self) -> SessionInfo:
        steady_frames: int | None = None
        try:
            steady_frames = int(self._runtime.peek_steady_output_num_frames())
        except Exception:
            steady_frames = None
        return SessionInfo(steady_output_frame_count=steady_frames)

    def next_step_request(self) -> StepRequest | None:
        if self._inference_session is not None:
            return self._inference_session.next_step_request()
        return self._runtime.next_step_request()

    def step(self, inputs: InferenceInput) -> StepResult:
        if self._inference_session is not None:
            result = self._inference_session.step(inputs)
        else:
            result = _run_on_event_loop(
                self._loop,
                self._runtime.step(
                    request=inputs.step[_STEP_REQUEST_KEY],
                    segments=list(inputs.step[_SEGMENTS_KEY]),
                    frame_times=list(inputs.step[_FRAME_TIMES_KEY]),
                ),
            )
            request = inputs.step[_STEP_REQUEST_KEY]
            if result.step_index != request.step_index:
                raise RuntimeError(
                    "Runtime result step does not match its request: "
                    f"requested {request.step_index}, got {result.step_index}."
                )
        if not isinstance(result, StepResult):
            raise TypeError(
                "WebRTC session steps must produce StepResult, got "
                f"{type(result).__name__}."
            )
        return result

    def reset(self, inputs: InferenceInput | None = None) -> None:
        session_input = None
        if inputs is not None:
            session_input = inputs.global_conditioning.get(_SESSION_INPUT_KEY)
        _run_on_event_loop(
            self._loop,
            self._runtime.reset_for_new_session(session_input=session_input),
        )
        if self._inference_session is not None:
            self._inference_session = _run_on_event_loop(
                self._loop,
                self._runtime.start_inference_session(),
            )

    def close(self) -> None:
        close = getattr(self._inference_session, "close", None)
        if callable(close):
            close()


class _LegacyWebRTCModelInputProvider:
    """Shared provider used until model-specific WebRTC providers land."""

    def __init__(self, *, runtime: Any, session_input: Any = None) -> None:
        self._runtime = runtime
        self._session_input = session_input
        self._uses_inference_session = _runtime_drives_inference_session(runtime)
        self._session_input_state_advanced = False
        self.capabilities = ProviderCapabilities(
            supports_realtime_clock=True,
            supports_reset=True,
            deterministic_given_inputs=False,
            user_input_schema=self._user_input_schema(),
        )

    def prepare_initial_input(self) -> InferenceInput:
        if self._session_input is None:
            return InferenceInput()
        return InferenceInput(
            global_conditioning={_SESSION_INPUT_KEY: self._session_input}
        )

    def prepare_step(
        self,
        *,
        request: Any,
        user_window: UserInputWindow,
    ) -> PreparedStep:
        if self._uses_inference_session:
            return PreparedStep(
                inference_input=self._prepare_inference_session_step(
                    request=request,
                    user_window=user_window,
                )
            )
        return PreparedStep(
            inference_input=self._prepare_segment_step(
                request=request,
                user_window=user_window,
            )
        )

    def reset(self, inputs: InferenceInput | None = None) -> None:
        del inputs
        self._session_input_state_advanced = False

    def close(self) -> None:
        return

    def _user_input_schema(self) -> UserInputSchema:
        schema = getattr(self._runtime, "input_source_schema", None)
        if isinstance(schema, UserInputSchema):
            return schema
        return WEBRTC_USER_INPUT_SCHEMA

    def _prepare_inference_session_step(
        self,
        *,
        request: Any,
        user_window: UserInputWindow,
    ) -> InferenceInput:
        self._advance_skipped_input_state(user_window)
        window_start = user_window.start_s
        if request.step_index == 0 and not self._session_input_state_advanced:
            window_start = 0.0
        window = TimeWindow(start_s=window_start, end_s=user_window.end_s)
        canonical_inputs = self._runtime.input_canonicalizer.canonicalize(
            user_window.inputs,
            window=window,
            source_schema=self._runtime.input_source_schema,
        )
        mapping = self._runtime.input_mapping
        inference_input = InferenceInput(
            metadata={
                **dict(user_window.metadata),
                "frame_times": tuple(user_window.frame_times),
                "window_start_s": window.start_s,
                "window_end_s": window.end_s,
            }
        )
        return mapping.map_step_inputs(
            canonical_inputs=canonical_inputs,
            inference_input=inference_input,
            request=_step_request_from_requirements(request, window=window),
        )

    def _advance_skipped_input_state(self, user_window: UserInputWindow) -> None:
        skipped_inputs = user_window.metadata.get(WEBRTC_SKIPPED_INPUTS_METADATA_KEY)
        skipped_window = user_window.metadata.get(WEBRTC_SKIPPED_WINDOW_METADATA_KEY)
        if not isinstance(skipped_inputs, UserInputs):
            return
        if not isinstance(skipped_window, tuple) or len(skipped_window) != 2:
            return
        start_value, end_value = skipped_window
        if not isinstance(start_value, int | float) or not isinstance(
            end_value,
            int | float,
        ):
            return
        start_s = float(start_value)
        end_s = float(end_value)
        if end_s <= start_s:
            return
        self._runtime.input_canonicalizer.canonicalize(
            skipped_inputs,
            window=TimeWindow(start_s=start_s, end_s=end_s),
            source_schema=self._runtime.input_source_schema,
        )
        self._session_input_state_advanced = True

    @staticmethod
    def _prepare_segment_step(
        *,
        request: Any,
        user_window: UserInputWindow,
    ) -> InferenceInput:
        segments = user_window.metadata.get(_LEGACY_SPARSE_KEY_SEGMENTS_METADATA_KEY)
        if not isinstance(segments, tuple):
            raise RuntimeError("WebRTC user window is missing resampled key segments.")
        window = TimeWindow(start_s=user_window.start_s, end_s=user_window.end_s)
        return InferenceInput(
            step={
                _STEP_REQUEST_KEY: _step_request_from_requirements(
                    request,
                    window=window,
                ),
                _SEGMENTS_KEY: tuple(segments),
                _FRAME_TIMES_KEY: tuple(user_window.frame_times),
            }
        )


class _LegacyWebRTCDemoAdapter:
    """Minimal adapter for the shared helper while WebRTC providers migrate."""

    model_id: str
    inference_input_schema = InferenceInputSchema()
    canonical_input_schema = CanonicalInputSchema()

    def __init__(
        self,
        *,
        runtime: Any,
        identity: str,
        session_input: Any = None,
    ) -> None:
        self._runtime = runtime
        self.model_id = identity
        self._session_input = session_input

    def supported_input_modes(self) -> tuple[str, ...]:
        return ("webrtc",)

    def supported_output_modes(self) -> tuple[str, ...]:
        return ("webrtc",)

    def default_input_mapping(self) -> InputMapping | None:
        return None

    def validate_config(self, config: Any) -> None:
        if config.model_id != self.model_id:
            raise ValueError(
                f"Expected WebRTC model_id={self.model_id!r}, got {config.model_id!r}."
            )

    def create_runtime(self, config: Any) -> Any:
        self.validate_config(config)
        return self._runtime

    def prepare_scenario(self, spec: Any) -> PreparedScenario:
        del spec
        return PreparedScenario(initial_inputs=self._initial_inputs())

    def create_model_input_provider(
        self,
        spec: Any,
        scenario: PreparedScenario,
    ) -> _LegacyWebRTCModelInputProvider:
        del spec, scenario
        return _LegacyWebRTCModelInputProvider(
            runtime=self._runtime,
            session_input=self._session_input,
        )

    def _initial_inputs(self) -> InferenceInput:
        if self._session_input is None:
            return InferenceInput()
        return InferenceInput(
            global_conditioning={_SESSION_INPUT_KEY: self._session_input}
        )


class _ManagedWebRTCSessionEdgeFactory:
    """Build shared realtime edges for one negotiated peer connection."""

    def __init__(
        self,
        *,
        manager: "BaseWebRTCSessionManager[Any, Any]",
        managed_session: "ManagedWebRTCSession",
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        self._manager = manager
        self._managed_session = managed_session
        self._loop = loop

    def create_session_edges(
        self,
        *,
        context: RunContext,
        spec: Any,
        scenario: PreparedScenario,
        provider: ModelInputProvider,
        adapter: Any,
    ) -> SessionEdges:
        del spec, scenario, provider, adapter
        input_source = self._managed_session.input_source
        transport = self._managed_session.transport
        if input_source is None or transport is None:
            raise RuntimeError("Managed WebRTC session is missing shared edges.")
        bridge = ThreadSafeWebRTCOutputBridge(
            loop=self._loop,
            video_encoder=self._managed_session.video_encoder,
            video_track=self._managed_session.video_track,
            close_track=not self._manager._keep_connection_after_completed,
            on_chunk_delivery=self._on_chunk_delivery,
            on_error=self._on_delivery_error,
        )
        return SessionEdges(
            input_source=input_source,
            output_sink=WebRTCOutputSink(bridge=bridge),
            cleanup_tasks=context.cleanup_tasks,
            metrics=InMemorySessionMetricsRecorder(),
            error_policy=WebRTCErrorPolicy(),
            transport=_GenerationTransportView(transport),
            clock=ResamplerRealtimeClock(
                resampler=self._managed_session.resampler,
                now_fn=self._loop.time,
                sleep_fn=asyncio.sleep,
            ),
            activation=WebRTCActivationPolicy(
                input_source=input_source,
                transport=transport,
                activate_without_input=self._manager._activate_without_input,
            ),
        )

    def _on_chunk_delivery(self, chunk: WebRTCChunkDelivery) -> None:
        self._manager._handle_shared_chunk_delivery(
            managed_session=self._managed_session,
            chunk=chunk,
        )

    def _on_delivery_error(self, exc: BaseException) -> None:
        self._manager._handle_shared_delivery_error(
            managed_session=self._managed_session,
            exc=exc,
        )
        if self._manager.fatal_generation_errors:
            self._loop.call_soon_threadsafe(
                lambda: asyncio.create_task(self._manager.close_active_session())
            )


@dataclass(frozen=True, slots=True)
class _GenerationTransportView:
    """Expose peer liveness without transferring peer-close ownership."""

    peer_transport: WebRTCTransportService
    """Peer-scoped transport owned by the managed WebRTC session."""

    def is_active(self) -> bool:
        """Return whether the underlying peer transport remains active."""
        return self.peer_transport.is_active()

    def close(self) -> None:
        """Release the generation view without closing the reusable peer."""


async def _wait_for_next_generation_or_disconnect(
    *,
    input_source: WebRTCInputSource,
    transport: WebRTCTransportService,
) -> bool:
    """Wait for another activation while draining both signal waiters.

    Returns:
        ``True`` when activation wins while the peer remains connected.
    """
    activated = asyncio.create_task(input_source.activation_signal.wait())
    closed = asyncio.create_task(transport.closed_signal.wait())
    waiters = (activated, closed)
    try:
        done, _ = await asyncio.wait(
            waiters,
            return_when=asyncio.FIRST_COMPLETED,
        )
        for waiter in done:
            waiter.result()
        return closed not in done and transport.is_active()
    finally:
        for waiter in waiters:
            if not waiter.done():
                waiter.cancel()
        await asyncio.gather(*waiters, return_exceptions=True)


@dataclass(slots=True)
class ManagedWebRTCSession:
    """Per-session state for the single active WebRTC peer connection."""

    runtime: Any
    video_track: BufferedVideoTrack | NVENCVideoTrack
    video_encoder: VideoEncoder
    peer_connection: Any
    resampler: RealtimeEventResampler
    legacy_segment_resampler: Any | None = None
    control_channel: Any | None = None
    generation_task: asyncio.Task[Any] | None = None
    first_action_received: asyncio.Event = field(default_factory=asyncio.Event)
    input_source: WebRTCInputSource | None = None
    transport: WebRTCTransportService | None = None
    reservation: Any | None = None
    pending_action_arrivals: deque[float] = field(default_factory=deque)
    inference_session: Any | None = None
    """Active ``InferenceSession``; ``None`` means call ``runtime.generate_chunk``."""
    session_steps_completed: int = 0
    session_input_state_advanced: bool = False
    user_events: deque[UserInputEvent] = field(default_factory=deque)
    """Raw user events awaiting canonicalization, oldest first."""
    coalesced_release_events: dict[str, UserInputEvent] = field(default_factory=dict)
    """Overflow key releases, coalesced by normalized key."""
    last_client_message_at: float = 0.0
    liveness_task: asyncio.Task[Any] | None = None
    closed: bool = False

    async def close(self) -> None:
        if self.closed:
            return
        self.closed = True

        current_task = asyncio.current_task()
        if (
            self.liveness_task is not None
            and self.liveness_task is not current_task
            and not self.liveness_task.done()
        ):
            self.liveness_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self.liveness_task
        self.liveness_task = None

        if (
            self.generation_task is not None
            and self.generation_task is not current_task
            and not self.generation_task.done()
        ):
            self.generation_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self.generation_task
        if self.generation_task is None or self.generation_task.done():
            reservation = self.reservation
            self.reservation = None
            if reservation is not None:
                reservation.release()
        self.generation_task = None

        await self.video_track.close()
        await self.peer_connection.close()


class BaseWebRTCSessionManager(Generic[_RuntimeT, _RuntimeConfigT]):
    """Owns one active WebRTC session and forwards actions into a model runtime."""

    _perf_log_interval_chunks: int = _DEFAULT_PERF_LOG_INTERVAL_CHUNKS

    def __init__(
        self,
        *,
        runtime: _RuntimeT,
        runtime_config: _RuntimeConfigT,
        fps: int,
        identity: str,
        busy_message: str = "A WebRTC session is already active.",
        warmup_label: str = "WebRTC",
        supported_control_keys: AbstractSet[str] | None = None,
        fatal_generation_errors: bool = False,
        client_liveness_timeout_s: float = DEFAULT_CLIENT_LIVENESS_TIMEOUT_S,
        activate_without_input: bool = False,
        model_warmup_plan: ModelWarmupPlan | None = None,
        shared_host: RuntimeHost | None = None,
        shared_adapter: Any | None = None,
        shared_spec: DemoSpec | None = None,
        shared_spec_factory: Callable[[Any], DemoSpec] | None = None,
        shared_scenario: PreparedScenario | None = None,
        shared_pipeline_factory: Callable[[], StepPipeline] | None = None,
        legacy_segment_resampler_factory: Callable[..., Any] | None = None,
        keep_connection_after_completed: bool = False,
    ) -> None:
        if client_liveness_timeout_s <= 0:
            raise ValueError("client_liveness_timeout_s must be > 0")
        if model_warmup_plan is not None and not isinstance(
            model_warmup_plan, ModelWarmupPlan
        ):
            raise TypeError("model_warmup_plan must be a ModelWarmupPlan.")
        self.runtime_config = runtime_config
        self.fps = fps
        self.identity = identity
        self.busy_message = busy_message
        self.warmup_label = warmup_label
        self.supported_control_keys = (
            None
            if supported_control_keys is None
            else frozenset(supported_control_keys)
        )
        self.fatal_generation_errors = fatal_generation_errors
        self.client_liveness_timeout_s = client_liveness_timeout_s
        self._activate_without_input = activate_without_input
        self._runtime = runtime
        self._warmup_complete = False
        self._active_session: ManagedWebRTCSession | None = None
        self._warmup_lock = asyncio.Lock()
        self._session_lock = asyncio.Lock()
        self._pending_session_input: Any = None
        self._shared_runtime_adapter: _LegacyWebRTCRuntimeAdapter | None = None
        self._model_warmup_plan = (
            ModelWarmupPlan() if model_warmup_plan is None else model_warmup_plan
        )
        self._shared_host: RuntimeHost | None = shared_host
        self._owns_shared_host = shared_host is not None
        self._shared_context: RunContext | None = None
        self._shared_adapter = shared_adapter
        self._shared_spec = shared_spec
        self._shared_spec_factory = shared_spec_factory
        self._shared_scenario = shared_scenario
        self._shared_pipeline_factory = shared_pipeline_factory
        self._keep_connection_after_completed = keep_connection_after_completed
        self._shared_video_encoder: VideoEncoder | None = None
        self._legacy_segment_resampler_factory = legacy_segment_resampler_factory
        self._lifecycle = WebRTCManagerLifecycle(
            busy_message=busy_message,
            client_liveness_timeout_s=client_liveness_timeout_s,
            health_check=lambda: self._shared_host is None
            or self._shared_host.is_healthy,
        )

    @property
    def _runtime_ready(self) -> bool:
        """Compatibility alias for readiness now owned by the lifecycle service."""
        return self._lifecycle.runtime_ready

    @_runtime_ready.setter
    def _runtime_ready(self, value: bool) -> None:
        if value:
            self._lifecycle.mark_ready()
        else:
            self._lifecycle.mark_unready()

    @property
    def pending_session_input(self) -> Any:
        """Input that will be applied to the next successfully negotiated session."""
        return self._pending_session_input

    @property
    def runtime(self) -> _RuntimeT:
        """Model runtime driven by this transport manager."""
        return self._runtime

    def browser_ui_config(self) -> dict[str, object]:
        """Return accepted control keys for the generic browser UI."""
        accepted_keys = self._effective_supported_control_keys() or ()
        return {"accepted_keys": sorted(accepted_keys)}

    def set_pending_session_input(self, session_input: Any) -> None:
        """Store validated model input for the next session."""
        if self.has_active_session():
            raise SessionBusyError(self.busy_message)
        self._pending_session_input = session_input

    def _make_resampler(self, *, start_v: float) -> RealtimeEventResampler:
        return self._make_resampler_at_fps(start_v=start_v, fps=self.fps)

    def _make_resampler_at_fps(
        self, *, start_v: float, fps: float
    ) -> RealtimeEventResampler:
        return RealtimeEventResampler(fps=fps, start_v=start_v)

    def _make_legacy_segment_resampler_at_fps(
        self, *, start_v: float, fps: float
    ) -> Any:
        factory = self._legacy_segment_resampler_factory
        if factory is None:
            raise RuntimeError(
                "Legacy WebRTC segment runtimes require "
                "legacy_segment_resampler_factory."
            )
        supported_control_keys = self._effective_supported_control_keys()
        kwargs: dict[str, object] = {
            "fps": fps,
            "start_v": start_v,
        }
        if supported_control_keys is not None:
            kwargs["supported_keys"] = supported_control_keys
        return factory(**kwargs)

    def _needs_legacy_segment_metadata(self) -> bool:
        return self._shared_adapter is None and not _runtime_drives_inference_session(
            self._runtime
        )

    def _effective_supported_control_keys(self) -> frozenset[str] | None:
        supported_control_keys = self.supported_control_keys
        if supported_control_keys is not None:
            return frozenset(supported_control_keys)
        legacy_supported_keys = getattr(self, "_resampler_supported_keys", None)
        if legacy_supported_keys is not None:
            return frozenset(legacy_supported_keys)
        return self._converter_supported_control_keys()

    def _feedable_converter_schemas(self) -> tuple[DeviceConverterSchema, ...]:
        scenario = self._shared_scenario
        if scenario is None:
            return ()
        schemas: list[DeviceConverterSchema] = []
        for converter in scenario.canonicalizer.converters_for(scenario.source_schema):
            schema = converter.schema
            if isinstance(schema, DeviceConverterSchema):
                schemas.append(schema)
        return tuple(schemas)

    def _converter_supported_control_keys(self) -> frozenset[str] | None:
        accepted_keys = frozenset(
            key
            for schema in self._feedable_converter_schemas()
            if schema.accepted_keys is not None
            for key in schema.accepted_keys
        )
        return accepted_keys or None

    @staticmethod
    def _positive_int_runtime_value(value: Any, *, label: str) -> int:
        try:
            parsed = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{label} must be an integer.") from exc
        if parsed <= 0:
            raise ValueError(f"{label} must be > 0.")
        return parsed

    @staticmethod
    def _positive_float_runtime_value(value: Any, *, label: str) -> float:
        try:
            parsed = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{label} must be numeric.") from exc
        if parsed <= 0.0:
            raise ValueError(f"{label} must be > 0.")
        return parsed

    def _runtime_input_fps(self, runtime: Any) -> float:
        peek_input_fps = getattr(runtime, "peek_input_fps", None)
        if not callable(peek_input_fps):
            return float(self.fps)
        return self._positive_float_runtime_value(
            peek_input_fps(),
            label="peek_input_fps",
        )

    def _runtime_next_step_request(self, runtime: Any) -> tuple[StepRequest, int]:
        request = runtime.next_step_request()
        if not isinstance(request, StepRequest):
            raise TypeError(
                "next_step_request must return StepRequest, "
                f"got {type(request).__name__}."
            )
        input_num_frames = self._positive_int_runtime_value(
            request.metadata.get("input_frame_count"),
            label="StepRequest.metadata['input_frame_count']",
        )
        return request, input_num_frames

    def _runtime_steady_output_num_frames(self, runtime: Any) -> int:
        peek_output_frames = getattr(runtime, "peek_steady_output_num_frames", None)
        if callable(peek_output_frames):
            return self._positive_int_runtime_value(
                peek_output_frames(),
                label="peek_steady_output_num_frames",
            )
        pipeline = getattr(runtime, "pipeline", None)
        get_num_frames = getattr(pipeline, "get_num_frames", None)
        if callable(get_num_frames):
            return self._positive_int_runtime_value(
                get_num_frames(1),
                label="pipeline.get_num_frames(1)",
            )
        return self._positive_int_runtime_value(
            1,
            label="fallback steady output frame count",
        )

    def _resolve_video_encoder(self) -> VideoEncoder:
        """Return the encoder to use for the next session.

        Default: read ``runtime.video_encoder`` if the runtime provides
        one through the shared thread-affine runtime;
        otherwise construct a session-scope :class:`DefaultRTCEncoder`.
        Runtimes that do not participate in encoder selection
        transparently get the software path without having to opt in.
        """
        encoder = getattr(self._runtime, "video_encoder", None)
        if encoder is None:
            encoder = self._shared_video_encoder
        if encoder is None:
            encoder = DefaultRTCEncoder(fps=self.fps)
        return encoder

    def _shared_run_context(self, loop: asyncio.AbstractEventLoop) -> RunContext:
        if self._shared_context is not None:
            return self._shared_context
        host = self._shared_host
        if host is None:
            runtime_adapter = _LegacyWebRTCRuntimeAdapter(
                runtime=self._runtime,
                loop=loop,
            )
            host = RuntimeHost(runtime_adapter)
            self._shared_runtime_adapter = runtime_adapter
            self._shared_host = host
        self._shared_context = RunContext(
            host=host,
            run_metrics=InMemorySessionMetricsRecorder(),
            admission=self._lifecycle.admission,
            model_warmup_plan=self._model_warmup_plan,
        )
        return self._shared_context

    def _shared_demo_spec(self) -> DemoSpec:
        return DemoSpec(
            model_id=self.identity,
            input_mode="webrtc",
            output=WebRTCOutputSpec(
                fps=self.fps,
                video_width=self.runtime_config.video_width,
                video_height=self.runtime_config.video_height,
                warmup_chunks=self.runtime_config.warmup_chunks,
                warmup_timeout_s=self.runtime_config.warmup_timeout_s,
                client_liveness_timeout_s=self.client_liveness_timeout_s,
            ),
        )

    async def _reset_runtime_for_session(
        self,
        *,
        context: RunContext,
        session_input: Any,
    ) -> None:
        reset = getattr(context.host.runtime, "reset_for_new_session", None)
        if not callable(reset):
            if self._shared_adapter is not None:
                return
            raise RuntimeError("WebRTC runtime adapter cannot reset sessions.")
        await context.host.call_async(reset, session_input)

    def _prefer_h264_video_codec(self, *, transceiver: Any) -> None:
        """Constrain the transceiver's codec preferences to H.264 variants.

        Required when the selected encoder emits pre-encoded H.264 packets
        (``av.Packet`` route through ``H264Encoder.pack()``): if the SDP
        negotiates VP8/VP9 instead, aiortc will pack the H.264 bitstream
        under the wrong codec header and the receiver will fail to decode.

        If the local aiortc build does not advertise H.264, no preference
        is set; the SDP-time fallback in ``_enforce_h264_or_fallback``
        will then swap the encoder to :class:`DefaultRTCEncoder`.
        """
        caps = RTCRtpSender.getCapabilities("video")
        h264_codecs = [c for c in caps.codecs if c.mimeType.lower() == "video/h264"]
        if not h264_codecs:
            return
        transceiver.setCodecPreferences(h264_codecs)

    async def _enforce_h264_or_fallback(
        self,
        *,
        transceiver: Any,
        managed_session: ManagedWebRTCSession,
        num_frames: int,
    ) -> None:
        """Verify H.264 was negotiated; swap to the software encoder if not.

        aiortc exposes the negotiated codec set on
        ``RTCRtpTransceiver._codecs`` after ``setLocalDescription``. We
        read it via that attribute (aiortc-internal, but stable in the
        pinned version) and, if H.264 did not land, close the hardware
        encoder and install a :class:`DefaultRTCEncoder` with a
        :class:`BufferedVideoTrack` on the same sender before the first
        RTP packet flies. ``replaceTrack`` does not renegotiate; aiortc's
        RTP loop will encode raw ``av.VideoFrame`` output with whatever
        codec (VP8/VP9/H.264) actually landed in the SDP.
        """
        negotiated = getattr(transceiver, "_codecs", None) or []
        if negotiated and negotiated[0].mimeType.lower() == "video/h264":
            logger.info(
                "Video codec negotiated: {} (hardware encoder path active).",
                negotiated[0].mimeType,
            )
            return

        chosen = negotiated[0].mimeType if negotiated else "<none>"
        logger.warning(
            "H.264 preferred by hardware encoder but SDP negotiation "
            "landed on {!r}; swapping to the software encoder before "
            "streaming begins.",
            chosen,
        )
        # Close the pre-encoded track before overwriting the reference
        # so its readyState transitions to "ended" and its packet queue
        # is drained. Otherwise ``ManagedWebRTCSession.close()`` would
        # only ever see the fallback track and never clean this one up.
        # The hardware encoder itself is owned by the runtime (created
        # once during runtime initialization and reused across
        # sessions), so it is intentionally NOT closed here — subsequent
        # sessions read the same object via ``runtime.video_encoder``
        # and expect it live. Runtime shutdown releases it.
        await managed_session.video_track.close()

        fallback_encoder = DefaultRTCEncoder(fps=self.fps)
        fallback_track = fallback_encoder.create_track(maxsize=num_frames)
        transceiver.sender.replaceTrack(fallback_track)
        managed_session.video_encoder = fallback_encoder
        managed_session.video_track = fallback_track

    async def _handle_event_message(
        self,
        *,
        managed_session: ManagedWebRTCSession,
        payload: dict[str, Any],
    ) -> bool:
        """Dispatch an optional model event message to runtimes that support it."""
        channel = managed_session.control_channel
        event_id = str(payload.get("event_id", payload.get("id", ""))).strip()
        state = str(payload.get("state", "trigger")).strip().lower() or "trigger"
        clear_states = {"clear", "release", "off", "none"}
        if not event_id and state not in clear_states:
            if channel is not None:
                self._send_json(
                    channel,
                    make_error_payload(
                        (
                            "Event payload must include non-empty 'event_id' "
                            "unless state clears the active event."
                        ),
                    ),
                )
            return False

        if managed_session.inference_session is not None:
            # On the session branch a text event is just another user event:
            # the mapping turns it into a session-global conditioning update
            # applied by the next step, so there is no separate runtime call.
            clears = state in clear_states
            try:
                event_payload = self._validate_user_event_payload(
                    managed_session=managed_session,
                    event_type="text_event",
                    payload={
                        "event_id": None if clears else event_id,
                        "state": state,
                    },
                )
                self._record_user_event(
                    managed_session=managed_session,
                    timestamp_s=asyncio.get_running_loop().time(),
                    event_type="text_event",
                    payload=event_payload,
                )
            except Exception as exc:
                if channel is not None:
                    self._send_json(channel, make_error_payload(str(exc)))
                return False
            if channel is not None:
                active_event_id = event_payload.get("event_id")
                ack_event_id = None if active_event_id is None else str(active_event_id)
                self._send_json(
                    channel,
                    make_event_ack_payload(
                        event_id=ack_event_id,
                        state=str(event_payload.get("state", state)),
                        result={"active_event_id": ack_event_id},
                    ),
                )
            return True

        trigger_event = getattr(managed_session.runtime, "trigger_event", None)
        if not callable(trigger_event):
            if channel is not None:
                self._send_json(
                    channel,
                    make_error_payload("This runtime does not support event messages."),
                )
            return False

        try:
            result = trigger_event(event_id=event_id, state=state)
            if inspect.isawaitable(result):
                result = await result
        except Exception as exc:
            if channel is not None:
                self._send_json(channel, make_error_payload(str(exc)))
            return False

        if channel is not None:
            self._send_json(
                channel,
                make_event_ack_payload(
                    event_id=event_id or None,
                    state=state,
                    result=result if isinstance(result, dict) else None,
                ),
            )
        return True

    @staticmethod
    def _drives_inference_session(runtime: Any) -> bool:
        """Return whether ``runtime`` should be driven through ``InferenceSession``."""
        return _runtime_drives_inference_session(runtime)

    def _record_user_event(
        self,
        *,
        managed_session: ManagedWebRTCSession,
        timestamp_s: float,
        event_type: str,
        payload: dict[str, Any],
    ) -> None:
        """Buffer one raw user event for the session branch.

        Timestamps use the same monotonic clock as the realtime resampler so
        chunk ``TimeWindow`` filtering and raw data-channel events agree.
        """
        if event_type in _KEY_USER_EVENT_TYPES and not self._supports_key_payload(
            payload
        ):
            return
        if len(managed_session.user_events) >= _MAX_SESSION_USER_EVENTS:
            if event_type in _RELEASE_USER_EVENT_TYPES:
                made_room = self._make_room_for_release_event(
                    managed_session=managed_session,
                    event_type=event_type,
                    payload=payload,
                )
                if not made_room:
                    self._record_coalesced_release_event(
                        managed_session=managed_session,
                        timestamp_s=timestamp_s,
                        event_type=event_type,
                        payload=payload,
                    )
                    return
            else:
                raise RuntimeError(
                    "Too many queued WebRTC user events; wait for inference to catch up."
                )
        managed_session.user_events.append(
            UserInputEvent(
                timestamp_s=timestamp_s,
                event_type=event_type,
                payload=payload,
                source="webrtc",
            )
        )

    def _make_room_for_release_event(
        self,
        *,
        managed_session: ManagedWebRTCSession,
        event_type: str,
        payload: dict[str, Any],
    ) -> bool:
        events = managed_session.user_events
        if not events:
            return False
        if event_type == "key_up":
            released_key = payload.get("key")
            normalized_released_key = (
                normalize_key(released_key) if isinstance(released_key, str) else None
            )
            if normalized_released_key is not None:
                for index, queued_event in enumerate(events):
                    queued_key = queued_event.payload.get("key")
                    if (
                        queued_event.event_type == "key_down"
                        and isinstance(queued_key, str)
                        and normalize_key(queued_key) == normalized_released_key
                    ):
                        del events[index]
                        return True
                for index, queued_event in enumerate(events):
                    queued_key = queued_event.payload.get("key")
                    if (
                        queued_event.event_type == "key_up"
                        and isinstance(queued_key, str)
                        and normalize_key(queued_key) == normalized_released_key
                    ):
                        del events[index]
                        return True
        return False

    def _record_coalesced_release_event(
        self,
        *,
        managed_session: ManagedWebRTCSession,
        timestamp_s: float,
        event_type: str,
        payload: dict[str, Any],
    ) -> None:
        if event_type != "key_up":
            return
        key = payload.get("key")
        if not isinstance(key, str):
            return
        managed_session.coalesced_release_events[normalize_key(key)] = UserInputEvent(
            timestamp_s=timestamp_s,
            event_type=event_type,
            payload=payload,
            source="webrtc",
        )

    def _supported_key_names(self) -> frozenset[str]:
        supported_keys = self._effective_supported_control_keys()
        if supported_keys is None:
            supported_keys = DEFAULT_SUPPORTED_KEYS
        return frozenset(normalize_key(key) for key in supported_keys)

    def _supports_key_payload(self, payload: dict[str, Any]) -> bool:
        key = payload.get("key")
        return (
            isinstance(key, str) and normalize_key(key) in self._supported_key_names()
        )

    @staticmethod
    def _pending_user_events(
        managed_session: ManagedWebRTCSession,
    ) -> tuple[UserInputEvent, ...]:
        return tuple(
            sorted(
                (
                    *managed_session.user_events,
                    *managed_session.coalesced_release_events.values(),
                ),
                key=lambda event: event.timestamp_s,
            )
        )

    def _catch_up_input_clock(
        self,
        *,
        managed_session: ManagedWebRTCSession,
        now: float,
        chunk_duration: float,
    ) -> None:
        """Skip stale input windows without skipping session input state."""
        resampler = managed_session.resampler
        lag = now - (resampler.next_chunk_start_v + chunk_duration)
        if lag <= chunk_duration:
            return
        latest_chunk_start = now - chunk_duration
        if managed_session.inference_session is not None:
            catch_up_start = (
                0.0
                if managed_session.session_steps_completed == 0
                else resampler.next_chunk_start_v
            )
            if latest_chunk_start > catch_up_start:
                self._advance_inference_input_state(
                    managed_session=managed_session,
                    window=TimeWindow(
                        start_s=catch_up_start,
                        end_s=latest_chunk_start,
                    ),
                )
        resampler.next_chunk_start_v = latest_chunk_start

    def _advance_inference_input_state(
        self,
        *,
        managed_session: ManagedWebRTCSession,
        window: TimeWindow,
    ) -> None:
        """Advance session input converters over a skipped raw-event window."""
        if managed_session.inference_session is None or window.end_s <= window.start_s:
            return
        runtime = managed_session.runtime
        runtime.input_canonicalizer.canonicalize(
            UserInputs(events=self._pending_user_events(managed_session)),
            window=window,
            source_schema=runtime.input_source_schema,
        )
        managed_session.session_input_state_advanced = True
        self._prune_consumed_user_events(
            managed_session,
            before_s=window.end_s,
        )

    def _validate_user_event_payload(
        self,
        *,
        managed_session: ManagedWebRTCSession,
        event_type: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        """Return a runtime-validated user-event payload."""
        validate = getattr(managed_session.runtime, "validate_user_event", None)
        if not callable(validate):
            return payload
        result = validate(event_type=event_type, payload=dict(payload))
        if result is None:
            return payload
        if not isinstance(result, dict):
            raise TypeError(
                "validate_user_event must return a payload dict or None, got "
                f"{type(result).__name__}."
            )
        return result

    @staticmethod
    def _prune_consumed_user_events(
        managed_session: ManagedWebRTCSession, *, before_s: float
    ) -> None:
        """Drop events already folded into converter state."""
        events = managed_session.user_events
        while events and events[0].timestamp_s < before_s:
            events.popleft()
        for key, event in tuple(managed_session.coalesced_release_events.items()):
            if event.timestamp_s < before_s:
                del managed_session.coalesced_release_events[key]

    async def _step_inference_session(
        self,
        *,
        managed_session: ManagedWebRTCSession,
        window: TimeWindow,
    ) -> StepResult:
        """Map this chunk's events into model inputs and run one session step."""
        session: Any = managed_session.inference_session
        if session is None:
            raise RuntimeError("Session branch invoked without an inference session.")
        request = session.next_step_request()
        if request is None:
            raise _InferenceSessionExhausted()
        if request.step_index == 0 and not managed_session.session_input_state_advanced:
            window = TimeWindow(start_s=0.0, end_s=window.end_s)
        request = replace(request, user_input_window=window)
        step_inputs = self._build_step_inputs(
            managed_session=managed_session,
            request=request,
            window=window,
        )
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, session.step, step_inputs)
        if not isinstance(result, StepResult):
            raise TypeError(
                "Inference session steps must produce StepResult, got "
                f"{type(result).__name__}."
            )
        self._prune_consumed_user_events(managed_session, before_s=window.start_s)
        managed_session.session_steps_completed += 1
        return result

    def _build_step_inputs(
        self,
        *,
        managed_session: ManagedWebRTCSession,
        request: Any,
        window: TimeWindow,
    ) -> InferenceInput:
        """Canonicalize this chunk's events and map them into model inputs."""
        runtime = managed_session.runtime
        canonical_inputs = runtime.input_canonicalizer.canonicalize(
            UserInputs(events=self._pending_user_events(managed_session)),
            window=window,
            source_schema=runtime.input_source_schema,
        )
        return runtime.input_mapping.map_step_inputs(
            canonical_inputs=canonical_inputs,
            inference_input=InferenceInput(),
            request=request,
        )

    def has_active_session(self) -> bool:
        return self._active_session is not None and not self._active_session.closed

    def is_runtime_ready(self) -> bool:
        return self._lifecycle.runtime_ready

    async def preload_runtime(self) -> None:
        async def initialize_runtime() -> None:
            initialize = getattr(self._runtime, "initialize", None)
            if callable(initialize):
                result = initialize()
                if inspect.isawaitable(result):
                    await result
            elif self._shared_host is not None:
                await asyncio.to_thread(self._shared_host.preload)
            if self._model_warmup_plan.sessions:
                host = self._shared_host
                if host is None:
                    host = self._shared_run_context(asyncio.get_running_loop()).host
                await asyncio.to_thread(host.warmup, self._model_warmup_plan)
            self._initialize_shared_video_encoder()

        await self._lifecycle.ensure_preloaded(initialize_runtime)
        async with self._warmup_lock:
            if not self._warmup_complete:
                await self._run_loopback_warmup_session(
                    num_chunks=(
                        0
                        if self._model_warmup_plan.sessions
                        else self.runtime_config.warmup_chunks
                    )
                )
                self._warmup_complete = True

    def _initialize_shared_video_encoder(self) -> None:
        if self._shared_video_encoder is not None:
            return
        if getattr(self._runtime, "video_encoder", None) is not None:
            return
        encoder_backend = getattr(self.runtime_config, "encoder_backend", None)
        if encoder_backend is None:
            return
        backend = _encoder_backend_from_config(encoder_backend)
        device_spec = str(getattr(self.runtime_config, "device", ""))
        device_type = device_spec.split(":", maxsplit=1)[0]
        if device_type != "cuda" and backend == "auto":
            backend = "default"
        if device_type != "cuda" and backend == "nvenc":
            raise RuntimeError("encoder_backend='nvenc' requires a CUDA device.")
        self._shared_video_encoder = select_encoder(
            backend=backend,
            width=self.runtime_config.video_width,
            height=self.runtime_config.video_height,
            fps=self.fps,
            bitrate=int(getattr(self.runtime_config, "encoder_bitrate_bps", 6_000_000)),
            gpu_id=_gpu_id_from_device_spec(device_spec),
            gop=int(getattr(self.runtime_config, "encoder_gop", self.fps)),
        )

    async def create_answer(self, *, offer_sdp: str, offer_type: str) -> dict[str, str]:
        if not self._lifecycle.runtime_ready or not self._warmup_complete:
            await self.preload_runtime()

        async with self._session_lock:
            if self._active_session is not None and not self._active_session.closed:
                raise SessionBusyError(self.busy_message)

            session_input = self._pending_session_input
            answer = await self._create_answer_with_runtime_ready_locked(
                offer_sdp=offer_sdp,
                offer_type=offer_type,
                session_input=session_input,
            )
            self._pending_session_input = None
            return answer

    async def _create_answer_with_runtime_ready_locked(
        self,
        *,
        offer_sdp: str,
        offer_type: str,
        session_input: Any = None,
        rtc_configuration: RTCConfiguration | None = None,
        enable_liveness_watchdog: bool = True,
    ) -> dict[str, str]:
        if self._active_session is not None and not self._active_session.closed:
            raise SessionBusyError(self.busy_message)
        if not self._lifecycle.runtime_ready:
            raise RuntimeError("Runtime is not initialized.")

        loop = asyncio.get_running_loop()
        context = self._shared_run_context(loop)
        reservation = context.admission.try_reserve()
        if reservation is None:
            raise SessionBusyError(self.busy_message)
        try:
            await self._reset_runtime_for_session(
                context=context,
                session_input=session_input,
            )
        except Exception:
            reservation.release()
            raise

        try:
            peer_connection = RTCPeerConnection(rtc_configuration)
            # Bounded queue sized to one *steady-state* chunk so the producer
            # is throttled to the consumer's drain rate. AR step 0 emits fewer
            # frames than steady state; sizing to it would force a per-chunk
            # stall, so we size to the steady-state count.
            num_frames = self._runtime_steady_output_num_frames(self._runtime)
            video_encoder = self._resolve_video_encoder()
            video_track = video_encoder.create_track(maxsize=num_frames)
            # Use ``addTransceiver`` (not ``addTrack``) so we can constrain the
            # SDP m-line's codec list via ``setCodecPreferences`` when the
            # encoder emits pre-encoded H.264 packets.
            video_transceiver = peer_connection.addTransceiver(
                video_track,
                direction="sendonly",
            )
            if video_encoder.prefers_codec == "h264":
                self._prefer_h264_video_codec(transceiver=video_transceiver)
            # Start the resampler's virtual clock at 0; the real anchor is set
            # in the ``on_datachannel`` handler so chunk 0's window starts when
            # input can actually arrive.
            resampler = self._make_resampler_at_fps(
                start_v=0.0,
                fps=self._runtime_input_fps(self._runtime),
            )
            legacy_segment_resampler = None
            if self._needs_legacy_segment_metadata():
                legacy_segment_resampler = self._make_legacy_segment_resampler_at_fps(
                    start_v=0.0,
                    fps=self._runtime_input_fps(self._runtime),
                )
            input_source = WebRTCInputSource(
                resampler=resampler,
                legacy_segment_resampler=legacy_segment_resampler,
                legacy_segments_metadata_key=(
                    _LEGACY_SPARSE_KEY_SEGMENTS_METADATA_KEY
                    if legacy_segment_resampler is not None
                    else None
                ),
            )
            transport = WebRTCTransportService(loop=loop)
            managed_session = ManagedWebRTCSession(
                runtime=self._runtime,
                video_track=video_track,
                video_encoder=video_encoder,
                peer_connection=peer_connection,
                resampler=resampler,
                legacy_segment_resampler=legacy_segment_resampler,
                input_source=input_source,
                transport=transport,
                reservation=reservation,
                last_client_message_at=loop.time(),
            )
        except Exception:
            reservation.release()
            raise
        self._active_session = managed_session
        if enable_liveness_watchdog:
            managed_session.liveness_task = asyncio.create_task(
                self._client_liveness_watchdog(managed_session=managed_session)
            )

        @peer_connection.on("datachannel")
        def on_datachannel(channel: Any) -> None:
            managed_session.control_channel = channel
            # Re-anchor the resampler at channel open. The real
            # virtual-clock anchor happens in ``WebRTCActivationPolicy`` once
            # the first browser event activates the shared realtime driver.
            channel_open_v = asyncio.get_running_loop().time()
            if managed_session.input_source is not None:
                managed_session.input_source.reset(start_v=channel_open_v)
            else:
                managed_session.resampler.reset(start_v=channel_open_v)

            @channel.on("message")
            def on_message(message: Any) -> None:
                asyncio.create_task(
                    self._handle_datachannel_message(
                        managed_session=managed_session,
                        raw_message=message,
                    )
                )

            # Spawn the shared realtime session once the channel is wired up so
            # ``chunk_done`` notifications have a channel to land on.
            managed_session.generation_task = asyncio.create_task(
                self._run_realtime_driver_session(
                    managed_session=managed_session,
                    context=context,
                    session_input=session_input,
                )
            )

            @channel.on("close")
            def on_close() -> None:
                logger.info("Control data channel closed; closing active session.")
                if managed_session.transport is not None:
                    managed_session.transport.disconnect("data channel closed")
                asyncio.create_task(self.close_active_session())

        @peer_connection.on("connectionstatechange")
        async def on_connectionstatechange() -> None:
            if peer_connection.connectionState in {
                "failed",
                "disconnected",
                "closed",
            }:
                await self.close_active_session()

        @peer_connection.on("iceconnectionstatechange")
        def on_iceconnectionstatechange() -> None:
            logger.info(
                "Peer ICE connection state changed: {}",
                peer_connection.iceConnectionState,
            )

        @peer_connection.on("icegatheringstatechange")
        def on_icegatheringstatechange() -> None:
            logger.debug(
                "Peer ICE gathering state changed: {}",
                peer_connection.iceGatheringState,
            )

        try:
            offer = RTCSessionDescription(sdp=offer_sdp, type=offer_type)
            logger.info(
                "Received WebRTC offer with {}.",
                _summarize_sdp_candidates(offer_sdp),
            )
            await peer_connection.setRemoteDescription(offer)
            answer = await peer_connection.createAnswer()
            await peer_connection.setLocalDescription(answer)
            await wait_for_ice_gathering_complete(peer_connection)
            if video_encoder.prefers_codec == "h264":
                await self._enforce_h264_or_fallback(
                    transceiver=video_transceiver,
                    managed_session=managed_session,
                    num_frames=num_frames,
                )
            local_description = peer_connection.localDescription
            if local_description is None:
                raise RuntimeError("Peer connection did not produce local description.")
            logger.info(
                "Created WebRTC answer with {}.",
                _summarize_sdp_candidates(local_description.sdp),
            )
            return {"sdp": local_description.sdp, "type": local_description.type}
        except Exception:
            logger.exception("WebRTC negotiation failed while creating an answer.")
            await managed_session.close()
            self._active_session = None
            raise

    async def _run_loopback_warmup_session(self, *, num_chunks: int) -> None:
        if not self._lifecycle.runtime_ready:
            raise RuntimeError("Runtime is not initialized.")
        await run_loopback_warmup_session(
            num_chunks=num_chunks,
            warmup_timeout_s=self.runtime_config.warmup_timeout_s,
            create_answer=self._create_loopback_warmup_answer,
            close_active_session=self.close_active_session,
            label=self.warmup_label,
            logger=logger,
        )

    async def _create_loopback_warmup_answer(
        self, *, offer_sdp: str, offer_type: str
    ) -> dict[str, str]:
        async with self._session_lock:
            return await self._create_answer_with_runtime_ready_locked(
                offer_sdp=offer_sdp,
                offer_type=offer_type,
                rtc_configuration=RTCConfiguration(iceServers=[]),
                enable_liveness_watchdog=False,
            )

    async def close_active_session(self) -> None:
        async with self._session_lock:
            if self._active_session is None:
                return
            active_session = self._active_session
            self._active_session = None
            await active_session.close()

    async def _client_liveness_watchdog(
        self, *, managed_session: ManagedWebRTCSession
    ) -> None:
        await self._lifecycle.watch_client_liveness(
            is_closed=lambda: managed_session.closed,
            last_message_at=lambda: managed_session.last_client_message_at,
            close_session=self.close_active_session,
            check_interval_s=_CLIENT_LIVENESS_CHECK_INTERVAL_S,
        )

    async def shutdown(self) -> None:
        await self.close_active_session()
        if self._shared_context is not None:
            await self._shared_context.close_async()
        if self._shared_host is not None:
            await asyncio.to_thread(self._shared_host.close)
        self._shared_context = None
        self._shared_host = None
        self._shared_runtime_adapter = None
        if self._shared_video_encoder is not None:
            self._shared_video_encoder.close()
            self._shared_video_encoder = None
        if not self._owns_shared_host:
            close = getattr(self._runtime, "close", None)
            if callable(close):
                result = close()
                if inspect.isawaitable(result):
                    await result
        self._lifecycle.mark_unready()
        self._warmup_complete = False

    def wait_for_termination(self) -> None:
        wait = getattr(self._runtime, "wait_for_termination", None)
        if callable(wait):
            wait()
            return
        if self._shared_host is not None:
            self._shared_host.run_worker_loop()

    def send_exit_signal(self) -> None:
        send = getattr(self._runtime, "send_exit_signal", None)
        if callable(send):
            send()

    async def _handle_datachannel_message(
        self,
        *,
        managed_session: ManagedWebRTCSession,
        raw_message: Any,
    ) -> None:
        channel = managed_session.control_channel
        if channel is None or managed_session.closed:
            return
        managed_session.last_client_message_at = asyncio.get_running_loop().time()
        if managed_session.transport is not None:
            managed_session.transport.mark_client_message(
                managed_session.last_client_message_at
            )

        if not isinstance(raw_message, str):
            self._send_json(channel, make_error_payload("Expected text payload."))
            return

        try:
            payload = json.loads(raw_message)
        except json.JSONDecodeError:
            self._send_json(channel, make_error_payload("Invalid JSON payload."))
            return

        if not isinstance(payload, dict):
            self._send_json(
                channel, make_error_payload("Payload must be a JSON object.")
            )
            return
        if self._is_filtered_key_action(payload):
            return
        if managed_session.input_source is not None:
            await self._handle_shared_datachannel_payload(
                managed_session=managed_session,
                payload=payload,
            )
            return
        message_type = str(payload.get("type", "")).strip().lower()
        if message_type == MESSAGE_TYPE_HEARTBEAT:
            return
        if message_type == MESSAGE_TYPE_DISCONNECT:
            logger.info("Client requested disconnect; closing active session.")
            await self.close_active_session()
            return
        if message_type == MESSAGE_TYPE_EVENT:
            handled = await self._handle_event_message(
                managed_session=managed_session,
                payload=payload,
            )
            if handled:
                # Text events intentionally count as first interaction: a client may
                # want the model to generate an idle-camera chunk with updated text.
                managed_session.first_action_received.set()
            return
        if message_type != MESSAGE_TYPE_ACTION:
            self._send_json(
                channel,
                make_error_payload(
                    "Unsupported message type, expected "
                    "'action', 'event', 'heartbeat', or 'disconnect'.",
                ),
            )
            return

        action_payload = payload.get("action", payload)
        if not isinstance(action_payload, dict):
            self._send_json(channel, make_error_payload("'action' must be an object."))
            return

        event = str(action_payload.get("event", "")).strip().lower()

        if event == "step":
            arrival_t = asyncio.get_running_loop().time()
            managed_session.pending_action_arrivals.append(arrival_t)
            managed_session.first_action_received.set()
            return
        if event not in ("keydown", "keyup"):
            self._send_json(
                channel,
                make_error_payload(
                    f"Unsupported event={event!r}; expected 'keydown' or 'keyup'.",
                ),
            )
            return
        key = str(action_payload.get("key", "")).strip()
        if not key:
            self._send_json(
                channel,
                make_error_payload("Action payload must include non-empty 'key'."),
            )
            return

        # Stamp arrival on the same monotonic clock that seeds the realtime
        # window clock so user-input windows can be compared directly.
        arrival_t = asyncio.get_running_loop().time()
        if managed_session.inference_session is not None:
            try:
                self._record_user_event(
                    managed_session=managed_session,
                    timestamp_s=arrival_t,
                    event_type="key_down" if event == "keydown" else "key_up",
                    payload={"key": key},
                )
            except Exception as exc:
                self._send_json(channel, make_error_payload(str(exc)))
                if event != "keyup":
                    return
        legacy_resampler = managed_session.legacy_segment_resampler
        if legacy_resampler is not None:
            legacy_resampler.on_edge(arrival_t=arrival_t, event=event, key=key)
        managed_session.pending_action_arrivals.append(arrival_t)
        # Releases the generation worker, which blocks on this until the
        # user actually interacts. Idempotent once already set.
        managed_session.first_action_received.set()

    def _is_filtered_key_action(self, payload: dict[str, Any]) -> bool:
        if str(payload.get("type", "")).strip().lower() != MESSAGE_TYPE_ACTION:
            return False
        action_payload = payload.get("action", payload)
        if not isinstance(action_payload, dict):
            return False
        event = str(action_payload.get("event", "")).strip().lower()
        if event not in {"keydown", "keyup"}:
            return False
        key = action_payload.get("key")
        return (
            isinstance(key, str)
            and bool(key.strip())
            and not self._supports_key_payload({"key": key})
        )

    async def _handle_shared_datachannel_payload(
        self,
        *,
        managed_session: ManagedWebRTCSession,
        payload: dict[str, Any],
    ) -> None:
        channel = managed_session.control_channel
        input_source = managed_session.input_source
        if channel is None or input_source is None:
            return
        message_type = str(payload.get("type", "")).strip().lower()
        if message_type == MESSAGE_TYPE_HEARTBEAT:
            return
        if message_type == MESSAGE_TYPE_DISCONNECT:
            logger.info("Client requested disconnect; closing active session.")
            if managed_session.transport is not None:
                managed_session.transport.disconnect("client disconnected")
            await self.close_active_session()
            return
        if message_type == MESSAGE_TYPE_EVENT:
            handled = self._record_shared_event_payload(
                managed_session=managed_session,
                payload=payload,
            )
            if handled:
                managed_session.first_action_received.set()
            return
        result = input_source.handle_browser_payload(
            payload,
            timestamp_s=asyncio.get_running_loop().time(),
        )
        if result.kind == "error":
            self._send_json(channel, make_error_payload(result.error or "Bad input."))
            return
        if result.activated:
            managed_session.first_action_received.set()

    def _record_shared_event_payload(
        self,
        *,
        managed_session: ManagedWebRTCSession,
        payload: dict[str, Any],
    ) -> bool:
        channel = managed_session.control_channel
        input_source = managed_session.input_source
        if channel is None or input_source is None:
            return False
        event_id = str(payload.get("event_id", payload.get("id", ""))).strip()
        state = str(payload.get("state", "trigger")).strip().lower() or "trigger"
        clear_states = {"clear", "release", "off", "none"}
        if not event_id and state not in clear_states:
            self._send_json(
                channel,
                make_error_payload(
                    (
                        "Event payload must include non-empty 'event_id' "
                        "unless state clears the active event."
                    ),
                ),
            )
            return False
        clears = state in clear_states
        try:
            event_payload = self._validate_user_event_payload(
                managed_session=managed_session,
                event_type="text_event",
                payload={
                    "event_id": None if clears else event_id,
                    "state": state,
                },
            )
            active_event_id = event_payload.get("event_id")
            source_event_id = None if active_event_id is None else str(active_event_id)
            input_source.record_user_event(
                timestamp_s=asyncio.get_running_loop().time(),
                event_type="text_event",
                payload=event_payload,
                source_event_id=source_event_id,
            )
        except Exception as exc:
            self._send_json(channel, make_error_payload(str(exc)))
            return False
        active_event_id = event_payload.get("event_id")
        ack_event_id = None if active_event_id is None else str(active_event_id)
        self._send_json(
            channel,
            make_event_ack_payload(
                event_id=ack_event_id,
                state=str(event_payload.get("state", state)),
                result={"active_event_id": ack_event_id},
            ),
        )
        return True

    async def _run_realtime_driver_session(
        self,
        *,
        managed_session: ManagedWebRTCSession,
        context: RunContext,
        session_input: Any,
    ) -> None:
        """Run one or more shared-demo generations on a peer connection."""
        run_mode = WebRTCRunMode(
            edge_factory=_ManagedWebRTCSessionEdgeFactory(
                manager=self,
                managed_session=managed_session,
                loop=asyncio.get_running_loop(),
            )
        )
        try:
            while not managed_session.closed:
                adapter = self._shared_adapter
                spec = self._shared_spec
                scenario = self._shared_scenario
                spec_factory = self._shared_spec_factory
                if adapter is None or spec is None:
                    adapter = _LegacyWebRTCDemoAdapter(
                        runtime=self._runtime,
                        identity=self.identity,
                        session_input=session_input,
                    )
                    spec = self._shared_demo_spec()
                elif spec_factory is not None and session_input is not None:
                    spec = spec_factory(session_input)
                    scenario = None
                if scenario is None:
                    scenario = adapter.prepare_scenario(spec)
                result = await run_demo_session_async(
                    context=context,
                    spec=spec,
                    scenario=scenario,
                    adapter=adapter,
                    run_mode=run_mode,
                    pipeline=(
                        self._shared_pipeline_factory()
                        if self._shared_pipeline_factory is not None
                        else StepPipeline()
                    ),
                    reservation=managed_session.reservation,
                )
                managed_session.reservation = None
                if result.status != "completed":
                    if (
                        self._keep_connection_after_completed
                        and result.status == "not_activated"
                        and result.reason == "transport closed"
                    ):
                        break
                    logger.warning(
                        "Shared WebRTC session ended with status={} reason={}",
                        result.status,
                        result.reason,
                    )
                    if result.reason and managed_session.control_channel is not None:
                        self._send_json(
                            managed_session.control_channel,
                            make_error_payload(result.reason),
                        )
                    break
                logger.info("Shared WebRTC generation completed.")
                if not self._keep_connection_after_completed:
                    break
                if managed_session.control_channel is not None:
                    self._send_json(
                        managed_session.control_channel,
                        {"type": "generation_complete"},
                    )
                input_source = managed_session.input_source
                transport = managed_session.transport
                if input_source is None or transport is None:
                    break
                input_source.reset(start_v=asyncio.get_running_loop().time())
                # Do not construct another driver until the user submits the
                # next prompt/generation. This keeps an idle T2V peer alive
                # without surfacing an expected disconnect as a failed run.
                if not await _wait_for_next_generation_or_disconnect(
                    input_source=input_source,
                    transport=transport,
                ):
                    break
        finally:
            managed_session.reservation = None
            if self._active_session is managed_session:
                await self.close_active_session()

    def _handle_shared_chunk_delivery(
        self,
        *,
        managed_session: ManagedWebRTCSession,
        chunk: WebRTCChunkDelivery,
    ) -> None:
        channel = managed_session.control_channel
        if channel is None or managed_session.closed:
            return
        delivery = chunk.delivery
        enqueued_frames = int(getattr(delivery, "num_frames", chunk.frame_count))
        encode_ms = float(getattr(delivery, "encode_ms", 0.0))
        play_ms = chunk.frame_count * 1000.0 / managed_session.video_track.fps
        queue_depth = managed_session.video_track.qsize()
        self._send_json(
            channel,
            make_chunk_done_payload(
                chunk_index=chunk.step_index,
                num_frames=chunk.frame_count,
                enqueued_frames=enqueued_frames,
                fps=managed_session.video_track.fps,
                width=self.runtime_config.video_width,
                height=self.runtime_config.video_height,
                model=self.identity,
                gen_ms=_stat_ms(chunk.metrics, "model_step_s"),
                enqueue_ms=encode_ms,
                play_ms=play_ms,
                queue_depth=queue_depth,
                lag_ms=0.0,
                control_latency_ms=None,
                consumed_actions=0,
                extra=chunk.metadata,
            ),
        )

    def _handle_shared_delivery_error(
        self,
        *,
        managed_session: ManagedWebRTCSession,
        exc: BaseException,
    ) -> None:
        channel = managed_session.control_channel
        if channel is not None:
            self._send_json(channel, make_error_payload(str(exc)))

    async def _generation_worker(
        self, *, managed_session: ManagedWebRTCSession
    ) -> None:
        """Drive back-to-back chunk generation aligned to the realtime clock.

        Sits idle until the first keyboard event arrives, then drives the
        chunk loop. Each iteration waits for wallclock to catch up to the
        *end* of the next chunk's virtual window, hands legacy segment data and
        frame times to the runtime, and pushes generated frames into the video
        track. The track's bounded queue then paces the loop to playback via
        backpressure on ``BufferedVideoTrack.enqueue_result``.
        """
        loop = asyncio.get_running_loop()
        runtime = managed_session.runtime
        resampler = managed_session.resampler
        video_track = managed_session.video_track
        video_encoder = managed_session.video_encoder

        # Stay idle until the user interacts. Generating eagerly would burn
        # GPU cycles on a still scene the viewer never sees. Once an event
        # arrives we re-anchor the resampler's virtual clock to ``now`` so
        # chunk 0's window starts at the moment of first interaction.
        logger.info("Generation worker idle; waiting for first action.")
        try:
            await managed_session.first_action_received.wait()
        except asyncio.CancelledError:
            logger.info("Generation worker cancelled before first action.")
            raise
        if managed_session.closed:
            return
        resampler.next_chunk_start_v = loop.time()
        logger.info(
            "First action received; starting generation at start_v={:.3f}",
            resampler.next_chunk_start_v,
        )
        perf_log_interval = max(0, int(self._perf_log_interval_chunks))
        perf_window_start = loop.time()
        perf_window_chunks = 0
        perf_window_frames = 0
        try:
            while not managed_session.closed:
                try:
                    request, input_num_frames = self._runtime_next_step_request(runtime)
                except RuntimeError:
                    logger.exception("Runtime not ready; stopping generation worker.")
                    return
                # Trigger when wallclock reaches the chunk's window end.
                chunk_duration = input_num_frames * resampler.dt
                trigger_wall = resampler.next_chunk_start_v + chunk_duration
                delay = trigger_wall - loop.time()
                if delay > 0:
                    await asyncio.sleep(delay)
                if managed_session.closed:
                    break

                # Catch the virtual clock up to wall if it has fallen more
                # than one chunk behind so end-to-end latency stays bounded.
                # The segment branch folds skipped edges through the resampler;
                # the session branch first advances its input canonicalizer
                # across the skipped raw-event window.
                now = loop.time()
                self._catch_up_input_clock(
                    managed_session=managed_session,
                    now=now,
                    chunk_duration=chunk_duration,
                )

                t_before_gen = loop.time()
                chunk_start_v = resampler.next_chunk_start_v
                frame_times = list(resampler.sample_chunk(input_num_frames))
                chunk_end_v = resampler.next_chunk_start_v
                segments: list[Any] = []
                legacy_resampler = managed_session.legacy_segment_resampler
                if legacy_resampler is not None:
                    legacy_resampler.next_chunk_start_v = chunk_start_v
                    segments, frame_times = legacy_resampler.sample_chunk(
                        input_num_frames
                    )
                segment_request = replace(
                    request,
                    user_input_window=TimeWindow(
                        start_s=chunk_start_v,
                        end_s=chunk_end_v,
                    ),
                )
                consumed_action_arrivals: list[float] = []
                while (
                    managed_session.pending_action_arrivals
                    and managed_session.pending_action_arrivals[0] <= chunk_end_v
                ):
                    consumed_action_arrivals.append(
                        managed_session.pending_action_arrivals.popleft()
                    )
                try:
                    if managed_session.inference_session is not None:
                        result = await self._step_inference_session(
                            managed_session=managed_session,
                            window=TimeWindow(
                                start_s=chunk_start_v,
                                end_s=chunk_end_v,
                            ),
                        )
                    else:
                        result = await runtime.step(
                            request=segment_request,
                            segments=segments,
                            frame_times=frame_times,
                        )
                        if result.step_index != segment_request.step_index:
                            raise RuntimeError(
                                "Runtime result step does not match its request: "
                                f"requested {segment_request.step_index}, "
                                f"got {result.step_index}."
                            )
                except _InferenceSessionExhausted:
                    logger.info(
                        "Inference session reported completion; closing WebRTC session."
                    )
                    await self.close_active_session()
                    return
                except Exception as exc:
                    logger.exception("Chunk generation failed.")
                    channel = managed_session.control_channel
                    if channel is not None:
                        self._send_json(channel, make_error_payload(str(exc)))
                    if self.fatal_generation_errors:
                        await self.close_active_session()
                        return
                    continue
                t_after_gen = loop.time()
                delivery = await video_encoder.deliver_chunk(
                    result,
                    video_track,
                    force_keyframe=False,
                )
                enqueued = delivery.num_frames
                t_after_enqueue = loop.time()

                gen_ms = (t_after_gen - t_before_gen) * 1e3
                enqueue_ms = (t_after_enqueue - t_after_gen) * 1e3
                play_ms = result.frame_count * 1000.0 / video_track.fps
                lag_ms = (t_after_enqueue - resampler.next_chunk_start_v) * 1e3
                control_latency_ms = (
                    (t_after_enqueue - consumed_action_arrivals[0]) * 1e3
                    if consumed_action_arrivals
                    else None
                )
                perf_window_chunks += 1
                perf_window_frames += result.frame_count
                if result.step_index == 0 or (
                    perf_log_interval > 0 and result.step_index % perf_log_interval == 0
                ):
                    interval_s = max(t_after_enqueue - perf_window_start, 1.0e-6)
                    interval_fps = perf_window_frames / interval_s
                    gen_fps = result.frame_count / max(
                        t_after_gen - t_before_gen, 1.0e-6
                    )
                    stats = result.metrics
                    logger.info(
                        "WebRTC perf chunk={} interval_chunks={} frames={} "
                        "gen_fps={:.1f} interval_fps={:.1f} playback_fps={} "
                        "gen_ms={:.0f} enqueue_ms={:.0f} model_ms={:.0f} "
                        "denoise_ms={:.0f} decode_ms={:.0f} pixel_post_ms={:.0f} "
                        "copy_ms={:.0f} cache_ms={:.0f} "
                        "cache_wait_ms={:.0f} cache_submit_ms={:.0f} "
                        "queue_depth={} lag_ms={:.0f} control_latency_ms={} "
                        "compile_active={} compile_start_step={} cuda_graph={} "
                        "cache_frames={} cache_tokens={}",
                        result.step_index,
                        perf_window_chunks,
                        perf_window_frames,
                        gen_fps,
                        interval_fps,
                        video_track.fps,
                        gen_ms,
                        enqueue_ms,
                        _stat_ms(stats, "model_step_s", gen_ms),
                        _stat_ms(stats, "denoise_s"),
                        _stat_ms(stats, "decode_s"),
                        _stat_ms(stats, "pixel_post_s"),
                        _stat_ms(stats, "gpu_to_cpu_copy_s"),
                        _stat_ms(stats, "cache_seed_prune_s"),
                        _stat_ms(stats, "cache_update_wait_s"),
                        _stat_ms(stats, "cache_update_submit_s"),
                        video_track.qsize(),
                        lag_ms,
                        "-"
                        if control_latency_ms is None
                        else f"{control_latency_ms:.0f}",
                        _stat_int(stats, "compile_denoise_active"),
                        _stat_int(stats, "compile_denoise_start_step"),
                        _stat_int(stats, "cuda_graph_captured"),
                        _stat_int(stats, "cache_frames"),
                        _stat_int(stats, "cache_tokens"),
                    )
                    perf_window_start = t_after_enqueue
                    perf_window_chunks = 0
                    perf_window_frames = 0
                logger.debug(
                    "Chunk done chunk={} input_frames={} output_frames={} "
                    "segments={} enqueued={} "
                    "gen_ms={:.1f} enqueue_ms={:.1f} play_ms={:.1f} queue_depth={} "
                    "lag_ms={:.1f}",
                    result.step_index,
                    input_num_frames,
                    result.frame_count,
                    len(segments),
                    enqueued,
                    gen_ms,
                    enqueue_ms,
                    play_ms,
                    video_track.qsize(),
                    lag_ms,
                )

                channel = managed_session.control_channel
                if channel is not None:
                    self._send_json(
                        channel,
                        make_chunk_done_payload(
                            chunk_index=result.step_index,
                            num_frames=result.frame_count,
                            enqueued_frames=enqueued,
                            fps=video_track.fps,
                            width=self.runtime_config.video_width,
                            height=self.runtime_config.video_height,
                            model=self.identity,
                            gen_ms=gen_ms,
                            enqueue_ms=enqueue_ms,
                            play_ms=play_ms,
                            queue_depth=video_track.qsize(),
                            lag_ms=lag_ms,
                            control_latency_ms=control_latency_ms,
                            consumed_actions=len(consumed_action_arrivals),
                            extra=result.metadata,
                        ),
                    )
        except asyncio.CancelledError:
            logger.info("Generation worker cancelled.")
            raise

    @staticmethod
    def _send_json(channel: Any, payload: dict[str, Any]) -> None:
        try:
            channel.send(json.dumps(payload))
        except Exception:
            # If the data channel is closing we just drop the message.
            return
