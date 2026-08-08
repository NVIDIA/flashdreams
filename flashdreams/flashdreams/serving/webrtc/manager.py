# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""WebRTC session lifecycle and control-message orchestration."""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import json
from collections import deque
from collections.abc import Mapping
from collections.abc import Set as AbstractSet
from dataclasses import dataclass, field, replace
from typing import Any, Generic, TypeVar

from aiortc import (
    RTCConfiguration,
    RTCPeerConnection,
    RTCRtpSender,
    RTCSessionDescription,
)
from loguru import logger

from flashdreams.runtime.inputs import (
    InferenceInput,
    TimeWindow,
    UserInputEvent,
    UserInputs,
)
from flashdreams.runtime.types import StepRequest, StepResult
from flashdreams.serving.realtime.input import (
    DEFAULT_SUPPORTED_KEYS,
    KeyboardResampler,
    normalize_key,
)
from flashdreams.serving.webrtc.encoders import (
    DefaultRTCEncoder,
    VideoEncoder,
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
    WebRTCSessionRuntime,
)
from flashdreams.serving.webrtc.server import SessionBusyError
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

_RuntimeT = TypeVar("_RuntimeT", bound=WebRTCSessionRuntime)
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


@dataclass(slots=True)
class ManagedWebRTCSession:
    """Per-session state for the single active WebRTC peer connection."""

    runtime: Any
    video_track: BufferedVideoTrack | NVENCVideoTrack
    video_encoder: VideoEncoder
    peer_connection: Any
    resampler: KeyboardResampler
    control_channel: Any | None = None
    generation_task: asyncio.Task[Any] | None = None
    first_action_received: asyncio.Event = field(default_factory=asyncio.Event)
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
    ) -> None:
        if client_liveness_timeout_s <= 0:
            raise ValueError("client_liveness_timeout_s must be > 0")
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
        self._runtime = runtime
        self._runtime_ready = False
        self._warmup_complete = False
        self._active_session: ManagedWebRTCSession | None = None
        self._preload_lock = asyncio.Lock()
        self._session_lock = asyncio.Lock()
        self._pending_session_input: Any = None

    @property
    def pending_session_input(self) -> Any:
        """Input that will be applied to the next successfully negotiated session."""
        return self._pending_session_input

    @property
    def runtime(self) -> _RuntimeT:
        """Model runtime driven by this transport manager."""
        return self._runtime

    def set_pending_session_input(self, session_input: Any) -> None:
        """Store validated model input for the next session."""
        if self.has_active_session():
            raise SessionBusyError(self.busy_message)
        self._pending_session_input = session_input

    def _make_resampler(self, *, start_v: float) -> KeyboardResampler:
        return self._make_resampler_at_fps(start_v=start_v, fps=self.fps)

    def _make_resampler_at_fps(
        self, *, start_v: float, fps: float
    ) -> KeyboardResampler:
        supported_control_keys = self._effective_supported_control_keys()
        if supported_control_keys is None:
            return KeyboardResampler(fps=fps, start_v=start_v)
        return KeyboardResampler(
            fps=fps,
            start_v=start_v,
            supported_keys=supported_control_keys,
        )

    def _effective_supported_control_keys(self) -> frozenset[str] | None:
        supported_control_keys = self.supported_control_keys
        if supported_control_keys is not None:
            return frozenset(supported_control_keys)
        legacy_supported_keys = getattr(self, "_resampler_supported_keys", None)
        if legacy_supported_keys is None:
            return None
        return frozenset(legacy_supported_keys)

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
        return self._positive_float_runtime_value(
            runtime.peek_input_fps(),
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
        return self._positive_int_runtime_value(
            runtime.peek_steady_output_num_frames(),
            label="peek_steady_output_num_frames",
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
            encoder = DefaultRTCEncoder(fps=self.fps)
        return encoder

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
        return callable(getattr(runtime, "start_inference_session", None))

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
        return self._runtime_ready

    async def preload_runtime(self) -> None:
        async with self._preload_lock:
            if not self._runtime_ready:
                await self._runtime.initialize()
                self._runtime_ready = True
            if not self._warmup_complete:
                await self._run_loopback_warmup_session(
                    num_chunks=self.runtime_config.warmup_chunks
                )
                self._warmup_complete = True

    async def create_answer(self, *, offer_sdp: str, offer_type: str) -> dict[str, str]:
        if not self._runtime_ready or not self._warmup_complete:
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
        if not self._runtime_ready:
            raise RuntimeError("Runtime is not initialized.")

        await self._runtime.reset_for_new_session(session_input=session_input)

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
        loop = asyncio.get_running_loop()
        managed_session = ManagedWebRTCSession(
            runtime=self._runtime,
            video_track=video_track,
            video_encoder=video_encoder,
            peer_connection=peer_connection,
            resampler=resampler,
            last_client_message_at=loop.time(),
        )
        session_runtime: Any = self._runtime
        if self._drives_inference_session(session_runtime):
            managed_session.inference_session = (
                await session_runtime.start_inference_session()
            )
        self._active_session = managed_session
        if enable_liveness_watchdog:
            managed_session.liveness_task = asyncio.create_task(
                self._client_liveness_watchdog(managed_session=managed_session)
            )

        @peer_connection.on("datachannel")
        def on_datachannel(channel: Any) -> None:
            managed_session.control_channel = channel
            # Re-anchor the resampler at channel open. The real
            # virtual-clock anchor happens in ``_generation_worker`` once
            # the first keyboard event arrives.
            channel_open_v = asyncio.get_running_loop().time()
            managed_session.resampler.reset(start_v=channel_open_v)

            @channel.on("message")
            def on_message(message: Any) -> None:
                asyncio.create_task(
                    self._handle_datachannel_message(
                        managed_session=managed_session,
                        raw_message=message,
                    )
                )

            # Spawn the generation worker once the channel is wired up so
            # ``chunk_done`` notifications have a channel to land on.
            managed_session.generation_task = asyncio.create_task(
                self._generation_worker(managed_session=managed_session)
            )

            @channel.on("close")
            def on_close() -> None:
                logger.info("Control data channel closed; closing active session.")
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
        if not self._runtime_ready:
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
        loop = asyncio.get_running_loop()
        try:
            while not managed_session.closed:
                elapsed_s = loop.time() - managed_session.last_client_message_at
                if elapsed_s >= self.client_liveness_timeout_s:
                    logger.warning(
                        "No client heartbeat/control message for {:.1f}s; "
                        "closing active session.",
                        elapsed_s,
                    )
                    await self.close_active_session()
                    return
                await asyncio.sleep(
                    min(
                        _CLIENT_LIVENESS_CHECK_INTERVAL_S,
                        self.client_liveness_timeout_s - elapsed_s,
                    )
                )
        except asyncio.CancelledError:
            raise

    async def shutdown(self) -> None:
        await self.close_active_session()
        await self._runtime.close()
        self._runtime_ready = False
        self._warmup_complete = False

    def wait_for_termination(self) -> None:
        self._runtime.wait_for_termination()

    def send_exit_signal(self) -> None:
        self._runtime.send_exit_signal()

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

        # Stamp arrival on the same monotonic clock that seeds the
        # resampler's ``next_chunk_start_v`` so virtual-time comparisons in
        # ``KeyboardResampler.sample_chunk`` are well-defined.
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
        managed_session.resampler.on_edge(arrival_t=arrival_t, event=event, key=key)
        managed_session.pending_action_arrivals.append(arrival_t)
        # Releases the generation worker, which blocks on this until the
        # user actually interacts. Idempotent once already set.
        managed_session.first_action_received.set()

    async def _generation_worker(
        self, *, managed_session: ManagedWebRTCSession
    ) -> None:
        """Drive back-to-back chunk generation aligned to the resampler clock.

        Sits idle until the first keyboard event arrives, then drives the
        chunk loop. Each iteration waits for wallclock to catch up to the
        *end* of the next chunk's virtual window, samples the chunk's
        piecewise-constant timeline, hands segments and frame times to the
        runtime, and pushes the generated frames into the video track. The
        track's bounded queue then paces the loop to playback via
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
                segments, frame_times = resampler.sample_chunk(input_num_frames)
                chunk_end_v = resampler.next_chunk_start_v
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
