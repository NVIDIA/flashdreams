# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""WebRTC session lifecycle and control-message orchestration."""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import json
from collections import deque
from collections.abc import Set as AbstractSet
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any

import torch
from aiortc import RTCConfiguration, RTCPeerConnection, RTCSessionDescription
from loguru import logger

from flashdreams.serving.realtime.input import KeyboardResampler
from flashdreams.serving.webrtc.media import BufferedVideoTrack
from flashdreams.serving.webrtc.server import SessionBusyError
from flashdreams.serving.webrtc.warmup import (
    run_loopback_warmup_session,
    wait_for_ice_gathering_complete,
)

# Close the active session if no client heartbeat/control message arrives
# within this many seconds. Browsers sends periodic heartbeats.
DEFAULT_CLIENT_LIVENESS_TIMEOUT_S = 10.0

# How often the liveness watchdog wakes to re-check the elapsed-since-last-message.
_CLIENT_LIVENESS_CHECK_INTERVAL_S = 1.0


class WebRTCControlSignal(IntEnum):
    """Rank-orchestration signals shared by the single-session runtimes."""

    INITIALIZE = 0
    RESET_SESSION = 1
    ACTION_STEP = 2
    CLOSE = 3
    EVENT = 4
    EXIT = 99


@dataclass(slots=True)
class WebRTCStepResult:
    """One generated chunk handed back by a model runtime's ``generate_chunk``."""

    chunk_index: int
    num_frames: int
    video_chunk: torch.Tensor
    stats: dict[str, float] | None


@dataclass(slots=True)
class ManagedWebRTCSession:
    """Per-session state for the single active WebRTC peer connection."""

    runtime: Any
    video_track: BufferedVideoTrack
    peer_connection: Any
    resampler: KeyboardResampler
    control_channel: Any | None = None
    generation_task: asyncio.Task[Any] | None = None
    first_action_received: asyncio.Event = field(default_factory=asyncio.Event)
    pending_action_arrivals: deque[float] = field(default_factory=deque)
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


class BaseWebRTCSessionManager:
    """Owns one active WebRTC session and forwards actions into a model runtime."""

    _busy_message: str = "A WebRTC session is already active."
    _warmup_label: str = "WebRTC"
    _runtime_error_types: tuple[type[Exception], ...] = (RuntimeError,)
    _close_session_on_generation_error: bool = False
    _resampler_supported_keys: AbstractSet[str] | None = None

    def __init__(
        self,
        *,
        runtime: Any,
        runtime_config: Any,
        fps: int,
        client_liveness_timeout_s: float = DEFAULT_CLIENT_LIVENESS_TIMEOUT_S,
    ) -> None:
        if client_liveness_timeout_s <= 0:
            raise ValueError("client_liveness_timeout_s must be > 0")
        self.runtime_config = runtime_config
        self.fps = fps
        self.client_liveness_timeout_s = client_liveness_timeout_s
        self._runtime = runtime
        self._runtime_ready = False
        self._warmup_complete = False
        self._active_session: ManagedWebRTCSession | None = None
        self._preload_lock = asyncio.Lock()
        self._session_lock = asyncio.Lock()

    def _model_name(self) -> str:
        """Human-readable model identifier reported in ``chunk_done``."""
        raise NotImplementedError

    def _peek_pending_session_input(self) -> Any:
        """Session input applied to the next ``create_answer`` (or ``None``)."""
        return None

    def _clear_pending_session_input(self) -> None:
        """Clear the pending session input after a successful answer."""

    async def _reset_runtime_for_session(self, session_input: Any) -> None:
        """Reset the runtime for a new rollout, honoring ``session_input``."""
        await self._runtime.reset_for_new_session()

    def _make_resampler(self, *, start_v: float) -> KeyboardResampler:
        if self._resampler_supported_keys is None:
            return KeyboardResampler(fps=self.fps, start_v=start_v)
        return KeyboardResampler(
            fps=self.fps,
            start_v=start_v,
            supported_keys=frozenset(self._resampler_supported_keys),
        )

    def _register_extra_peer_handlers(self, peer_connection: Any) -> None:
        """Register optional extra peer-connection event handlers."""

    def _on_offer_received(self, offer_sdp: str) -> None:
        """Hook invoked with the remote offer SDP before negotiation."""

    def _on_answer_created(self, answer_sdp: str) -> None:
        """Hook invoked with the local answer SDP after negotiation."""

    def _chunk_done_extra(self) -> dict[str, Any]:
        """Extra fields merged into every ``chunk_done`` payload."""
        return {}

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
                    {
                        "type": "error",
                        "message": (
                            "Event payload must include non-empty 'event_id' "
                            "unless state clears the active event."
                        ),
                    },
                )
            return False

        trigger_event = getattr(managed_session.runtime, "trigger_event", None)
        if not callable(trigger_event):
            if channel is not None:
                self._send_json(
                    channel,
                    {
                        "type": "error",
                        "message": "This runtime does not support event messages.",
                    },
                )
            return False

        try:
            result = trigger_event(event_id=event_id, state=state)
            if inspect.isawaitable(result):
                result = await result
        except Exception as exc:
            if channel is not None:
                self._send_json(channel, {"type": "error", "message": str(exc)})
            return False

        if channel is not None:
            ack: dict[str, Any] = {
                "type": "event_ack",
                "event_id": event_id or None,
                "state": state,
            }
            if isinstance(result, dict):
                for key, value in result.items():
                    if key not in ack:
                        ack[key] = value
            self._send_json(channel, ack)
        return True

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
                raise SessionBusyError(self._busy_message)

            session_input = self._peek_pending_session_input()
            answer = await self._create_answer_with_runtime_ready_locked(
                offer_sdp=offer_sdp,
                offer_type=offer_type,
                session_input=session_input,
            )
            self._clear_pending_session_input()
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
            raise SessionBusyError(self._busy_message)
        if not self._runtime_ready:
            raise self._runtime_error_types[0]("Runtime is not initialized.")

        await self._reset_runtime_for_session(session_input)

        peer_connection = RTCPeerConnection(rtc_configuration)
        # Bounded queue sized to one *steady-state* chunk so the producer
        # is throttled to the consumer's drain rate. AR step 0 emits fewer
        # frames than steady state; sizing to it would force a per-chunk
        # stall, so we size to the steady-state count.
        num_frames = self._runtime.peek_steady_chunk_num_frames()
        video_track = BufferedVideoTrack(fps=self.fps, maxsize=num_frames)
        peer_connection.addTrack(video_track)
        # Start the resampler's virtual clock at 0; the real anchor is set
        # in the ``on_datachannel`` handler so chunk 0's window starts when
        # input can actually arrive.
        resampler = self._make_resampler(start_v=0.0)
        loop = asyncio.get_running_loop()
        managed_session = ManagedWebRTCSession(
            runtime=self._runtime,
            video_track=video_track,
            peer_connection=peer_connection,
            resampler=resampler,
            last_client_message_at=loop.time(),
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

        self._register_extra_peer_handlers(peer_connection)

        try:
            offer = RTCSessionDescription(sdp=offer_sdp, type=offer_type)
            self._on_offer_received(offer_sdp)
            await peer_connection.setRemoteDescription(offer)
            answer = await peer_connection.createAnswer()
            await peer_connection.setLocalDescription(answer)
            await wait_for_ice_gathering_complete(peer_connection)
            local_description = peer_connection.localDescription
            if local_description is None:
                raise RuntimeError("Peer connection did not produce local description.")
            self._on_answer_created(local_description.sdp)
            return {"sdp": local_description.sdp, "type": local_description.type}
        except Exception:
            logger.exception("WebRTC negotiation failed while creating an answer.")
            await managed_session.close()
            self._active_session = None
            raise

    async def _run_loopback_warmup_session(self, *, num_chunks: int) -> None:
        if not self._runtime_ready:
            raise self._runtime_error_types[0]("Runtime is not initialized.")
        await run_loopback_warmup_session(
            num_chunks=num_chunks,
            warmup_timeout_s=self.runtime_config.warmup_timeout_s,
            create_answer=self._create_loopback_warmup_answer,
            close_active_session=self.close_active_session,
            label=self._warmup_label,
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
            self._send_json(
                channel, {"type": "error", "message": "Expected text payload."}
            )
            return

        try:
            payload = json.loads(raw_message)
        except json.JSONDecodeError:
            self._send_json(
                channel, {"type": "error", "message": "Invalid JSON payload."}
            )
            return

        if not isinstance(payload, dict):
            self._send_json(
                channel, {"type": "error", "message": "Payload must be a JSON object."}
            )
            return
        message_type = str(payload.get("type", "")).strip().lower()
        if message_type == "heartbeat":
            return
        if message_type == "disconnect":
            logger.info("Client requested disconnect; closing active session.")
            await self.close_active_session()
            return
        if message_type == "event":
            handled = await self._handle_event_message(
                managed_session=managed_session,
                payload=payload,
            )
            if handled:
                # Text events intentionally count as first interaction: a client may
                # want the model to generate an idle-camera chunk with updated text.
                managed_session.first_action_received.set()
            return
        if message_type != "action":
            self._send_json(
                channel,
                {
                    "type": "error",
                    "message": "Unsupported message type, expected "
                    "'action', 'event', 'heartbeat', or 'disconnect'.",
                },
            )
            return

        action_payload = payload.get("action", payload)
        if not isinstance(action_payload, dict):
            self._send_json(
                channel, {"type": "error", "message": "'action' must be an object."}
            )
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
                {
                    "type": "error",
                    "message": f"Unsupported event={event!r}; "
                    "expected 'keydown' or 'keyup'.",
                },
            )
            return
        key = str(action_payload.get("key", "")).strip()
        if not key:
            self._send_json(
                channel,
                {
                    "type": "error",
                    "message": "Action payload must include non-empty 'key'.",
                },
            )
            return

        # Stamp arrival on the same monotonic clock that seeds the
        # resampler's ``next_chunk_start_v`` so virtual-time comparisons in
        # ``KeyboardResampler.sample_chunk`` are well-defined.
        arrival_t = asyncio.get_running_loop().time()
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
        backpressure on ``BufferedVideoTrack.enqueue_chunk``.
        """
        loop = asyncio.get_running_loop()
        runtime = managed_session.runtime
        resampler = managed_session.resampler
        video_track = managed_session.video_track

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
        try:
            while not managed_session.closed:
                try:
                    num_frames = runtime.peek_next_chunk_num_frames()
                except self._runtime_error_types:
                    logger.exception("Runtime not ready; stopping generation worker.")
                    return
                # Trigger when wallclock reaches the chunk's window end.
                chunk_duration = num_frames * resampler.dt
                trigger_wall = resampler.next_chunk_start_v + chunk_duration
                delay = trigger_wall - loop.time()
                if delay > 0:
                    await asyncio.sleep(delay)
                if managed_session.closed:
                    break

                # Catch the virtual clock up to wall if it has fallen more
                # than one chunk behind so end-to-end latency stays bounded.
                # Held-key continuity is preserved because ``sample_chunk``
                # folds every event below the new window start into the
                # carried state.
                now = loop.time()
                lag = now - (resampler.next_chunk_start_v + chunk_duration)
                if lag > chunk_duration:
                    resampler.next_chunk_start_v = now - chunk_duration

                t_before_gen = loop.time()
                segments, frame_times = resampler.sample_chunk(num_frames)
                chunk_end_v = resampler.next_chunk_start_v
                consumed_action_arrivals: list[float] = []
                while (
                    managed_session.pending_action_arrivals
                    and managed_session.pending_action_arrivals[0] <= chunk_end_v
                ):
                    consumed_action_arrivals.append(
                        managed_session.pending_action_arrivals.popleft()
                    )
                try:
                    result = await runtime.generate_chunk(
                        segments=segments, frame_times=frame_times
                    )
                except Exception as exc:
                    logger.exception("Chunk generation failed.")
                    channel = managed_session.control_channel
                    if channel is not None:
                        self._send_json(channel, {"type": "error", "message": str(exc)})
                    if self._close_session_on_generation_error:
                        await self.close_active_session()
                        return
                    continue
                t_after_gen = loop.time()
                enqueued = await video_track.enqueue_chunk(result.video_chunk)
                t_after_enqueue = loop.time()

                gen_ms = (t_after_gen - t_before_gen) * 1e3
                enqueue_ms = (t_after_enqueue - t_after_gen) * 1e3
                play_ms = result.num_frames * 1000.0 / video_track.fps
                lag_ms = (t_after_enqueue - resampler.next_chunk_start_v) * 1e3
                control_latency_ms = (
                    (t_after_enqueue - consumed_action_arrivals[0]) * 1e3
                    if consumed_action_arrivals
                    else None
                )
                logger.debug(
                    "Chunk done chunk={} num_frames={} segments={} enqueued={} "
                    "gen_ms={:.1f} enqueue_ms={:.1f} play_ms={:.1f} queue_depth={} "
                    "lag_ms={:.1f}",
                    result.chunk_index,
                    result.num_frames,
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
                    payload: dict[str, Any] = {
                        "type": "chunk_done",
                        "chunk_index": result.chunk_index,
                        "num_frames": result.num_frames,
                        "enqueued_frames": enqueued,
                        "fps": video_track.fps,
                        "resolution": {
                            "width": self.runtime_config.video_width,
                            "height": self.runtime_config.video_height,
                        },
                        "model": self._model_name(),
                        "gen_ms": round(gen_ms, 1),
                        "enqueue_ms": round(enqueue_ms, 1),
                        "play_ms": round(play_ms, 1),
                        "queue_depth": video_track.qsize(),
                        "lag_ms": round(lag_ms, 1),
                    }
                    payload.update(self._chunk_done_extra())
                    if control_latency_ms is not None:
                        payload["latency_ms"] = round(control_latency_ms, 1)
                        payload["control_latency_ms"] = round(control_latency_ms, 1)
                        payload["consumed_actions"] = len(consumed_action_arrivals)
                    self._send_json(channel, payload)
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
