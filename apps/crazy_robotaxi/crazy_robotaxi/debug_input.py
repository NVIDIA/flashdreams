# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Scheduled headless input on the gameplay-ready monotonic clock."""

from __future__ import annotations

import heapq
import threading
import time
from collections.abc import Callable
from queue import SimpleQueue
from typing import TYPE_CHECKING, Any

import numpy as np

from flashdreams.api_v2.user_input_event import UserInputEvent
from flashdreams.runtime_v2.user_input_event import (
    CloseUserInputEvent,
    FocusUserInputEvent,
    GamepadUserInputEvent,
    KeyboardInputState,
    KeyboardUserInputEvent,
)
from flashdreams.runtime_v2.user_input_events import UserInputEvents

if TYPE_CHECKING:
    from crazy_robotaxi.debug_capture import CaptureRecorder

DEBUG_CONTROL_KEY = "robotaxi_debug_control"


def control_event(payload: Any) -> UserInputEvent:
    """Validate a command and construct an ordinary runtime input event."""
    if not isinstance(payload, dict):
        raise ValueError("Input must be a JSON object")
    identifier = payload.get("id")
    timestamp = payload.get("timestamp_us")
    if not isinstance(identifier, str) or not identifier or len(identifier) > 128:
        raise ValueError("Input id must be a nonempty string of at most 128 characters")
    if type(timestamp) is not int or not 0 <= timestamp < 2**63:
        raise ValueError("timestamp_us must be a nonnegative integer below 2**63")
    common = {"id", "timestamp_us", "type"}
    kind = payload.get("type")
    if kind == "keyboard" and set(payload) == common | {"key", "pressed"}:
        if (
            isinstance(payload["key"], str)
            and 0 < len(payload["key"]) <= 128
            and isinstance(payload["pressed"], bool)
        ):
            return KeyboardUserInputEvent(
                timestamp=np.uint64(timestamp),
                key=payload["key"],
                state=KeyboardInputState.PRESSED
                if payload["pressed"]
                else KeyboardInputState.RELEASED,
            )
    elif kind == "gamepad" and common | {"axes", "buttons"} <= set(
        payload
    ) <= common | {"axes", "buttons", "action", "index"}:
        axes, buttons = payload["axes"], payload["buttons"]
        action, index = payload.get("action", "state"), payload.get("index", 0)
        if (
            action in ("connected", "disconnected", "state")
            and type(index) is int
            and 0 <= index < 2**31
            and isinstance(axes, list)
            and len(axes) <= 32
            and isinstance(buttons, list)
            and len(buttons) <= 64
            and all(type(v) in (int, float) and -1 <= v <= 1 for v in axes)
            and all(type(v) in (int, float) and 0 <= v <= 1 for v in buttons)
        ):
            return GamepadUserInputEvent(
                timestamp=np.uint64(timestamp),
                action=action,
                index=index,
                mapping="standard",
                axes=tuple(float(v) for v in axes),
                buttons=tuple(float(v) for v in buttons),
                pressed=tuple(v > 0.5 for v in buttons),
            )
    elif kind == "focus" and set(payload) == common | {"focused"}:
        if isinstance(payload["focused"], bool):
            return FocusUserInputEvent(
                timestamp=np.uint64(timestamp), focused=payload["focused"]
            )
    elif kind == "close" and set(payload) == common:
        return CloseUserInputEvent(timestamp=np.uint64(timestamp))
    raise ValueError("Expected a keyboard, gamepad, focus, or close input event")


class HeadlessInput:
    """Queue agent and file commands until their deadlines, with acknowledgments.

    Receipt means acceptance by the application, not a sender's socket write.
    Commands accepted on time remain valid when a delayed UI/model reads them.
    All timestamps count from ``start()``, after gameplay initialization.
    """

    def __init__(self, clock_ns: Callable[[], int] = time.monotonic_ns) -> None:
        self.presentation_source: dict[str, Any] | None = None
        self.presentation_feedback: dict[str, Any] | None = None
        """Values submitted to the ordinary HUD draw calls on this UI tick."""
        self.capture: CaptureRecorder | None = None
        self.termination_reason: str | None = None
        self.clock_ns = clock_ns
        self.ready_ns: int | None = None
        self.model_ident: int | None = None
        self.native_thread_id: int | None = None
        self.generation_delays_ms: tuple[float, ...] = (0.0,)
        """Repeating additional generation waits; configured before the session starts."""
        self.messages: SimpleQueue[tuple[str | None, dict[str, Any]]] = SimpleQueue()
        self._lock = threading.RLock()
        self._seen: set[str] = set()
        self._pending: list[tuple[int, int, UserInputEvent, dict, str | None]] = []
        self._unconsumed: dict[int, tuple[UserInputEvent, dict, str | None]] = {}
        self._gameplay_receiver: Callable[[UserInputEvents], None] | None = None

    def attach_gameplay(
        self, receiver: Callable[[UserInputEvents], None] | None
    ) -> None:
        """Send accepted commands ahead to the simulation's timestamp queue.

        The UI still receives due events through the normal runtime path. Driving
        has one consumer: the independent simulation, never the model loop.
        """
        with self._lock:
            self._gameplay_receiver = receiver
            if receiver is not None:
                receiver(
                    UserInputEvents(
                        [event for event, _, _ in self._unconsumed.values()]
                    )
                )

    def elapsed_us(self) -> int | None:
        """Return elapsed gameplay wall time, or ``None`` during loading."""
        return (
            None if self.ready_ns is None else (self.clock_ns() - self.ready_ns) // 1000
        )

    def delay_generation(self, chunk: int, stop: threading.Event) -> None:
        """Delay a captured chunk after conditioning, without delaying UI dispatch.

        This simulates slow output availability, not CPU/GPU contention. Warmup
        does not call this hook. Shutdown interrupts the additional wait.
        """
        delay_ms = self.generation_delays_ms[chunk % len(self.generation_delays_ms)]
        if delay_ms == 0:
            return
        started = self.elapsed_us()
        interrupted = stop.wait(delay_ms / 1000)
        self.messages.put(
            (
                None,
                {
                    "status": "generation_delay",
                    "autoregressive_index": chunk,
                    "requested_ms": delay_ms,
                    "started_us": started,
                    "completed_us": self.elapsed_us(),
                    "interrupted": interrupted,
                },
            )
        )

    def start(self) -> None:
        """Start the capture clock once, after model and gameplay warmup."""
        with self._lock:
            if self.ready_ns is None:
                self.ready_ns = self.clock_ns()
                if self.model_ident is None:
                    self.model_ident = threading.get_ident()
                    self.native_thread_id = threading.get_native_id()
                self.messages.put((None, {"status": "ready", "elapsed_us": 0}))

    def accept(self, payload: Any, peer: str | None = None) -> None:
        """Accept a future event or report a late, duplicate, or invalid command."""
        with self._lock:
            now = self.elapsed_us()
            identifier = payload.get("id") if isinstance(payload, dict) else None
            valid_id = isinstance(identifier, str) and 0 < len(identifier) <= 128
            response: dict[str, Any] = {
                "id": identifier if valid_id else None,
                "received_us": now,
            }
            if (
                valid_id
                and isinstance(payload, dict)
                and payload.get("type") == "status"
                and set(payload) == {"id", "type"}
            ):
                self.messages.put(
                    (
                        peer,
                        {
                            **response,
                            "status": "ready" if now is not None else "loading",
                            "elapsed_us": now,
                        },
                    )
                )
                return
            try:
                event = control_event(payload)
                assert isinstance(identifier, str)
                if identifier in self._seen:
                    raise ValueError("duplicate_id")
                if len(self._seen) >= 100_000:
                    raise ValueError("command_limit")
                if now is not None and int(event.timestamp) < now:
                    raise ValueError("late")
            except (ValueError, TypeError) as error:
                self.messages.put(
                    (peer, {**response, "status": "rejected", "reason": str(error)})
                )
                return
            self._seen.add(identifier)
            response.update(timestamp_us=int(event.timestamp), command=dict(payload))
            heapq.heappush(
                self._pending,
                (int(event.timestamp), len(self._seen), event, response, peer),
            )
            self.messages.put((peer, {**response, "status": "accepted"}))
            if not isinstance(event, CloseUserInputEvent):
                self._unconsumed[id(event)] = (event, response, peer)
                if self._gameplay_receiver is not None:
                    self._gameplay_receiver(UserInputEvents([event]))

    def due(self) -> list[UserInputEvent]:
        """Dispatch due events in timestamp order, retaining their original time."""
        with self._lock:
            now = self.elapsed_us()
            events = []
            while now is not None and self._pending and self._pending[0][0] <= now:
                _, _, event, response, peer = heapq.heappop(self._pending)
                if isinstance(event, CloseUserInputEvent):
                    self.termination_reason = "close_command"
                self.messages.put(
                    (
                        peer,
                        {
                            **response,
                            "status": "dispatched"
                            if id(event) in self._unconsumed
                            or isinstance(event, CloseUserInputEvent)
                            else "ui_dispatched",
                            "dispatched_us": now,
                        },
                    )
                )
                events.append(event)
            return events

    def consumed(
        self,
        events: UserInputEvents,
        *,
        step: int | None = None,
        gameplay_tick: int | None = None,
    ) -> None:
        """Acknowledge the actual owner that consumed an input batch."""
        with self._lock:
            for event in events.get_events():
                entry = self._unconsumed.pop(id(event), None)
                if entry is not None:
                    _, response, peer = entry
                    self.messages.put(
                        (
                            peer,
                            {
                                **response,
                                "status": "consumed",
                                "consumed_us": self.elapsed_us(),
                                **(
                                    {"model_step": step}
                                    if step is not None
                                    else {"gameplay_tick": gameplay_tick}
                                ),
                            },
                        )
                    )

    def close(self) -> None:
        """Report commands not consumed before capture shutdown."""
        with self._lock:
            pending = [
                (response, peer)
                for _, _, event, response, peer in self._pending
                if isinstance(event, CloseUserInputEvent)
            ]
            pending.extend(
                (response, peer) for _, response, peer in self._unconsumed.values()
            )
            self._pending.clear()
            self._unconsumed.clear()
            self._gameplay_receiver = None
            for response, peer in pending:
                self.messages.put(
                    (
                        peer,
                        {**response, "status": "cancelled", "reason": "capture_closed"},
                    )
                )
