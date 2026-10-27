# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Scheduled headless input on the gameplay-ready monotonic clock."""

from __future__ import annotations

import heapq
import threading
import time
from collections.abc import Callable
from queue import SimpleQueue
from typing import Any

import numpy as np

from flashdreams.api_v2.user_input_event import UserInputEvent
from flashdreams.runtime_v2.user_input_event import (
    CloseUserInputEvent,
    FocusUserInputEvent,
    KeyboardInputState,
    KeyboardUserInputEvent,
)
from flashdreams.runtime_v2.user_input_events import UserInputEvents

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
    elif kind == "focus" and set(payload) == common | {"focused"}:
        if isinstance(payload["focused"], bool):
            return FocusUserInputEvent(
                timestamp=np.uint64(timestamp), focused=payload["focused"]
            )
    elif kind == "close" and set(payload) == common:
        return CloseUserInputEvent(timestamp=np.uint64(timestamp))
    raise ValueError("Expected a keyboard, focus, or close input event")


class HeadlessInput:
    """Queue agent and file commands until their deadlines, with acknowledgments.

    Receipt means acceptance by the application, not a sender's socket write.
    Commands accepted on time remain valid when a delayed UI/model reads them.
    All timestamps count from ``start()``, after gameplay initialization.
    """

    def __init__(self, clock_ns: Callable[[], int] = time.monotonic_ns) -> None:
        self.clock_ns = clock_ns
        self.ready_ns: int | None = None
        self.model_ident: int | None = None
        self.native_thread_id: int | None = None
        self.messages: SimpleQueue[tuple[str | None, dict[str, Any]]] = SimpleQueue()
        self._lock = threading.RLock()
        self._seen: set[str] = set()
        self._pending: list[tuple[int, int, UserInputEvent, dict, str | None]] = []
        self._dispatched: dict[int, tuple[dict, str | None]] = {}

    def elapsed_us(self) -> int | None:
        """Return elapsed gameplay wall time, or ``None`` during loading."""
        return (
            None if self.ready_ns is None else (self.clock_ns() - self.ready_ns) // 1000
        )

    def start(self) -> None:
        """Start the capture clock once, after model and gameplay warmup."""
        with self._lock:
            if self.ready_ns is None:
                self.ready_ns = self.clock_ns()
                self.model_ident = threading.get_ident()
                self.native_thread_id = threading.get_native_id()
                self.messages.put((None, {"status": "ready", "elapsed_us": 0}))

    def accept(self, payload: Any, peer: str | None = None) -> None:
        """Accept a future event or report a late, duplicate, or invalid command."""
        with self._lock:
            now = self.elapsed_us()
            identifier = payload.get("id") if isinstance(payload, dict) else None
            valid_id = isinstance(identifier, str) and 0 < len(identifier) <= 128
            response = {"id": identifier if valid_id else None, "received_us": now}
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

    def due(self) -> list[UserInputEvent]:
        """Dispatch due events in timestamp order, retaining their original time."""
        with self._lock:
            now = self.elapsed_us()
            events = []
            while now is not None and self._pending and self._pending[0][0] <= now:
                _, _, event, response, peer = heapq.heappop(self._pending)
                if not isinstance(event, CloseUserInputEvent):
                    self._dispatched[id(event)] = (response, peer)
                self.messages.put(
                    (peer, {**response, "status": "dispatched", "dispatched_us": now})
                )
                events.append(event)
            return events

    def consumed(self, events: UserInputEvents, *, step: int) -> None:
        """Acknowledge events read by the normal model-thread input consumer."""
        with self._lock:
            for event in events.get_events():
                entry = self._dispatched.pop(id(event), None)
                if entry is not None:
                    response, peer = entry
                    self.messages.put(
                        (
                            peer,
                            {
                                **response,
                                "status": "consumed",
                                "consumed_us": self.elapsed_us(),
                                "model_step": step,
                            },
                        )
                    )

    def close(self) -> None:
        """Report commands not consumed before capture shutdown."""
        with self._lock:
            pending = [(response, peer) for _, _, _, response, peer in self._pending]
            pending.extend(self._dispatched.values())
            self._pending.clear()
            self._dispatched.clear()
            for response, peer in pending:
                self.messages.put(
                    (
                        peer,
                        {**response, "status": "cancelled", "reason": "capture_closed"},
                    )
                )
