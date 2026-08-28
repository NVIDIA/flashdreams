# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""SlangPy local-window input canonicalization."""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from threading import Lock
from typing import Any

from flashdreams.demo.io import InputHandler, SessionInfo
from flashdreams.infra.time import TimeWindow
from flashdreams.runtime.canonical import (
    DRIVER_COMMAND,
    InputCanonicalizer,
)
from flashdreams.runtime.gamepad import (
    GAMEPAD_STATE_CAPABILITY,
    GAMEPAD_STATE_EVENT,
    DrivingInputConverter,
    GamepadState,
    gamepad_state_payload,
)
from flashdreams.runtime.inputs import (
    CanonicalInputSchema,
    CanonicalInputWindow,
    UserInputCapability,
    UserInputEvent,
    UserInputs,
    UserInputSchema,
)

_LOCAL_SOURCE_SCHEMA = UserInputSchema(
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
        GAMEPAD_STATE_CAPABILITY,
    ),
    description="SlangPy local-window keyboard and gamepad events.",
)
"""Raw event schema emitted by the SlangPy window callback."""


class SlangPyLocalInputHandler(InputHandler):
    """Convert SlangPy window events into application canonical inputs."""

    def __init__(
        self,
        input_schema: CanonicalInputSchema,
        *,
        process_events: Callable[[], None] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Create a handler for one application input schema.

        Args:
            input_schema: Canonical modalities requested by the application.
            process_events: Optional callback that pumps the owning local window.
            clock: Monotonic clock used for session-relative event timestamps.

        Raises:
            ValueError: The schema requests a modality the local window cannot
                produce.
        """
        converters = []
        unsupported: list[str] = []
        for modality in input_schema.modalities:
            if modality.name != DRIVER_COMMAND.name:
                unsupported.append(modality.name)
                continue
            if not modality.is_satisfied_by(DRIVER_COMMAND):
                unsupported.append(modality.name)
                continue
            if not converters:
                converters.append(DrivingInputConverter())
        if unsupported:
            raise ValueError(
                "Local-window input cannot provide canonical modalities: "
                f"{sorted(set(unsupported))}."
            )

        self._requested_names = frozenset(
            modality.name for modality in input_schema.modalities
        )
        self._canonicalizer = InputCanonicalizer(converters)
        self._process_events = process_events
        self._clock = clock
        self._events: list[UserInputEvent] = []
        self._event_lock = Lock()
        self._session_start_s = 0.0
        self._window_start_s = 0.0
        self._opened = False

    @property
    def accepts_window_events(self) -> bool:
        """Return whether this handler needs callbacks from the local window."""
        return bool(self._requested_names)

    def open(self, session_info: SessionInfo) -> None:
        """Open the handler and reset device state for one session."""
        del session_info
        self._canonicalizer.reset()
        with self._event_lock:
            self._events.clear()
        self._session_start_s = self._clock()
        self._window_start_s = 0.0
        self._opened = True

    def current_inputs(self) -> CanonicalInputWindow:
        """Pump events and return canonical input levels for the elapsed window."""
        if not self._opened:
            raise RuntimeError("Cannot fetch inputs from a closed input handler.")
        if self._process_events is not None:
            self._process_events()

        now_s = max(0.0, self._clock() - self._session_start_s)
        with self._event_lock:
            events = tuple(self._events)
            self._events.clear()
        if events and events[-1].timestamp_s >= now_s:
            now_s = math.nextafter(events[-1].timestamp_s, math.inf)
        window = TimeWindow(start_s=self._window_start_s, end_s=now_s)
        self._window_start_s = now_s
        canonical = self._canonicalizer.canonicalize(
            UserInputs(events=events),
            window=window,
            source_schema=_LOCAL_SOURCE_SCHEMA,
        )
        values = {
            name: value
            for name, value in canonical.values.items()
            if name in self._requested_names
        }
        metadata = dict(canonical.metadata)

        return CanonicalInputWindow(
            values=values,
            metadata=metadata,
            window=window,
        )

    def close(self) -> None:
        """Close the handler and discard queued device events."""
        self._opened = False
        with self._event_lock:
            self._events.clear()

    def on_keyboard_event(self, event: Any) -> None:
        """Record one SlangPy keyboard edge from the window event pump."""
        if not self._opened or DRIVER_COMMAND.name not in self._requested_names:
            return
        is_press = _event_flag(event, "is_key_press")
        is_release = _event_flag(event, "is_key_release")
        is_repeat = _event_flag(event, "is_key_repeat")
        if not (is_press or is_release or is_repeat):
            return
        key = _slangpy_enum_name(getattr(event, "key", None))
        if key is None:
            return
        event_type = "key_up" if is_release else "key_down"
        raw_event = UserInputEvent(
            timestamp_s=max(0.0, self._clock() - self._session_start_s),
            event_type=event_type,
            payload={"key": key},
            source="slangpy-keyboard",
        )
        with self._event_lock:
            self._events.append(raw_event)

    def on_gamepad_event(self, event: Any) -> None:
        """Track SlangPy gamepad connection changes."""
        if not self._opened:
            return
        if _event_flag(event, "is_disconnect"):
            self._record_gamepad_state(GamepadState(False, 0.0, 0.0, 0.0))

    def on_gamepad_state(self, state: Any) -> None:
        """Record the latest SDL gamepad driving state."""
        if not self._opened or DRIVER_COMMAND.name not in self._requested_names:
            return
        self._record_gamepad_state(
            GamepadState(
                connected=True,
                steer=-_clamp(float(getattr(state, "left_x", 0.0)), -1.0, 1.0),
                throttle=_clamp(
                    float(getattr(state, "right_trigger", 0.0)),
                    0.0,
                    1.0,
                ),
                brake=_clamp(
                    float(getattr(state, "left_trigger", 0.0)),
                    0.0,
                    1.0,
                ),
            )
        )

    def _record_gamepad_state(self, state: GamepadState) -> None:
        """Append one normalized gamepad event."""
        event = UserInputEvent(
            timestamp_s=max(0.0, self._clock() - self._session_start_s),
            event_type=GAMEPAD_STATE_EVENT,
            payload=gamepad_state_payload(state),
            source="slangpy-gamepad",
        )
        with self._event_lock:
            self._events.append(event)


def _event_flag(event: Any, method_name: str) -> bool:
    method = getattr(event, method_name, None)
    return bool(method()) if callable(method) else False


def _slangpy_enum_name(value: Any) -> str | None:
    if value is None:
        return None
    name = getattr(value, "name", None)
    if isinstance(name, str):
        return name.lower()
    if isinstance(value, str):
        return value.rsplit(".", 1)[-1].lower()
    return str(value).rsplit(".", 1)[-1].lower()


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


__all__ = ["SlangPyLocalInputHandler"]
