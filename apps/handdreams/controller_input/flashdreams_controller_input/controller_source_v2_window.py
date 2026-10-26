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

"""V2 presentation paired with an asynchronous controller source."""

from __future__ import annotations

import math
import threading
import time
from collections import deque
from collections.abc import Callable

from numpy import uint64

from flashdreams.api_v2.client_window import ClientWindowLink, IClientWindow
from flashdreams.api_v2.user_input_event import UserInputEvent
from flashdreams.runtime_v2.session_desc import SessionDesc
from flashdreams.runtime_v2.step_result import StepResult
from flashdreams.runtime_v2.user_input_event import (
    FocusUserInputEvent,
    KeyboardInputState,
    KeyboardUserInputEvent,
    ResetUserInputEvent,
    XRControllerUserInputEvent,
)
from flashdreams.runtime_v2.user_input_events import UserInputEvents
from flashdreams_controller_input.controller_types import (
    InputSource,
    Observation,
)
from flashdreams_controller_input.events import (
    ControllerConnectionStatus,
    ControllerFrameEvent,
    ControllerSnapshot,
    ControllerStatusEvent,
)

_POLL_INTERVAL_SECONDS = 1.0 / 240.0
_SOURCE_THREAD_JOIN_SECONDS = 5.0
_SOURCE_SILENCE_TIMEOUT_SECONDS = 1.0
_MAX_PENDING_OBSERVATIONS = 256
"""Bound input retained across slow UI ticks; overflow breaks pose continuity."""


def controller_frame_event(
    observation: Observation,
    *,
    session_origin_ns: int,
    frame_id: int,
    pose_convention: str,
    source_name: str,
    client_capture_keys: tuple[str, ...] = ("client_capture_ns",),
) -> ControllerFrameEvent:
    """Snapshot a source frame or disconnect, with absent hands untracked."""
    timestamp = uint64(max(0, (observation.capture_ns - session_origin_ns) // 1_000))
    frame = observation.frame
    if frame is None:
        return ControllerFrameEvent(timestamp=timestamp, action="disconnected")
    client_capture_ns = observation.capture_ns
    for key in reversed(client_capture_keys):
        client_capture_ns = frame.get(key, client_capture_ns)
    hands: dict[str, ControllerSnapshot] = {}
    for side in ("left", "right"):
        controller = frame.get(side)
        if controller is None:
            hands[side] = ControllerSnapshot(
                position=(0.0, 0.0, 0.0),
                orientation=(0.0, 0.0, 0.0, 1.0),
                tracked=False,
            )
            continue
        if not isinstance(controller, dict):
            raise ValueError(f"{source_name} frame has an invalid {side} controller")
        hands[side] = ControllerSnapshot(
            position=controller["position"],
            orientation=controller["orientation"],
            axes=controller.get("axes", ()),
            trigger=controller.get("trigger", 0.0),
            grip=controller.get("grip", 0.0),
            tracked=controller.get("tracked", True),
            battery=controller.get("battery"),
            primary_button=controller.get("primary_button", False),
            secondary_button=controller.get("secondary_button", False),
        )
    return ControllerFrameEvent(
        timestamp=timestamp,
        action="state",
        frame_id=frame_id,
        client_capture_ns=client_capture_ns,
        pose_convention=pose_convention,
        left=hands["left"],
        right=hands["right"],
        is_emulated=frame.get("is_emulated", False),
    )


class ControllerSourceWindow(IClientWindow):
    """Standard V2 presentation plus one long-lived controller source.

    Source startup and polling run on a worker because controller transports may
    take minutes to initialize. A bounded source history survives slow UI ticks. A
    replacement session receives a fresh timestamp epoch while the source and
    its external runtime remain alive.
    """

    def __init__(
        self,
        *,
        output_window: IClientWindow,
        source: InputSource,
        source_name: str,
        pose_convention: str,
        source_thread_name: str,
        source_failure_message: str,
        initial_window_size: tuple[int, int] | None = None,
        client_capture_keys: tuple[str, ...] = ("client_capture_ns",),
        status_provider: Callable[[], ControllerConnectionStatus] | None = None,
        restart_headset: Callable[[], None] | None = None,
        _source_silence_timeout_s: float = _SOURCE_SILENCE_TIMEOUT_SECONDS,
        _silence_clock_ns: Callable[[], int] = time.perf_counter_ns,
    ) -> None:
        if (
            not math.isfinite(_source_silence_timeout_s)
            or _source_silence_timeout_s <= 0.0
        ):
            raise ValueError("source silence timeout must be finite and positive")
        if not source_name:
            raise ValueError("source name must be non-empty")
        if not pose_convention:
            raise ValueError("pose convention must be non-empty")
        if not client_capture_keys or any(not key for key in client_capture_keys):
            raise ValueError("client capture keys must be non-empty")
        self.output_window = output_window
        self._initial_window_size = initial_window_size
        self._source = source
        self._source_name = source_name
        self._pose_convention = pose_convention
        self._source_failure_message = source_failure_message
        self._client_capture_keys = client_capture_keys
        self._status_provider = status_provider
        self._restart_headset = restart_headset
        self._restart_key_pressed = False
        self._last_status: ControllerConnectionStatus | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._observations: deque[Observation] = deque()
        self._observations_overflowed = False
        self._source_failure: BaseException | None = None
        self._session_origin_ns: int | None = None
        self._last_source_activity_ns: int | None = None
        self._source_seen_in_session = False
        self._source_generation: int | None = None
        self._disconnect_emitted = False
        self._source_silence_timeout_ns = int(_source_silence_timeout_s * 1e9)
        self._silence_clock_ns = _silence_clock_ns
        self._next_fallback_frame_id = 0
        self._closed = False
        self._source_stopped = False
        self._source_thread_stopped = False
        self._output_closed = False
        self._cleanup_complete = False
        self._close_lock = threading.Lock()
        self._source_thread = threading.Thread(
            target=self._run_source,
            name=source_thread_name,
            daemon=True,
        )
        self._source_thread.start()

    def _run_source(self) -> None:
        """Start and continuously sample the controller source off the UI thread."""
        # The worker may not be scheduled until after close() has already
        # stopped the source. Never resurrect an external runtime during
        # teardown.
        if self._stop.is_set():
            return
        try:
            self._source.start()
            # stop() may have won after the guard but before start() acquired
            # its resources. Re-stop after startup so close() cannot miss a
            # late-created transport, even for a merely idempotent source.
            if self._stop.is_set():
                self._source.stop()
                return
            while not self._stop.is_set():
                observation = self._source.poll()
                if observation is not None:
                    with self._lock:
                        if len(self._observations) >= _MAX_PENDING_OBSERVATIONS:
                            self._observations.clear()
                            self._observations_overflowed = True
                        self._observations.append(observation)
                        self._last_source_activity_ns = self._silence_clock_ns()
                self._stop.wait(_POLL_INTERVAL_SECONDS)
        except BaseException as error:
            with self._lock:
                if not self._closed:
                    self._source_failure = error
            self._stop.set()
            self._source.stop()

    def request_hide_cursor(self, hide_cursor: bool) -> None:
        """Delegate cursor visibility."""
        self.output_window.request_hide_cursor(hide_cursor)

    def request_lock_cursor_to_window(self, lock_cursor_to_window: bool) -> None:
        """Delegate pointer capture."""
        self.output_window.request_lock_cursor_to_window(lock_cursor_to_window)

    def request_new_window_size(self, new_window_size: tuple[int, int]) -> None:
        """Forward runtime resize requests to the presentation backend."""
        self.output_window.request_new_window_size(new_window_size)

    def request_links(self, links: tuple[ClientWindowLink, ...]) -> None:
        """Forward clickable regions to the operator's presentation backend."""
        self.output_window.request_links(links)

    def open(self, session_desc: SessionDesc) -> None:
        """Open presentation output and begin a fresh input timestamp epoch."""
        self.output_window.open(session_desc)
        if self._initial_window_size is not None:
            # Scene replacement must preserve the operator's subsequent resizing.
            self.request_new_window_size(self._initial_window_size)
            self._initial_window_size = None
        with self._lock:
            self._observations.clear()
            self._observations_overflowed = False
            self._session_origin_ns = time.perf_counter_ns()
            self._source_seen_in_session = False
            self._source_generation = None
            self._disconnect_emitted = False
            self._last_status = None

    def put_user_input_event(self, event: UserInputEvent) -> None:
        """Forward event to the presentation backend, merged with its input."""
        self.output_window.put_user_input_event(event)

    def get_user_input_events(self) -> UserInputEvents:
        """Merge source-clock controller history with buffered window input."""
        # The source owns XR input. Operator B only signals its headset worker;
        # another active controller source must retain its poses and calibration.
        window_events = []
        for event in self.output_window.get_user_input_events().get_events():
            if isinstance(event, XRControllerUserInputEvent):
                continue
            if isinstance(event, FocusUserInputEvent) and not event.focused:
                self._restart_key_pressed = False
            if (
                self._restart_headset is not None
                and isinstance(event, KeyboardUserInputEvent)
                and event.key.lower() == "b"
            ):
                pressed = event.state is KeyboardInputState.PRESSED
                if pressed and not self._restart_key_pressed and not self._closed:
                    self._restart_headset()
                self._restart_key_pressed = pressed
                continue
            window_events.append(event)
        with self._lock:
            source_failure = self._source_failure
            observations = list(self._observations)
            self._observations.clear()
            overflowed, self._observations_overflowed = (
                self._observations_overflowed,
                False,
            )
            session_origin_ns = self._session_origin_ns
            last_source_activity_ns = self._last_source_activity_ns
            source_seen_in_session = self._source_seen_in_session
            disconnect_emitted = self._disconnect_emitted
        if source_failure is not None:
            raise RuntimeError(self._source_failure_message) from source_failure
        if session_origin_ns is not None and self._status_provider is not None:
            status = self._status_provider()
            if status != self._last_status:
                window_events.append(
                    ControllerStatusEvent(
                        timestamp=uint64(
                            max(
                                0, (time.perf_counter_ns() - session_origin_ns) // 1_000
                            )
                        ),
                        status=status,
                    )
                )
                self._last_status = status
        source_silent = (
            last_source_activity_ns is not None
            and self._silence_clock_ns() - last_source_activity_ns
            >= self._source_silence_timeout_ns
        )
        if not observations or session_origin_ns is None or source_silent:
            if (
                session_origin_ns is not None
                and (source_seen_in_session or observations)
                and not disconnect_emitted
                and source_silent
            ):
                with self._lock:
                    self._disconnect_emitted = True
                timestamp = uint64(
                    max(0, (time.perf_counter_ns() - session_origin_ns) // 1_000)
                )
                return UserInputEvents(
                    window_events
                    + [ControllerFrameEvent(timestamp=timestamp, action="disconnected")]
                )
            return UserInputEvents(window_events)
        observations = [
            observation
            for observation in observations
            if observation.capture_ns >= session_origin_ns
        ]
        if not observations:
            return UserInputEvents(window_events)

        with self._lock:
            self._source_seen_in_session = True

        if overflowed:
            window_events.append(
                ControllerFrameEvent(
                    action="disconnected",
                    timestamp=uint64(
                        max(
                            0, (observations[0].capture_ns - session_origin_ns) // 1_000
                        )
                    ),
                )
            )
            disconnect_emitted = True
        for observation in observations:
            previous_generation = self._source_generation
            if previous_generation is not None:
                if observation.generation < previous_generation:
                    continue
                if observation.generation != previous_generation:
                    timestamp = uint64(
                        max(0, (observation.capture_ns - session_origin_ns) // 1_000)
                    )
                    # Runtime reset leaves pending inputs in the event buffer.
                    # The boundary also invalidates poses from the old origin.
                    window_events.extend(
                        [
                            ControllerFrameEvent(
                                timestamp=timestamp, action="disconnected"
                            ),
                            ResetUserInputEvent(timestamp=timestamp),
                        ]
                    )
                    disconnect_emitted = True
            self._source_generation = observation.generation
            if observation.frame is None:
                if not disconnect_emitted:
                    window_events.append(
                        ControllerFrameEvent(
                            timestamp=uint64(
                                max(
                                    0,
                                    (observation.capture_ns - session_origin_ns)
                                    // 1_000,
                                )
                            ),
                            action="disconnected",
                        )
                    )
                disconnect_emitted = True
                continue
            frame = observation.frame or {}
            source_frame_index = frame.get("source_frame_index")
            with self._lock:
                if (
                    isinstance(source_frame_index, int)
                    and not isinstance(source_frame_index, bool)
                    and source_frame_index >= 0
                ):
                    frame_id = source_frame_index
                    self._next_fallback_frame_id = max(
                        self._next_fallback_frame_id,
                        frame_id + 1,
                    )
                else:
                    frame_id = self._next_fallback_frame_id
                    self._next_fallback_frame_id += 1
            event = controller_frame_event(
                observation,
                session_origin_ns=session_origin_ns,
                frame_id=frame_id,
                pose_convention=self._pose_convention,
                source_name=self._source_name,
                client_capture_keys=self._client_capture_keys,
            )
            window_events.append(event)
            disconnect_emitted = False
        with self._lock:
            self._disconnect_emitted = disconnect_emitted
        return UserInputEvents(window_events)

    def write(self, result: StepResult) -> None:
        """Delegate generated output to the presentation window."""
        self.output_window.write(result)

    def metrics_snapshot(self) -> dict[str, float | int]:
        """Return diagnostics when supported by the presentation backend."""
        snapshot = getattr(self.output_window, "metrics_snapshot", None)
        return snapshot() if callable(snapshot) else {}

    def close(self) -> None:
        """Stop the controller worker/source and close presentation output."""
        with self._close_lock:
            with self._lock:
                if self._cleanup_complete:
                    return
                self._closed = True
            self._stop.set()
            failures: list[BaseException] = []
            if not self._source_stopped:
                try:
                    self._source.stop()
                    self._source_stopped = True
                except BaseException as error:
                    failures.append(error)
            if not self._source_thread_stopped:
                self._source_thread.join(timeout=_SOURCE_THREAD_JOIN_SECONDS)
                self._source_thread_stopped = not self._source_thread.is_alive()
                if not self._source_thread_stopped:
                    failures.append(
                        RuntimeError(f"{self._source_name} input worker did not stop")
                    )
            if not self._output_closed:
                try:
                    self.output_window.close()
                    self._output_closed = True
                except BaseException as error:
                    failures.append(error)
            with self._lock:
                self._cleanup_complete = (
                    self._source_stopped
                    and self._source_thread_stopped
                    and self._output_closed
                )
            if failures:
                raise failures[0]


__all__ = [
    "ControllerSourceWindow",
    "controller_frame_event",
]
