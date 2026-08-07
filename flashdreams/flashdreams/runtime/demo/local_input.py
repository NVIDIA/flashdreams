# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native-window input source for the shared runtime loop."""

from __future__ import annotations

import time
from collections.abc import Callable

from PIL import Image, ImageDraw

from flashdreams.runtime.inputs import (
    TimeWindow,
    UserInputCapability,
    UserInputEvent,
    UserInputs,
    UserInputSchema,
)
from flashdreams.runtime.sources import QueuedUserInputSource
from flashdreams.serving.presentation.base import (
    HudOverlay,
    KeyEvent,
    PointerEvent,
    Rect,
)
from flashdreams.serving.presentation.composite import CompositeOverlay
from flashdreams.serving.presentation.frame import DisplayFrame


def _local_window_source_schema() -> UserInputSchema:
    return UserInputSchema(
        capabilities=(
            UserInputCapability(
                event_type="key_down",
                payload_fields=frozenset({"key"}),
            ),
            UserInputCapability(
                event_type="key_up",
                payload_fields=frozenset({"key"}),
            ),
            UserInputCapability(
                event_type="pointer_move",
                payload_fields=frozenset({"position"}),
            ),
            UserInputCapability(
                event_type="pointer_down",
                payload_fields=frozenset({"position", "button"}),
            ),
            UserInputCapability(
                event_type="pointer_up",
                payload_fields=frozenset({"position", "button"}),
            ),
            UserInputCapability(
                event_type="wheel_state",
                payload_fields=frozenset({"steer", "throttle", "brake", "reverse"}),
            ),
        )
    )


class LocalWindowInputSource:
    """Queue native-window events using session-relative timestamps."""

    def __init__(
        self,
        *,
        clock: Callable[[], float] = time.monotonic,
        source_name: str = "local-window",
    ) -> None:
        self._clock = clock
        self._source_name = source_name
        self._origin_s = clock()
        self._queue = QueuedUserInputSource()
        self._source_schema = _local_window_source_schema()
        self._overlay = _InputForwardingOverlay(self)

    @property
    def source_schema(self) -> UserInputSchema:
        """Return raw event capabilities produced by the native window."""
        return self._source_schema

    def compose_overlay(self, overlay: HudOverlay) -> CompositeOverlay:
        """Place input forwarding under demo chrome in one overlay graph."""
        return CompositeOverlay(layers=(self._overlay, overlay))

    def reset(self) -> None:
        """Start a fresh session clock and discard queued events."""
        self._origin_s = self._clock()
        self._queue.reset()

    def start_session(self) -> None:
        """Start the input timeline after the model session is ready."""
        self.reset()

    def window(self, time_window: TimeWindow) -> UserInputs:
        """Return events inside one model-requested time window."""
        return self._queue.window(time_window)

    def append_event(self, event: UserInputEvent) -> None:
        """Validate and append one app/device-produced event."""
        self._source_schema.validate_event(event)
        self._queue.append(event)

    def append_wheel_state(
        self,
        *,
        steer: float,
        throttle: float,
        brake: float,
        reverse: bool,
        stop: bool = False,
        observed_s: float | None = None,
    ) -> None:
        """Append one absolute wheel/pedal snapshot."""
        timestamp_s = self._clock() if observed_s is None else observed_s
        self.append_event(
            UserInputEvent(
                timestamp_s=self._relative_timestamp(timestamp_s),
                event_type="wheel_state",
                payload={
                    "steer": steer,
                    "throttle": throttle,
                    "brake": brake,
                    "reverse": reverse,
                    "stop": stop,
                },
                source=self._source_name,
            )
        )

    def _append_key(self, event: KeyEvent) -> None:
        if event.action == "repeat":
            return
        queued = UserInputEvent(
            timestamp_s=self._relative_timestamp(event.timestamp_s),
            event_type="key_down" if event.action == "press" else "key_up",
            payload={"key": event.key},
            source=self._source_name,
        )
        self.append_event(queued)

    def _append_pointer(self, event: PointerEvent) -> None:
        event_types = {
            "move": "pointer_move",
            "press": "pointer_down",
            "release": "pointer_up",
        }
        if event.action != "move" and event.button is None:
            return
        payload: dict[str, object] = {"position": event.position}
        if event.button is not None:
            payload["button"] = event.button
        queued = UserInputEvent(
            timestamp_s=self._relative_timestamp(event.timestamp_s),
            event_type=event_types[event.action],
            payload=payload,
            source=self._source_name,
        )
        self.append_event(queued)

    def _relative_timestamp(self, observed_s: float) -> float:
        return max(0.0, observed_s - self._origin_s)


class _InputForwardingOverlay:
    """Forward events unconsumed by demo chrome into a live input source."""

    def __init__(self, source: LocalWindowInputSource) -> None:
        self._source = source

    def camera_area(self, canvas_size: tuple[int, int]) -> Rect:
        return (0, 0, canvas_size[0], canvas_size[1])

    def draw(
        self,
        canvas: Image.Image,
        draw: ImageDraw.ImageDraw,
        *,
        frame: DisplayFrame,
        camera_area: Rect,
    ) -> None:
        del canvas, draw, frame, camera_area

    def draw_placeholder(
        self,
        canvas: Image.Image,
        draw: ImageDraw.ImageDraw,
        *,
        camera_area: Rect,
    ) -> None:
        del canvas, draw, camera_area

    def prepare(self, frame: DisplayFrame) -> None:
        del frame

    def on_canvas_resized(self, canvas_size: tuple[int, int]) -> None:
        del canvas_size

    def on_key(self, event: KeyEvent) -> bool:
        self._source._append_key(event)
        return True

    def on_pointer(self, event: PointerEvent) -> bool:
        self._source._append_pointer(event)
        return True

    def close(self) -> None:
        return


__all__ = ["LocalWindowInputSource"]
