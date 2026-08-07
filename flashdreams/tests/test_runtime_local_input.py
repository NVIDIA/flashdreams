# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
from flashdreams.runtime import TimeWindow
from flashdreams.runtime.demo.local_input import LocalWindowInputSource
from flashdreams.serving.presentation import (
    DisplayFrame,
    KeyEvent,
    PointerEvent,
    Rect,
)
from PIL import Image, ImageDraw

pytestmark = pytest.mark.ci_cpu


class _Chrome:
    def __init__(self, *, consume_keys: bool = False) -> None:
        self.consume_keys = consume_keys

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
        del event
        return self.consume_keys

    def on_pointer(self, event: PointerEvent) -> bool:
        del event
        return False

    def close(self) -> None:
        return


def test_native_key_edges_become_session_relative_user_events() -> None:
    source = LocalWindowInputSource(clock=lambda: 100.0)
    overlay = source.compose_overlay(_Chrome())

    assert overlay.on_key(KeyEvent(key="w", action="press", timestamp_s=101.25))
    assert overlay.on_key(KeyEvent(key="w", action="release", timestamp_s=101.75))

    inputs = source.window(TimeWindow(start_s=1.0, end_s=2.0))
    assert [
        (event.timestamp_s, event.event_type, event.payload["key"])
        for event in inputs.events
    ] == [
        (1.25, "key_down", "w"),
        (1.75, "key_up", "w"),
    ]


def test_demo_chrome_can_consume_ui_keys_before_model_input_sees_them() -> None:
    source = LocalWindowInputSource(clock=lambda: 10.0)
    overlay = source.compose_overlay(_Chrome(consume_keys=True))

    assert overlay.on_key(KeyEvent(key="escape", action="press", timestamp_s=11.0))

    assert source.window(TimeWindow(start_s=0.0, end_s=2.0)).events == ()


def test_key_repeats_do_not_duplicate_level_triggered_input() -> None:
    source = LocalWindowInputSource(clock=lambda: 0.0)
    overlay = source.compose_overlay(_Chrome())

    overlay.on_key(KeyEvent(key="w", action="repeat", timestamp_s=0.5))

    assert source.window(TimeWindow(start_s=0.0, end_s=1.0)).events == ()


def test_pointer_events_preserve_position_and_button() -> None:
    source = LocalWindowInputSource(clock=lambda: 5.0)
    overlay = source.compose_overlay(_Chrome())

    overlay.on_pointer(
        PointerEvent(
            action="press",
            position=(12, 34),
            timestamp_s=5.25,
            button="left",
        )
    )

    event = source.window(TimeWindow(start_s=0.0, end_s=1.0)).events[0]
    assert event.event_type == "pointer_down"
    assert event.payload == {"position": (12, 34), "button": "left"}


def test_pointer_transition_without_a_known_button_is_dropped() -> None:
    source = LocalWindowInputSource(clock=lambda: 5.0)
    overlay = source.compose_overlay(_Chrome())

    overlay.on_pointer(
        PointerEvent(
            action="press",
            position=(12, 34),
            timestamp_s=5.25,
            button=None,
        )
    )

    assert source.window(TimeWindow(start_s=0.0, end_s=1.0)).events == ()


def test_reset_restarts_the_session_clock_and_clears_pending_events() -> None:
    times = iter((10.0, 20.0))
    source = LocalWindowInputSource(clock=lambda: next(times))
    overlay = source.compose_overlay(_Chrome())
    overlay.on_key(KeyEvent(key="w", action="press", timestamp_s=11.0))

    source.reset()
    overlay.on_key(KeyEvent(key="s", action="press", timestamp_s=20.5))

    inputs = source.window(TimeWindow(start_s=0.0, end_s=1.0))
    assert [(event.timestamp_s, event.payload["key"]) for event in inputs.events] == [
        (0.5, "s")
    ]
