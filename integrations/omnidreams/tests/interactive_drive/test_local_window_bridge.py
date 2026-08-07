# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Translation between the engine's frame contract and the shared presenter."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from flashdreams.serving.presentation import DisplayFrame, KeyEvent, PointerEvent
from interactive_drive_app.input.keyboard import KeyboardState
from interactive_drive_app.overlays import BEV_OVERLAY_KEY, SceneHeaderWidget
from omnidreams.interactive_drive.local_window_bridge import (
    _build_overlay,
    _display_frame,
    _KeyboardSlot,
)
from omnidreams.interactive_drive.types import PresentedFrame
from PIL import Image, ImageDraw

pytestmark = pytest.mark.ci_cpu


def _frame(
    *,
    timestamp_us: int = 1234,
    rgb_host_uint8: Any = None,
    model_rgb_host_uint8: Any = None,
    bev_host_uint8: Any = None,
    status_message: str | None = None,
) -> PresentedFrame:
    return PresentedFrame(
        timestamp_us=timestamp_us,
        rgb_host_uint8=(
            np.zeros((4, 4, 3), dtype=np.uint8)
            if rgb_host_uint8 is None
            else rgb_host_uint8
        ),
        depth_host_f32=None,
        model_rgb_host_uint8=model_rgb_host_uint8,
        bev_host_uint8=bev_host_uint8,
        status_message=status_message,
    )


## Keyboard rebinding


def test_rebinding_the_keyboard_reaches_chrome_the_presenter_already_captured() -> None:
    """The demo builds the presenter against a placeholder, then hands over the
    engine's real keyboard. Because the presenter captures the overlay at
    construction, the rebind has to reach the existing overlay rather than
    replace it."""
    placeholder, engine = KeyboardState(), KeyboardState()
    slot = _KeyboardSlot(placeholder)
    overlay = _build_overlay(slot)

    slot.keyboard = engine
    overlay.on_key(KeyEvent(key="w", action="press", timestamp_s=0.0))

    assert engine.command().throttle > 0
    assert placeholder.command().throttle == 0
    overlay.close()


def test_rebinding_leaves_the_overlay_object_identity_untouched() -> None:
    """Identity is the invariant: the presenter and its compositor hold this
    object, so a swap must not produce a new one."""
    slot = _KeyboardSlot(KeyboardState())
    overlay = _build_overlay(slot)
    layers = overlay.layers

    slot.keyboard = KeyboardState()

    assert overlay.layers is layers
    overlay.close()


def test_drive_keys_route_to_whichever_keyboard_is_currently_bound() -> None:
    first, second = KeyboardState(), KeyboardState()
    slot = _KeyboardSlot(first)
    overlay = _build_overlay(slot)

    overlay.on_key(KeyEvent(key="a", action="press", timestamp_s=0.0))
    slot.keyboard = second
    overlay.on_key(KeyEvent(key="d", action="press", timestamp_s=0.0))

    # ``a`` steers positive and ``d`` negative in this codebase's convention;
    # the point is that each press landed on the keyboard bound at the time.
    assert first.command().steer > 0
    assert second.command().steer < 0
    overlay.close()


def test_reset_request_follows_the_rebind() -> None:
    placeholder, engine = KeyboardState(), KeyboardState()
    slot = _KeyboardSlot(placeholder)
    overlay = _build_overlay(slot)

    slot.keyboard = engine
    overlay.on_key(KeyEvent(key="r", action="press", timestamp_s=0.0))

    assert engine.consume_reset_request()
    assert not placeholder.consume_reset_request()
    overlay.close()


## Frame translation


def test_model_rgb_view_shows_the_world_model_output() -> None:
    model = np.ones((2, 2, 3), dtype=np.uint8)
    display = _display_frame(_frame(model_rgb_host_uint8=model), "model_rgb")

    assert display.image is model


def test_model_rgb_view_falls_back_to_raster_when_the_model_has_no_frame() -> None:
    """Early chunks arrive before the world model produces anything."""
    raster = np.ones((2, 2, 3), dtype=np.uint8)
    display = _display_frame(_frame(rgb_host_uint8=raster), "model_rgb")

    assert display.image is raster


def test_raster_view_ignores_an_available_model_frame() -> None:
    raster = np.ones((2, 2, 3), dtype=np.uint8)
    display = _display_frame(
        _frame(rgb_host_uint8=raster, model_rgb_host_uint8=np.zeros((8, 8, 3))),
        "rgb",
    )

    assert display.image is raster


def test_only_the_model_view_may_grow_the_window() -> None:
    """The raster view is already rendered at window resolution, so letting it
    drive a resize would fight the user's own sizing."""
    model = _display_frame(_frame(model_rgb_host_uint8=np.ones((2, 2, 3))), "model_rgb")
    raster = _display_frame(_frame(), "rgb")

    assert model.allow_window_resize
    assert not raster.allow_window_resize


def test_bev_travels_as_opaque_overlay_data() -> None:
    """The presenter never inspects this; only the BEV widget knows the key."""
    bev = np.ones((3, 3, 3), dtype=np.uint8)
    display = _display_frame(_frame(bev_host_uint8=bev), "rgb")

    assert display.overlay_data[BEV_OVERLAY_KEY] is bev


def test_status_message_and_timestamp_survive_translation() -> None:
    display = _display_frame(
        _frame(timestamp_us=99, status_message="Respawning..."), "rgb"
    )

    assert display.status_message == "Respawning..."
    assert display.timestamp_us == 99


## Post-process toggle


def _header_rect(widget: SceneHeaderWidget) -> tuple[int, int, int, int]:
    return (0, 0, 500, widget.measure(500))


def _draw(widget: SceneHeaderWidget) -> None:
    """Drawing is what establishes the toggle's hit rectangle."""
    canvas = Image.new("RGBA", (500, 300))
    widget.draw(
        canvas,
        ImageDraw.Draw(canvas),
        frame=DisplayFrame(),
        rect=_header_rect(widget),
    )


def _toggle_click(widget: SceneHeaderWidget) -> PointerEvent:
    row_top = 2 * 36 + 10
    return PointerEvent(
        action="press", position=(20, row_top), timestamp_s=0.0, button="left"
    )


def test_the_toggle_row_appears_only_once_a_preset_is_bound() -> None:
    """The pipeline is built after the window, so the binding arrives late."""
    widget = SceneHeaderWidget(scene_label=lambda: "s", variant_label=lambda: "v")
    before = widget.measure(500)

    widget.set_postprocess_control(
        preset="upsample2x", enabled=False, callback=lambda _: None
    )

    assert widget.measure(500) > before


def test_clicking_the_toggle_reaches_the_pipeline() -> None:
    toggled: list[bool] = []
    widget = SceneHeaderWidget(scene_label=lambda: "s", variant_label=lambda: "v")
    widget.set_postprocess_control(
        preset="upsample2x", enabled=False, callback=toggled.append
    )
    _draw(widget)

    assert widget.on_pointer(_toggle_click(widget))
    assert toggled == [True]


def test_the_toggle_alternates_on_repeated_clicks() -> None:
    toggled: list[bool] = []
    widget = SceneHeaderWidget(scene_label=lambda: "s", variant_label=lambda: "v")
    widget.set_postprocess_control(
        preset="upsample2x", enabled=False, callback=toggled.append
    )

    for _ in range(2):
        _draw(widget)
        widget.on_pointer(_toggle_click(widget))

    assert toggled == [True, False]


def test_without_a_preset_there_is_nothing_to_click() -> None:
    widget = SceneHeaderWidget(scene_label=lambda: "s", variant_label=lambda: "v")
    _draw(widget)

    assert not widget.on_pointer(_toggle_click(widget))
