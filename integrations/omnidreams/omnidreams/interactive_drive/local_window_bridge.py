# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Compatibility bridge from the retired engine frame contract to presentation.

The plug-compatible app/runtime route does not use this module. It remains for
the explicit legacy ``interactive-drive --presenter-backend local-window``
fallback until that command is removed.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from flashdreams.serving.presentation import (
    CompositeOverlay,
    DisplayFrame,
    KeyEvent,
    LocalWindowPresenter,
    PanelOverlay,
    PointerEvent,
    Rect,
    WindowConfig,
    measure_text,
    resolve_font,
)
from interactive_drive_app.input.keyboard import KeyboardState
from interactive_drive_app.overlays import (
    BEV_OVERLAY_KEY,
    MIN_CAMERA_WIDTH,
    NVIDIA_GREEN,
    PANEL_BG,
    PANEL_WIDTH,
    BevWidget,
    PedalsWidget,
    SceneHeaderWidget,
    SpeedWidget,
    TitleWidget,
    WheelWidget,
)
from loguru import logger
from omnidreams.interactive_drive.cuda_env import DISABLE_CUDA_INTEROP_ENV, env_truthy
from omnidreams.interactive_drive.types import PresentedFrame
from PIL import Image, ImageDraw

BG_COLOR: tuple[int, int, int] = (20, 20, 30)
TEXT_COLOR: tuple[int, int, int] = (220, 220, 230)
LABEL_COLOR: tuple[int, int, int] = (150, 150, 170)

_READOUT_MARGIN = 18
_READOUT_LINE_GAP = 6

_DRIVE_KEYS: frozenset[str] = frozenset(
    {"w", "a", "s", "d", "up", "down", "left", "right", "space"}
)
"""Keys forwarded to the simulation. Arrow names match ``KeyboardState``'s
lower-case vocabulary, unlike the HUD's capitalised keysyms."""


class _KeyboardSlot:
    """Indirection letting a scene switch swap the keyboard under live chrome.

    ``LocalWindowPresenter`` and its compositor capture the overlay once at
    construction, so rebuilding the overlay on a scene change would leave them
    drawing the old one. The overlay graph therefore has to outlive the switch;
    only the state it reads is replaced.
    """

    __slots__ = ("keyboard",)

    def __init__(self, keyboard: KeyboardState) -> None:
        self.keyboard = keyboard


class MinimalDrivingOverlay:
    """Corner readout over a full-window camera.

    Sits under the panel and owns the driving key bindings: it turns the
    presenter's normalized key events into ``KeyboardState`` writes, and draws
    the corner readout that confirms frames and input are arriving.
    """

    def __init__(self, slot: _KeyboardSlot) -> None:
        self._slot = slot
        self._font = resolve_font(18)
        self._frames = 0
        self._held: set[str] = set()

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
        del canvas, frame
        self._frames += 1
        held = " ".join(sorted(self._held)) or "-"
        self._draw_readout(
            draw,
            camera_area,
            (
                f"frames {self._frames}",
                f"keys {held}",
                "wasd drive | r reset | esc quit",
            ),
        )

    def draw_placeholder(
        self,
        canvas: Image.Image,
        draw: ImageDraw.ImageDraw,
        *,
        camera_area: Rect,
    ) -> None:
        del canvas
        self._draw_readout(draw, camera_area, ("waiting for frames",))

    def prepare(self, frame: DisplayFrame) -> None:
        del frame

    def on_canvas_resized(self, canvas_size: tuple[int, int]) -> None:
        del canvas_size

    def on_key(self, event: KeyEvent) -> bool:
        # Repeats are ignored: the engine reads level-triggered key state, so
        # a repeat carries no information a press has not already delivered.
        if event.action == "repeat":
            return True
        pressed = event.action == "press"
        if event.key in _DRIVE_KEYS:
            self._slot.keyboard.set_key(event.key, pressed)
            self._held.add(event.key) if pressed else self._held.discard(event.key)
            return True
        if not pressed:
            return True
        if event.key == "r":
            self._slot.keyboard.request_reset()
        elif event.key == "1":
            self._slot.keyboard.set_view_mode("model_rgb")
        elif event.key == "2":
            self._slot.keyboard.set_view_mode("rgb")
        return True

    def on_pointer(self, event: PointerEvent) -> bool:
        del event
        return False

    def close(self) -> None:
        return

    def _draw_readout(
        self, draw: ImageDraw.ImageDraw, area: Rect, lines: tuple[str, ...]
    ) -> None:
        left, top, _right, _bottom = area
        x = left + _READOUT_MARGIN
        y = top + _READOUT_MARGIN
        for index, text in enumerate(lines):
            bbox = measure_text(self._font, text)
            draw.text(
                (x - bbox[0], y - bbox[1]),
                text,
                fill=TEXT_COLOR if index == 0 else LABEL_COLOR,
                font=self._font,
            )
            y += bbox[3] - bbox[1] + _READOUT_LINE_GAP


class LocalWindowPresenterBridge:
    """Adapt the engine's presenter contract onto the shared presenter.

    The engine speaks ``PresentedFrame`` plus a view mode; the shared
    presenter speaks :class:`DisplayFrame`. Selecting which rendered source
    to show is the translation this performs.
    """

    def __init__(
        self,
        keyboard: KeyboardState,
        *,
        width: int = 1920,
        height: int = 1080,
        title: str = "interactive-drive (local-window)",
        scene_label: str = "Scene",
        variant_label: str = "default",
    ) -> None:
        self._slot = _KeyboardSlot(keyboard)
        self._header = SceneHeaderWidget(
            scene_label=lambda: scene_label,
            variant_label=lambda: variant_label,
        )
        self._presenter = LocalWindowPresenter(
            overlay=_build_overlay(
                self._slot,
                header=self._header,
            ),
            config=WindowConfig(
                width=width,
                height=height,
                title=title,
                background=BG_COLOR,
                text_color=TEXT_COLOR,
            ),
            # Honour the same opt-out the rasterizer and the legacy presenters
            # read, so one variable still forces the whole demo onto host paths.
            cuda_interop_disabled=env_truthy(DISABLE_CUDA_INTEROP_ENV),
        )

    @property
    def should_close(self) -> bool:
        return self._presenter.should_close

    def process_events(self) -> None:
        self._presenter.process_events()

    def prepare_frame(self, frame: PresentedFrame, view_mode: str) -> None:
        self._presenter.prepare_frame(_display_frame(frame, view_mode))

    def present_frame(self, frame: PresentedFrame, view_mode: str) -> None:
        self._presenter.present_frame(_display_frame(frame, view_mode))

    def close(self) -> None:
        self._presenter.close()

    def bind_keyboard(self, keyboard: KeyboardState) -> None:
        """Point the live chrome at a new engine's keyboard.

        The demo builds this presenter against a placeholder keyboard and then
        hands over the engine's real one, and does so again on every scene
        switch. Swapping the slot rather than the overlay is what makes that
        work, since the presenter already captured the overlay.
        """
        self._slot.keyboard = keyboard

    def set_model_status(self, **kwargs: Any) -> None:
        """Accept the HUD's status wiring so demo callers stay uniform."""
        del kwargs

    def set_postprocess_control(
        self,
        *,
        preset: str,
        enabled: bool,
        callback: Callable[[bool], None],
    ) -> None:
        """Bind the panel's post-process toggle to the model pipeline."""
        self._header.set_postprocess_control(
            preset=preset, enabled=enabled, callback=callback
        )


def _build_overlay(
    slot: _KeyboardSlot,
    *,
    header: SceneHeaderWidget | None = None,
    control_assets: Any | None = None,
) -> CompositeOverlay:
    """Stack the chrome this demo wants over the shared presenter.

    The panel owns its widgets and lays them out top to bottom in the order
    listed here, so reordering the column is a matter of reordering this
    tuple. Widgets are added as they move across from the legacy HUD.
    """
    drive_state = _DriveStateReader(slot)
    if control_assets is None:
        control_assets = _bundled_control_assets()
    panel = PanelOverlay(
        width=PANEL_WIDTH,
        min_camera_width=MIN_CAMERA_WIDTH,
        background=PANEL_BG,
        accent=NVIDIA_GREEN,
        children=(
            TitleWidget(),
            header
            if header is not None
            else SceneHeaderWidget(
                scene_label=lambda: "Scene", variant_label=lambda: "default"
            ),
            SpeedWidget(lambda: _current_speed_mps(slot)),
            WheelWidget(drive_state, control_assets=control_assets),
            PedalsWidget(drive_state, control_assets=control_assets),
            BevWidget(
                marker_y_fraction=_bev_marker_y_fraction,
                recolor=_bev_recolor(),
            ),
        ),
    )
    return CompositeOverlay(layers=(MinimalDrivingOverlay(slot), panel))


def _bundled_control_assets() -> Any | None:
    """Load the wheel and pedal sprites that ship with the demo.

    Late import: ``demo`` reaches this module through the presenter factory,
    so a top-level import would be circular. Returns ``None`` when the assets
    cannot be loaded, which leaves the widgets on their drawn fallbacks.
    """
    try:
        from omnidreams.interactive_drive.demo import _load_control_assets

        return _load_control_assets(None)
    except Exception as exc:  # noqa: BLE001 -- chrome sprites are optional
        logger.info(f"[presenter] control assets unavailable; using fallbacks ({exc})")
        return None


@dataclass(frozen=True, slots=True)
class _DriveStateReader:
    """Adapt ``KeyboardState``'s command to the chrome's drive-state shape.

    The legacy HUD read a wheel object with ``steering`` / ``throttle`` /
    ``brake``; the keyboard path exposes the same values under a
    ``DriverCommand``'s ``steer``. Normalizing here keeps the overlays free
    of either representation.
    """

    slot: _KeyboardSlot

    def __call__(self) -> _DriveValues:
        command = self.slot.keyboard.command()
        return _DriveValues(
            steering=float(command.steer),
            throttle=float(command.throttle),
            brake=float(command.brake),
        )


@dataclass(frozen=True, slots=True)
class _DriveValues:
    steering: float
    throttle: float
    brake: float


def _current_speed_mps(slot: _KeyboardSlot) -> float:
    """Read ego speed from the telemetry the loop publishes each chunk.

    Returns ``0.0`` before the first chunk publishes state, and after a reset
    clears it.
    """
    state = slot.keyboard.vehicle_state
    return 0.0 if state is None else float(state.speed_mps)


def _display_frame(frame: PresentedFrame, view_mode: str) -> DisplayFrame:
    """Project an engine frame onto the presenter's display contract.

    Only a world-model frame has a native resolution worth growing the window
    for; the raster view is already rendered at the window's own resolution.
    """
    model_rgb = frame.model_rgb_host_uint8
    show_model = view_mode == "model_rgb" and model_rgb is not None
    return DisplayFrame(
        image=model_rgb if show_model else frame.rgb_host_uint8,
        timestamp_us=frame.timestamp_us,
        status_message=frame.status_message,
        allow_window_resize=show_model,
        overlay_data={BEV_OVERLAY_KEY: frame.bev_host_uint8},
    )


def _bev_recolor() -> Callable[[Image.Image], Image.Image] | None:
    """The map palette the legacy minimap uses.

    Returns ``None`` when unavailable, which leaves the minimap in the
    renderer's raw colours rather than failing the whole overlay.
    """
    try:
        from omnidreams.interactive_drive.demo import _apply_googlemaps_filter

        return _apply_googlemaps_filter
    except Exception as exc:  # noqa: BLE001 -- recolouring is cosmetic
        logger.info(f"[presenter] BEV recolour unavailable ({exc})")
        return None


def _bev_marker_y_fraction() -> float:
    """Where the ego chevron sits vertically in the minimap.

    The BEV camera looks ahead of the vehicle, so the ego is below centre.
    Late import for the same circularity reason as the control assets.
    """
    try:
        from omnidreams.interactive_drive.demo import _bev_marker_y_rel

        return float(_bev_marker_y_rel())
    except Exception:  # noqa: BLE001 -- centre is a reasonable default
        return 0.5


__all__ = [
    "LocalWindowPresenterBridge",
    "MinimalDrivingOverlay",
]
