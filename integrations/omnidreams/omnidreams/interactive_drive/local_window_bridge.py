# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Run interactive-drive on the shared local-window presenter.

A deliberately small overlay plus the adapter between the engine's
``PresenterBackend`` and
:class:`~flashdreams.serving.presentation.LocalWindowPresenter`. The full
driving chrome still lives in :mod:`.slangpy_hud_presenter`; this path exists
so the shared presenter can be exercised end to end while that chrome is
ported across a piece at a time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from loguru import logger
from omnidreams.interactive_drive.cuda_env import DISABLE_CUDA_INTEROP_ENV, env_truthy
from omnidreams.interactive_drive.input.keyboard import KeyboardState
from omnidreams.interactive_drive.overlays import (
    BEV_OVERLAY_KEY,
    BevOverlay,
    DrivingPanelOverlay,
    PedalsOverlay,
    SpeedOverlay,
    WheelOverlay,
)
from omnidreams.interactive_drive.types import PresentedFrame
from PIL import Image, ImageDraw

from flashdreams.serving.presentation import (
    CompositeOverlay,
    DisplayFrame,
    KeyEvent,
    LocalWindowPresenter,
    PanelLayout,
    PointerEvent,
    Rect,
    WindowConfig,
    measure_text,
    resolve_font,
)

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


class MinimalDrivingOverlay:
    """Corner readout over a full-window camera.

    Intentionally not the driving HUD: no panel, speedometer, wheel, pedals,
    minimap, or scene picker. It draws only what confirms the presenter is
    live, so the shared window path can be validated before that chrome moves.
    """

    def __init__(self, keyboard: KeyboardState) -> None:
        self._keyboard = keyboard
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
            self._keyboard.set_key(event.key, pressed)
            self._held.add(event.key) if pressed else self._held.discard(event.key)
            return True
        if not pressed:
            return True
        if event.key == "r":
            self._keyboard.request_reset()
        elif event.key == "1":
            self._keyboard.set_view_mode("model_rgb")
        elif event.key == "2":
            self._keyboard.set_view_mode("rgb")
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
    ) -> None:
        self._overlay = _build_overlay(keyboard)
        self._presenter = LocalWindowPresenter(
            overlay=self._overlay,
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
        """Rebind to a new engine's keyboard across a scene switch."""
        self._overlay = _build_overlay(keyboard)

    def set_model_status(self, **kwargs: Any) -> None:
        """Accept the HUD's status wiring so demo callers stay uniform."""
        del kwargs

    def set_postprocess_control(self, **kwargs: Any) -> None:
        del kwargs


def _build_overlay(
    keyboard: KeyboardState, *, control_assets: Any | None = None
) -> CompositeOverlay:
    """Stack the chrome this demo wants over the shared presenter.

    The panel layer must come first: it reserves the column and opens the
    shared layout that every widget after it reserves rows from, in draw
    order. Widgets are added here as they move across from the legacy HUD.
    """
    layout = PanelLayout()
    drive_state = _DriveStateReader(keyboard)
    if control_assets is None:
        control_assets = _bundled_control_assets()
    return CompositeOverlay(
        layers=(
            MinimalDrivingOverlay(keyboard),
            DrivingPanelOverlay(layout),
            SpeedOverlay(layout, lambda: _current_speed_mps(keyboard)),
            WheelOverlay(layout, drive_state, control_assets=control_assets),
            PedalsOverlay(layout, drive_state, control_assets=control_assets),
            BevOverlay(layout, marker_y_fraction=_bev_marker_y_fraction),
        )
    )


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

    keyboard: KeyboardState

    def __call__(self) -> _DriveValues:
        command = self.keyboard.command()
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


def _current_speed_mps(keyboard: KeyboardState) -> float:
    """Read ego speed from the telemetry the loop publishes each chunk.

    Returns ``0.0`` before the first chunk publishes state, and after a reset
    clears it.
    """
    state = keyboard.vehicle_state
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
