# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Single-process slangpy-window HUD presenter for ``interactive-drive``.

Plugs into the same engine seam as ``SlangPyPresenter`` (``--no-hud``), but
draws PIL chrome (panel, dropdowns, BEV minimap, speed/wheel/pedals) over the
camera frame -- composited on CUDA when interop is available, else on the CPU.
Input goes straight to ``KeyboardState``; dropdown scene/variant changes are
handled by the demo's outer loop over this same long-lived window.
"""

from __future__ import annotations

import contextlib
import math as _math
import time
from collections import OrderedDict
from collections.abc import Callable
from typing import Any

import numpy as np
from loguru import logger
from omnidreams.interactive_drive.config import RasterConfig
from omnidreams.interactive_drive.cuda_env import DISABLE_CUDA_INTEROP_ENV
from omnidreams.interactive_drive.input.keyboard import KeyboardState
from omnidreams.interactive_drive.presenter import (
    _CudaRGBInterop,
    _env_truthy,
)
from omnidreams.interactive_drive.types import DriverCommand, PresentedFrame
from PIL import Image, ImageDraw, ImageFont

# Colour palette mirrors :mod:`omnidreams.interactive_drive.demo` for a
# consistent visual identity.
NVIDIA_GREEN: tuple[int, int, int] = (118, 185, 0)
BG_COLOR: tuple[int, int, int] = (20, 20, 30)
PANEL_BG: tuple[int, int, int] = (25, 25, 35)
TEXT_COLOR: tuple[int, int, int] = (220, 220, 230)
LABEL_COLOR: tuple[int, int, int] = (150, 150, 170)
HEADER_BG: tuple[int, int, int] = (35, 35, 50)
HOVER_BG: tuple[int, int, int] = (50, 60, 80)
ACTIVE_BG: tuple[int, int, int] = (30, 80, 30)
ACCENT_AMBER: tuple[int, int, int] = (200, 150, 50)
GMAPS_LAND_RGB: tuple[int, int, int] = (234, 226, 209)

# Initial windowed dimensions and minimum size.
DEFAULT_WINDOW_W = 1920
DEFAULT_WINDOW_H = 1080
MIN_WINDOW_W = 640
MIN_WINDOW_H = 360
HUD_PANEL_WIDTH = 500

# BEV minimap geometry (in panel-local pixels).
BEV_PANEL_TOP_GAP = 12
BEV_PANEL_SIDE_MARGIN = 14
BEV_PANEL_BOTTOM_MARGIN = 12
BEV_PANEL_MIN_HEIGHT = 100

# Quantisation buckets for the steering-wheel rotation cache. ±450° / 3°
# = 300 buckets in the worst case; cached PIL images are small (radius
# ~120 px) so the memory cost is negligible and we save a 2 ms
# Image.rotate per render tick.
WHEEL_ROTATION_QUANTUM_DEG = 3

# Render loop sleep target between event polls; 5 ms keeps input latency
# low without burning a core.
EVENT_POLL_INTERVAL_S = 0.005

# Metres-per-second to miles-per-hour, for the speed digit.
MPS_TO_MPH = 2.2369362920544

# Drive-key release debounce window. See the
# ``_pending_drive_releases`` field documentation in
# :class:`SlangPyHudPresenter`.
DRIVE_KEY_RELEASE_DEBOUNCE_S = 0.08


def _allocate_canvas(width: int, height: int) -> tuple[np.ndarray, Image.Image]:
    """Allocate the chrome buffer and a PIL Image view sharing its memory.

    ``Image.frombuffer`` (RGBA "raw", Pillow >= 9) aliases ``buf``, so PIL
    draws write into it directly and we can hand ``buf`` straight to slangpy's
    ``copy_from_numpy`` with no PIL-to-numpy memcpy. ``readonly = 0`` is
    required or ``ImageDraw`` rejects the image as a draw target.
    """
    buf = np.empty((height, width, 4), dtype=np.uint8)
    buf[..., :3] = BG_COLOR
    buf[..., 3] = 255
    img = Image.frombuffer("RGBA", (width, height), buf, "raw", "RGBA", 0, 1)
    img.readonly = 0
    return buf, img


class _LRUCache(OrderedDict):
    """Tiny ordered-dict-backed LRU.

    Used for the speed-digit / wheel-rotation / pedal-sprite caches so
    the per-bucket render artefacts don't pile up forever. The OrderedDict
    move-to-end on every ``get`` keeps the LRU semantics correct.
    """

    def __init__(self, maxsize: int) -> None:
        super().__init__()
        self._maxsize = int(maxsize)

    def get_or_compute(self, key: Any, build: Any) -> Any:
        existing = self.get(key)
        if existing is not None:
            self.move_to_end(key)
            return existing
        value = build()
        self[key] = value
        if len(self) > self._maxsize:
            self.popitem(last=False)
        return value


def _resolve_font(size: int) -> Any:
    """Find a host TrueType font (DejaVu Sans / Arial / Segoe UI), else PIL's default.

    PIL has no sysfont resolver, so we probe well-known paths;
    ``ImageFont.load_default`` is the last-resort bitmap fallback.
    """
    candidates = (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
        "/Library/Fonts/Arial.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
    )
    for path in candidates:
        try:
            return ImageFont.truetype(path, size=size)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def _measure_text(font: Any, text: str) -> tuple[int, int, int, int]:
    """Wrapper for :meth:`ImageFont.FreeTypeFont.getbbox` that handles legacy bitmap fallback."""
    if hasattr(font, "getbbox"):
        bbox = font.getbbox(text)
        return (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
    # The 9.x-era bitmap fallback only has ``getsize``.
    width, height = font.getsize(text)  # type: ignore[attr-defined]
    return (0, 0, int(width), int(height))


def _truncate_text_to_width(
    font: Any, text: str, max_width: int, ellipsis: str = "\u2026"
) -> str:
    """Shrink ``text`` (with a trailing ``…``) until it fits ``max_width`` pixels.

    PIL doesn't auto-clip ``ImageDraw.text``, so a long label would overflow
    the panel; we measure progressively shorter prefixes until one fits.
    """
    bbox = _measure_text(font, text)
    if bbox[2] - bbox[0] <= max_width:
        return text
    # Greedy shrink. The header is short (a UUID + label), so the
    # quadratic cost of re-measuring on every truncation is fine.
    for end in range(len(text), 0, -1):
        candidate = text[:end] + ellipsis
        cb = _measure_text(font, candidate)
        if cb[2] - cb[0] <= max_width:
            return candidate
    return ellipsis


class KeyboardStateDriveSink:
    """Duck-typed control sink that writes drive commands straight to ``KeyboardState``.

    Consumed by :class:`~omnidreams.interactive_drive.demo.WheelBridge` and
    :class:`~omnidreams.interactive_drive.demo.KeyboardDriveState`. ``set_key`` /
    ``pulse`` are unused no-ops kept only so the full control surface exists.
    """

    def __init__(self, keyboard: KeyboardState) -> None:
        self._keyboard = keyboard

    def set_drive(
        self, *, steer: float, throttle: float, brake: float, reverse: bool = False
    ) -> None:
        # ``manual_control`` + ``steer_is_direct`` keep the engine state
        # identical regardless of transport. ``reverse`` is set by a
        # wheel/controller's bound reverse button (keyboard leaves it False).
        self._keyboard.set_drive_command(
            DriverCommand(
                throttle=max(0.0, min(1.0, throttle)),
                brake=max(0.0, min(1.0, brake)),
                steer=max(-1.0, min(1.0, steer)),
                reverse=bool(reverse),
                steer_is_direct=True,
                manual_control=True,
            )
        )

    def release_all(self) -> None:
        self._keyboard.set_drive_command(None)

    def request_reset(self) -> None:
        # Lets a wheel/controller's bound reset button trigger the same
        # rollout reset the ``R`` key does.
        self._keyboard.request_reset()

    def request_exit_scene(self) -> None:
        # Lets a wheel/controller's bound exit button drop back to the scene
        # selector, the same as the ``X`` key. The presenter drains this on
        # its event-pump thread and converts it into an exit-to-selection.
        self._keyboard.request_exit_scene()

    # No-ops in-process: the HUD writes key events directly to ``KeyboardState``
    # from its ``on_keyboard_event`` callback. Kept so the full control surface
    # exists for duck-typed callers.
    def set_key(self, key: str, down: bool) -> None:  # noqa: ARG002 -- unused in-process
        return

    def pulse(self, key: str) -> None:  # noqa: ARG002 -- unused in-process
        return

    def stop(self) -> None:
        return


class SlangPyHudPresenter:
    """Single-process slangpy-window HUD with PIL-rendered chrome.

    Implements the ``PresenterBackend`` Protocol that
    :class:`~omnidreams.interactive_drive.app.InteractiveDriveApp` expects. Owns
    a :class:`slangpy.Window` (the same SDL3-backed Vulkan swapchain
    ``--no-hud`` uses), a CPU-side PIL canvas where chrome is composited
    with the camera frame, the input event handlers, and the
    sprite/font/panel caches.
    """

    def __init__(
        self,
        raster: RasterConfig,
        keyboard: KeyboardState,
        *,
        args: Any,
        scene_options: tuple[Any, ...],
        control_assets: Any,
        wheel: Any | None,
    ) -> None:
        try:
            import slangpy as spy
        except ImportError as exc:
            raise RuntimeError(
                "SlangPy is required for the interactive-drive HUD;"
                " install with `uv sync --package flashdreams-omnidreams --extra interactive-drive`."
            ) from exc

        self._spy = spy
        self._raster = raster
        self._keyboard = keyboard
        self._args = args
        self._scene_options = scene_options
        self._control_assets = control_assets
        self._wheel = wheel

        # Late-imports of helpers we need at runtime; ``demo`` imports
        # this module via the presenter factory, so direct top-level
        # imports would be circular.
        from omnidreams.interactive_drive.demo import (
            KeyboardDriveState,
            _bev_marker_y_rel,
            _scene_label,
        )

        self._keyboard_drive = KeyboardDriveState(KeyboardStateDriveSink(keyboard))
        self._bev_marker_y_rel = _bev_marker_y_rel
        self._scene_label_fn = _scene_label

        # Window + device + surface setup mirrors SlangPyPresenter's
        # but with a resizable HUD-sized window and a display texture
        # we re-create on resize.
        self._cuda_interop_unavailable_reason: str | None = None
        self._cuda_hud_error_logged = False
        self._window = spy.Window(
            width=DEFAULT_WINDOW_W,
            height=DEFAULT_WINDOW_H,
            title="interactive-drive HUD",
            resizable=True,
        )
        self._device = self._create_device()
        logger.info(f"[presenter] device={self._device.info.adapter_name}")
        self._surface = self._device.create_surface(self._window)
        self._surface_format = self._choose_surface_format()
        self._display_format = spy.Format.rgba8_unorm
        logger.info(
            f"[presenter] surface preferred={self._surface.info.preferred_format}"
            f" chosen={self._surface_format} display={self._display_format}",
        )
        # Trust the ACTUAL window size after creation rather than the
        # requested defaults: SDL3 may clamp the window down to fit the
        # display (or scale for HiDPI), and configuring a surface with
        # the wrong size makes ``acquireNextImage`` fail at first
        # present with a generic SLANG_FAIL. ``window.size`` is
        # ``math.uint2``, indexed like a 2-vector.
        self._configured_size = self._current_window_size()
        self._configure_surface(*self._configured_size)
        self._display_texture = self._build_display_texture(*self._configured_size)
        self._cuda_hud_interop = self._create_cuda_hud_interop(*self._configured_size)
        self._retired_cuda_hud_interops: list[Any] = []
        self._cuda_hud_resize_logged = False
        # ``_pending_resize`` is set by the on_resize callback (which
        # runs on the windowing thread) and consumed by ``present_frame``
        # on the main thread, where it's safe to recreate Vulkan
        # resources.
        self._pending_resize: tuple[int, int] | None = None
        self._window.on_resize = self._on_resize
        self._window.on_keyboard_event = self._on_keyboard_event
        self._window.on_mouse_event = self._on_mouse_event

        self._font_tiny = _resolve_font(14)
        self._font_small = _resolve_font(18)
        self._font_medium = _resolve_font(22)
        self._font_large = _resolve_font(44)
        self._font_speed = _resolve_font(76)

        self._panel_chrome_cache_key: tuple[Any, ...] | None = None
        self._panel_chrome_cache: Image.Image | None = None
        self._speed_chip_cache: _LRUCache = _LRUCache(maxsize=64)
        self._wheel_base_image: Image.Image | None = None
        self._wheel_base_size: int | None = None
        self._wheel_rotation_cache: _LRUCache = _LRUCache(maxsize=480)
        self._pedal_cache: _LRUCache = _LRUCache(maxsize=16)
        self._scene_thumb_cache: dict[Any, Image.Image | None] = {}
        self._variant_thumb_cache: dict[tuple[Any, str], Image.Image | None] = {}
        self._bev_panel_cache_key: tuple[int, int, int] | None = None
        self._bev_panel_cache: Image.Image | None = None

        self._latest_camera_pil: Image.Image | None = None
        self._latest_bev_pil: Image.Image | None = None
        # Numpy view of the latest world-model frame (RGBA8 with alpha
        # padded to 255) used by the GPU camera path. Lazily filled on
        # demand from ``_latest_camera_pil`` so we don't pay for the
        # RGB->RGBA expansion on warmup ticks that take the CPU
        # fallback path anyway.
        self._latest_camera_rgba: np.ndarray | None = None
        self._latest_camera_src_size: tuple[int, int] | None = None  # (w, h)
        self._camera_resize_cache_key: tuple[int, int, int] | None = None
        self._camera_resize_cache: Image.Image | None = None

        # GPU camera path: world-model frames upload into a source-sized
        # texture, get GPU-scaled into a fit-sized texture via
        # ``encoder.blit`` (full-extent linear filter == hardware
        # bilinear resize, ~0.1 ms vs the ~5 ms PIL ``Image.resize``
        # we used to pay on the CPU), and finally copy into a centred
        # rectangle inside the display texture via
        # ``encoder.copy_texture``. Skipped (CPU fallback) only when
        # ``status_message`` is set so the warmup "Loading world
        # model..." overlay still composites over the loading frame.
        self._camera_texture: Any | None = None
        self._camera_texture_size: tuple[int, int] | None = None
        self._camera_fit_texture: Any | None = None
        self._camera_fit_size: tuple[int, int] | None = None
        # Pre-allocated RGBA staging buffer used by the GPU camera
        # upload. See :meth:`_ensure_camera_texture_uploaded` for the
        # rationale; in short, we reuse one ``(src_h, src_w, 4)``
        # numpy buffer with the alpha channel pre-filled to 255 so
        # the per-tick work is a single RGB slice copy instead of
        # an alpha alloc + ``np.concatenate`` + redundant
        # ``ascontiguousarray``.
        self._camera_rgba_staging: np.ndarray | None = None

        # Numpy-backed RGBA canvas: PIL writes into the same buffer
        # slangpy uploads to per frame. See :func:`_allocate_canvas`.
        self._canvas_buffer, self._canvas = _allocate_canvas(*self._configured_size)

        self._scene_dropdown_open = False
        self._variant_dropdown_open = False
        self._scene_header_rect: tuple[int, int, int, int] | None = None
        self._variant_header_rect: tuple[int, int, int, int] | None = None
        self._scene_item_rects: list[tuple[tuple[int, int, int, int], Any]] = []
        self._variant_item_rects: list[tuple[tuple[int, int, int, int], str]] = []
        self._hovered_scene_label: str | None = None
        self._hovered_variant: str | None = None
        self._mouse_pos: tuple[int, int] = (0, 0)
        self._speed_mph: float = 0.0
        self._is_fullscreen = False
        self._should_close_flag = False

        self._current_scene = args.scene
        self._selected_variant = args.variant
        self._has_camera_frame = False
        # ``_engine_active`` is False during the initial scene-selection
        # wait (when the user hasn't picked a scene yet AND
        # ``--auto-start`` was off) and during the brief gap between
        # scene changes. Drives the camera-area placeholder text together
        # with the model-warmup state below. Toggled by the demo wrapper
        # via :meth:`set_engine_active` around each scene's run.
        self._engine_active = False
        # Model-warmup status, wired by the demo via :meth:`set_model_status`.
        # ``_model_can_prewarm`` is True when the model loads at startup
        # (so the selection wait shows "Loading world model..." instead of
        # "Load Scene"); ``_model_ready_probe`` returns True once warmup
        # has finished. Defaults are inert so a presenter used without the
        # wiring (or before it) behaves like the old "Load Scene" prompt.
        self._model_can_prewarm = False
        self._model_ready_probe: Callable[[], bool] = lambda: True
        # Scene-selection lock, wired by the demo via
        # :meth:`set_scene_selection_locked` when --preload-scenes is on.
        # While the probe returns True the scene/variant dropdowns ignore
        # clicks and the placeholder shows a "Preloading scenes..." hint, so
        # the user can't pick a scene until every scene is cached.
        self._scene_selection_locked_probe: Callable[[], bool] = lambda: False

        # Scene-change request set by the dropdown click handlers. The
        # outer demo loop checks this after each ``app.run_scene`` returns:
        # if non-None, it calls ``app.load_scene`` for the requested scene
        # and re-enters the engine over the SAME presenter so the slangpy
        # window (and the warmed model) stay alive.
        self._pending_scene_change: tuple[Any, str] | None = None
        # Exit-to-selection request set by the ``x`` key or a wheel's bound
        # exit button. The outer demo loop checks this (ahead of
        # ``pending_scene_change``) after each ``app.run_scene`` returns: when
        # set it tears down the rollout and re-enters the scene selector over
        # the SAME presenter, so a long-running demo can stop the video model
        # generating without closing the window or reloading the model.
        self._pending_exit_scene = False

        self._key_codes = self._build_key_codes()
        # Drive-key release debounce. Some SDL3 builds send a
        # ``release + press`` cycle for OS-level key repeats instead of
        # the dedicated ``key_repeat`` event we filter out, which made
        # ``KeyboardDriveState`` toggle the key state off and on at the
        # OS repeat rate (~30 Hz) and produced visible steering jitter
        # while the user was actually still holding the key. We defer
        # release calls by ``DRIVE_KEY_RELEASE_DEBOUNCE_S`` so a fresh
        # press / repeat within that window cancels the release; real
        # releases incur an 80 ms delay before the wheel starts
        # returning, which is below conscious latency.
        self._pending_drive_releases: dict[str, float] = {}

    # -- PresenterBackend protocol ---------------------------------

    @property
    def should_close(self) -> bool:
        return self._should_close_flag or self._window.should_close()

    def process_events(self) -> None:
        self._window.process_events()
        # A wheel/controller's bound exit button posts its request onto the
        # shared ``KeyboardState`` from the wheel reader thread; drain it here
        # on the main thread and convert it into the presenter's own
        # exit-to-selection signal (the ``x`` key takes the direct path in
        # ``_on_keyboard_event``).
        if self._keyboard.consume_exit_scene_request():
            self.exit_scene()

    def prepare_frame(self, frame: PresentedFrame, view_mode: str) -> None:
        rgb = self._select_view_rgb(frame, view_mode)
        if self._cuda_hud_interop is None or not _has_cuda_tensor(rgb):
            _prefetch_to_numpy(rgb)
        if frame.bev_host_uint8 is not None:
            _prefetch_to_numpy(frame.bev_host_uint8)

    def present_frame(self, frame: PresentedFrame, view_mode: str) -> None:
        # Apply any pending resize before touching the display texture
        # this frame. Done here (not inside on_resize) so Vulkan
        # resources are only ever rebuilt on the main thread.
        if self._pending_resize is not None:
            new_size = self._pending_resize
            self._pending_resize = None
            self._apply_resize(new_size[0], new_size[1])

        rgb = self._select_view_rgb(frame, view_mode)
        try:
            if self._present_cuda_hud_frame(frame, rgb):
                return
        except Exception as exc:
            if not self._cuda_hud_error_logged:
                logger.warning(
                    "[presenter] hud_cuda_interop=failed; disabling and using "
                    f"host HUD upload ({exc})",
                )
                self._cuda_hud_error_logged = True
            if self._cuda_hud_interop is not None:
                with contextlib.suppress(Exception):
                    self._cuda_hud_interop.close()
                self._cuda_hud_interop = None
        self._update_camera_pil(rgb)
        if frame.bev_host_uint8 is not None:
            self._update_bev_pil(frame.bev_host_uint8)
        self._render_canvas(frame.status_message)
        self._present_canvas(use_gpu_camera=frame.status_message is None)

    def present_world_model_loading(self, *, process_events: bool = True) -> None:
        """Paint the HUD's world-model loading state during blocking setup work."""
        if process_events:
            self.process_events()
        self.set_engine_active(True)
        self._render_canvas("Loading World Model")
        self._present_canvas(use_gpu_camera=False)

    def _present_cuda_hud_frame(self, frame: PresentedFrame, rgb: object) -> bool:
        if self._cuda_hud_interop is None:
            return False

        cuda_frame = self._cuda_hud_interop.as_cuda_rgb_source(rgb)
        if cuda_frame is None:
            return False

        if not cuda_frame.ready:
            self._submit_ready_cuda_hud()
            return True

        if frame.bev_host_uint8 is not None:
            self._update_bev_pil(frame.bev_host_uint8)
        self._has_camera_frame = True
        self._render_canvas(frame.status_message, camera_transparent=True)
        overlay = np.array(self._canvas, dtype=np.uint8)
        camera_area, _panel_rect = self._layout_regions()

        submitted = self._submit_ready_cuda_hud()
        queued = self._cuda_hud_interop.enqueue_camera_to_shared_rgba(
            cuda_frame,
            overlay_rgba=overlay,
            camera_area=camera_area,
            bg_rgb=BG_COLOR,
        )
        if not queued:
            return True
        if not submitted:
            self._submit_ready_cuda_hud()
        return True

    def close(self) -> None:
        self._should_close_flag = True
        if self._cuda_hud_interop is not None:
            with contextlib.suppress(Exception):
                self._cuda_hud_interop.close()
            self._cuda_hud_interop = None
        retired_interops = getattr(self, "_retired_cuda_hud_interops", [])
        for interop in retired_interops:
            with contextlib.suppress(Exception):
                interop.close()
        retired_interops.clear()
        if self._wheel is not None:
            try:
                self._wheel.stop()
            except Exception as exc:  # noqa: BLE001 -- defensive teardown
                logger.warning(f"[presenter] wheel.stop() failed: {exc!r}")
            self._wheel = None
        with contextlib.suppress(Exception):
            self._window.close()

    # -- Frame helpers ---------------------------------------------

    @staticmethod
    def _select_view_rgb(frame: PresentedFrame, view_mode: str) -> object:
        if view_mode == "model_rgb" and frame.model_rgb_host_uint8 is not None:
            return frame.model_rgb_host_uint8
        return frame.rgb_host_uint8

    def _update_camera_pil(self, rgb: object) -> None:
        rgb = _as_rgb_host_uint8(rgb)
        # ``Image.fromarray`` over a contiguous numpy buffer is zero-copy
        # at the C level (PIL keeps a buffer-protocol reference). The
        # resulting Image's ``.tobytes()`` would copy, but we only ever
        # use this image as a paste source which doesn't trigger a copy.
        if not rgb.flags["C_CONTIGUOUS"]:
            rgb = np.ascontiguousarray(rgb)
        self._latest_camera_pil = Image.fromarray(rgb, mode="RGB")
        # Source dimensions for the GPU camera path. ``slangpy.Texture``
        # uploads need RGBA; the chunk pipeline produces RGB, so we
        # expand to RGBA lazily in :meth:`_ensure_camera_texture_uploaded`
        # only on ticks that actually take the GPU path.
        src_h, src_w = rgb.shape[:2]
        self._latest_camera_src_size = (src_w, src_h)
        # Force re-upload of the GPU camera texture (the chunk pipeline
        # reuses its scratch buffer, so ``id(rgb)`` is stable across
        # frames with different contents). Clearing the cached RGBA
        # expansion forces a fresh ``np.dstack`` / ``copy_from_numpy``
        # on the next ``_ensure_camera_texture_uploaded`` call.
        self._latest_camera_rgba = None
        # Invalidate the CPU resize cache: same buffer reuse story
        # applies to the PIL fallback path.
        self._camera_resize_cache_key = None
        self._camera_resize_cache = None
        self._has_camera_frame = True

    def _update_bev_pil(self, bev_rgb: object) -> None:
        bev_rgb = _as_rgb_host_uint8(bev_rgb)
        # Wrap the raw BEV; the GoogleMaps recolour runs in
        # :meth:`_get_bev_panel_image` *after* the panel-sized resize so the
        # float32 pipeline processes ~0.22 MP instead of 1 MP.
        if not bev_rgb.flags["C_CONTIGUOUS"]:
            bev_rgb = np.ascontiguousarray(bev_rgb)
        try:
            self._latest_bev_pil = Image.fromarray(bev_rgb, mode="RGB")
        except (ValueError, OSError):
            return
        self._bev_panel_cache_key = None
        self._bev_panel_cache = None

    # -- Vulkan / surface plumbing ---------------------------------

    def _create_device(self) -> Any:
        existing_device_handles = self._cuda_existing_device_handles()
        enable_cuda_interop = not _env_truthy(DISABLE_CUDA_INTEROP_ENV)
        if not enable_cuda_interop:
            self._cuda_interop_unavailable_reason = (
                f"disabled by {DISABLE_CUDA_INTEROP_ENV}"
            )
        device_kwargs = {
            "type": self._spy.DeviceType.vulkan,
            "enable_debug_layers": False,
            "enable_cuda_interop": enable_cuda_interop,
            "enable_cuda_launch_from_gfx": False,
            "enable_ray_tracing": False,
        }
        if existing_device_handles:
            device_kwargs["existing_device_handles"] = existing_device_handles
        try:
            return self._spy.Device(**device_kwargs)
        except RuntimeError as exc:
            logger.warning(
                "[presenter] CUDA interop device creation failed; retrying Vulkan without "
                f"interop ({exc})",
            )
            self._cuda_interop_unavailable_reason = "device creation failed"
            return self._spy.Device(
                type=self._spy.DeviceType.vulkan,
                enable_debug_layers=False,
                enable_cuda_launch_from_gfx=False,
                enable_ray_tracing=False,
            )

    def _cuda_existing_device_handles(self) -> list[Any]:
        if _env_truthy(DISABLE_CUDA_INTEROP_ENV):
            return []
        try:
            import torch
        except ImportError:
            return []
        try:
            if not torch.cuda.is_initialized():
                return []
        except Exception:
            return []

        get_handles = getattr(
            self._spy, "get_cuda_current_context_native_handles", None
        )
        if not callable(get_handles):
            return []
        try:
            handles: Any = get_handles()
            return list(handles)
        except Exception:
            return []

    def _create_cuda_hud_interop(
        self, width: int, height: int
    ) -> _CudaRGBInterop | None:
        if _env_truthy(DISABLE_CUDA_INTEROP_ENV):
            logger.info(
                "[presenter] hud_cuda_interop=disabled by "
                f"{DISABLE_CUDA_INTEROP_ENV}; using host HUD upload",
            )
            return None
        if not self._device.supports_cuda_interop:
            reason = self._cuda_interop_unavailable_reason or "unsupported"
            logger.info(
                f"[presenter] hud_cuda_interop={reason}; using host HUD upload",
            )
            return None
        try:
            interop = _CudaRGBInterop(
                spy=self._spy,
                device=self._device,
                width=width,
                height=height,
            )
        except Exception as exc:
            logger.warning(
                f"[presenter] hud_cuda_interop=unavailable; using host HUD upload ({exc})",
            )
            return None
        logger.info("[presenter] hud_cuda_interop=enabled")
        return interop

    def _choose_surface_format(self) -> Any:
        """Pick a linear surface format (no implicit sRGB encode).

        Identical to :class:`SlangPyPresenter._choose_surface_format`.
        Mismatched gamma between display texture and swapchain causes
        washed-out colours, so we explicitly pick a linear format that
        the surface advertises support for.
        """
        spy = self._spy
        linear_pairs = {
            spy.Format.rgba8_unorm_srgb: spy.Format.rgba8_unorm,
            spy.Format.bgra8_unorm_srgb: spy.Format.bgra8_unorm,
            spy.Format.bgrx8_unorm_srgb: spy.Format.bgrx8_unorm,
        }
        preferred = self._surface.info.preferred_format
        supported = list(self._surface.info.formats)
        for candidate in (
            spy.Format.rgba8_unorm,
            spy.Format.bgra8_unorm,
            spy.Format.bgrx8_unorm,
        ):
            if candidate in supported:
                return candidate
        preferred_linear = linear_pairs.get(preferred, preferred)
        if preferred_linear in supported:
            return preferred_linear
        raise RuntimeError(
            f"Presenter requires a linear swapchain, but the surface only supports: {supported}"
        )

    def _configure_surface(self, width: int, height: int) -> None:
        self._surface.configure(width=width, height=height, format=self._surface_format)

    def _build_display_texture(self, width: int, height: int) -> Any:
        spy = self._spy
        return self._device.create_texture(
            format=self._display_format,
            width=width,
            height=height,
            usage=(
                spy.TextureUsage.shader_resource
                | spy.TextureUsage.unordered_access
                | spy.TextureUsage.copy_destination
            ),
            label="hud_display_texture",
        )

    def _apply_resize(self, width: int, height: int, *, force: bool = False) -> bool:
        width, height = self._normalise_present_size(width, height)
        previous_size = self._configured_size
        size_changed = (width, height) != previous_size
        if not force and not size_changed:
            return True
        try:
            display_texture = self._build_display_texture(width, height)
            canvas_buffer, canvas = _allocate_canvas(width, height)
            self._configure_surface(width, height)
        except Exception as exc:
            logger.warning(
                "[presenter] window resize failed; keeping previous presenter "
                f"texture size {previous_size} ({exc})",
            )
            return False
        self._configured_size = (width, height)
        # Re-create only presenter-owned display resources. The world-model
        # raster/inference resolution stays fixed by AppConfig/manifest; this
        # texture is only the final HUD swapchain upload target.
        self._display_texture = display_texture
        if size_changed:
            self._recreate_cuda_hud_interop_after_resize(width, height)
        # Drop the chrome panel cache (its size depends on screen size)
        # and reallocate the canvas. Other caches are size-independent.
        self._panel_chrome_cache_key = None
        self._panel_chrome_cache = None
        self._bev_panel_cache_key = None
        self._bev_panel_cache = None
        self._wheel_rotation_cache.clear()
        self._pedal_cache.clear()
        # Camera fit-texture's size is derived from the camera area in
        # the resized display, so it needs to be re-built next frame.
        # The source-sized camera_texture only depends on world-model
        # output dims, so it stays valid across window resizes.
        self._camera_fit_texture = None
        self._camera_fit_size = None
        self._canvas_buffer, self._canvas = canvas_buffer, canvas
        return True

    def _on_resize(self, width: int, height: int) -> None:
        # Stash the new dimensions; ``present_frame`` recreates Vulkan
        # resources on the next tick. Doing it in the callback would
        # race with whatever frame is in flight.
        self._pending_resize = self._normalise_present_size(width, height)

    def _submit_ready_cuda_hud(self) -> bool:
        interop = self._cuda_hud_interop
        if interop is None:
            return False
        interop_frame = interop.ready_rgba_buffer()
        if interop_frame is None:
            return False
        rgba_buffer, cuda_stream = interop_frame
        self._sync_window_size()
        if self._cuda_hud_interop is not interop:
            return False
        if not self._surface.config:
            return False
        try:
            surface_texture = self._surface.acquire_next_image()
        except RuntimeError as exc:
            logger.warning(
                f"[presenter] swapchain acquire failed ({exc}); reconfiguring",
            )
            self._reconfigure_surface()
            return False
        if not surface_texture:
            time.sleep(0.001)
            return False

        try:
            width, height = self._configured_size
            encoder = self._device.create_command_encoder()
            encoder.copy_buffer_to_texture(
                self._display_texture,
                0,
                0,
                [0, 0, 0],
                rgba_buffer.buffer,
                0,
                rgba_buffer.size_bytes,
                rgba_buffer.row_pitch,
                [width, height, 1],
            )
            encoder.blit(surface_texture, self._display_texture)
            submit_id = self._device.submit_command_buffer(
                encoder.finish(),
                cuda_stream=cuda_stream,
            )
            interop.mark_submitted(rgba_buffer, submit_id)
            del surface_texture
            self._surface.present()
        except RuntimeError as exc:
            logger.warning(
                f"[presenter] swapchain present failed ({exc}); reconfiguring",
            )
            self._reconfigure_surface()
            return False
        return True

    def _present_canvas(self, use_gpu_camera: bool = False) -> None:
        # Sync to the window's CURRENT size before every present.
        # SDL3 doesn't always fire on_resize for compositor-side rezies
        # (window manager fitting the window to the screen on first
        # map, hidpi scaling, etc.), so we belt-and-braces compare
        # ``window.size`` to our last-configured size each tick.
        self._sync_window_size()
        if not self._surface.config:
            return
        try:
            surface_texture = self._surface.acquire_next_image()
        except RuntimeError as exc:
            # NVIDIA's Vulkan driver returns ``VK_ERROR_OUT_OF_DATE_KHR``
            # (surfaced here as a generic ``SLANG_FAIL``) when the
            # swapchain has gotten out of sync with the surface --
            # typically after a resize SDL didn't tell us about, or
            # after the swapchain has been idle long enough that the
            # OS reclaimed it. The fix is to reconfigure the surface
            # at the current window size; the next tick will retry.
            logger.warning(
                f"[presenter] swapchain acquire failed ({exc}); reconfiguring",
            )
            self._reconfigure_surface()
            return
        if not surface_texture:
            time.sleep(0.001)
            return
        # ``self._canvas_buffer`` is the same memory PIL drew into this tick
        # (see :func:`_allocate_canvas`), so this is a direct upload with no
        # PIL-to-numpy memcpy.
        try:
            self._display_texture.copy_from_numpy(self._canvas_buffer)
            encoder = self._device.create_command_encoder()
            if use_gpu_camera:
                self._composite_camera_gpu(encoder)
            encoder.blit(surface_texture, self._display_texture)
            self._device.submit_command_buffer(encoder.finish())
            del surface_texture
            self._surface.present()
        except RuntimeError as exc:
            logger.warning(
                f"[presenter] swapchain present failed ({exc}); reconfiguring",
            )
            self._reconfigure_surface()

    # -- GPU camera composite --------------------------------------

    def _composite_camera_gpu(self, encoder: Any) -> None:
        """Stamp the camera frame into the display texture on the GPU.

        Hardware bilinear blit + sub-region copy (<1 ms) over the chrome
        canvas the caller already uploaded; just fills the centred fit rect.
        """
        fit = self._compute_camera_fit()
        if fit is None:
            return
        fit_w, fit_h, offset_x, offset_y = fit
        if fit_w <= 0 or fit_h <= 0:
            return
        if not self._ensure_camera_texture_uploaded():
            return
        self._ensure_camera_fit_texture(fit_w, fit_h)
        # Hardware bilinear resize: source-sized texture to fit-sized
        # texture (whole-extent blit with linear filter).
        encoder.blit(self._camera_fit_texture, self._camera_texture)
        # Sub-region copy: fit-sized texture into the centred rect. Uses the
        # int-layer / int-mip ``copy_texture`` overload because this slangpy
        # build's ``SubresourceRange`` ctor only accepts a dict, not kwargs.
        spy = self._spy
        encoder.copy_texture(
            self._display_texture,
            0,  # dst_layer
            0,  # dst_mip
            spy.math.uint3(offset_x, offset_y, 0),
            self._camera_fit_texture,
            0,  # src_layer
            0,  # src_mip
            spy.math.uint3(0, 0, 0),
        )

    def _compute_camera_fit(self) -> tuple[int, int, int, int] | None:
        """Centered cover-fit for the current camera frame.

        Returns ``(fit_w, fit_h, offset_x, offset_y)`` in display-texture
        coordinates, or ``None`` if no camera frame is available. The
        offsets put the camera centred inside the camera area (left of
        the panel column).
        """
        if self._latest_camera_src_size is None:
            return None
        src_w, src_h = self._latest_camera_src_size
        screen_w, screen_h = self._configured_size
        panel_w = (
            HUD_PANEL_WIDTH if screen_w > HUD_PANEL_WIDTH + MIN_WINDOW_W // 2 else 0
        )
        cam_w = max(1, screen_w - panel_w)
        cam_h = screen_h
        if src_w <= 0 or src_h <= 0:
            return None
        scale = min(cam_w / src_w, cam_h / src_h)
        fit_w = max(1, int(src_w * scale))
        fit_h = max(1, int(src_h * scale))
        offset_x = (cam_w - fit_w) // 2
        offset_y = (cam_h - fit_h) // 2
        return (fit_w, fit_h, offset_x, offset_y)

    def _ensure_camera_texture_uploaded(self) -> bool:
        """Upload the latest world-model frame to the GPU camera texture.

        Lazily (re)allocates the source-sized RGBA8 texture, pads the source
        RGB into an RGBA8 staging buffer, and uploads via ``copy_from_numpy``.
        The expansion is cached so back-to-back ticks with the same frame skip
        the copy.
        """
        if self._latest_camera_pil is None or self._latest_camera_src_size is None:
            return False
        src_w, src_h = self._latest_camera_src_size
        if self._camera_texture is None or self._camera_texture_size != (src_w, src_h):
            spy = self._spy
            self._camera_texture = self._device.create_texture(
                format=spy.Format.rgba8_unorm,
                width=src_w,
                height=src_h,
                usage=spy.TextureUsage.shader_resource
                | spy.TextureUsage.unordered_access,
                label="hud_camera_src",
            )
            self._camera_texture_size = (src_w, src_h)
            self._latest_camera_rgba = None
            # Drop the staging buffer too -- it follows source-size.
            self._camera_rgba_staging = None
        # Re-use a single alpha-pre-filled RGBA staging buffer per source size
        # so the per-tick work is one RGB slice copy, not an alpha alloc +
        # fresh RGBA alloc + concatenate.
        if self._camera_rgba_staging is None or self._camera_rgba_staging.shape[:2] != (
            src_h,
            src_w,
        ):
            self._camera_rgba_staging = np.empty((src_h, src_w, 4), dtype=np.uint8)
            # One-time alpha fill -- the GPU camera path only ever
            # writes the RGB slice from here on, so alpha stays 255.
            self._camera_rgba_staging[..., 3] = 255
            # Force the RGB refill below since the buffer is fresh.
            self._latest_camera_rgba = None
        if self._latest_camera_rgba is None:
            # Single strided RGB copy into the alpha-padded staging buffer;
            # ``np.asarray(pil)`` is a zero-copy view of the world-model frame.
            self._camera_rgba_staging[..., :3] = np.asarray(self._latest_camera_pil)
            self._latest_camera_rgba = self._camera_rgba_staging
        self._camera_texture.copy_from_numpy(self._latest_camera_rgba)
        return True

    def _ensure_camera_fit_texture(self, fit_w: int, fit_h: int) -> None:
        """Lazily (re)allocate the fit-sized GPU camera texture."""
        if self._camera_fit_texture is not None and self._camera_fit_size == (
            fit_w,
            fit_h,
        ):
            return
        spy = self._spy
        self._camera_fit_texture = self._device.create_texture(
            format=spy.Format.rgba8_unorm,
            width=fit_w,
            height=fit_h,
            usage=spy.TextureUsage.shader_resource | spy.TextureUsage.unordered_access,
            label="hud_camera_fit",
        )
        self._camera_fit_size = (fit_w, fit_h)

    def _sync_window_size(self) -> None:
        """If the window's current size differs from our last
        configuration, reconfigure the surface + canvas before the
        next present.
        """
        new_size = self._current_window_size()
        if new_size != self._configured_size:
            self._apply_resize(*new_size)

    def _reconfigure_surface(self) -> None:
        """Rebuild the surface configuration at the current window size.

        Used on the swapchain-lost path.
        """
        self._apply_resize(*self._current_window_size(), force=True)

    def _normalise_present_size(self, width: int, height: int) -> tuple[int, int]:
        return max(1, int(width)), max(1, int(height))

    def _current_window_size(self) -> tuple[int, int]:
        actual = self._window.size
        return self._normalise_present_size(actual.x, actual.y)

    def _recreate_cuda_hud_interop_after_resize(self, width: int, height: int) -> None:
        if self._cuda_hud_interop is None:
            return
        self._retired_cuda_hud_interops.append(self._cuda_hud_interop)
        self._cuda_hud_interop = None
        self._cuda_hud_interop = self._create_cuda_hud_interop(width, height)
        if self._cuda_hud_interop is not None:
            logger.info(
                "[presenter] hud_cuda_interop=recreated after window resize",
            )
            self._cuda_hud_resize_logged = False
            return
        if not self._cuda_hud_resize_logged:
            logger.warning(
                "[presenter] hud_cuda_interop=disabled after window resize; "
                "could not recreate shared CUDA/Vulkan resources",
            )
            self._cuda_hud_resize_logged = True

    # -- Render ------------------------------------------------------

    def _layout_regions(
        self,
    ) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int]]:
        screen_w, screen_h = self._canvas.size
        panel_w = (
            HUD_PANEL_WIDTH if screen_w > HUD_PANEL_WIDTH + MIN_WINDOW_W // 2 else 0
        )
        camera_area = (0, 0, max(1, screen_w - panel_w), screen_h)
        panel_rect = (camera_area[2], 0, screen_w, screen_h)
        return camera_area, panel_rect

    def _render_canvas(
        self,
        status_message: str | None,
        *,
        camera_transparent: bool = False,
    ) -> None:
        """Composite camera + chrome into ``self._canvas`` for this frame.

        Steps: fill background, draw camera (or placeholder), paste the cached
        panel chrome, draw dynamic chrome (speed/wheel/pedals/BEV), then any
        open dropdown and status overlay. Drawing is in-place on ``self._canvas``
        to avoid a fresh RGBA alloc each frame.
        """
        # Apply any debounced drive-key releases whose grace window has
        # elapsed. Done here because ``_render_canvas`` runs once per
        # tick and is the only consumer of ``_keyboard_drive`` state;
        # putting the expiry inline guarantees real releases land
        # within one tick of the debounce window expiring.
        self._expire_pending_drive_releases()

        canvas = self._canvas
        screen_w, screen_h = canvas.size
        camera_area, panel_rect = self._layout_regions()
        panel_w = panel_rect[2] - panel_rect[0]

        draw = ImageDraw.Draw(canvas)
        # No full-canvas clear: the panel paste and ``_draw_camera`` fully
        # cover their regions every frame and the letterbox bars stay at
        # BG_COLOR, so skipping the 2 MP RGBA fill saves ~3-8 ms / tick at
        # 1080p. Only the placeholder branch wipes the camera area (below).
        if camera_transparent:
            # CUDA HUD mode composites the camera on the GPU, so keep
            # only the camera area transparent before drawing any
            # status/dropdown overlay that should sit above it.
            draw.rectangle(camera_area, fill=(0, 0, 0, 0))

        camera_drawn = False
        if camera_transparent:
            camera_drawn = True
        elif self._latest_camera_pil is not None:
            if status_message is None:
                # GPU camera path fills the centred fit rect after the canvas
                # upload; here we only repaint the letterbox bars (~0.3 ms) so
                # they don't show last frame's content when the fit rect resizes.
                draw.rectangle(camera_area, fill=BG_COLOR + (255,))
            else:
                # CPU camera path: composite onto canvas so the status
                # overlay (drawn after this) sits on top of the
                # loading-frame contents. Only used during warmup.
                self._draw_camera(canvas, self._latest_camera_pil, camera_area)
            camera_drawn = True
        if not camera_drawn:
            # Wipe the camera area so the previous tick's placeholder / camera
            # frame doesn't ghost behind the new placeholder (placeholder ticks
            # only, so cheaper than an always-on full-screen clear).
            draw.rectangle(camera_area, fill=BG_COLOR + (255,))
            if not self._engine_active:
                if self._model_can_prewarm and not self._model_ready_probe():
                    placeholder = "Loading world model..."
                elif self._scene_selection_locked():
                    placeholder = "Preloading scenes..."
                elif self._model_can_prewarm:
                    placeholder = "Ready - pick a scene"
                else:
                    placeholder = "Load Scene"
            elif not self._model_ready_probe():
                placeholder = "Loading World Model"
            else:
                placeholder = "Loading Scene..."
            self._draw_camera_placeholder(canvas, draw, camera_area, placeholder)

        # Poll the drive sink *every* tick (before the conditional panel draw):
        # ``_keyboard_drive.update()`` publishes key state to the simulation, so
        # gating it on ``_draw_panel`` would freeze drive input whenever the
        # panel is hidden (narrow window). Speed smoothing also reads this.
        wheel_state = self._poll_drive_state()
        self._update_speed(wheel_state)

        if panel_w > 0:
            self._draw_panel(canvas, draw, panel_rect, wheel_state)

        if self._scene_dropdown_open:
            self._draw_scene_dropdown(canvas, draw)
        if self._variant_dropdown_open:
            self._draw_variant_dropdown(canvas, draw)

        if status_message:
            self._draw_status_overlay(canvas, draw, camera_area, status_message)

    # -- Camera area -------------------------------------------------

    def _draw_camera(
        self,
        canvas: Image.Image,
        camera: Image.Image,
        area: tuple[int, int, int, int],
    ) -> None:
        # Cover-fit with letterbox bars: preserve aspect, centre in area,
        # leave the unused gap as the surrounding ``BG_COLOR`` fill.
        ax, ay, ar, ab = area
        aw, ah = ar - ax, ab - ay
        fw, fh = camera.size
        if fw <= 0 or fh <= 0 or aw <= 0 or ah <= 0:
            return
        scale = min(aw / fw, ah / fh)
        target_w = max(1, int(fw * scale))
        target_h = max(1, int(fh * scale))
        cache_key = (id(camera), target_w, target_h)
        if (
            cache_key != self._camera_resize_cache_key
            or self._camera_resize_cache is None
        ):
            if (target_w, target_h) == (fw, fh):
                resized = camera
            else:
                resized = camera.resize(
                    (target_w, target_h),
                    Image.Resampling.LANCZOS
                    if scale < 1.0
                    else Image.Resampling.BILINEAR,
                )
            self._camera_resize_cache = resized
            self._camera_resize_cache_key = cache_key
        else:
            resized = self._camera_resize_cache
        x = ax + (aw - target_w) // 2
        y = ay + (ah - target_h) // 2
        if resized.mode != "RGBA":
            canvas.paste(resized, (x, y))
        else:
            canvas.alpha_composite(resized, (x, y))

    def _draw_camera_placeholder(
        self,
        canvas: Image.Image,
        draw: ImageDraw.ImageDraw,
        area: tuple[int, int, int, int],
        message: str,
    ) -> None:
        ax, ay, ar, ab = area
        cx, cy = (ax + ar) // 2, (ay + ab) // 2
        bbox = _measure_text(self._font_large, message)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        draw.text(
            (cx - text_w // 2 - bbox[0], cy - text_h // 2 - bbox[1]),
            message,
            fill=TEXT_COLOR,
            font=self._font_large,
        )
        if message in (
            "Load Scene",
            "Loading Scene...",
            "Loading world model...",
            "Ready - pick a scene",
            "Preloading scenes...",
        ):
            hint = (
                "Preloading scenes, please wait..."
                if self._scene_selection_locked()
                else "Pick a scene from the panel on the right"
            )
            hbox = _measure_text(self._font_small, hint)
            hw = hbox[2] - hbox[0]
            draw.text(
                (cx - hw // 2 - hbox[0], cy + text_h // 2 + 12 - hbox[1]),
                hint,
                fill=LABEL_COLOR,
                font=self._font_small,
            )

    def _draw_status_overlay(
        self,
        canvas: Image.Image,
        draw: ImageDraw.ImageDraw,
        area: tuple[int, int, int, int],
        message: str,
    ) -> None:
        ax, ay, ar, ab = area
        cx, cy = (ax + ar) // 2, (ay + ab) // 2
        bbox = _measure_text(self._font_large, message)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        pad = 24
        box_left = cx - text_w // 2 - pad
        box_right = cx + text_w // 2 + pad
        box_top = cy - text_h // 2 - pad
        box_bottom = cy + text_h // 2 + pad
        # Semi-transparent dark callout. PIL's draw.rectangle on the
        # alpha-composited canvas just writes the alpha channel through.
        draw.rectangle(
            (box_left, box_top, box_right, box_bottom),
            fill=(20, 20, 20, 230),
            outline=(240, 240, 240, 255),
            width=2,
        )
        draw.text(
            (cx - text_w // 2 - bbox[0], cy - text_h // 2 - bbox[1]),
            message,
            fill=TEXT_COLOR,
            font=self._font_large,
        )

    # -- Panel chrome ------------------------------------------------

    def _poll_drive_state(self) -> Any:
        """Read the current drive state (wheel if connected, else keyboard).

        Pulled out of ``_draw_panel`` so the keyboard-drive sink's
        ``update()`` side-effect (publishing key state into the
        simulation) and the speed-digit smoothing in
        :meth:`_update_speed` run on every tick, including ticks where
        the side panel is not drawn (narrow window / camera-only mode).
        """
        if self._wheel is not None and self._wheel.state.connected:
            return self._wheel.state
        return self._keyboard_drive.update()

    def _draw_panel(
        self,
        canvas: Image.Image,
        draw: ImageDraw.ImageDraw,
        panel_rect: tuple[int, int, int, int],
        wheel_state: Any,
    ) -> None:
        px, py, pr, pb = panel_rect
        panel_size = (pr - px, pb - py)
        chrome = self._get_panel_chrome(panel_size)
        canvas.paste(chrome, (px, py))

        # Hit-test rectangles must stay in screen-space so the
        # ``on_mouse_event`` handler can compare against them directly.
        margin = 10
        bar_h = 32
        header_x = px + margin
        header_w = panel_size[0] - margin * 2
        header_y = py + 8
        variant_y = header_y + bar_h + 4
        self._scene_header_rect = (
            header_x,
            header_y,
            header_x + header_w,
            header_y + bar_h,
        )
        self._variant_header_rect = (
            header_x,
            variant_y,
            header_x + header_w,
            variant_y + bar_h,
        )

        center_x = px + panel_size[0] // 2
        # ``speed_y`` is the top of the speed-digit chip. PIL renders
        # text into a tight glyph-bbox image (no leading above the
        # glyph), so positioning the chip-top right after the variant
        # bar would still land the visible glyph inside the bar. Add a
        # ~12 px clearance below ``variant_y + bar_h`` so the digit
        # never overlaps the headers.
        speed_y = variant_y + bar_h + 12
        self._draw_speed(canvas, draw, center_x, speed_y, int(self._speed_mph))

        # Light the reverse indicator red when reverse is engaged; the cached
        # chrome only draws the inactive grey "R" box at the same spot.
        if getattr(wheel_state, "reverse", False):
            rx0, ry0 = px + 14, speed_y + 70
            draw.rounded_rectangle(
                (rx0, ry0, px + 54, speed_y + 102), radius=5, fill=(200, 60, 60, 255)
            )
            rbox = _measure_text(self._font_tiny, "R")
            rw, rh = rbox[2] - rbox[0], rbox[3] - rbox[1]
            draw.text(
                (rx0 + (40 - rw) // 2 - rbox[0], ry0 + (32 - rh) // 2 - rbox[1]),
                "R",
                fill=(255, 255, 255),
                font=self._font_tiny,
            )

        # The wheel sits a little below the speed readout. Pedals are
        # anchored to ``speed_y`` (NOT the wheel center) below, so this
        # vertical nudge can be tuned without dragging the pedals / BEV
        # down with it -- and so the pedal + BEV geometry stays in lockstep
        # with the cached chrome in ``_get_panel_chrome``.
        wheel_center = (center_x, speed_y + 210)
        self._draw_wheel(canvas, draw, wheel_center, 112, wheel_state.steering)

        angle_text = f"{int(wheel_state.steering * 450):+}\u00b0"
        abox = _measure_text(self._font_medium, angle_text)
        aw = abox[2] - abox[0]
        draw.text(
            (center_x - aw // 2 - abox[0], wheel_center[1] + 128 - abox[1]),
            angle_text,
            fill=ACCENT_AMBER,
            font=self._font_medium,
        )

        pedals_y = speed_y + 365
        self._draw_pedals(canvas, draw, panel_rect, pedals_y, wheel_state)

        controls_bottom_y = pedals_y + 220
        self._draw_bev(canvas, draw, panel_rect, controls_bottom_y)

    def _get_panel_chrome(self, panel_size: tuple[int, int]) -> Image.Image:
        current_scene_option = self._current_scene_option()
        has_multiple_variants = (
            current_scene_option is not None and len(current_scene_option.variants) > 1
        )
        # ``_engine_active`` is part of the cache key because the scene
        # header label changes shape ("Select Scene" when the engine
        # isn't running, "Running clipgt-...\u2026" when it is). The
        # demo wrapper also explicitly invalidates the cache around
        # ``set_engine_active``; the key entry here is belt-and-braces.
        key = (
            panel_size,
            str(self._current_scene),
            self._selected_variant,
            self._scene_dropdown_open,
            self._variant_dropdown_open,
            has_multiple_variants,
            self._engine_active,
            # Scene header reads "Preloading scenes..." while locked, so the
            # lock state has to invalidate the cached chrome too.
            self._scene_selection_locked(),
        )
        if key == self._panel_chrome_cache_key and self._panel_chrome_cache is not None:
            return self._panel_chrome_cache

        panel_w, panel_h = panel_size
        chrome = Image.new("RGBA", (panel_w, panel_h), PANEL_BG + (255,))
        d = ImageDraw.Draw(chrome)
        # Vertical green divider on the panel's left edge (signature
        # NVIDIA touch, matches the pygame HUD).
        d.rectangle((0, 0, 3, panel_h), fill=NVIDIA_GREEN + (255,))

        margin = 10
        bar_h = 32
        header_w = panel_w - margin * 2
        header_y = 8

        # Scene header bar. Reserve room on the left for the green
        # status dot and on the right for the dropdown arrow; the
        # remaining width is what the scene label gets to use, and we
        # truncate-with-ellipsis to fit.
        scene_rect = (margin, header_y, margin + header_w, header_y + bar_h)
        d.rounded_rectangle(scene_rect, radius=6, fill=HEADER_BG + (255,))
        d.ellipse(
            (margin + 8, header_y + 11, margin + 18, header_y + 21),
            fill=NVIDIA_GREEN + (255,),
        )
        if self._engine_active:
            scene_label_full = (
                f"Running {self._scene_label_fn(self._current_scene)}\u2026"
            )
        elif self._scene_selection_locked():
            scene_label_full = "Preloading scenes\u2026"
        else:
            scene_label_full = "Select Scene"
        scene_label_max_w = header_w - 26 - 30  # 26 left for dot, 30 right for arrow
        scene_label = _truncate_text_to_width(
            self._font_small, scene_label_full, scene_label_max_w
        )
        d.text(
            (margin + 26, header_y + 6),
            scene_label,
            fill=TEXT_COLOR,
            font=self._font_small,
        )
        scene_arrow = "\u25b2" if self._scene_dropdown_open else "\u25bc"
        d.text(
            (margin + header_w - 24, header_y + 6),
            scene_arrow,
            fill=LABEL_COLOR,
            font=self._font_small,
        )

        # Variant header bar. Same truncation pattern in case the
        # variant string is unusually long.
        variant_y = header_y + bar_h + 4
        variant_rect = (margin, variant_y, margin + header_w, variant_y + bar_h)
        d.rounded_rectangle(variant_rect, radius=6, fill=HEADER_BG + (255,))
        variant_full = f"Variant: {self._selected_variant}"
        variant_max_w = header_w - 10 - (30 if has_multiple_variants else 10)
        variant_label = _truncate_text_to_width(
            self._font_small, variant_full, variant_max_w
        )
        d.text(
            (margin + 10, variant_y + 6),
            variant_label,
            fill=TEXT_COLOR,
            font=self._font_small,
        )
        # Only advertise the dropdown affordance when a scene is loaded; the
        # header isn't clickable otherwise (see _handle_click).
        if has_multiple_variants and self._engine_active:
            v_arrow = "\u25b2" if self._variant_dropdown_open else "\u25bc"
            d.text(
                (margin + header_w - 24, variant_y + 6),
                v_arrow,
                fill=LABEL_COLOR,
                font=self._font_small,
            )

        # ``mph`` label baseline + reverse-indicator box. Speed-y must
        # match the live ``_draw_panel`` calculation; both place the
        # speed-digit chip-top ~12 px below the variant bar so PIL's
        # tight-bbox glyph chip clears the headers.
        center_x = panel_w // 2
        speed_y = variant_y + bar_h + 12
        mbox = _measure_text(self._font_tiny, "mph")
        mw = mbox[2] - mbox[0]
        d.text(
            (center_x - mw // 2 - mbox[0], speed_y + 76 - mbox[1]),
            "mph",
            fill=TEXT_COLOR,
            font=self._font_tiny,
        )
        d.rounded_rectangle(
            (14, speed_y + 70, 54, speed_y + 102),
            radius=5,
            fill=(60, 60, 70, 255),
        )
        rbox = _measure_text(self._font_tiny, "R")
        rw = rbox[2] - rbox[0]
        rh = rbox[3] - rbox[1]
        d.text(
            (14 + (40 - rw) // 2 - rbox[0], speed_y + 70 + (32 - rh) // 2 - rbox[1]),
            "R",
            fill=(100, 100, 110),
            font=self._font_tiny,
        )

        # BEV chrome (cream background + green outline + title). Pedals are
        # anchored to ``speed_y`` here; keep this in lockstep with the same
        # ``pedals_y`` / ``controls_bottom_y`` computation in ``_draw_panel``
        # so the BEV foreground lands on this cached background.
        pedals_y = speed_y + 365
        controls_bottom_y = pedals_y + 220
        bev_top = controls_bottom_y + BEV_PANEL_TOP_GAP
        bev_height = panel_h - bev_top - BEV_PANEL_BOTTOM_MARGIN
        if bev_height >= BEV_PANEL_MIN_HEIGHT:
            bev_left = BEV_PANEL_SIDE_MARGIN
            bev_right = panel_w - BEV_PANEL_SIDE_MARGIN
            bev_rect = (bev_left, bev_top, bev_right, bev_top + bev_height)
            tbox = _measure_text(self._font_small, "BEV Map")
            d.text(
                (bev_left + 2, bev_top - (tbox[3] - tbox[1]) - 4 - tbox[1]),
                "BEV Map",
                fill=NVIDIA_GREEN,
                font=self._font_small,
            )
            d.rounded_rectangle(bev_rect, radius=10, fill=GMAPS_LAND_RGB + (255,))
            d.rounded_rectangle(
                bev_rect, radius=10, outline=NVIDIA_GREEN + (255,), width=2
            )

        self._panel_chrome_cache = chrome
        self._panel_chrome_cache_key = key
        return chrome

    # -- Speed digit -------------------------------------------------

    def _draw_speed(
        self,
        canvas: Image.Image,
        draw: ImageDraw.ImageDraw,
        center_x: int,
        speed_y: int,
        mph: int,
    ) -> None:
        chip = self._speed_chip_cache.get_or_compute(
            mph, lambda: self._render_speed_chip(mph)
        )
        cw, ch = chip.size
        canvas.alpha_composite(chip, (center_x - cw // 2, speed_y))

    def _render_speed_chip(self, mph: int) -> Image.Image:
        text = f"{mph:d}"
        bbox = _measure_text(self._font_speed, text)
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        chip = Image.new("RGBA", (max(1, w), max(1, h)), (0, 0, 0, 0))
        ImageDraw.Draw(chip).text(
            (-bbox[0], -bbox[1]), text, fill=NVIDIA_GREEN, font=self._font_speed
        )
        return chip

    # -- Steering wheel ----------------------------------------------

    def _draw_wheel(
        self,
        canvas: Image.Image,
        draw: ImageDraw.ImageDraw,
        center: tuple[int, int],
        radius: int,
        steering: float,
    ) -> None:
        base = self._get_wheel_base(radius)
        if base is None:
            self._draw_wheel_fallback(draw, center, radius, steering)
            return
        angle_deg = steering * 450.0
        bucket = (
            int(round(angle_deg / WHEEL_ROTATION_QUANTUM_DEG))
            * WHEEL_ROTATION_QUANTUM_DEG
        )
        rotated = self._wheel_rotation_cache.get_or_compute(
            bucket,
            lambda b=bucket, base=base: base.rotate(
                b, resample=Image.Resampling.BILINEAR
            ),
        )
        rw, rh = rotated.size
        canvas.alpha_composite(rotated, (center[0] - rw // 2, center[1] - rh // 2))

    def _get_wheel_base(self, radius: int) -> Image.Image | None:
        if self._wheel_base_size == radius and self._wheel_base_image is not None:
            return self._wheel_base_image
        pil = self._control_assets.steering_wheel
        if pil is None:
            return None
        diameter = max(2, radius * 2)
        scaled = pil.copy()
        scaled.thumbnail((diameter, diameter), Image.Resampling.BILINEAR)
        if scaled.mode != "RGBA":
            scaled = scaled.convert("RGBA")
        self._wheel_base_image = scaled
        self._wheel_base_size = radius
        self._wheel_rotation_cache.clear()
        return scaled

    def _draw_wheel_fallback(
        self,
        draw: ImageDraw.ImageDraw,
        center: tuple[int, int],
        radius: int,
        steering: float,
    ) -> None:
        cx, cy = center
        draw.ellipse(
            (cx - radius, cy - radius, cx + radius, cy + radius),
            outline=(60, 60, 80, 255),
            width=4,
        )
        angle = -steering * _math.radians(450)
        tip_x = cx + int(_math.sin(angle) * (radius - 6))
        tip_y = cy - int(_math.cos(angle) * (radius - 6))
        draw.line((cx, cy, tip_x, tip_y), fill=NVIDIA_GREEN + (255,), width=4)

    # -- Pedals ------------------------------------------------------

    def _draw_pedals(
        self,
        canvas: Image.Image,
        draw: ImageDraw.ImageDraw,
        panel_rect: tuple[int, int, int, int],
        pedals_y: int,
        wheel_state: Any,
    ) -> None:
        target_w = 80
        target_h = 160
        center_x = panel_rect[0] + (panel_rect[2] - panel_rect[0]) // 2
        gap = 24
        throttle_x = center_x + gap
        brake_x = center_x - gap - target_w
        throttle_pressed = wheel_state.throttle > 0.05
        brake_pressed = wheel_state.brake > 0.05
        throttle_pil = (
            self._control_assets.throttle_pressed
            if throttle_pressed
            else self._control_assets.throttle_unpressed
        )
        brake_pil = (
            self._control_assets.brake_pressed
            if brake_pressed
            else self._control_assets.brake_unpressed
        )
        # Sprite when the user has the AlpaSim pedal PNGs installed,
        # otherwise a CPU-rendered fill bar so the chrome is informative
        # even without the optional asset pack. The bar fills upward
        # from the bottom proportional to the pedal value, mirroring how
        # a real pedal travels.
        if throttle_pil is not None:
            throttle_img = self._fit_pedal(throttle_pil, "T", target_w, target_h)
            canvas.alpha_composite(throttle_img, (throttle_x, pedals_y))
        else:
            self._draw_pedal_bar(
                draw,
                throttle_x,
                pedals_y,
                target_w,
                target_h,
                wheel_state.throttle,
                NVIDIA_GREEN,
            )
        if brake_pil is not None:
            brake_img = self._fit_pedal(brake_pil, "B", target_w, target_h)
            # The brake sprite is wide/short, so aspect-fitting it into the
            # tall pedal slot leaves it hugging the top. Center it vertically
            # in the slot so it sits lower, roughly level with the throttle
            # (mirrors AlpaSim's ``(throttle_h - brake_h) // 2`` offset).
            brake_dy = max(0, (target_h - brake_img.height) // 2)
            canvas.alpha_composite(brake_img, (brake_x, pedals_y + brake_dy))
        else:
            self._draw_pedal_bar(
                draw,
                brake_x,
                pedals_y,
                target_w,
                target_h,
                wheel_state.brake,
                # Soft red. ``ACCENT_AMBER`` is for the steering angle
                # readout; brake should read as "stop" without competing
                # with the steering colour.
                (220, 80, 80),
            )

        labels_y = pedals_y + target_h + 8
        for cx_offset, text in (
            (throttle_x + target_w // 2, f"Throttle {wheel_state.throttle:0.2f}"),
            (brake_x + target_w // 2, f"Brake {wheel_state.brake:0.2f}"),
        ):
            tbox = _measure_text(self._font_tiny, text)
            tw = tbox[2] - tbox[0]
            draw.text(
                (cx_offset - tw // 2 - tbox[0], labels_y - tbox[1]),
                text,
                fill=TEXT_COLOR,
                font=self._font_tiny,
            )

    @staticmethod
    def _draw_pedal_bar(
        draw: ImageDraw.ImageDraw,
        x: int,
        y: int,
        w: int,
        h: int,
        fraction: float,
        fill_color: tuple[int, int, int],
    ) -> None:
        """Vertical pedal-style fill bar, used when no sprite is available.

        ``fraction`` is clamped to ``[0, 1]``. The fill grows upward from
        the bottom (matching the visual metaphor of a pedal being
        depressed). Outer track + 2 px padded inner fill so the rounded
        corners stay clean even when fully filled.
        """
        f = max(0.0, min(1.0, float(fraction)))
        # Outer track: dark fill + lighter outline for visual weight.
        draw.rounded_rectangle(
            (x, y, x + w, y + h),
            radius=8,
            fill=(40, 40, 50, 255),
            outline=(80, 80, 90, 255),
            width=2,
        )
        # Inner track inset by 4 px on every side so the fill stays
        # entirely inside the rounded outer border.
        inner_top = y + 4
        inner_bottom = y + h - 4
        inner_left = x + 4
        inner_right = x + w - 4
        inner_h = inner_bottom - inner_top
        if inner_h <= 0 or f <= 0.0:
            return
        fill_h = int(round(inner_h * f))
        if fill_h <= 0:
            return
        fill_top = inner_bottom - fill_h
        draw.rounded_rectangle(
            (inner_left, fill_top, inner_right, inner_bottom),
            radius=4,
            fill=fill_color + (255,),
        )

    def _fit_pedal(
        self, pil_image: Image.Image, kind: str, target_w: int, target_h: int
    ) -> Image.Image:
        key = (id(pil_image), kind, target_w, target_h)

        def _build() -> Image.Image:
            scaled = pil_image.copy()
            scaled.thumbnail((target_w, target_h), Image.Resampling.BILINEAR)
            if scaled.mode != "RGBA":
                scaled = scaled.convert("RGBA")
            return scaled

        return self._pedal_cache.get_or_compute(key, _build)

    # -- BEV minimap -------------------------------------------------

    def _draw_bev(
        self,
        canvas: Image.Image,
        draw: ImageDraw.ImageDraw,
        panel_rect: tuple[int, int, int, int],
        controls_bottom_y: int,
    ) -> None:
        bev_top = controls_bottom_y + BEV_PANEL_TOP_GAP
        bev_height = panel_rect[3] - bev_top - BEV_PANEL_BOTTOM_MARGIN
        if bev_height < BEV_PANEL_MIN_HEIGHT:
            return
        bev_left = panel_rect[0] + BEV_PANEL_SIDE_MARGIN
        bev_right = panel_rect[2] - BEV_PANEL_SIDE_MARGIN
        bev_rect = (bev_left, bev_top, bev_right, bev_top + bev_height)
        inner = (bev_rect[0] + 4, bev_rect[1] + 4, bev_rect[2] - 4, bev_rect[3] - 4)
        inner_w = inner[2] - inner[0]
        inner_h = inner[3] - inner[1]

        if self._latest_bev_pil is None:
            text = "WAITING FOR BEV..."
            tbox = _measure_text(self._font_tiny, text)
            tw = tbox[2] - tbox[0]
            cx = (bev_rect[0] + bev_rect[2]) // 2
            cy = (bev_rect[1] + bev_rect[3]) // 2
            draw.text(
                (cx - tw // 2 - tbox[0], cy - (tbox[3] - tbox[1]) // 2 - tbox[1]),
                text,
                fill=LABEL_COLOR,
                font=self._font_tiny,
            )
            return

        panel_image = self._get_bev_panel_image((inner_w, inner_h))
        if panel_image is not None:
            canvas.paste(panel_image, (inner[0], inner[1]))

        # Ego marker (Google-Maps chevron) over the BEV panel.
        marker_cx = inner[0] + inner_w // 2
        marker_cy = inner[1] + int(inner_h * self._bev_marker_y_rel())
        marker_size = max(10, min(inner_w, inner_h) // 14)
        self._draw_bev_marker(draw, marker_cx, marker_cy, marker_size)

    def _get_bev_panel_image(self, target_size: tuple[int, int]) -> Image.Image | None:
        if self._latest_bev_pil is None:
            return None
        target_w, target_h = target_size
        if target_w <= 0 or target_h <= 0:
            return None
        key = (id(self._latest_bev_pil), target_w, target_h)
        if key == self._bev_panel_cache_key and self._bev_panel_cache is not None:
            return self._bev_panel_cache
        from omnidreams.interactive_drive.demo import _apply_googlemaps_filter

        bev = self._latest_bev_pil
        # Cover-fit + crop, then GoogleMaps filter. Two ordering choices keep
        # this cheap (~10 ms vs ~57 ms / tick at 1024 -> ~470 panel):
        #   1) Resize before filtering so the per-pixel float32 filter runs on
        #      ~0.22 MP not 1 MP; it commutes with bilinear resampling to
        #      within ~2 channel units (visually identical).
        #   2) BILINEAR not LANCZOS -- far cheaper, and the tint blend masks
        #      the sharpness difference at minimap scale.
        scale = max(target_w / bev.width, target_h / bev.height)
        scaled_w = max(1, int(bev.width * scale))
        scaled_h = max(1, int(bev.height * scale))
        scaled = bev.resize((scaled_w, scaled_h), Image.Resampling.BILINEAR)
        crop_left = (scaled_w - target_w) // 2
        crop_top = (scaled_h - target_h) // 2
        cropped = scaled.crop(
            (crop_left, crop_top, crop_left + target_w, crop_top + target_h)
        )
        filtered = _apply_googlemaps_filter(cropped)
        self._bev_panel_cache = filtered
        self._bev_panel_cache_key = key
        return filtered

    @staticmethod
    def _draw_bev_marker(
        draw: ImageDraw.ImageDraw, cx: int, cy: int, size: int
    ) -> None:
        # Soft drop shadow.
        shadow_size = size + 4
        draw.ellipse(
            (
                cx - shadow_size,
                cy - shadow_size + 2,
                cx + shadow_size,
                cy + shadow_size + 2,
            ),
            fill=(0, 0, 0, 60),
        )
        # White outer ring.
        draw.ellipse(
            (cx - size, cy - size, cx + size, cy + size), fill=(255, 255, 255, 255)
        )
        # Forward chevron in Google-Maps blue.
        chevron = size - 4
        draw.polygon(
            [
                (cx, cy - chevron),
                (cx - int(chevron * 0.7), cy + int(chevron * 0.55)),
                (cx, cy + int(chevron * 0.18)),
                (cx + int(chevron * 0.7), cy + int(chevron * 0.55)),
            ],
            fill=(66, 133, 244, 255),
        )

    # -- Dropdowns ---------------------------------------------------

    def _draw_scene_dropdown(
        self, canvas: Image.Image, draw: ImageDraw.ImageDraw
    ) -> None:
        if self._scene_header_rect is None:
            return
        sx, _sy, sr, sb = self._scene_header_rect
        if not self._scene_options:
            empty = (sx, sb + 2, sr, sb + 36)
            draw.rounded_rectangle(empty, radius=6, fill=(70, 35, 35, 255))
            draw.text(
                (sx + 12, sb + 9),
                f"No scenes found in {self._args.scene_dir}",
                fill=(255, 220, 220),
                font=self._font_tiny,
            )
            return

        item_h = 80
        items_top = sb + 2
        bg = (sx, items_top - 1, sr, items_top + len(self._scene_options) * item_h + 1)
        draw.rounded_rectangle(bg, radius=6, fill=(35, 35, 50, 255))
        draw.rounded_rectangle(bg, radius=6, outline=(60, 60, 80, 255), width=1)

        self._scene_item_rects = []
        for idx, scene in enumerate(self._scene_options):
            top = items_top + idx * item_h
            rect = (sx, top, sr, top + item_h)
            self._scene_item_rects.append((rect, scene))
            if self._scene_option_matches_current(scene):
                draw.rectangle(rect, fill=ACTIVE_BG + (255,))
            elif scene.label == self._hovered_scene_label:
                draw.rectangle(rect, fill=HOVER_BG + (255,))
            text_x = rect[0] + 12
            text_y = top + item_h // 2 - 8
            thumb = self._get_scene_thumbnail(scene)
            if thumb is not None:
                tw, th = thumb.size
                tx = rect[0] + 6
                ty = top + max(0, (item_h - th) // 2)
                canvas.paste(thumb, (tx, ty))
                draw.rectangle(
                    (tx, ty, tx + tw, ty + th), outline=(60, 60, 80, 255), width=1
                )
                text_x = tx + tw + 10
            label = _truncate_text_to_width(
                self._font_tiny, scene.label, max(0, rect[2] - text_x - 8)
            )
            draw.text((text_x, text_y), label, fill=TEXT_COLOR, font=self._font_tiny)

    def _draw_variant_dropdown(
        self, canvas: Image.Image, draw: ImageDraw.ImageDraw
    ) -> None:
        if self._variant_header_rect is None:
            return
        scene_option = self._current_scene_option()
        if scene_option is None or len(scene_option.variants) <= 1:
            return
        vx, vy, vr, vb = self._variant_header_rect
        # Taller rows when the scene ships per-variant previews, matching the
        # scene dropdown; fall back to compact text-only rows otherwise.
        has_thumbs = bool(scene_option.variant_thumbnails)
        item_h = 80 if has_thumbs else 34
        items_top = vb + 2
        bg = (
            vx,
            items_top - 1,
            vr,
            items_top + len(scene_option.variants) * item_h + 1,
        )
        draw.rounded_rectangle(bg, radius=6, fill=(35, 35, 50, 255))
        draw.rounded_rectangle(bg, radius=6, outline=(60, 60, 80, 255), width=1)
        self._variant_item_rects = []
        for idx, variant in enumerate(scene_option.variants):
            top = items_top + idx * item_h
            rect = (vx, top, vr, top + item_h)
            self._variant_item_rects.append((rect, variant))
            if variant == self._selected_variant:
                draw.rectangle(rect, fill=ACTIVE_BG + (255,))
            elif variant == self._hovered_variant:
                draw.rectangle(rect, fill=HOVER_BG + (255,))
            text_x = rect[0] + 12
            text_y = top + item_h // 2 - 8
            if has_thumbs:
                thumb = self._get_variant_thumbnail(scene_option, variant)
                if thumb is not None:
                    tw, th = thumb.size
                    tx = rect[0] + 6
                    ty = top + max(0, (item_h - th) // 2)
                    canvas.paste(thumb, (tx, ty))
                    draw.rectangle(
                        (tx, ty, tx + tw, ty + th), outline=(60, 60, 80, 255), width=1
                    )
                    text_x = tx + tw + 10
            label = _truncate_text_to_width(
                self._font_tiny, variant, max(0, rect[2] - text_x - 8)
            )
            draw.text((text_x, text_y), label, fill=TEXT_COLOR, font=self._font_tiny)

    def _get_scene_thumbnail(self, scene: Any) -> Image.Image | None:
        if scene.path in self._scene_thumb_cache:
            return self._scene_thumb_cache[scene.path]
        if scene.thumbnail is None:
            self._scene_thumb_cache[scene.path] = None
            return None
        thumb = scene.thumbnail
        if thumb.mode != "RGBA":
            thumb = thumb.convert("RGBA")
        self._scene_thumb_cache[scene.path] = thumb
        return thumb

    def _get_variant_thumbnail(self, scene: Any, variant: str) -> Image.Image | None:
        key = (scene.path, variant)
        if key in self._variant_thumb_cache:
            return self._variant_thumb_cache[key]
        thumb = scene.variant_thumbnails.get(variant)
        if thumb is not None and thumb.mode != "RGBA":
            thumb = thumb.convert("RGBA")
        self._variant_thumb_cache[key] = thumb
        return thumb

    def _current_scene_option(self) -> Any:
        for option in self._scene_options:
            if self._scene_option_matches_current(option):
                return option
        return None

    def _scene_option_matches_current(self, option: Any) -> bool:
        current = self._current_scene
        if option.path == current or str(option.path) == str(current):
            return True
        for path in getattr(option, "variant_paths", {}).values():
            if path == current or str(path) == str(current):
                return True
        return False

    def _update_speed(self, wheel_state: Any) -> None:
        # Drive the digit from the authoritative ego speed
        # (``KeyboardState.vehicle_state``, same source as the MJPEG /state),
        # not the HUD's ``target_speed_mps`` integrator which drifts and never
        # resets on R/respawn. Magnitude only; reverse shows via the "R" box.
        telemetry = self._keyboard.vehicle_state
        if telemetry is not None:
            target_mph = abs(telemetry.speed_mps) * MPS_TO_MPH
        else:
            # No chunk yet (warmup / between scenes): hold at zero rather than
            # showing the integrator's creep-to-10mph target before the ego
            # has actually moved.
            target_mph = 0.0
        delta = target_mph - self._speed_mph
        self._speed_mph += delta * 0.18

    # -- Input -------------------------------------------------------

    def _build_key_codes(self) -> dict[str, Any]:
        spy = self._spy
        return {
            "escape": _lookup_key(spy.KeyCode, "escape"),
            "f11": _lookup_key(spy.KeyCode, "f11"),
            "w": _lookup_key(spy.KeyCode, "w"),
            "a": _lookup_key(spy.KeyCode, "a"),
            "s": _lookup_key(spy.KeyCode, "s"),
            "d": _lookup_key(spy.KeyCode, "d"),
            "r": _lookup_key(spy.KeyCode, "r"),
            "x": _lookup_key(spy.KeyCode, "x"),
            "space": _lookup_key(spy.KeyCode, "space"),
            "up": _lookup_key(spy.KeyCode, "up", "arrow_up"),
            "down": _lookup_key(spy.KeyCode, "down", "arrow_down"),
            "left": _lookup_key(spy.KeyCode, "left", "arrow_left"),
            "right": _lookup_key(spy.KeyCode, "right", "arrow_right"),
            "key1": _lookup_key(spy.KeyCode, "key1", "digit1", "num_1"),
            "key2": _lookup_key(spy.KeyCode, "key2", "digit2", "num_2"),
        }

    def _on_keyboard_event(self, event: Any) -> None:
        # Treat the dedicated ``is_key_repeat`` events as presses so OS
        # auto-repeat keeps the key marked "held" even on SDL3 builds
        # that interleave release+press around each repeat (the
        # observed source of the steering-jitter bug).
        is_press = event.is_key_press() if hasattr(event, "is_key_press") else False
        is_release = (
            event.is_key_release() if hasattr(event, "is_key_release") else False
        )
        is_repeat = event.is_key_repeat() if hasattr(event, "is_key_repeat") else False
        if not (is_press or is_release or is_repeat):
            return
        key = event.key
        if self._key_matches(key, "escape") and is_press:
            self._should_close_flag = True
            return
        # Drive keys flow through ``_keyboard_drive`` so the smoothed
        # steer / throttle / brake the wheel + speed-digit chrome reads
        # also reflects user input. The ``KeyboardDriveState.update()``
        # call inside ``_poll_drive_state`` (invoked unconditionally once
        # per tick from ``_render_canvas``) posts the smoothed values to
        # ``KeyboardState`` via ``set_drive``, so the simulation reads
        # the same values the chrome shows. (Bypassing this path and
        # writing to ``KeyboardState.set_key`` directly would be
        # ineffective: ``KeyboardState.command()`` gives ``_drive_command``
        # priority over the pressed-key set when set, and the per-frame
        # ``_keyboard_drive.update()`` always sets it.)
        drive_keysym = self._drive_keysym_for(key)
        if drive_keysym is not None:
            if is_press or is_repeat:
                # Press / repeat both reaffirm the key is held; cancel
                # any pending debounced release for this key.
                self._pending_drive_releases.pop(drive_keysym, None)
                self._keyboard_drive.set_key(drive_keysym, True)
                if drive_keysym == "space":
                    self._keyboard.set_key("space", True)
            else:
                # Schedule the release; per-frame ``_expire_pending_releases``
                # commits it after ``DRIVE_KEY_RELEASE_DEBOUNCE_S`` if no
                # press / repeat lands first. This filters out the
                # release+press cycles SDL3 sometimes emits for OS-level
                # key repeat.
                self._pending_drive_releases[drive_keysym] = time.monotonic()
            return
        if not is_press:
            return
        if self._key_matches(key, "key1"):
            self._keyboard.set_view_mode("model_rgb")
        elif self._key_matches(key, "key2"):
            self._keyboard.set_view_mode("rgb")
        elif self._key_matches(key, "r"):
            self._keyboard.request_reset()
        elif self._key_matches(key, "x"):
            self.exit_scene()

    def _expire_pending_drive_releases(self) -> None:
        """Commit any debounced release whose grace window has passed.

        Called once per render tick from :meth:`_render_canvas`. A
        release whose timestamp is older than
        ``DRIVE_KEY_RELEASE_DEBOUNCE_S`` is treated as final and
        propagated to ``_keyboard_drive`` (and ``KeyboardState`` for
        space). Anything younger stays pending; if a fresh press /
        repeat for the same key arrives in the meantime, the
        ``_on_keyboard_event`` handler discards the pending release.
        """
        if not self._pending_drive_releases:
            return
        now = time.monotonic()
        expired = [
            keysym
            for keysym, ts in self._pending_drive_releases.items()
            if now - ts >= DRIVE_KEY_RELEASE_DEBOUNCE_S
        ]
        for keysym in expired:
            self._keyboard_drive.set_key(keysym, False)
            if keysym == "space":
                self._keyboard.set_key("space", False)
            self._pending_drive_releases.pop(keysym, None)

    # Map slangpy ``KeyCode`` to the keysym vocabulary
    # :func:`omnidreams.interactive_drive.demo._keyboard_drive_key` expects;
    # cardinal arrows are capitalised ("Up"/"Down"/"Left"/"Right").
    _DRIVE_KEYSYMS: tuple[tuple[str, str], ...] = (
        ("w", "w"),
        ("a", "a"),
        ("s", "s"),
        ("d", "d"),
        ("up", "Up"),
        ("down", "Down"),
        ("left", "Left"),
        ("right", "Right"),
        ("space", "space"),
    )

    def _drive_keysym_for(self, event_key: Any) -> str | None:
        for name, keysym in self._DRIVE_KEYSYMS:
            if self._key_matches(event_key, name):
                return keysym
        return None

    def _key_matches(self, event_key: Any, name: str) -> bool:
        code = self._key_codes.get(name)
        return code is not None and event_key == code

    def _on_mouse_event(self, event: Any) -> None:
        spy = self._spy
        # ``pos`` is float2 in window-relative pixels. We round to int
        # for hit-testing against our integer panel rects.
        pos = event.pos
        try:
            self._mouse_pos = (int(pos.x), int(pos.y))
        except AttributeError:
            self._mouse_pos = (int(pos[0]), int(pos[1]))

        etype = event.type
        if etype == spy.MouseEventType.move:
            self._update_hover(self._mouse_pos)
            return
        if (
            etype == spy.MouseEventType.button_down
            and event.button == spy.MouseButton.left
        ):
            self._handle_click(self._mouse_pos)

    def _update_hover(self, pos: tuple[int, int]) -> None:
        self._hovered_scene_label = None
        self._hovered_variant = None
        if self._scene_dropdown_open:
            for rect, scene in self._scene_item_rects:
                if _rect_contains(rect, pos):
                    self._hovered_scene_label = scene.label
                    break
        if self._variant_dropdown_open:
            for rect, variant in self._variant_item_rects:
                if _rect_contains(rect, pos):
                    self._hovered_variant = variant
                    break

    def _handle_click(self, pos: tuple[int, int]) -> None:
        # While scenes are still preloading, the scene/variant dropdowns are
        # locked (the only mouse-clickable HUD elements), so ignore clicks
        # until every scene is cached and selection is instant.
        if self._scene_selection_locked():
            return
        # Variant dropdown sits on top of the scene dropdown items, so
        # check it first.
        if self._variant_dropdown_open:
            for rect, variant in self._variant_item_rects:
                if _rect_contains(rect, pos):
                    self._restart_variant(variant)
                    return
            if self._variant_header_rect and _rect_contains(
                self._variant_header_rect, pos
            ):
                self._variant_dropdown_open = False
                return
            self._variant_dropdown_open = False
            return

        if self._scene_dropdown_open:
            for rect, scene in self._scene_item_rects:
                if _rect_contains(rect, pos):
                    self._restart_backend(scene)
                    return
            if self._scene_header_rect and _rect_contains(self._scene_header_rect, pos):
                self._scene_dropdown_open = False
                return
            self._scene_dropdown_open = False
            return

        if self._scene_header_rect and _rect_contains(self._scene_header_rect, pos):
            self._scene_dropdown_open = True
            self._variant_dropdown_open = False
            self._panel_chrome_cache_key = None
            return

        # The variant dropdown is only meaningful once a scene is actually
        # loaded/running. Before that (the initial selection wait and the gap
        # between scene switches) the engine is inactive, so ignore clicks on
        # the variant header.
        current_scene_option = self._current_scene_option()
        if (
            self._engine_active
            and self._variant_header_rect
            and _rect_contains(self._variant_header_rect, pos)
            and current_scene_option is not None
            and len(current_scene_option.variants) > 1
        ):
            self._variant_dropdown_open = True
            self._scene_dropdown_open = False
            self._panel_chrome_cache_key = None

    # -- Scene / variant restart -------------------------------------

    def _restart_backend(self, scene: Any) -> None:
        logger.info(f"[demo] switching scene -> {scene.label}")
        new_variant = scene.variants[0] if scene.variants else "default"
        self._signal_scene_change(scene.path, new_variant)

    def _restart_variant(self, variant: str) -> None:
        if variant == self._selected_variant:
            self._variant_dropdown_open = False
            return
        logger.info(f"[demo] switching variant -> {variant}")
        self._variant_dropdown_open = False
        self._signal_scene_change(self._current_scene, variant)

    def _signal_scene_change(self, scene_path: Any, variant: str) -> None:
        """Tell the engine to exit (window stays alive) and stash the next scene.

        Sets ``_pending_scene_change`` + the close flag so ``run_main_loop``
        exits and the demo's outer loop loads the new scene over this same
        presenter, keeping the swapchain and resident model alive.
        """
        self._args.scene = scene_path
        self._args.variant = variant
        self._pending_scene_change = (scene_path, variant)
        # An explicit scene pick supersedes any pending exit-to-selection.
        self._pending_exit_scene = False
        self._should_close_flag = True
        # Drop the wheel-set DriverCommand so a stale steer/throttle doesn't
        # apply to the next scene before the user touches a key.
        self._keyboard.set_drive_command(None)

    @property
    def pending_scene_change(self) -> tuple[Any, str] | None:
        """``(scene_path, variant)`` if a dropdown click is pending, else None."""
        return self._pending_scene_change

    def exit_scene(self) -> None:
        """Request a return to the scene selector, keeping the window alive.

        Like :meth:`_signal_scene_change` but sets ``_pending_exit_scene`` so
        the outer loop re-enters :meth:`wait_for_scene_selection`. No-op unless
        a scene is running.
        """
        if not self._engine_active:
            return
        self._pending_exit_scene = True
        # An explicit exit overrides any scene change picked in the same tick.
        self._pending_scene_change = None
        self._should_close_flag = True
        # Clean input state so a stale steer/throttle can't leak into the
        # next scene the user eventually picks.
        self._keyboard.set_drive_command(None)

    @property
    def pending_exit_scene(self) -> bool:
        """True when the user asked to exit back to the scene selector."""
        return self._pending_exit_scene

    def acknowledge_exit_scene(self) -> None:
        """Clear the exit request and reset per-rollout view state for the selector.

        Called before the outer loop re-enters :meth:`wait_for_scene_selection`;
        resets the close flag, the selected variant, and the last rollout's
        camera/BEV/speed so the selector doesn't ghost them.
        """
        self._pending_exit_scene = False
        self._should_close_flag = False
        self._reset_selected_variant_to_default()
        self._reset_scene_view_state()

    def _reset_selected_variant_to_default(self) -> None:
        """Point ``_selected_variant`` at the current scene's first variant.

        Otherwise the "Variant:" header keeps showing the exited rollout's
        weather variant, which a fresh scene pick (always ``scene.variants[0]``)
        won't load. Falls back to ``"default"`` if the scene can't be resolved.
        """
        option = self._current_scene_option()
        self._selected_variant = (
            option.variants[0] if option is not None and option.variants else "default"
        )

    def set_model_status(
        self, *, can_prewarm: bool, ready_probe: Callable[[], bool]
    ) -> None:
        """Wire the camera-placeholder text to model-warmup progress.

        ``can_prewarm`` True (default world-model path) shows "Loading world
        model..." then "Ready - pick a scene"; False keeps "Load Scene".
        ``ready_probe`` (polled each tick) flips to ready once warmup finishes.
        """
        self._model_can_prewarm = bool(can_prewarm)
        self._model_ready_probe = ready_probe

    def set_scene_selection_locked(self, probe: Callable[[], bool]) -> None:
        """Lock scene/variant selection while ``probe()`` returns True (--preload-scenes).

        Dropdowns ignore clicks until every scene is cached; the placeholder
        shows a "Preloading scenes..." hint.
        """
        self._scene_selection_locked_probe = probe

    def _scene_selection_locked(self) -> bool:
        return self._scene_selection_locked_probe()

    def set_engine_active(self, active: bool) -> None:
        """Toggle the scene-running chrome / placeholder text.

        ``active=False`` is the selection wait and the gap between switches;
        ``True`` is a scene running/loading. Called by the demo around each run.
        """
        self._engine_active = bool(active)
        if not self._engine_active:
            # No scene loaded -> the variant dropdown isn't selectable, so a
            # previously-open one must not linger into the no-scene state.
            self._variant_dropdown_open = False
        # Drop the chrome cache so the panel redraws promptly (the cache key
        # doesn't include engine activity).
        self._panel_chrome_cache_key = None
        self._panel_chrome_cache = None

    def wait_for_scene_selection(self) -> tuple[Any, str] | None:
        """Run a chrome-only event loop until the user picks a scene.

        Opens the HUD window with no engine and a "Load Scene" placeholder;
        returns ``(scene_path, variant)`` on selection or ``None`` if the
        window closes first. ~60 fps (5 ms sleep) of chrome render + present.
        """
        prior_engine_active = self._engine_active
        self.set_engine_active(False)
        try:
            while not self.should_close:
                self.process_events()
                if self._pending_scene_change is not None:
                    request = self._pending_scene_change
                    return request
                # Render chrome + "Load Scene" placeholder.
                self._render_canvas(None)
                self._present_canvas()
                time.sleep(EVENT_POLL_INTERVAL_S)
            return None
        finally:
            self.set_engine_active(prior_engine_active)

    def wait_while_preloading(self, in_progress: Callable[[], bool]) -> None:
        """Pump the "Preloading scenes..." chrome until ``in_progress()`` clears.

        Used by ``--auto-start`` + ``--preload-scenes`` so the auto-loaded
        scene waits for the background preloader to finish (and is served from
        its cache) instead of racing it with a second parse of the same USDZ.
        Returns early if the window closes. Keeps the engine inactive so the
        camera area shows the locked "Preloading scenes..." placeholder.
        """
        prior_engine_active = self._engine_active
        self.set_engine_active(False)
        try:
            while in_progress() and not self.should_close:
                self.process_events()
                self._render_canvas(None)
                self._present_canvas()
                time.sleep(EVENT_POLL_INTERVAL_S)
        finally:
            self.set_engine_active(prior_engine_active)

    def acknowledge_scene_change(self, scene_path: Any, variant: str) -> None:
        """Accept the scene change and prep the presenter for the next scene.

        Resets the close flag, clears per-scene view state, and updates
        ``_current_scene`` / ``_selected_variant`` for the chrome.
        """
        self._pending_scene_change = None
        self._pending_exit_scene = False
        self._should_close_flag = False
        self._current_scene = scene_path
        self._selected_variant = variant
        self._reset_scene_view_state()

    def _reset_scene_view_state(self) -> None:
        """Drop all per-rollout view state (camera, BEV, chrome, speed, telemetry).

        Shared by :meth:`acknowledge_scene_change` and
        :meth:`acknowledge_exit_scene` so the next state starts clean instead
        of ghosting the just-ended rollout.
        """
        self._scene_dropdown_open = False
        self._variant_dropdown_open = False
        # The next backend renders into a fresh ``rgb_host_uint8`` buffer
        # so the camera resize cache (keyed on ``id(buffer)``) is now
        # stale; drop it. Same for the BEV cache.
        self._camera_resize_cache_key = None
        self._camera_resize_cache = None
        self._latest_camera_pil = None
        self._latest_bev_pil = None
        self._bev_panel_cache_key = None
        self._bev_panel_cache = None
        # Panel chrome shows the scene label, so its cache key changes
        # naturally; explicitly invalidate to be safe.
        self._panel_chrome_cache_key = None
        self._panel_chrome_cache = None
        self._has_camera_frame = False
        self._speed_mph = 0.0
        # Forget the previous rollout's speed so the digit doesn't ramp back
        # toward it; a new rollout republishes telemetry as soon as it starts.
        self._keyboard.clear_telemetry()
        self._pending_drive_releases.clear()

    def set_wheel(self, wheel: Any | None) -> None:
        """Attach (or detach) a :class:`WheelBridge` after construction.

        The demo builds the wheel after the engine (so its sink targets the
        app's keyboard) and attaches it here, before the selection wait, so the
        steering / pedal chrome reacts to the device while picking a scene.
        """
        self._wheel = wheel

    def bind_keyboard(self, keyboard: KeyboardState) -> None:
        """Rebind to the engine's long-lived ``KeyboardState``.

        :class:`InteractiveDriveApp` calls this once at construction; updates
        our reference + the ``KeyboardDriveState`` smoother that wraps it.
        """
        from omnidreams.interactive_drive.demo import KeyboardDriveState

        self._keyboard = keyboard
        self._keyboard_drive = KeyboardDriveState(KeyboardStateDriveSink(keyboard))


# -- Module-level helpers ---------------------------------------------


def _lookup_key(key_enum: Any, *names: str) -> Any:
    for name in names:
        value = getattr(key_enum, name, None)
        if value is not None:
            return value
    return None


def _rect_contains(rect: tuple[int, int, int, int], pos: tuple[int, int]) -> bool:
    x, y = pos
    return rect[0] <= x < rect[2] and rect[1] <= y < rect[3]


def _prefetch_to_numpy(frame: object) -> None:
    prefetch = getattr(frame, "prefetch_to_numpy", None)
    if callable(prefetch):
        prefetch()


def _has_cuda_tensor(frame: object) -> bool:
    return callable(getattr(frame, "to_cuda_tensor", None))


def _as_rgb_host_uint8(frame: object) -> np.ndarray:
    to_numpy = getattr(frame, "to_numpy", None)
    if callable(to_numpy):
        frame = to_numpy()
    return np.ascontiguousarray(np.asarray(frame, dtype=np.uint8)[..., :3])


__all__ = [
    "KeyboardStateDriveSink",
    "SlangPyHudPresenter",
]
