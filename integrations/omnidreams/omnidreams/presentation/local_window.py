# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""SlangPy-backed local-window presenter: Vulkan swapchain plus PIL chrome.

Owns the window, device, surface, and camera composite for the
``local-window`` output mode. Everything model-specific -- layout, chrome
pixels, and what a key or click means -- is delegated to a
:class:`~omnidreams.presentation.base.HudOverlay`, so an integration adds a
native demo by writing an overlay rather than another presenter.
"""

from __future__ import annotations

import contextlib
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
from loguru import logger
from PIL import Image, ImageDraw

from omnidreams.presentation.base import (
    HudOverlay,
    InputSink,
    KeyAction,
    KeyEvent,
    PointerEvent,
    Rect,
)
from omnidreams.presentation.canvas import (
    allocate_canvas,
    draw_status_overlay,
    fit_rect,
    resolve_font,
)
from omnidreams.presentation.cuda_interop import CudaRGBInterop
from omnidreams.presentation.frame import (
    DisplayFrame,
    as_rgb_host_uint8,
    has_cuda_tensor,
    prefetch_frame,
    rgb_source_size,
)

_ARROW_ALIASES: dict[str, tuple[str, ...]] = {
    "up": ("up", "arrow_up"),
    "down": ("down", "arrow_down"),
    "left": ("left", "arrow_left"),
    "right": ("right", "arrow_right"),
}
"""SlangPy has spelled the cardinal arrows both ways across releases."""

_NAMED_KEYS: tuple[str, ...] = (
    "escape",
    "space",
    "enter",
    "tab",
    "backspace",
    *(f"f{index}" for index in range(1, 13)),
)

_POINTER_BUTTONS: tuple[str, ...] = ("left", "middle", "right")


@dataclass(frozen=True, kw_only=True, slots=True)
class WindowConfig:
    """Window, canvas, and status-overlay settings for a local-window presenter."""

    width: int = 1920
    height: int = 1080

    title: str = "flashdreams"

    resizable: bool = True

    min_height: int = 360
    """Floor applied when auto-resizing to a source frame, so a short frame
    cannot collapse the window to an unusable height."""

    background: tuple[int, int, int] = (20, 20, 30)
    """Canvas clear colour, also used for camera letterbox bars."""

    text_color: tuple[int, int, int] = (220, 220, 230)
    """Status-overlay text colour."""

    status_font_size: int = 44

    auto_resize_to_source: bool = True
    """Grow the window when a source frame needs more room than the current
    camera area, keeping native pixels instead of downscaling. Later user
    resizes stay authoritative until the source resolution changes again."""


class LocalWindowPresenter:
    """Present frames into a resizable SlangPy/Vulkan window with PIL chrome.

    Composites through one of three paths per tick, in preference order:
    CUDA interop (camera stays on the GPU, chrome uploads as an RGBA
    overlay), host upload with a hardware blit for the camera, and a full
    CPU composite used while a status message is showing.
    """

    def __init__(
        self,
        *,
        overlay: HudOverlay,
        input_sink: InputSink | None = None,
        config: WindowConfig | None = None,
        cuda_interop_disabled: bool = False,
    ) -> None:
        try:
            import slangpy as spy
        except ImportError as exc:
            raise RuntimeError(
                "SlangPy is required for the local-window presenter; install with"
                " `uv sync --package flashdreams-omnidreams --extra interactive-drive`."
            ) from exc

        self._spy = spy
        self._config = config or WindowConfig()
        self._overlay = overlay
        self._input_sink = input_sink
        self._cuda_interop_requested = not cuda_interop_disabled
        self._cuda_interop_unavailable_reason: str | None = None
        self._cuda_error_logged = False
        self._cuda_resize_logged = False
        self._should_close_flag = False

        self._window = spy.Window(
            width=self._config.width,
            height=self._config.height,
            title=self._config.title,
            resizable=self._config.resizable,
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
        # Trust the ACTUAL window size after creation rather than the requested
        # defaults: SDL3 may clamp the window to fit the display (or scale for
        # HiDPI), and configuring a surface at the wrong size makes
        # ``acquire_next_image`` fail at first present with a generic SLANG_FAIL.
        self._configured_size = self._current_window_size()
        self._configure_surface(*self._configured_size)
        self._display_texture = self._build_display_texture(*self._configured_size)
        self._cuda_interop = self._create_cuda_interop(*self._configured_size)
        self._retired_cuda_interops: list[CudaRGBInterop] = []

        # Set by the on_resize callback on the windowing thread and consumed by
        # ``present_frame`` on the main thread, where rebuilding Vulkan
        # resources is safe.
        self._pending_resize: tuple[int, int] | None = None
        self._auto_sized_source_size: tuple[int, int] | None = None

        self._canvas_buffer, self._canvas = allocate_canvas(
            *self._configured_size, background=self._config.background
        )
        self._status_font = resolve_font(self._config.status_font_size)

        self._camera_image: Image.Image | None = None
        self._camera_src_size: tuple[int, int] | None = None
        self._camera_rgba: np.ndarray | None = None
        self._camera_rgba_staging: np.ndarray | None = None
        self._camera_texture: Any | None = None
        self._camera_texture_size: tuple[int, int] | None = None
        self._camera_fit_texture: Any | None = None
        self._camera_fit_size: tuple[int, int] | None = None
        self._camera_resize_cache: Image.Image | None = None
        self._camera_resize_cache_key: tuple[int, int, int] | None = None
        self._has_camera_frame = False

        self._key_names = self._build_key_names()
        self._pointer_buttons = self._build_pointer_buttons()
        self._window.on_resize = self._on_resize
        self._window.on_keyboard_event = self._on_keyboard_event
        self._window.on_mouse_event = self._on_mouse_event

    ## PresenterBackend protocol

    @property
    def should_close(self) -> bool:
        return self._should_close_flag or self._window.should_close()

    def process_events(self) -> None:
        self._window.process_events()

    def prepare_frame(self, frame: DisplayFrame) -> None:
        """Start host materialization off the presentation path.

        Skipped for a CUDA-resident image while interop is live, since that
        path never needs a host copy.
        """
        image = frame.image
        if image is not None and (
            self._cuda_interop is None or not has_cuda_tensor(image)
        ):
            prefetch_frame(image)
        self._overlay.prepare(frame)

    def present_frame(self, frame: DisplayFrame) -> None:
        # Apply a pending resize before touching the display texture, so Vulkan
        # resources are only ever rebuilt on this thread.
        if self._pending_resize is not None:
            width, height = self._pending_resize
            self._pending_resize = None
            self._apply_resize(width, height)

        image = frame.image
        if image is not None and self._resize_window_for_source(image):
            # Resources are rebuilt at the start of the next tick. Drop this
            # transition frame rather than presenting old-size buffers against
            # the newly resized window.
            return

        try:
            if self._present_cuda_frame(frame):
                return
        except Exception as exc:  # noqa: BLE001 -- interop is never required
            self._disable_cuda_interop(exc)

        if image is not None:
            self._update_camera_image(image)
        self._render_canvas(frame)
        self._present_canvas(use_gpu_camera=frame.status_message is None)

    def close(self) -> None:
        self._should_close_flag = True
        with contextlib.suppress(Exception):
            self._overlay.close()
        if self._cuda_interop is not None:
            with contextlib.suppress(Exception):
                self._cuda_interop.close()
            self._cuda_interop = None
        for interop in self._retired_cuda_interops:
            with contextlib.suppress(Exception):
                interop.close()
        self._retired_cuda_interops.clear()
        with contextlib.suppress(Exception):
            self._window.close()

    ## Status presentation

    def present_status(self, message: str, *, process_events: bool = True) -> None:
        """Paint ``message`` over the current canvas during blocking setup work."""
        if process_events:
            self.process_events()
        self._render_canvas(DisplayFrame(status_message=message))
        self._present_canvas(use_gpu_camera=False)

    @property
    def window_size(self) -> tuple[int, int]:
        """Current configured canvas size in pixels."""
        return self._configured_size

    ## CUDA composite path

    def _present_cuda_frame(self, frame: DisplayFrame) -> bool:
        """Composite this frame through CUDA interop.

        Returns:
            ``True`` when the frame was handled (including when the producer
            event is not ready yet and the tick was intentionally skipped),
            ``False`` to fall through to the host path.
        """
        if self._cuda_interop is None or frame.image is None:
            return False

        cuda_frame = self._cuda_interop.as_cuda_rgb_source(frame.image)
        if cuda_frame is None:
            return False

        if not cuda_frame.ready:
            self._submit_ready_cuda_buffer()
            return True

        self._has_camera_frame = True
        self._render_canvas(frame, camera_transparent=True)
        overlay_rgba = np.array(self._canvas, dtype=np.uint8)

        submitted = self._submit_ready_cuda_buffer()
        queued = self._cuda_interop.enqueue_camera_to_shared_rgba(
            cuda_frame,
            overlay_rgba=overlay_rgba,
            camera_area=self._camera_area(),
            background=self._config.background,
        )
        if not queued:
            return True
        if not submitted:
            self._submit_ready_cuda_buffer()
        return True

    def _submit_ready_cuda_buffer(self) -> bool:
        interop = self._cuda_interop
        if interop is None:
            return False
        ready = interop.ready_rgba_buffer()
        if ready is None:
            return False
        rgba_buffer, _cuda_stream = ready
        self._sync_window_size()
        # ``_sync_window_size`` can rebuild interop; the buffer we hold would
        # then belong to a retired ring.
        if self._cuda_interop is not interop:
            return False
        if not self._surface.config:
            return False
        surface_texture = self._acquire_surface_texture()
        if surface_texture is None:
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
            submit_id = self._device.submit_command_buffer(encoder.finish())
            interop.mark_submitted(rgba_buffer, submit_id)
            self._surface.present()
            del surface_texture
        except RuntimeError as exc:
            logger.warning(
                f"[presenter] swapchain present failed ({exc}); reconfiguring",
            )
            self._reconfigure_surface()
            return False
        return True

    def _disable_cuda_interop(self, exc: BaseException) -> None:
        if not self._cuda_error_logged:
            logger.warning(
                "[presenter] cuda_interop=failed; disabling and using host "
                f"upload ({exc})",
            )
            self._cuda_error_logged = True
        if self._cuda_interop is not None:
            with contextlib.suppress(Exception):
                self._cuda_interop.close()
            self._cuda_interop = None

    ## Host composite path

    def _present_canvas(self, *, use_gpu_camera: bool) -> None:
        # SDL3 doesn't always fire on_resize for compositor-side resizes
        # (window manager fitting on first map, HiDPI scaling), so compare
        # against the live window size every tick.
        self._sync_window_size()
        if not self._surface.config:
            return
        surface_texture = self._acquire_surface_texture()
        if surface_texture is None:
            return
        # ``_canvas_buffer`` is the same memory PIL drew into this tick, so
        # this is a direct upload with no PIL-to-numpy memcpy.
        try:
            self._display_texture.copy_from_numpy(self._canvas_buffer)
            encoder = self._device.create_command_encoder()
            if use_gpu_camera:
                self._composite_camera_gpu(encoder)
            encoder.blit(surface_texture, self._display_texture)
            self._device.submit_command_buffer(encoder.finish())
            self._surface.present()
            del surface_texture
        except RuntimeError as exc:
            logger.warning(
                f"[presenter] swapchain present failed ({exc}); reconfiguring",
            )
            self._reconfigure_surface()

    def _composite_camera_gpu(self, encoder: Any) -> None:
        """Stamp the camera frame into the display texture on the GPU.

        Hardware bilinear blit plus a sub-region copy over the chrome canvas
        the caller already uploaded, filling only the centred fit rect.
        """
        if self._camera_src_size is None:
            return
        target = fit_rect(source_size=self._camera_src_size, area=self._camera_area())
        if target is None:
            return
        offset_x, offset_y, right, bottom = target
        fit_w, fit_h = right - offset_x, bottom - offset_y
        if not self._ensure_camera_texture_uploaded():
            return
        self._ensure_camera_fit_texture(fit_w, fit_h)
        # Whole-extent blit with a linear filter is a hardware bilinear resize,
        # ~0.1 ms against the ~5 ms a PIL resize costs at this size.
        encoder.blit(self._camera_fit_texture, self._camera_texture)
        # Int-layer / int-mip overload: this slangpy build's SubresourceRange
        # constructor only accepts a dict, not kwargs.
        spy = self._spy
        encoder.copy_texture(
            self._display_texture,
            0,
            0,
            spy.math.uint3(offset_x, offset_y, 0),
            self._camera_fit_texture,
            0,
            0,
            spy.math.uint3(0, 0, 0),
        )

    def _ensure_camera_texture_uploaded(self) -> bool:
        """Upload the latest camera frame into the source-sized GPU texture.

        Reuses one alpha-pre-filled RGBA staging buffer per source size, so
        the per-tick cost is a single RGB slice copy rather than a fresh
        allocation and concatenate.
        """
        if self._camera_image is None or self._camera_src_size is None:
            return False
        src_w, src_h = self._camera_src_size
        if self._camera_texture is None or self._camera_texture_size != (src_w, src_h):
            spy = self._spy
            self._camera_texture = self._device.create_texture(
                format=spy.Format.rgba8_unorm,
                width=src_w,
                height=src_h,
                usage=spy.TextureUsage.shader_resource
                | spy.TextureUsage.unordered_access,
                label="local_window_camera_src",
            )
            self._camera_texture_size = (src_w, src_h)
            self._camera_rgba = None
            self._camera_rgba_staging = None
        if self._camera_rgba_staging is None or self._camera_rgba_staging.shape[:2] != (
            src_h,
            src_w,
        ):
            self._camera_rgba_staging = np.empty((src_h, src_w, 4), dtype=np.uint8)
            # One-time alpha fill; this path only ever writes the RGB slice.
            self._camera_rgba_staging[..., 3] = 255
            self._camera_rgba = None
        if self._camera_rgba is None:
            self._camera_rgba_staging[..., :3] = np.asarray(self._camera_image)
            self._camera_rgba = self._camera_rgba_staging
        self._camera_texture.copy_from_numpy(self._camera_rgba)
        return True

    def _ensure_camera_fit_texture(self, fit_w: int, fit_h: int) -> None:
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
            label="local_window_camera_fit",
        )
        self._camera_fit_size = (fit_w, fit_h)

    def _update_camera_image(self, image: object) -> None:
        rgb = as_rgb_host_uint8(image)
        # ``Image.fromarray`` over a contiguous buffer is zero-copy at the C
        # level; this image is only ever used as a paste source, which does
        # not trigger a copy either.
        if not rgb.flags["C_CONTIGUOUS"]:
            rgb = np.ascontiguousarray(rgb)
        self._camera_image = Image.fromarray(rgb, mode="RGB")
        src_h, src_w = rgb.shape[:2]
        self._camera_src_size = (src_w, src_h)
        # Producers reuse their scratch buffers, so identity is stable across
        # frames with different contents; drop the derived caches explicitly
        # rather than relying on a key comparison.
        self._camera_rgba = None
        self._camera_resize_cache_key = None
        self._camera_resize_cache = None
        self._has_camera_frame = True

    ## Canvas rendering

    def _camera_area(self) -> Rect:
        return self._overlay.camera_area(self._canvas.size)

    def _render_canvas(
        self, frame: DisplayFrame, *, camera_transparent: bool = False
    ) -> None:
        """Composite camera and chrome into the canvas for this tick.

        No full-canvas clear: the overlay and camera paths cover their own
        regions every frame and the letterbox bars stay at the background
        colour, which saves a 2 MP RGBA fill per tick at 1080p.
        """
        canvas = self._canvas
        camera_area = self._camera_area()
        draw = ImageDraw.Draw(canvas)
        background = self._config.background

        if camera_transparent:
            # The GPU composite supplies the camera pixels, so leave a hole
            # for it and let chrome drawn afterwards sit on top.
            draw.rectangle(camera_area, fill=(0, 0, 0, 0))
        elif self._camera_image is not None:
            if frame.status_message is None:
                # The GPU fills the centred fit rect after this canvas is
                # uploaded; repaint only the letterbox bars so they don't
                # show the previous frame when the fit rect changes size.
                draw.rectangle(camera_area, fill=background + (255,))
            else:
                # CPU composite so the status callout sits over the image.
                self._draw_camera(canvas, camera_area)
        else:
            # Wipe first so the previous tick doesn't ghost behind the
            # placeholder; placeholder ticks only, so cheaper than an
            # always-on full-canvas clear.
            draw.rectangle(camera_area, fill=background + (255,))
            self._overlay.draw_placeholder(canvas, draw, camera_area=camera_area)

        self._overlay.draw(canvas, draw, frame=frame, camera_area=camera_area)

        if frame.status_message:
            draw_status_overlay(
                draw,
                area=camera_area,
                message=frame.status_message,
                font=self._status_font,
                text_color=self._config.text_color,
            )

    def _draw_camera(self, canvas: Image.Image, area: Rect) -> None:
        camera = self._camera_image
        if camera is None:
            return
        target = fit_rect(source_size=camera.size, area=area)
        if target is None:
            return
        left, top, right, bottom = target
        target_w, target_h = right - left, bottom - top
        cache_key = (id(camera), target_w, target_h)
        if (
            cache_key != self._camera_resize_cache_key
            or self._camera_resize_cache is None
        ):
            if (target_w, target_h) == camera.size:
                resized = camera
            else:
                resized = camera.resize(
                    (target_w, target_h),
                    Image.Resampling.LANCZOS
                    if target_w < camera.size[0]
                    else Image.Resampling.BILINEAR,
                )
            self._camera_resize_cache = resized
            self._camera_resize_cache_key = cache_key
        else:
            resized = self._camera_resize_cache
        if resized.mode == "RGBA":
            canvas.alpha_composite(resized, (left, top))
        else:
            canvas.paste(resized, (left, top))

    ## Device, surface, and resize plumbing

    def _create_device(self) -> Any:
        existing_device_handles = self._cuda_existing_device_handles()
        enable_cuda_interop = self._cuda_interop_requested and bool(
            existing_device_handles
        )
        if not self._cuda_interop_requested:
            self._cuda_interop_unavailable_reason = "disabled by caller"
        elif not existing_device_handles:
            self._cuda_interop_unavailable_reason = "CUDA context unavailable"
        device_kwargs: dict[str, Any] = {
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
                "[presenter] CUDA interop device creation failed; retrying Vulkan "
                f"without interop ({exc})",
            )
            self._cuda_interop_unavailable_reason = "device creation failed"
            return self._spy.Device(
                type=self._spy.DeviceType.vulkan,
                enable_debug_layers=False,
                enable_cuda_launch_from_gfx=False,
                enable_ray_tracing=False,
            )

    def _cuda_existing_device_handles(self) -> list[Any]:
        if not self._cuda_interop_requested:
            return []
        try:
            import torch
        except ImportError:
            return []
        try:
            if not torch.cuda.is_initialized():
                torch.cuda.init()
            # Presenters are constructed before the model backend, so CUDA is
            # otherwise still lazy here. Materialize the primary context on
            # this thread so slangpy binds interop to the device the backend
            # will go on to use.
            torch.cuda.current_stream()
        except Exception:  # noqa: BLE001 -- absent CUDA just means no interop
            return []

        get_handles = getattr(
            self._spy, "get_cuda_current_context_native_handles", None
        )
        if not callable(get_handles):
            return []
        try:
            handles: Any = get_handles()
            return list(handles)
        except Exception:  # noqa: BLE001 -- absent CUDA just means no interop
            return []

    def _create_cuda_interop(self, width: int, height: int) -> CudaRGBInterop | None:
        if not self._cuda_interop_requested:
            logger.info("[presenter] cuda_interop=disabled; using host upload")
            return None
        if not self._device.supports_cuda_interop:
            reason = self._cuda_interop_unavailable_reason or "unsupported"
            logger.info(f"[presenter] cuda_interop={reason}; using host upload")
            return None
        try:
            interop = CudaRGBInterop(
                spy=self._spy,
                device=self._device,
                width=width,
                height=height,
            )
        except Exception as exc:  # noqa: BLE001 -- interop is never required
            logger.warning(
                f"[presenter] cuda_interop=unavailable; using host upload ({exc})",
            )
            return None
        logger.info("[presenter] cuda_interop=enabled")
        return interop

    def _choose_surface_format(self) -> Any:
        """Pick a linear surface format so no implicit sRGB encode is applied.

        Mismatched gamma between the display texture and the swapchain washes
        colours out, so require a linear format the surface advertises.

        Raises:
            RuntimeError: The surface offers no linear format.
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
            "Presenter requires a linear swapchain, but the surface only "
            f"supports: {supported}"
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
            label="local_window_display_texture",
        )

    def _acquire_surface_texture(self) -> Any | None:
        try:
            surface_texture = self._surface.acquire_next_image()
        except RuntimeError as exc:
            # NVIDIA's Vulkan driver reports VK_ERROR_OUT_OF_DATE_KHR as a
            # generic SLANG_FAIL when the swapchain has drifted from the
            # surface -- typically an unreported resize, or the OS reclaiming
            # an idle swapchain. Reconfiguring at the live window size fixes
            # it and the next tick retries.
            logger.warning(
                f"[presenter] swapchain acquire failed ({exc}); reconfiguring",
            )
            self._reconfigure_surface()
            return None
        if not surface_texture:
            time.sleep(0.001)
            return None
        return surface_texture

    def _apply_resize(self, width: int, height: int, *, force: bool = False) -> bool:
        width, height = self._normalise_size(width, height)
        previous_size = self._configured_size
        size_changed = (width, height) != previous_size
        if not force and not size_changed:
            return True
        try:
            display_texture = self._build_display_texture(width, height)
            canvas_buffer, canvas = allocate_canvas(
                width, height, background=self._config.background
            )
            self._configure_surface(width, height)
        except Exception as exc:  # noqa: BLE001 -- keep presenting at the old size
            logger.warning(
                "[presenter] window resize failed; keeping previous texture size "
                f"{previous_size} ({exc})",
            )
            return False
        self._configured_size = (width, height)
        # Only presenter-owned display resources are rebuilt. Render/inference
        # resolution is fixed elsewhere; this texture is the swapchain upload
        # target and nothing more.
        self._display_texture = display_texture
        if size_changed:
            self._recreate_cuda_interop_after_resize(width, height)
        self._canvas_buffer, self._canvas = canvas_buffer, canvas
        # The fit texture is sized from the camera area, which moved. The
        # source-sized camera texture only tracks producer dimensions and
        # stays valid across window resizes.
        self._camera_fit_texture = None
        self._camera_fit_size = None
        self._camera_resize_cache_key = None
        self._camera_resize_cache = None
        return True

    def _on_resize(self, width: int, height: int) -> None:
        # Runs on the windowing thread; stash only. ``present_frame`` rebuilds
        # Vulkan resources on the next tick, where it cannot race a frame in
        # flight.
        self._pending_resize = self._normalise_size(width, height)

    def _resize_window_for_source(self, image: object) -> bool:
        """Grow the window when a source resolution needs more room.

        Smaller frames stay centred at native resolution rather than being
        upscaled. Larger frames grow only the dimensions required to fit the
        source alongside whatever width the overlay reserves.

        Returns:
            ``True`` when a resize was requested and this frame should be
            dropped while presentation resources are rebuilt.
        """
        if not self._config.auto_resize_to_source:
            return False
        source_size = rgb_source_size(image)
        if source_size is None or source_size == self._auto_sized_source_size:
            return False
        source_width, source_height = source_size
        current_width, current_height = self._current_window_size()
        left, _top, right, _bottom = self._camera_area()
        reserved_width = max(0, self._canvas.size[0] - (right - left))
        target_size = (
            max(current_width, source_width + reserved_width),
            max(current_height, source_height, self._config.min_height),
        )
        if target_size == (current_width, current_height):
            self._auto_sized_source_size = source_size
            return False
        try:
            self._window.resize(*target_size)
        except Exception as exc:  # noqa: BLE001 -- keep presenting at the old size
            logger.warning(
                "[presenter] source-driven window resize failed "
                f"source={source_size} target={target_size} ({exc})",
            )
            return False
        # Some SDL/window-manager combinations deliver the resize callback
        # asynchronously, so record the request too; the next tick then always
        # rebuilds before presenting at the new dimensions.
        self._pending_resize = target_size
        self._auto_sized_source_size = source_size
        logger.info(
            "[presenter] source-driven window resize "
            f"source={source_size} target={target_size}",
        )
        return True

    def _sync_window_size(self) -> None:
        new_size = self._current_window_size()
        if new_size != self._configured_size:
            self._apply_resize(*new_size)

    def _reconfigure_surface(self) -> None:
        self._apply_resize(*self._current_window_size(), force=True)

    def _normalise_size(self, width: int, height: int) -> tuple[int, int]:
        return max(1, int(width)), max(1, int(height))

    def _current_window_size(self) -> tuple[int, int]:
        actual = self._window.size
        return self._normalise_size(actual.x, actual.y)

    def _recreate_cuda_interop_after_resize(self, width: int, height: int) -> None:
        if self._cuda_interop is None:
            return
        # The old ring may still have buffers referenced by in-flight submits,
        # so retire it for teardown instead of closing it here.
        self._retired_cuda_interops.append(self._cuda_interop)
        self._cuda_interop = self._create_cuda_interop(width, height)
        if self._cuda_interop is not None:
            logger.info("[presenter] cuda_interop=recreated after window resize")
            self._cuda_resize_logged = False
            return
        if not self._cuda_resize_logged:
            logger.warning(
                "[presenter] cuda_interop=disabled after window resize; could not "
                "recreate shared CUDA/Vulkan resources",
            )
            self._cuda_resize_logged = True

    ## Input routing

    def _on_keyboard_event(self, event: Any) -> None:
        action = _key_action(event)
        if action is None:
            return
        key = self._key_names.get(event.key)
        if key is None:
            return
        if key == "escape" and action == "press":
            self._should_close_flag = True
            return
        normalized = KeyEvent(key=key, action=action, timestamp_s=time.monotonic())
        if self._overlay.on_key(normalized):
            return
        if self._input_sink is not None:
            self._input_sink.key_event(normalized)

    def _on_mouse_event(self, event: Any) -> None:
        spy = self._spy
        # ``pos`` is float2 in window-relative pixels; round for integer
        # hit-testing against chrome rectangles.
        pos = event.pos
        try:
            position = (int(pos.x), int(pos.y))
        except AttributeError:
            position = (int(pos[0]), int(pos[1]))

        event_type = event.type
        if event_type == spy.MouseEventType.move:
            action = "move"
            button = None
        elif event_type == spy.MouseEventType.button_down:
            action = "press"
            button = self._pointer_buttons.get(event.button)
        elif event_type == spy.MouseEventType.button_up:
            action = "release"
            button = self._pointer_buttons.get(event.button)
        else:
            return

        normalized = PointerEvent(
            action=action,
            position=position,
            timestamp_s=time.monotonic(),
            button=button,
        )
        if self._overlay.on_pointer(normalized):
            return
        if self._input_sink is not None:
            self._input_sink.pointer_event(normalized)

    def _build_key_names(self) -> dict[Any, str]:
        """Map slangpy key codes to normalized names, skipping absent codes."""
        key_code = self._spy.KeyCode
        names: dict[Any, str] = {}

        def register(name: str, *candidates: str) -> None:
            for candidate in candidates or (name,):
                code = getattr(key_code, candidate, None)
                if code is not None:
                    names.setdefault(code, name)
                    return

        for letter in "abcdefghijklmnopqrstuvwxyz":
            register(letter)
        for digit in range(10):
            register(str(digit), f"key{digit}", f"digit{digit}", f"num_{digit}")
        for name, candidates in _ARROW_ALIASES.items():
            register(name, *candidates)
        for name in _NAMED_KEYS:
            register(name)
        return names

    def _build_pointer_buttons(self) -> dict[Any, str]:
        button_enum = self._spy.MouseButton
        buttons: dict[Any, str] = {}
        for name in _POINTER_BUTTONS:
            code = getattr(button_enum, name, None)
            if code is not None:
                buttons[code] = name
        return buttons


def _key_action(event: Any) -> KeyAction | None:
    """Classify a slangpy keyboard event, or ``None`` when it is not a transition."""
    if getattr(event, "is_key_press", None) and event.is_key_press():
        return "press"
    if getattr(event, "is_key_release", None) and event.is_key_release():
        return "release"
    if getattr(event, "is_key_repeat", None) and event.is_key_repeat():
        return "repeat"
    return None


__all__ = ["LocalWindowPresenter", "WindowConfig"]
