# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Server-rendered status and camera-control HUD for Cam2V applications."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from torch import Tensor

from flashdreams.api_v2.loop import IUILoop
from flashdreams.runtime.keyboard import KeyboardState, normalize_key
from flashdreams.runtime_v2.step_result import InputEventTrace, StepResult
from flashdreams.runtime_v2.user_input_event import (
    FocusUserInputEvent,
    KeyboardInputState,
    KeyboardUserInputEvent,
    ResetUserInputEvent,
)
from flashdreams.runtime_v2.user_input_events import UserInputEvents

_REFERENCE_WIDTH = 832
_REFERENCE_HEIGHT = 464

_PANEL_BG = (16, 17, 18, 190)
_PANEL_BORDER = (255, 255, 255, 38)
_TEXT = (243, 247, 240, 255)
_MUTED = (185, 192, 191, 255)
_ACCENT = (142, 240, 28, 255)
_ACCENT_STRONG = (169, 255, 47, 255)
_WARNING = (255, 191, 79, 255)
_KEY_BG = (12, 14, 15, 196)
_KEY_BORDER = (255, 255, 255, 64)
_KEY_ACTIVE_BG = (142, 240, 28, 78)
_KEY_ACTIVE_BORDER = (142, 240, 28, 220)

_PRESENTATION_STREAM_PRIORITY = -1
"""Prefer short interactive presentation kernels over queued model work."""


@dataclass(frozen=True, slots=True)
class _ControlGroup:
    label: str
    keys: tuple[str, ...]


_CONTROL_GROUPS = (
    _ControlGroup("DRIVE / TURN", ("w", "a", "s", "d")),
    _ControlGroup("STRAFE", ("q", "e")),
    _ControlGroup("PITCH", ("i", "k")),
    _ControlGroup("LOOK", ("j", "l")),
)
"""Lingbot's browser control grouping, retained in the server-side HUD."""

_CAMERA_KEY_ORDER = tuple(key for group in _CONTROL_GROUPS for key in group.keys)
"""Stable order used for caching and rendering held camera controls."""

_CAMERA_KEYS = frozenset(_CAMERA_KEY_ORDER)
"""Keyboard controls recognized by the shared camera pose integrator."""

RECENT_MODEL_FPS_WINDOW_SECONDS = 2.0
"""Trailing AR-step completion window displayed in the model status panel."""

_IDLE_STATUS_REFRESH_SECONDS = 0.25
"""Maximum four-Hz HUD refresh while a recent model rate can expire."""


@dataclass(frozen=True, slots=True)
class Cam2VModelStepTiming:
    """One completed autoregressive model step used for recent throughput."""

    completed_at: float
    """Timestamp recorded after the step and required CUDA synchronization."""

    frame_count: int
    """Generated frames returned by this autoregressive step."""

    wall_s: float
    """Wall time for input preparation, generation, finalization, and CUDA sync."""


@dataclass(frozen=True, slots=True)
class Cam2VUIStatus:
    """Latest model-generation status copied to the UI loop."""

    completed_blocks: int
    """Number of autoregressive blocks completed in this rollout."""

    frames_generated: int
    """Number of video frames generated in this rollout."""

    chunk_fps: float
    """Frame throughput measured across the latest model step."""

    recent_model_steps: tuple[Cam2VModelStepTiming, ...] | None
    """Post-warmup model steps completed in the recent sampling window."""

    model_step_wall_s: float
    """Wall time spent producing the latest model chunk."""

    def recent_model_fps(self, now: float | None = None) -> float | None:
        """Return recent post-warmup model-step throughput at ``now``."""
        steps = self.recent_model_steps
        if steps is None:
            return None
        return _recent_model_fps(
            steps,
            now=time.perf_counter() if now is None else now,
            window_seconds=RECENT_MODEL_FPS_WINDOW_SECONDS,
        )


@dataclass(slots=True)
class Cam2VUIState:
    """Mutable Cam2V HUD state owned exclusively by the UI loop."""

    total_blocks: int
    """Number of autoregressive blocks requested for the rollout."""

    target_fps: int
    """Configured generated-video frame rate."""

    warmup_blocks: int
    """Leading blocks excluded from steady-state throughput."""

    held_keys: set[str] = field(default_factory=set)
    """Camera-control keys currently held by the client."""

    _keyboard_state: KeyboardState = field(
        default_factory=lambda: KeyboardState(supported_keys=_CAMERA_KEYS),
    )
    """UI-thread-owned source-aware keyboard state."""

    status: Cam2VUIStatus | None = None
    """Latest model status received from the model-generation loop."""

    def update_status(self, status: Cam2VUIStatus) -> None:
        """Replace the displayed model-generation status."""
        self.status = status

    def reset(self) -> None:
        """Clear transient controls and model status for a new generation."""
        self.held_keys.clear()
        self._keyboard_state = KeyboardState(supported_keys=_CAMERA_KEYS)
        self.status = None


@dataclass(frozen=True, slots=True)
class _Cam2VHUDLayout:
    """Resolved HUD geometry cached by one renderer."""

    scale: float
    controls_rect: tuple[int, int, int, int]
    status_rect: tuple[int, int, int, int]
    key_rects: dict[str, tuple[int, int, int, int]]
    group_labels: tuple[tuple[str, tuple[int, int]], ...]


class _Cam2VHUDRenderer(Protocol):
    def render(
        self,
        state: Cam2VUIState,
        *,
        device: torch.device,
        dtype: torch.dtype,
        status_sampled_at: float | None = None,
    ) -> Tensor:
        """Return a normalized RGBA HUD shaped ``[4, H, W]``."""
        ...

    def reset(self) -> None:
        """Clear renderer caches after a rollout reset."""
        ...

    def presentation_stream(self, device: torch.device) -> torch.cuda.Stream:
        """Return the renderer-owned stream used for CUDA composition."""
        ...

    def close(self) -> None:
        """Release renderer resources."""
        ...


@dataclass(slots=True)
class _UploadSlot:
    """Pinned host storage retained until its asynchronous upload completes."""

    host: Tensor
    ready: torch.cuda.Event | None = None


class Cam2VHUDRenderer:
    """Rasterize the legacy Lingbot visual language into a torch RGBA layer.

    Static Pillow chrome is drawn once and control layers are reused across
    status updates. Host overlays stay as compact uint8 tensors. CUDA uploads
    use the session's high-priority presentation stream and a two-slot pinned
    staging pool, so the io-thread avoids a synchronous full-float-frame copy.
    """

    def __init__(
        self,
        *,
        width: int,
        height: int,
        device: torch.device | None = None,
        presentation_stream: torch.cuda.Stream | None = None,
    ) -> None:
        if width <= 0 or height <= 0:
            raise ValueError("Cam2V HUD dimensions must be > 0.")
        self.width = int(width)
        self.height = int(height)
        self.layout = _build_layout(self.width, self.height)
        self._fonts = _build_fonts(self.layout.scale)
        self._static_image = _draw_static_hud(self)
        self._controls_signature: tuple[str, ...] | None = None
        self._controls_image: Image.Image | None = None
        self._cache_signature: tuple[object, ...] | None = None
        self._cached_rgba8: Tensor | None = None
        self._cached_output: Tensor | None = None
        self._cached_device: torch.device | None = None
        self._cached_dtype: torch.dtype | None = None
        self._cached_ready: torch.cuda.Event | None = None
        self._upload_device: torch.device | None = None
        self._upload_stream = presentation_stream
        self._upload_slots: list[_UploadSlot] = []
        self._next_upload_slot = 0

        if presentation_stream is not None:
            self._upload_device = _canonical_device(presentation_stream.device)

        if device is not None:
            target_device = _canonical_device(torch.device(device))
            if target_device.type == "cuda" and torch.cuda.is_available():
                self._initialize_cuda_upload(target_device)

    def render(
        self,
        state: Cam2VUIState,
        *,
        device: torch.device,
        dtype: torch.dtype,
        status_sampled_at: float | None = None,
    ) -> Tensor:
        """Draw or reuse the current HUD on ``device`` with ``dtype``."""
        if not dtype.is_floating_point:
            raise ValueError("Cam2V HUD output dtype must be floating point.")
        held_keys = tuple(key for key in _CAMERA_KEY_ORDER if key in state.held_keys)
        recent_model_fps = (
            None
            if state.status is None
            else state.status.recent_model_fps(status_sampled_at)
        )
        signature = (
            state.total_blocks,
            state.target_fps,
            state.warmup_blocks,
            state.status,
            None if recent_model_fps is None else round(recent_model_fps, 1),
            held_keys,
        )
        if signature != self._cache_signature or self._cached_rgba8 is None:
            rgba8 = _draw_hud(self, state, recent_model_fps=recent_model_fps)
            self._cached_rgba8 = torch.from_numpy(rgba8).permute(2, 0, 1).contiguous()
            self._cache_signature = signature
            self._cached_output = None
            self._cached_ready = None

        target_device = _canonical_device(torch.device(device))
        if (
            self._cached_output is None
            or self._cached_device != target_device
            or self._cached_dtype != dtype
        ):
            assert self._cached_rgba8 is not None
            if target_device.type == "cuda":
                self._cached_output, self._cached_ready = self._upload_cuda(
                    self._cached_rgba8,
                    device=target_device,
                    dtype=dtype,
                )
            else:
                self._cached_output = _normalize_rgba8(
                    self._cached_rgba8,
                    device=target_device,
                    dtype=dtype,
                )
                self._cached_ready = None
            self._cached_device = target_device
            self._cached_dtype = dtype

        if target_device.type == "cuda":
            assert self._cached_ready is not None
            consumer = torch.cuda.current_stream(target_device)
            consumer.wait_event(self._cached_ready)
            self._cached_output.record_stream(consumer)
        return self._cached_output

    def reset(self) -> None:
        """Invalidate dynamic caches after a new rollout starts."""
        self._synchronize_uploads()
        self._cache_signature = None
        self._cached_rgba8 = None
        self._cached_output = None
        self._cached_device = None
        self._cached_dtype = None
        self._cached_ready = None
        self._controls_signature = None
        self._controls_image = None
        for slot in self._upload_slots:
            slot.ready = None

    def close(self) -> None:
        """Release cached host and device tensors."""
        self.reset()
        self._upload_slots.clear()
        self._upload_stream = None
        self._upload_device = None

    def presentation_stream(self, device: torch.device) -> torch.cuda.Stream:
        """Return the high-priority stream shared by HUD upload and composition."""
        target_device = _canonical_device(device)
        self._initialize_cuda_upload(target_device)
        assert self._upload_stream is not None
        return self._upload_stream

    def _initialize_cuda_upload(self, device: torch.device) -> None:
        """Allocate upload resources before model generation starts."""
        if self._upload_device == device and self._upload_slots:
            return
        if self._upload_stream is not None and self._upload_device != device:
            raise ValueError("The Cam2V presentation stream and HUD device must match.")
        self._synchronize_uploads()
        self._upload_slots.clear()
        with torch.cuda.device(device):
            if self._upload_stream is None:
                self._upload_stream = torch.cuda.Stream(
                    device=device,
                    priority=_PRESENTATION_STREAM_PRIORITY,
                )
            self._upload_slots = [
                _UploadSlot(
                    torch.empty(
                        (4, self.height, self.width),
                        dtype=torch.uint8,
                        device="cpu",
                        pin_memory=True,
                    )
                )
                for _ in range(2)
            ]
        self._upload_device = device
        self._next_upload_slot = 0

    def _upload_cuda(
        self,
        rgba8: Tensor,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[Tensor, torch.cuda.Event]:
        """Enqueue one compact host overlay upload on the presentation stream."""
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA HUD rendering requested but CUDA is unavailable.")
        self._initialize_cuda_upload(device)
        assert self._upload_stream is not None
        slot = self._acquire_upload_slot()
        slot.host.copy_(rgba8)
        with torch.cuda.stream(self._upload_stream):
            raw = slot.host.to(device=device, non_blocking=True)
            output = raw.to(dtype=dtype)
            output[:3].mul_(2.0 / 255.0).sub_(1.0)
            output[3:4].mul_(1.0 / 255.0)
            raw.record_stream(self._upload_stream)
            ready = torch.cuda.Event()
            ready.record(self._upload_stream)
        slot.ready = ready
        return output, ready

    def _acquire_upload_slot(self) -> _UploadSlot:
        """Return a staging slot whose preceding transfer has completed."""
        slot_count = len(self._upload_slots)
        for offset in range(slot_count):
            index = (self._next_upload_slot + offset) % slot_count
            slot = self._upload_slots[index]
            if slot.ready is None or slot.ready.query():
                self._next_upload_slot = (index + 1) % slot_count
                return slot
        # Status changes are much less frequent than transfers. This is a rare
        # safety path that prevents overwriting pinned storage still in flight.
        slot = self._upload_slots[self._next_upload_slot]
        assert slot.ready is not None
        slot.ready.synchronize()
        self._next_upload_slot = (self._next_upload_slot + 1) % slot_count
        return slot

    def _synchronize_uploads(self) -> None:
        if self._upload_stream is not None:
            self._upload_stream.synchronize()


def _canonical_device(device: torch.device) -> torch.device:
    if device.type == "cuda" and device.index is None and torch.cuda.is_available():
        return torch.device("cuda", torch.cuda.current_device())
    return device


def _normalize_rgba8(
    rgba8: Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    output = rgba8.to(device=device, dtype=dtype)
    output[:3].mul_(2.0 / 255.0).sub_(1.0)
    output[3:4].mul_(1.0 / 255.0)
    return output


class Cam2VHUDLoop(IUILoop[Cam2VUIState]):
    """Composite a styled Cam2V HUD over the currently presented model frame."""

    def __init__(
        self,
        *,
        renderer: _Cam2VHUDRenderer | None = None,
        width: int | None = None,
        height: int | None = None,
        device: torch.device | None = None,
        presentation_stream: torch.cuda.Stream | None = None,
    ) -> None:
        if renderer is None:
            if width is None or height is None:
                raise ValueError(
                    "width and height are required when renderer is not supplied."
                )
            renderer = Cam2VHUDRenderer(
                width=width,
                height=height,
                device=device,
                presentation_stream=presentation_stream,
            )
        self.renderer = renderer
        self._last_redraw_at: float | None = None
        self._last_rendered_recent_fps: float | None = None

    def should_redraw_for_input(self, events: UserInputEvents) -> bool:
        """Redraw only for input that can change the server-rendered HUD."""
        for event in events.get_events():
            if isinstance(event, ResetUserInputEvent):
                return True
            if isinstance(event, FocusUserInputEvent):
                if not event.focused:
                    return True
                continue
            if (
                isinstance(event, KeyboardUserInputEvent)
                and normalize_key(event.key) in _CAMERA_KEYS
            ):
                return True
        return False

    def should_redraw_on_idle(self) -> bool:
        """Refresh an expired recent-model rate at most four times per second."""
        status = self.state.status
        last_redraw_at = self._last_redraw_at
        if status is None or last_redraw_at is None:
            return False
        now = time.perf_counter()
        if now - last_redraw_at < _IDLE_STATUS_REFRESH_SECONDS:
            return False
        recent_model_fps = status.recent_model_fps(now)
        rounded_fps = None if recent_model_fps is None else round(recent_model_fps, 1)
        return rounded_fps != self._last_rendered_recent_fps

    def step(self, step_index: int, events: UserInputEvents) -> StepResult:
        """Apply browser input and draw one server-rendered HUD frame."""
        _apply_ui_input(self.state, events)
        status_sampled_at = time.perf_counter()
        recent_model_fps = (
            None
            if self.state.status is None
            else self.state.status.recent_model_fps(status_sampled_at)
        )
        # Peek only to select the device-specific presentation stream. The
        # accessor below establishes the producer dependency before any tensor
        # operation is enqueued.
        frame = self._presentation_manager.presented_frame(0)
        device = frame.device if frame is not None else torch.device("cpu")
        dtype = (
            frame.dtype
            if frame is not None and frame.is_floating_point()
            else torch.float32
        )
        output_ready_event = None
        if device.type == "cuda":
            presentation_stream = self.renderer.presentation_stream(device)
            if _canonical_device(presentation_stream.device) != _canonical_device(
                device
            ):
                raise ValueError("The Cam2V presentation stream device changed.")
            frame = self.presented_model_frame(0, stream=presentation_stream)
            with torch.cuda.device(device), torch.cuda.stream(presentation_stream):
                back_buffer = _normalized_float_frame(frame)
                overlay = self.renderer.render(
                    self.state,
                    device=device,
                    dtype=dtype,
                    status_sampled_at=status_sampled_at,
                )
                output = self._presentation_manager.composite(back_buffer, overlay)
                output_ready_event = torch.cuda.Event()
                output_ready_event.record(presentation_stream)
        else:
            frame = self.presented_model_frame(0)
            back_buffer = _normalized_float_frame(frame)
            overlay = self.renderer.render(
                self.state,
                device=device,
                dtype=dtype,
                status_sampled_at=status_sampled_at,
            )
            output = self._presentation_manager.composite(back_buffer, overlay)
        self._last_redraw_at = time.perf_counter()
        self._last_rendered_recent_fps = (
            None if recent_model_fps is None else round(recent_model_fps, 1)
        )
        return StepResult(
            step_index=step_index,
            output=output.unsqueeze(0),
            frame_count=1,
            output_layout=self.output_layout,
            input_event_traces=_ui_input_event_traces(events),
            output_ready_event=output_ready_event,
        )

    def reset(self) -> None:
        """Clear UI-loop state for a new generation."""
        self.state.reset()
        self.renderer.reset()
        self._last_redraw_at = None
        self._last_rendered_recent_fps = None

    def close(self) -> None:
        """Release HUD rendering resources."""
        self.renderer.close()


def _build_layout(width: int, height: int) -> _Cam2VHUDLayout:
    scale = min(width / _REFERENCE_WIDTH, height / _REFERENCE_HEIGHT)
    scale = min(1.25, max(0.25, scale))

    def px(value: float) -> int:
        return max(1, round(value * scale))

    margin = px(18)

    controls_width = min(px(390), max(1, width - 2 * margin))
    controls_height = min(px(246), max(1, height - 2 * margin))
    controls_rect = (
        margin,
        max(margin, height - margin - controls_height),
        margin + controls_width,
        max(margin, height - margin - controls_height) + controls_height,
    )

    status_width = min(px(254), max(1, width - 2 * margin))
    status_height = min(px(118), max(1, height - 2 * margin))
    status_rect = (
        max(margin, width - margin - status_width),
        margin,
        max(margin, width - margin - status_width) + status_width,
        margin + status_height,
    )

    key_size = px(34)
    key_gap = px(7)
    row_gap = px(7)
    rows_top = controls_rect[1] + px(48)
    keys_left = controls_rect[0] + px(17)
    label_left = controls_rect[0] + px(194)
    key_rects: dict[str, tuple[int, int, int, int]] = {}
    group_labels: list[tuple[str, tuple[int, int]]] = []
    for row_index, group in enumerate(_CONTROL_GROUPS):
        top = rows_top + row_index * (key_size + row_gap)
        for key_index, key in enumerate(group.keys):
            left = keys_left + key_index * (key_size + key_gap)
            key_rects[key] = (left, top, left + key_size, top + key_size)
        group_labels.append((group.label, (label_left, top + key_size // 2)))

    return _Cam2VHUDLayout(
        scale=scale,
        controls_rect=controls_rect,
        status_rect=status_rect,
        key_rects=key_rects,
        group_labels=tuple(group_labels),
    )


def _build_fonts(
    scale: float,
) -> dict[str, ImageFont.ImageFont | ImageFont.FreeTypeFont]:
    def font(size: float) -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
        return ImageFont.load_default(size=max(6, round(size * scale)))

    return {
        "tiny": font(10),
        "small": font(12),
        "body": font(14),
        "key": font(17),
        "rate": font(24),
        "brand": font(18),
    }


def _draw_static_hud(renderer: Cam2VHUDRenderer) -> Image.Image:
    """Render shadows, panels, branding, and inactive controls once."""
    image = Image.new("RGBA", (renderer.width, renderer.height), (0, 0, 0, 0))
    layout = renderer.layout
    if renderer.width < 160 or renderer.height < 96:
        return image

    def px(value: float) -> int:
        return max(1, round(value * layout.scale))

    _draw_panel_shadow(image, layout.controls_rect, radius=px(9))
    _draw_panel_shadow(image, layout.status_rect, radius=px(9))
    draw = ImageDraw.Draw(image, "RGBA")
    draw.rounded_rectangle(
        layout.controls_rect,
        radius=px(8),
        fill=_PANEL_BG,
        outline=_PANEL_BORDER,
        width=px(1),
    )
    draw.rounded_rectangle(
        layout.status_rect,
        radius=px(8),
        fill=_PANEL_BG,
        outline=_PANEL_BORDER,
        width=px(1),
    )

    _draw_brand(draw, renderer)
    _draw_static_controls(draw, renderer)
    return image


def _draw_hud(
    renderer: Cam2VHUDRenderer,
    state: Cam2VUIState,
    *,
    recent_model_fps: float | None,
) -> np.ndarray:
    if renderer.width < 160 or renderer.height < 96:
        image = Image.new("RGBA", (renderer.width, renderer.height), (0, 0, 0, 0))
        _draw_compact_hud(ImageDraw.Draw(image, "RGBA"), renderer, state)
        return np.asarray(image, dtype=np.uint8).copy()

    image = renderer._static_image.copy()
    held_keys = tuple(key for key in _CAMERA_KEY_ORDER if key in state.held_keys)
    if held_keys != renderer._controls_signature or renderer._controls_image is None:
        renderer._controls_image = _draw_active_controls(renderer, held_keys)
        renderer._controls_signature = held_keys
    image.alpha_composite(renderer._controls_image)
    draw = ImageDraw.Draw(image, "RGBA")
    _draw_status(draw, renderer, state, recent_model_fps=recent_model_fps)
    return np.asarray(image, dtype=np.uint8).copy()


def _draw_compact_hud(
    draw: ImageDraw.ImageDraw,
    renderer: Cam2VHUDRenderer,
    state: Cam2VUIState,
) -> None:
    """Draw bounded chrome when a test or integration uses a tiny frame."""
    right = renderer.width - 1
    bottom = renderer.height - 1
    border = _ACCENT if state.held_keys else _PANEL_BORDER
    draw.rectangle(
        (0, 0, right, bottom),
        fill=(16, 17, 18, 150),
        outline=border,
        width=1,
    )
    if renderer.width < 48 or renderer.height < 14:
        return
    label = "+".join(key.upper() for key in _CAMERA_KEY_ORDER if key in state.held_keys)
    draw.text(
        (4, renderer.height // 2),
        label or "CAM2V",
        fill=_ACCENT_STRONG if label else _TEXT,
        font=renderer._fonts["tiny"],
        anchor="lm",
    )


def _draw_panel_shadow(
    image: Image.Image,
    rect: tuple[int, int, int, int],
    *,
    radius: int,
) -> None:
    shadow = Image.new("RGBA", image.size, (0, 0, 0, 0))
    shadow_draw = ImageDraw.Draw(shadow, "RGBA")
    offset = max(1, radius // 2)
    shifted = (rect[0], rect[1] + offset, rect[2], rect[3] + offset)
    shadow_draw.rounded_rectangle(shifted, radius=radius, fill=(0, 0, 0, 120))
    image.alpha_composite(shadow.filter(ImageFilter.GaussianBlur(radius)))


def _draw_brand(draw: ImageDraw.ImageDraw, renderer: Cam2VHUDRenderer) -> None:
    scale = renderer.layout.scale

    def px(value: float) -> int:
        return max(1, round(value * scale))

    x = px(18)
    y = px(18)
    draw.rounded_rectangle(
        (x, y, x + px(4), y + px(40)),
        radius=px(2),
        fill=_ACCENT,
    )
    draw.text(
        (x + px(13), y - px(1)),
        "FLASHDREAMS",
        fill=_TEXT,
        font=renderer._fonts["brand"],
    )
    draw.text(
        (x + px(13), y + px(24)),
        "CAMERA  /  WORLD MODEL",
        fill=_MUTED,
        font=renderer._fonts["tiny"],
    )


def _draw_static_controls(
    draw: ImageDraw.ImageDraw,
    renderer: Cam2VHUDRenderer,
) -> None:
    layout = renderer.layout
    scale = layout.scale

    def px(value: float) -> int:
        return max(1, round(value * scale))

    left, top, _, bottom = layout.controls_rect
    title_x = left + px(17)
    title_y = top + px(14)
    draw.rounded_rectangle(
        (title_x, title_y, title_x + px(3), title_y + px(20)),
        radius=px(2),
        fill=_ACCENT,
    )
    draw.text(
        (title_x + px(11), title_y - px(1)),
        "CAMERA CONTROLS",
        fill=_TEXT,
        font=renderer._fonts["body"],
    )

    for key in layout.key_rects:
        _draw_key(draw, renderer, key, active=False)

    for label, position in layout.group_labels:
        draw.text(
            position,
            label,
            fill=_TEXT,
            font=renderer._fonts["small"],
            anchor="lm",
        )

    hint = "HOLD KEYS OR ARROWS  ·  CLICK VIDEO TO FOCUS"
    draw.text(
        (left + px(17), bottom - px(16)),
        hint,
        fill=_MUTED,
        font=renderer._fonts["tiny"],
        anchor="lm",
    )


def _draw_active_controls(
    renderer: Cam2VHUDRenderer,
    held_keys: tuple[str, ...],
) -> Image.Image:
    """Render only changing key highlights over the inactive control chrome."""
    image = Image.new("RGBA", (renderer.width, renderer.height), (0, 0, 0, 0))
    if not held_keys:
        return image
    layout = renderer.layout

    def px(value: float) -> int:
        return max(1, round(value * layout.scale))

    glow = Image.new("RGBA", image.size, (0, 0, 0, 0))
    glow_draw = ImageDraw.Draw(glow, "RGBA")
    for key in held_keys:
        rect = layout.key_rects[key]
        expanded = (
            rect[0] - px(3),
            rect[1] - px(3),
            rect[2] + px(3),
            rect[3] + px(3),
        )
        glow_draw.rounded_rectangle(
            expanded,
            radius=px(7),
            fill=(142, 240, 28, 78),
        )
    image.alpha_composite(glow.filter(ImageFilter.GaussianBlur(px(6))))
    draw = ImageDraw.Draw(image, "RGBA")
    for key in held_keys:
        _draw_key(draw, renderer, key, active=True)
    return image


def _draw_key(
    draw: ImageDraw.ImageDraw,
    renderer: Cam2VHUDRenderer,
    key: str,
    *,
    active: bool,
) -> None:
    scale = renderer.layout.scale

    def px(value: float) -> int:
        return max(1, round(value * scale))

    base_rect = renderer.layout.key_rects[key]
    offset = px(1) if active else 0
    rect = (
        base_rect[0],
        base_rect[1] + offset,
        base_rect[2],
        base_rect[3] + offset,
    )
    draw.rounded_rectangle(
        rect,
        radius=px(6),
        fill=_KEY_ACTIVE_BG if active else _KEY_BG,
        outline=_KEY_ACTIVE_BORDER if active else _KEY_BORDER,
        width=px(1),
    )
    if not active:
        draw.line(
            (rect[0] + px(4), rect[3] - px(2), rect[2] - px(4), rect[3] - px(2)),
            fill=(0, 0, 0, 90),
            width=px(2),
        )
    draw.text(
        ((rect[0] + rect[2]) // 2, (rect[1] + rect[3]) // 2),
        key.upper(),
        fill=_ACCENT_STRONG if active else _TEXT,
        font=renderer._fonts["key"],
        anchor="mm",
    )


def _draw_status(
    draw: ImageDraw.ImageDraw,
    renderer: Cam2VHUDRenderer,
    state: Cam2VUIState,
    *,
    recent_model_fps: float | None,
) -> None:
    layout = renderer.layout
    scale = layout.scale

    def px(value: float) -> int:
        return max(1, round(value * scale))

    left, top, right, bottom = layout.status_rect
    status = state.status

    has_recent_output = (
        status is not None and recent_model_fps is not None and recent_model_fps > 0.0
    )
    is_complete = status is not None and status.completed_blocks >= state.total_blocks
    if is_complete:
        status_label = "MODEL COMPLETE"
    elif has_recent_output:
        status_label = "MODEL GENERATION"
    elif recent_model_fps is not None:
        status_label = "WAITING FOR OUTPUT"
    elif status is not None:
        status_label = "MODEL WARMUP"
    else:
        status_label = "WAITING FOR MODEL"
    dot_color = _ACCENT if has_recent_output or is_complete else _WARNING
    dot_x = left + px(17)
    dot_y = top + px(20)
    dot_radius = px(4)
    draw.ellipse(
        (
            dot_x - dot_radius,
            dot_y - dot_radius,
            dot_x + dot_radius,
            dot_y + dot_radius,
        ),
        fill=dot_color,
    )
    draw.text(
        (dot_x + px(11), dot_y),
        status_label,
        fill=_MUTED,
        font=renderer._fonts["tiny"],
        anchor="lm",
    )

    if status is None:
        rate_text = "AR -- FPS"
        detail = f"VIDEO RATE {state.target_fps} FPS  ·  WAITING FOR FIRST CHUNK"
        completed = 0
    elif recent_model_fps is None:
        display_rate = status.chunk_fps
        rate_text = f"AR {display_rate:.1f} FPS"
        detail = (
            f"WARMUP {min(status.completed_blocks, state.warmup_blocks)} / "
            f"{state.warmup_blocks}  ·  "
            f"{status.frames_generated} FRAMES  ·  "
            f"{status.model_step_wall_s * 1_000.0:.0f} MS"
        )
        completed = status.completed_blocks
    else:
        rate_text = f"AR {recent_model_fps:.1f} FPS"
        detail = (
            f"BLOCK {status.completed_blocks} / {state.total_blocks}  ·  "
            f"{status.frames_generated} FRAMES  ·  "
            f"{status.model_step_wall_s * 1_000.0:.0f} MS"
        )
        completed = status.completed_blocks
    draw.text(
        (left + px(17), top + px(34)),
        rate_text,
        fill=(
            _ACCENT_STRONG
            if has_recent_output or is_complete
            else _WARNING
            if status is not None
            else _TEXT
        ),
        font=renderer._fonts["rate"],
    )
    draw.text(
        (left + px(17), top + px(72)),
        detail,
        fill=_MUTED,
        font=renderer._fonts["tiny"],
    )
    bar_rect = (
        left + px(17),
        bottom - px(17),
        right - px(17),
        bottom - px(11),
    )
    draw.rounded_rectangle(bar_rect, radius=px(3), fill=(255, 255, 255, 28))
    fraction = min(1.0, max(0.0, completed / max(1, state.total_blocks)))
    if fraction > 0.0:
        fill_right = bar_rect[0] + max(1, round((bar_rect[2] - bar_rect[0]) * fraction))
        draw.rounded_rectangle(
            (bar_rect[0], bar_rect[1], fill_right, bar_rect[3]),
            radius=px(3),
            fill=_ACCENT,
        )


def _recent_model_fps(
    steps: tuple[Cam2VModelStepTiming, ...],
    *,
    now: float,
    window_seconds: float,
) -> float:
    """Aggregate model-step throughput completed in a trailing time window."""
    if not steps:
        return 0.0
    now = max(float(now), steps[-1].completed_at)
    cutoff = now - window_seconds
    recent_steps = tuple(step for step in steps if step.completed_at > cutoff)
    elapsed_s = sum(step.wall_s for step in recent_steps)
    if elapsed_s <= 0.0:
        return 0.0
    return sum(step.frame_count for step in recent_steps) / elapsed_s


def _normalized_float_frame(frame: Tensor | None) -> Tensor | None:
    if frame is None:
        return None
    if frame.is_floating_point():
        return frame
    return frame.to(torch.float32).mul_(2.0 / 255.0).sub_(1.0)


def _apply_ui_input(state: Cam2VUIState, events: UserInputEvents) -> None:
    for event in events.get_events():
        if isinstance(event, FocusUserInputEvent) and not event.focused:
            state.held_keys.clear()
            state._keyboard_state = KeyboardState(supported_keys=_CAMERA_KEYS)
            continue
        if not isinstance(event, KeyboardUserInputEvent):
            continue
        if not state._keyboard_state.apply_event(
            event=("keydown" if event.state is KeyboardInputState.PRESSED else "keyup"),
            key=event.key,
        ):
            continue
        state.held_keys.clear()
        state.held_keys.update(state._keyboard_state.snapshot())


def _ui_input_event_traces(
    events: UserInputEvents,
) -> tuple[InputEventTrace, ...]:
    """Acknowledge every correlated HUD control processed by this UI step."""
    return tuple(
        InputEventTrace(event_id=event.event_id, frame_index=0)
        for event in events.get_events()
        if isinstance(event, KeyboardUserInputEvent)
        and event.event_id is not None
        and normalize_key(event.key) in _CAMERA_KEYS
    )


__all__ = [
    "Cam2VHUDLoop",
    "Cam2VHUDRenderer",
    "Cam2VModelStepTiming",
    "Cam2VUIState",
    "Cam2VUIStatus",
]
