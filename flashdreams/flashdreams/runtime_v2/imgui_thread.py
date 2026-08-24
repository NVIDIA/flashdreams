# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SlangPy ImGui UI thread and renderer."""

import importlib
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, Generic, Protocol, TypeVar, cast, final

import torch
from torch import Tensor

from flashdreams.api_v2.thread import UIThread
from flashdreams.runtime_v2.presentation_manager import PresentationManager
from flashdreams.runtime_v2.step_result import StepResult
from flashdreams.runtime_v2.user_input_event import (
    KeyboardInputState,
    KeyboardUserInputEventData,
    MouseUserInputEventData,
)
from flashdreams.runtime_v2.user_input_events import UserInputEvents
from flashdreams.runtime_v2.video_tensor import VideoTensorLayout

StateT = TypeVar("StateT")


class ImGUIRenderer(Protocol):
    """Rendering backend needed by :class:`ImGUIThread`."""

    def render(
        self,
        step_index: int,
        events: UserInputEvents,
        draw_ui: Callable[[Any, int, UserInputEvents], None],
    ) -> Tensor:
        """Render one UI frame as normalized ``[C, H, W]`` output."""
        ...

    def reset(self) -> None:
        """Reset renderer input and transient state."""
        ...

    def close(self) -> None:
        """Release renderer resources."""
        ...


class ImGUIThread(UIThread[StateT], ABC, Generic[StateT]):
    """Render an ImGui UI over an optional model frame."""

    def __init__(
        self,
        *,
        state: StateT,
        frequency: int,
        output_layout: VideoTensorLayout,
        presentation_manager: PresentationManager,
        renderer: ImGUIRenderer | None = None,
        width: int | None = None,
        height: int | None = None,
    ) -> None:
        """Configure an ImGui thread without creating GPU resources.

        Args:
            state: State used by the UI thread.
            frequency: Maximum UI iterations per second.
            output_layout: Layout used for the compositing result.
            presentation_manager: Buffer containing model frames.
            renderer: Rendering backend; ``None`` creates the SlangPy backend.
            width: Render-target width, required for the default renderer.
            height: Render-target height, required for the default renderer.

        Raises:
            ValueError: ``output_layout`` is not ``tchw``, or the default
                renderer has no output dimensions.
        """
        if output_layout is not VideoTensorLayout.tchw:
            raise ValueError("ImGui rendering requires tchw output.")
        super().__init__(
            state=state,
            frequency=frequency,
            output_layout=output_layout,
            presentation_manager=presentation_manager,
        )
        if renderer is None:
            if width is None or height is None:
                raise ValueError(
                    "width and height are required when renderer is not supplied."
                )
            renderer = SlangPyImGUIRenderer(width=width, height=height)
        self.renderer = renderer

    @abstractmethod
    def draw_ui(
        self, ui: Any, step_index: int, events: UserInputEvents
    ) -> Tensor | None:
        """Draw widgets and optionally return the frame beneath them."""
        ...

    @final
    def step_ui(self, step_index: int, events: UserInputEvents) -> StepResult:
        """Render ImGui over the optional back-buffer returned by :meth:`draw_ui`."""
        back_buffer: Tensor | None = None

        def draw(ui: Any, index: int, current_events: UserInputEvents) -> None:
            nonlocal back_buffer
            back_buffer = self.draw_ui(ui, index, current_events)

        overlay = self.renderer.render(step_index, events, draw)
        frame = self._presentation_manager.composite(back_buffer, overlay)
        return StepResult(
            step_index=step_index,
            output=frame.unsqueeze(0),
            frame_count=1,
            output_layout=self.output_layout,
        )

    def reset(self) -> None:
        """Reset renderer state after a session reset event."""
        self.renderer.reset()

    def close(self) -> None:
        """Release the renderer."""
        self.renderer.close()


class SlangPyImGUIRenderer:
    """Render SlangPy's native ImGui widgets through CUDA interop."""

    def __init__(
        self,
        *,
        width: int,
        height: int,
        slangpy_module: Any | None = None,
    ) -> None:
        """Configure a renderer whose native resources are created lazily.

        Args:
            width: Render-target width in pixels.
            height: Render-target height in pixels.
            slangpy_module: Injected SlangPy module for tests.

        Raises:
            ValueError: A render dimension is not positive.
        """
        if width <= 0 or height <= 0:
            raise ValueError("ImGui render dimensions must be > 0.")
        self.width = int(width)
        self.height = int(height)
        self._slangpy = slangpy_module
        self._ui: _SlangPyUI | None = None
        self._device: Any | None = None
        self._ui_context: Any | None = None
        self._target: Any | None = None
        self._rgba_buffer: Any | None = None
        self._rgba_tensor: Tensor | None = None
        self._rgba_buffer_size = 0
        self._rgba_row_pitch = 0
        self._has_rendered = False

    def render(
        self,
        step_index: int,
        events: UserInputEvents,
        draw_ui: Callable[[Any, int, UserInputEvents], None],
    ) -> Tensor:
        """Queue input and render one ImGui frame into shared RGBA storage."""
        self._ensure_initialized()
        assert self._device is not None
        assert self._slangpy is not None
        assert self._ui is not None
        assert self._ui_context is not None
        assert self._target is not None
        assert self._rgba_buffer is not None
        assert self._rgba_tensor is not None

        if self._has_rendered:
            self._device.sync_to_cuda(_current_cuda_stream())
        _route_input_events(
            events,
            ui_context=self._ui_context,
            slangpy=self._slangpy,
            width=self.width,
            height=self.height,
        )
        draw_ui(self._ui, step_index, events)
        self._ui_context.begin_frame(self.width, self.height)

        encoder = self._device.create_command_encoder()
        encoder.clear_texture_float(
            self._target,
            clear_value=(0.0, 0.0, 0.0, 0.0),
        )
        self._ui_context.end_frame(self._target, encoder)
        encoder.copy_texture_to_buffer(
            self._rgba_buffer,
            0,
            self._rgba_buffer_size,
            self._rgba_row_pitch,
            self._target,
            0,
            0,
            [0, 0, 0],
            [self.width, self.height, 1],
        )
        self._device.submit_command_buffer(encoder.finish())
        self._has_rendered = True
        self._device.sync_to_device(_current_cuda_stream())
        return _rgba8_to_compositing_frame(self._rgba_tensor)

    def reset(self) -> None:
        """Keep the retained SlangPy widget tree for the next generation."""

    def close(self) -> None:
        """Release native UI and GPU resources after pending work completes."""
        if self._device is not None:
            torch.cuda.current_stream().synchronize()
            self._device.wait_for_idle()
        self._ui = None
        self._rgba_tensor = None
        self._rgba_buffer = None
        self._rgba_buffer_size = 0
        self._rgba_row_pitch = 0
        self._target = None
        self._ui_context = None
        self._device = None

    def _ensure_initialized(self) -> None:
        if self._device is not None:
            return
        slangpy = self._slangpy
        if slangpy is None:
            try:
                slangpy = importlib.import_module("slangpy")
            except ImportError as error:
                raise RuntimeError(
                    "ImGui rendering requires the FlashDreams 'local-window' extra."
                ) from error
            self._slangpy = slangpy
        if not torch.cuda.is_available():
            raise RuntimeError("SlangPy ImGui rendering requires CUDA.")
        if not torch.cuda.is_initialized():
            torch.cuda.init()
        torch.cuda.set_device(torch.cuda.current_device())
        torch.cuda.current_stream()
        handles = list(slangpy.get_cuda_current_context_native_handles())
        if not handles:
            raise RuntimeError("Could not obtain the current CUDA context handles.")
        device = slangpy.Device(
            type=slangpy.DeviceType.vulkan,
            enable_debug_layers=False,
            enable_cuda_interop=True,
            enable_cuda_launch_from_gfx=False,
            enable_ray_tracing=False,
            existing_device_handles=handles,
        )
        if not device.supports_cuda_interop:
            raise RuntimeError("The Vulkan device does not support CUDA interop.")

        target = device.create_texture(
            format=slangpy.Format.rgba8_unorm,
            width=self.width,
            height=self.height,
            usage=(
                slangpy.TextureUsage.render_target
                | slangpy.TextureUsage.shader_resource
                | slangpy.TextureUsage.copy_source
            ),
            label="flashdreams_imgui_target",
        )
        layout = target.get_subresource_layout(0)
        size_bytes = int(layout.size_in_bytes)
        row_pitch = int(layout.row_pitch)
        rgba_buffer = device.create_buffer(
            size=size_bytes,
            usage=slangpy.BufferUsage.shared | slangpy.BufferUsage.copy_destination,
            label="flashdreams_imgui_rgba",
        )
        rgba_tensor = cast(
            Tensor,
            rgba_buffer.to_torch(
                type=slangpy.DataType.uint8,
                shape=[self.height, self.width, 4],
                strides=[row_pitch, 4, 1],
            ),
        )

        ui_context = slangpy.ui.Context(device)

        self._device = device
        self._ui_context = ui_context
        self._ui = _SlangPyUI(slangpy.ui, ui_context.screen)
        self._target = target
        self._rgba_buffer = rgba_buffer
        self._rgba_tensor = rgba_tensor
        self._rgba_buffer_size = size_bytes
        self._rgba_row_pitch = row_pitch


class _SlangPyUI:
    """Expose native SlangPy widget types with their root screen."""

    def __init__(self, module: Any, screen: Any) -> None:
        self.screen = screen
        self._module = module

    def __getattr__(self, name: str) -> Any:
        return getattr(self._module, name)


def _route_input_events(
    events: UserInputEvents,
    *,
    ui_context: Any,
    slangpy: Any,
    width: int,
    height: int,
) -> None:
    """Route supported runtime input events into SlangPy's UI context."""
    for event in events.get_events():
        data = event.get_event_data()
        if isinstance(data, KeyboardUserInputEventData):
            pressed = data.state is KeyboardInputState.PRESSED
            key = _resolve_slangpy_key(slangpy, data.key)
            if key is not None:
                key_event = slangpy.KeyboardEvent()
                key_event.type = (
                    slangpy.KeyboardEventType.key_press
                    if pressed
                    else slangpy.KeyboardEventType.key_release
                )
                key_event.key = key
                key_event.mods = slangpy.KeyModifierFlags.none
                ui_context.handle_keyboard_event(key_event)
            if pressed and len(data.key) == 1:
                text_event = slangpy.KeyboardEvent()
                text_event.type = slangpy.KeyboardEventType.input
                text_event.codepoint = ord(data.key)
                text_event.mods = slangpy.KeyModifierFlags.none
                ui_context.handle_keyboard_event(text_event)
        elif isinstance(data, MouseUserInputEventData):
            mouse_event = slangpy.MouseEvent()
            mouse_event.pos = (data.x * width, data.y * height)
            mouse_event.mods = slangpy.KeyModifierFlags.none
            if data.action == "button":
                buttons = (
                    slangpy.MouseButton.left,
                    slangpy.MouseButton.middle,
                    slangpy.MouseButton.right,
                )
                if not 0 <= data.button < len(buttons):
                    continue
                mouse_event.type = (
                    slangpy.MouseEventType.button_down
                    if data.pressed
                    else slangpy.MouseEventType.button_up
                )
                mouse_event.button = buttons[data.button]
            elif data.action == "wheel":
                mouse_event.type = slangpy.MouseEventType.scroll
                mouse_event.scroll = (data.wheel_x, data.wheel_y)
            else:
                mouse_event.type = slangpy.MouseEventType.move
            ui_context.handle_mouse_event(mouse_event)


def _rgba8_to_compositing_frame(frame: Tensor) -> Tensor:
    """Convert shared ``[H, W, 4]`` bytes into normalized ``[4, H, W]``."""
    rgba = frame.permute(2, 0, 1).to(torch.float32)
    color = rgba[:3].mul_(2.0 / 255.0).sub_(1.0)
    alpha = rgba[3:4].mul_(1.0 / 255.0)
    return torch.cat((color, alpha), dim=0)


def _resolve_slangpy_key(slangpy: Any, key: str) -> Any | None:
    normalized = key.strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        " ": "space",
        "alt": "left_alt",
        "arrowdown": "down",
        "arrowleft": "left",
        "arrowright": "right",
        "arrowup": "up",
        "control": "left_control",
        "ctrl": "left_control",
        "esc": "escape",
        "meta": "left_super",
        "return": "enter",
        "shift": "left_shift",
    }
    punctuation = {
        "'": "apostrophe",
        ",": "comma",
        ".": "period",
        "/": "slash",
        ";": "semicolon",
        "=": "equal",
        "[": "left_bracket",
        "\\": "backslash",
        "]": "right_bracket",
        "`": "grave_accent",
    }
    normalized = aliases.get(key.lower(), aliases.get(normalized, normalized))
    normalized = punctuation.get(key, normalized)
    if len(normalized) == 1 and normalized.isdigit():
        normalized = f"key{normalized}"
    return getattr(slangpy.KeyCode, normalized, None)


def _current_cuda_stream() -> int:
    """Return the current PyTorch CUDA stream handle."""
    return int(torch.cuda.current_stream().cuda_stream)


__all__ = ["ImGUIRenderer", "ImGUIThread", "SlangPyImGUIRenderer"]
