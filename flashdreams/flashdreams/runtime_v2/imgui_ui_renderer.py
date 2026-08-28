# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dear ImGui rendering through SlangPy's Vulkan/CUDA bridge."""

from __future__ import annotations

import importlib
from collections.abc import Callable
from typing import Any, cast

import numpy as np
import torch
from torch import Tensor

from flashdreams.runtime_v2.slangpy_ui_renderer import (
    _current_cuda_stream,
    _resolve_slangpy_key,
    _rgba8_to_compositing_frame,
)
from flashdreams.runtime_v2.user_input_event import (
    KeyboardInputState,
    KeyboardUserInputEvent,
    MouseUserInputEvent,
)
from flashdreams.runtime_v2.user_input_events import UserInputEvents


class _ImGui:
    """Proxy Dear ImGui and add image-like pixel uploads backed by SlangPy."""

    def __init__(self, device: Any, slangpy: Any, imgui: Any, bridge: Any) -> None:
        self._device = device
        self._slangpy = slangpy
        self._imgui = imgui
        self._bridge = bridge
        self._textures: dict[str, tuple[Any, tuple[int, ...]]] = {}

    def __getattr__(self, name: str) -> Any:
        """Expose the complete ``imgui_bundle.imgui`` API."""
        return getattr(self._imgui, name)

    def image(
        self,
        texture_or_key: Any,
        pixels_or_size: Any,
        *args: Any,
        size: tuple[float, float] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Draw an ImGui texture or upload and draw image-like pixels.

        The regular ``imgui.image(texture_ref, ImVec2(...))`` form is passed
        through unchanged. ``imgui.image(key, pixels, size=(w, h))`` uploads
        RGB/RGBA pixels into a cached SlangPy texture before drawing it.
        """
        if not isinstance(texture_or_key, str) or size is None:
            return self._imgui.image(
                texture_or_key,
                pixels_or_size,
                *args,
                **kwargs,
            )
        rgba = _rgba_pixels(pixels_or_size)
        cached = self._textures.get(texture_or_key)
        if cached is None or cached[1] != rgba.shape:
            texture = self._device.create_texture(
                format=self._slangpy.Format.rgba8_unorm_srgb,
                width=int(rgba.shape[1]),
                height=int(rgba.shape[0]),
                usage=self._slangpy.TextureUsage.shader_resource,
                label=f"flashdreams_imgui_{texture_or_key}",
            )
            self._textures[texture_or_key] = (texture, rgba.shape)
        else:
            texture = cached[0]
        texture.copy_from_numpy(rgba)
        return self._imgui.image(
            self._bridge.texture_ref(texture),
            self._imgui.ImVec2(*size),
        )

    def clear_textures(self) -> None:
        """Release application-owned SlangPy texture references."""
        self._textures.clear()


class _ImGuiUIRenderer:
    """Render immediate Dear ImGui draw data through SlangPy CUDA interop."""

    def __init__(
        self,
        *,
        width: int,
        height: int,
        slangpy_module: Any | None = None,
        imgui_module: Any | None = None,
        bridge_module: Any | None = None,
    ) -> None:
        if width <= 0 or height <= 0:
            raise ValueError("ImGui UI render dimensions must be > 0.")
        self.width = int(width)
        self.height = int(height)
        self._slangpy = slangpy_module
        self._imgui = imgui_module
        self._bridge = bridge_module
        self._device: Any | None = None
        self._ui_context: Any | None = None
        self._imgui_context: Any | None = None
        self._ui: _ImGui | None = None
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
        step_ui: Callable[[Any, int, UserInputEvents], None],
    ) -> Tensor:
        """Render one transparent immediate-mode UI frame."""
        self._ensure_initialized()
        assert self._device is not None
        assert self._slangpy is not None
        assert self._imgui is not None
        assert self._bridge is not None
        assert self._ui_context is not None
        assert self._imgui_context is not None
        assert self._ui is not None
        assert self._target is not None
        assert self._rgba_buffer is not None
        assert self._rgba_tensor is not None

        if self._has_rendered:
            self._device.sync_to_cuda(_current_cuda_stream())
        self._imgui.set_current_context(self._imgui_context)
        _route_imgui_input_events(
            events,
            slangpy=self._slangpy,
            bridge=self._bridge,
            width=self.width,
            height=self.height,
        )
        self._bridge.begin_frame(self.width, self.height)
        step_ui(self._ui, step_index, events)
        self._imgui.render()
        draw_data = self._imgui.get_draw_data()
        self._bridge.sync_draw_data_textures(
            self._device,
            self._ui_context,
            draw_data,
        )

        encoder = self._device.create_command_encoder()
        encoder.clear_texture_float(
            self._target,
            clear_value=(0.0, 0.0, 0.0, 0.0),
        )
        self._bridge.render_imgui_draw_data(
            self._ui_context,
            draw_data,
            self._target,
            encoder,
        )
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
        """Preserve cached textures across model-session resets."""

    def close(self) -> None:
        """Release ImGui, UI, and GPU resources after pending work completes."""
        if self._device is not None:
            torch.cuda.current_stream().synchronize()
            self._device.wait_for_idle()
        if self._ui is not None:
            self._ui.clear_textures()
        if self._imgui is not None and self._imgui_context is not None:
            self._imgui.destroy_context(self._imgui_context)
        self._ui = None
        self._rgba_tensor = None
        self._rgba_buffer = None
        self._rgba_buffer_size = 0
        self._rgba_row_pitch = 0
        self._target = None
        self._imgui_context = None
        self._ui_context = None
        self._device = None

    def _ensure_initialized(self) -> None:
        if self._device is not None:
            return
        try:
            slangpy = self._slangpy or importlib.import_module("slangpy")
            imgui = self._imgui or importlib.import_module("imgui_bundle.imgui")
            bridge = self._bridge or importlib.import_module("slangpy.ui.imgui_bundle")
        except ImportError as error:
            raise RuntimeError(
                "ImGui UI rendering requires the FlashDreams 'local-window' extra."
            ) from error
        if not torch.cuda.is_available():
            raise RuntimeError("ImGui UI rendering requires CUDA.")
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
            label="flashdreams_imgui_ui_target",
        )
        layout = target.get_subresource_layout(0)
        size_bytes = int(layout.size_in_bytes)
        row_pitch = int(layout.row_pitch)
        rgba_buffer = device.create_buffer(
            size=size_bytes,
            usage=slangpy.BufferUsage.shared | slangpy.BufferUsage.copy_destination,
            label="flashdreams_imgui_ui_rgba",
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
        imgui_context = bridge.create_imgui_context(self.width, self.height)

        self._slangpy = slangpy
        self._imgui = imgui
        self._bridge = bridge
        self._device = device
        self._ui_context = ui_context
        self._imgui_context = imgui_context
        self._ui = _ImGui(device, slangpy, imgui, bridge)
        self._target = target
        self._rgba_buffer = rgba_buffer
        self._rgba_tensor = rgba_tensor
        self._rgba_buffer_size = size_bytes
        self._rgba_row_pitch = row_pitch


def _route_imgui_input_events(
    events: UserInputEvents,
    *,
    slangpy: Any,
    bridge: Any,
    width: int,
    height: int,
) -> None:
    """Translate runtime keyboard and mouse events into Dear ImGui events."""
    for event in events.get_events():
        if isinstance(event, KeyboardUserInputEvent):
            pressed = event.state is KeyboardInputState.PRESSED
            key = _resolve_slangpy_key(slangpy, event.key)
            if key is not None:
                key_event = slangpy.KeyboardEvent()
                key_event.type = (
                    slangpy.KeyboardEventType.key_press
                    if pressed
                    else slangpy.KeyboardEventType.key_release
                )
                key_event.key = key
                key_event.mods = slangpy.KeyModifierFlags.none
                bridge.handle_keyboard_event(key_event)
            if pressed and len(event.key) == 1:
                text_event = slangpy.KeyboardEvent()
                text_event.type = slangpy.KeyboardEventType.input
                text_event.codepoint = ord(event.key)
                text_event.mods = slangpy.KeyModifierFlags.none
                bridge.handle_keyboard_event(text_event)
        elif isinstance(event, MouseUserInputEvent):
            mouse_event = slangpy.MouseEvent()
            mouse_event.pos = (event.x * width, event.y * height)
            mouse_event.mods = slangpy.KeyModifierFlags.none
            if event.action == "button":
                buttons = (
                    slangpy.MouseButton.left,
                    slangpy.MouseButton.middle,
                    slangpy.MouseButton.right,
                )
                if not 0 <= event.button < len(buttons):
                    continue
                mouse_event.type = (
                    slangpy.MouseEventType.button_down
                    if event.pressed
                    else slangpy.MouseEventType.button_up
                )
                mouse_event.button = buttons[event.button]
            elif event.action == "wheel":
                mouse_event.type = slangpy.MouseEventType.scroll
                mouse_event.scroll = (event.wheel_x, event.wheel_y)
            else:
                mouse_event.type = slangpy.MouseEventType.move
            bridge.handle_mouse_event(mouse_event)


def _rgba_pixels(value: Any) -> np.ndarray[Any, np.dtype[np.uint8]]:
    """Materialize an image-like value as contiguous RGBA bytes."""
    to_numpy = getattr(value, "to_numpy", None)
    if callable(to_numpy):
        value = to_numpy()
    if isinstance(value, Tensor):
        value = value.detach().cpu().numpy()
    array = np.asarray(value, dtype=np.uint8)
    if array.ndim != 3 or array.shape[2] not in (3, 4):
        raise ValueError(f"Expected HWC RGB/RGBA pixels, received {array.shape}.")
    if array.shape[2] == 3:
        alpha = np.full((*array.shape[:2], 1), 255, dtype=np.uint8)
        array = np.concatenate((array, alpha), axis=2)
    return np.array(array, dtype=np.uint8, copy=True, order="C")


__all__: list[str] = []
