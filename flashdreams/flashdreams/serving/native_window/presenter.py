# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SlangPy presenter for shared local-window demos."""

from __future__ import annotations

import importlib
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

DEFAULT_NATIVE_KEY_BINDINGS: Mapping[str, Sequence[str]] = {
    "w": ("w", "up", "arrow_up"),
    "a": ("a", "left", "arrow_left"),
    "s": ("s", "down", "arrow_down"),
    "d": ("d", "right", "arrow_right"),
    "q": ("q",),
    "e": ("e",),
    "i": ("i",),
    "j": ("j",),
    "k": ("k",),
    "l": ("l",),
    "r": ("r",),
    "g": ("g",),
    "b": ("b",),
    "space": ("space",),
    "shift": ("left_shift", "right_shift", "shift"),
    "control": ("left_control", "right_control", "control", "ctrl"),
}


class SlangPyNativePresenter:
    """Present RGB frames and forward configured keyboard controls."""

    def __init__(
        self,
        *,
        width: int,
        height: int,
        title: str,
        on_key: Callable[[str, str], None],
        key_bindings: Mapping[str, Sequence[str]] | None = None,
    ) -> None:
        try:
            spy = importlib.import_module("slangpy")
        except ImportError as exc:
            raise RuntimeError(
                "Local-window output requires the 'native-window' extra: "
                "pip install 'flashdreams[native-window]'."
            ) from exc
        self._width = width
        self._height = height
        self._on_key_callback = on_key
        self._closed = False
        self._window = spy.Window(
            width=width,
            height=height,
            title=title,
            resizable=False,
        )
        self._device = spy.Device(
            type=spy.DeviceType.vulkan,
            enable_debug_layers=False,
            enable_cuda_launch_from_gfx=False,
            enable_ray_tracing=False,
        )
        self._surface = self._device.create_surface(self._window)
        self._surface.configure(
            width=width,
            height=height,
            format=_surface_format(spy, self._surface),
        )
        self._texture = self._device.create_texture(
            format=spy.Format.rgba8_unorm,
            width=width,
            height=height,
            usage=(
                spy.TextureUsage.shader_resource
                | spy.TextureUsage.unordered_access
                | spy.TextureUsage.copy_destination
            ),
        )
        self._rgba = np.empty((height, width, 4), dtype=np.uint8)
        self._keys = _build_key_map(
            spy,
            key_bindings or DEFAULT_NATIVE_KEY_BINDINGS,
        )
        self._escape = getattr(spy.KeyCode, "escape", None)
        self._window.on_keyboard_event = self._on_keyboard

    @property
    def should_close(self) -> bool:
        return self._closed or self._window.should_close()

    def process_events(self) -> None:
        self._window.process_events()

    def present_frame(self, frame: object) -> None:
        target = self._surface.acquire_next_image()
        if not target:
            time.sleep(0.001)
            return
        rgb = _as_rgb(frame)
        if rgb.shape != (self._height, self._width, 3):
            raise ValueError(
                f"Native frame shape {rgb.shape} does not match "
                f"{(self._height, self._width, 3)}."
            )
        self._rgba[..., :3], self._rgba[..., 3] = rgb, 255
        self._texture.copy_from_numpy(self._rgba)
        encoder = self._device.create_command_encoder()
        encoder.blit(target, self._texture)
        self._device.submit_command_buffer(encoder.finish())
        self._surface.present()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._window.close()

    def _on_keyboard(self, event: Any) -> None:
        pressed = event.is_key_press()
        released = event.is_key_release()
        if pressed and event.key == self._escape:
            self.close()
            return
        key = self._keys.get(event.key)
        if key is not None and (pressed or released):
            self._on_key_callback("keydown" if pressed else "keyup", key)


def _surface_format(spy: Any, surface: Any) -> Any:
    supported = surface.info.formats
    for name in ("rgba8_unorm", "bgra8_unorm", "bgrx8_unorm"):
        value = getattr(spy.Format, name, None)
        if value is not None and value in supported:
            return value
    raise RuntimeError("Local-window output requires a linear surface format.")


def _build_key_map(
    spy: Any,
    bindings: Mapping[str, Sequence[str]],
) -> dict[object, str]:
    keys: dict[object, str] = {}
    for action, names in bindings.items():
        for name in names:
            value = getattr(spy.KeyCode, name, None)
            if value is not None:
                keys[value] = action
    return keys


def _as_rgb(frame: object) -> np.ndarray:
    to_numpy = getattr(frame, "to_numpy", None)
    if callable(to_numpy):
        frame = to_numpy()
    return np.ascontiguousarray(np.asarray(frame, dtype=np.uint8)[..., :3])


__all__ = ["DEFAULT_NATIVE_KEY_BINDINGS", "SlangPyNativePresenter"]
