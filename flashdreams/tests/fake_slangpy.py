# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""A fake ``slangpy`` standing in for a Vulkan window, device, and swapchain.

Lets :class:`~flashdreams.serving.presentation.LocalWindowPresenter` be driven
on a headless CI host: the resize, swapchain-recovery, and camera-upload logic
is where the real bugs live, and none of it needs a GPU to exercise. Records
the calls a test wants to assert on and can be told to fail a specific
operation so recovery paths are reachable.
"""

from __future__ import annotations

import enum
import types
from dataclasses import dataclass, field
from typing import Any


class Format(enum.Enum):
    rgba8_unorm = "rgba8_unorm"
    bgra8_unorm = "bgra8_unorm"
    bgrx8_unorm = "bgrx8_unorm"
    rgba8_unorm_srgb = "rgba8_unorm_srgb"
    bgra8_unorm_srgb = "bgra8_unorm_srgb"
    bgrx8_unorm_srgb = "bgrx8_unorm_srgb"


class TextureUsage(enum.IntFlag):
    shader_resource = 1
    unordered_access = 2
    copy_destination = 4
    copy_source = 8
    render_target = 16


class DeviceType(enum.Enum):
    vulkan = "vulkan"


class KeyCode(enum.Enum):
    a = "a"
    d = "d"
    r = "r"
    w = "w"
    s = "s"
    key1 = "key1"
    key2 = "key2"
    escape = "escape"
    space = "space"
    left = "left"
    right = "right"
    up = "up"
    down = "down"


class MouseButton(enum.Enum):
    left = "left"
    middle = "middle"
    right = "right"


class MouseEventType(enum.Enum):
    move = "move"
    button_down = "button_down"
    button_up = "button_up"
    scroll = "scroll"


@dataclass(frozen=True)
class Int2:
    x: int
    y: int


@dataclass(frozen=True)
class Uint3:
    x: int
    y: int
    z: int


math = types.SimpleNamespace(uint3=lambda x, y, z: Uint3(x, y, z))


@dataclass
class FakeTexture:
    width: int
    height: int
    format: Any = Format.rgba8_unorm
    label: str | None = None
    uploads: list[Any] = field(default_factory=list)
    """A copy of the pixels of each upload, so a test can tell whether new
    content actually reached the GPU rather than only counting calls."""

    def copy_from_numpy(self, array: Any) -> None:
        self.uploads.append(array.copy())


class FakeCommandEncoder:
    def __init__(self, log: list[str]) -> None:
        self._log = log

    def blit(self, dst: Any, src: Any) -> None:
        del dst, src
        self._log.append("blit")

    def copy_texture(self, *args: Any) -> None:
        del args
        self._log.append("copy_texture")

    def finish(self) -> str:
        return "command_buffer"


@dataclass
class FakeSurfaceInfo:
    preferred_format: Any = Format.rgba8_unorm
    formats: tuple[Any, ...] = (Format.rgba8_unorm, Format.bgra8_unorm)


class FakeSurface:
    """Swapchain stand-in that can refuse to acquire, like a stale swapchain."""

    def __init__(self, info: FakeSurfaceInfo | None = None) -> None:
        self.info = info if info is not None else FakeSurfaceInfo()
        self.config: tuple[int, int] | None = None
        self.configure_calls: list[tuple[int, int]] = []
        self.present_count = 0
        self.acquire_error: Exception | None = None
        self.present_error: Exception | None = None
        self.acquire_returns_none = False

    def configure(self, *, width: int, height: int, format: Any) -> None:
        del format
        self.config = (width, height)
        self.configure_calls.append((width, height))

    def acquire_next_image(self) -> Any:
        if self.acquire_error is not None:
            error, self.acquire_error = self.acquire_error, None
            raise error
        if self.acquire_returns_none:
            return None
        return FakeTexture(width=1, height=1, label="surface")

    def present(self) -> None:
        if self.present_error is not None:
            error, self.present_error = self.present_error, None
            raise error
        self.present_count += 1


@dataclass
class FakeDeviceInfo:
    adapter_name: str = "fake-adapter"


class FakeDevice:
    def __init__(
        self,
        *,
        supports_cuda_interop: bool = False,
        surface_info: FakeSurfaceInfo | None = None,
        **kwargs: Any,
    ) -> None:
        del kwargs
        self.info = FakeDeviceInfo()
        self.supports_cuda_interop = supports_cuda_interop
        self._surface_info = surface_info
        self.textures: list[FakeTexture] = []
        self.encoder_log: list[str] = []
        self.submits = 0
        self.texture_error: Exception | None = None
        """Raised by every ``create_texture`` until the test clears it."""
        self.surface: FakeSurface | None = None

    def create_surface(self, window: Any) -> FakeSurface:
        del window
        self.surface = FakeSurface(self._surface_info)
        return self.surface

    def create_texture(
        self,
        *,
        format: Any,
        width: int,
        height: int,
        usage: Any = None,
        label: str | None = None,
        **kwargs: Any,
    ) -> FakeTexture:
        del usage, kwargs
        if self.texture_error is not None:
            raise self.texture_error
        texture = FakeTexture(width=width, height=height, format=format, label=label)
        self.textures.append(texture)
        return texture

    def create_command_encoder(self) -> FakeCommandEncoder:
        return FakeCommandEncoder(self.encoder_log)

    def submit_command_buffer(self, buffer: Any) -> int:
        del buffer
        self.submits += 1
        return self.submits


class FakeWindow:
    def __init__(
        self,
        *,
        width: int = 1920,
        height: int = 1080,
        title: str = "",
        resizable: bool = True,
    ) -> None:
        self._size = Int2(width, height)
        self.title = title
        self.resizable = resizable
        self.closed = False
        self.process_event_calls = 0
        self.resize_requests: list[tuple[int, int]] = []
        self.resize_error: Exception | None = None
        self._should_close = False
        self.on_resize: Any = None
        self.on_keyboard_event: Any = None
        self.on_mouse_event: Any = None

    @property
    def size(self) -> Int2:
        return self._size

    def should_close(self) -> bool:
        return self._should_close

    def process_events(self) -> None:
        self.process_event_calls += 1

    def resize(self, width: int, height: int) -> None:
        if self.resize_error is not None:
            error, self.resize_error = self.resize_error, None
            raise error
        self.resize_requests.append((width, height))
        self._size = Int2(width, height)

    def close(self) -> None:
        self.closed = True

    ## Test-side driving

    def set_size(self, width: int, height: int) -> None:
        """Resize without notifying, as a compositor-side resize would."""
        self._size = Int2(width, height)

    def notify_resize(self, width: int, height: int) -> None:
        """Resize and fire the callback, in the order a real window does.

        The size is live before the callback arrives, which matters: the
        presenter re-reads the window size while presenting and would otherwise
        treat the new configuration as stale and revert it.
        """
        self.set_size(width, height)
        if self.on_resize is not None:
            self.on_resize(width, height)

    def request_close(self) -> None:
        self._should_close = True


@dataclass
class FakeKeyboardEvent:
    key: Any
    action: str = "press"

    def is_key_press(self) -> bool:
        return self.action == "press"

    def is_key_release(self) -> bool:
        return self.action == "release"

    def is_key_repeat(self) -> bool:
        return self.action == "repeat"


@dataclass
class FakeMouseEvent:
    type: Any
    pos: Any
    button: Any = None


@dataclass
class FakeSlangpy(types.SimpleNamespace):
    """The module object a presenter sees in place of real ``slangpy``.

    ``Window`` and ``Device`` are capitalised to mirror the real module's
    constructors, which is what the presenter calls.
    """

    windows: list[FakeWindow] = field(default_factory=list)
    devices: list[FakeDevice] = field(default_factory=list)
    supports_cuda_interop: bool = False

    clamp_window_to: tuple[int, int] | None = None
    """Size the window manager forces, ignoring what was requested. Mirrors
    SDL3 clamping to the display or scaling for HiDPI."""

    surface_info: FakeSurfaceInfo | None = None
    """Formats the swapchain advertises; defaults to offering linear ones."""

    def __post_init__(self) -> None:
        self.Format = Format
        self.TextureUsage = TextureUsage
        self.DeviceType = DeviceType
        self.KeyCode = KeyCode
        self.MouseButton = MouseButton
        self.MouseEventType = MouseEventType
        self.math = math

    def Window(self, **kwargs: Any) -> FakeWindow:
        window = FakeWindow(**kwargs)
        if self.clamp_window_to is not None:
            window.set_size(*self.clamp_window_to)
        self.windows.append(window)
        return window

    def Device(self, **kwargs: Any) -> FakeDevice:
        device = FakeDevice(
            supports_cuda_interop=self.supports_cuda_interop,
            surface_info=self.surface_info,
            **kwargs,
        )
        self.devices.append(device)
        return device

    @property
    def window(self) -> FakeWindow:
        return self.windows[-1]

    @property
    def device(self) -> FakeDevice:
        return self.devices[-1]

    @property
    def surface(self) -> FakeSurface:
        surface = self.device.surface
        assert surface is not None, "no surface created yet"
        return surface


__all__ = [
    "FakeDevice",
    "FakeKeyboardEvent",
    "FakeMouseEvent",
    "FakeSlangpy",
    "FakeSurface",
    "FakeSurfaceInfo",
    "FakeTexture",
    "FakeWindow",
    "Format",
    "KeyCode",
    "MouseButton",
    "MouseEventType",
]
