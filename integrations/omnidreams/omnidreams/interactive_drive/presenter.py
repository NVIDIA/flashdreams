# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import time

import numpy as np
from omnidreams.interactive_drive.config import RasterConfig
from omnidreams.interactive_drive.input.keyboard import KeyboardState
from omnidreams.interactive_drive.loading_overlay import render_loading_overlay
from omnidreams.interactive_drive.types import PresentedFrame


class SlangPyPresenter:
    def __init__(self, raster: RasterConfig, keyboard: KeyboardState) -> None:
        try:
            import slangpy as spy
        except ImportError as exc:
            raise RuntimeError(
                "SlangPy is required for the presenter. Install with"
                " `uv sync --package flashdreams-omnidreams`."
            ) from exc

        self._spy = spy
        self._raster = raster
        self._keyboard = keyboard
        self._window = spy.Window(
            width=raster.width,
            height=raster.height,
            title="interactive_drive",
            resizable=False,
        )
        self._device = spy.Device(
            type=spy.DeviceType.vulkan,
            enable_debug_layers=False,
            # Workaround: avoid cuDNN MHA crash on NVIDIA Blackwell + R595.
            enable_cuda_launch_from_gfx=False,
            enable_ray_tracing=False,
        )
        print(f"[presenter] device={self._device.info.adapter_name}", flush=True)
        self._surface = self._device.create_surface(self._window)
        self._surface_format = self._choose_surface_format()
        self._display_format = spy.Format.rgba8_unorm
        print(
            f"[presenter] surface preferred={self._surface.info.preferred_format} chosen={self._surface_format} display={self._display_format}",
            flush=True,
        )
        self._surface.configure(
            width=raster.width, height=raster.height, format=self._surface_format
        )
        self._display_texture = self._device.create_texture(
            format=self._display_format,
            width=raster.width,
            height=raster.height,
            usage=spy.TextureUsage.shader_resource | spy.TextureUsage.unordered_access,
            label="display_texture",
        )
        self._key_codes = self._build_key_codes()
        self._window.on_keyboard_event = self._on_keyboard_event

    @property
    def should_close(self) -> bool:
        return self._window.should_close()

    def close(self) -> None:
        self._window.close()

    def process_events(self) -> None:
        self._window.process_events()

    def present_frame(self, frame: PresentedFrame, view_mode: str) -> None:
        if view_mode == "model_rgb" and frame.model_rgb_host_uint8 is not None:
            self._present_array(
                _with_status_overlay(frame.model_rgb_host_uint8, frame.status_message)
            )
            return
        self._present_array(
            _with_status_overlay(frame.rgb_host_uint8, frame.status_message)
        )

    def _present_array(self, rgb_host_uint8: np.ndarray) -> None:
        if not self._surface.config:
            return
        surface_texture = self._surface.acquire_next_image()
        if not surface_texture:
            time.sleep(0.001)
            return

        upload = self._pack_surface_pixels(rgb_host_uint8)
        self._display_texture.copy_from_numpy(upload)

        command_encoder = self._device.create_command_encoder()
        command_encoder.blit(surface_texture, self._display_texture)
        self._device.submit_command_buffer(command_encoder.finish())
        del surface_texture
        self._surface.present()

    def _choose_surface_format(self):
        linear_pairs = {
            self._spy.Format.rgba8_unorm_srgb: self._spy.Format.rgba8_unorm,
            self._spy.Format.bgra8_unorm_srgb: self._spy.Format.bgra8_unorm,
            self._spy.Format.bgrx8_unorm_srgb: self._spy.Format.bgrx8_unorm,
        }
        preferred = self._surface.info.preferred_format
        supported = list(self._surface.info.formats)

        for candidate in (
            self._spy.Format.rgba8_unorm,
            self._spy.Format.bgra8_unorm,
            self._spy.Format.bgrx8_unorm,
        ):
            if candidate in supported:
                return candidate

        preferred_linear = linear_pairs.get(preferred, preferred)
        if preferred_linear in supported:
            return preferred_linear

        raise RuntimeError(
            f"Presenter requires a linear swapchain, but the surface only supports: {supported}"
        )

    def _pack_surface_pixels(self, rgb_host_uint8: np.ndarray) -> np.ndarray:
        upload = np.zeros((self._raster.height, self._raster.width, 4), dtype=np.uint8)
        upload[..., :3] = rgb_host_uint8
        upload[..., 3] = 255
        return upload

    def _on_keyboard_event(self, event) -> None:
        is_press = event.is_key_press() if hasattr(event, "is_key_press") else False
        is_release = (
            event.is_key_release() if hasattr(event, "is_key_release") else False
        )
        if not (is_press or is_release):
            return

        if self._matches_key(event.key, "escape") and is_press:
            self.close()
            return

        key_map = {
            self._key_codes["w"]: "w",
            self._key_codes["a"]: "a",
            self._key_codes["s"]: "s",
            self._key_codes["d"]: "d",
            self._key_codes["up"]: "up",
            self._key_codes["left"]: "left",
            self._key_codes["down"]: "down",
            self._key_codes["right"]: "right",
        }
        key_map = {
            key_code: name for key_code, name in key_map.items() if key_code is not None
        }
        if event.key in key_map:
            self._keyboard.set_key(key_map[event.key], is_press)
            return

        # Two view modes: ``1`` = world-model RGB (the generated drive view,
        # which is the point of the demo), ``2`` = HDMap with traffic (the
        # rasterizer's conditioning input). No depth mode.
        if is_press and self._matches_key(event.key, "key1"):
            self._keyboard.set_view_mode("model_rgb")
        elif is_press and self._matches_key(event.key, "key2"):
            self._keyboard.set_view_mode("rgb")
        elif is_press and self._matches_key(event.key, "r"):
            # ``r`` restarts the rollout: the simulation loop bumps its
            # trajectory back to the scene's start pose and begins a new
            # world-model session (new KV cache, new first chunk).
            self._keyboard.request_reset()

    def _build_key_codes(self) -> dict[str, object | None]:
        return {
            "escape": self._lookup_key_code("escape"),
            "w": self._lookup_key_code("w"),
            "a": self._lookup_key_code("a"),
            "s": self._lookup_key_code("s"),
            "d": self._lookup_key_code("d"),
            "r": self._lookup_key_code("r"),
            "up": self._lookup_key_code("up", "arrow_up"),
            "left": self._lookup_key_code("left", "arrow_left"),
            "down": self._lookup_key_code("down", "arrow_down"),
            "right": self._lookup_key_code("right", "arrow_right"),
            "key1": self._lookup_key_code("key1", "digit1", "num_1"),
            "key2": self._lookup_key_code("key2", "digit2", "num_2"),
        }

    def _lookup_key_code(self, *names: str) -> object | None:
        for name in names:
            value = getattr(self._spy.KeyCode, name, None)
            if value is not None:
                return value
        return None

    def _matches_key(self, event_key: object, name: str) -> bool:
        key_code = self._key_codes.get(name)
        return key_code is not None and event_key == key_code


def _with_status_overlay(rgb_host_uint8: np.ndarray, message: str | None) -> np.ndarray:
    if message is None:
        return rgb_host_uint8
    return render_loading_overlay(rgb_host_uint8, message=message)
