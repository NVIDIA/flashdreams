# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Presenter wrapper: unsharp filter + coin sprites on the model frame.

Decorator-style presenter in the mold of ``CausalFrameAlignmentPresenter``
and ``AlignmentDiagnosticPresenter`` (composition + ``__getattr__``
passthrough), inserted *inside* the alignment wrapper at the composition
root so it sees frame-synchronized ``rig_to_world``:

    self._presenter = CausalFrameAlignmentPresenter(
        LiveEditPresenter(presenter, ...)
    )

It rewrites ``PresentedFrame.model_rgb_host_uint8`` with a host-composited
copy wrapped in a lazy-source shim, so both downstream branches (the
SlangPy CUDA fast path via ``to_cuda_tensor`` and the PIL/MJPEG path via
``to_numpy``) keep working. CPU cost is milliseconds per frame; a
GPU-native torch compositing path is a follow-up (TODO below).
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
from omnidreams_game_engine.camera import FThetaCameraModel
from omnidreams_game_engine.types import CameraCalibration, PresentedFrame
from PIL import Image, ImageDraw, ImageFilter

from crazy_robotaxi.live_edit.coin_ability import CoinAbility, coin_squash
from crazy_robotaxi.live_edit.config import LiveEditConfig
from crazy_robotaxi.live_edit.style_ability import StyleAbility

_COUNTER_MARGIN_PX = 12
_COUNTER_TEXT_RGBA = (255, 255, 255, 255)
_COUNTER_CHIP_RGBA = (30, 30, 30, 180)
_COIN_GOLD = (250, 200, 40, 255)
_COIN_RIM = (170, 120, 10, 255)


def unsharp_rgb(frame: np.ndarray, *, sigma: float, amount: float) -> np.ndarray:
    """Sharpen an HWC uint8 RGB frame with the validated cas-style mask.

    Matches ``composite_track_items_residential.cas_sharpen``
    (``addWeighted 1+amount / -amount`` around a Gaussian blur) without the
    cv2 dependency.
    """
    if amount <= 0.0:
        return frame
    image = Image.fromarray(frame, mode="RGB")
    blurred = np.asarray(
        image.filter(ImageFilter.GaussianBlur(radius=sigma)), dtype=np.float32
    )
    sharpened = (1.0 + amount) * frame.astype(np.float32) - amount * blurred
    return np.clip(sharpened, 0.0, 255.0).astype(np.uint8)


def procedural_coin_sprite(size_px: int = 96) -> Image.Image:
    """Render a simple golden coin RGBA sprite (placeholder asset)."""
    sprite = Image.new("RGBA", (size_px, size_px), (0, 0, 0, 0))
    draw = ImageDraw.Draw(sprite)
    margin = size_px // 12
    draw.ellipse(
        [margin, margin, size_px - margin, size_px - margin],
        fill=_COIN_GOLD,
        outline=_COIN_RIM,
        width=max(2, size_px // 16),
    )
    inner = size_px // 4
    draw.ellipse(
        [inner, inner, size_px - inner, size_px - inner],
        outline=_COIN_RIM,
        width=max(1, size_px // 24),
    )
    return sprite


class _HostRGBFrame:
    """Composited host frame exposing the LazyRGBFrame duck-type."""

    def __init__(self, rgb_host_uint8: np.ndarray) -> None:
        self._rgb = np.ascontiguousarray(rgb_host_uint8)

    @property
    def shape(self) -> tuple[int, ...]:
        return self._rgb.shape

    def prefetch_to_numpy(self) -> None:
        return None

    def to_numpy(self) -> np.ndarray:
        return self._rgb

    def __array__(self, dtype: Any = None) -> np.ndarray:
        return self._rgb if dtype is None else self._rgb.astype(dtype)

    def to_cuda_event(self) -> None:
        return None

    def to_cuda_tensor(self) -> Any:
        # TODO(gpu): host round-trip; a torch-native compositing path
        # should keep the frame on-device instead of re-uploading here.
        import torch

        return torch.from_numpy(self._rgb).cuda()


class LiveEditPresenter:
    """Composite live-edit pixels into frames before HUD drawing/encoding."""

    def __init__(
        self,
        inner: Any,
        config: LiveEditConfig,
        *,
        coin_ability: CoinAbility | None = None,
        style_ability: StyleAbility | None = None,
    ) -> None:
        self._inner = inner
        self._config = config
        self._coin_ability = coin_ability
        self._style_ability = style_ability
        self._camera_model: FThetaCameraModel | None = None
        self._camera_calibration: CameraCalibration | None = None
        self._frame_index = 0
        self._last_timestamp_us: int | None = None
        self._last_processed: PresentedFrame | None = None
        self._coin_sprite = self._load_sprite(config.coins.sprite_path)

    def set_coin_ability(self, coin_ability: CoinAbility | None) -> None:
        """Bind the per-rollout coin ability (rebuilt on scene load / reset)."""
        self._coin_ability = coin_ability

    def configure_taxi_camera(self, calibration: CameraCalibration) -> None:
        """Intercept the scene camera and forward it down the chain."""
        self._camera_calibration = calibration
        self._camera_model = None
        forward = getattr(self._inner, "configure_taxi_camera", None)
        if callable(forward):
            forward(calibration)

    def prepare_frame(self, frame: PresentedFrame, view_mode: str) -> None:
        prepare = getattr(self._inner, "prepare_frame", None)
        if callable(prepare):
            prepare(self._process(frame, view_mode), view_mode)

    def present_frame(self, frame: PresentedFrame, view_mode: str) -> None:
        self._inner.present_frame(self._process(frame, view_mode), view_mode)

    def close(self) -> None:
        self._inner.close()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def _process(self, frame: PresentedFrame, view_mode: str) -> PresentedFrame:
        if view_mode != "model_rgb" or frame.model_rgb_host_uint8 is None:
            return frame
        if not self._anything_active():
            return frame
        if (
            self._last_timestamp_us == frame.timestamp_us
            and self._last_processed is not None
        ):
            # prepare_frame runs speculatively ahead of present_frame; the
            # lazy source can only be materialized once, so reuse the result.
            return replace(
                frame,
                model_rgb_host_uint8=self._last_processed.model_rgb_host_uint8,
            )
        rgb = np.ascontiguousarray(
            np.asarray(frame.model_rgb_host_uint8, dtype=np.uint8)[..., :3]
        )
        rgb = self._apply_style_filter(rgb)
        rgb = self._composite_coins(rgb, frame)
        rgb = self._draw_hud_chips(rgb)
        self._frame_index += 1
        processed = replace(frame, model_rgb_host_uint8=_HostRGBFrame(rgb))
        self._last_timestamp_us = frame.timestamp_us
        self._last_processed = processed
        return processed

    def _anything_active(self) -> bool:
        coins_active = self._coin_ability is not None and self._coin_ability.enabled
        style_active = (
            self._style_ability is not None
            and self._style_ability.active_skin_name != "base"
        )
        return coins_active or style_active

    def _apply_style_filter(self, rgb: np.ndarray) -> np.ndarray:
        style = self._style_ability
        if style is None or style.active_skin_name == "base":
            return rgb
        return unsharp_rgb(
            rgb,
            sigma=self._config.sharpen_sigma,
            amount=self._config.sharpen_amount,
        )

    def _composite_coins(self, rgb: np.ndarray, frame: PresentedFrame) -> np.ndarray:
        coins = self._coin_ability
        if coins is None or not coins.enabled or frame.rig_to_world is None:
            return rgb
        height, width = rgb.shape[:2]
        camera_model = self._require_camera_model(width, height)
        if camera_model is None:
            return rgb
        sprites = coins.visible_sprites(
            np.asarray(frame.rig_to_world, dtype=np.float32),
            camera_model,
            image_width=width,
            image_height=height,
        )
        canvas = Image.fromarray(rgb, mode="RGB").convert("RGBA")
        for sprite in sprites:
            squash = coin_squash(sprite.spin_phase, self._frame_index)
            sprite_h = max(3, round(sprite.height_px))
            sprite_w = max(2, round(sprite.height_px * squash))
            resized = self._coin_sprite.resize(
                (sprite_w, sprite_h), Image.Resampling.LANCZOS
            )
            if sprite.alpha < 1.0:
                faded = np.asarray(resized).copy()
                faded[..., 3] = (
                    faded[..., 3].astype(np.float32) * sprite.alpha
                ).astype(np.uint8)
                resized = Image.fromarray(faded)
            # TODO: port luminance harmonization + contact shadows from
            # composite_track_items.py once the look is reviewed on GPU runs.
            canvas.alpha_composite(
                resized,
                (
                    round(sprite.center_uv[0] - sprite_w / 2.0),
                    round(sprite.center_uv[1] - sprite_h / 2.0),
                ),
            )
        return np.asarray(canvas.convert("RGB"))

    def _draw_hud_chips(self, rgb: np.ndarray) -> np.ndarray:
        """Draw the skin-name and coin-counter chips into the frame.

        Drawn into pixels so both the native window and the MJPEG stream
        show them without touching ``_draw_taxi_hud``; the polished version
        belongs in the taxi HUD panel (needs an upstream hook or edit).
        """
        labels: list[str] = []
        style = self._style_ability
        if style is not None:
            labels.append(f"SKIN {style.active_skin_name.upper()}")
        coins = self._coin_ability
        if coins is not None and coins.enabled:
            labels.append(f"COINS {coins.collected_count}")
        if not labels:
            return rgb
        canvas = Image.fromarray(rgb, mode="RGB").convert("RGBA")
        draw = ImageDraw.Draw(canvas)
        y0 = _COUNTER_MARGIN_PX
        for label in labels:
            x0 = _COUNTER_MARGIN_PX
            text_box = draw.textbbox((x0 + 10, y0 + 6), label)
            draw.rounded_rectangle(
                [x0, y0, text_box[2] + 10, text_box[3] + 6],
                radius=6,
                fill=_COUNTER_CHIP_RGBA,
            )
            draw.text((x0 + 10, y0 + 6), label, fill=_COUNTER_TEXT_RGBA)
            y0 = text_box[3] + 6 + 8
        return np.asarray(canvas.convert("RGB"))

    def _require_camera_model(
        self, width: int, height: int
    ) -> FThetaCameraModel | None:
        """Return the FTheta model scaled to the presented frame size.

        The model frame usually differs from the calibration resolution, so
        the projection is scaled per source size exactly as the taxi marker
        path does (``hud_presenter._draw_taxi_world_marker``).
        """
        if self._camera_calibration is None:
            return None
        model = self._camera_model
        if (
            model is None
            or model.output_width != width
            or model.output_height != height
        ):
            self._camera_model = FThetaCameraModel(
                self._camera_calibration,
                output_width=width,
                output_height=height,
            )
        return self._camera_model

    @staticmethod
    def _load_sprite(sprite_path: Path | None) -> Image.Image:
        if sprite_path is None:
            return procedural_coin_sprite()
        with Image.open(sprite_path) as image:
            return image.convert("RGBA")
