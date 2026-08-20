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

import math
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
from omnidreams_game_engine.camera import FThetaCameraModel
from omnidreams_game_engine.types import CameraCalibration, PresentedFrame
from PIL import Image, ImageDraw, ImageFilter

from crazy_robotaxi.live_edit.coin_ability import CoinAbility, coin_squash
from crazy_robotaxi.live_edit.config import LiveEditConfig
from crazy_robotaxi.live_edit.obstacle_ability import ObstacleAbility
from crazy_robotaxi.live_edit.style_ability import StyleAbility

_COUNTER_MARGIN_PX = 12
_COUNTER_TEXT_RGBA = (255, 255, 255, 255)
_COUNTER_CHIP_RGBA = (30, 30, 30, 180)
_COIN_GOLD = (250, 200, 40, 255)
_COIN_RIM = (170, 120, 10, 255)

# Soft contact-shadow tuning: kept light so the gold coin pops.
_SHADOW_MAX_ALPHA = 60
_SHADOW_WIDTH_FRACTION = 0.9
_SHADOW_HEIGHT_FRACTION = 0.22
_SHADOW_DROP_FRACTION = 0.62


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
    """Render a simple golden coin RGBA sprite (the default coin; a custom
    sprite can be supplied via ``--live-edit-coin-sprite``)."""
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
        self._obstacle_ability: ObstacleAbility | None = None
        self._camera_model: FThetaCameraModel | None = None
        self._camera_calibration: CameraCalibration | None = None
        self._frame_index = 0
        self._last_timestamp_us: int | None = None
        self._last_processed: PresentedFrame | None = None
        self._coin_sprite = self._load_sprite(config.coins.sprite_path)

    def set_coin_ability(self, coin_ability: CoinAbility | None) -> None:
        """Bind the per-rollout coin ability (rebuilt on scene load / reset)."""
        self._coin_ability = coin_ability

    def set_obstacle_ability(self, obstacle_ability: ObstacleAbility | None) -> None:
        """Bind the per-rollout obstacle ability (chips + box annotation)."""
        self._obstacle_ability = obstacle_ability

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
        rgb = self._annotate_obstacle(rgb, frame)
        rgb = self._draw_hud_chips(rgb)
        self._frame_index += 1
        processed = replace(frame, model_rgb_host_uint8=_HostRGBFrame(rgb))
        self._last_timestamp_us = frame.timestamp_us
        self._last_processed = processed
        return processed

    def _anything_active(self) -> bool:
        coins_active = self._coin_ability is not None and self._coin_ability.enabled
        style_active = self._style_ability is not None and (
            self._style_ability.active_skin_name != "base"
            or getattr(self._style_ability, "active_weather_name", "clear") != "clear"
        )
        obstacle_active = (
            self._obstacle_ability is not None and self._obstacle_ability.active
        )
        return coins_active or style_active or obstacle_active

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
            sprite_w, sprite_h = scaled_sprite_size(
                self._coin_sprite.size, sprite.height_px, squash
            )
            resized = self._coin_sprite.resize(
                (sprite_w, sprite_h), Image.Resampling.LANCZOS
            )
            if sprite.alpha < 1.0:
                faded = np.asarray(resized).copy()
                faded[..., 3] = (
                    faded[..., 3].astype(np.float32) * sprite.alpha
                ).astype(np.uint8)
                resized = Image.fromarray(faded)
            self._composite_contact_shadow(
                canvas, sprite.center_uv, sprite_w, sprite_h, sprite.alpha
            )
            _alpha_composite_clipped(
                canvas,
                resized,
                round(sprite.center_uv[0] - sprite_w / 2.0),
                round(sprite.center_uv[1] - sprite_h / 2.0),
            )
        return np.asarray(canvas.convert("RGB"))

    @staticmethod
    def _composite_contact_shadow(
        canvas: Image.Image,
        center_uv: tuple[float, float],
        sprite_w: int,
        sprite_h: int,
        alpha: float,
    ) -> None:
        """Blend a soft elliptical shadow under one coin (light on purpose)."""
        shadow_w = max(2, round(sprite_w * _SHADOW_WIDTH_FRACTION))
        shadow_h = max(1, round(sprite_h * _SHADOW_HEIGHT_FRACTION))
        blur = max(1, shadow_h // 3)
        # Pad well past the blur radius so the falloff isn't clipped into a
        # visible rectangle on flat road textures.
        pad = 3 * blur + 2
        shadow = Image.new(
            "RGBA", (shadow_w + 2 * pad, shadow_h + 2 * pad), (0, 0, 0, 0)
        )
        draw = ImageDraw.Draw(shadow)
        draw.ellipse(
            [pad, pad, shadow_w + pad, shadow_h + pad],
            fill=(0, 0, 0, round(_SHADOW_MAX_ALPHA * alpha)),
        )
        shadow = shadow.filter(ImageFilter.GaussianBlur(radius=blur))
        _alpha_composite_clipped(
            canvas,
            shadow,
            round(center_uv[0] - shadow.width / 2.0),
            round(center_uv[1] + sprite_h * _SHADOW_DROP_FRACTION),
        )

    def _annotate_obstacle(self, rgb: np.ndarray, frame: PresentedFrame) -> np.ndarray:
        """Outline the obstacle clone's 3D box (evidence aid, flag-gated)."""
        obstacle = self._obstacle_ability
        if (
            not self._config.obstacle.annotate
            or obstacle is None
            or obstacle.event is None
            or frame.rig_to_world is None
        ):
            return rgb
        event = obstacle.event
        center = event.center_at(int(frame.timestamp_us))
        if center is None:
            return rgb
        height, width = rgb.shape[:2]
        camera_model = self._require_camera_model(width, height)
        if camera_model is None:
            return rgb
        # Nearest-sample orientation is plenty for an annotation outline.
        sample = int(
            np.argmin(np.abs(event.timestamps_us - np.int64(frame.timestamp_us)))
        )
        rotation = _quat_to_matrix(event.orientations_xyzw[sample])
        half = np.asarray(event.dimensions_lwh, dtype=np.float32) / 2.0
        signs = np.array(
            [[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)],
            dtype=np.float32,
        )
        corners = center[None, :] + (signs * half[None, :]) @ rotation.T
        uv, _depth, forward = camera_model.project_world(
            corners, np.asarray(frame.rig_to_world, dtype=np.float32)
        )
        if not forward.all():
            return rgb
        canvas = Image.fromarray(rgb, mode="RGB")
        draw = ImageDraw.Draw(canvas)
        edges = (
            (0, 1),
            (0, 2),
            (1, 3),
            (2, 3),  # bottom/top faces per z pairing
            (4, 5),
            (4, 6),
            (5, 7),
            (6, 7),
            (0, 4),
            (1, 5),
            (2, 6),
            (3, 7),
        )
        for a, b in edges:
            draw.line(
                [tuple(uv[a].tolist()), tuple(uv[b].tolist())],
                fill=(255, 60, 60),
                width=2,
            )
        return np.asarray(canvas)

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
            weather_name = getattr(style, "active_weather_name", "clear")
            if weather_name != "clear":
                labels.append(f"WEATHER {weather_name.upper()}")
        coins = self._coin_ability
        if coins is not None and coins.enabled:
            labels.append(f"COINS {coins.collected_count}")
        obstacle = self._obstacle_ability
        if obstacle is not None and obstacle.active:
            labels.append("OBSTACLE!")
        if obstacle is not None and obstacle.hit_count:
            labels.append(f"HITS {obstacle.hit_count}")
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


def scaled_sprite_size(
    sprite_wh: tuple[int, int], height_px: float, squash: float
) -> tuple[int, int]:
    """On-screen sprite size from the FTheta-projected vertical extent.

    Height is authoritative (projected coin diameter); width preserves the
    sprite's native aspect ratio and carries the spin squash.
    """
    native_w, native_h = sprite_wh
    if native_w <= 0 or native_h <= 0:
        raise ValueError("sprite dimensions must be positive")
    sprite_h = max(3, round(height_px))
    sprite_w = max(2, round(height_px * (native_w / native_h) * squash))
    return sprite_w, sprite_h


def _alpha_composite_clipped(
    canvas: Image.Image, source: Image.Image, left: int, top: int
) -> None:
    """``alpha_composite`` that tolerates destinations off the canvas edge."""
    crop_left = max(0, -left)
    crop_top = max(0, -top)
    crop_right = min(source.width, canvas.width - left)
    crop_bottom = min(source.height, canvas.height - top)
    if crop_left >= crop_right or crop_top >= crop_bottom:
        return
    if (crop_left, crop_top) != (0, 0) or (crop_right, crop_bottom) != source.size:
        source = source.crop((crop_left, crop_top, crop_right, crop_bottom))
    canvas.alpha_composite(source, (left + crop_left, top + crop_top))


def _quat_to_matrix(quat_xyzw: np.ndarray) -> np.ndarray:
    """Rotation matrix from a normalized xyzw quaternion (numpy-only)."""
    x, y, z, w = (float(v) for v in np.asarray(quat_xyzw, dtype=np.float64))
    norm = math.sqrt(x * x + y * y + z * z + w * w) or 1.0
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )
