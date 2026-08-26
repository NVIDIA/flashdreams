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

It rewrites ``PresentedFrame.model_rgb_host_uint8`` with a composited copy
wrapped in a lazy-source shim, so both downstream branches (the SlangPy
CUDA fast path via ``to_cuda_tensor`` and the PIL/MJPEG path via
``to_numpy``) keep working.

Two compositing paths, selected per frame by the source's residency:

- CUDA source (the native fast path's lazy CUDA uint8 HWC frame, also the
  MJPEG path before its host prefetch): torch ops on the device via
  :class:`~.gpu_compositor.LiveEditFrameCompositor` — no host round trip,
  the result is re-wrapped as a ``LazyCudaFrame`` whose CUDA event orders
  downstream consumers after the compositing;
- host source (plain numpy arrays, already-materialized frames): the
  original PIL path. The obstacle box-outline annotation
  (``--live-edit-obstacle-annotate``, a debug/evidence aid) always takes
  the host path.
"""

from __future__ import annotations

import math
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger
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


_ITEM_PLACEHOLDER_COLORS = {
    "rain": ((70, 130, 240, 255), (20, 60, 160, 255), "R"),
    "snow": ((235, 245, 255, 255), (120, 160, 210, 255), "S"),
    "mystery": ((250, 170, 40, 255), (160, 90, 10, 255), "?"),
    "nitro": ((90, 225, 110, 255), (20, 120, 40, 255), "N"),
}


def procedural_item_sprite(item_type: str, size_px: int = 96) -> Image.Image:
    """Render a placeholder RGBA icon for one effect-item type.

    The default when no sprite path is configured (real item sprites are
    local-only files, never bundled — same policy as the coin sprite):
    a filled rounded square in a per-type color carrying its letter.
    """
    if item_type not in _ITEM_PLACEHOLDER_COLORS:
        raise ValueError(f"unknown item type {item_type!r}")
    fill, rim, letter = _ITEM_PLACEHOLDER_COLORS[item_type]
    sprite = Image.new("RGBA", (size_px, size_px), (0, 0, 0, 0))
    draw = ImageDraw.Draw(sprite)
    margin = size_px // 10
    draw.rounded_rectangle(
        [margin, margin, size_px - margin, size_px - margin],
        radius=size_px // 6,
        fill=fill,
        outline=rim,
        width=max(2, size_px // 16),
    )
    text_box = draw.textbbox((0, 0), letter)
    draw.text(
        (
            (size_px - (text_box[2] - text_box[0])) / 2.0,
            (size_px - (text_box[3] - text_box[1])) / 2.0 - text_box[1],
        ),
        letter,
        fill=rim,
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
        # Host-composited frame (host source or annotate fallback): the
        # native consumer re-uploads it. CUDA sources take the torch
        # compositing path instead and never round-trip through here.
        import torch

        return torch.from_numpy(self._rgb).cuda()


class _LiveEditPerfLog:
    """Rolling p50/p95 self-report of live-edit per-frame costs.

    Enabled by ``--live-edit-perf-log N`` / ``LIVE_EDIT_PERF_LOG=N``; logs
    one line every ``N`` composited frames so remote users can report the
    coin-update CPU cost and the compositor's enqueue/GPU cost from their
    machine. GPU timings come from CUDA event pairs resolved lazily at
    report time — pairs still in flight are skipped, never synchronized on,
    so the report itself adds no GPU sync point.
    """

    def __init__(self, every_frames: int) -> None:
        self._every = every_frames
        self._coin_ms: list[float] = []
        self._enqueue_ms: list[float] = []
        self._sprite_counts: list[int] = []
        self._gpu_event_pairs: list[tuple[Any, Any]] = []

    def record(
        self,
        *,
        coin_ms: float,
        enqueue_ms: float,
        sprite_count: int,
        gpu_events: tuple[Any, Any] | None,
    ) -> None:
        """Add one frame's samples; emit the report every N frames."""
        self._coin_ms.append(coin_ms)
        self._enqueue_ms.append(enqueue_ms)
        self._sprite_counts.append(sprite_count)
        if gpu_events is not None:
            self._gpu_event_pairs.append(gpu_events)
        if len(self._coin_ms) >= self._every:
            self._report()

    def _report(self) -> None:
        gpu_ms = [
            start.elapsed_time(end)
            for start, end in self._gpu_event_pairs
            if end.query()
        ]
        gpu_summary = (
            f"compositor_gpu_ms p50={np.percentile(gpu_ms, 50):.3f} "
            f"p95={np.percentile(gpu_ms, 95):.3f} (n={len(gpu_ms)})"
            if gpu_ms
            else "compositor_gpu_ms n/a"
        )
        logger.info(
            f"[live-edit] perf over {len(self._coin_ms)} frames: "
            f"coin_cpu_ms p50={np.percentile(self._coin_ms, 50):.3f} "
            f"p95={np.percentile(self._coin_ms, 95):.3f} | "
            f"compositor_enqueue_cpu_ms p50={np.percentile(self._enqueue_ms, 50):.3f} "
            f"p95={np.percentile(self._enqueue_ms, 95):.3f} | "
            f"{gpu_summary} | "
            f"sprites p50={np.percentile(self._sprite_counts, 50):.0f} "
            f"max={max(self._sprite_counts)}"
        )
        self._coin_ms.clear()
        self._enqueue_ms.clear()
        self._sprite_counts.clear()
        self._gpu_event_pairs.clear()


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
        self._item_ability: Any | None = None
        self._nitro_ability: Any | None = None
        self._item_sprites: dict[str, Image.Image] = (
            self._load_item_sprites(config.items) if config.items.enabled else {}
        )
        self._gpu_compositor: Any | None = None
        self._perf_log = (
            _LiveEditPerfLog(config.perf_log_every_frames)
            if config.perf_log_every_frames > 0
            else None
        )
        # Test hook: lets CPU torch tensors exercise the torch path.
        self._allow_cpu_tensor_source = False

    def set_coin_ability(self, coin_ability: CoinAbility | None) -> None:
        """Bind the per-rollout coin ability (rebuilt on scene load / reset)."""
        self._coin_ability = coin_ability

    def set_obstacle_ability(self, obstacle_ability: ObstacleAbility | None) -> None:
        """Bind the per-rollout obstacle controller (chips + box annotation)."""
        self._obstacle_ability = obstacle_ability

    def set_item_ability(self, item_ability: Any | None) -> None:
        """Bind the per-rollout effect-item ability (sprites + HUD flash)."""
        self._item_ability = item_ability

    def set_nitro_ability(self, nitro_ability: Any | None) -> None:
        """Bind the nitro ability (boost chip with countdown)."""
        self._nitro_ability = nitro_ability

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
        composited = self._process_tensor(frame)
        if composited is None:
            rgb = np.ascontiguousarray(
                np.asarray(frame.model_rgb_host_uint8, dtype=np.uint8)[..., :3]
            )
            rgb = self._apply_style_filter(rgb)
            rgb = self._composite_coins(rgb, frame)
            rgb = self._annotate_obstacle(rgb, frame)
            rgb = self._draw_hud_chips(rgb)
            composited = _HostRGBFrame(rgb)
        self._frame_index += 1
        processed = replace(frame, model_rgb_host_uint8=composited)
        self._last_timestamp_us = frame.timestamp_us
        self._last_processed = processed
        return processed

    def _process_tensor(self, frame: PresentedFrame) -> Any | None:
        """Composite on the source's device when it is a torch tensor.

        Returns ``None`` to fall back to the host/PIL path: for plain
        numpy sources, already-host-materialized lazy frames, and whenever
        the obstacle box-outline annotation (host-only debug aid) is on.
        The returned frame is a ``LazyCudaFrame`` whose source event is
        recorded after the compositing ops, so downstream consumers
        (Vulkan interop copy stream, MJPEG host prefetch) order correctly.
        """
        if self._config.obstacle.annotate and self._obstacle_ability is not None:
            return None
        source = frame.model_rgb_host_uint8
        to_cuda_tensor = getattr(source, "to_cuda_tensor", None)
        if not callable(to_cuda_tensor):
            return None
        import torch

        try:
            tensor = to_cuda_tensor()
        except RuntimeError:
            # Already materialized on the host (numpy XOR cuda contract).
            return None
        if (
            not torch.is_tensor(tensor)
            or tensor.ndim != 3
            or tensor.dtype != torch.uint8
        ):
            return None
        if not tensor.is_cuda and not self._allow_cpu_tensor_source:
            return None
        if tensor.is_cuda:
            to_cuda_event = getattr(source, "to_cuda_event", None)
            event = to_cuda_event() if callable(to_cuda_event) else None
            if event is not None:
                torch.cuda.current_stream(tensor.device).wait_event(event)
        source_tensor = tensor[..., :3]

        compositor = self._require_gpu_compositor()
        style = self._style_ability
        sharpen = style is not None and style.active_skin_name != "base"
        perf = self._perf_log
        coin_start = time.perf_counter()
        height, width = source_tensor.shape[:2]
        sprites = self._gather_sprites(frame, width, height)
        enqueue_start = time.perf_counter()
        gpu_events = None
        if perf is not None and source_tensor.is_cuda:
            gpu_events = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            gpu_events[0].record(torch.cuda.current_stream(source_tensor.device))
        work = compositor.composite(
            source_tensor,
            sprites=sprites,
            frame_index=self._frame_index,
            labels=self._hud_labels(),
            sharpen_sigma=self._config.sharpen_sigma if sharpen else 0.0,
            sharpen_amount=self._config.sharpen_amount if sharpen else 0.0,
        )
        if gpu_events is not None:
            gpu_events[1].record(torch.cuda.current_stream(work.device))
        if perf is not None:
            enqueue_end = time.perf_counter()
            perf.record(
                coin_ms=(enqueue_start - coin_start) * 1.0e3,
                enqueue_ms=(enqueue_end - enqueue_start) * 1.0e3,
                sprite_count=len(sprites),
                gpu_events=gpu_events,
            )

        done_event = None
        if work.is_cuda:
            done_event = torch.cuda.Event()
            done_event.record(torch.cuda.current_stream(work.device))
        from flashdreams.infra.acceleration.frame_prefetch import LazyCudaFrame

        return LazyCudaFrame(
            work.unsqueeze(0),
            0,
            source_event=done_event,
            lost_source_message=(
                "Live-edit composited frame lost its tensor before materialization."
            ),
            already_materialized_message=(
                "Live-edit composited frame was already materialized on the host."
            ),
        )

    def _require_gpu_compositor(self) -> Any:
        if self._gpu_compositor is None:
            from crazy_robotaxi.live_edit.gpu_compositor import (
                LiveEditFrameCompositor,
            )

            self._gpu_compositor = LiveEditFrameCompositor(
                self._coin_sprite, sprite_bank=self._item_sprites
            )
        return self._gpu_compositor

    def _anything_active(self) -> bool:
        coins_active = self._coin_ability is not None and self._coin_ability.enabled
        items_active = self._item_ability is not None and self._item_ability.enabled
        style_active = self._style_ability is not None and (
            self._style_ability.active_skin_name != "base"
            or getattr(self._style_ability, "active_weather_name", "clear") != "clear"
        )
        obstacle_active = (
            self._obstacle_ability is not None and self._obstacle_ability.active
        )
        return coins_active or items_active or style_active or obstacle_active

    def _apply_style_filter(self, rgb: np.ndarray) -> np.ndarray:
        style = self._style_ability
        if style is None or style.active_skin_name == "base":
            return rgb
        return unsharp_rgb(
            rgb,
            sigma=self._config.sharpen_sigma,
            amount=self._config.sharpen_amount,
        )

    def _gather_sprites(
        self, frame: PresentedFrame, width: int, height: int
    ) -> tuple[Any, ...]:
        """Coin + effect-item sprites merged far-to-near for one frame."""
        if frame.rig_to_world is None:
            return ()
        camera_model = self._require_camera_model(width, height)
        if camera_model is None:
            return ()
        rig_to_world = np.asarray(frame.rig_to_world, dtype=np.float32)
        sprites: list[Any] = []
        for ability in (self._coin_ability, self._item_ability):
            if ability is not None and ability.enabled:
                sprites.extend(
                    ability.visible_sprites(
                        rig_to_world,
                        camera_model,
                        image_width=width,
                        image_height=height,
                    )
                )
        if self._coin_ability is not None and self._item_ability is not None:
            # Each ability emits far-to-near; the merged painter's order
            # must too.
            sprites.sort(key=lambda sprite: -sprite.distance_m)
        return tuple(sprites)

    def _sprite_image(self, sprite_key: str) -> Image.Image:
        """Bank sprite for one key on the host path (coin fallback)."""
        return self._item_sprites.get(sprite_key, self._coin_sprite)

    def _composite_coins(self, rgb: np.ndarray, frame: PresentedFrame) -> np.ndarray:
        height, width = rgb.shape[:2]
        sprites = self._gather_sprites(frame, width, height)
        if not sprites:
            return rgb
        canvas = Image.fromarray(rgb, mode="RGB").convert("RGBA")
        for sprite in sprites:
            image = self._sprite_image(sprite.sprite_key)
            squash = (
                coin_squash(sprite.spin_phase, self._frame_index)
                if sprite.spin
                else 1.0
            )
            sprite_w, sprite_h = scaled_sprite_size(
                image.size, sprite.height_px, squash
            )
            resized = image.resize((sprite_w, sprite_h), Image.Resampling.LANCZOS)
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
        """Outline each obstacle event's 3D box (evidence aid, flag-gated)."""
        obstacle = self._obstacle_ability
        if (
            not self._config.obstacle.annotate
            or obstacle is None
            or not obstacle.events
            or frame.rig_to_world is None
        ):
            return rgb
        height, width = rgb.shape[:2]
        camera_model = self._require_camera_model(width, height)
        if camera_model is None:
            return rgb
        canvas: Image.Image | None = None
        draw: ImageDraw.ImageDraw | None = None
        signs = np.array(
            [[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)],
            dtype=np.float32,
        )
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
        for event in obstacle.events:
            center = event.center_at(int(frame.timestamp_us))
            if center is None:
                continue
            orientation = event.orientation_at(int(frame.timestamp_us))
            if orientation is None:
                continue
            rotation = _quat_to_matrix(orientation)
            half = np.asarray(event.dimensions_lwh, dtype=np.float32) / 2.0
            corners = center[None, :] + (signs * half[None, :]) @ rotation.T
            uv, _depth, forward = camera_model.project_world(
                corners, np.asarray(frame.rig_to_world, dtype=np.float32)
            )
            if not forward.all():
                continue
            if canvas is None:
                canvas = Image.fromarray(rgb, mode="RGB")
                draw = ImageDraw.Draw(canvas)
            for a, b in edges:
                draw.line(
                    [tuple(uv[a].tolist()), tuple(uv[b].tolist())],
                    fill=(255, 60, 60),
                    width=2,
                )
        return rgb if canvas is None else np.asarray(canvas)

    def _hud_labels(self) -> list[str]:
        """Chip labels for the current ability state (shared by both paths)."""
        labels: list[str] = []
        style = self._style_ability
        if style is not None:
            skin_label = f"SKIN {style.active_skin_name.upper()}"
            remaining_s = getattr(style, "skin_seconds_remaining", None)
            if remaining_s is not None:
                # Timed power-up mode: countdown at chunk granularity.
                skin_label += f" {remaining_s:.1f}s"
            labels.append(skin_label)
            weather_name = getattr(style, "active_weather_name", "clear")
            if weather_name != "clear":
                weather_label = f"WEATHER {weather_name.upper()}"
                weather_s = getattr(style, "weather_seconds_remaining", None)
                if weather_s is not None:
                    # Timed weather: countdown at chunk granularity.
                    weather_label += f" {weather_s:.1f}s"
                labels.append(weather_label)
        coins = self._coin_ability
        if coins is not None and coins.enabled:
            labels.append(f"COINS {coins.collected_count}")
        items = self._item_ability
        if items is not None and items.enabled:
            flash = items.flash_label
            if flash is not None:
                labels.append(flash)
        nitro = self._nitro_ability
        if nitro is not None and nitro.active:
            # Boost chip with countdown (game-time seconds, 0.1 s steps).
            labels.append(f"NITRO x{nitro.boost:.1f} {nitro.seconds_remaining:.1f}s")
        obstacle = self._obstacle_ability
        if obstacle is not None and obstacle.active:
            n = len(obstacle.events)
            labels.append("OBSTACLE!" if n <= 1 else f"TRAFFIC x{n}")
        if obstacle is not None and obstacle.hit_count:
            labels.append(f"HITS {obstacle.hit_count}")
        return labels

    def _draw_hud_chips(self, rgb: np.ndarray) -> np.ndarray:
        """Draw the skin-name and coin-counter chips into the frame.

        Drawn into pixels so both the native window and the MJPEG stream
        show them without touching ``_draw_taxi_hud``; the polished version
        belongs in the taxi HUD panel (needs an upstream hook or edit).
        """
        labels = self._hud_labels()
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

    @classmethod
    def _load_item_sprites(cls, items_config: Any) -> dict[str, Image.Image]:
        """Per-type effect-item sprites (procedural placeholders by default)."""
        from crazy_robotaxi.live_edit.config import ITEM_TYPES

        sprites: dict[str, Image.Image] = {}
        for item_type in ITEM_TYPES:
            path = items_config.sprite_path(item_type)
            sprites[item_type] = (
                procedural_item_sprite(item_type)
                if path is None
                else cls._load_sprite(path)
            )
        return sprites


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
