# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Torch compositor for live-edit pixels on device-resident model frames.

The native window path hands ``PresentedFrame.model_rgb_host_uint8`` to the
Vulkan HUD as a CUDA uint8 HWC tensor (``LazyCudaFrame``); materializing it
to host numpy for PIL compositing forces a GPU->CPU->CPU-composite->GPU
round trip per frame (~10 fps observed). This module keeps the frame on
device: sprites, contact shadows, and HUD chips are pre-rendered once (PIL,
host) and uploaded as cached tensors; the per-frame work is a handful of
small alpha-blended ROI writes plus an optional separable-Gaussian unsharp
mask, all plain torch ops on the frame's device.

Every operation is device-agnostic (CPU tensors run the identical code),
so the compositing math is unit-testable without a GPU. Visual parity with
the PIL path is approximate by design: sprites scale with bilinear
interpolation instead of Lanczos, and the contact shadow is one canonical
blurred ellipse rescaled per coin instead of a per-coin Gaussian blur.
"""

from __future__ import annotations

import math
from collections import OrderedDict
from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFilter
from torch import Tensor

if TYPE_CHECKING:
    from crazy_robotaxi.live_edit.coin_ability import CoinSprite

_CHIP_CACHE_MAX = 64
"""Cached chip textures (labels change only on pickups/state switches)."""

_SPRITE_REF_PX = 96
"""Canonical sprite edge used for the pre-uploaded coin/shadow textures."""

_SCALED_CACHE_MAX = 1024
"""Cached per-size sprite/shadow textures (a few KB each)."""


def _rgba_to_tensors(image: Image.Image, device: torch.device) -> tuple[Tensor, Tensor]:
    """Split an RGBA image into ``([3,H,W] rgb 0..255, [1,H,W] alpha 0..1)``."""
    array = np.asarray(image.convert("RGBA"), dtype=np.float32)
    rgba = torch.from_numpy(array).to(device).permute(2, 0, 1)
    return rgba[:3], rgba[3:] / 255.0


def _gaussian_kernel1d(sigma: float, device: torch.device) -> Tensor:
    """Normalized 1D Gaussian, radius 3 sigma (PIL GaussianBlur parity-ish)."""
    radius = max(1, math.ceil(3.0 * sigma))
    x = torch.arange(-radius, radius + 1, dtype=torch.float32, device=device)
    kernel = torch.exp(-(x * x) / (2.0 * sigma * sigma))
    return kernel / kernel.sum()


def alpha_blend_(
    canvas_hwc_uint8: Tensor,
    rgb: Tensor | None,
    alpha: Tensor,
    left: int,
    top: int,
) -> None:
    """Alpha-composite one texture into the canvas, clipped at the edges.

    Args:
        canvas_hwc_uint8: ``[H,W,3]`` uint8 frame, written in place.
        rgb: ``[3,h,w]`` float source colors in 0..255; ``None`` blends
            black (shadow).
        alpha: ``[1,h,w]`` float coverage in 0..1.
        left, top: Destination of the texture's top-left corner; may lie
            (partly) off the canvas.
    """
    height, width = canvas_hwc_uint8.shape[:2]
    src_h, src_w = alpha.shape[-2:]
    x0, y0 = max(0, left), max(0, top)
    x1, y1 = min(width, left + src_w), min(height, top + src_h)
    if x0 >= x1 or y0 >= y1:
        return
    sx, sy = x0 - left, y0 - top
    a = alpha[:, sy : sy + (y1 - y0), sx : sx + (x1 - x0)].permute(1, 2, 0)
    roi = canvas_hwc_uint8[y0:y1, x0:x1].to(torch.float32)
    if rgb is None:
        out = roi * (1.0 - a)
    else:
        c = rgb[:, sy : sy + (y1 - y0), sx : sx + (x1 - x0)].permute(1, 2, 0)
        out = roi * (1.0 - a) + c * a
    canvas_hwc_uint8[y0:y1, x0:x1] = out.round_().clamp_(0.0, 255.0).to(torch.uint8)


def _blend_float_(
    canvas_hwc_f32: Tensor,
    premultiplied_rgb_hwc: Tensor | None,
    one_minus_alpha_hw1: Tensor,
    left: int,
    top: int,
    fade: float = 1.0,
) -> None:
    """In-place premultiplied blend on a float32 HWC canvas (hot path).

    Same clipping semantics as :func:`alpha_blend_`, but the canvas stays
    float across all blends of a frame (one uint8 round-trip per frame
    instead of one per blend) and the textures are pre-baked so each blend
    at full opacity is ``roi = roi * (1 - a) [+ rgb * a]`` — one or two
    small kernels. ``premultiplied_rgb_hwc=None`` darkens toward black
    (contact shadow).
    """
    height, width = canvas_hwc_f32.shape[:2]
    src_h, src_w = one_minus_alpha_hw1.shape[:2]
    x0, y0 = max(0, left), max(0, top)
    x1, y1 = min(width, left + src_w), min(height, top + src_h)
    if x0 >= x1 or y0 >= y1:
        return
    sx, sy = x0 - left, y0 - top
    om = one_minus_alpha_hw1[sy : sy + (y1 - y0), sx : sx + (x1 - x0)]
    roi = canvas_hwc_f32[y0:y1, x0:x1]
    # With fade f the factor on the canvas is 1 - f*(1-om) = (1-f) + f*om.
    roi.mul_(om if fade >= 1.0 else (1.0 - fade) + fade * om)
    if premultiplied_rgb_hwc is not None:
        c = premultiplied_rgb_hwc[sy : sy + (y1 - y0), sx : sx + (x1 - x0)]
        roi.add_(c if fade >= 1.0 else fade * c)


class LiveEditFrameCompositor:
    """Pre-uploaded textures + per-frame ROI blends for one coin sprite.

    Mirrors the PIL path in :mod:`crazy_robotaxi.live_edit.presenter`
    (:func:`~.presenter.unsharp_rgb`, coin/shadow compositing, HUD chips)
    with torch ops on the frame's device. One instance per presenter;
    texture caches are keyed by device so CPU tests and CUDA serving share
    the code.
    """

    def __init__(self, coin_sprite: Image.Image) -> None:
        self._coin_sprite_image = coin_sprite.convert("RGBA")
        self._sprite_cache: dict[torch.device, tuple[Tensor, Tensor]] = {}
        self._shadow_cache: dict[torch.device, Tensor] = {}
        self._chip_cache: OrderedDict[tuple[str, torch.device], tuple[Tensor, Tensor]]
        self._chip_cache = OrderedDict()
        self._kernel_cache: dict[tuple[float, torch.device], Tensor] = {}
        # Per-size texture caches: coin sizes quantize to a few dozen
        # (height from distance, width from the 36-frame squash cycle), so
        # the per-frame hot path is dictionary lookups + fused lerp blends,
        # no per-frame F.interpolate.
        self._scaled_sprite_cache: OrderedDict[
            tuple[torch.device, int, int], Tensor
        ] = OrderedDict()
        self._scaled_shadow_cache: OrderedDict[
            tuple[torch.device, int, int], Tensor
        ] = OrderedDict()

    ## Texture caches

    def _sprite(self, device: torch.device) -> tuple[Tensor, Tensor]:
        cached = self._sprite_cache.get(device)
        if cached is None:
            cached = _rgba_to_tensors(self._coin_sprite_image, device)
            self._sprite_cache[device] = cached
        return cached

    def _shadow(self, device: torch.device) -> Tensor:
        """Canonical blurred contact-shadow alpha at max strength.

        Rendered once with the exact PIL routine of the host path at the
        reference sprite size; per-coin scaling stretches it, which also
        scales the blur falloff proportionally.
        """
        cached = self._shadow_cache.get(device)
        if cached is None:
            from crazy_robotaxi.live_edit.presenter import (
                _SHADOW_HEIGHT_FRACTION,
                _SHADOW_MAX_ALPHA,
                _SHADOW_WIDTH_FRACTION,
            )

            shadow_w = max(2, round(_SPRITE_REF_PX * _SHADOW_WIDTH_FRACTION))
            shadow_h = max(1, round(_SPRITE_REF_PX * _SHADOW_HEIGHT_FRACTION))
            blur = max(1, shadow_h // 3)
            pad = 3 * blur + 2
            image = Image.new(
                "RGBA", (shadow_w + 2 * pad, shadow_h + 2 * pad), (0, 0, 0, 0)
            )
            draw = ImageDraw.Draw(image)
            draw.ellipse(
                [pad, pad, shadow_w + pad, shadow_h + pad],
                fill=(0, 0, 0, _SHADOW_MAX_ALPHA),
            )
            image = image.filter(ImageFilter.GaussianBlur(radius=blur))
            _, alpha = _rgba_to_tensors(image, device)
            cached = alpha.unsqueeze(0)  # [1,1,H,W] for interpolate
            self._shadow_cache[device] = cached
        return cached

    def _chip(self, label: str, device: torch.device) -> tuple[Tensor, Tensor]:
        """Chip texture ``([h,w,3] rgb*a, [h,w,1] 1-a)``, rendered per label."""
        key = (label, device)
        cached = self._chip_cache.get(key)
        if cached is not None:
            self._chip_cache.move_to_end(key)
            return cached
        from crazy_robotaxi.live_edit.presenter import (
            _COUNTER_CHIP_RGBA,
            _COUNTER_TEXT_RGBA,
        )

        probe = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
        text_box = probe.textbbox((10, 6), label)
        image = Image.new("RGBA", (text_box[2] + 10 + 1, text_box[3] + 6 + 1), (0,) * 4)
        draw = ImageDraw.Draw(image)
        draw.rounded_rectangle(
            [0, 0, text_box[2] + 10, text_box[3] + 6],
            radius=6,
            fill=_COUNTER_CHIP_RGBA,
        )
        draw.text((10, 6), label, fill=_COUNTER_TEXT_RGBA)
        rgb, alpha = _rgba_to_tensors(image, device)
        rgb, alpha = rgb.permute(1, 2, 0), alpha.permute(1, 2, 0)
        cached = ((rgb * alpha).contiguous(), (1.0 - alpha).contiguous())
        self._chip_cache[key] = cached
        while len(self._chip_cache) > _CHIP_CACHE_MAX:
            self._chip_cache.popitem(last=False)
        return cached

    def _scaled_sprite(
        self, device: torch.device, width: int, height: int
    ) -> tuple[Tensor, Tensor]:
        """Coin texture at one size: ``([h,w,3] rgb*a, [h,w,1] 1-a)``.

        Sizes quantize to a few dozen (height from distance, width from the
        36-frame squash cycle), so per-frame work is a cache lookup.
        """
        key = (device, width, height)
        cached = self._scaled_sprite_cache.get(key)
        if cached is None:
            sprite_rgb, sprite_alpha = self._sprite(device)
            scaled = F.interpolate(
                torch.cat([sprite_rgb, sprite_alpha], dim=0).unsqueeze(0),
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            )[0].permute(1, 2, 0)
            alpha = scaled[..., 3:]
            cached = (
                (scaled[..., :3] * alpha).contiguous(),
                (1.0 - alpha).contiguous(),
            )
            self._scaled_sprite_cache[key] = cached
            while len(self._scaled_sprite_cache) > _SCALED_CACHE_MAX:
                self._scaled_sprite_cache.popitem(last=False)
        else:
            self._scaled_sprite_cache.move_to_end(key)
        return cached

    def _scaled_shadow(self, device: torch.device, width: int, height: int) -> Tensor:
        """One-minus-alpha shadow texture ``[h,w,1]`` at one size."""
        key = (device, width, height)
        cached = self._scaled_shadow_cache.get(key)
        if cached is None:
            shadow = self._shadow(device)
            scaled = F.interpolate(
                shadow, size=(height, width), mode="bilinear", align_corners=False
            )[0].permute(1, 2, 0)
            cached = (1.0 - scaled).contiguous()
            self._scaled_shadow_cache[key] = cached
            while len(self._scaled_shadow_cache) > _SCALED_CACHE_MAX:
                self._scaled_shadow_cache.popitem(last=False)
        else:
            self._scaled_shadow_cache.move_to_end(key)
        return cached

    ## Frame operations

    def composite(
        self,
        frame_hwc_uint8: Tensor,
        *,
        sprites: Sequence[CoinSprite] = (),
        frame_index: int = 0,
        labels: Sequence[str] = (),
        sharpen_sigma: float = 0.0,
        sharpen_amount: float = 0.0,
    ) -> Tensor:
        """All live-edit pixels in one pass; returns a new uint8 frame.

        The canvas is converted to float32 once, every texture blend is a
        fused in-place lerp on it, and the single round/clamp/uint8 cast
        happens at the end — the per-frame kernel count stays small enough
        for a sub-millisecond budget with a dozen coins on screen.
        """
        canvas = frame_hwc_uint8.to(torch.float32)
        if sharpen_amount > 0.0:
            canvas = self._unsharp_float(
                canvas, sigma=sharpen_sigma, amount=sharpen_amount
            )
        self._blend_coins(canvas, sprites, frame_index)
        self._blend_chips(canvas, labels)
        return canvas.round_().clamp_(0.0, 255.0).to(torch.uint8)

    def unsharp(
        self, frame_hwc_uint8: Tensor, *, sigma: float, amount: float
    ) -> Tensor:
        """Separable-Gaussian unsharp mask (torch port of ``unsharp_rgb``)."""
        if amount <= 0.0:
            return frame_hwc_uint8
        sharpened = self._unsharp_float(
            frame_hwc_uint8.to(torch.float32), sigma=sigma, amount=amount
        )
        return sharpened.clamp_(0.0, 255.0).round_().to(torch.uint8)

    def _unsharp_float(
        self, canvas_hwc_f32: Tensor, *, sigma: float, amount: float
    ) -> Tensor:
        device = canvas_hwc_f32.device
        key = (float(sigma), device)
        kernel = self._kernel_cache.get(key)
        if kernel is None:
            kernel = _gaussian_kernel1d(sigma, device)
            self._kernel_cache[key] = kernel
        radius = (kernel.numel() - 1) // 2
        image = canvas_hwc_f32.permute(2, 0, 1).unsqueeze(0)
        padded = F.pad(image, (radius, radius, 0, 0), mode="replicate")
        blurred = F.conv2d(
            padded, kernel.view(1, 1, 1, -1).expand(3, 1, 1, -1), groups=3
        )
        padded = F.pad(blurred, (0, 0, radius, radius), mode="replicate")
        blurred = F.conv2d(
            padded, kernel.view(1, 1, -1, 1).expand(3, 1, -1, 1), groups=3
        )
        sharpened = (1.0 + amount) * image - amount * blurred
        return sharpened.squeeze(0).permute(1, 2, 0).contiguous()

    def composite_coins(
        self,
        frame_hwc_uint8: Tensor,
        sprites: Sequence[CoinSprite],
        frame_index: int,
    ) -> None:
        """Blend the projected coin sprites in place (uint8 convenience)."""
        frame_hwc_uint8.copy_(
            self.composite(frame_hwc_uint8, sprites=sprites, frame_index=frame_index)
        )

    def draw_chips(self, frame_hwc_uint8: Tensor, labels: Sequence[str]) -> None:
        """Blend the stacked HUD chips in place (uint8 convenience)."""
        frame_hwc_uint8.copy_(self.composite(frame_hwc_uint8, labels=labels))

    def _blend_coins(
        self,
        canvas_hwc_f32: Tensor,
        sprites: Sequence[CoinSprite],
        frame_index: int,
    ) -> None:
        """Blend sprites far-to-near (input order) onto the float canvas."""
        if not sprites:
            return
        from crazy_robotaxi.live_edit.coin_ability import coin_squash
        from crazy_robotaxi.live_edit.presenter import (
            _SHADOW_DROP_FRACTION,
            scaled_sprite_size,
        )

        device = canvas_hwc_f32.device
        shadow_ref = self._shadow(device)
        shadow_scale_w = shadow_ref.shape[-1] / _SPRITE_REF_PX
        shadow_scale_h = shadow_ref.shape[-2] / _SPRITE_REF_PX
        for sprite in sprites:
            squash = coin_squash(sprite.spin_phase, frame_index)
            sprite_w, sprite_h = scaled_sprite_size(
                self._coin_sprite_image.size, sprite.height_px, squash
            )
            shadow_w = max(2, round(sprite_w * shadow_scale_w))
            shadow_h = max(2, round(sprite_h * shadow_scale_h))
            _blend_float_(
                canvas_hwc_f32,
                None,
                self._scaled_shadow(device, shadow_w, shadow_h),
                round(sprite.center_uv[0] - shadow_w / 2.0),
                round(sprite.center_uv[1] + sprite_h * _SHADOW_DROP_FRACTION),
                fade=sprite.alpha,
            )
            premultiplied, one_minus = self._scaled_sprite(device, sprite_w, sprite_h)
            _blend_float_(
                canvas_hwc_f32,
                premultiplied,
                one_minus,
                round(sprite.center_uv[0] - sprite_w / 2.0),
                round(sprite.center_uv[1] - sprite_h / 2.0),
                fade=sprite.alpha,
            )

    def _blend_chips(self, canvas_hwc_f32: Tensor, labels: Sequence[str]) -> None:
        if not labels:
            return
        from crazy_robotaxi.live_edit.presenter import _COUNTER_MARGIN_PX

        device = canvas_hwc_f32.device
        y0 = _COUNTER_MARGIN_PX
        for label in labels:
            premultiplied, one_minus = self._chip(label, device)
            _blend_float_(
                canvas_hwc_f32, premultiplied, one_minus, _COUNTER_MARGIN_PX, y0
            )
            y0 += one_minus.shape[0] - 1 + 8
