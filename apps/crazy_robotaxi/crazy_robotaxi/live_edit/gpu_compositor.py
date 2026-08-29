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
import os
from collections import OrderedDict
from collections.abc import Sequence
from typing import TYPE_CHECKING, Protocol

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
"""Cached per-size merged coin textures (a few KB each)."""

_FADE_STEPS = 16
"""Distance-fade quantization for the texture cache. Baking the fade into
the cached texture makes every coin a single full-opacity blend (the hot
path is CPU-launch-bound, so per-coin torch-op count dominates); 1/16 alpha
steps are imperceptible on the 20 m fade ramp."""

_COUNTER_MARGIN_PX = 12
_COUNTER_TEXT_RGBA = (255, 255, 255, 255)
_COUNTER_CHIP_RGBA = (30, 30, 30, 180)
_SHADOW_MAX_ALPHA = 60
_SHADOW_WIDTH_FRACTION = 0.9
_SHADOW_HEIGHT_FRACTION = 0.22
_SHADOW_DROP_FRACTION = 0.62


def _scaled_sprite_size(
    original_wh: tuple[int, int], target_height: float, squash: float
) -> tuple[int, int]:
    """Scale a sprite to its projected height and horizontal spin squash."""
    original_width, original_height = original_wh
    height = max(1, round(target_height))
    width = max(1, round(height * original_width / original_height * squash))
    return width, height


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
    canvas_hwc: Tensor,
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
    height, width = canvas_hwc.shape[:2]
    src_h, src_w = one_minus_alpha_hw1.shape[:2]
    x0, y0 = max(0, left), max(0, top)
    x1, y1 = min(width, left + src_w), min(height, top + src_h)
    if x0 >= x1 or y0 >= y1:
        return
    sx, sy = x0 - left, y0 - top
    om = one_minus_alpha_hw1[sy : sy + (y1 - y0), sx : sx + (x1 - x0)]
    roi = canvas_hwc[y0:y1, x0:x1]
    # With fade f the factor on the canvas is 1 - f*(1-om) = (1-f) + f*om.
    roi.mul_(om if fade >= 1.0 else (1.0 - fade) + fade * om)
    if premultiplied_rgb_hwc is not None:
        c = premultiplied_rgb_hwc[sy : sy + (y1 - y0), sx : sx + (x1 - x0)]
        roi.add_(c if fade >= 1.0 else fade * c)


def _blend_uint8_(
    canvas_hwc: Tensor,
    premultiplied_rgb_hwc: Tensor | None,
    one_minus_alpha_hw1: Tensor,
    left: int,
    top: int,
    fade: float = 1.0,
) -> None:
    """ROI-local premultiplied blend directly on a uint8 HWC canvas.

    Same math and clipping as :func:`_blend_float_`, but only the sprite's
    ROI is converted to float and back — the full frame never leaves uint8,
    so per-frame GPU memory traffic scales with on-screen sprite area
    instead of the ~5-frame-sized traffic of the float32 canvas round trip
    (selected via ``LIVE_EDIT_COMPOSITOR=roi``). Values can differ from the
    float path by at most 1 LSB where blends overlap (each ROI blend rounds
    independently).
    """
    height, width = canvas_hwc.shape[:2]
    src_h, src_w = one_minus_alpha_hw1.shape[:2]
    x0, y0 = max(0, left), max(0, top)
    x1, y1 = min(width, left + src_w), min(height, top + src_h)
    if x0 >= x1 or y0 >= y1:
        return
    sx, sy = x0 - left, y0 - top
    om = one_minus_alpha_hw1[sy : sy + (y1 - y0), sx : sx + (x1 - x0)]
    if fade < 1.0:
        om = (1.0 - fade) + fade * om
    roi = canvas_hwc[y0:y1, x0:x1]
    out = roi * om
    if premultiplied_rgb_hwc is not None:
        c = premultiplied_rgb_hwc[sy : sy + (y1 - y0), sx : sx + (x1 - x0)]
        out += c if fade >= 1.0 else fade * c
    roi.copy_(out.round_().clamp_(0.0, 255.0))


class _BlendFn(Protocol):
    """Signature shared by :func:`_blend_float_` and :func:`_blend_uint8_`."""

    def __call__(
        self,
        canvas_hwc: Tensor,
        premultiplied_rgb_hwc: Tensor | None,
        one_minus_alpha_hw1: Tensor,
        left: int,
        top: int,
        fade: float = 1.0,
    ) -> None: ...


class LiveEditFrameCompositor:
    """Pre-uploaded textures + per-frame ROI blends for one coin sprite.

    Mirrors the PIL path in :mod:`crazy_robotaxi.live_edit.presenter`
    (:func:`~.presenter.unsharp_rgb`, coin/shadow compositing, HUD chips)
    with torch ops on the frame's device. One instance per presenter;
    texture caches are keyed by device so CPU tests and CUDA serving share
    the code.
    """

    def __init__(
        self,
        coin_sprite: Image.Image,
        sprite_bank: dict[str, Image.Image] | None = None,
    ) -> None:
        self._coin_sprite_image = coin_sprite.convert("RGBA")
        # Sprite bank: "coin" plus any effect-item sprites, selected per
        # sprite by CoinSprite.sprite_key. Items reuse the whole texture
        # pipeline (shadow, fade quantization, per-size cache).
        self._sprite_images: dict[str, Image.Image] = {
            "coin": self._coin_sprite_image,
            **{
                key: image.convert("RGBA") for key, image in (sprite_bank or {}).items()
            },
        }
        # A/B switch for remote perf triage (LIVE_EDIT_COMPOSITOR=roi):
        # "roi" keeps the frame uint8 and blends each sprite slice in place
        # (minimal GPU memory traffic, ~5 tiny kernels per coin); "float"
        # (default) uses one full-frame float32 canvas with 2 kernels per
        # coin. Both paths are launch-count-bound in every measurement on
        # GB300 (GPU execution fully hidden), which makes "float" ~2x faster
        # wall-clock there; "roi" exists for machines where the compositor's
        # full-frame traffic on the inference stream is the actual cost.
        # Pair with --live-edit-perf-log to compare.
        self._roi_blends = os.environ.get("LIVE_EDIT_COMPOSITOR", "float") == "roi"
        self._sprite_cache: dict[tuple[str, torch.device], tuple[Tensor, Tensor]] = {}
        self._shadow_cache: dict[torch.device, Tensor] = {}
        self._chip_cache: OrderedDict[tuple[str, torch.device], tuple[Tensor, Tensor]]
        self._chip_cache = OrderedDict()
        self._kernel_cache: dict[tuple[float, torch.device], Tensor] = {}
        # Merged per-size coin textures (contact shadow + coin + quantized
        # distance fade pre-composited): coin sizes quantize to a few dozen
        # (height from distance, width from the 36-frame squash cycle), so
        # the per-frame hot path is one dictionary lookup and ONE blend per
        # coin — no per-frame F.interpolate, no separate shadow pass.
        self._coin_texture_cache: OrderedDict[
            tuple[str, torch.device, int, int, int], tuple[Tensor, Tensor, int]
        ] = OrderedDict()

    ## Texture caches

    def sprite_image(self, key: str) -> Image.Image:
        """The bank sprite for one key (unknown keys fall back to the coin)."""
        return self._sprite_images.get(key, self._coin_sprite_image)

    def _sprite(self, key: str, device: torch.device) -> tuple[Tensor, Tensor]:
        cached = self._sprite_cache.get((key, device))
        if cached is None:
            cached = _rgba_to_tensors(self.sprite_image(key), device)
            self._sprite_cache[(key, device)] = cached
        return cached

    def _shadow(self, device: torch.device) -> Tensor:
        """Canonical blurred contact-shadow alpha at max strength.

        Rendered once with the exact PIL routine of the host path at the
        reference sprite size; per-coin scaling stretches it, which also
        scales the blur falloff proportionally.
        """
        cached = self._shadow_cache.get(device)
        if cached is None:
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
        probe = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
        text_box = probe.textbbox((10, 6), label)
        image = Image.new(
            "RGBA",
            (round(text_box[2]) + 11, round(text_box[3]) + 7),
            (0,) * 4,
        )
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

    def _coin_texture(
        self,
        key: str,
        device: torch.device,
        width: int,
        height: int,
        alpha_q: int,
    ) -> tuple[Tensor, Tensor, int]:
        """Merged coin+shadow texture at one size and quantized fade.

        The contact shadow, the coin sprite (composited over the shadow with
        premultiplied "over", exactly associativity-equivalent to the old
        two-pass blend), and the ``alpha_q / _FADE_STEPS`` distance fade are
        all baked in, so the per-frame cost per coin is one blend.

        Returns:
            ``([h,w,3] premultiplied rgb, [h,w,1] 1-alpha, coin_left)``
            where ``coin_left`` is the coin rect's x offset inside the
            texture (the texture is anchored at the coin's top edge).
        """
        cache_key = (key, device, width, height, alpha_q)
        cached = self._coin_texture_cache.get(cache_key)
        if cached is not None:
            self._coin_texture_cache.move_to_end(cache_key)
            return cached
        sprite_rgb, sprite_alpha = self._sprite(key, device)
        scaled = F.interpolate(
            torch.cat([sprite_rgb, sprite_alpha], dim=0).unsqueeze(0),
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )[0]
        coin_rgb, coin_a = scaled[:3], scaled[3:]
        shadow_ref = self._shadow(device)
        shadow_w = max(2, round(width * shadow_ref.shape[-1] / _SPRITE_REF_PX))
        shadow_h = max(2, round(height * shadow_ref.shape[-2] / _SPRITE_REF_PX))
        shadow_a = F.interpolate(
            shadow_ref, size=(shadow_h, shadow_w), mode="bilinear", align_corners=False
        )[0]
        tex_w = max(width, shadow_w)
        coin_x = (tex_w - width) // 2
        shadow_x = (tex_w - shadow_w) // 2
        shadow_y = round(height * (0.5 + _SHADOW_DROP_FRACTION))
        tex_h = max(height, shadow_y + shadow_h)
        alpha = torch.zeros((1, tex_h, tex_w), device=device)
        rgb = torch.zeros((3, tex_h, tex_w), device=device)
        alpha[:, shadow_y : shadow_y + shadow_h, shadow_x : shadow_x + shadow_w] = (
            shadow_a
        )
        coin_region = alpha[:, :height, coin_x : coin_x + width]
        coin_region.copy_(coin_a + coin_region * (1.0 - coin_a))
        rgb[:, :height, coin_x : coin_x + width] = coin_rgb * coin_a
        fade = alpha_q / _FADE_STEPS
        cached = (
            (rgb * fade).permute(1, 2, 0).contiguous(),
            (1.0 - alpha * fade).permute(1, 2, 0).contiguous(),
            coin_x,
        )
        self._coin_texture_cache[cache_key] = cached
        while len(self._coin_texture_cache) > _SCALED_CACHE_MAX:
            self._coin_texture_cache.popitem(last=False)
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

        Every coin is exactly one blend of a merged (shadow+coin+fade)
        cached texture. The default canvas is float32 full-frame (convert
        once, fused in-place lerps, single round/clamp/cast at the end);
        ``LIVE_EDIT_COMPOSITOR=roi`` keeps the frame uint8 end to end and
        blends only the sprite ROIs (:func:`_blend_uint8_`) so per-frame
        memory traffic scales with sprite area instead of frame size. The
        unsharp mask always forces the float path (it filters the whole
        frame anyway).
        """
        if sharpen_amount <= 0.0 and self._roi_blends:
            canvas = frame_hwc_uint8.clone()
            self._blend_coins(canvas, sprites, frame_index, _blend_uint8_)
            self._blend_chips(canvas, labels, _blend_uint8_)
            return canvas
        canvas = frame_hwc_uint8.to(torch.float32)
        if sharpen_amount > 0.0:
            canvas = self._unsharp_float(
                canvas, sigma=sharpen_sigma, amount=sharpen_amount
            )
        self._blend_coins(canvas, sprites, frame_index, _blend_float_)
        self._blend_chips(canvas, labels, _blend_float_)
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
        canvas_hwc: Tensor,
        sprites: Sequence[CoinSprite],
        frame_index: int,
        blend: _BlendFn,
    ) -> None:
        """Blend sprites far-to-near (input order) onto the canvas."""
        if not sprites:
            return
        from crazy_robotaxi.live_edit.coin_ability import coin_squash

        device = canvas_hwc.device
        for sprite in sprites:
            alpha_q = min(_FADE_STEPS, round(sprite.alpha * _FADE_STEPS))
            if alpha_q <= 0:
                continue
            key = getattr(sprite, "sprite_key", "coin")
            spin = getattr(sprite, "spin", True)
            squash = coin_squash(sprite.spin_phase, frame_index) if spin else 1.0
            sprite_w, sprite_h = _scaled_sprite_size(
                self.sprite_image(key).size, sprite.height_px, squash
            )
            premultiplied, one_minus, coin_x = self._coin_texture(
                key, device, sprite_w, sprite_h, alpha_q
            )
            blend(
                canvas_hwc,
                premultiplied,
                one_minus,
                round(sprite.center_uv[0] - sprite_w / 2.0) - coin_x,
                round(sprite.center_uv[1] - sprite_h / 2.0),
            )

    def _blend_chips(
        self, canvas_hwc: Tensor, labels: Sequence[str], blend: _BlendFn
    ) -> None:
        if not labels:
            return
        device = canvas_hwc.device
        y0 = _COUNTER_MARGIN_PX
        for label in labels:
            premultiplied, one_minus = self._chip(label, device)
            blend(canvas_hwc, premultiplied, one_minus, _COUNTER_MARGIN_PX, y0)
            y0 += one_minus.shape[0] - 1 + 8
