# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Reusable Cosmos LiDAR tokenizer, streaming encoder, and streaming decoder."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, replace
from math import isqrt
from typing import Any, Literal

import torch
import torch.nn.functional as F
from torch import Tensor

from cosmos_interactive._lidar_codec import LidarCodec, network_config
from cosmos_interactive._lidar_codec.range_image import (
    circular_pad_range_video_3ch,
    crop_range_video_width,
    metric_lidar_to_network,
    network_lidar_to_metric_clip,
)
from cosmos_interactive._patchify import patchify_latent, unpatchify_latent


@dataclass(frozen=True)
class LidarGeometry:
    """A range tokenizer latent grid and its square diffusion patches."""

    channels: int = 128
    height: int = 16
    width: int = 225
    patch: int = 2
    fps: float = 10.0

    def __post_init__(self) -> None:
        import math

        if min(self.channels, self.height, self.width, self.patch) < 1:
            raise ValueError("LiDAR dimensions must be positive")
        if not math.isfinite(self.fps) or self.fps <= 0:
            raise ValueError("LiDAR fps must be finite and positive")

    @property
    def grid(self) -> tuple[int, int]:
        return (
            (self.height + self.patch - 1) // self.patch,
            (self.width + self.patch - 1) // self.patch,
        )

    @property
    def spatial_tokens(self) -> int:
        h, w = self.grid
        return h * w

    @property
    def patch_dim(self) -> int:
        return self.channels * self.patch**2


class LidarTokenizer:
    """Patch normalized range-VAE latents without changing their statistics.

    ``backend`` owns the native V1 inference network. Encoding/decoding use its
    checkpoint-owned affine statistics and calibrated range limits. Construction
    is separate from streaming state so controls and observations share weights.
    """

    def __init__(self, backend: Any, geometry: LidarGeometry = LidarGeometry()):
        self.backend = backend
        self.geometry = geometry

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: str,
        *,
        device: str = "cuda",
        version: Literal["v1p1", "v1p2"] = "v1p1",
        contiguous_linears: bool = False,
    ) -> LidarTokenizer:
        """Load the explicit tokenizer architecture, independent of checkpoint filename."""
        config = network_config(version)
        spatial = 2 ** len(config["depths"])
        geometry = LidarGeometry(
            height=config["resolution"][0] // spatial,
            width=config["resolution"][1] // spatial,
        )
        backend = LidarCodec(checkpoint, version=version, device=device)
        if contiguous_linears:
            # Flatten FP32 projections into one GEMM instead of broadcasted BMMs.
            for module in backend.model.modules():
                if isinstance(module, torch.nn.Linear):
                    module.register_forward_pre_hook(
                        lambda _module, inputs: (inputs[0].contiguous(),)
                    )
        return cls(backend, geometry)

    def with_patch_dim(self, patch_dim: int) -> LidarTokenizer:
        """Reuse codec weights with square diffusion patches matching the model adapter."""
        patch = isqrt(max(0, patch_dim) // self.geometry.channels)
        if patch < 1 or self.geometry.channels * patch**2 != patch_dim:
            raise ValueError(
                f"Width {patch_dim} cannot form square LiDAR diffusion patches "
                f"with {self.geometry.channels} latent channels"
            )
        return LidarTokenizer(self.backend, replace(self.geometry, patch=patch))

    def to_model_canvas(self, ranges: Tensor) -> Tensor:
        """Wrap semantic azimuth rays onto the tokenizer's padded model canvas."""
        projection = getattr(self.backend, "range_projection", None)
        if projection is None:
            return ranges
        if tuple(ranges.shape[-2:]) != (
            projection.semantic_height,
            projection.semantic_width,
        ):
            raise ValueError("LiDAR ranges must use the tokenizer's semantic ray grid")
        return circular_pad_range_video_3ch(ranges, model_width=projection.model_width)

    def to_semantic_canvas(self, ranges: Tensor) -> Tensor:
        """Remove model padding while preserving the calibrated semantic ray grid."""
        projection = getattr(self.backend, "range_projection", None)
        if projection is None:
            return ranges
        if tuple(ranges.shape[-2:]) != (
            projection.semantic_height,
            projection.model_width,
        ):
            raise ValueError("Decoded LiDAR ranges disagree with the model canvas")
        return crop_range_video_width(ranges, semantic_width=projection.semantic_width)

    def patchify(self, latent: Tensor) -> Tensor:
        """Convert ``[1,C,T,H,W]`` to ``[T*grid_h*grid_w,C*p*p]`` with zero padding."""
        g = self.geometry
        if (
            latent.ndim != 5
            or latent.shape[0] != 1
            or latent.shape[1] != g.channels
            or tuple(latent.shape[-2:]) != (g.height, g.width)
        ):
            raise ValueError(
                f"Expected [1,{g.channels},T,{g.height},{g.width}], got {tuple(latent.shape)}"
            )
        h, w = g.grid
        x = F.pad(latent, (0, w * g.patch - g.width, 0, h * g.patch - g.height))
        return patchify_latent(x[0], g.patch)

    def unpatchify(self, tokens: Tensor) -> Tensor:
        """Restore ``[1,C,T,H,W]`` and remove only the spatial patch padding."""
        g = self.geometry
        if (
            tokens.ndim != 2
            or tokens.shape[1] != g.patch_dim
            or tokens.shape[0] % g.spatial_tokens
        ):
            raise ValueError("LiDAR tokens disagree with the tokenizer geometry")
        h, w = g.grid
        x = unpatchify_latent(
            tokens,
            channels=g.channels,
            grid_t=len(tokens) // g.spatial_tokens,
            grid_h=h,
            grid_w=w,
            patch_spatial=g.patch,
        )
        return x[None, ..., : g.height, : g.width].contiguous()


@contextmanager
def _tf32(enabled: bool):
    """Allow TF32 matmuls/convolutions for one codec call, then restore.

    The codec is FP32; TF32 roughly halves its GB300 time. Scoped rather than
    global so the BF16 model and anything else FP32 keep their precision.
    """
    if not enabled:
        yield
        return
    matmul, cudnn = (
        torch.backends.cuda.matmul.allow_tf32,
        torch.backends.cudnn.allow_tf32,
    )
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = matmul
        torch.backends.cudnn.allow_tf32 = cudnn


class LidarEncoder:
    """One causal range-image stream, with a bounded nine-sweep temporal cache."""

    def __init__(
        self, tokenizer: LidarTokenizer, *, context_frames: int = 9, tf32: bool = False
    ):
        if context_frames < 1:
            raise ValueError("context_frames must be positive")
        self.tokenizer = tokenizer
        self.context_frames = context_frames
        self.tf32 = tf32
        self.cache: Any = None

    def reset(self) -> None:
        self.cache = None

    @torch.inference_mode()
    def __call__(self, ranges: Tensor) -> Tensor:
        """Encode metric range/unit intensity/validity, ``[1,3,T,H,W]``."""
        b = self.tokenizer.backend
        if (
            ranges.ndim != 5
            or ranges.shape[0] != 1
            or ranges.shape[1] != 3
            or not 0 < ranges.shape[2] <= self.context_frames
        ):
            raise ValueError("Expected [1,3,T,H,W], with 1 <= T <= context_frames")
        x, _ = metric_lidar_to_network(
            self.tokenizer.to_model_canvas(ranges.to(device=b.device, dtype=b.dtype)),
            min_range=b.min_range,
            max_range=b.max_range,
        )
        cache = b.model._limit_streaming_cache(
            self.cache, self.context_frames, new_frames=x.shape[2]
        )
        with _tf32(self.tf32):
            _, (mean, _), self.cache = b.model.encode_stream(x, cache)
        return (mean.to(b.dtype) - b.latent_mean) / b.latent_std


class LidarDecoder:
    """One causal latent stream, sharing tokenizer weights with independent state."""

    def __init__(
        self, tokenizer: LidarTokenizer, *, context_frames: int = 9, tf32: bool = False
    ):
        if context_frames < 1:
            raise ValueError("context_frames must be positive")
        self.tokenizer = tokenizer
        self.context_frames = context_frames
        self.tf32 = tf32
        self.cache: Any = None
        self._range_bounds: tuple[Tensor, Tensor] | None = None

    def reset(self) -> None:
        self.cache = None

    @torch.inference_mode()
    def __call__(self, latent: Tensor) -> Tensor:
        """Decode normalized ``[1,C,T,H,W]`` to metric range/intensity/validity."""
        b = self.tokenizer.backend
        g = self.tokenizer.geometry
        if (
            latent.ndim != 5
            or latent.shape[0] != 1
            or latent.shape[1] != g.channels
            or tuple(latent.shape[-2:]) != (g.height, g.width)
            or not 0 < latent.shape[2] <= self.context_frames
        ):
            raise ValueError(
                "LiDAR decode shape or chunk length disagrees with the tokenizer"
            )
        z = latent.to(device=b.device, dtype=b.dtype) * b.latent_std + b.latent_mean
        cache = b.model._limit_streaming_cache(
            self.cache, self.context_frames, new_frames=z.shape[2]
        )
        with _tf32(self.tf32):
            output, self.cache = b.model.decode_stream(z, cache)
        if self._range_bounds is None:
            # Resident once, so the metric conversion is not handed Python floats
            # to copy to the device -- a synchronizing copy -- on every chunk.
            self._range_bounds = (
                torch.as_tensor(b.min_range, dtype=b.dtype, device=b.device),
                torch.as_tensor(b.max_range, dtype=b.dtype, device=b.device),
            )
        return self.tokenizer.to_semantic_canvas(
            network_lidar_to_metric_clip(
                output,
                output[:, 2:3].sigmoid(),
                min_range=self._range_bounds[0],
                max_range=self._range_bounds[1],
                apply_validity_mask=True,
                validity_threshold=b.validity_threshold,
            )
        )
