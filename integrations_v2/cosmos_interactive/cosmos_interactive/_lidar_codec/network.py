# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FlashDreams checkpoint and streaming interface for the vLLM-Omni LiDAR VAE."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import einops
import torch
from torch import nn

from .encoding import generate_polar_coords
from .transformer_vae import Decoder, Encoder


class TransformerVAE(nn.Module):
    """Run the symmetric V1 codec with causal attention caches.

    This inference module intentionally excludes training, temporal compression,
    and experimental architecture variants. The public encoder uses the mean.
    """

    coords: torch.Tensor

    def __init__(
        self,
        resolution: Sequence[int],
        depths: Sequence[int],
        num_heads: Sequence[int],
        *,
        z_dim: int = 128,
        base_channels: int = 128,
        patch_size: tuple[int, int] = (2, 2),
        window_size: tuple[int, int] = (5, 45),
        mlp_ratio: float = 3.0,
        max_t: int = 32,
    ) -> None:
        super().__init__()
        self.z_dim = z_dim
        self.resolution = tuple(resolution)
        self.spatial_compression = tuple(p * 2 ** (len(depths) - 1) for p in patch_size)
        self.temporal_compression = 1
        self.supports_temporal_kv_cache = True
        self.register_buffer(
            "coords", generate_polar_coords(*resolution, device=torch.empty(0).device)
        )
        options: dict[str, Any] = dict(
            resolution=resolution,
            z_dim=z_dim,
            base_channels=base_channels,
            patch_size=patch_size,
            window_size=window_size,
            depths=depths,
            num_heads=num_heads,
            dilation=(1,) * len(depths),
            temporal_downsample=(False,) * (len(depths) - 1),
            mlp_ratio=mlp_ratio,
            bottleneck_3d=True,
            bottleneck_3d_max_t=max_t,
            bottleneck_3d_causal_time=True,
            bottleneck_3d_rope=True,
        )
        self.encoder = Encoder(in_channels=3, **options)
        self.quant_conv = nn.Conv2d(z_dim * 2, z_dim * 2, 1)
        self.post_quant_conv = nn.Conv2d(z_dim, z_dim, 1)
        self.decoder = Decoder(out_channels=3, **options)

    def encode_stream(
        self, x_new: torch.Tensor, temporal_kv_cache: dict | None = None
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor], dict]:
        """Return the posterior mean and updated cache for new sweeps."""
        B, _, T, _, _ = x_new.shape
        parameters, cache = self.encoder.forward_stream(
            x_new, self.coords.to(x_new), dict(temporal_kv_cache or {})
        )
        parameters = self.quant_conv(
            einops.rearrange(parameters, "B C T H W -> (B T) C H W")
        )
        parameters = einops.rearrange(parameters, "(B T) C H W -> B C T H W", B=B, T=T)
        mean, logvar = parameters.chunk(2, dim=1)
        mean, logvar = mean.contiguous(), logvar.contiguous().clamp(-30.0, 20.0)
        # Preserve the reference RNG consumption as well as its return contract;
        # LidarEncoder selects the mean, independent of this posterior sample.
        sample = mean + torch.exp(0.5 * logvar) * torch.randn_like(mean)
        return sample, (mean, logvar), cache

    def decode_stream(
        self, z_new: torch.Tensor, temporal_kv_cache: dict | None = None
    ) -> tuple[torch.Tensor, dict]:
        """Decode new latents to range/intensity values and validity logits."""
        B, _, T, _, _ = z_new.shape
        z_new = self.post_quant_conv(
            einops.rearrange(z_new, "B C T H W -> (B T) C H W")
        )
        z_new = einops.rearrange(z_new, "(B T) C H W -> B C T H W", B=B, T=T)
        return self.decoder(
            z_new,
            self.coords.to(z_new),
            temporal_kv_cache=dict(temporal_kv_cache or {}),
            return_temporal_kv_cache=True,
        )

    @staticmethod
    def _limit_streaming_cache(
        cache: dict | None, context_frames: int | None, *, new_frames: int
    ) -> dict | None:
        """Reserve room for the new chunk within the total sweep context."""
        if cache is None or context_frames is None:
            return cache
        keep_past = context_frames - new_frames
        if keep_past <= 0:
            return None
        return {
            key: (k[:, :, -keep_past:], v[:, :, -keep_past:])
            for key, (k, v) in cache.items()
        }
