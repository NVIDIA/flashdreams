# SPDX-FileCopyrightText: Copyright 2026 The MiniMax and HuggingFace Teams. All rights reserved.
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

"""MiniMax H3 video codec with independent encoder and decoder loading."""

# Adapted from Diffusers' MiniMax H3 autoencoder: native checkpoint modules,
# FlashDreams attention, and independently constructed inference-only halves.

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, fields
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from flashdreams.accelerated.multi_head_attention import (
    AttentionConfig,
    AttentionType,
    RoPEConfig,
    RoPEStyle,
)
from flashdreams.accelerated.multi_head_attention.torch import TorchMultiHeadAttention


@dataclass(frozen=True, kw_only=True)
class VideoVAEConfig:
    """Checkpoint geometry and normalization for the H3 video codec."""

    in_channels: int = 3
    """Input pixel channels."""
    out_channels: int = 3
    """Decoded pixel channels."""
    latent_channels: int = 24
    """Channels in the unnormalized latent domain."""
    block_out_channels: tuple[int, ...] = (128, 256, 256, 512, 512, 1024)
    """Encoder width at each residual stage."""
    layers_per_block: int = 2
    """Residual blocks in each encoder stage."""
    spatial_downsample_factors: tuple[int, ...] = (2, 2, 2, 2, 1, 1)
    """Spatial stride of each encoder stage."""
    temporal_downsample_factors: tuple[int, ...] = (1, 2, 2, 1, 1, 1)
    """Temporal stride of each encoder stage."""
    norm_num_groups: int = 32
    """Frame-isolated encoder normalization groups."""
    norm_eps: float = 1e-6
    """Encoder group-normalization epsilon."""
    spatial_padding_mode: str = "reflect"
    """Spatial convolution padding; temporal padding always uses zeros."""
    decoder_num_layers: int = 36
    """Number of noncausal decoder transformer blocks."""
    decoder_num_attention_heads: int = 32
    """Heads per decoder self-attention layer."""
    decoder_attention_head_dim: int = 64
    """Feature width of each decoder attention head."""
    decoder_num_register_tokens: int = 4
    """Learned tokens appended before the all-zero auxiliary token."""
    decoder_ffn_mult: int = 4
    """Decoder feed-forward expansion factor."""
    decoder_rope_theta: float = 100.0
    """Frequency base for normalized-coordinate rotary embeddings."""
    decoder_rope_dim_ratio: float = 0.75
    """Fraction of each attention head rotated, leaving the remainder intact."""
    decoder_norm_eps: float = 1e-5
    """Decoder RMS- and layer-normalization epsilon."""
    clip_length: int = 17
    """Pixel frames independently encoded in each temporal clip."""
    token_drop: int = 3
    """Trailing latent frames removed from the concatenated encoder output."""
    latents_mean: tuple[float, ...] = (0.0,) * 24
    """Raw latent means, applied by the conditioning and decode facade."""
    latents_std: tuple[float, ...] = (1.0,) * 24
    """Raw latent standard deviations, applied by the facade."""

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> VideoVAEConfig:
        """Read checkpoint metadata, rejecting unknown architecture fields."""
        names = {item.name for item in fields(cls)}
        unknown = {key for key in values if not key.startswith("_")} - names
        if unknown:
            raise ValueError(
                f"Unknown video VAE configuration fields: {sorted(unknown)}"
            )
        return cls(**{key: value for key, value in values.items() if key in names})

    @property
    def spatial_compression_ratio(self) -> int:
        """Return the product of spatial encoder strides."""
        return math.prod(self.spatial_downsample_factors)

    @property
    def temporal_compression_ratio(self) -> int:
        """Return the convolutional stride, not the overlapped clip mapping."""
        return math.prod(self.temporal_downsample_factors)


class _CausalConv3d(nn.Conv3d):
    """Convolution with spatial reflect padding and causal temporal zero padding."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int | tuple[int, int, int] = 1,
        spatial_padding: int = 0,
        temporal_padding: int = 0,
        spatial_padding_mode: str = "reflect",
    ) -> None:
        super().__init__(in_channels, out_channels, kernel_size, stride, padding=0)
        self.spatial_padding = spatial_padding
        self.temporal_padding = temporal_padding
        self.spatial_padding_mode = spatial_padding_mode

    def forward(self, x: Tensor) -> Tensor:
        """Pad the two domains separately and convolve."""
        if self.spatial_padding:
            p = self.spatial_padding
            x = F.pad(x, (p, p, p, p, 0, 0), mode=self.spatial_padding_mode)
        if self.temporal_padding:
            x = F.pad(x, (0, 0, 0, 0, self.temporal_padding, 0))
        return super().forward(x)


class _FrameGroupNorm(nn.GroupNorm):
    """Group normalization with statistics isolated to each pixel frame."""

    def forward(self, x: Tensor) -> Tensor:
        """Fold time into the batch before group normalization."""
        b, c, t, h, w = x.shape
        x = x.permute(0, 2, 1, 3, 4).contiguous().view(b * t, c, 1, h, w)
        x = super().forward(x)
        return x.view(b, t, c, h, w).permute(0, 2, 1, 3, 4).contiguous()


class _ResnetBlock(nn.Module):
    """Two causal convolutions with frame-isolated group normalization."""

    def __init__(self, in_channels: int, out_channels: int, config: VideoVAEConfig):
        super().__init__()
        self.norm1 = _FrameGroupNorm(
            config.norm_num_groups, in_channels, eps=config.norm_eps
        )
        self.norm2 = _FrameGroupNorm(
            config.norm_num_groups, out_channels, eps=config.norm_eps
        )
        self.conv1 = _CausalConv3d(
            in_channels,
            out_channels,
            3,
            spatial_padding=1,
            temporal_padding=2,
            spatial_padding_mode=config.spatial_padding_mode,
        )
        self.conv2 = _CausalConv3d(
            out_channels,
            out_channels,
            3,
            spatial_padding=1,
            temporal_padding=2,
            spatial_padding_mode=config.spatial_padding_mode,
        )
        self.conv_shortcut = (
            _CausalConv3d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else None
        )

    def forward(self, x: Tensor) -> Tensor:
        """Add the projected shortcut to the two-convolution residual."""
        residual = x if self.conv_shortcut is None else self.conv_shortcut(x)
        return residual + self.conv2(
            F.silu(self.norm2(self.conv1(F.silu(self.norm1(x)))))
        )


class _Downsample(nn.Module):
    """Causal strided convolution with asymmetric spatial padding."""

    def __init__(
        self,
        channels: int,
        temporal_stride: int,
        spatial_stride: int,
        config: VideoVAEConfig,
    ):
        super().__init__()
        self.spatial_stride = spatial_stride
        self.spatial_padding_mode = config.spatial_padding_mode
        self.conv = _CausalConv3d(
            channels,
            channels,
            3,
            stride=(temporal_stride, spatial_stride, spatial_stride),
            temporal_padding=2,
            spatial_padding_mode=config.spatial_padding_mode,
        )

    def forward(self, x: Tensor) -> Tensor:
        """Pad bottom and right before each spatial downsample."""
        if self.spatial_stride == 2:
            x = F.pad(x, (0, 1, 0, 1, 0, 0), mode=self.spatial_padding_mode)
        return self.conv(x)


class _DownBlock(nn.Module):
    """Encoder residual stack and optional downsampling operation."""

    def __init__(
        self, in_channels: int, out_channels: int, index: int, config: VideoVAEConfig
    ):
        super().__init__()
        self.resnets = nn.ModuleList(
            [
                _ResnetBlock(
                    in_channels if i == 0 else out_channels, out_channels, config
                )
                for i in range(config.layers_per_block)
            ]
        )
        spatial = config.spatial_downsample_factors[index]
        temporal = config.temporal_downsample_factors[index]
        self.downsamplers = (
            nn.ModuleList([_Downsample(out_channels, temporal, spatial, config)])
            if spatial * temporal > 1
            else None
        )

    def forward(self, x: Tensor) -> Tensor:
        """Run residuals followed by the optional downsample."""
        for block in self.resnets:
            x = block(x)
        if self.downsamplers is not None:
            for downsample in self.downsamplers:
                x = downsample(x)
        return x


class _Encoder(nn.Module):
    """Causal CNN producing unnormalized Gaussian moments."""

    def __init__(self, config: VideoVAEConfig):
        super().__init__()
        channels = config.block_out_channels
        self.conv_in = _CausalConv3d(
            config.in_channels,
            channels[0],
            3,
            spatial_padding=1,
            temporal_padding=2,
            spatial_padding_mode=config.spatial_padding_mode,
        )
        self.down_blocks = nn.ModuleList(
            [
                _DownBlock(channels[max(i - 1, 0)], out_channels, i, config)
                for i, out_channels in enumerate(channels)
            ]
        )
        self.norm_out = _FrameGroupNorm(
            config.norm_num_groups, channels[-1], eps=config.norm_eps
        )
        self.conv_out = _CausalConv3d(
            channels[-1],
            2 * config.latent_channels,
            3,
            spatial_padding=1,
            temporal_padding=2,
            spatial_padding_mode=config.spatial_padding_mode,
        )

    def forward(self, x: Tensor) -> Tensor:
        """Encode pixels without latent sampling or normalization."""
        x = self.conv_in(x)
        for block in self.down_blocks:
            x = block(x)
        return self.conv_out(F.silu(self.norm_out(x)))


class _FloatRMSNorm(nn.RMSNorm):
    """RMS normalization in FP32 with activation-precision output."""

    def forward(self, x: Tensor) -> Tensor:
        """Normalize in FP32 and restore the input dtype."""
        return super().forward(x.float()).to(x.dtype)


class _VideoAttention(TorchMultiHeadAttention):
    """Checkpoint-named projections using shared cacheless attention and RoPE."""

    def __init__(self, dim: int, heads: int, head_dim: int, eps: float):
        super().__init__(
            AttentionType.SELF_ATTENTION,
            AttentionConfig(
                query_dim=dim,
                n_heads=heads,
                head_dim=head_dim,
                qk_norm_eps=eps,
                rope_config=RoPEConfig(style=RoPEStyle.SPLIT),
            ),
        )
        self.to_q = nn.Linear(dim, heads * head_dim)
        self.to_k = nn.Linear(dim, heads * head_dim)
        self.to_v = nn.Linear(dim, heads * head_dim)
        self.to_out = nn.ModuleList([nn.Linear(heads * head_dim, dim)])
        self.norm_q = _FloatRMSNorm(head_dim, eps=eps, elementwise_affine=False)
        self.norm_k = _FloatRMSNorm(head_dim, eps=eps, elementwise_affine=False)

    @property
    def query_projection(self) -> nn.Linear:
        return self.to_q

    @property
    def key_projection(self) -> nn.Linear:
        return self.to_k

    @property
    def value_projection(self) -> nn.Linear:
        return self.to_v

    @property
    def output_projection(self) -> nn.Linear:
        return self.to_out[0]

    @property
    def query_norm(self) -> nn.Module:
        return self.norm_q

    @property
    def key_norm(self) -> nn.Module:
        return self.norm_k


class _SwiGLU(nn.Module):
    """Fused gate projection retaining the checkpoint's parameter names."""

    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.proj = nn.Linear(dim, hidden_dim * 2)

    def forward(self, x: Tensor) -> Tensor:
        """Multiply the value half by the SiLU gate half."""
        value, gate = self.proj(x).chunk(2, dim=-1)
        return value * F.silu(gate)


class _FeedForward(nn.Module):
    """SwiGLU feed-forward network with checkpoint-native sequential indices."""

    def __init__(self, dim: int, mult: int):
        super().__init__()
        self.net = nn.Sequential(
            _SwiGLU(dim, dim * mult), nn.Identity(), nn.Linear(dim * mult, dim)
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class _TransformerBlock(nn.Module):
    """Decoder transformer with learned per-channel residual scales."""

    def __init__(self, config: VideoVAEConfig):
        super().__init__()
        dim = config.decoder_num_attention_heads * config.decoder_attention_head_dim
        eps = config.decoder_norm_eps
        self.norm1 = _FloatRMSNorm(dim, eps=eps)
        self.attn = _VideoAttention(
            dim,
            config.decoder_num_attention_heads,
            config.decoder_attention_head_dim,
            eps,
        )
        self.scale1 = nn.Parameter(torch.zeros(dim))
        self.norm2 = _FloatRMSNorm(dim, eps=eps)
        self.ff = _FeedForward(dim, config.decoder_ffn_mult)
        self.scale2 = nn.Parameter(torch.zeros(dim))

    def forward(self, x: Tensor, rope_freqs: Tensor) -> Tensor:
        """Apply attention and feed-forward residuals with FP32 normalization."""
        x = x + self.attn(self.norm1(x), rope_freqs=rope_freqs) * self.scale1
        return x + self.ff(self.norm2(x)) * self.scale2


class _ViTDecoder(nn.Module):
    """Noncausal decoder with register tokens and normalized-coordinate RoPE."""

    def __init__(self, config: VideoVAEConfig):
        super().__init__()
        self.config = config
        dim = config.decoder_num_attention_heads * config.decoder_attention_head_dim
        rotary_dim = int(
            config.decoder_attention_head_dim * config.decoder_rope_dim_ratio
        )
        if rotary_dim <= 0 or rotary_dim % 6:
            raise ValueError(
                "Video decoder rotary dimension must be positive and divisible by 6"
            )
        self.register_buffer(
            "inv_freq",
            1.0
            / config.decoder_rope_theta
            ** torch.arange(0, 1, 6 / rotary_dim, dtype=torch.float32),
            persistent=False,
        )
        self.proj_in = nn.Linear(config.latent_channels, dim)
        self.register_tokens = nn.Parameter(
            torch.zeros(1, config.decoder_num_register_tokens, dim)
        )
        self.transformer_blocks = nn.ModuleList(
            [_TransformerBlock(config) for _ in range(config.decoder_num_layers)]
        )
        self.norm_out = nn.LayerNorm(dim, eps=config.decoder_norm_eps)
        self.proj_out = nn.Linear(
            dim,
            config.out_channels
            * config.temporal_compression_ratio
            * config.spatial_compression_ratio**2,
        )

    def forward(self, z: Tensor) -> Tensor:
        """Decode one tile into ImageNet-normalized pixel patches."""
        b, c, t, h, w = z.shape
        x = self.proj_in(z.permute(0, 2, 3, 4, 1).reshape(b, t * h * w, c))
        num_patches = x.shape[1]
        x = torch.cat(
            [x, self.register_tokens.expand(b, -1, -1), torch.zeros_like(x[:, :1])],
            dim=1,
        )
        grids = [
            2 * (torch.arange(0.5, size, device=z.device, dtype=torch.float32) / size)
            - 1
            for size in (t, h, w)
        ]
        positions = torch.stack(torch.meshgrid(*grids, indexing="ij"), dim=-1).flatten(
            0, 2
        )
        positions = torch.cat(
            [
                positions,
                positions.new_zeros((self.config.decoder_num_register_tokens + 1, 3)),
            ]
        )
        angles = 2.0 * math.pi * positions[:, :, None] * self.inv_freq[None, None, :]
        rope_freqs = angles.flatten(1, 2).tile(2)[:, None, None, :]
        for block in self.transformer_blocks:
            x = block(x, rope_freqs)
        x = self.proj_out(self.norm_out(x))[:, :num_patches]
        spatial, temporal = (
            self.config.spatial_compression_ratio,
            self.config.temporal_compression_ratio,
        )
        x = x.view(b, t, h, w, self.config.out_channels, temporal, spatial, spatial)
        return (
            x.permute(0, 4, 1, 5, 2, 6, 3, 7)
            .contiguous()
            .reshape(
                b, self.config.out_channels, t * temporal, h * spatial, w * spatial
            )
        )


class _TiledCodec(nn.Module):
    """Model-owned spatial tiling shared by the encoder and decoder halves."""

    def __init__(self, config: VideoVAEConfig):
        super().__init__()
        count = len(config.block_out_channels)
        if (
            not count
            or len(config.spatial_downsample_factors) != count
            or len(config.temporal_downsample_factors) != count
        ):
            raise ValueError(
                "Video VAE encoder widths and stride lists must have equal nonzero lengths"
            )
        if any(
            value <= 0
            for value in (
                config.in_channels,
                config.out_channels,
                config.latent_channels,
                config.layers_per_block,
                config.norm_num_groups,
                config.clip_length,
            )
        ):
            raise ValueError(
                "Video VAE channel counts, layer counts, groups and clip length must be positive"
            )
        if any(
            width <= 0 or width % config.norm_num_groups
            for width in config.block_out_channels
        ):
            raise ValueError(
                "Video VAE encoder widths must be positive multiples of norm_num_groups"
            )
        if any(
            stride not in (1, 2)
            for stride in (
                *config.spatial_downsample_factors,
                *config.temporal_downsample_factors,
            )
        ):
            raise ValueError("Video VAE convolutional strides must be 1 or 2")
        if (
            not 0
            <= config.token_drop
            < math.ceil(config.clip_length / config.temporal_compression_ratio)
        ):
            raise ValueError("Video VAE token_drop must fit within one latent clip")
        if (
            len(config.latents_mean) != config.latent_channels
            or len(config.latents_std) != config.latent_channels
        ):
            raise ValueError(
                "Video VAE latent normalization must match latent_channels"
            )
        if any(not math.isfinite(value) for value in config.latents_mean) or any(
            not math.isfinite(value) or value <= 0 for value in config.latents_std
        ):
            raise ValueError(
                "Video VAE latent means must be finite and standard deviations positive"
            )
        self.config = config
        self.use_tiling = True
        self.tile_sample_min_height = 256
        self.tile_sample_min_width = 256
        self.tile_sample_min_overlap_height = 64
        self.tile_sample_min_overlap_width = 64

    @property
    def spatial_compression_ratio(self) -> int:
        return self.config.spatial_compression_ratio

    @property
    def temporal_compression_ratio(self) -> int:
        return self.config.temporal_compression_ratio

    def _split_tiles(
        self, length: int, tile_size: int, min_overlap: int
    ) -> tuple[list[int], list[int], list[int]]:
        if tile_size <= 0 or not 0 <= min_overlap < tile_size:
            raise ValueError(
                "Tile overlap must be nonnegative and smaller than tile size"
            )
        if any(
            value % self.spatial_compression_ratio
            for value in (length, tile_size, min_overlap)
        ):
            raise ValueError(
                "Video dimensions and tile geometry must align to the spatial compression ratio"
            )
        if tile_size >= length:
            return [0], [length], []
        num_tiles = math.ceil(length / tile_size)
        while tile_size * num_tiles - min_overlap * (num_tiles - 1) < length:
            num_tiles += 1
        overlaps = [min_overlap] * (num_tiles - 1)
        remaining = tile_size * num_tiles - sum(overlaps) - length
        for i in range(remaining // self.spatial_compression_ratio):
            overlaps[i % (num_tiles - 1)] += self.spatial_compression_ratio
        starts = [0]
        for overlap in overlaps:
            starts.append(starts[-1] + tile_size - overlap)
        return starts, [tile_size] * num_tiles, overlaps

    @staticmethod
    def _blend(a: Tensor, b: Tensor, blend_extent: int, dim: int) -> Tensor:
        extent = min(a.shape[dim], b.shape[dim], blend_extent)
        if extent == 0:
            return b
        positions = torch.arange(extent, device=b.device, dtype=b.dtype)
        shape = [1] * a.ndim
        shape[dim] = extent
        weight_a, weight_b = (
            (1 - positions / extent).view(shape),
            (positions / extent).view(shape),
        )
        blended = (
            a.narrow(dim, a.shape[dim] - extent, extent) * weight_a
            + b.narrow(dim, 0, extent) * weight_b
        )
        return torch.cat(
            [blended, b.narrow(dim, extent, b.shape[dim] - extent)], dim=dim
        )

    def _stitch_tiles(
        self,
        tiles: list[list[Tensor]],
        height_overlaps: list[int],
        width_overlaps: list[int],
    ) -> Tensor:
        result_rows = []
        for i, row in enumerate(tiles):
            result_row = []
            for j, tile in enumerate(row):
                if i > 0:
                    tile = self._blend(
                        tiles[i - 1][j], tile, height_overlaps[i - 1], -2
                    )
                if j > 0:
                    tile = self._blend(row[j - 1], tile, width_overlaps[j - 1], -1)
                if i < len(tiles) - 1 and height_overlaps[i]:
                    tile = tile[..., : -height_overlaps[i], :]
                if j < len(row) - 1 and width_overlaps[j]:
                    tile = tile[..., :, : -width_overlaps[j]]
                result_row.append(tile)
            result_rows.append(torch.cat(result_row, dim=-1))
        return torch.cat(result_rows, dim=-2)

    def _validate_video(self, x: Tensor, channels: int) -> None:
        if x.ndim != 5 or x.shape[1] != channels or any(size <= 0 for size in x.shape):
            raise ValueError(
                f"Expected nonempty [B,{channels},T,H,W] video, got {tuple(x.shape)}"
            )
        if next(self.parameters()).dtype != torch.float32:
            raise ValueError("H3 video VAE weights must remain float32")


class MiniMaxH3VideoEncoder(_TiledCodec):
    """Independently loadable video encoder with checkpoint-native keys."""

    def __init__(self, config: VideoVAEConfig):
        super().__init__(config)
        self.encoder = _Encoder(config)
        self.quant_conv = nn.Conv3d(
            2 * config.latent_channels, 2 * config.latent_channels, 1
        )

    def _encode_clip(self, x: Tensor) -> Tensor:
        if not self.use_tiling:
            return self.quant_conv(self.encoder(x))
        ys, heights, yo = self._split_tiles(
            x.shape[-2],
            self.tile_sample_min_height,
            self.tile_sample_min_overlap_height,
        )
        xs, widths, xo = self._split_tiles(
            x.shape[-1], self.tile_sample_min_width, self.tile_sample_min_overlap_width
        )
        rows = [
            [
                self.quant_conv(self.encoder(x[..., y : y + h, s : s + w]))
                for s, w in zip(xs, widths)
            ]
            for y, h in zip(ys, heights)
        ]
        ratio = self.spatial_compression_ratio
        return self._stitch_tiles(
            rows, [n // ratio for n in yo], [n // ratio for n in xo]
        )

    def encode(self, pixels: Tensor) -> Tensor:
        """Encode ImageNet-normalized pixels into raw mean/log-variance moments.

        Args:
            pixels: FP32 pixels in ``[B,3,T,H,W]`` layout.

        Returns:
            Unnormalized moments shaped ``[B,2*latent_channels,Tl,Hl,Wl]``.
        """
        self._validate_video(pixels, self.config.in_channels)
        if any(size % self.spatial_compression_ratio for size in pixels.shape[-2:]):
            raise ValueError(
                "Video encoder pixels must align to the spatial compression ratio"
            )
        with torch.autocast(device_type=pixels.device.type, enabled=False):
            pixels = pixels.float()
            if pixels.shape[2] == 1:
                return self._encode_clip(pixels)
            clip = self.config.clip_length
            pad = (-pixels.shape[2]) % clip
            if pad:
                pixels = torch.cat(
                    [pixels, pixels[:, :, -1:].repeat(1, 1, pad, 1, 1)], dim=2
                )
            moments = torch.cat(
                [self._encode_clip(chunk) for chunk in pixels.split(clip, dim=2)], dim=2
            )
            if self.config.token_drop:
                moments = moments[:, :, : -self.config.token_drop]
            return moments

    def sample(self, pixels: Tensor, *, generator: torch.Generator) -> Tensor:
        """Sample raw latents using the supplied generator's device and sequence."""
        mean, logvar = self.encode(pixels).chunk(2, dim=1)
        std = torch.exp(0.5 * logvar.clamp(-30.0, 20.0))
        noise = torch.randn(
            mean.shape, generator=generator, device=generator.device, dtype=mean.dtype
        ).to(mean.device)
        return mean + std * noise

    def forward(self, pixels: Tensor) -> Tensor:
        """Return raw encoder moments."""
        return self.encode(pixels)


class MiniMaxH3VideoDecoder(_TiledCodec):
    """Independently loadable ViT decoder preserving spatial and temporal blends."""

    def __init__(self, config: VideoVAEConfig):
        super().__init__(config)
        self.post_quant_conv = nn.Conv3d(
            config.latent_channels, config.latent_channels, 1
        )
        self.decoder = _ViTDecoder(config)

    def _decode_clip(self, z: Tensor) -> Tensor:
        if not self.use_tiling:
            return self.decoder(self.post_quant_conv(z))
        ratio = self.spatial_compression_ratio
        ys, heights, yo = self._split_tiles(
            z.shape[-2] * ratio,
            self.tile_sample_min_height,
            self.tile_sample_min_overlap_height,
        )
        xs, widths, xo = self._split_tiles(
            z.shape[-1] * ratio,
            self.tile_sample_min_width,
            self.tile_sample_min_overlap_width,
        )
        rows = [
            [
                self.decoder(
                    self.post_quant_conv(
                        z[
                            ...,
                            y // ratio : (y + h) // ratio,
                            x // ratio : (x + w) // ratio,
                        ]
                    )
                )
                for x, w in zip(xs, widths)
            ]
            for y, h in zip(ys, heights)
        ]
        return self._stitch_tiles(rows, yo, xo)

    def _decode(self, z: Tensor) -> Tensor:
        ratio = self.temporal_compression_ratio
        clip, drop = self.config.clip_length, self.config.token_drop
        chunk_size = math.ceil(clip / ratio)
        token_overlap = (-drop) % chunk_size
        frame_pre_padding = (-clip) % ratio
        frame_overlap = max(token_overlap * ratio - frame_pre_padding, 0)
        chunk_frames = chunk_size * ratio
        num_tokens = z.shape[2] + drop
        pad_tokens = (-num_tokens) % chunk_size
        num_chunks = (num_tokens + pad_tokens) // chunk_size - int(drop > 0)
        if num_chunks <= 0:
            raise ValueError(
                "Too few video latent frames for the configured temporal decoder"
            )
        if pad_tokens:
            z = torch.cat([z, z[:, :, -1:].repeat(1, 1, pad_tokens, 1, 1)], dim=2)
        chunks, overlap = [], None
        for i in range(num_chunks):
            start = i * chunk_size
            decoded = self._decode_clip(
                z[:, :, start : start + chunk_size + token_overlap]
            )
            for j in range(int(drop > 0) + 1):
                chunk = decoded[:, :, j * chunk_frames : (j + 1) * chunk_frames]
                chunk = chunk[:, :, frame_pre_padding:]
                if j == 0:
                    if overlap is not None:
                        chunk = self._blend(overlap, chunk, frame_overlap, -3)
                    chunks.append(chunk)
                else:
                    overlap = chunk
        if overlap is not None:
            chunks.append(overlap)
        video = torch.cat(chunks, dim=2)
        if pad_tokens:
            intra_tail = clip % ratio
            before_pad = z.shape[2] - pad_tokens
            pad_frames = sum(
                intra_tail
                if intra_tail and (before_pad + k) % chunk_size == 0
                else ratio
                for k in range(pad_tokens)
            )
            video = video[:, :, :-pad_frames]
        return video

    def decode(self, latents: Tensor) -> Tensor:
        """Decode raw denormalized latents into ImageNet-normalized RGB.

        FP32 parameters are mandatory. CUDA uses FP16 autocast; CPU remains FP32.
        Output normalization and conversion to application layout belong to the
        caller, not to the checkpoint model.
        """
        self._validate_video(latents, self.config.latent_channels)
        with torch.autocast(
            device_type=latents.device.type,
            dtype=torch.float16,
            enabled=latents.device.type == "cuda",
        ):
            return self._decode(latents.float())

    def forward(self, latents: Tensor) -> Tensor:
        """Return decoded raw pixels."""
        return self.decode(latents)
