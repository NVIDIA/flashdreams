# SPDX-FileCopyrightText: Copyright 2025 The MiniMax authors and The HuggingFace Team. All rights reserved.
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

"""MiniMax H3 reference-audio encoder without decoder or posterior wrappers."""

# Adapted from Diffusers' MiniMax H3 audio autoencoder: encoder-only modules
# with native checkpoint names and shared FlashDreams causal attention.

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, fields
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.utils import weight_norm

from flashdreams.accelerated.multi_head_attention.sdpa import (
    scaled_dot_product_attention,
)


@dataclass(frozen=True, kw_only=True)
class AudioEncoderConfig:
    """Encoder-only checkpoint geometry and per-channel latent normalization."""

    encoder_dim: int = 64
    """Initial waveform-encoder feature width."""
    encoder_rates: tuple[int, ...] = (2, 4, 4, 5, 5)
    """Strides of the channel-doubling encoder blocks."""
    latent_dim: int = 2048
    """Encoder trunk width before the attention projection."""
    latent_channels: int = 32
    """Channels in the raw audio latent domain."""
    num_attention_heads: int = 8
    """Causal projection heads, mean-pooled before latent projection."""
    sampling_rate: int = 32000
    """Reference waveform sample rate."""
    latents_mean: tuple[float, ...] = (0.0,) * 32
    """Raw latent means applied by the conditioning facade."""
    latents_std: tuple[float, ...] = (1.0,) * 32
    """Raw latent standard deviations applied by the conditioning facade."""

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> AudioEncoderConfig:
        """Read a full audio VAE config while excluding decoder-only settings."""
        names = {item.name for item in fields(cls)}
        decoder_names = {
            "decoder_dim",
            "decoder_rates",
            "decoder_kernel_sizes",
            "resblock_kernel_sizes",
            "resblock_dilation_sizes",
        }
        unknown = (
            {key for key in values if not key.startswith("_")} - names - decoder_names
        )
        if unknown:
            raise ValueError(
                f"Unknown audio VAE configuration fields: {sorted(unknown)}"
            )
        return cls(**{key: value for key, value in values.items() if key in names})

    @property
    def hop_length(self) -> int:
        """Return waveform samples per encoded latent."""
        return math.prod(self.encoder_rates)


class _Snake(nn.Module):
    """DAC encoder activation with checkpoint-native per-channel frequency."""

    def __init__(self, channels: int):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, channels, 1))

    def forward(self, x: Tensor) -> Tensor:
        """Add the learned periodic residual to the waveform features."""
        return x + (self.alpha + 1e-9).reciprocal() * torch.sin(self.alpha * x).pow(2)


class _ResidualUnit(nn.Module):
    """Weight-normalized DAC residual unit."""

    def __init__(self, dim: int, dilation: int):
        super().__init__()
        self.block = nn.Sequential(
            _Snake(dim),
            weight_norm(
                nn.Conv1d(dim, dim, 7, dilation=dilation, padding=3 * dilation)
            ),
            _Snake(dim),
            weight_norm(nn.Conv1d(dim, dim, 1)),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Add the residual, center-cropping its shortcut when required."""
        residual = self.block(x)
        pad = (x.shape[-1] - residual.shape[-1]) // 2
        if pad > 0:
            x = x[..., pad:-pad]
        return x + residual


class _EncoderBlock(nn.Module):
    """Three residual units followed by strided channel doubling."""

    def __init__(self, dim: int, stride: int):
        super().__init__()
        self.block = nn.Sequential(
            *[_ResidualUnit(dim // 2, dilation) for dilation in (1, 3, 9)],
            _Snake(dim // 2),
            weight_norm(
                nn.Conv1d(
                    dim // 2,
                    dim,
                    2 * stride,
                    stride=stride,
                    padding=math.ceil(stride / 2),
                )
            ),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class _Encoder(nn.Module):
    """Mono waveform encoder retaining checkpoint-native sequential indices."""

    def __init__(self, config: AudioEncoderConfig):
        super().__init__()
        dim = config.encoder_dim
        blocks: list[nn.Module] = [weight_norm(nn.Conv1d(1, dim, 7, padding=3))]
        for stride in config.encoder_rates:
            dim *= 2
            blocks.append(_EncoderBlock(dim, stride))
        blocks.extend(
            [_Snake(dim), weight_norm(nn.Conv1d(dim, config.latent_dim, 3, padding=1))]
        )
        self.block = nn.Sequential(*blocks)

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class _CausalAttention(nn.Module):
    """Causal attention with head-mean and adaptive feature pooling."""

    def __init__(self, in_dim: int, out_dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = in_dim // num_heads
        self.out_dim = out_dim
        self.qkv = nn.Linear(in_dim, in_dim * 3, bias=False)
        self.q_bias = nn.Parameter(torch.zeros(in_dim))
        self.v_bias = nn.Parameter(torch.zeros(in_dim))
        self.register_buffer("zero_k_bias", torch.zeros(in_dim))
        self.proj = nn.Linear(out_dim, out_dim)

    def forward(self, x: Tensor) -> Tensor:
        """Attend causally, then pool heads and feature width independently."""
        b, length, _ = x.shape
        qkv = F.linear(
            x, self.qkv.weight, torch.cat([self.q_bias, self.zero_k_bias, self.v_bias])
        )
        q, k, v = (
            qkv.reshape(b, length, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 1, 3, 4)
            .unbind(0)
        )
        x = scaled_dot_product_attention(q, k, v, is_causal=True)
        x = F.adaptive_avg_pool1d(x.mean(dim=2), self.out_dim)
        return self.proj(x)


class _GeGLU(nn.Module):
    """Pre-normalized GeGLU projection with checkpoint-native names."""

    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.w0 = nn.Linear(dim, hidden_dim)
        self.w1 = nn.Linear(dim, hidden_dim)
        self.w2 = nn.Linear(hidden_dim, dim)

    def forward(self, x: Tensor) -> Tensor:
        x = self.norm(x)
        return self.w2(F.gelu(self.w0(x), approximate="tanh") * self.w1(x))


class _AttentionProjection(nn.Module):
    """Residual causal-attention projection from trunk to latent width."""

    def __init__(self, in_dim: int, out_dim: int, num_heads: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(in_dim)
        self.attn = _CausalAttention(in_dim, out_dim, num_heads)
        self.proj = nn.Linear(in_dim, out_dim)
        self.norm3 = nn.LayerNorm(in_dim)
        self.norm2 = nn.LayerNorm(out_dim)
        self.mlp = _GeGLU(out_dim, out_dim * 2)

    def forward(self, x: Tensor) -> Tensor:
        x = self.proj(self.norm3(x)) + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class MiniMaxH3AudioEncoder(nn.Module):
    """FP32 reference-audio encoder returning only the consumed posterior mean."""

    def __init__(self, config: AudioEncoderConfig):
        super().__init__()
        if any(
            value <= 0
            for value in (
                config.encoder_dim,
                config.latent_dim,
                config.latent_channels,
                config.num_attention_heads,
                config.sampling_rate,
            )
        ):
            raise ValueError(
                "Audio encoder widths, head count and sampling rate must be positive"
            )
        if (
            len(config.latents_mean) != config.latent_channels
            or len(config.latents_std) != config.latent_channels
        ):
            raise ValueError("Audio latent normalization must match latent_channels")
        if any(not math.isfinite(value) for value in config.latents_mean) or any(
            not math.isfinite(value) or value <= 0 for value in config.latents_std
        ):
            raise ValueError(
                "Audio latent means must be finite and standard deviations positive"
            )
        if (
            config.latent_dim % config.latent_channels
            or config.latent_dim % config.num_attention_heads
        ):
            raise ValueError(
                "Audio trunk width must be divisible by latent channels and attention heads"
            )
        if not config.encoder_rates or any(rate <= 0 for rate in config.encoder_rates):
            raise ValueError("Audio encoder strides must be positive")
        self.config = config
        self.encoder = _Encoder(config)
        self.pre_block = _AttentionProjection(
            config.latent_dim, config.latent_channels, config.num_attention_heads
        )
        self.mean_proj = nn.Conv1d(config.latent_channels, config.latent_channels, 1)

    def encode(self, waveform: Tensor) -> Tensor:
        """Return raw posterior means for mono waveforms shaped ``[B,1,S]``.

        Stereo references occupy two batch items. Right-pad to a whole encoder
        hop; the caller applies the checkpoint's per-channel mean and std.
        """
        if (
            waveform.ndim != 3
            or waveform.shape[1] != 1
            or any(size <= 0 for size in waveform.shape)
        ):
            raise ValueError(
                f"Expected nonempty mono waveform [B,1,S], got {tuple(waveform.shape)}"
            )
        if next(self.parameters()).dtype != torch.float32:
            raise ValueError("H3 audio encoder weights must remain float32")
        with torch.autocast(device_type=waveform.device.type, enabled=False):
            waveform = F.pad(
                waveform.float(), (0, (-waveform.shape[-1]) % self.config.hop_length)
            )
            x = self.encoder(waveform)
            x = self.pre_block(x.transpose(1, 2)).transpose(1, 2)
            return self.mean_proj(x)

    def forward(self, waveform: Tensor) -> Tensor:
        """Return raw audio posterior means."""
        return self.encode(waveform)
