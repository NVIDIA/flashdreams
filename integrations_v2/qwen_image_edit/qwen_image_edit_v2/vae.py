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

"""Native image path through the causal Qwen Image variational autoencoder."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

LATENTS_MEAN = (
    -0.7571,
    -0.7089,
    -0.9113,
    0.1075,
    -0.1745,
    0.9653,
    -0.1517,
    1.5508,
    0.4134,
    -0.0715,
    0.5517,
    -0.3632,
    -0.1922,
    -0.9497,
    0.2503,
    -0.2921,
)
"""Per-channel mean of Qwen Image VAE latents."""

LATENTS_STD = (
    2.8184,
    1.4541,
    2.3275,
    2.6558,
    1.2196,
    1.7708,
    2.6052,
    2.0743,
    3.2687,
    2.1526,
    2.8652,
    1.5579,
    1.6382,
    1.1253,
    2.8251,
    1.9160,
)
"""Per-channel standard deviation of Qwen Image VAE latents."""


class QwenImageCausalConv3d(nn.Conv3d):
    """Apply left-causal temporal padding around a 3D convolution."""

    _padding: tuple[int, int, int, int, int, int]

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int, int],
        stride: int | tuple[int, int, int] = 1,
        padding: int | tuple[int, int, int] = 0,
    ) -> None:
        super().__init__(in_channels, out_channels, kernel_size, stride, padding)
        self._padding = (
            int(self.padding[2]),
            int(self.padding[2]),
            int(self.padding[1]),
            int(self.padding[1]),
            2 * int(self.padding[0]),
            0,
        )
        self.padding = (0, 0, 0)

    def forward(self, hidden_states: Tensor) -> Tensor:
        return super().forward(F.pad(hidden_states, self._padding))


class QwenImageRMS_norm(nn.Module):
    """Apply Qwen's channel-first RMS normalization."""

    def __init__(
        self,
        dim: int,
        *,
        channel_first: bool = True,
        images: bool = True,
        bias: bool = False,
    ) -> None:
        super().__init__()
        broadcast = (1, 1, 1) if not images else (1, 1)
        shape = (dim, *broadcast) if channel_first else (dim,)
        self.channel_first = channel_first
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(shape))
        self.bias = nn.Parameter(torch.zeros(shape)) if bias else 0.0

    def forward(self, hidden_states: Tensor) -> Tensor:
        dimension = 1 if self.channel_first else -1
        normalized = F.normalize(hidden_states.float(), dim=dimension).to(
            hidden_states.dtype
        )
        return normalized * self.scale * self.gamma + self.bias


class QwenImageUpsample(nn.Upsample):
    """Upsample in fp32 and restore the input dtype."""

    def forward(self, hidden_states: Tensor) -> Tensor:
        return super().forward(hidden_states.float()).to(hidden_states.dtype)


class QwenImageResample(nn.Module):
    """Resample Qwen VAE image features while retaining checkpoint topology."""

    def __init__(self, dim: int, mode: str) -> None:
        super().__init__()
        self.dim = dim
        self.mode = mode
        if mode in {"upsample2d", "upsample3d"}:
            self.resample = nn.Sequential(
                QwenImageUpsample(scale_factor=(2.0, 2.0), mode="nearest-exact"),
                nn.Conv2d(dim, dim // 2, 3, padding=1),
            )
            if mode == "upsample3d":
                self.time_conv = QwenImageCausalConv3d(
                    dim, dim * 2, (3, 1, 1), padding=(1, 0, 0)
                )
        elif mode in {"downsample2d", "downsample3d"}:
            self.resample = nn.Sequential(
                nn.ZeroPad2d((0, 1, 0, 1)),
                nn.Conv2d(dim, dim, 3, stride=(2, 2)),
            )
            if mode == "downsample3d":
                self.time_conv = QwenImageCausalConv3d(
                    dim, dim, (3, 1, 1), stride=(2, 1, 1)
                )
        else:
            self.resample = nn.Identity()

    def forward(self, hidden_states: Tensor) -> Tensor:
        batch, channels, frames, height, width = hidden_states.shape
        image = hidden_states.permute(0, 2, 1, 3, 4).reshape(
            batch * frames, channels, height, width
        )
        image = self.resample(image)
        return image.view(batch, frames, *image.shape[1:]).permute(0, 2, 1, 3, 4)


class QwenImageResidualBlock(nn.Module):
    """Apply two causal convolutions with a residual shortcut."""

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm1 = QwenImageRMS_norm(in_dim, images=False)
        self.conv1 = QwenImageCausalConv3d(in_dim, out_dim, 3, padding=1)
        self.norm2 = QwenImageRMS_norm(out_dim, images=False)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = QwenImageCausalConv3d(out_dim, out_dim, 3, padding=1)
        self.conv_shortcut = (
            QwenImageCausalConv3d(in_dim, out_dim, 1)
            if in_dim != out_dim
            else nn.Identity()
        )

    def forward(self, hidden_states: Tensor) -> Tensor:
        residual = self.conv_shortcut(hidden_states)
        hidden_states = self.conv1(F.silu(self.norm1(hidden_states)))
        hidden_states = self.conv2(self.dropout(F.silu(self.norm2(hidden_states))))
        return hidden_states + residual


class QwenImageAttentionBlock(nn.Module):
    """Apply single-head spatial attention independently to each frame."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.norm = QwenImageRMS_norm(dim)
        self.to_qkv = nn.Conv2d(dim, dim * 3, 1)
        self.proj = nn.Conv2d(dim, dim, 1)

    def forward(self, hidden_states: Tensor) -> Tensor:
        residual = hidden_states
        batch, channels, frames, height, width = hidden_states.shape
        image = hidden_states.permute(0, 2, 1, 3, 4).reshape(
            batch * frames, channels, height, width
        )
        qkv = self.to_qkv(self.norm(image)).reshape(batch * frames, 1, channels * 3, -1)
        query, key, value = qkv.permute(0, 1, 3, 2).contiguous().chunk(3, dim=-1)
        image = F.scaled_dot_product_attention(query, key, value)
        image = self.proj(
            image.squeeze(1)
            .permute(0, 2, 1)
            .reshape(batch * frames, channels, height, width)
        )
        image = image.view(batch, frames, channels, height, width).permute(
            0, 2, 1, 3, 4
        )
        return image + residual


class QwenImageMidBlock(nn.Module):
    """Apply the VAE bottleneck residual-attention-residual sequence."""

    def __init__(self, dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.resnets = nn.ModuleList(
            [
                QwenImageResidualBlock(dim, dim, dropout),
                QwenImageResidualBlock(dim, dim, dropout),
            ]
        )
        self.attentions = nn.ModuleList([QwenImageAttentionBlock(dim)])

    def forward(self, hidden_states: Tensor) -> Tensor:
        hidden_states = self.resnets[0](hidden_states)
        return self.resnets[1](self.attentions[0](hidden_states))


class QwenImageEncoder3d(nn.Module):
    """Encode one RGB image into Gaussian latent moments."""

    def __init__(
        self,
        dim: int,
        z_dim: int,
        dim_mult: tuple[int, ...],
        num_res_blocks: int,
        temporal_downsample: tuple[bool, ...],
        dropout: float,
        input_channels: int,
    ) -> None:
        super().__init__()
        dimensions = [dim * value for value in (1, *dim_mult)]
        self.conv_in = QwenImageCausalConv3d(
            input_channels, dimensions[0], 3, padding=1
        )
        self.down_blocks = nn.ModuleList()
        for index, (in_dim, out_dim) in enumerate(zip(dimensions[:-1], dimensions[1:])):
            for _ in range(num_res_blocks):
                self.down_blocks.append(
                    QwenImageResidualBlock(in_dim, out_dim, dropout)
                )
                in_dim = out_dim
            if index != len(dim_mult) - 1:
                mode = "downsample3d" if temporal_downsample[index] else "downsample2d"
                self.down_blocks.append(QwenImageResample(out_dim, mode))
        self.mid_block = QwenImageMidBlock(out_dim, dropout)
        self.norm_out = QwenImageRMS_norm(out_dim, images=False)
        self.conv_out = QwenImageCausalConv3d(out_dim, z_dim, 3, padding=1)

    def forward(self, hidden_states: Tensor) -> Tensor:
        hidden_states = self.conv_in(hidden_states)
        for layer in self.down_blocks:
            hidden_states = layer(hidden_states)
        hidden_states = self.mid_block(hidden_states)
        return self.conv_out(F.silu(self.norm_out(hidden_states)))


class QwenImageUpBlock(nn.Module):
    """Apply residual blocks followed by optional spatial upsampling."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_res_blocks: int,
        dropout: float,
        upsample_mode: str | None,
    ) -> None:
        super().__init__()
        resnets = []
        for _ in range(num_res_blocks + 1):
            resnets.append(QwenImageResidualBlock(in_dim, out_dim, dropout))
            in_dim = out_dim
        self.resnets = nn.ModuleList(resnets)
        self.upsamplers = (
            nn.ModuleList([QwenImageResample(out_dim, upsample_mode)])
            if upsample_mode is not None
            else None
        )

    def forward(self, hidden_states: Tensor) -> Tensor:
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states)
        if self.upsamplers is not None:
            hidden_states = self.upsamplers[0](hidden_states)
        return hidden_states


class QwenImageDecoder3d(nn.Module):
    """Decode one Qwen latent frame into RGB pixels."""

    def __init__(
        self,
        dim: int,
        z_dim: int,
        dim_mult: tuple[int, ...],
        num_res_blocks: int,
        temporal_upsample: tuple[bool, ...],
        dropout: float,
        input_channels: int,
    ) -> None:
        super().__init__()
        dimensions = [dim * value for value in (dim_mult[-1], *dim_mult[::-1])]
        self.conv_in = QwenImageCausalConv3d(z_dim, dimensions[0], 3, padding=1)
        self.mid_block = QwenImageMidBlock(dimensions[0], dropout)
        self.up_blocks = nn.ModuleList()
        for index, (in_dim, out_dim) in enumerate(zip(dimensions[:-1], dimensions[1:])):
            if index > 0:
                in_dim //= 2
            mode = None
            if index != len(dim_mult) - 1:
                mode = "upsample3d" if temporal_upsample[index] else "upsample2d"
            self.up_blocks.append(
                QwenImageUpBlock(in_dim, out_dim, num_res_blocks, dropout, mode)
            )
        self.norm_out = QwenImageRMS_norm(out_dim, images=False)
        self.conv_out = QwenImageCausalConv3d(out_dim, input_channels, 3, padding=1)

    def forward(self, hidden_states: Tensor) -> Tensor:
        hidden_states = self.mid_block(self.conv_in(hidden_states))
        for block in self.up_blocks:
            hidden_states = block(hidden_states)
        return self.conv_out(F.silu(self.norm_out(hidden_states)))


class QwenImageVAE(nn.Module):
    """Qwen Image VAE specialized for independent image frames."""

    latents_mean: Tensor
    latents_std: Tensor

    spatial_compression_ratio = 8
    """Pixel-to-latent spatial compression ratio."""

    def __init__(
        self,
        *,
        base_dim: int = 96,
        z_dim: int = 16,
        dim_mult: tuple[int, ...] = (1, 2, 4, 4),
        num_res_blocks: int = 2,
        temporal_downsample: tuple[bool, ...] = (False, True, True),
        dropout: float = 0.0,
        input_channels: int = 3,
    ) -> None:
        super().__init__()
        self.z_dim = z_dim
        self.encoder = QwenImageEncoder3d(
            base_dim,
            z_dim * 2,
            dim_mult,
            num_res_blocks,
            temporal_downsample,
            dropout,
            input_channels,
        )
        self.quant_conv = QwenImageCausalConv3d(z_dim * 2, z_dim * 2, 1)
        self.post_quant_conv = QwenImageCausalConv3d(z_dim, z_dim, 1)
        self.decoder = QwenImageDecoder3d(
            base_dim,
            z_dim,
            dim_mult,
            num_res_blocks,
            temporal_downsample[::-1],
            dropout,
            input_channels,
        )
        self.register_buffer(
            "latents_mean",
            torch.tensor(LATENTS_MEAN).view(1, z_dim, 1, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "latents_std",
            torch.tensor(LATENTS_STD).view(1, z_dim, 1, 1, 1),
            persistent=False,
        )

    def encode(self, image: Tensor) -> Tensor:
        """Encode ``[B, 3, H, W]`` pixels into normalized mode latents."""
        moments = self.quant_conv(self.encoder(image.unsqueeze(2)))
        mean, _ = moments.chunk(2, dim=1)
        return (mean - self.latents_mean) / self.latents_std

    def decode(self, latents: Tensor) -> Tensor:
        """Decode normalized ``[B, 16, H, W]`` latents into RGB pixels."""
        denormalized = latents.unsqueeze(2) * self.latents_std + self.latents_mean
        decoded = self.decoder(self.post_quant_conv(denormalized))[:, :, 0]
        return decoded.clamp(-1.0, 1.0)


__all__ = ["QwenImageVAE"]
