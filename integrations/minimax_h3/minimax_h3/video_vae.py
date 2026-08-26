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

"""Native MiniMax H3 causal encoder and ViT video decoder.

Modified from the Apache-2.0 H3 video VAE in Hugging Face Diffusers commit
``175fe6b2419a01db9c2ceabd01ec37d2c0305fc2``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import cast

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from flashdreams.core.checkpoint.load import load_checkpoint
from flashdreams.infra.config import InstantiateConfig

H3_VIDEO_VAE_CHECKPOINT = (
    "https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/"
    "42ed227ee7df40d41602854ae760620d6eb651fe/vae/"
    "diffusion_pytorch_model.safetensors.index.json"
)
"""Immutable released MiniMax H3 sharded video VAE index."""

_PIXEL_MEAN = (0.485, 0.456, 0.406)
"""ImageNet RGB mean expected by the H3 video VAE."""

_PIXEL_STD = (0.229, 0.224, 0.225)
"""ImageNet RGB standard deviation expected by the H3 video VAE."""

_VIDEO_LATENTS_MEAN = (
    0.858090341091156,
    -0.9606591463088989,
    1.0661640167236328,
    -0.5090325474739075,
    -0.2727581858634949,
    -1.3675414323806763,
    -0.2553254961967468,
    -0.26907554268836975,
    -0.5376840829849243,
    -0.0464097298681736,
    0.6657370328903198,
    0.19690127670764923,
    -0.5460608005523682,
    -0.4035342037677765,
    -0.23683024942874908,
    0.25928452610969543,
    -0.30133944749832153,
    0.211341992020607,
    -1.1206848621368408,
    0.3581933379173279,
    -0.04225143790245056,
    0.2604829967021942,
    0.22864092886447906,
    0.7056031823158264,
)
"""Released per-channel mean for H3 video diffusion latents."""

_VIDEO_LATENTS_STD = (
    1.2223774194717407,
    1.2767263650894165,
    1.6831774711608887,
    1.7549455165863037,
    1.5636216402053833,
    2.194143533706665,
    0.9653137922286987,
    1.0569885969161987,
    0.841948926448822,
    0.7729952931404114,
    1.8955937623977661,
    0.946841835975647,
    0.7996809482574463,
    0.44988900423049927,
    0.7197399735450745,
    0.6936293244361877,
    2.961095094680786,
    2.7694199085235596,
    3.0496184825897217,
    2.1088054180145264,
    3.276226282119751,
    3.1627357006073,
    2.2816812992095947,
    2.6127843856811523,
)
"""Released per-channel standard deviation for H3 video diffusion latents."""


@dataclass
class MiniMaxH3VideoEncoderOutput:
    """Video posterior returned by the native causal encoder."""

    latent_dist: MiniMaxH3VideoDiagonalGaussianDistribution
    """Posterior over the encoded video latents."""


class MiniMaxH3VideoDiagonalGaussianDistribution:
    """Diagonal Gaussian posterior represented by concatenated moments."""

    def __init__(self, parameters: Tensor) -> None:
        self.parameters = parameters
        self.mean, self.logvar = parameters.chunk(2, dim=1)
        self.logvar = self.logvar.clamp(-30.0, 20.0)
        self.std = torch.exp(0.5 * self.logvar)

    def mode(self) -> Tensor:
        """Return the posterior mean."""
        return self.mean

    def sample(self, generator: torch.Generator | None = None) -> Tensor:
        """Draw one posterior sample with an optional device generator."""
        noise_device = self.mean.device if generator is None else generator.device
        noise = torch.randn(
            self.mean.shape,
            generator=generator,
            device=noise_device,
            dtype=self.mean.dtype,
        ).to(self.mean.device)
        return self.mean + self.std * noise


class MiniMaxH3VideoCausalConv3d(nn.Conv3d):
    """
    3D convolution used throughout the MiniMax-H3 video encoder.

    Spatial padding is symmetric and uses ``spatial_padding_mode``. Temporal
    padding is causal: preceding frames are zero-padded and no frames follow.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int, int],
        stride: int | tuple[int, int, int] = 1,
        spatial_padding: int = 0,
        temporal_padding: int = 0,
        spatial_padding_mode: str = "reflect",
    ) -> None:
        super().__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
        )
        self.spatial_padding = spatial_padding
        self.temporal_padding = temporal_padding
        self.spatial_padding_mode = spatial_padding_mode

    def forward(self, hidden_states: Tensor) -> Tensor:
        """Apply the causal convolution to one video tensor."""
        if self.spatial_padding > 0:
            padding = self.spatial_padding
            hidden_states = F.pad(
                hidden_states,
                (padding, padding, padding, padding, 0, 0),
                mode=self.spatial_padding_mode,
            )
        if self.temporal_padding > 0:
            hidden_states = F.pad(
                hidden_states,
                (0, 0, 0, 0, self.temporal_padding, 0),
                mode="constant",
            )
        return F.conv3d(
            hidden_states,
            self.weight,
            self.bias,
            stride=self.stride,
            padding=0,
            dilation=self.dilation,
        )


class MiniMaxH3VideoGroupNorm(nn.GroupNorm):
    """
    Group normalization applied to each latent frame in isolation.

    The temporal axis is folded into the batch axis so statistics never mix
    across frames.
    """

    def forward(self, hidden_states: Tensor) -> Tensor:
        """Normalize each video frame independently."""
        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        hidden_states = hidden_states.permute(0, 2, 1, 3, 4).contiguous()
        hidden_states = hidden_states.view(
            batch_size * num_frames, num_channels, 1, height, width
        )
        hidden_states = super().forward(hidden_states)
        hidden_states = hidden_states.view(
            batch_size, num_frames, num_channels, height, width
        )
        return hidden_states.permute(0, 2, 1, 3, 4).contiguous()


class MiniMaxH3VideoResnetBlock3d(nn.Module):
    """One causal residual block in the H3 video encoder."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        norm_num_groups: int = 32,
        norm_eps: float = 1e-6,
        spatial_padding_mode: str = "reflect",
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.norm1 = MiniMaxH3VideoGroupNorm(
            norm_num_groups, in_channels, eps=norm_eps, affine=True
        )
        self.conv1 = MiniMaxH3VideoCausalConv3d(
            in_channels,
            out_channels,
            kernel_size=3,
            spatial_padding=1,
            temporal_padding=2,
            spatial_padding_mode=spatial_padding_mode,
        )
        self.norm2 = MiniMaxH3VideoGroupNorm(
            norm_num_groups, out_channels, eps=norm_eps, affine=True
        )
        self.conv2 = MiniMaxH3VideoCausalConv3d(
            out_channels,
            out_channels,
            kernel_size=3,
            spatial_padding=1,
            temporal_padding=2,
            spatial_padding_mode=spatial_padding_mode,
        )
        self.conv_shortcut = None
        if in_channels != out_channels:
            self.conv_shortcut = MiniMaxH3VideoCausalConv3d(
                in_channels, out_channels, kernel_size=1
            )

    def forward(self, hidden_states: Tensor) -> Tensor:
        """Apply the causal residual block."""
        residual = hidden_states
        hidden_states = F.silu(self.norm1(hidden_states))
        hidden_states = self.conv1(hidden_states)
        hidden_states = F.silu(self.norm2(hidden_states))
        hidden_states = self.conv2(hidden_states)
        if self.conv_shortcut is not None:
            residual = self.conv_shortcut(residual)
        return residual + hidden_states


class MiniMaxH3VideoDownsample3d(nn.Module):
    """
    Strided 3x3x3 downsampling convolution. A spatial stride of 2 is preceded by an asymmetric bottom/right pad of 1
    (the convolution itself carries no spatial padding), so the output is exactly ``ceil(size / 2)``.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temporal_stride: int = 1,
        spatial_stride: int = 2,
        spatial_padding_mode: str = "reflect",
    ) -> None:
        super().__init__()
        self.spatial_stride = spatial_stride
        self.spatial_padding_mode = spatial_padding_mode
        self.conv = MiniMaxH3VideoCausalConv3d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=(temporal_stride, spatial_stride, spatial_stride),
            spatial_padding=0,
            temporal_padding=2,
            spatial_padding_mode=spatial_padding_mode,
        )

    def forward(self, hidden_states: Tensor) -> Tensor:
        """Downsample one video tensor."""
        if self.spatial_stride == 2:
            hidden_states = F.pad(
                hidden_states,
                (0, 1, 0, 1, 0, 0),
                mode=self.spatial_padding_mode,
            )
        return self.conv(hidden_states)


class MiniMaxH3VideoDownBlock3d(nn.Module):
    """Residual encoder stage with optional spatiotemporal downsampling."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_layers: int,
        temporal_downsample_factor: int,
        spatial_downsample_factor: int,
        norm_num_groups: int = 32,
        norm_eps: float = 1e-6,
        spatial_padding_mode: str = "reflect",
    ) -> None:
        super().__init__()
        self.resnets = nn.ModuleList(
            [
                MiniMaxH3VideoResnetBlock3d(
                    in_channels=in_channels if i == 0 else out_channels,
                    out_channels=out_channels,
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                    spatial_padding_mode=spatial_padding_mode,
                )
                for i in range(num_layers)
            ]
        )
        self.downsamplers = None
        if temporal_downsample_factor * spatial_downsample_factor > 1:
            self.downsamplers = nn.ModuleList(
                [
                    MiniMaxH3VideoDownsample3d(
                        out_channels,
                        out_channels,
                        temporal_stride=temporal_downsample_factor,
                        spatial_stride=spatial_downsample_factor,
                        spatial_padding_mode=spatial_padding_mode,
                    )
                ]
            )

        self.gradient_checkpointing = False

    def forward(self, hidden_states: Tensor) -> Tensor:
        """Apply the encoder stage and its optional downsampler."""
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states)
        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)
        return hidden_states


class MiniMaxH3VideoEncoder3d(nn.Module):
    """
    Causal 3D CNN encoder. ``block_out_channels`` gives the channel count of every level; the per-level
    ``spatial_downsample_factors`` / ``temporal_downsample_factors`` multiply out to the total compression ratios.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 48,
        block_out_channels: tuple[int, ...] = (128, 256, 256, 512, 512, 1024),
        layers_per_block: int = 2,
        spatial_downsample_factors: tuple[int, ...] = (2, 2, 2, 2, 1, 1),
        temporal_downsample_factors: tuple[int, ...] = (1, 2, 2, 1, 1, 1),
        norm_num_groups: int = 32,
        norm_eps: float = 1e-6,
        spatial_padding_mode: str = "reflect",
    ) -> None:
        super().__init__()

        self.conv_in = MiniMaxH3VideoCausalConv3d(
            in_channels,
            block_out_channels[0],
            kernel_size=3,
            spatial_padding=1,
            temporal_padding=2,
            spatial_padding_mode=spatial_padding_mode,
        )

        block_in_channels = (block_out_channels[0],) + tuple(block_out_channels[:-1])
        self.down_blocks = nn.ModuleList(
            [
                MiniMaxH3VideoDownBlock3d(
                    in_channels=block_in_channels[i],
                    out_channels=block_out_channels[i],
                    num_layers=layers_per_block,
                    temporal_downsample_factor=temporal_downsample_factors[i],
                    spatial_downsample_factor=spatial_downsample_factors[i],
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                    spatial_padding_mode=spatial_padding_mode,
                )
                for i in range(len(block_out_channels))
            ]
        )

        self.norm_out = MiniMaxH3VideoGroupNorm(
            norm_num_groups,
            block_out_channels[-1],
            eps=norm_eps,
            affine=True,
        )
        self.conv_out = MiniMaxH3VideoCausalConv3d(
            block_out_channels[-1],
            out_channels,
            kernel_size=3,
            spatial_padding=1,
            temporal_padding=2,
            spatial_padding_mode=spatial_padding_mode,
        )

    def forward(self, hidden_states: Tensor) -> Tensor:
        """Encode one ImageNet-normalized video tensor."""
        hidden_states = self.conv_in(hidden_states)
        for down_block in self.down_blocks:
            hidden_states = down_block(hidden_states)
        hidden_states = F.silu(self.norm_out(hidden_states))
        return self.conv_out(hidden_states)


class MiniMaxH3VideoRotaryPosEmbed(nn.Module):
    """
    Three-axis rotary embedding for the ViT decoder.

    Coordinates are length-normalized to ``[-1, 1)`` per axis. The resulting
    time, height, and width angles rotate the configured prefix of each head.
    """

    def __init__(self, dim: int, theta: float = 100.0, num_axes: int = 3) -> None:
        super().__init__()
        if dim % (2 * num_axes) != 0:
            raise ValueError(
                f"dim {dim} must be divisible by 2 * num_axes {2 * num_axes}."
            )
        inv_freq = 1.0 / theta ** torch.arange(
            0, 1, 2 * num_axes / dim, dtype=torch.float32
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, position_ids: Tensor) -> tuple[Tensor, Tensor]:
        """Build cosine and sine tensors for three-axis RoPE."""
        inv_freq = cast(Tensor, self.inv_freq)
        angles = (
            2.0 * math.pi * position_ids[:, :, :, None] * inv_freq[None, None, None, :]
        )
        angles = angles.flatten(2, 3).tile(2).unsqueeze(2)
        return angles.cos(), angles.sin()


class _MiniMaxH3VideoSwiGLU(nn.Module):
    """Checkpoint-compatible SwiGLU input projection."""

    def __init__(self, dim: int, inner_dim: int, *, bias: bool = True) -> None:
        super().__init__()
        self.proj = nn.Linear(dim, inner_dim * 2, bias=bias)

    def forward(self, hidden_states: Tensor) -> Tensor:
        """Apply the value-times-SiLU-gate projection."""
        value, gate = self.proj(hidden_states).chunk(2, dim=-1)
        return value * F.silu(gate)


class _MiniMaxH3VideoFeedForward(nn.Module):
    """Checkpoint-compatible decoder SwiGLU feed-forward network."""

    def __init__(self, dim: int, mult: int, *, bias: bool = True) -> None:
        super().__init__()
        inner_dim = dim * mult
        self.net = nn.ModuleList(
            [
                _MiniMaxH3VideoSwiGLU(dim, inner_dim, bias=bias),
                nn.Dropout(0.0),
                nn.Linear(inner_dim, dim, bias=bias),
            ]
        )

    def forward(self, hidden_states: Tensor) -> Tensor:
        """Apply the checkpoint-compatible decoder MLP."""
        for module in self.net:
            hidden_states = module(hidden_states)
        return hidden_states


class MiniMaxH3VideoAttention(nn.Module):
    """Full self-attention used by the non-causal H3 ViT decoder."""

    def __init__(
        self,
        dim: int,
        heads: int,
        dim_head: int,
        eps: float = 1e-5,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        inner_dim = heads * dim_head
        self.norm_q = nn.RMSNorm(dim_head, eps=eps, elementwise_affine=False)
        self.norm_k = nn.RMSNorm(dim_head, eps=eps, elementwise_affine=False)
        self.to_q = nn.Linear(dim, inner_dim, bias=bias)
        self.to_k = nn.Linear(dim, inner_dim, bias=bias)
        self.to_v = nn.Linear(dim, inner_dim, bias=bias)
        self.to_out = nn.ModuleList(
            [nn.Linear(inner_dim, dim, bias=bias), nn.Dropout(0.0)]
        )

    def forward(
        self,
        hidden_states: Tensor,
        rotary_emb: tuple[Tensor, Tensor] | None = None,
    ) -> Tensor:
        """Apply rotary full self-attention to one token sequence."""
        query = self.to_q(hidden_states).unflatten(2, (self.heads, -1))
        key = self.to_k(hidden_states).unflatten(2, (self.heads, -1))
        value = self.to_v(hidden_states).unflatten(2, (self.heads, -1))
        query = self.norm_q(query.float()).to(query.dtype)
        key = self.norm_k(key.float()).to(key.dtype)
        if rotary_emb is not None:
            cos, sin = (value.to(query.dtype) for value in rotary_emb)
            rotary_dim = cos.shape[-1]
            query_rotary = query[..., :rotary_dim]
            query_pass = query[..., rotary_dim:]
            key_rotary = key[..., :rotary_dim]
            key_pass = key[..., rotary_dim:]
            query_first, query_second = query_rotary.chunk(2, dim=-1)
            key_first, key_second = key_rotary.chunk(2, dim=-1)
            query_rotated = torch.cat((-query_second, query_first), dim=-1)
            key_rotated = torch.cat((-key_second, key_first), dim=-1)
            query = torch.cat(
                (query_rotary * cos + query_rotated * sin, query_pass), dim=-1
            )
            key = torch.cat((key_rotary * cos + key_rotated * sin, key_pass), dim=-1)
        attended = F.scaled_dot_product_attention(
            query.transpose(1, 2),
            key.transpose(1, 2),
            value.transpose(1, 2),
        ).transpose(1, 2)
        return self.to_out[0](attended.flatten(2, 3))


class MiniMaxH3VideoTransformerBlock(nn.Module):
    """One scaled attention and SwiGLU block in the H3 ViT decoder."""

    def __init__(
        self,
        dim: int,
        heads: int,
        dim_head: int,
        ffn_mult: int = 4,
        eps: float = 1e-5,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.norm1 = nn.RMSNorm(dim, eps=eps, elementwise_affine=True)
        self.attn = MiniMaxH3VideoAttention(
            dim=dim, heads=heads, dim_head=dim_head, eps=eps, bias=bias
        )
        self.scale1 = nn.Parameter(torch.zeros(dim))
        self.norm2 = nn.RMSNorm(dim, eps=eps, elementwise_affine=True)
        self.ff = _MiniMaxH3VideoFeedForward(dim, ffn_mult, bias=bias)
        self.scale2 = nn.Parameter(torch.zeros(dim))

    def forward(
        self,
        hidden_states: Tensor,
        rotary_emb: tuple[Tensor, Tensor] | None = None,
    ) -> Tensor:
        """Apply one scaled decoder transformer block."""
        normalized = self.norm1(hidden_states.float()).to(hidden_states.dtype)
        hidden_states = hidden_states + self.attn(normalized, rotary_emb) * self.scale1
        normalized = self.norm2(hidden_states.float()).to(hidden_states.dtype)
        return hidden_states + self.ff(normalized) * self.scale2


class MiniMaxH3VideoViTDecoder3d(nn.Module):
    """
    Non-causal ViT decoder from latent voxels to pixel patches.

    Learned register tokens and one zero token join the voxel sequence for full
    self-attention, then are removed before the pixel patch projection.
    """

    def __init__(
        self,
        in_channels: int = 24,
        out_channels: int = 3,
        patch_size: int = 16,
        patch_size_t: int = 4,
        num_layers: int = 36,
        num_attention_heads: int = 32,
        attention_head_dim: int = 64,
        num_register_tokens: int = 4,
        ffn_mult: int = 4,
        rope_theta: float = 100.0,
        rope_dim_ratio: float = 0.75,
        norm_eps: float = 1e-5,
    ) -> None:
        super().__init__()
        dim = num_attention_heads * attention_head_dim
        self.patch_size = patch_size
        self.patch_size_t = patch_size_t
        self.out_channels = out_channels
        self.num_register_tokens = num_register_tokens

        self.rope = MiniMaxH3VideoRotaryPosEmbed(
            int(attention_head_dim * rope_dim_ratio), theta=rope_theta
        )
        self.proj_in = nn.Linear(in_channels, dim)
        self.register_tokens = nn.Parameter(torch.zeros(1, num_register_tokens, dim))
        self.transformer_blocks = nn.ModuleList(
            [
                MiniMaxH3VideoTransformerBlock(
                    dim=dim,
                    heads=num_attention_heads,
                    dim_head=attention_head_dim,
                    ffn_mult=ffn_mult,
                    eps=norm_eps,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm_out = nn.LayerNorm(dim, elementwise_affine=True, eps=norm_eps)
        self.proj_out = nn.Linear(
            dim, out_channels * patch_size_t * patch_size * patch_size
        )

        self.gradient_checkpointing = False

    def forward(self, hidden_states: Tensor) -> Tensor:
        """Decode latent voxels into spatiotemporal RGB patches."""
        batch_size, num_channels, num_frames, height, width = hidden_states.shape

        hidden_states = hidden_states.permute(0, 2, 3, 4, 1).reshape(
            batch_size, num_frames * height * width, num_channels
        )
        hidden_states = self.proj_in(hidden_states)
        num_patches = hidden_states.shape[1]

        register_tokens = self.register_tokens.expand(batch_size, -1, -1)
        cls_token = torch.zeros_like(hidden_states[:, :1, :])
        hidden_states = torch.cat([hidden_states, register_tokens, cls_token], dim=1)

        grids = [
            2.0
            * (
                torch.arange(
                    0.5, size, dtype=torch.float32, device=hidden_states.device
                )
                / size
            )
            - 1.0
            for size in (num_frames, height, width)
        ]
        position_ids = torch.stack(
            torch.meshgrid(*grids, indexing="ij"), dim=-1
        ).flatten(0, 2)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1, -1)
        suffix_ids = position_ids.new_zeros(
            (batch_size, self.num_register_tokens + 1, 3)
        )
        position_ids = torch.cat([position_ids, suffix_ids], dim=1)
        rotary_emb = self.rope(position_ids)

        for block in self.transformer_blocks:
            hidden_states = block(hidden_states, rotary_emb)

        hidden_states = self.norm_out(hidden_states)
        hidden_states = self.proj_out(hidden_states)
        hidden_states = hidden_states[:, :num_patches, :]

        patch_size, patch_size_t = self.patch_size, self.patch_size_t
        hidden_states = hidden_states.view(
            batch_size,
            num_frames,
            height,
            width,
            self.out_channels,
            patch_size_t,
            patch_size,
            patch_size,
        )
        hidden_states = hidden_states.permute(0, 4, 1, 5, 2, 6, 3, 7).contiguous()
        return hidden_states.reshape(
            batch_size,
            self.out_channels,
            num_frames * patch_size_t,
            height * patch_size,
            width * patch_size,
        )


@dataclass(kw_only=True)
class MiniMaxH3VideoVAEConfig(InstantiateConfig):
    """Configure the native FP32 MiniMax H3 video autoencoder."""

    _target: type[MiniMaxH3VideoVAE] = field(default_factory=lambda: MiniMaxH3VideoVAE)
    checkpoint_path: str | None = H3_VIDEO_VAE_CHECKPOINT
    """Sharded checkpoint index URL or local path; ``None`` skips loading."""

    checkpoint_min_free_gb: float | None = None
    """Optional free-space floor used before checkpoint downloads."""

    device: str = "cpu"
    """Device on which parameters are constructed and inference runs."""

    in_channels: int = 3
    """Input RGB channel count."""

    out_channels: int = 3
    """Decoded RGB channel count."""

    latent_channels: int = 24
    """Width of the normalized H3 video diffusion latent."""

    block_out_channels: tuple[int, ...] = (128, 256, 256, 512, 512, 1024)
    """Channel width of each causal encoder stage."""

    layers_per_block: int = 2
    """Residual units in every causal encoder stage."""

    spatial_downsample_factors: tuple[int, ...] = (2, 2, 2, 2, 1, 1)
    """Per-stage spatial compression factors."""

    temporal_downsample_factors: tuple[int, ...] = (1, 2, 2, 1, 1, 1)
    """Per-stage temporal compression factors."""

    norm_num_groups: int = 32
    """Group count for frame-isolated encoder normalization."""

    norm_eps: float = 1e-6
    """Encoder group-normalization epsilon."""

    spatial_padding_mode: str = "reflect"
    """Spatial padding mode used by causal encoder convolutions."""

    decoder_num_layers: int = 36
    """Number of non-causal ViT decoder blocks."""

    decoder_num_attention_heads: int = 32
    """Self-attention head count in each decoder block."""

    decoder_attention_head_dim: int = 64
    """Width of each decoder attention head."""

    decoder_num_register_tokens: int = 4
    """Learned decoder register-token count."""

    decoder_ffn_mult: int = 4
    """SwiGLU hidden-width multiplier in decoder blocks."""

    decoder_rope_theta: float = 100.0
    """Base frequency for decoder rotary embeddings."""

    decoder_rope_dim_ratio: float = 0.75
    """Fraction of every decoder attention head rotated by RoPE."""

    decoder_norm_eps: float = 1e-5
    """Decoder RMS- and layer-normalization epsilon."""

    clip_length: int = 17
    """Pixel frames encoded in each released temporal chunk."""

    token_drop: int = 3
    """Trailing latent frames removed after temporal chunk encoding."""

    latents_mean: tuple[float, ...] = _VIDEO_LATENTS_MEAN
    """Released per-channel video latent means."""

    latents_std: tuple[float, ...] = _VIDEO_LATENTS_STD
    """Released per-channel video latent standard deviations."""

    use_slicing: bool = False
    """Decode batch items separately to reduce peak memory."""

    use_tiling: bool = True
    """Preserve the released spatially tiled encode/decode behavior."""

    tile_sample_min_height: int = 256
    """Pixel-space height of an encoder or decoder tile."""

    tile_sample_min_width: int = 256
    """Pixel-space width of an encoder or decoder tile."""

    tile_sample_min_overlap_height: int = 64
    """Minimum vertical overlap between neighboring tiles."""

    tile_sample_min_overlap_width: int = 64
    """Minimum horizontal overlap between neighboring tiles."""


class MiniMaxH3VideoVAE(nn.Module):
    """Encode H3 conditioning and decode generated video in released geometry."""

    config: MiniMaxH3VideoVAEConfig

    def __init__(self, config: MiniMaxH3VideoVAEConfig) -> None:
        super().__init__()
        self.config = config
        self._validate_config()
        self.spatial_compression_ratio = math.prod(config.spatial_downsample_factors)
        self.temporal_compression_ratio = math.prod(config.temporal_downsample_factors)
        with torch.device(config.device):
            self.encoder = MiniMaxH3VideoEncoder3d(
                in_channels=config.in_channels,
                out_channels=2 * config.latent_channels,
                block_out_channels=config.block_out_channels,
                layers_per_block=config.layers_per_block,
                spatial_downsample_factors=config.spatial_downsample_factors,
                temporal_downsample_factors=config.temporal_downsample_factors,
                norm_num_groups=config.norm_num_groups,
                norm_eps=config.norm_eps,
                spatial_padding_mode=config.spatial_padding_mode,
            )
            self.quant_conv = nn.Conv3d(
                2 * config.latent_channels,
                2 * config.latent_channels,
                kernel_size=1,
            )
            self.post_quant_conv = nn.Conv3d(
                config.latent_channels,
                config.latent_channels,
                kernel_size=1,
            )
            self.decoder = MiniMaxH3VideoViTDecoder3d(
                in_channels=config.latent_channels,
                out_channels=config.out_channels,
                patch_size=self.spatial_compression_ratio,
                patch_size_t=self.temporal_compression_ratio,
                num_layers=config.decoder_num_layers,
                num_attention_heads=config.decoder_num_attention_heads,
                attention_head_dim=config.decoder_attention_head_dim,
                num_register_tokens=config.decoder_num_register_tokens,
                ffn_mult=config.decoder_ffn_mult,
                rope_theta=config.decoder_rope_theta,
                rope_dim_ratio=config.decoder_rope_dim_ratio,
                norm_eps=config.decoder_norm_eps,
            )
        self.frame_pre_padding = (-config.clip_length) % self.temporal_compression_ratio
        self.tokens_chunk_size = math.ceil(
            config.clip_length / self.temporal_compression_ratio
        )
        self.token_overlap = (-config.token_drop) % self.tokens_chunk_size
        self.frame_overlap = max(
            self.token_overlap * self.temporal_compression_ratio
            - self.frame_pre_padding,
            0,
        )
        self.use_slicing = config.use_slicing
        self.use_tiling = config.use_tiling
        self.tile_sample_min_height = config.tile_sample_min_height
        self.tile_sample_min_width = config.tile_sample_min_width
        self.tile_sample_min_overlap_height = config.tile_sample_min_overlap_height
        self.tile_sample_min_overlap_width = config.tile_sample_min_overlap_width
        self.float()
        if config.checkpoint_path is not None:
            load_checkpoint(
                config.checkpoint_path,
                model=self,
                checkpoint_min_free_gb=config.checkpoint_min_free_gb,
            )
        self.eval()

    def _validate_config(self) -> None:
        config = self.config
        integer_values = {
            "in_channels": config.in_channels,
            "out_channels": config.out_channels,
            "latent_channels": config.latent_channels,
            "layers_per_block": config.layers_per_block,
            "norm_num_groups": config.norm_num_groups,
            "decoder_num_layers": config.decoder_num_layers,
            "decoder_num_attention_heads": config.decoder_num_attention_heads,
            "decoder_attention_head_dim": config.decoder_attention_head_dim,
            "decoder_num_register_tokens": config.decoder_num_register_tokens,
            "decoder_ffn_mult": config.decoder_ffn_mult,
            "clip_length": config.clip_length,
        }
        if any(
            type(value) is not int or value <= 0 for value in integer_values.values()
        ):
            raise ValueError(
                "Video VAE dimensions and counts must be positive integers."
            )
        if type(config.token_drop) is not int or config.token_drop < 0:
            raise ValueError("token_drop must be a non-negative integer.")
        block_count = len(config.block_out_channels)
        if block_count == 0 or len(config.spatial_downsample_factors) != block_count:
            raise ValueError("Each encoder block requires a spatial downsample factor.")
        if len(config.temporal_downsample_factors) != block_count:
            raise ValueError(
                "Each encoder block requires a temporal downsample factor."
            )
        sequences = (
            config.block_out_channels,
            config.spatial_downsample_factors,
            config.temporal_downsample_factors,
        )
        if any(
            type(value) is not int or value <= 0
            for values in sequences
            for value in values
        ):
            raise ValueError(
                "Encoder widths and downsample factors must be positive integers."
            )
        if any(width % config.norm_num_groups for width in config.block_out_channels):
            raise ValueError("Encoder widths must be divisible by norm_num_groups.")
        if config.spatial_padding_mode not in {"reflect", "replicate"}:
            raise ValueError("spatial_padding_mode must be reflect or replicate.")
        if not math.isfinite(config.norm_eps) or config.norm_eps <= 0:
            raise ValueError("norm_eps must be positive and finite.")
        if not math.isfinite(config.decoder_norm_eps) or config.decoder_norm_eps <= 0:
            raise ValueError("decoder_norm_eps must be positive and finite.")
        if (
            not math.isfinite(config.decoder_rope_theta)
            or config.decoder_rope_theta <= 0
        ):
            raise ValueError("decoder_rope_theta must be positive and finite.")
        if not math.isfinite(config.decoder_rope_dim_ratio) or not (
            0 < config.decoder_rope_dim_ratio <= 1
        ):
            raise ValueError("decoder_rope_dim_ratio must be in (0, 1].")
        rope_dim = int(
            config.decoder_attention_head_dim * config.decoder_rope_dim_ratio
        )
        if rope_dim % 6:
            raise ValueError("The decoder rotary width must be divisible by six.")
        temporal_ratio = math.prod(config.temporal_downsample_factors)
        tokens_chunk_size = math.ceil(config.clip_length / temporal_ratio)
        if config.token_drop >= tokens_chunk_size:
            raise ValueError("token_drop must be smaller than the latent chunk size.")
        if (
            len(config.latents_mean) != config.latent_channels
            or len(config.latents_std) != config.latent_channels
        ):
            raise ValueError("Video latent statistics must match latent_channels.")
        if not all(math.isfinite(value) for value in config.latents_mean):
            raise ValueError("Video latent means must be finite.")
        if not all(math.isfinite(value) and value > 0 for value in config.latents_std):
            raise ValueError("Video latent standard deviations must be positive.")
        tile_values = (
            config.tile_sample_min_height,
            config.tile_sample_min_width,
            config.tile_sample_min_overlap_height,
            config.tile_sample_min_overlap_width,
        )
        if any(type(value) is not int or value <= 0 for value in tile_values):
            raise ValueError("Video tile sizes and overlaps must be positive integers.")
        spatial_ratio = math.prod(config.spatial_downsample_factors)
        if any(value % spatial_ratio for value in tile_values):
            raise ValueError("Video tile geometry must align to the spatial ratio.")
        if (
            config.tile_sample_min_overlap_height >= config.tile_sample_min_height
            or config.tile_sample_min_overlap_width >= config.tile_sample_min_width
        ):
            raise ValueError("Video tile overlaps must be smaller than tile sizes.")
        if type(config.use_slicing) is not bool or type(config.use_tiling) is not bool:
            raise ValueError("Video slicing and tiling switches must be booleans.")

    @property
    def device(self) -> torch.device:
        """Return the device of the first autoencoder parameter."""
        return next(self.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        """Return the autoencoder parameter dtype."""
        return next(self.parameters()).dtype

    def _require_fp32(self) -> None:
        if self.dtype != torch.float32:
            raise RuntimeError("MiniMax H3 video VAE weights must remain float32.")

    def enable_tiling(
        self,
        tile_sample_min_height: int | None = None,
        tile_sample_min_width: int | None = None,
        tile_sample_min_overlap_height: int | None = None,
        tile_sample_min_overlap_width: int | None = None,
    ) -> None:
        """
        Enable tiled VAE encoding/decoding. When this option is enabled, the VAE splits the frames into tiles, encodes
        or decodes each tile separately and linearly blends the overlaps back together. This lowers the memory
        requirement and allows processing larger frames.

        Args:
            tile_sample_min_height (``int``, *optional*):
                The tile height in pixel space. Frames taller than this are split along the height dimension.
            tile_sample_min_width (``int``, *optional*):
                The tile width in pixel space. Frames wider than this are split along the width dimension.
            tile_sample_min_overlap_height (``int``, *optional*):
                The minimum overlap, in pixels, between two consecutive vertical tiles.
            tile_sample_min_overlap_width (``int``, *optional*):
                The minimum overlap, in pixels, between two consecutive horizontal tiles.
        """
        self.use_tiling = True
        self.tile_sample_min_height = (
            tile_sample_min_height or self.tile_sample_min_height
        )
        self.tile_sample_min_width = tile_sample_min_width or self.tile_sample_min_width
        self.tile_sample_min_overlap_height = (
            tile_sample_min_overlap_height or self.tile_sample_min_overlap_height
        )
        self.tile_sample_min_overlap_width = (
            tile_sample_min_overlap_width or self.tile_sample_min_overlap_width
        )

    def _split_tiles(
        self, length: int, tile_size: int, min_overlap: int
    ) -> tuple[list[int], list[int], list[int]]:
        """
        Lay aligned overlapping tiles over one pixel-space dimension.

        Slack is distributed across overlaps in whole spatial-compression steps
        so that every tile boundary remains latent-aligned.
        """
        if tile_size >= length:
            return [0], [length], []

        num_tiles = math.ceil(length / tile_size)
        while tile_size * num_tiles - min_overlap * (num_tiles - 1) - length < 0:
            num_tiles += 1

        overlaps = [min_overlap] * (num_tiles - 1)
        remaining = tile_size * num_tiles - sum(overlaps) - length
        for i in range(remaining // self.spatial_compression_ratio):
            overlaps[i % (num_tiles - 1)] += self.spatial_compression_ratio

        tile_start_indices = [0]
        for i in range(num_tiles - 1):
            tile_start_indices.append(tile_start_indices[-1] + tile_size - overlaps[i])
        return tile_start_indices, [tile_size] * num_tiles, overlaps

    def _blend(self, a: Tensor, b: Tensor, blend_extent: int, dim: int) -> Tensor:
        blend_extent = min(a.shape[dim], b.shape[dim], blend_extent)
        positions = torch.arange(blend_extent, device=b.device, dtype=b.dtype)
        shape = [1] * a.ndim
        shape[dim] = blend_extent
        weight_a = (1 - positions / blend_extent).view(shape)
        weight_b = (positions / blend_extent).view(shape)

        slice_a = [slice(None)] * a.ndim
        slice_a[dim] = slice(-blend_extent, None)
        slice_b = [slice(None)] * b.ndim
        slice_b[dim] = slice(0, blend_extent)
        blended = a[tuple(slice_a)] * weight_a + b[tuple(slice_b)] * weight_b

        if blend_extent == b.shape[dim]:
            return blended
        slice_rest = [slice(None)] * b.ndim
        slice_rest[dim] = slice(blend_extent, None)
        return torch.cat([blended, b[tuple(slice_rest)]], dim=dim)

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
                        tiles[i - 1][j], tile, height_overlaps[i - 1], dim=-2
                    )
                if j > 0:
                    tile = self._blend(row[j - 1], tile, width_overlaps[j - 1], dim=-1)
                if i < len(tiles) - 1:
                    tile = tile[..., : -height_overlaps[i], :]
                if j < len(row) - 1:
                    tile = tile[..., :, : -width_overlaps[j]]
                result_row.append(tile)
            result_rows.append(torch.cat(result_row, dim=-1))
        return torch.cat(result_rows, dim=-2)

    def _encode_clip(self, x: Tensor) -> Tensor:
        """Encode one temporal clip, spatially tiled when tiling is enabled."""
        if not self.use_tiling:
            return self.quant_conv(self.encoder(x))

        height, width = x.shape[-2], x.shape[-1]
        y_indices, y_lengths, y_overlaps = self._split_tiles(
            height, self.tile_sample_min_height, self.tile_sample_min_overlap_height
        )
        x_indices, x_lengths, x_overlaps = self._split_tiles(
            width, self.tile_sample_min_width, self.tile_sample_min_overlap_width
        )

        rows = []
        for i_pos, i_len in zip(y_indices, y_lengths):
            row = []
            for j_pos, j_len in zip(x_indices, x_lengths):
                tile = x[..., i_pos : i_pos + i_len, j_pos : j_pos + j_len]
                row.append(self.quant_conv(self.encoder(tile)))
            rows.append(row)

        latent_y_overlaps = [
            overlap // self.spatial_compression_ratio for overlap in y_overlaps
        ]
        latent_x_overlaps = [
            overlap // self.spatial_compression_ratio for overlap in x_overlaps
        ]
        return self._stitch_tiles(rows, latent_y_overlaps, latent_x_overlaps)

    def _decode_clip(self, z: Tensor) -> Tensor:
        """Decode one temporal clip, spatially tiled when tiling is enabled."""
        if not self.use_tiling:
            return self.decoder(self.post_quant_conv(z))

        # Tiles are laid out in pixel space and then mapped back onto the latent grid.
        height = z.shape[-2] * self.spatial_compression_ratio
        width = z.shape[-1] * self.spatial_compression_ratio
        y_indices, y_lengths, y_overlaps = self._split_tiles(
            height, self.tile_sample_min_height, self.tile_sample_min_overlap_height
        )
        x_indices, x_lengths, x_overlaps = self._split_tiles(
            width, self.tile_sample_min_width, self.tile_sample_min_overlap_width
        )

        ratio = self.spatial_compression_ratio
        rows = []
        for i_pos, i_len in zip(y_indices, y_lengths):
            row = []
            for j_pos, j_len in zip(x_indices, x_lengths):
                tile = z[
                    ...,
                    i_pos // ratio : i_pos // ratio + i_len // ratio,
                    j_pos // ratio : j_pos // ratio + j_len // ratio,
                ]
                row.append(self.decoder(self.post_quant_conv(tile)))
            rows.append(row)

        return self._stitch_tiles(rows, y_overlaps, x_overlaps)

    def _encode(self, x: Tensor) -> Tensor:
        """
        Encode a video in ``clip_length``-frame chunks and drop the ``token_drop`` trailing latent frames.

        A single frame has no temporal extent, so it goes through the spatial
        encoder alone. Longer videos are padded to whole temporal chunks.
        """
        clip_length = self.config.clip_length
        num_frames = x.shape[2]
        if num_frames == 1:
            return self._encode_clip(x)
        if num_frames % clip_length != 0:
            pad_frames = x[:, :, -1:].repeat(1, 1, (-num_frames) % clip_length, 1, 1)
            x = torch.cat([x, pad_frames], dim=2)

        moments = torch.cat(
            [
                self._encode_clip(x[:, :, i * clip_length : (i + 1) * clip_length])
                for i in range(x.shape[2] // clip_length)
            ],
            dim=2,
        )
        if self.config.token_drop > 0:
            moments = moments[:, :, : -self.config.token_drop]
        return moments

    def _decode(self, z: Tensor) -> Tensor:
        """
        Decode a latent video, mirroring the chunking that ``_encode`` applied.

        Consecutive chunks overlap because encoding removed trailing tokens.
        The decoder linearly cross-fades those overlaps and trims repeated tails.
        """
        tokens_chunk_size = self.tokens_chunk_size
        token_drop = self.config.token_drop
        temporal_ratio = self.temporal_compression_ratio
        chunk_num_frames = tokens_chunk_size * temporal_ratio

        num_tokens = z.shape[2] + token_drop
        pad_tokens = (-num_tokens) % tokens_chunk_size
        num_chunks = (num_tokens + pad_tokens) // tokens_chunk_size - int(
            token_drop > 0
        )
        if pad_tokens > 0:
            z = torch.cat([z, z[:, :, -1:].repeat(1, 1, pad_tokens, 1, 1)], dim=2)

        decoded_chunks = []
        overlap = None
        for i in range(num_chunks):
            start = i * tokens_chunk_size
            clip = self._decode_clip(
                z[:, :, start : start + tokens_chunk_size + self.token_overlap]
            )
            for j in range(int(token_drop > 0) + 1):
                frame_start = j * chunk_num_frames
                chunk = clip[:, :, frame_start : frame_start + chunk_num_frames]
                chunk = chunk[:, :, self.frame_pre_padding :]
                if j == 0:
                    if overlap is not None:
                        chunk = self._blend(overlap, chunk, self.frame_overlap, dim=-3)
                    decoded_chunks.append(chunk)
                else:
                    overlap = chunk
        if overlap is not None:
            decoded_chunks.append(overlap)

        dec = torch.cat(decoded_chunks, dim=2)

        # ``pad_tokens`` repeated latent frames produced trailing pixel frames that were never requested. A chunk's
        # The last latent in a chunk may cover fewer than ``temporal_ratio`` frames.
        if pad_tokens > 0:
            intra_tail = self.config.clip_length % temporal_ratio
            num_tokens_before_pad = z.shape[2] - pad_tokens
            pad_frames = sum(
                intra_tail
                if intra_tail and (num_tokens_before_pad + k) % tokens_chunk_size == 0
                else temporal_ratio
                for k in range(pad_tokens)
            )
            dec = dec[:, :, :-pad_frames]
        return dec

    @torch.no_grad()
    def encode(self, sample: Tensor) -> MiniMaxH3VideoEncoderOutput:
        """Encode ImageNet-normalized videos into a diagonal posterior."""
        self._validate_video(sample, channels=self.config.in_channels, name="sample")
        self._require_fp32()
        sample = sample.to(device=self.device, dtype=torch.float32)
        if self.use_slicing and sample.shape[0] > 1:
            moments = torch.cat([self._encode(value) for value in sample.split(1)])
        else:
            moments = self._encode(sample)
        return MiniMaxH3VideoEncoderOutput(
            latent_dist=MiniMaxH3VideoDiagonalGaussianDistribution(moments)
        )

    @torch.no_grad()
    def decode(self, latents: Tensor) -> Tensor:
        """Decode denormalized video latents into ImageNet-normalized RGB."""
        self._validate_video(
            latents, channels=self.config.latent_channels, name="latents"
        )
        self._require_fp32()
        latents = latents.to(device=self.device, dtype=torch.float32)
        if self.use_slicing and latents.shape[0] > 1:
            return torch.cat([self._decode(value) for value in latents.split(1)])
        return self._decode(latents)

    @staticmethod
    def _validate_video(value: Tensor, *, channels: int, name: str) -> None:
        if (
            value.ndim != 5
            or value.shape[0] <= 0
            or value.shape[1] != channels
            or any(size <= 0 for size in value.shape[2:])
        ):
            raise ValueError(
                f"{name} must have shape [batch, {channels}, frames, height, width]."
            )
        if not value.is_floating_point() or not bool(torch.isfinite(value).all()):
            raise ValueError(f"{name} must contain finite floating-point values.")

    def normalize_latents(self, latents: Tensor) -> Tensor:
        """Apply the released per-channel video latent normalization."""
        self._validate_video(
            latents, channels=self.config.latent_channels, name="latents"
        )
        mean = latents.new_tensor(self.config.latents_mean).view(1, -1, 1, 1, 1)
        std = latents.new_tensor(self.config.latents_std).view(1, -1, 1, 1, 1)
        return (latents - mean) / std

    def denormalize_latents(self, latents: Tensor) -> Tensor:
        """Undo the released per-channel video latent normalization."""
        self._validate_video(
            latents, channels=self.config.latent_channels, name="latents"
        )
        mean = latents.new_tensor(self.config.latents_mean).view(1, -1, 1, 1, 1)
        std = latents.new_tensor(self.config.latents_std).view(1, -1, 1, 1, 1)
        return latents * std + mean

    @torch.no_grad()
    def encode_pixels(
        self,
        pixels: Tensor,
        *,
        sample_posterior: bool = False,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        """Encode base-range RGB pixels into normalized H3 video latents."""
        posterior = self._encode_pixel_posterior(pixels)
        latents = (
            posterior.sample(generator=generator)
            if sample_posterior
            else posterior.mode()
        )
        return self.normalize_latents(latents)

    def _encode_pixel_posterior(
        self, pixels: Tensor
    ) -> MiniMaxH3VideoDiagonalGaussianDistribution:
        """Build the video posterior for base-range RGB pixels."""
        self._validate_video(pixels, channels=self.config.in_channels, name="pixels")
        if bool((pixels < 0).any()) or bool((pixels > 1).any()):
            raise ValueError("pixels must stay within [0, 1].")
        mean = pixels.new_tensor(_PIXEL_MEAN).view(1, -1, 1, 1, 1)
        std = pixels.new_tensor(_PIXEL_STD).view(1, -1, 1, 1, 1)
        return self.encode((pixels - mean) / std).latent_dist

    @torch.no_grad()
    def encode_condition_pixels(self, pixels: Tensor, *, seed: int = 42) -> Tensor:
        """Encode visual conditioning with H3's seeded, rounded posterior.

        The conditioning posterior owns a fresh CPU generator and therefore
        never consumes the request's denoising RNG. Its sampled latent is
        deliberately rounded through FP16 before per-channel normalization.
        """
        if type(seed) is not int or seed < 0:
            raise ValueError("conditioning seed must be a non-negative integer")
        posterior = self._encode_pixel_posterior(pixels)
        generator = torch.Generator(device="cpu").manual_seed(seed)
        latents = posterior.sample(generator=generator)
        latents = latents.to(torch.float16).float().cpu()
        return self.normalize_latents(latents)

    @torch.no_grad()
    def decode_output(self, normalized_latents: Tensor) -> Tensor:
        """Decode normalized H3 latents into finite base-range RGB frames."""
        latents = self.denormalize_latents(normalized_latents)
        with torch.autocast(
            device_type=self.device.type,
            dtype=torch.float16,
            enabled=self.device.type == "cuda",
        ):
            decoded = self.decode(latents)
        mean = decoded.new_tensor(_PIXEL_MEAN).view(1, -1, 1, 1, 1)
        std = decoded.new_tensor(_PIXEL_STD).view(1, -1, 1, 1, 1)
        pixels = (decoded.float() * std + mean).clamp(0.0, 1.0)
        if not bool(torch.isfinite(pixels).all()):
            raise RuntimeError("Decoded MiniMax H3 video contains non-finite pixels.")
        return pixels

    def forward(
        self,
        sample: Tensor,
        *,
        sample_posterior: bool = False,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        """Encode and decode one ImageNet-normalized video batch."""
        posterior = self.encode(sample).latent_dist
        latents = (
            posterior.sample(generator=generator)
            if sample_posterior
            else posterior.mode()
        )
        return self.decode(latents)


__all__ = [
    "H3_VIDEO_VAE_CHECKPOINT",
    "MiniMaxH3VideoDiagonalGaussianDistribution",
    "MiniMaxH3VideoEncoderOutput",
    "MiniMaxH3VideoVAE",
    "MiniMaxH3VideoVAEConfig",
]
