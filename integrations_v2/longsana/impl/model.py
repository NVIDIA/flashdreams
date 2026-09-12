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

"""Checkpoint-compatible LongSana DiT with a constant-size recurrent state."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from sana_wm.impl.stage1_model import (
    RMSNorm,
    SanaWMStage1Spec,
    Stage1CrossAttention,
    TextEmbedder,
    TimestepEmbedder,
)
from torch import Tensor

from flashdreams.infra.config import InstantiateConfig
from longsana.impl.constants import MAX_ROPE_POSITION

LONGSANA_SPEC = SanaWMStage1Spec(
    latent_channels=16,
    hidden_size=2240,
    text_dim=2304,
    timestep_dim=256,
    depth=20,
    num_heads=20,
    head_dim=112,
    max_text_length=300,
    latent_grid_size=(30, 52),
    mlp_ratio=3,
    temporal_kernel_size=3,
)
"""Architecture of the released LongSana 2B 480p generator."""


@dataclass(kw_only=True)
class LongSanaBlockState:
    """Constant-size recurrent state for one LongSana transformer block."""

    value_key: Tensor | None = None
    """Cumulative rotated value-key product, shaped [B, H, D, D]."""

    key_sum: Tensor | None = None
    """Cumulative unrotated positive key sum, shaped [B, H, 1, D]."""

    conv_tail: Tensor | None = None
    """Last spatial-FFN frame, shaped [B, C, 1, H*W]."""

    def num_bytes(self) -> int:
        """Return bytes currently occupied by this block's recurrent tensors."""
        return sum(
            tensor.numel() * tensor.element_size()
            for tensor in (self.value_key, self.key_sum, self.conv_tail)
            if tensor is not None
        )


@dataclass(kw_only=True)
class LongSanaNetworkConfig(InstantiateConfig):
    """Config for the released LongSana generator architecture."""

    _target: type["LongSanaModel"] = field(default_factory=lambda: LongSanaModel)

    spec: SanaWMStage1Spec = LONGSANA_SPEC
    """Checkpoint-facing dimensions. Tests may substitute a small stand-in."""

    patch_size: tuple[int, int, int] = (1, 2, 2)
    """Temporal, height, and width patch factors."""

    fp32_attention: bool = True
    """Accumulate the rotated linear-attention numerator in float32."""


class _PatchEmbed3D(nn.Module):
    """3D convolutional patch embedder with upstream-compatible parameter names."""

    def __init__(
        self,
        in_channels: int,
        hidden_size: int,
        patch_size: tuple[int, int, int],
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.kernel_size = patch_size
        self.proj = nn.Conv3d(
            in_channels,
            hidden_size,
            kernel_size=patch_size,
            stride=patch_size,
            bias=True,
        )

    def forward(self, x: Tensor) -> Tensor:
        """Project and flatten BCTHW latents into BND tokens."""
        return self.proj(x).flatten(2).transpose(1, 2)


class LongSanaLinearAttention(nn.Module):
    """ReLU-kernel linear attention with absolute RoPE and recurrent sums."""

    def __init__(self, spec: SanaWMStage1Spec, *, fp32_attention: bool) -> None:
        super().__init__()
        self.heads = spec.num_heads
        self.dim = spec.head_dim
        self.eps = 1e-8
        self.fp32_attention = fp32_attention
        self.qkv = nn.Linear(spec.hidden_size, 3 * spec.hidden_size, bias=False)
        self.q_norm = RMSNorm(spec.hidden_size, eps=1e-5)
        self.k_norm = RMSNorm(spec.hidden_size, eps=1e-5)
        self.proj = nn.Linear(spec.hidden_size, spec.hidden_size)

    def forward(
        self,
        x: Tensor,
        *,
        rotary_emb: Tensor,
        state: LongSanaBlockState,
        update_state: bool,
    ) -> Tensor:
        """Apply attention using prior state and optionally commit this block."""
        batch, tokens, channels = x.shape
        if channels != self.heads * self.dim:
            raise ValueError(
                f"channels={channels} != heads*head_dim={self.heads * self.dim}."
            )

        qkv = self.qkv(x).reshape(batch, tokens, 3, channels)
        q, k, v = qkv.unbind(dim=2)
        dtype = q.dtype

        q = self.q_norm(q).transpose(-1, -2)
        k = self.k_norm(k).transpose(-1, -2)
        v = v.transpose(-1, -2)
        q = F.relu(q.reshape(batch, self.heads, self.dim, tokens))
        k = F.relu(k.reshape(batch, self.heads, self.dim, tokens))
        v = v.reshape(batch, self.heads, self.dim, tokens)

        q_rotated = _apply_causal_rope(q, rotary_emb)
        k_rotated = _apply_causal_rope(k, rotary_emb)
        if self.fp32_attention:
            q_rotated = q_rotated.float()
            k_rotated = k_rotated.float()
            v = v.float()

        current_key_sum = k.sum(dim=-1, keepdim=True).transpose(-2, -1)
        current_value_key = torch.matmul(v, k_rotated.transpose(-1, -2))
        total_key_sum = current_key_sum
        total_value_key = current_value_key
        if state.value_key is not None or state.key_sum is not None:
            if state.value_key is None or state.key_sum is None:
                raise RuntimeError(
                    "LongSana attention state is only partially initialized."
                )
            total_value_key = current_value_key + state.value_key
            total_key_sum = current_key_sum + state.key_sum

        denominator = 1.0 / (total_key_sum @ q + self.eps)
        out = torch.matmul(total_value_key, q_rotated)
        out = (out * denominator).to(dtype)
        out = self.proj(out.reshape(batch, channels, tokens).permute(0, 2, 1))

        if update_state:
            _set_or_copy(state, "value_key", total_value_key)
            _set_or_copy(state, "key_sum", total_key_sum)
        return out


class LongSanaCausalGLUMBConvTemp(nn.Module):
    """SANA GLUMB feed-forward layer with one-frame causal temporal state."""

    def __init__(self, spec: SanaWMStage1Spec) -> None:
        super().__init__()
        inner = spec.mlp_inner_size
        gated = spec.gated_mlp_size
        self.inverted_conv = _Conv2dContainer(spec.hidden_size, inner, 1)
        self.depth_conv = _Conv2dContainer(
            inner,
            inner,
            3,
            groups=inner,
            padding=1,
        )
        self.point_conv = _Conv2dContainer(
            gated,
            spec.hidden_size,
            1,
            bias=False,
        )
        self.t_conv = nn.Conv2d(
            spec.hidden_size,
            spec.hidden_size,
            kernel_size=(spec.temporal_kernel_size, 1),
            padding=(spec.temporal_kernel_size // 2, 0),
            bias=False,
        )

    def forward(
        self,
        x: Tensor,
        *,
        frames: int,
        height: int,
        width: int,
        state: LongSanaBlockState,
        update_state: bool,
    ) -> Tensor:
        """Run spatial GLUMB and a causal temporal convolution."""
        batch, tokens, channels = x.shape
        if tokens != frames * height * width:
            raise ValueError(
                f"tokens={tokens} != frames*height*width={frames * height * width}."
            )
        x_2d = x.reshape(batch * frames, height, width, channels).permute(0, 3, 1, 2)
        x_2d = F.silu(self.inverted_conv(x_2d), inplace=True)
        x_2d = self.depth_conv(x_2d)
        value, gate = x_2d.chunk(2, dim=1)
        x_2d = self.point_conv(value * F.silu(gate))

        spatial = x_2d.view(batch, frames, channels, height * width).permute(0, 2, 1, 3)
        padding = int(self.t_conv.kernel_size[0]) // 2
        conv_input = spatial
        prefix = 0
        if state.conv_tail is not None:
            conv_input = torch.cat((state.conv_tail[:, :, -padding:], spatial), dim=2)
            prefix = conv_input.shape[2] - spatial.shape[2]

        temporal = self.t_conv(conv_input)[:, :, prefix:]
        out = spatial + temporal
        if update_state:
            _set_or_copy(state, "conv_tail", spatial[:, :, -padding:])
        return out.permute(0, 2, 3, 1).reshape(batch, tokens, channels)


class _Conv2dContainer(nn.Module):
    """Expose a convolution under the checkpoint-compatible conv name."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        *,
        groups: int = 1,
        bias: bool = True,
        padding: int = 0,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            groups=groups,
            bias=bias,
            padding=padding,
        )

    def forward(self, x: Tensor) -> Tensor:
        """Apply the contained convolution."""
        return self.conv(x)


class LongSanaBlock(nn.Module):
    """One checkpoint-compatible LongSana transformer block."""

    def __init__(self, spec: SanaWMStage1Spec, *, fp32_attention: bool) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(
            spec.hidden_size,
            elementwise_affine=False,
            eps=1e-6,
        )
        self.attn = LongSanaLinearAttention(
            spec,
            fp32_attention=fp32_attention,
        )
        self.cross_attn = Stage1CrossAttention(spec)
        self.norm2 = nn.LayerNorm(
            spec.hidden_size,
            elementwise_affine=False,
            eps=1e-6,
        )
        self.mlp = LongSanaCausalGLUMBConvTemp(spec)
        self.scale_shift_table = nn.Parameter(torch.empty(6, spec.hidden_size))

    def forward(
        self,
        x: Tensor,
        y: Tensor,
        timestep_modulation: Tensor,
        *,
        frames: int,
        height: int,
        width: int,
        mask: Tensor | None,
        rotary_emb: Tensor,
        state: LongSanaBlockState,
        update_state: bool,
    ) -> Tensor:
        """Run self-attention, text cross-attention, and causal GLUMB."""
        batch, _tokens, _channels = x.shape
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.scale_shift_table[None] + timestep_modulation.reshape(batch, 6, -1)
        ).chunk(6, dim=1)

        attn_input = _modulate(self.norm1(x), shift_msa, scale_msa)
        attn_output = self.attn(
            attn_input,
            rotary_emb=rotary_emb,
            state=state,
            update_state=update_state,
        )
        x = x + gate_msa * attn_output
        x = x + self.cross_attn(x, y, mask=mask)

        mlp_input = _modulate(self.norm2(x), shift_mlp, scale_mlp)
        mlp_output = self.mlp(
            mlp_input,
            frames=frames,
            height=height,
            width=width,
            state=state,
            update_state=update_state,
        )
        return x + gate_mlp * mlp_output


class _LongSanaFinalLayer(nn.Module):
    """Final AdaLN and spatial unpatch projection."""

    def __init__(
        self,
        spec: SanaWMStage1Spec,
        patch_size: tuple[int, int, int],
    ) -> None:
        super().__init__()
        self.norm_final = nn.LayerNorm(
            spec.hidden_size,
            elementwise_affine=False,
            eps=1e-6,
        )
        self.linear = nn.Linear(
            spec.hidden_size,
            math.prod(patch_size) * spec.latent_channels,
        )
        self.scale_shift_table = nn.Parameter(torch.empty(2, spec.hidden_size))

    def forward(self, x: Tensor, timestep_embedding: Tensor) -> Tensor:
        """Project hidden tokens into patched latent channels."""
        shift, scale = (
            self.scale_shift_table[None] + timestep_embedding[:, None]
        ).chunk(2, dim=1)
        return self.linear(_modulate(self.norm_final(x), shift, scale))


class LongSanaModel(nn.Module):
    """Released LongSana 2B model with Runtime V2-owned recurrent state."""

    def __init__(self, config: LongSanaNetworkConfig) -> None:
        super().__init__()
        self.config = config
        self.spec = config.spec
        self.patch_size = config.patch_size
        self.register_buffer(
            "pos_embed",
            torch.zeros(1, 1800, self.spec.hidden_size),
        )
        if self.patch_size[0] != 1:
            raise ValueError("LongSana requires temporal patch size 1.")
        self.x_embedder = _PatchEmbed3D(
            self.spec.latent_channels,
            self.spec.hidden_size,
            self.patch_size,
        )
        self.t_embedder = TimestepEmbedder(self.spec)
        self.t_block = nn.Sequential(
            nn.SiLU(),
            nn.Linear(self.spec.hidden_size, 6 * self.spec.hidden_size),
        )
        self.y_embedder = TextEmbedder(self.spec)
        self.attention_y_norm = RMSNorm(self.spec.hidden_size, eps=1e-5)
        with torch.no_grad():
            self.attention_y_norm.weight.fill_(0.01)
        self.blocks = nn.ModuleList(
            [
                LongSanaBlock(
                    self.spec,
                    fp32_attention=config.fp32_attention,
                )
                for _ in range(self.spec.depth)
            ]
        )
        self.final_layer = _LongSanaFinalLayer(self.spec, self.patch_size)

    def prepare_condition(self, condition: Tensor) -> Tensor:
        """Project static Gemma features once for an entire rollout."""
        y = self.y_embedder(condition.to(dtype=self.dtype))
        return self.attention_y_norm(y)

    def forward(
        self,
        x: Tensor,
        timestep: Tensor,
        projected_condition: Tensor,
        condition_mask: Tensor,
        block_states: list[LongSanaBlockState],
        *,
        start_frame: int,
        update_state: bool,
    ) -> Tensor:
        """Predict flow and optionally advance all per-block recurrent states."""
        if x.ndim != 5:
            raise ValueError(
                f"LongSana expects BCTHW input, got shape {tuple(x.shape)}."
            )
        if len(block_states) != len(self.blocks):
            raise ValueError(
                f"Expected {len(self.blocks)} block states, got {len(block_states)}."
            )
        batch, channels, frames, latent_height, latent_width = x.shape
        if channels != self.spec.latent_channels:
            raise ValueError(
                f"Expected {self.spec.latent_channels} latent channels, got {channels}."
            )
        if latent_height % self.patch_size[1] or latent_width % self.patch_size[2]:
            raise ValueError(
                "LongSana latent height and width must be divisible by spatial "
                f"patch size {self.patch_size[1:]}, got {(latent_height, latent_width)}."
            )

        x = x.to(dtype=self.dtype)
        height = latent_height // self.patch_size[1]
        width = latent_width // self.patch_size[2]
        tokens = self.x_embedder(x)
        rope = causal_wan_rope(
            head_dim=self.spec.head_dim,
            start_frame=start_frame,
            frames=frames,
            height=height,
            width=width,
            device=x.device,
        )

        model_timestep = timestep.reshape(()).expand(batch).long().float()
        timestep_embedding = self.t_embedder(model_timestep)
        modulation = self.t_block(timestep_embedding)
        y = projected_condition.to(dtype=self.dtype)
        mask = condition_mask.to(device=x.device)

        for block, state in zip(self.blocks, block_states):
            tokens = block(
                tokens,
                y,
                modulation,
                frames=frames,
                height=height,
                width=width,
                mask=mask,
                rotary_emb=rope,
                state=state,
                update_state=update_state,
            )

        output = self.final_layer(tokens, timestep_embedding)
        return _unpatchify(
            output,
            frames=frames,
            height=height,
            width=width,
            channels=self.spec.latent_channels,
            patch_size=self.patch_size,
        )

    @property
    def dtype(self) -> torch.dtype:
        """Return the checkpoint parameter dtype."""
        return self.x_embedder.proj.weight.dtype


def causal_wan_rope(
    *,
    head_dim: int,
    start_frame: int,
    frames: int,
    height: int,
    width: int,
    device: torch.device,
    max_sequence_length: int = MAX_ROPE_POSITION,
) -> Tensor:
    """Build upstream-compatible complex128 Wan RoPE at absolute frame positions."""
    end_frame = start_frame + frames
    if start_frame < 0 or min(frames, height, width) <= 0:
        raise ValueError(
            "LongSana RoPE dimensions must be positive and start_frame non-negative."
        )
    if max(end_frame, height, width) > max_sequence_length:
        raise ValueError(
            "LongSana RoPE position exceeds the released "
            f"{MAX_ROPE_POSITION}-position table: "
            f"end_frame={end_frame}, height={height}, width={width}."
        )

    temporal_complex = head_dim // 2 - 2 * (head_dim // 6)
    height_complex = head_dim // 6
    width_complex = head_dim // 6
    temporal = _axis_rope(
        max_sequence_length,
        temporal_complex,
        device,
    )[start_frame:end_frame]
    vertical = _axis_rope(max_sequence_length, height_complex, device)[:height]
    horizontal = _axis_rope(max_sequence_length, width_complex, device)[:width]

    temporal = temporal[:, None, None].expand(frames, height, width, -1)
    vertical = vertical[None, :, None].expand(frames, height, width, -1)
    horizontal = horizontal[None, None, :].expand(frames, height, width, -1)
    return torch.cat((temporal, vertical, horizontal), dim=-1).reshape(
        1, 1, frames * height * width, head_dim // 2
    )


def _axis_rope(length: int, complex_dims: int, device: torch.device) -> Tensor:
    if complex_dims == 0:
        return torch.empty(length, 0, dtype=torch.complex128, device=device)
    dim = complex_dims * 2
    positions = torch.arange(length, device=device)
    frequency = 1.0 / (
        10000.0 ** (torch.arange(0, dim, 2, dtype=torch.float64, device=device) / dim)
    )
    angles = torch.outer(positions, frequency)
    return torch.polar(torch.ones_like(angles), angles)


def _apply_causal_rope(hidden_states: Tensor, frequencies: Tensor) -> Tensor:
    """Apply complex RoPE exactly as the released cached attention module."""
    complex_states = torch.view_as_complex(
        hidden_states.permute(0, 1, 3, 2).to(torch.float64).unflatten(3, (-1, 2))
    )
    rotated = torch.view_as_real(complex_states * frequencies)
    return rotated.flatten(3, 4).permute(0, 1, 3, 2).type_as(hidden_states)


def _unpatchify(
    x: Tensor,
    *,
    frames: int,
    height: int,
    width: int,
    channels: int,
    patch_size: tuple[int, int, int],
) -> Tensor:
    batch = x.shape[0]
    pt, ph, pw = patch_size
    expected_tokens = frames * height * width
    if x.shape[1] != expected_tokens:
        raise ValueError(f"Expected {expected_tokens} output tokens, got {x.shape[1]}.")
    x = x.reshape(batch, frames, height, width, pt, ph, pw, channels)
    x = x.permute(0, 7, 1, 4, 2, 5, 3, 6)
    return x.reshape(batch, channels, frames * pt, height * ph, width * pw)


def _modulate(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    return x * (1 + scale) + shift


def _set_or_copy(state: LongSanaBlockState, name: str, value: Tensor) -> None:
    detached = value.detach()
    current = getattr(state, name)
    if current is None:
        setattr(state, name, detached.clone())
        return
    if current.shape != detached.shape or current.dtype != detached.dtype:
        raise ValueError(
            f"LongSana cache slot {name} changed shape or dtype: "
            f"{tuple(current.shape)}/{current.dtype} -> "
            f"{tuple(detached.shape)}/{detached.dtype}."
        )
    current.copy_(detached)
