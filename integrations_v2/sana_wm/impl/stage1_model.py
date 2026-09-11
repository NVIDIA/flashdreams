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

"""SANA-WM Stage-1 DiT module definitions.

The module names and tensor shapes are checkpoint-facing API: keep them stable
unless the public checkpoint contract changes.
"""

from __future__ import annotations

import math
import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

if TYPE_CHECKING:
    import triton
    import triton.language as tl
else:
    try:
        import triton
        import triton.language as tl
    except ImportError:  # pragma: no cover - exercised in minimal CPU environments.
        triton = None
        tl = None


@dataclass(frozen=True)
class SanaWMStage1Spec:
    """Static architecture values for the public SANA-WM bidirectional DiT."""

    latent_channels: int = 128
    hidden_size: int = 2240
    text_dim: int = 2304
    timestep_dim: int = 256
    depth: int = 20
    num_heads: int = 20
    head_dim: int = 112
    max_text_length: int = 300
    latent_grid_size: tuple[int, int] = (22, 40)
    mlp_ratio: int = 3
    conv_kernel_size: int = 4
    temporal_kernel_size: int = 3
    plucker_channels: int = 48
    raymap_channels: int = 3
    softmax_every_n: int = 4
    chunk_size: int | None = None
    chunk_split_strategy: str = "first_chunk_plus_one"

    @property
    def mlp_inner_size(self) -> int:
        """Return the GLUMBConv hidden expansion width."""
        return self.hidden_size * self.mlp_ratio * 2

    @property
    def gated_mlp_size(self) -> int:
        """Return the width consumed by the pointwise output projection."""
        return self.hidden_size * self.mlp_ratio

    def block_uses_gdn(self, index: int) -> bool:
        """Return whether a block has GDN convolution checkpoint tensors."""
        return (index + 1) % self.softmax_every_n != 0


SANA_WM_STAGE1_SPEC = SanaWMStage1Spec()
"""Architecture spec for the public SANA-WM bidirectional Stage-1 checkpoint."""

SANA_WM_STREAMING_STAGE1_SPEC = SanaWMStage1Spec(chunk_size=3)
"""Architecture spec for the public SANA-WM streaming Stage-1 checkpoint."""


def _env_flag_enabled(name: str) -> bool:
    return os.environ.get(name, "").lower() in {"1", "true", "yes", "on"}


_DISABLE_UCPE_FAST = _env_flag_enabled("SANA_WM_STAGE1_DISABLE_UCPE_FAST")
_DISABLE_GLU_FAST = _env_flag_enabled("SANA_WM_STAGE1_DISABLE_GLU_FAST")
_DISABLE_RMS_RELU_FAST = _env_flag_enabled("SANA_WM_STAGE1_DISABLE_RMS_RELU_FAST")


@dataclass(frozen=True)
class _CameraProjectionCache:
    raymats: Tensor
    proj: Tensor
    proj_q: Tensor
    proj_kv: Tensor
    rope_cam: Tensor | None


class RMSNorm(nn.Module):
    """RMSNorm parameter container matching the SANA checkpoint schema."""

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(hidden_size))
        self.eps = eps
        nn.init.ones_(self.weight)

    def forward(self, x: Tensor) -> Tensor:
        """Apply RMS normalization using the stored scale parameter."""
        dtype = x.dtype
        normed = x.float() * torch.rsqrt(
            x.float().pow(2).mean(dim=-1, keepdim=True) + self.eps
        )
        return (normed * self.weight).to(dtype=dtype)


class Conv3dProjector(nn.Module):
    """Named 1x1x1 projection used by latent, ray, and Plucker embedders."""

    def __init__(self, in_channels: int, hidden_size: int) -> None:
        super().__init__()
        self.proj = nn.Conv3d(in_channels, hidden_size, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        """Project a 5D tensor to hidden channels."""
        return self.proj(x)


class TimestepEmbedder(nn.Module):
    """Two-layer timestep embedder with checkpoint-compatible names."""

    def __init__(self, spec: SanaWMStage1Spec) -> None:
        super().__init__()
        self.timestep_dim = int(spec.timestep_dim)
        self.mlp = nn.Sequential(
            nn.Linear(spec.timestep_dim, spec.hidden_size),
            nn.SiLU(),
            nn.Linear(spec.hidden_size, spec.hidden_size),
        )

    def forward(self, t: Tensor) -> Tensor:
        """Project timestep features into the DiT hidden width."""
        first_layer = self.mlp[0]
        if not isinstance(first_layer, nn.Linear):
            raise TypeError(
                "TimestepEmbedder expected the first MLP layer to be Linear."
            )
        t_freq = _timestep_embedding(t.flatten(), self.timestep_dim)
        return self.mlp(t_freq.to(device=t.device, dtype=first_layer.weight.dtype))


class TextProjection(nn.Module):
    """Two-layer text projection matching ``y_embedder.y_proj`` keys."""

    def __init__(self, spec: SanaWMStage1Spec) -> None:
        super().__init__()
        self.fc1 = nn.Linear(spec.text_dim, spec.hidden_size)
        self.act = nn.SiLU()
        self.fc2 = nn.Linear(spec.hidden_size, spec.hidden_size)

    def forward(self, y: Tensor) -> Tensor:
        """Project text encoder activations into the DiT hidden width."""
        return self.fc2(self.act(self.fc1(y)))


class TextEmbedder(nn.Module):
    """Text embedding/projection container matching SANA checkpoint keys."""

    def __init__(self, spec: SanaWMStage1Spec) -> None:
        super().__init__()
        self.y_embedding = nn.Parameter(
            torch.empty(spec.max_text_length, spec.text_dim)
        )
        self.y_proj = TextProjection(spec)

    def forward(self, y: Tensor | None) -> Tensor:
        """Project text embeddings, using the learned null embedding when absent."""
        if y is None:
            y = self.y_embedding.unsqueeze(0)
        if y.ndim == 4 and y.shape[1] == 1:
            y = y.squeeze(1)
        return self.y_proj(y)


class FinalLayer(nn.Module):
    """Final AdaLN and latent-channel projection container."""

    def __init__(self, spec: SanaWMStage1Spec) -> None:
        super().__init__()
        self.norm_final = nn.LayerNorm(
            spec.hidden_size,
            elementwise_affine=False,
            eps=1e-6,
        )
        self.scale_shift_table = nn.Parameter(torch.empty(2, spec.hidden_size))
        self.linear = nn.Linear(spec.hidden_size, spec.latent_channels)

    def forward(self, x: Tensor, t: Tensor, *, frames: int) -> Tensor:
        """Project hidden tokens back to latent channels."""
        if t.ndim > 2:
            batch, tokens, channels = x.shape
            shift, scale = (
                self.scale_shift_table[None, None, :, :] + t.transpose(1, 2)
            ).chunk(2, dim=-2)
            x = _modulate(
                self.norm_final(x).reshape(batch, frames, -1, channels),
                shift,
                scale,
            ).reshape(batch, tokens, channels)
        else:
            shift, scale = (self.scale_shift_table[None] + t[:, None]).chunk(
                2,
                dim=1,
            )
            x = _modulate(self.norm_final(x), shift, scale)
        return self.linear(x)


class Conv2dContainer(nn.Module):
    """Expose a convolution under a stable ``.conv`` attribute."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        *,
        groups: int = 1,
        bias: bool = True,
        padding: int | tuple[int, int] = 0,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            groups=groups,
            bias=bias,
            padding=padding,
        )

    def forward(self, x: Tensor) -> Tensor:
        """Apply the contained convolution."""
        return self.conv(x)


class GLUMBConvTemp(nn.Module):
    """Checkpoint-compatible GLUMBConvTemp feed-forward container."""

    def __init__(self, spec: SanaWMStage1Spec) -> None:
        super().__init__()
        self.inverted_conv = Conv2dContainer(spec.hidden_size, spec.mlp_inner_size, 1)
        self.depth_conv = Conv2dContainer(
            spec.mlp_inner_size,
            spec.mlp_inner_size,
            3,
            groups=spec.mlp_inner_size,
            padding=1,
        )
        self.point_conv = Conv2dContainer(
            spec.gated_mlp_size,
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
        nn.init.zeros_(self.t_conv.weight)

    def forward(self, x: Tensor, *, frames: int, height: int, width: int) -> Tensor:
        """Run spatial GLU plus temporal aggregation."""
        batch, tokens, channels = x.shape
        x_2d = x.reshape(batch * frames, height, width, channels).permute(0, 3, 1, 2)
        x_2d = F.silu(self.inverted_conv(x_2d), inplace=True)
        x_2d = self.depth_conv(x_2d)
        value, gate = x_2d.chunk(2, dim=1)
        x_2d = self.point_conv(
            _silu_multiply(value, gate, inplace=not torch.is_grad_enabled())
        )

        x_time = x_2d.view(batch, frames, channels, height * width).permute(0, 2, 1, 3)
        x_time = x_time + self.t_conv(x_time)
        return x_time.permute(0, 2, 3, 1).reshape(batch, tokens, channels)


class _LinearizedPointwiseConv2d(nn.Module):
    """Run an existing 1x1 ``Conv2dContainer`` through a Linear module."""

    def __init__(self, conv_layer: nn.Module) -> None:
        super().__init__()
        conv = getattr(conv_layer, "conv", None)
        if not isinstance(conv, nn.Conv2d):
            raise ValueError("expected Conv2dContainer.conv to be nn.Conv2d.")
        if (
            conv.kernel_size != (1, 1)
            or conv.stride != (1, 1)
            or conv.padding != (0, 0)
        ):
            raise ValueError("only exact 1x1 pointwise Conv2d can be linearized.")
        if conv.dilation != (1, 1) or conv.groups != 1:
            raise ValueError(
                "grouped or dilated pointwise Conv2d cannot be linearized."
            )
        self.linear = nn.Linear(
            conv.in_channels,
            conv.out_channels,
            bias=conv.bias is not None,
            device=conv.weight.device,
            dtype=conv.weight.dtype,
        )
        with torch.no_grad():
            self.linear.weight.copy_(conv.weight.flatten(1))
            if conv.bias is not None:
                self.linear.bias.copy_(conv.bias)

    def forward(self, x: Tensor) -> Tensor:
        """Apply the pointwise projection while preserving NCHW layout."""
        if x.dim() != 4:
            raise ValueError(f"expected NCHW input, got shape {tuple(x.shape)}.")
        batch, _channels, height, width = x.shape
        x_nhwc = x.permute(0, 2, 3, 1).reshape(batch * height * width, -1)
        y = self.linear(x_nhwc)
        return y.reshape(batch, height, width, -1).permute(0, 3, 1, 2).contiguous()


def linearize_stage1_ffn_for_quant(module: nn.Module) -> tuple[int, int]:
    """Expose Stage-1 pointwise FFN convolutions as Linears for quantization."""
    converted = 0
    skipped = 0
    for child in module.modules():
        if not isinstance(child, GLUMBConvTemp):
            continue
        for attr in ("inverted_conv", "point_conv"):
            pointwise = getattr(child, attr)
            if isinstance(pointwise, _LinearizedPointwiseConv2d):
                continue
            try:
                replacement = _LinearizedPointwiseConv2d(pointwise)
            except ValueError:
                skipped += 1
                continue
            replacement.train(pointwise.training)
            setattr(child, attr, replacement)
            converted += 1
    return converted, skipped


class Stage1SelfAttention(nn.Module):
    """Self/camera attention parameter container for one Stage-1 block."""

    def __init__(self, spec: SanaWMStage1Spec, *, use_gdn_convs: bool) -> None:
        super().__init__()
        self.heads = spec.num_heads
        self.dim = spec.head_dim
        self.eps = 1e-6
        self.use_gdn_convs = use_gdn_convs
        self.patch_size = (1, 1, 1)
        self.A_log = nn.Parameter(torch.empty(spec.num_heads))
        self.beta_proj = nn.Linear(spec.hidden_size, spec.num_heads)
        self.dt_bias = nn.Parameter(torch.empty(spec.num_heads))
        self.gate_proj = nn.Linear(spec.hidden_size, spec.num_heads)
        self.k_norm = RMSNorm(spec.hidden_size)
        self.k_norm_cam = RMSNorm(spec.hidden_size)
        self.k_proj_cam = nn.Linear(spec.hidden_size, spec.hidden_size)
        self.out_proj_cam = nn.Linear(spec.hidden_size, spec.hidden_size)
        self.output_gate = nn.Linear(spec.hidden_size, spec.hidden_size)
        self.proj = nn.Linear(spec.hidden_size, spec.hidden_size)
        self.q_norm = RMSNorm(spec.hidden_size)
        self.q_norm_cam = RMSNorm(spec.hidden_size)
        self.q_proj_cam = nn.Linear(spec.hidden_size, spec.hidden_size)
        self.qkv = nn.Linear(spec.hidden_size, 3 * spec.hidden_size, bias=False)
        self.recall_gate = nn.Parameter(torch.empty(1))
        self.v_proj_cam = nn.Linear(spec.hidden_size, spec.hidden_size)
        if use_gdn_convs:
            self.conv_k = nn.Conv1d(
                spec.hidden_size,
                spec.hidden_size,
                kernel_size=spec.conv_kernel_size,
                groups=spec.hidden_size,
                bias=False,
            )
            self.conv_k_cam = nn.Conv1d(
                spec.hidden_size,
                spec.hidden_size,
                kernel_size=spec.conv_kernel_size,
                groups=spec.hidden_size,
                bias=False,
            )
        nn.init.zeros_(self.A_log)
        nn.init.constant_(self.dt_bias, -5.0)
        nn.init.zeros_(self.recall_gate)

    def forward(
        self,
        x: Tensor,
        *_args: object,
        HW: tuple[int, int, int] | None = None,
        rotary_emb: Tensor | None = None,
        camera_conditions: Tensor | None = None,
        camera_cache: _CameraProjectionCache | None = None,
        apply_output_gate: bool = True,
        **kwargs: object,
    ) -> Tensor:
        """Run SANA-WM self/camera attention."""
        if HW is None:
            raise ValueError("SANA-WM Stage-1 attention requires HW=(T, H, W).")
        batch, tokens, channels = x.shape
        if channels != self.heads * self.dim:
            raise ValueError(
                f"channels={channels} != heads*dim={self.heads * self.dim}"
            )

        precomputed_gates = self._compute_frame_gates(x, HW)
        main_raw = self._forward_gdn_main(
            x,
            HW=HW,
            rotary_emb=rotary_emb,
            precomputed_gates=precomputed_gates,
        )

        cam_contrib: Tensor | int = 0
        if camera_conditions is not None:
            if self.use_gdn_convs:
                cam_raw = self._forward_gdn_camera(
                    x,
                    HW=HW,
                    rotary_emb=rotary_emb,
                    camera_conditions=camera_conditions,
                    camera_cache=camera_cache,
                    precomputed_gates=precomputed_gates,
                )
            else:
                raw_chunk_size = kwargs.get("chunk_size")
                chunk_size = raw_chunk_size if isinstance(raw_chunk_size, int) else None
                raw_chunk_index = kwargs.get("chunk_index")
                chunk_index = (
                    cast(list[int], raw_chunk_index)
                    if isinstance(raw_chunk_index, list)
                    and all(isinstance(index, int) for index in raw_chunk_index)
                    else None
                )
                cam_raw = self._forward_softmax_camera(
                    x,
                    HW=HW,
                    rotary_emb=rotary_emb,
                    camera_conditions=camera_conditions,
                    camera_cache=camera_cache,
                    chunk_size=chunk_size,
                    chunk_split_strategy=str(
                        kwargs.get("chunk_split_strategy", "uniform")
                    ),
                    chunk_index=chunk_index,
                )
            cam_contrib = self.out_proj_cam(cam_raw)

        combined = main_raw + cam_contrib
        if apply_output_gate:
            combined = _apply_output_gate(
                combined,
                x,
                self.output_gate.weight,
                self.output_gate.bias,
            )
            combined = self.proj(combined.to(dtype=self.proj.weight.dtype))
        del kwargs
        return combined

    def _forward_gdn_main(
        self,
        x: Tensor,
        *,
        HW: tuple[int, int, int],
        rotary_emb: Tensor | None,
        precomputed_gates: tuple[Tensor, Tensor],
    ) -> Tensor:
        batch, tokens, channels = x.shape
        qkv = self.qkv(x).reshape(batch, tokens, 3, self.heads, self.dim)
        if hasattr(self, "conv_k"):
            k_raw = qkv[:, :, 1].reshape(batch, tokens, channels)
            k_conv = _apply_bidirectional_temporal_conv(k_raw, self.conv_k, HW)
            qkv[:, :, 1] = k_conv.reshape(batch, tokens, self.heads, self.dim)

        q = _rmsnorm_relu_heads(
            qkv[:, :, 0],
            self.q_norm.weight,
            self.q_norm.eps,
        )
        k = _rmsnorm_relu_heads(
            qkv[:, :, 1],
            self.k_norm.weight,
            self.k_norm.eps,
            scale=_gdn_key_scale(self.dim, HW),
        )
        v = qkv[:, :, 2]
        v = v.float()
        q_rot = _apply_complex_rope(q, rotary_emb)
        k_rot = _apply_complex_rope(k, rotary_emb)
        beta, decay = precomputed_gates
        out = _bidirectional_gdn_scan(
            q=q,
            k=k,
            q_rot=q_rot,
            k_rot=k_rot,
            v=v,
            beta=beta,
            decay=decay,
            HW=HW,
            eps=self.eps,
        )
        return out.reshape(batch, tokens, channels).to(dtype=x.dtype)

    def _forward_softmax_main(
        self,
        x: Tensor,
        *,
        HW: tuple[int, int, int],
        rotary_emb: Tensor | None,
        chunk_size: int | None,
        chunk_split_strategy: str,
        chunk_index: list[int] | None,
    ) -> Tensor:
        batch, tokens, channels = x.shape
        qkv = self.qkv(x).reshape(batch, tokens, 3, self.heads, self.dim)
        q, k, v = qkv.unbind(dim=2)
        q = self.q_norm(q.reshape(batch, tokens, channels)).reshape(
            batch,
            tokens,
            self.heads,
            self.dim,
        )
        k = self.k_norm(k.reshape(batch, tokens, channels)).reshape(
            batch,
            tokens,
            self.heads,
            self.dim,
        )
        q = _apply_complex_rope(q, rotary_emb)
        k = _apply_complex_rope(k, rotary_emb)
        out = _scaled_dot_product_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            HW=HW,
            chunk_size=chunk_size,
            chunk_split_strategy=chunk_split_strategy,
            chunk_index=chunk_index,
        )
        return out.transpose(1, 2).reshape(batch, tokens, channels).to(dtype=x.dtype)

    def _forward_gdn_camera(
        self,
        x: Tensor,
        *,
        HW: tuple[int, int, int],
        rotary_emb: Tensor | None,
        camera_conditions: Tensor,
        camera_cache: _CameraProjectionCache | None,
        precomputed_gates: tuple[Tensor, Tensor],
    ) -> Tensor:
        q_cam, k_cam, v_cam = _camera_qkv(self, x, HW)
        q_trans, k_trans, v_trans, inflation_sq, output_projector = _prepare_ucpe_qkv(
            q_cam,
            k_cam,
            v_cam,
            camera_conditions=camera_conditions,
            HW=HW,
            rotary_emb=rotary_emb,
            camera_cache=camera_cache,
            q_norm_weight=self.q_norm_cam.weight,
            k_norm_weight=self.k_norm_cam.weight,
            norm_eps=self.q_norm_cam.eps,
        )
        beta, decay = precomputed_gates
        frame_inflation = inflation_sq.reshape(
            x.shape[0],
            self.heads,
            HW[0],
            HW[1] * HW[2],
        ).mean(dim=-1)
        beta = beta / frame_inflation.unsqueeze(-1).clamp_min(1.0)
        out = _bidirectional_numerator_scan(
            q=q_trans,
            k=k_trans,
            v=v_trans,
            beta=beta,
            decay=decay,
            HW=HW,
        )
        out = output_projector(out)
        return out.reshape(x.shape[0], x.shape[1], -1).to(dtype=x.dtype)

    def _forward_softmax_camera(
        self,
        x: Tensor,
        *,
        HW: tuple[int, int, int],
        rotary_emb: Tensor | None,
        camera_conditions: Tensor,
        camera_cache: _CameraProjectionCache | None,
        chunk_size: int | None,
        chunk_split_strategy: str,
        chunk_index: list[int] | None,
    ) -> Tensor:
        q_cam, k_cam, v_cam = _camera_qkv(self, x, HW)
        q_trans, k_trans, v_trans, output_projector = _prepare_ucpe_qkv_softmax(
            q_cam,
            k_cam,
            v_cam,
            camera_conditions=camera_conditions,
            HW=HW,
            rotary_emb=rotary_emb,
            camera_cache=camera_cache,
            q_norm_weight=self.q_norm_cam.weight,
            k_norm_weight=self.k_norm_cam.weight,
            norm_eps=self.q_norm_cam.eps,
        )
        out = _scaled_dot_product_attention(
            q_trans.transpose(1, 2),
            k_trans.transpose(1, 2),
            v_trans.transpose(1, 2),
            HW=HW,
            chunk_size=chunk_size,
            chunk_split_strategy=chunk_split_strategy,
            chunk_index=chunk_index,
        )
        return (
            output_projector(out.transpose(1, 2))
            .reshape(
                x.shape[0],
                x.shape[1],
                -1,
            )
            .to(dtype=x.dtype)
        )

    def _compute_frame_gates(
        self,
        x: Tensor,
        HW: tuple[int, int, int],
    ) -> tuple[Tensor, Tensor]:
        batch, tokens, channels = x.shape
        frames, height, width = HW
        spatial = height * width
        if tokens != frames * spatial:
            raise ValueError(f"tokens={tokens} != T*H*W={frames * spatial}")
        beta, decay = _compute_frame_gates(
            x,
            frames,
            spatial,
            self.heads,
            self.beta_proj.weight,
            self.beta_proj.bias,
            self.gate_proj.weight,
            self.gate_proj.bias,
            self.dt_bias,
            self.A_log,
        )
        return beta.float(), decay.float()


class Stage1CrossAttention(nn.Module):
    """Cross-attention parameter container for one Stage-1 block."""

    def __init__(self, spec: SanaWMStage1Spec) -> None:
        super().__init__()
        self.num_heads = spec.num_heads
        self.head_dim = spec.hidden_size // spec.num_heads
        self.k_norm = RMSNorm(spec.hidden_size)
        self.kv_linear = nn.Linear(spec.hidden_size, 2 * spec.hidden_size)
        self.proj = nn.Linear(spec.hidden_size, spec.hidden_size)
        self.q_linear = nn.Linear(spec.hidden_size, spec.hidden_size)
        self.q_norm = RMSNorm(spec.hidden_size)

    def forward(
        self,
        x: Tensor,
        y: Tensor,
        *,
        mask: Tensor | None = None,
        **_kwargs: object,
    ) -> Tensor:
        """Run text cross-attention with checkpoint-compatible Q/K norms."""
        batch, tokens, channels = x.shape
        q = self.q_norm(self.q_linear(x)).view(
            batch,
            tokens,
            self.num_heads,
            self.head_dim,
        )
        kv = self.kv_linear(y).view(batch, -1, 2, channels)
        k, v = kv.unbind(dim=2)
        k = self.k_norm(k).view(batch, -1, self.num_heads, self.head_dim)
        v = v.view(batch, -1, self.num_heads, self.head_dim)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        attn_mask = None
        if mask is not None:
            attn_mask = (1 - mask.to(dtype=q.dtype)) * -10000.0
            if attn_mask.ndim == 2:
                attn_mask = attn_mask[:, None, None].repeat(1, self.num_heads, 1, 1)
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=0.0,
            is_causal=False,
        )
        out = out.transpose(1, 2).reshape(batch, tokens, channels)
        return self.proj(out.to(dtype=self.proj.weight.dtype))


class SanaWMStage1Block(nn.Module):
    """One checkpoint-compatible SANA-WM Stage-1 transformer block."""

    def __init__(self, spec: SanaWMStage1Spec, *, index: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(
            spec.hidden_size,
            elementwise_affine=False,
            eps=1e-6,
        )
        self.norm2 = nn.LayerNorm(
            spec.hidden_size,
            elementwise_affine=False,
            eps=1e-6,
        )
        self.scale_shift_table = nn.Parameter(torch.empty(6, spec.hidden_size))
        self.attn = Stage1SelfAttention(
            spec,
            use_gdn_convs=spec.block_uses_gdn(index),
        )
        self.cross_attn = Stage1CrossAttention(spec)
        self.mlp = GLUMBConvTemp(spec)
        self.plucker_proj = nn.Linear(spec.hidden_size, spec.hidden_size)

    def forward(
        self,
        x: Tensor,
        y: Tensor,
        t: Tensor,
        *,
        frames: int,
        height: int,
        width: int,
        mask: Tensor | None = None,
        plucker_emb: Tensor | None = None,
        **kwargs: object,
    ) -> Tensor:
        """Run one Stage-1 transformer block."""
        batch, tokens, channels = x.shape
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.scale_shift_table[None, None, :, :] + t.reshape(batch, frames, 6, -1)
        ).chunk(6, dim=-2)

        x_norm = self.norm1(x).reshape(batch, frames, -1, channels)
        attn_in = _modulate(x_norm, shift_msa, scale_msa).reshape(
            batch, tokens, channels
        )
        attn_out = self.attn(attn_in, **kwargs).reshape(batch, frames, -1, channels)
        x = x + (gate_msa * attn_out).reshape(batch, tokens, channels)

        if plucker_emb is not None:
            x = x + self.plucker_proj(plucker_emb)

        x = x + self.cross_attn(x, y, mask=mask)

        x_norm = self.norm2(x).reshape(batch, frames, -1, channels)
        mlp_in = _modulate(x_norm, shift_mlp, scale_mlp).reshape(
            batch, tokens, channels
        )
        mlp_out = self.mlp(mlp_in, frames=frames, height=height, width=width)
        mlp_out = mlp_out.reshape(batch, frames, -1, channels)
        return x + (gate_mlp * mlp_out).reshape(batch, tokens, channels)


class SanaWMStage1Model(nn.Module):
    """Checkpoint-compatible Stage-1 SANA-WM DiT shell."""

    def __init__(self, spec: SanaWMStage1Spec = SANA_WM_STAGE1_SPEC) -> None:
        super().__init__()
        self.spec = spec
        self.x_embedder = Conv3dProjector(spec.latent_channels, spec.hidden_size)
        self.t_embedder = TimestepEmbedder(spec)
        self.t_block = nn.Sequential(
            nn.SiLU(), nn.Linear(spec.hidden_size, 6 * spec.hidden_size)
        )
        self.y_embedder = TextEmbedder(spec)
        self.attention_y_norm = RMSNorm(spec.hidden_size)
        self.raymap_embedder = Conv3dProjector(spec.raymap_channels, spec.hidden_size)
        self.plucker_embedder = Conv3dProjector(spec.plucker_channels, spec.hidden_size)
        self.blocks = nn.ModuleList(
            SanaWMStage1Block(spec, index=index) for index in range(spec.depth)
        )
        num_pos_tokens = spec.latent_grid_size[0] * spec.latent_grid_size[1] // 2 + 44
        self.pos_embed = nn.Parameter(torch.empty(1, num_pos_tokens, spec.hidden_size))
        self.final_layer = FinalLayer(spec)

    def forward(
        self,
        x: Tensor,
        timestep: Tensor,
        y: Tensor,
        *,
        mask: Tensor | None = None,
        chunk_plucker: Tensor | None = None,
        **kwargs: object,
    ) -> Tensor:
        """Run the SANA-WM Stage-1 DiT."""
        batch, _channels, frames, height, width = x.shape
        x = self.x_embedder(x)
        x = x.permute(0, 2, 3, 4, 1).reshape(batch, frames * height * width, -1)

        plucker_emb = kwargs.get("chunk_plucker_emb")
        if isinstance(plucker_emb, Tensor):
            if (
                plucker_emb.ndim != 3
                or plucker_emb.shape[1:] != x.shape[1:]
                or plucker_emb.shape[0] not in (1, batch)
            ):
                raise ValueError(
                    "chunk_plucker_emb must have shape [1|B, T*H*W, hidden], "
                    f"got {tuple(plucker_emb.shape)} for x={tuple(x.shape)}."
                )
            plucker_emb = plucker_emb.to(device=x.device, dtype=x.dtype)
        elif chunk_plucker is not None:
            plucker_emb = self.prepare_plucker_embedding(chunk_plucker)

        rotary_emb = kwargs.get("rotary_emb")
        if isinstance(rotary_emb, Tensor):
            rotary_emb = rotary_emb.to(device=x.device)
        else:
            rotary_emb = _wan_rope_complex(
                self.spec.head_dim,
                frames,
                height,
                width,
                x.device,
            )
        raw_camera_conditions = kwargs.get("camera_conditions")
        camera_conditions = (
            raw_camera_conditions.to(device=x.device, dtype=x.dtype)
            if isinstance(raw_camera_conditions, Tensor)
            else None
        )
        camera_cache_value = kwargs.get("camera_cache")
        camera_cache = (
            _camera_projection_cache_to(
                camera_cache_value,
                device=x.device,
                dtype=x.dtype,
            )
            if isinstance(camera_cache_value, _CameraProjectionCache)
            else (
                _prepare_camera_projection_cache(
                    camera_conditions,
                    HW=(frames, height, width),
                    rotary_emb=rotary_emb,
                    head_dim=self.spec.head_dim,
                )
                if camera_conditions is not None
                else None
            )
        )
        camera_signal = (
            camera_cache.raymats if camera_cache is not None else camera_conditions
        )
        block_kwargs = {
            key: value
            for key, value in kwargs.items()
            if key
            not in {
                "camera_conditions",
                "chunk_plucker_emb",
                "camera_cache",
                "rotary_emb",
            }
        }

        y = self.y_embedder(y)
        y = self.attention_y_norm(y)
        if mask is not None and mask.ndim > 2:
            mask = mask.squeeze(1).squeeze(1)

        timestep_embed = self.t_embedder(timestep.flatten())
        timestep_embed = timestep_embed.unflatten(0, timestep.shape)
        block_t = self.t_block(timestep_embed).reshape(
            batch,
            1,
            frames,
            6 * self.spec.hidden_size,
        )

        for block in self.blocks:
            x = block(
                x,
                y,
                block_t,
                frames=frames,
                height=height,
                width=width,
                mask=mask,
                plucker_emb=plucker_emb,
                HW=(frames, height, width),
                rotary_emb=rotary_emb,
                camera_conditions=camera_signal,
                camera_cache=camera_cache,
                chunk_size=self.spec.chunk_size,
                chunk_split_strategy=self.spec.chunk_split_strategy,
                **block_kwargs,
            )

        x = self.final_layer(x, timestep_embed, frames=frames)
        return x.reshape(batch, frames, height, width, -1).permute(0, 4, 1, 2, 3)

    def prepare_plucker_embedding(self, chunk_plucker: Tensor) -> Tensor:
        """Project static Plucker conditioning into Stage-1 token space."""
        batch = chunk_plucker.shape[0]
        plucker_emb = self.plucker_embedder(chunk_plucker)
        return plucker_emb.permute(0, 2, 3, 4, 1).reshape(
            batch,
            -1,
            self.spec.hidden_size,
        )

    def prepare_camera_projection_cache(
        self,
        camera_conditions: Tensor,
        *,
        frames: int,
        height: int,
        width: int,
    ) -> tuple[Tensor, _CameraProjectionCache]:
        """Precompute static RoPE and camera projection tensors for a rollout."""
        param = next(self.parameters())
        camera_conditions = camera_conditions.to(device=param.device, dtype=param.dtype)
        rotary_emb = _wan_rope_complex(
            self.spec.head_dim,
            frames,
            height,
            width,
            param.device,
        )
        camera_cache = _prepare_camera_projection_cache(
            camera_conditions,
            HW=(frames, height, width),
            rotary_emb=rotary_emb,
            head_dim=self.spec.head_dim,
        )
        return rotary_emb, camera_cache


def _modulate(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    return x * (1 + scale) + shift


def _timestep_embedding(t: Tensor, dim: int, max_period: int = 10000) -> Tensor:
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device)
        / half
    )
    args = t[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


def _gdn_key_scale(head_dim: int, HW: tuple[int, int, int]) -> float:
    return (head_dim**-0.5) * ((HW[1] * HW[2]) ** -0.5)


def _sdpa_needs_head_pad(head_dim: int) -> bool:
    return head_dim not in (32, 64, 128, 256) and head_dim < 256


def _scaled_dot_product_attention(
    xq: Tensor,
    xk: Tensor,
    xv: Tensor,
    *,
    HW: tuple[int, int, int] | None = None,
    chunk_size: int | None = None,
    chunk_split_strategy: str = "uniform",
    chunk_index: list[int] | None = None,
) -> Tensor:
    if HW is not None and chunk_size is not None and chunk_size < HW[0]:
        frames, height, width = HW
        spatial = height * width
        boundaries = _normalize_chunk_index(
            chunk_index,
            frames,
            chunk_size,
            chunk_split_strategy,
        )
        out_chunks = []
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            q_chunk = xq[:, :, start * spatial : end * spatial]
            out_chunks.append(
                _scaled_dot_product_attention_full(
                    q_chunk,
                    xk[:, :, : end * spatial],
                    xv[:, :, : end * spatial],
                ),
            )
        return torch.cat(out_chunks, dim=2)
    return _scaled_dot_product_attention_full(xq, xk, xv)


def _scaled_dot_product_attention_full(xq: Tensor, xk: Tensor, xv: Tensor) -> Tensor:
    head_dim = xq.shape[-1]
    if not _sdpa_needs_head_pad(head_dim):
        return F.scaled_dot_product_attention(
            xq,
            xk,
            xv,
            dropout_p=0.0,
            is_causal=False,
        )
    pad_to = 128 if head_dim <= 128 else 256
    pad_size = pad_to - head_dim
    xq_padded = F.pad(xq, (0, pad_size))
    xk_padded = F.pad(xk, (0, pad_size))
    xv_padded = F.pad(xv, (0, pad_size))
    out = F.scaled_dot_product_attention(
        xq_padded,
        xk_padded,
        xv_padded,
        dropout_p=0.0,
        is_causal=False,
        scale=head_dim**-0.5,
    )
    return out[..., :head_dim]


def _normalize_chunk_index(
    chunk_index: list[int] | None,
    frames: int,
    chunk_size: int,
    chunk_split_strategy: str,
) -> list[int]:
    if chunk_index is None:
        chunk_index = _chunk_index_from_chunk_size(
            frames,
            chunk_size,
            chunk_split_strategy,
        )
    else:
        chunk_index = list(chunk_index)
    if not chunk_index or chunk_index[0] != 0:
        chunk_index = [0] + [idx for idx in chunk_index if idx > 0]
    chunk_index = [idx for idx in chunk_index if idx < frames]
    if not chunk_index:
        chunk_index = [0]
    if chunk_index[-1] != frames:
        chunk_index.append(frames)
    return chunk_index


def _chunk_index_from_chunk_size(
    frames: int,
    chunk_size: int,
    chunk_split_strategy: str,
) -> list[int]:
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be > 0, got {chunk_size}.")
    if frames <= 0:
        raise ValueError(f"frames must be > 0, got {frames}.")
    strategy = "uniform" if chunk_split_strategy is None else str(chunk_split_strategy)
    strategy = strategy.lower()
    if strategy in ("uniform", "default"):
        indices = list(range(0, frames, chunk_size))
        if len(indices) > 1 and (frames - indices[-1]) < chunk_size:
            indices.pop()
        return indices
    if strategy in ("first_frame", "first_frame_alone", "first_frame_only"):
        if frames <= 1:
            return [0]
        indices = [0] + list(range(1, frames, chunk_size))
        if len(indices) > 2 and (frames - indices[-1]) < chunk_size:
            indices.pop()
        return indices
    if strategy in ("first_plus_one", "first_chunk_plus_one"):
        if frames <= chunk_size + 1:
            return [0]
        indices = [0] + list(range(chunk_size + 1, frames, chunk_size))
        if len(indices) > 1 and (frames - indices[-1]) < chunk_size:
            indices.pop()
        return indices
    raise ValueError(
        "Unknown chunk_split_strategy "
        f"{chunk_split_strategy!r}; expected uniform, first_frame, or first_plus_one."
    )


def _prepare_camera_projection_cache(
    camera_conditions: Tensor,
    *,
    HW: tuple[int, int, int],
    rotary_emb: Tensor | None,
    head_dim: int,
) -> _CameraProjectionCache:
    batch, frames = camera_conditions.shape[:2]
    latent_frames, height, width = HW
    if frames != latent_frames:
        raise ValueError(
            f"camera_conditions has {frames} frames but latent grid has {latent_frames}."
        )
    tokens = frames * height * width
    raymats = _camera_ray_mats(camera_conditions.float(), HW).to(
        dtype=camera_conditions.dtype,
    )
    proj = raymats.reshape(batch, tokens, 4, 4)
    proj_q = proj.transpose(-1, -2).contiguous()
    proj_kv = _invert_se3(proj).contiguous()
    rope_cam = _slice_rope_for_camera(rotary_emb, head_dim)
    return _CameraProjectionCache(
        raymats=raymats,
        proj=proj,
        proj_q=proj_q,
        proj_kv=proj_kv,
        rope_cam=rope_cam,
    )


def _camera_projection_cache_to(
    cache: _CameraProjectionCache,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> _CameraProjectionCache:
    if (
        cache.raymats.device == device
        and cache.proj.device == device
        and cache.proj_q.device == device
        and cache.proj_kv.device == device
        and cache.raymats.dtype == dtype
        and cache.proj.dtype == dtype
        and cache.proj_q.dtype == dtype
        and cache.proj_kv.dtype == dtype
        and (cache.rope_cam is None or cache.rope_cam.device == device)
    ):
        return cache
    return _CameraProjectionCache(
        raymats=cache.raymats.to(device=device, dtype=dtype),
        proj=cache.proj.to(device=device, dtype=dtype),
        proj_q=cache.proj_q.to(device=device, dtype=dtype),
        proj_kv=cache.proj_kv.to(device=device, dtype=dtype),
        rope_cam=(
            cache.rope_cam.to(device=device) if cache.rope_cam is not None else None
        ),
    )


@torch.compile
def _apply_output_gate(
    out: Tensor,
    gate_x: Tensor,
    gate_weight: Tensor,
    gate_bias: Tensor,
) -> Tensor:
    gate_values = F.silu(F.linear(gate_x, gate_weight, gate_bias).float())
    return out * gate_values.to(dtype=out.dtype)


@torch.compile
def _compute_frame_gates(
    x: Tensor,
    frames: int,
    spatial: int,
    heads: int,
    beta_weight: Tensor,
    beta_bias: Tensor,
    gate_weight: Tensor,
    gate_bias: Tensor,
    dt_bias: Tensor,
    A_log: Tensor,
) -> tuple[Tensor, Tensor]:
    batch, tokens, channels = x.shape
    beta = F.linear(x, beta_weight, beta_bias).sigmoid()
    beta = beta.reshape(batch, frames, spatial, heads).permute(0, 3, 1, 2)
    x_frame = x.reshape(batch, frames, spatial, channels).mean(dim=2)
    gate = F.linear(x_frame, gate_weight, gate_bias).float()
    dt = dt_bias.float().view(1, 1, -1)
    a_val = A_log.float().exp().view(1, 1, -1)
    decay = (-a_val * F.softplus(gate + dt)).exp().transpose(1, 2)
    return beta, decay


if triton is not None and tl is not None:

    @triton.jit
    def _bidirectional_temporal_conv_kernel(
        x_ptr,
        weight_ptr,
        out_ptr,
        x_stride_b,
        x_stride_t,
        x_stride_c,
        weight_stride_c,
        weight_stride_k,
        out_stride_b,
        out_stride_t,
        out_stride_c,
        frames: tl.constexpr,
        spatial: tl.constexpr,
        channels,
        kernel_size: tl.constexpr,
        block_f: tl.constexpr,
        block_s: tl.constexpr,
        block_c: tl.constexpr,
    ) -> None:
        batch_idx = tl.program_id(0)
        spatial_offsets = tl.program_id(1) * block_s + tl.arange(0, block_s)
        channel_offsets = tl.program_id(2) * block_c + tl.arange(0, block_c)
        frame_offsets = tl.arange(0, block_f)

        frame_mask = frame_offsets < frames
        spatial_mask = spatial_offsets < spatial
        channel_mask = channel_offsets < channels
        element_mask = (
            frame_mask[:, None, None]
            & spatial_mask[None, :, None]
            & channel_mask[None, None, :]
        )

        token_offsets = (
            frame_offsets[:, None, None] * spatial
            + spatial_offsets[
                None,
                :,
                None,
            ]
        )
        x_offsets = (
            batch_idx * x_stride_b
            + token_offsets * x_stride_t
            + channel_offsets[None, None, :] * x_stride_c
        )
        out_offsets = (
            batch_idx * out_stride_b
            + token_offsets * out_stride_t
            + channel_offsets[None, None, :] * out_stride_c
        )

        center = tl.load(
            weight_ptr
            + channel_offsets * weight_stride_c
            + (kernel_size - 1) * weight_stride_k,
            mask=channel_mask,
            other=0.0,
        ).to(tl.float32)
        value = tl.load(x_ptr + x_offsets, mask=element_mask, other=0.0).to(tl.float32)
        acc = value * center[None, None, :]

        for offset in range(1, kernel_size):
            coeff = tl.load(
                weight_ptr
                + channel_offsets * weight_stride_c
                + (kernel_size - 1 - offset) * weight_stride_k,
                mask=channel_mask,
                other=0.0,
            ).to(tl.float32)

            prev_frame_offsets = frame_offsets - offset
            prev_token_offsets = (
                prev_frame_offsets[:, None, None] * spatial
                + (spatial_offsets[None, :, None])
            )
            prev_mask = element_mask & (frame_offsets[:, None, None] >= offset)
            prev_offsets = (
                batch_idx * x_stride_b
                + prev_token_offsets * x_stride_t
                + channel_offsets[None, None, :] * x_stride_c
            )
            prev = tl.load(x_ptr + prev_offsets, mask=prev_mask, other=0.0).to(
                tl.float32
            )

            next_frame_offsets = frame_offsets + offset
            next_token_offsets = (
                next_frame_offsets[:, None, None] * spatial
                + (spatial_offsets[None, :, None])
            )
            next_mask = element_mask & (next_frame_offsets[:, None, None] < frames)
            next_offsets = (
                batch_idx * x_stride_b
                + next_token_offsets * x_stride_t
                + channel_offsets[None, None, :] * x_stride_c
            )
            next_value = tl.load(x_ptr + next_offsets, mask=next_mask, other=0.0).to(
                tl.float32
            )
            acc += (prev + next_value) * coeff[None, None, :]

        tl.store(out_ptr + out_offsets, acc, mask=element_mask)

    @triton.jit
    def _ucpe_first_half_kernel(
        x_ptr,
        matrix_ptr,
        out_ptr,
        x_stride_b,
        x_stride_n,
        x_stride_h,
        x_stride_d,
        matrix_stride_b,
        matrix_stride_n,
        matrix_stride_i,
        matrix_stride_j,
        out_stride_b,
        out_stride_n,
        out_stride_h,
        out_stride_d,
        tokens: tl.constexpr,
        groups4: tl.constexpr,
        matrix_batch: tl.constexpr,
        block_n: tl.constexpr,
    ) -> None:
        batch_idx = tl.program_id(0)
        head_idx = tl.program_id(1)
        packed_idx = tl.program_id(2)
        group_idx = packed_idx % groups4
        token_block = packed_idx // groups4
        token_offsets = token_block * block_n + tl.arange(0, block_n)
        token_mask = token_offsets < tokens

        matrix_batch_idx = 0 if matrix_batch == 1 else batch_idx
        col0 = group_idx * 4
        col1 = col0 + 1
        col2 = col0 + 2
        col3 = col0 + 3

        x_base = (
            batch_idx * x_stride_b + token_offsets * x_stride_n + head_idx * x_stride_h
        )
        x0 = tl.load(
            x_ptr + x_base + col0 * x_stride_d,
            mask=token_mask,
            other=0.0,
        ).to(tl.float32)
        x1 = tl.load(
            x_ptr + x_base + col1 * x_stride_d,
            mask=token_mask,
            other=0.0,
        ).to(tl.float32)
        x2 = tl.load(
            x_ptr + x_base + col2 * x_stride_d,
            mask=token_mask,
            other=0.0,
        ).to(tl.float32)
        x3 = tl.load(
            x_ptr + x_base + col3 * x_stride_d,
            mask=token_mask,
            other=0.0,
        ).to(tl.float32)

        matrix_base = (
            matrix_batch_idx * matrix_stride_b + token_offsets * matrix_stride_n
        )
        out_base = (
            batch_idx * out_stride_b
            + token_offsets * out_stride_n
            + head_idx * out_stride_h
        )
        for row in range(4):
            row_base = matrix_base + row * matrix_stride_i
            m0 = tl.load(
                matrix_ptr + row_base + 0 * matrix_stride_j,
                mask=token_mask,
                other=0.0,
            ).to(tl.float32)
            m1 = tl.load(
                matrix_ptr + row_base + 1 * matrix_stride_j,
                mask=token_mask,
                other=0.0,
            ).to(tl.float32)
            m2 = tl.load(
                matrix_ptr + row_base + 2 * matrix_stride_j,
                mask=token_mask,
                other=0.0,
            ).to(tl.float32)
            m3 = tl.load(
                matrix_ptr + row_base + 3 * matrix_stride_j,
                mask=token_mask,
                other=0.0,
            ).to(tl.float32)
            value = (x0 * m0 + x1 * m1) + (x2 * m2 + x3 * m3)
            tl.store(
                out_ptr + out_base + (col0 + row) * out_stride_d,
                value,
                mask=token_mask,
            )

    @triton.jit
    def _ucpe_rope_second_half_kernel(
        x_ptr,
        rope_ptr,
        out_ptr,
        x_stride_b,
        x_stride_n,
        x_stride_h,
        x_stride_d,
        rope_stride_n,
        rope_stride_p,
        rope_stride_ri,
        out_stride_b,
        out_stride_n,
        out_stride_h,
        out_stride_d,
        tokens: tl.constexpr,
        half: tl.constexpr,
        rope_pairs: tl.constexpr,
        inverse_rope: tl.constexpr,
        block_n: tl.constexpr,
    ) -> None:
        batch_idx = tl.program_id(0)
        head_idx = tl.program_id(1)
        packed_idx = tl.program_id(2)
        pair_idx = packed_idx % rope_pairs
        token_block = packed_idx // rope_pairs
        token_offsets = token_block * block_n + tl.arange(0, block_n)
        token_mask = token_offsets < tokens

        real_d = half + pair_idx * 2
        imag_d = real_d + 1
        x_base = (
            batch_idx * x_stride_b + token_offsets * x_stride_n + head_idx * x_stride_h
        )
        real = tl.load(
            x_ptr + x_base + real_d * x_stride_d,
            mask=token_mask,
            other=0.0,
        ).to(tl.float32)
        imag = tl.load(
            x_ptr + x_base + imag_d * x_stride_d,
            mask=token_mask,
            other=0.0,
        ).to(tl.float32)
        rope_base = token_offsets * rope_stride_n + pair_idx * rope_stride_p
        rope_real = tl.load(
            rope_ptr + rope_base + 0 * rope_stride_ri,
            mask=token_mask,
            other=0.0,
        ).to(tl.float32)
        rope_imag = tl.load(
            rope_ptr + rope_base + 1 * rope_stride_ri,
            mask=token_mask,
            other=0.0,
        ).to(tl.float32)
        if inverse_rope:
            rope_imag = -rope_imag

        out_base = (
            batch_idx * out_stride_b
            + token_offsets * out_stride_n
            + head_idx * out_stride_h
        )
        tl.store(
            out_ptr + out_base + real_d * out_stride_d,
            real * rope_real - imag * rope_imag,
            mask=token_mask,
        )
        tl.store(
            out_ptr + out_base + imag_d * out_stride_d,
            real * rope_imag + imag * rope_real,
            mask=token_mask,
        )

    @triton.jit
    def _ucpe_norms_kernel(
        x_ptr,
        out_ptr,
        pre_ptr,
        post_ptr,
        x_stride_b,
        x_stride_n,
        x_stride_h,
        x_stride_d,
        out_stride_b,
        out_stride_n,
        out_stride_h,
        out_stride_d,
        pre_stride_b,
        pre_stride_h,
        pre_stride_n,
        post_stride_b,
        post_stride_h,
        post_stride_n,
        tokens: tl.constexpr,
        dim: tl.constexpr,
        block_n: tl.constexpr,
        block_d: tl.constexpr,
    ) -> None:
        batch_idx = tl.program_id(0)
        head_idx = tl.program_id(1)
        token_block = tl.program_id(2)
        token_offsets = token_block * block_n + tl.arange(0, block_n)
        dim_offsets = tl.arange(0, block_d)
        mask = (token_offsets[:, None] < tokens) & (dim_offsets[None, :] < dim)

        x_offsets = (
            batch_idx * x_stride_b
            + token_offsets[:, None] * x_stride_n
            + head_idx * x_stride_h
            + dim_offsets[None, :] * x_stride_d
        )
        out_offsets = (
            batch_idx * out_stride_b
            + token_offsets[:, None] * out_stride_n
            + head_idx * out_stride_h
            + dim_offsets[None, :] * out_stride_d
        )
        x_values = tl.load(x_ptr + x_offsets, mask=mask, other=0.0).to(tl.float32)
        out_values = tl.load(out_ptr + out_offsets, mask=mask, other=0.0).to(tl.float32)
        pre = tl.sum(x_values * x_values, axis=1)
        post = tl.sum(out_values * out_values, axis=1)
        token_mask = token_offsets < tokens
        pre_offsets = (
            batch_idx * pre_stride_b
            + head_idx * pre_stride_h
            + token_offsets * pre_stride_n
        )
        post_offsets = (
            batch_idx * post_stride_b
            + head_idx * post_stride_h
            + token_offsets * post_stride_n
        )
        tl.store(pre_ptr + pre_offsets, pre, mask=token_mask)
        tl.store(post_ptr + post_offsets, post, mask=token_mask)

    @triton.jit
    def _silu_multiply_kernel(
        value_ptr,
        gate_ptr,
        out_ptr,
        value_stride_n,
        value_stride_c,
        value_stride_h,
        value_stride_w,
        gate_stride_n,
        gate_stride_c,
        gate_stride_h,
        gate_stride_w,
        channels: tl.constexpr,
        height: tl.constexpr,
        width: tl.constexpr,
        total: tl.constexpr,
        block: tl.constexpr,
    ) -> None:
        offsets = tl.program_id(0) * block + tl.arange(0, block)
        mask = offsets < total
        channel_offsets = offsets % channels
        spatial_offsets = offsets // channels
        width_offsets = spatial_offsets % width
        height_offsets = (spatial_offsets // width) % height
        batch_offsets = spatial_offsets // (height * width)

        value_offsets = (
            batch_offsets * value_stride_n
            + channel_offsets * value_stride_c
            + height_offsets * value_stride_h
            + width_offsets * value_stride_w
        )
        gate_offsets = (
            batch_offsets * gate_stride_n
            + channel_offsets * gate_stride_c
            + height_offsets * gate_stride_h
            + width_offsets * gate_stride_w
        )
        value = tl.load(value_ptr + value_offsets, mask=mask, other=0.0).to(tl.float32)
        gate = tl.load(gate_ptr + gate_offsets, mask=mask, other=0.0).to(tl.float32)
        silu_gate = gate / (1.0 + tl.exp(-gate))
        tl.store(out_ptr + offsets, value * silu_gate, mask=mask)

    @triton.jit
    def _rmsnorm_relu_heads_kernel(
        x_ptr,
        weight_ptr,
        out_ptr,
        x_stride_b,
        x_stride_n,
        x_stride_h,
        x_stride_d,
        out_stride_b,
        out_stride_n,
        out_stride_h,
        out_stride_d,
        tokens: tl.constexpr,
        heads: tl.constexpr,
        dim: tl.constexpr,
        channels: tl.constexpr,
        eps: tl.constexpr,
        scale: tl.constexpr,
        block_c: tl.constexpr,
    ) -> None:
        row = tl.program_id(0)
        batch_idx = row // tokens
        token_idx = row - batch_idx * tokens
        channel_offsets = tl.arange(0, block_c)
        mask = channel_offsets < channels
        head_offsets = channel_offsets // dim
        dim_offsets = channel_offsets - head_offsets * dim

        x_offsets = (
            batch_idx * x_stride_b
            + token_idx * x_stride_n
            + head_offsets * x_stride_h
            + dim_offsets * x_stride_d
        )
        values = tl.load(x_ptr + x_offsets, mask=mask, other=0.0).to(tl.float32)
        mean_sq = tl.sum(values * values, axis=0) / channels
        inv_rms = tl.rsqrt(mean_sq + eps)
        weights = tl.load(weight_ptr + channel_offsets, mask=mask, other=0.0).to(
            tl.float32
        )
        out_values = tl.maximum(values * inv_rms * weights * scale, 0.0)
        out_offsets = (
            batch_idx * out_stride_b
            + token_idx * out_stride_n
            + head_offsets * out_stride_h
            + dim_offsets * out_stride_d
        )
        tl.store(out_ptr + out_offsets, out_values, mask=mask)

else:
    _bidirectional_temporal_conv_kernel = None
    _ucpe_first_half_kernel = None
    _ucpe_rope_second_half_kernel = None
    _ucpe_norms_kernel = None
    _silu_multiply_kernel = None
    _rmsnorm_relu_heads_kernel = None


def _apply_bidirectional_temporal_conv_fast(
    x: Tensor,
    conv: nn.Conv1d,
    HW: tuple[int, int, int],
) -> Tensor:
    batch, tokens, channels = x.shape
    frames, height, width = HW
    spatial = height * width
    out = torch.empty_like(x)
    assert triton is not None
    assert _bidirectional_temporal_conv_kernel is not None
    block_f = triton.next_power_of_2(frames)
    block_s = 4
    block_c = 128
    grid = (batch, triton.cdiv(spatial, block_s), triton.cdiv(channels, block_c))
    weight = conv.weight[:, 0, :]
    _bidirectional_temporal_conv_kernel[grid](
        x,
        weight,
        out,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        weight.stride(0),
        weight.stride(1),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        frames,
        spatial,
        channels,
        int(conv.kernel_size[0]),
        block_f,
        block_s,
        block_c,
    )
    return out


def _apply_bidirectional_temporal_conv_eager(
    x: Tensor,
    conv: nn.Conv1d,
    HW: tuple[int, int, int],
) -> Tensor:
    batch, tokens, channels = x.shape
    frames, height, width = HW
    spatial = height * width
    if tokens != frames * spatial:
        raise ValueError(f"tokens={tokens} != T*H*W={frames * spatial}")
    temporal = (
        x.reshape(batch, frames, spatial, channels)
        .permute(0, 2, 3, 1)
        .reshape(batch * spatial, channels, frames)
    )
    y_fwd = _causal_depthwise_conv(temporal, conv)
    y_bwd = _causal_depthwise_conv(temporal.flip(-1), conv).flip(-1)
    center = conv.weight[:, 0, -1].view(1, channels, 1)
    out = y_fwd + y_bwd - temporal * center
    return (
        out.reshape(batch, spatial, channels, frames)
        .permute(0, 3, 1, 2)
        .reshape(batch, tokens, channels)
        .to(dtype=x.dtype)
    )


def _apply_bidirectional_temporal_conv(
    x: Tensor,
    conv: nn.Conv1d,
    HW: tuple[int, int, int],
) -> Tensor:
    if (
        x.is_cuda
        and conv.weight.is_cuda
        and conv.weight.dtype == x.dtype
        and conv.bias is None
        and conv.in_channels == x.shape[-1]
        and conv.out_channels == x.shape[-1]
        and conv.groups == x.shape[-1]
        and conv.stride == (1,)
        and conv.padding == (0,)
        and conv.dilation == (1,)
        and not torch.is_grad_enabled()
        and triton is not None
        and _bidirectional_temporal_conv_kernel is not None
    ):
        batch, tokens, _channels = x.shape
        frames, height, width = HW
        if tokens == frames * height * width:
            return _apply_bidirectional_temporal_conv_fast(x, conv, HW)
    return _apply_bidirectional_temporal_conv_eager(x, conv, HW)


def _causal_depthwise_conv(x: Tensor, conv: nn.Conv1d) -> Tensor:
    kernel = int(conv.kernel_size[0])
    padded = F.pad(x, (kernel - 1, 0))
    return F.conv1d(
        padded,
        conv.weight.to(dtype=x.dtype),
        bias=None,
        stride=1,
        padding=0,
        dilation=1,
        groups=conv.groups,
    )


def _wan_rope_complex(
    head_dim: int,
    frames: int,
    height: int,
    width: int,
    device: torch.device,
) -> Tensor:
    t_size = head_dim // 2 - 2 * (head_dim // 6)
    h_size = head_dim // 6
    w_size = head_dim // 6
    freqs_t = _axis_rope_complex(frames, t_size, device)
    freqs_h = _axis_rope_complex(height, h_size, device)
    freqs_w = _axis_rope_complex(width, w_size, device)
    expanded_t = freqs_t[:, None, None, :].expand(frames, height, width, t_size)
    expanded_h = freqs_h[None, :, None, :].expand(frames, height, width, h_size)
    expanded_w = freqs_w[None, None, :, :].expand(frames, height, width, w_size)
    freqs = torch.cat([expanded_t, expanded_h, expanded_w], dim=-1)
    return freqs.reshape(1, 1, frames * height * width, -1)


def _axis_rope_complex(length: int, complex_dims: int, device: torch.device) -> Tensor:
    if complex_dims == 0:
        return torch.empty(length, 0, dtype=torch.complex64, device=device)
    dim = complex_dims * 2
    exponent = torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim
    freqs = 1.0 / (10000.0**exponent)
    positions = torch.arange(length, dtype=torch.float32, device=device)
    angles = positions[:, None] * freqs[None]
    return torch.polar(torch.ones_like(angles), angles)


def _apply_complex_rope(x: Tensor, rotary_emb: Tensor | None) -> Tensor:
    if rotary_emb is None:
        return x
    batch, tokens, heads, dim = x.shape
    freqs = rotary_emb.squeeze(0).squeeze(0)
    if freqs.shape[0] != tokens or freqs.shape[1] != dim // 2:
        raise ValueError(
            f"RoPE shape {tuple(freqs.shape)} is incompatible with {(tokens, dim)}."
        )
    x_float = x.float()
    x_complex = torch.view_as_complex(
        x_float.reshape(batch, tokens, heads, dim // 2, 2)
    )
    rotated = torch.view_as_real(x_complex * freqs[None, :, None, :]).flatten(-2)
    return rotated.to(dtype=x.dtype)


def _slice_rope_for_camera(rotary_emb: Tensor | None, head_dim: int) -> Tensor | None:
    if rotary_emb is None:
        return None
    orig_t_size = head_dim // 2 - 2 * (head_dim // 6)
    orig_h_size = head_dim // 6
    new_head_dim = head_dim // 2
    new_t_size = new_head_dim // 2 - 2 * (new_head_dim // 6)
    new_h_size = new_head_dim // 6
    new_w_size = new_head_dim // 6
    t_part = rotary_emb[..., :new_t_size]
    h_part = rotary_emb[..., orig_t_size : orig_t_size + new_h_size]
    w_part = rotary_emb[
        ...,
        orig_t_size + orig_h_size : orig_t_size + orig_h_size + new_w_size,
    ]
    return torch.cat([t_part, h_part, w_part], dim=-1)


def _bidirectional_gdn_scan(
    *,
    q: Tensor,
    k: Tensor,
    q_rot: Tensor,
    k_rot: Tensor,
    v: Tensor,
    beta: Tensor,
    decay: Tensor,
    HW: tuple[int, int, int],
    eps: float,
) -> Tensor:
    m_hist, z_hist = _gdn_histories(
        k=k,
        k_rot=k_rot,
        v=v,
        beta=beta,
        decay=decay,
        HW=HW,
        include_denominator=True,
    )
    batch, tokens, heads, dim = q.shape
    frames, height, width = HW
    spatial = height * width
    q = q.reshape(batch, frames, spatial, heads, dim).permute(0, 3, 1, 2, 4)
    q_rot = q_rot.reshape(batch, frames, spatial, heads, dim).permute(
        0,
        3,
        1,
        2,
        4,
    )
    num = torch.einsum("bhfsd,bhfde->bhfse", q_rot.float(), m_hist)
    den = torch.einsum("bhfsd,bhfd->bhfs", q.float(), z_hist)
    out = num / (den[..., None] + eps)
    return out.permute(0, 2, 3, 1, 4).reshape(batch, tokens, heads, dim)


def _bidirectional_numerator_scan(
    *,
    q: Tensor,
    k: Tensor,
    v: Tensor,
    beta: Tensor,
    decay: Tensor,
    HW: tuple[int, int, int],
) -> Tensor:
    m_hist, _z_hist = _gdn_histories(
        k=k,
        k_rot=k,
        v=v,
        beta=beta,
        decay=decay,
        HW=HW,
        include_denominator=False,
    )
    batch, tokens, heads, dim = q.shape
    frames, height, width = HW
    spatial = height * width
    q = q.reshape(batch, frames, spatial, heads, dim).permute(0, 3, 1, 2, 4)
    out = torch.einsum("bhfsd,bhfde->bhfse", q.float(), m_hist)
    return out.permute(0, 2, 3, 1, 4).reshape(batch, tokens, heads, dim)


def _gdn_histories(
    *,
    k: Tensor,
    k_rot: Tensor,
    v: Tensor,
    beta: Tensor,
    decay: Tensor,
    HW: tuple[int, int, int],
    include_denominator: bool,
) -> tuple[Tensor, Tensor | None]:
    batch, tokens, heads, dim = k.shape
    frames, height, width = HW
    spatial = height * width
    k = k.reshape(batch, frames, spatial, heads, dim).permute(0, 3, 1, 2, 4).float()
    k_rot = (
        k_rot.reshape(batch, frames, spatial, heads, dim).permute(0, 3, 1, 2, 4).float()
    )
    v = v.reshape(batch, frames, spatial, heads, dim).permute(0, 3, 1, 2, 4).float()
    eye = torch.eye(dim, dtype=torch.float32, device=k.device).view(1, 1, 1, dim, dim)

    # The per-frame GDN transition is linear in the running state, so the
    # spatial reductions that build each frame's transition are batched over all
    # frames in one einsum, leaving only the (light) sequential state scan as a
    # loop — run forward and reverse to form the bidirectional history.
    p_kv = torch.einsum("bhfsd,bhfse->bhfde", k_rot, beta[..., None] * k_rot)
    a_kv = torch.einsum("bhfsd,bhfse->bhfde", k_rot, beta[..., None] * v)
    trans_m = decay[..., None, None] * (eye - p_kv)
    m_state = torch.zeros(batch, heads, dim, dim, dtype=torch.float32, device=k.device)
    m_forward: list[Tensor] = []
    for frame in range(frames):
        m_state = torch.einsum("bhde,bhef->bhdf", trans_m[:, :, frame], m_state)
        m_state = m_state + a_kv[:, :, frame]
        m_forward.append(m_state)
    m_hist = torch.stack(m_forward, dim=2)
    m_state = torch.zeros(batch, heads, dim, dim, dtype=torch.float32, device=k.device)
    for src in range(frames - 1, 0, -1):
        m_state = torch.einsum("bhde,bhef->bhdf", trans_m[:, :, src], m_state)
        m_state = m_state + a_kv[:, :, src]
        m_hist[:, :, src - 1] = m_hist[:, :, src - 1] + m_state

    z_hist = None
    if include_denominator:
        beta_k = beta[..., None] * k
        p_z = torch.einsum("bhfsd,bhfse->bhfde", k, beta_k)
        b_z = beta_k.sum(dim=3)
        trans_z = decay[..., None, None] * (eye - p_z)
        z_state = torch.zeros(batch, heads, dim, dtype=torch.float32, device=k.device)
        z_forward: list[Tensor] = []
        for frame in range(frames):
            z_state = torch.einsum("bhde,bhe->bhd", trans_z[:, :, frame], z_state)
            z_state = z_state + b_z[:, :, frame]
            z_forward.append(z_state)
        z_hist = torch.stack(z_forward, dim=2)
        z_state = torch.zeros(batch, heads, dim, dtype=torch.float32, device=k.device)
        for src in range(frames - 1, 0, -1):
            z_state = torch.einsum("bhde,bhe->bhd", trans_z[:, :, src], z_state)
            z_state = z_state + b_z[:, :, src]
            z_hist[:, :, src - 1] = z_hist[:, :, src - 1] + z_state

    return m_hist, z_hist


def _camera_qkv(
    module: Stage1SelfAttention,
    x: Tensor,
    HW: tuple[int, int, int],
) -> tuple[Tensor, Tensor, Tensor]:
    batch, tokens, _channels = x.shape
    qkv_weight = torch.cat(
        [
            module.q_proj_cam.weight,
            module.k_proj_cam.weight,
            module.v_proj_cam.weight,
        ],
    )
    qkv_bias = torch.cat(
        [
            module.q_proj_cam.bias,
            module.k_proj_cam.bias,
            module.v_proj_cam.bias,
        ],
    )
    q_raw_flat, k_raw_flat, v_raw_flat = F.linear(x, qkv_weight, qkv_bias).chunk(
        3,
        dim=-1,
    )
    if hasattr(module, "conv_k_cam"):
        k_raw_flat = _apply_bidirectional_temporal_conv(
            k_raw_flat, module.conv_k_cam, HW
        )
    q_raw = q_raw_flat.reshape(batch, tokens, module.heads, module.dim)
    k_raw = k_raw_flat.reshape(batch, tokens, module.heads, module.dim)
    v_raw = v_raw_flat.reshape(batch, tokens, module.heads, module.dim)
    return q_raw, k_raw, v_raw


def _prepare_ucpe_qkv(
    q_raw: Tensor,
    k_raw: Tensor,
    v_raw: Tensor,
    *,
    camera_conditions: Tensor,
    HW: tuple[int, int, int],
    rotary_emb: Tensor | None,
    camera_cache: _CameraProjectionCache | None,
    q_norm_weight: Tensor,
    k_norm_weight: Tensor,
    norm_eps: float,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Callable[[Tensor], Tensor]]:
    batch, tokens, heads, dim = q_raw.shape
    q_norm = _rmsnorm_relu_heads(q_raw, q_norm_weight, norm_eps)
    k_norm = _rmsnorm_relu_heads(
        k_raw,
        k_norm_weight,
        norm_eps,
        scale=_gdn_key_scale(dim, HW),
    )
    v_float = v_raw.float()
    if camera_cache is None:
        camera_cache = _prepare_camera_projection_cache(
            camera_conditions,
            HW=HW,
            rotary_emb=rotary_emb,
            head_dim=dim,
        )

    q_trans = _ucpe_transform_apply(
        q_norm,
        camera_cache.proj_q,
        camera_cache.rope_cam,
        inverse_rope=False,
    )
    k_trans, k_pre_sq, k_post_sq = _ucpe_transform(
        k_norm,
        camera_cache.proj_kv,
        camera_cache.rope_cam,
        inverse_rope=False,
    )
    v_trans = _ucpe_transform_apply(
        v_float,
        camera_cache.proj_kv,
        camera_cache.rope_cam,
        inverse_rope=False,
    )
    inflation_sq = k_post_sq.clamp_min(1e-12) / k_pre_sq.clamp_min(1e-12)

    def output_projector(out: Tensor) -> Tensor:
        projected = _ucpe_transform_apply(
            out.float(),
            camera_cache.proj,
            camera_cache.rope_cam,
            inverse_rope=True,
        )
        return projected.to(dtype=out.dtype)

    return q_trans, k_trans, v_trans, inflation_sq, output_projector


def _prepare_ucpe_qkv_softmax(
    q_raw: Tensor,
    k_raw: Tensor,
    v_raw: Tensor,
    *,
    camera_conditions: Tensor,
    HW: tuple[int, int, int],
    rotary_emb: Tensor | None,
    camera_cache: _CameraProjectionCache | None,
    q_norm_weight: Tensor,
    k_norm_weight: Tensor,
    norm_eps: float,
) -> tuple[Tensor, Tensor, Tensor, Callable[[Tensor], Tensor]]:
    batch, tokens, heads, dim = q_raw.shape
    q_inv = _inv_rms(q_raw, norm_eps)
    k_inv = _inv_rms(k_raw, norm_eps)
    q_weight = q_norm_weight.float().view(heads, dim)
    k_weight = k_norm_weight.float().view(heads, dim)
    q_norm = (q_raw.float() * q_inv[:, :, None, None] * q_weight[None, None]).to(
        dtype=q_raw.dtype
    )
    k_norm = (k_raw.float() * k_inv[:, :, None, None] * k_weight[None, None]).to(
        dtype=k_raw.dtype
    )
    if camera_cache is None:
        camera_cache = _prepare_camera_projection_cache(
            camera_conditions,
            HW=HW,
            rotary_emb=rotary_emb,
            head_dim=dim,
        )
    q_trans = _ucpe_transform_apply(
        q_norm,
        camera_cache.proj_q,
        camera_cache.rope_cam,
        inverse_rope=False,
    ).to(dtype=q_raw.dtype)
    kv_trans = _ucpe_transform_apply(
        torch.cat([k_norm, v_raw], dim=2),
        camera_cache.proj_kv,
        camera_cache.rope_cam,
        inverse_rope=False,
    ).to(dtype=q_raw.dtype)
    k_trans, v_trans = kv_trans.chunk(2, dim=2)

    def output_projector(out: Tensor) -> Tensor:
        projected = _ucpe_transform_apply(
            out.to(dtype=q_raw.dtype),
            camera_cache.proj,
            camera_cache.rope_cam,
            inverse_rope=True,
        )
        return projected.to(dtype=out.dtype)

    return q_trans, k_trans, v_trans, output_projector


def _inv_rms(x: Tensor, eps: float) -> Tensor:
    batch, tokens, heads, dim = x.shape
    channels = heads * dim
    return torch.rsqrt(x.float().pow(2).sum(dim=(-1, -2)) / channels + eps)


def _rmsnorm_relu_heads(
    x: Tensor,
    weight: Tensor,
    eps: float,
    *,
    scale: float = 1.0,
) -> Tensor:
    fast = _rmsnorm_relu_heads_fast(x, weight, eps, scale=scale)
    if fast is not None:
        return fast
    batch, tokens, heads, dim = x.shape
    inv_rms = _inv_rms(x, eps)
    norm_weight = weight.float().view(heads, dim)
    out = F.relu(x.float() * inv_rms[:, :, None, None] * norm_weight[None, None])
    if scale != 1.0:
        out = out * scale
    return out


def _rmsnorm_relu_heads_fast(
    x: Tensor,
    weight: Tensor,
    eps: float,
    *,
    scale: float,
) -> Tensor | None:
    batch, tokens, heads, dim = x.shape
    channels = heads * dim
    block_c = 1 << (channels - 1).bit_length()
    if (
        _DISABLE_RMS_RELU_FAST
        or not x.is_cuda
        or not weight.is_cuda
        or torch.is_grad_enabled()
        or triton is None
        or _rmsnorm_relu_heads_kernel is None
        or weight.numel() != channels
        or block_c > 4096
    ):
        return None
    out = torch.empty(
        (batch, tokens, heads, dim),
        device=x.device,
        dtype=torch.float32,
    )
    grid = (batch * tokens,)
    _rmsnorm_relu_heads_kernel[grid](
        x,
        weight,
        out,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        x.stride(3),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out.stride(3),
        tokens,
        heads,
        dim,
        channels,
        float(eps),
        float(scale),
        block_c,
    )
    return out


def _ucpe_transform(
    x: Tensor,
    matrix: Tensor,
    rotary_emb: Tensor | None,
    *,
    inverse_rope: bool,
) -> tuple[Tensor, Tensor, Tensor]:
    out = _ucpe_transform_apply(x, matrix, rotary_emb, inverse_rope=inverse_rope)
    fast_norms = _ucpe_transform_norms_fast(x, out)
    if fast_norms is not None:
        pre_sq, post_sq = fast_norms
        return out, pre_sq, post_sq
    pre_sq = _ucpe_transform_norms_eager(x)
    post_sq = _ucpe_transform_norms_eager(out)
    return out, pre_sq, post_sq


def _ucpe_transform_norms_eager(x: Tensor) -> Tensor:
    return x.float().pow(2).sum(dim=-1).transpose(1, 2).contiguous()


def _ucpe_transform_apply(
    x: Tensor,
    matrix: Tensor,
    rotary_emb: Tensor | None,
    *,
    inverse_rope: bool,
) -> Tensor:
    fast = _ucpe_transform_apply_fast(
        x,
        matrix,
        rotary_emb,
        inverse_rope=inverse_rope,
    )
    if fast is not None:
        return fast
    return _ucpe_transform_apply_eager(
        x,
        matrix,
        rotary_emb,
        inverse_rope=inverse_rope,
    )


def _ucpe_transform_apply_eager(
    x: Tensor,
    matrix: Tensor,
    rotary_emb: Tensor | None,
    *,
    inverse_rope: bool,
) -> Tensor:
    batch, tokens, heads, dim = x.shape
    half = dim // 2
    if half % 4 != 0:
        raise ValueError(f"UCPE requires head_dim/2 divisible by 4, got {half}.")
    first = x[..., :half].reshape(batch, tokens, heads, half // 4, 4)
    first_out = torch.einsum("bnij,bnhgj->bnhgi", matrix.float(), first.float())
    first_out = first_out.reshape(batch, tokens, heads, half)
    second = x[..., half:]
    if inverse_rope and rotary_emb is not None:
        second_out = _apply_complex_rope(second, rotary_emb.conj())
    else:
        second_out = _apply_complex_rope(second, rotary_emb)
    out = torch.cat([first_out, second_out], dim=-1)
    return out


def _ucpe_transform_apply_fast(
    x: Tensor,
    matrix: Tensor,
    rotary_emb: Tensor | None,
    *,
    inverse_rope: bool,
) -> Tensor | None:
    batch, tokens, heads, dim = x.shape
    half = dim // 2
    if half % 4 != 0:
        raise ValueError(f"UCPE requires head_dim/2 divisible by 4, got {half}.")
    if (
        _DISABLE_UCPE_FAST
        or not x.is_cuda
        or not matrix.is_cuda
        or rotary_emb is None
        or not rotary_emb.is_cuda
        or torch.is_grad_enabled()
        or triton is None
        or _ucpe_first_half_kernel is None
        or _ucpe_rope_second_half_kernel is None
        or matrix.shape[-2:] != (4, 4)
        or matrix.shape[1] != tokens
        or matrix.shape[0] not in (1, batch)
    ):
        return None
    rope = torch.view_as_real(rotary_emb.squeeze(0).squeeze(0))
    if rope.shape != (tokens, half // 2, 2):
        return None

    out = torch.empty_like(x, dtype=torch.float32)
    groups4 = half // 4
    rope_pairs = half // 2
    block_n = 128
    first_grid = (
        batch,
        heads,
        groups4 * triton.cdiv(tokens, block_n),
    )
    _ucpe_first_half_kernel[first_grid](
        x,
        matrix,
        out,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        x.stride(3),
        matrix.stride(0),
        matrix.stride(1),
        matrix.stride(2),
        matrix.stride(3),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out.stride(3),
        tokens,
        groups4,
        matrix.shape[0],
        block_n,
    )
    rope_grid = (
        batch,
        heads,
        rope_pairs * triton.cdiv(tokens, block_n),
    )
    _ucpe_rope_second_half_kernel[rope_grid](
        x,
        rope,
        out,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        x.stride(3),
        rope.stride(0),
        rope.stride(1),
        rope.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out.stride(3),
        tokens,
        half,
        rope_pairs,
        inverse_rope,
        block_n,
    )
    return out


def _ucpe_transform_norms_fast(x: Tensor, out: Tensor) -> tuple[Tensor, Tensor] | None:
    batch, tokens, heads, dim = x.shape
    block_d = 1 << (dim - 1).bit_length()
    if (
        _DISABLE_UCPE_FAST
        or not x.is_cuda
        or not out.is_cuda
        or torch.is_grad_enabled()
        or triton is None
        or _ucpe_norms_kernel is None
        or out.shape != x.shape
        or block_d > 256
    ):
        return None

    pre_sq = torch.empty((batch, heads, tokens), device=x.device, dtype=torch.float32)
    post_sq = torch.empty_like(pre_sq)
    block_n = 16
    grid = (
        batch,
        heads,
        triton.cdiv(tokens, block_n),
    )
    _ucpe_norms_kernel[grid](
        x,
        out,
        pre_sq,
        post_sq,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        x.stride(3),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out.stride(3),
        pre_sq.stride(0),
        pre_sq.stride(1),
        pre_sq.stride(2),
        post_sq.stride(0),
        post_sq.stride(1),
        post_sq.stride(2),
        tokens,
        dim,
        block_n,
        block_d,
    )
    return pre_sq, post_sq


def _silu_multiply(value: Tensor, gate: Tensor, *, inplace: bool) -> Tensor:
    fast = _silu_multiply_fast(value, gate)
    if fast is not None:
        return fast
    return value * F.silu(gate, inplace=inplace)


def _silu_multiply_fast(value: Tensor, gate: Tensor) -> Tensor | None:
    if (
        _DISABLE_GLU_FAST
        or value.shape != gate.shape
        or value.dim() != 4
        or not value.is_cuda
        or not gate.is_cuda
        or torch.is_grad_enabled()
        or triton is None
        or _silu_multiply_kernel is None
    ):
        return None
    batch, channels, height, width = value.shape
    total = batch * channels * height * width
    if total <= 0:
        return torch.empty_like(value, memory_format=torch.channels_last)
    out = torch.empty(
        (batch, channels, height, width),
        device=value.device,
        dtype=value.dtype,
        memory_format=torch.channels_last,
    )
    block = 128
    grid = (triton.cdiv(total, block),)
    _silu_multiply_kernel[grid](
        value,
        gate,
        out,
        value.stride(0),
        value.stride(1),
        value.stride(2),
        value.stride(3),
        gate.stride(0),
        gate.stride(1),
        gate.stride(2),
        gate.stride(3),
        channels,
        height,
        width,
        total,
        block,
    )
    return out


def _camera_ray_mats(
    camera_conditions: Tensor,
    HW: tuple[int, int, int],
) -> Tensor:
    batch, frames = camera_conditions.shape[:2]
    latent_frames, height, width = HW
    if frames != latent_frames:
        raise ValueError(
            f"camera_conditions has {frames} frames but latent grid has {latent_frames}."
        )
    c2w = camera_conditions[..., :16].reshape(batch, frames, 4, 4)
    fx = camera_conditions[..., 16]
    fy = camera_conditions[..., 17]
    cx = camera_conditions[..., 18]
    cy = camera_conditions[..., 19]
    y_grid, x_grid = torch.meshgrid(
        torch.arange(height, dtype=torch.float32, device=camera_conditions.device),
        torch.arange(width, dtype=torch.float32, device=camera_conditions.device),
        indexing="ij",
    )
    x = (x_grid.view(1, 1, height, width) - cx[:, :, None, None]) / fx[
        :,
        :,
        None,
        None,
    ].clamp_min(1e-6)
    y = (y_grid.view(1, 1, height, width) - cy[:, :, None, None]) / fy[
        :,
        :,
        None,
        None,
    ].clamp_min(1e-6)
    dirs_cam = torch.stack([x, y, torch.ones_like(x)], dim=-1)
    dirs_cam = F.normalize(dirs_cam, dim=-1, eps=1e-6)
    rotation = c2w[..., :3, :3]
    translation = c2w[..., :3, 3]
    dirs_world = torch.einsum("btij,bthwj->bthwi", rotation, dirs_cam)
    cam_y = rotation[..., :, 1].view(batch, frames, 1, 1, 3).expand_as(dirs_world)
    z_ray = F.normalize(dirs_world, dim=-1, eps=1e-6)
    x_ray = F.normalize(torch.cross(cam_y, z_ray, dim=-1), dim=-1, eps=1e-6)
    y_ray = F.normalize(torch.cross(z_ray, x_ray, dim=-1), dim=-1, eps=1e-6)
    ray_to_world = torch.stack([x_ray, y_ray, z_ray], dim=-1)
    world_to_ray = ray_to_world.transpose(-1, -2)
    origin = translation.view(batch, frames, 1, 1, 3).expand_as(dirs_world)
    trans = -torch.einsum("bthwij,bthwj->bthwi", world_to_ray, origin)
    mats = torch.zeros(
        batch,
        frames,
        height,
        width,
        4,
        4,
        dtype=torch.float32,
        device=camera_conditions.device,
    )
    mats[..., :3, :3] = world_to_ray
    mats[..., :3, 3] = trans
    mats[..., 3, 3] = 1.0
    return mats


def _invert_se3(transforms: Tensor) -> Tensor:
    rotation_inv = transforms[..., :3, :3].transpose(-1, -2)
    out = torch.zeros_like(transforms)
    out[..., :3, :3] = rotation_inv
    out[..., :3, 3] = -torch.einsum(
        "...ij,...j->...i",
        rotation_inv,
        transforms[..., :3, 3],
    )
    out[..., 3, 3] = 1.0
    return out


__all__ = [
    "SANA_WM_STAGE1_SPEC",
    "SANA_WM_STREAMING_STAGE1_SPEC",
    "GLUMBConvTemp",
    "RMSNorm",
    "SanaWMStage1Block",
    "SanaWMStage1Model",
    "SanaWMStage1Spec",
    "Stage1CrossAttention",
    "Stage1SelfAttention",
    "linearize_stage1_ffn_for_quant",
]
