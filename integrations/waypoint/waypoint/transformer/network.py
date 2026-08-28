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

"""Checkpoint-compatible Waypoint DiT topology and local tensor operations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import cast

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.attention.flex_attention import BlockMask

from flashdreams.infra.config import InstantiateConfig
from waypoint.spec import WAYPOINT_1_5, WaypointModelSpec
from waypoint.transformer.cache import WaypointKVCache
from waypoint.transformer.norm import adaptive_gate, adaptive_rms_norm
from waypoint.transformer.rope import WaypointOrthoRoPEAngles, apply_waypoint_ortho_rope


# Compile the pure fixed-attention operation so FlexAttention receives the
# block-index representation required by the fixed cache.
@torch.compile(dynamic=False)
def _compiled_fixed_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    block_mask: BlockMask,
) -> Tensor:
    """Run fixed-cache GQA through FlexAttention's compiled kernel path."""
    from torch.nn.attention.flex_attention import flex_attention

    return flex_attention(
        query,
        key,
        value,
        block_mask=block_mask,
        enable_gqa=True,
    )


def sinusoidal_noise_embedding(
    dim: int,
    sigma: Tensor,
    *,
    frequencies: Tensor | None = None,
) -> Tensor:
    """Embed continuous noise levels with Waypoint's Wan-style Fourier basis.

    Args:
        dim: Even output feature width.
        sigma: Noise levels with arbitrary leading shape.
        frequencies: Optional precomputed positive Fourier frequencies. Supplying
            the conditioner's buffer preserves its checkpoint-runtime numeric
            behavior after a precision move.

    Returns:
        Cosine/sine features with shape ``[*sigma.shape, dim]``.

    Raises:
        ValueError: ``dim`` is not even.
    """
    if dim % 2:
        raise ValueError(f"noise embedding width must be even, got {dim}")
    half = dim // 2
    sigma = sigma.to(dtype=torch.float32)
    if frequencies is None:
        frequencies = torch.logspace(
            0,
            -1,
            steps=half,
            base=10_000.0,
            device=sigma.device,
            dtype=torch.float32,
        )
    elif frequencies.ndim != 1 or frequencies.numel() != half:
        raise ValueError(
            f"expected {half} one-dimensional Fourier frequencies, got "
            f"{tuple(frequencies.shape)}"
        )
    frequencies = frequencies.to(device=sigma.device, dtype=torch.float32)
    angles = sigma[..., None] * 1_000 * frequencies
    return torch.cat((torch.sin(angles), torch.cos(angles)), dim=-1) * (2**0.5)


class _TwoLayerMLP(nn.Module):
    """Bias-free two-projection MLP with raw checkpoint names."""

    def __init__(
        self, in_features: int, hidden_features: int, out_features: int
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features, bias=False)
        self.fc2 = nn.Linear(hidden_features, out_features, bias=False)

    def forward(self, input: Tensor) -> Tensor:
        """Apply the SiLU MLP used by the public control and noise conditioners."""
        return self.fc2(F.silu(self.fc1(input)))


class _NullControl(nn.Module):
    """Classifier-free null-control embedding container."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.null_emb = nn.Parameter(torch.empty(1, 1, d_model))


class _ConditionHead(nn.Module):
    """Three-projection adaptive-conditioning parameter group."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.bias_in = nn.Parameter(torch.empty(d_model))
        self.cond_proj = nn.ModuleList(
            [nn.Linear(d_model, d_model, bias=False) for _ in range(3)]
        )

    def forward(self, conditioning: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Project one conditioner into an AdaLN scale, shift, and gate tuple."""
        features = F.silu(conditioning + self.bias_in)
        return tuple(projection(features) for projection in self.cond_proj)  # type: ignore[return-value]


class _WaypointAttention(nn.Module):
    """Grouped-query attention projections and value-residual coefficient."""

    def __init__(self, spec: WaypointModelSpec) -> None:
        super().__init__()
        kv_dim = spec.n_kv_heads * spec.head_dim
        self.n_heads = spec.n_heads
        self.n_kv_heads = spec.n_kv_heads
        self.head_dim = spec.head_dim
        self.k_proj = nn.Linear(spec.d_model, kv_dim, bias=False)
        self.out_proj = nn.Linear(spec.d_model, spec.d_model, bias=False)
        self.q_proj = nn.Linear(spec.d_model, spec.d_model, bias=False)
        self.v_lamb = nn.Parameter(torch.empty(()))
        self.v_proj = nn.Linear(spec.d_model, kv_dim, bias=False)

    def project_qkv(self, tokens: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Project tokens into RMS-normalized Q/K and unnormalized grouped V.

        Args:
            tokens: Hidden states shaped ``[batch, tokens, d_model]``.

        Returns:
            Query, key, and value tensors shaped ``[batch, tokens, heads, head_dim]``.

        Raises:
            ValueError: ``tokens`` does not have Waypoint's model width.
        """
        if tokens.ndim != 3 or tokens.shape[-1] != self.q_proj.in_features:
            raise ValueError(
                "Waypoint attention requires [batch, tokens, d_model] input; "
                f"got {tuple(tokens.shape)}"
            )
        batch_size, token_count, _ = tokens.shape
        query = self.q_proj(tokens).reshape(
            batch_size, token_count, self.n_heads, self.head_dim
        )
        key = self.k_proj(tokens).reshape(
            batch_size, token_count, self.n_kv_heads, self.head_dim
        )
        value = self.v_proj(tokens).reshape(
            batch_size, token_count, self.n_kv_heads, self.head_dim
        )
        return (
            F.rms_norm(query, (self.head_dim,), weight=None, eps=None),
            F.rms_norm(key, (self.head_dim,), weight=None, eps=None),
            value,
        )

    def blend_value_residual(
        self, current_value: Tensor, initial_value: Tensor | None
    ) -> tuple[Tensor, Tensor]:
        """Blend current values with the first block's value stream.

        Args:
            current_value: Values projected by the current transformer block.
            initial_value: First-block values; ``None`` on the first block.

        Returns:
            Values for attention and the value stream retained for later blocks.

        Raises:
            ValueError: A retained value stream does not match the current layout.
        """
        if initial_value is None:
            return current_value, current_value
        if initial_value.shape != current_value.shape:
            raise ValueError(
                "Waypoint value residual requires matching current and initial "
                f"value shapes, got {tuple(current_value.shape)} and "
                f"{tuple(initial_value.shape)}"
            )
        return torch.lerp(current_value, initial_value, self.v_lamb), initial_value

    def forward(
        self,
        tokens: Tensor,
        *,
        cosine: Tensor,
        sine: Tensor,
        layer_index: int,
        frame_index: int,
        kv_cache: WaypointKVCache,
        initial_value: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        """Attend over the sparse causal history of one Waypoint block.

        Args:
            tokens: Normalized hidden states shaped ``[B, S, D]``.
            cosine: Current-frame packed RoPE cosine factors.
            sine: Current-frame packed RoPE sine factors.
            layer_index: Zero-indexed transformer block owning the K/V history.
            frame_index: Zero-indexed latent action being denoised.
            kv_cache: Per-rollout sparse K/V history.
            initial_value: Value tensor retained from the first transformer block.

        Returns:
            Attention residual in ``[B, S, D]`` layout and the value stream to
            retain for subsequent blocks.
        """
        query, key, current_value = self.project_qkv(tokens)
        query = apply_waypoint_ortho_rope(query, cosine, sine)
        key = apply_waypoint_ortho_rope(key, cosine, sine)
        value, retained_value = self.blend_value_residual(current_value, initial_value)

        view = kv_cache.update(
            layer_index=layer_index,
            frame_index=frame_index,
            key=key.transpose(1, 2),
            value=value.transpose(1, 2),
        )
        query = query.transpose(1, 2)
        if view.block_mask is None:
            attention = F.scaled_dot_product_attention(
                query,
                view.key,
                view.value,
                enable_gqa=True,
            )
        else:
            attention = _compiled_fixed_attention(
                query, view.key, view.value, view.block_mask
            )
        batch_size, _, token_count, _ = attention.shape
        attention = attention.transpose(1, 2).reshape(
            batch_size, token_count, self.out_proj.in_features
        )
        return self.out_proj(attention), retained_value


class _ControlFusion(nn.Module):
    """Controller-fusion projections for every third Waypoint block."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.fc1_c = nn.Linear(d_model, d_model, bias=False)
        self.fc1_x = nn.Linear(d_model, d_model, bias=False)
        self.fc2 = nn.Linear(d_model, d_model, bias=False)

    def forward(self, tokens: Tensor, control: Tensor) -> Tensor:
        """Fuse controller features through the published residual MLP path."""
        return self.fc2(F.silu(self.fc1_x(tokens) + self.fc1_c(control)))


class _WaypointBlock(nn.Module):
    """One checkpoint-compatible Waypoint transformer block."""

    def __init__(self, spec: WaypointModelSpec, *, has_control_fusion: bool) -> None:
        super().__init__()
        self.attn = _WaypointAttention(spec)
        self.attn_cond_head = _ConditionHead(spec.d_model)
        self.ctrl_mlpfusion: _ControlFusion | None = (
            _ControlFusion(spec.d_model) if has_control_fusion else None
        )
        self.dit_mlp = _TwoLayerMLP(
            spec.d_model,
            spec.d_model * spec.mlp_ratio,
            spec.d_model,
        )
        self.mlp_cond_head = _ConditionHead(spec.d_model)

    def forward(
        self,
        tokens: Tensor,
        *,
        conditioning: Tensor,
        control: Tensor | None,
        cosine: Tensor,
        sine: Tensor,
        layer_index: int,
        frame_index: int,
        kv_cache: WaypointKVCache,
        initial_value: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        """Apply one conditionally gated Waypoint transformer block.

        Args:
            tokens: Hidden states in ``[B, T * S, D]`` layout.
            conditioning: Noise/control features in ``[B, T, D]`` layout.
            control: Per-token controller features; ``None`` skips periodic fusion.
            cosine: Current-frame packed RoPE cosine factors.
            sine: Current-frame packed RoPE sine factors.
            layer_index: Zero-indexed block index for sparse-history selection.
            frame_index: Zero-indexed latent action being denoised.
            kv_cache: Per-rollout sparse attention K/V history.
            initial_value: First-block value stream; ``None`` for block zero.

        Returns:
            Updated hidden states and the first-block value stream.

        Raises:
            ValueError: Control tokens do not match the hidden-state layout.
        """
        attn_scale, attn_bias, attn_gate = self.attn_cond_head(conditioning)
        attn_residual, initial_value = self.attn(
            adaptive_rms_norm(tokens, attn_scale, attn_bias),
            cosine=cosine,
            sine=sine,
            layer_index=layer_index,
            frame_index=frame_index,
            kv_cache=kv_cache,
            initial_value=initial_value,
        )
        tokens = tokens + adaptive_gate(attn_residual, attn_gate)

        if control is not None and self.ctrl_mlpfusion is not None:
            if control.shape != tokens.shape:
                raise ValueError(
                    "Waypoint control tokens must match hidden states, got "
                    f"control={tuple(control.shape)}, tokens={tuple(tokens.shape)}"
                )
            channels = tokens.shape[-1]
            fused_tokens = F.rms_norm(tokens, (channels,), weight=None, eps=None)
            fused_control = F.rms_norm(control, (channels,), weight=None, eps=None)
            tokens = tokens + self.ctrl_mlpfusion(fused_tokens, fused_control)

        mlp_scale, mlp_bias, mlp_gate = self.mlp_cond_head(conditioning)
        mlp_residual = self.dit_mlp(adaptive_rms_norm(tokens, mlp_scale, mlp_bias))
        return tokens + adaptive_gate(mlp_residual, mlp_gate), initial_value


class _WaypointBlockStack(nn.Module):
    """Ordered Waypoint transformer blocks with periodic control fusion."""

    def __init__(self, spec: WaypointModelSpec) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                _WaypointBlock(
                    spec,
                    has_control_fusion=(
                        layer_index % spec.controller_conditioning_period == 0
                    ),
                )
                for layer_index in range(spec.n_layers)
            ]
        )


class _ControlEmbedder(nn.Module):
    """Waypoint controller embedding MLP."""

    def __init__(self, spec: WaypointModelSpec) -> None:
        super().__init__()
        self.mlp = _TwoLayerMLP(
            spec.n_buttons + 3,
            spec.d_model * spec.mlp_ratio,
            spec.d_model,
        )


class _NoiseEmbedder(nn.Module):
    """Waypoint diffusion-noise embedding MLP."""

    freq: Tensor
    mlp: _TwoLayerMLP

    def __init__(self, spec: WaypointModelSpec) -> None:
        super().__init__()
        if spec.noise_embedding_dim % 2:
            raise ValueError("Waypoint noise embedding width must be even")
        self.register_buffer(
            "freq",
            torch.logspace(
                0,
                -1,
                steps=spec.noise_embedding_dim // 2,
                base=10_000.0,
                dtype=torch.float32,
            ),
            persistent=False,
        )
        self.mlp = _TwoLayerMLP(
            spec.noise_embedding_dim,
            spec.d_model * spec.mlp_ratio,
            spec.d_model,
        )

    def _apply(self, fn):
        """Move the conditioner while retaining FP32 Fourier-MLP arithmetic."""

        def keep_dtype(tensor: Tensor) -> Tensor:
            return fn(tensor).to(dtype=tensor.dtype)

        return super()._apply(keep_dtype)


class _OutputNorm(nn.Module):
    """Checkpoint-compatible final adaptive-normalization projection."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.fc = nn.Linear(d_model, 2 * d_model, bias=False)


@dataclass(kw_only=True)
class WaypointDiTConfig(InstantiateConfig):
    """Static construction config for the published Waypoint 1.5 DiT."""

    _target: type["WaypointDiT"] = field(default_factory=lambda: WaypointDiT)

    spec: WaypointModelSpec = WAYPOINT_1_5
    """Immutable architecture contract for the target checkpoint."""


class WaypointDiT(nn.Module):
    """Denoise one controllable autoregressive Waypoint latent frame.

    Waypoint generates a four-frame video chunk from each 32-channel latent
    frame, then feeds that latent history back into the next action. Its control
    embedding, grouped-query transformer widths, and fixed checkpoint namespace
    are coupled to that rollout contract, so it cannot be represented by a
    generic image-to-video DiT without changing the learned function.
    """

    ctrl_cfg: _NullControl
    ctrl_emb: _ControlEmbedder
    denoise_step_emb: _NoiseEmbedder
    out_norm: _OutputNorm
    patchify: nn.Conv2d
    transformer: _WaypointBlockStack
    rope_angles: WaypointOrthoRoPEAngles
    unpatchify: nn.ConvTranspose2d

    def __init__(self, config: WaypointDiTConfig) -> None:
        super().__init__()
        self.config = config
        self.spec = config.spec
        self.ctrl_cfg = _NullControl(self.spec.d_model)
        self.ctrl_emb = _ControlEmbedder(self.spec)
        self.denoise_step_emb = _NoiseEmbedder(self.spec)
        self.out_norm = _OutputNorm(self.spec.d_model)
        self.patchify = nn.Conv2d(
            self.spec.channels,
            self.spec.d_model,
            kernel_size=(self.spec.patch_height, self.spec.patch_width),
            stride=(self.spec.patch_height, self.spec.patch_width),
            bias=False,
        )
        self.transformer = _WaypointBlockStack(self.spec)
        self.rope_angles = WaypointOrthoRoPEAngles(self.spec)
        self.unpatchify = nn.ConvTranspose2d(
            self.spec.d_model,
            self.spec.channels,
            kernel_size=(self.spec.patch_height, self.spec.patch_width),
            stride=(self.spec.patch_height, self.spec.patch_width),
            bias=True,
        )

    def patchify_latent(self, latent: Tensor) -> Tensor:
        """Patchify one or more Waypoint latent frames into DiT tokens.

        Args:
            latent: Raw latent video with shape ``[B, T, C, H, W]``.

        Returns:
            Tokens ordered by ``(T, H, W)`` with shape ``[B, T * L, D]``.

        Raises:
            ValueError: The latent shape differs from the published contract.
        """
        if latent.ndim != 5:
            raise ValueError(
                "Waypoint latent must have shape [B, T, C, H, W], "
                f"got {tuple(latent.shape)}"
            )
        batch_size, frames, channels, height, width = latent.shape
        expected = (self.spec.channels, self.spec.latent_height, self.spec.latent_width)
        if (channels, height, width) != expected:
            raise ValueError(
                "Waypoint latent C/H/W mismatch: "
                f"expected {expected}, got {(channels, height, width)}"
            )
        x = self.patchify(latent.reshape(batch_size * frames, channels, height, width))
        patch_height, patch_width = x.shape[-2:]
        x = x.reshape(batch_size, frames, self.spec.d_model, patch_height, patch_width)
        return x.permute(0, 1, 3, 4, 2).reshape(batch_size, -1, self.spec.d_model)

    def unpatchify_tokens(self, tokens: Tensor, *, frames: int = 1) -> Tensor:
        """Unpatchify DiT tokens into raw Waypoint latents.

        Args:
            tokens: Tokens with shape ``[B, T * L, D]``.
            frames: Latent-frame count ``T`` represented by ``tokens``.

        Returns:
            Latent video with shape ``[B, T, C, H, W]``.

        Raises:
            ValueError: Token rank, width, count, or frame count is invalid.
        """
        if tokens.ndim != 3:
            raise ValueError(f"Waypoint tokens must have rank 3, got {tokens.ndim}")
        if frames < 1:
            raise ValueError(f"frames must be positive, got {frames}")
        batch_size, token_count, width = tokens.shape
        if width != self.spec.d_model:
            raise ValueError(
                f"Waypoint token width must be {self.spec.d_model}, got {width}"
            )
        tokens_per_frame = self.spec.tokens_per_latent_frame
        if token_count != frames * tokens_per_frame:
            raise ValueError(
                f"Waypoint token count must be frames * {tokens_per_frame}, "
                f"got frames={frames}, token_count={token_count}"
            )
        patch_height = self.spec.latent_height // self.spec.patch_height
        patch_width = self.spec.latent_width // self.spec.patch_width
        # The published tensor is laid out as a convolution kernel, but its
        # inference operator emits the ``C * patch_h * patch_w`` pixel vector
        # independently for every token. Keep the raw kernel layout in the
        # module state dict, then expose that learned operator directly here.
        output_weight = self.unpatchify.weight.permute(1, 2, 3, 0).reshape(
            -1, self.spec.d_model
        )
        output_bias = cast(Tensor, self.unpatchify.bias)
        output_bias = (
            output_bias[:, None, None]
            .expand(-1, self.spec.patch_height, self.spec.patch_width)
            .reshape(-1)
        )
        x = F.linear(tokens, output_weight, output_bias)
        x = x.reshape(
            batch_size,
            frames,
            patch_height,
            patch_width,
            self.spec.channels,
            self.spec.patch_height,
            self.spec.patch_width,
        )
        return x.permute(0, 1, 4, 2, 5, 3, 6).reshape(
            batch_size,
            frames,
            self.spec.channels,
            self.spec.latent_height,
            self.spec.latent_width,
        )

    def embed_control(self, *, button: Tensor, mouse: Tensor, scroll: Tensor) -> Tensor:
        """Embed one controller state per autoregressive latent frame.

        Args:
            button: Multi-hot button tensor with shape ``[B, T, 256]``.
            mouse: Pointer deltas with shape ``[B, T, 2]``.
            scroll: Wheel direction with shape ``[B, T, 1]``.

        Returns:
            Controller embeddings with shape ``[B, T, D]``.

        Raises:
            ValueError: Controller tensors do not share the published shapes.
        """
        expected_prefix = button.shape[:-1]
        if (
            button.ndim != 3
            or button.shape[-1] != self.spec.n_buttons
            or mouse.shape != expected_prefix + (2,)
            or scroll.shape != expected_prefix + (1,)
        ):
            raise ValueError(
                "Waypoint controls require button=[B, T, 256], mouse=[B, T, 2], "
                f"scroll=[B, T, 1]; got button={tuple(button.shape)}, "
                f"mouse={tuple(mouse.shape)}, scroll={tuple(scroll.shape)}"
            )
        controls = torch.cat((mouse, button, scroll), dim=-1)
        return self.ctrl_emb.mlp(controls)

    def embed_noise(self, sigma: Tensor) -> Tensor:
        """Embed continuous rectified-flow noise levels.

        Args:
            sigma: Scalar or batch-shaped noise level.

        Returns:
            Noise embedding with shape ``[*sigma.shape, D]``.
        """
        features = sinusoidal_noise_embedding(
            self.spec.noise_embedding_dim,
            sigma,
            frequencies=self.denoise_step_emb.freq,
        )
        return self.denoise_step_emb.mlp(features).to(dtype=self.patchify.weight.dtype)

    def forward(
        self,
        latent: Tensor,
        *,
        sigma: Tensor,
        frame_index: int,
        kv_cache: WaypointKVCache,
        button: Tensor | None = None,
        mouse: Tensor | None = None,
        scroll: Tensor | None = None,
    ) -> Tensor:
        """Predict rectified-flow velocity for one autoregressive latent action.

        Args:
            latent: Noisy latent video in ``[B, 1, C, H, W]`` layout.
            sigma: One noise level per batch item, shaped ``[B]``.
            frame_index: Zero-indexed latent action shared by the batch.
            kv_cache: Per-rollout sparse attention K/V history.
            button: Optional multi-hot buttons in ``[B, 1, 256]`` layout.
            mouse: Optional pointer deltas in ``[B, 1, 2]`` layout.
            scroll: Optional wheel directions in ``[B, 1, 1]`` layout.

        Returns:
            Rectified-flow velocity with the same shape as ``latent``.

        Raises:
            ValueError: The latent, noise, or partial controller state does not
                match Waypoint's one-action execution contract.
        """
        if latent.ndim != 5 or latent.shape[1] != 1:
            raise ValueError(
                "WaypointDiT forward expects one latent action in [B, 1, C, H, W] "
                f"layout, got {tuple(latent.shape)}"
            )
        if sigma.ndim != 1 or sigma.shape[0] != latent.shape[0]:
            raise ValueError(
                f"sigma must have one value per batch item, got {tuple(sigma.shape)}"
            )
        if frame_index < 0:
            raise ValueError(f"frame_index must be non-negative, got {frame_index}")
        controls = (button, mouse, scroll)
        if any(control is not None for control in controls) and any(
            control is None for control in controls
        ):
            raise ValueError("button, mouse, and scroll must be supplied together")

        tokens = self.patchify_latent(latent)
        batch_size, token_count, _ = tokens.shape
        conditioning = self.embed_noise(sigma.to(device=tokens.device)).to(tokens.dtype)

        if button is None:
            control_frame = self.ctrl_cfg.null_emb.to(dtype=tokens.dtype).expand(
                batch_size, 1, -1
            )
        else:
            mouse = cast(Tensor, mouse)
            scroll = cast(Tensor, scroll)
            control_frame = self.embed_control(
                button=button.to(device=tokens.device, dtype=tokens.dtype),
                mouse=mouse.to(device=tokens.device, dtype=tokens.dtype),
                scroll=scroll.to(device=tokens.device, dtype=tokens.dtype),
            )
            if control_frame.shape[:2] != (batch_size, 1):
                raise ValueError(
                    "Waypoint controller state must describe one frame per batch item, "
                    f"got {tuple(control_frame.shape)}"
                )
        control_tokens = control_frame.expand(-1, token_count, -1)
        cosine, sine = self._current_rope_angles(
            frame_index=frame_index, device=tokens.device
        )

        initial_value: Tensor | None = None
        conditioning = conditioning[:, None]
        for layer_index, block in enumerate(self.transformer.blocks):
            tokens, initial_value = block(
                tokens,
                conditioning=conditioning,
                control=control_tokens,
                cosine=cosine,
                sine=sine,
                layer_index=layer_index,
                frame_index=frame_index,
                kv_cache=kv_cache,
                initial_value=initial_value,
            )

        scale, bias = self.out_norm.fc(F.silu(conditioning)).chunk(2, dim=-1)
        tokens = F.silu(adaptive_rms_norm(tokens, scale, bias))
        return self.unpatchify_tokens(tokens)

    def _current_rope_angles(
        self, *, frame_index: int, device: torch.device
    ) -> tuple[Tensor, Tensor]:
        """Build the fixed spatial and current temporal RoPE factors."""
        rows = torch.arange(self.spec.patch_grid_height, device=device)
        columns = torch.arange(self.spec.patch_grid_width, device=device)
        row_index = rows.repeat_interleave(self.spec.patch_grid_width)
        column_index = columns.repeat(self.spec.patch_grid_height)
        # The cache advances once per latent action.  Temporal RoPE uses the
        # checkpoint's base-rate timestamp, whose stride is part of the
        # checkpoint contract (and is one for Waypoint 1.5).
        frame_indices = torch.full_like(
            row_index,
            frame_index * self.spec.frame_timestamp_stride,
        )
        return self.rope_angles(
            frame_index=frame_indices,
            row_index=row_index,
            column_index=column_index,
        )
