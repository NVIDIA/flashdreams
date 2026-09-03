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

"""Native dual-stream Qwen Image diffusion transformer."""

from __future__ import annotations

import math
from math import prod

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def timestep_embedding(
    timesteps: Tensor,
    embedding_dim: int,
    *,
    flip_sin_to_cos: bool = False,
    downscale_freq_shift: float = 1.0,
    scale: float = 1.0,
    max_period: int = 10_000,
) -> Tensor:
    """Build sinusoidal diffusion timestep embeddings."""
    if timesteps.ndim != 1:
        raise ValueError(f"timesteps must be one-dimensional, got {timesteps.shape}")
    half = embedding_dim // 2
    exponent = -math.log(max_period) * torch.arange(
        half, dtype=torch.float32, device=timesteps.device
    )
    exponent = exponent / (half - downscale_freq_shift)
    phases = timesteps[:, None].float() * torch.exp(exponent)[None] * scale
    result = torch.cat([torch.sin(phases), torch.cos(phases)], dim=-1)
    if flip_sin_to_cos:
        result = torch.cat([result[:, half:], result[:, :half]], dim=-1)
    if embedding_dim % 2:
        result = F.pad(result, (0, 1))
    return result


class RMSNorm(nn.Module):
    """Normalize the final dimension by its root-mean-square magnitude."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, hidden_states: Tensor) -> Tensor:
        input_dtype = hidden_states.dtype
        variance = hidden_states.float().square().mean(-1, keepdim=True)
        normalized = hidden_states * torch.rsqrt(variance + self.eps)
        return normalized.to(self.weight.dtype) * self.weight.to(input_dtype)


class TimestepEmbedding(nn.Module):
    """Project sinusoidal timesteps into the transformer width."""

    def __init__(self, in_channels: int, time_embed_dim: int) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(in_channels, time_embed_dim)
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim)

    def forward(self, sample: Tensor) -> Tensor:
        return self.linear_2(F.silu(self.linear_1(sample)))


class QwenTimestepProjEmbeddings(nn.Module):
    """Create Qwen's scaled sinusoidal timestep conditioning."""

    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.timestep_embedder = TimestepEmbedding(256, embedding_dim)

    def forward(self, timestep: Tensor, hidden_states: Tensor) -> Tensor:
        projected = timestep_embedding(
            timestep,
            256,
            flip_sin_to_cos=True,
            downscale_freq_shift=0,
            scale=1000,
        )
        return self.timestep_embedder(projected.to(hidden_states.dtype))


class GELU(nn.Module):
    """Linear projection followed by approximate GELU."""

    def __init__(self, dim_in: int, dim_out: int) -> None:
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out)

    def forward(self, hidden_states: Tensor) -> Tensor:
        return F.gelu(self.proj(hidden_states), approximate="tanh")


class FeedForward(nn.Module):
    """Qwen transformer feed-forward network."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.net = nn.ModuleList(
            [GELU(dim, dim * 4), nn.Dropout(0.0), nn.Linear(dim * 4, dim)]
        )

    def forward(self, hidden_states: Tensor) -> Tensor:
        for module in self.net:
            hidden_states = module(hidden_states)
        return hidden_states


class AdaLayerNormContinuous(nn.Module):
    """Apply timestep-conditioned scale and shift after layer normalization."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(dim, dim * 2)
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(self, hidden_states: Tensor, conditioning: Tensor) -> Tensor:
        scale, shift = self.linear(F.silu(conditioning).to(hidden_states.dtype)).chunk(
            2, dim=1
        )
        return self.norm(hidden_states) * (1 + scale[:, None]) + shift[:, None]


def _rope_params(index: Tensor, dim: int, theta: int = 10_000) -> Tensor:
    if dim % 2:
        raise ValueError(f"RoPE axis width must be even, got {dim}")
    phases = torch.outer(
        index,
        1.0 / torch.pow(theta, torch.arange(0, dim, 2).float().div(float(dim))),
    )
    return torch.polar(torch.ones_like(phases), phases)


class QwenEmbedLayer3DRope(nn.Module):
    """Build Qwen's layered temporal-height-width rotary frequencies."""

    pos_freqs: Tensor
    neg_freqs: Tensor

    def __init__(
        self, axes_dim: tuple[int, int, int] = (16, 56, 56), theta: int = 10_000
    ) -> None:
        super().__init__()
        positive = torch.arange(4096)
        negative = torch.arange(4096).flip(0) * -1 - 1
        self.axes_dim = axes_dim
        self.register_buffer(
            "pos_freqs",
            torch.cat([_rope_params(positive, dim, theta) for dim in axes_dim], dim=1),
            persistent=False,
        )
        self.register_buffer(
            "neg_freqs",
            torch.cat([_rope_params(negative, dim, theta) for dim in axes_dim], dim=1),
            persistent=False,
        )

    def _apply(self, fn, recurse: bool = True):
        """Move RoPE caches without applying a lossy real dtype cast."""
        del recurse
        device = fn(torch.empty(0, device=self.pos_freqs.device)).device
        self.pos_freqs = self.pos_freqs.to(device=device)
        self.neg_freqs = self.neg_freqs.to(device=device)
        return self

    def _image_freqs(
        self,
        shape: tuple[int, int, int],
        layer_index: int,
        *,
        condition: bool,
    ) -> Tensor:
        frames, height, width = shape
        positive = self.pos_freqs.split([dim // 2 for dim in self.axes_dim], dim=1)
        negative = self.neg_freqs.split([dim // 2 for dim in self.axes_dim], dim=1)
        if condition:
            frame_freqs = negative[0][-1:]
        else:
            frame_freqs = positive[0][layer_index : layer_index + frames]
        frame_freqs = frame_freqs.view(frames, 1, 1, -1).expand(
            frames, height, width, -1
        )
        height_freqs = torch.cat(
            [negative[1][-(height - height // 2) :], positive[1][: height // 2]],
            dim=0,
        ).view(1, height, 1, -1)
        width_freqs = torch.cat(
            [negative[2][-(width - width // 2) :], positive[2][: width // 2]],
            dim=0,
        ).view(1, 1, width, -1)
        return torch.cat(
            [
                frame_freqs,
                height_freqs.expand(frames, height, width, -1),
                width_freqs.expand(frames, height, width, -1),
            ],
            dim=-1,
        ).reshape(frames * height * width, -1)

    def forward(
        self, image_shapes: list[tuple[int, int, int]], text_sequence_length: int
    ) -> tuple[Tensor, Tensor]:
        """Return image and text complex rotary frequencies."""
        image = torch.cat(
            [
                self._image_freqs(
                    shape,
                    index,
                    condition=index == len(image_shapes) - 1,
                )
                for index, shape in enumerate(image_shapes)
            ],
            dim=0,
        )
        max_image_index = max(
            len(image_shapes) - 1,
            *(max(height // 2, width // 2) for _, height, width in image_shapes),
        )
        text = self.pos_freqs[max_image_index : max_image_index + text_sequence_length]
        return image, text


def apply_rotary_embedding(hidden_states: Tensor, frequencies: Tensor) -> Tensor:
    """Apply complex rotary frequencies to ``[B, S, H, D]`` states."""
    complex_states = torch.view_as_complex(
        hidden_states.float().reshape(*hidden_states.shape[:-1], -1, 2)
    )
    rotated = complex_states * frequencies[None, :, None]
    return torch.view_as_real(rotated).flatten(-2).to(hidden_states.dtype)


class QwenJointAttention(nn.Module):
    """Jointly attend over image and text streams."""

    def __init__(self, dim: int, heads: int, head_dim: int) -> None:
        super().__init__()
        self.heads = heads
        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)
        self.add_k_proj = nn.Linear(dim, dim)
        self.add_v_proj = nn.Linear(dim, dim)
        self.add_q_proj = nn.Linear(dim, dim)
        self.to_out = nn.ModuleList([nn.Linear(dim, dim), nn.Dropout(0.0)])
        self.to_add_out = nn.Linear(dim, dim)
        self.norm_q = RMSNorm(head_dim)
        self.norm_k = RMSNorm(head_dim)
        self.norm_added_q = RMSNorm(head_dim)
        self.norm_added_k = RMSNorm(head_dim)

    def _heads(self, hidden_states: Tensor) -> Tensor:
        return hidden_states.unflatten(-1, (self.heads, -1))

    def forward(
        self,
        image: Tensor,
        text: Tensor,
        rotary: tuple[Tensor, Tensor],
        attention_mask: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        image_q = self.norm_q(self._heads(self.to_q(image)))
        image_k = self.norm_k(self._heads(self.to_k(image)))
        image_v = self._heads(self.to_v(image))
        text_q = self.norm_added_q(self._heads(self.add_q_proj(text)))
        text_k = self.norm_added_k(self._heads(self.add_k_proj(text)))
        text_v = self._heads(self.add_v_proj(text))

        image_freqs, text_freqs = rotary
        image_q = apply_rotary_embedding(image_q, image_freqs)
        image_k = apply_rotary_embedding(image_k, image_freqs)
        text_q = apply_rotary_embedding(text_q, text_freqs)
        text_k = apply_rotary_embedding(text_k, text_freqs)

        text_length = text.shape[1]
        query = torch.cat([text_q, image_q], dim=1).transpose(1, 2)
        key = torch.cat([text_k, image_k], dim=1).transpose(1, 2)
        value = torch.cat([text_v, image_v], dim=1).transpose(1, 2)
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
        ).transpose(1, 2)
        attended = attended.flatten(2)
        text_output = self.to_add_out(attended[:, :text_length].contiguous())
        image_output = self.to_out[0](attended[:, text_length:].contiguous())
        return self.to_out[1](image_output), text_output


class QwenImageTransformerBlock(nn.Module):
    """Process one dual-stream Qwen transformer layer."""

    def __init__(self, dim: int, heads: int, head_dim: int) -> None:
        super().__init__()
        self.img_mod = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
        self.img_norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = QwenJointAttention(dim, heads, head_dim)
        self.img_norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.img_mlp = FeedForward(dim)
        self.txt_mod = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
        self.txt_norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.txt_norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.txt_mlp = FeedForward(dim)

    @staticmethod
    def _modulate(
        hidden_states: Tensor, parameters: Tensor, token_kind: Tensor | None = None
    ) -> tuple[Tensor, Tensor]:
        shift, scale, gate = parameters.chunk(3, dim=-1)
        if token_kind is None:
            return (
                hidden_states * (1 + scale[:, None]) + shift[:, None],
                gate[:, None],
            )
        batch = shift.shape[0] // 2
        selector = token_kind[..., None].bool()
        shift = torch.where(selector, shift[batch:, None], shift[:batch, None])
        scale = torch.where(selector, scale[batch:, None], scale[:batch, None])
        gate = torch.where(selector, gate[batch:, None], gate[:batch, None])
        return hidden_states * (1 + scale) + shift, gate

    def forward(
        self,
        image: Tensor,
        text: Tensor,
        timestep_embedding: Tensor,
        rotary: tuple[Tensor, Tensor],
        token_kind: Tensor,
        attention_mask: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        image_mod1, image_mod2 = self.img_mod(timestep_embedding).chunk(2, dim=-1)
        text_timestep = timestep_embedding.chunk(2, dim=0)[0]
        text_mod1, text_mod2 = self.txt_mod(text_timestep).chunk(2, dim=-1)
        image_input, image_gate = self._modulate(
            self.img_norm1(image), image_mod1, token_kind
        )
        text_input, text_gate = self._modulate(self.txt_norm1(text), text_mod1)
        image_attention, text_attention = self.attn(
            image_input, text_input, rotary, attention_mask
        )
        image = image + image_gate * image_attention
        text = text + text_gate * text_attention
        image_input, image_gate = self._modulate(
            self.img_norm2(image), image_mod2, token_kind
        )
        text_input, text_gate = self._modulate(self.txt_norm2(text), text_mod2)
        image = image + image_gate * self.img_mlp(image_input)
        text = text + text_gate * self.txt_mlp(text_input)
        return text, image


class QwenImageTransformer(nn.Module):
    """Qwen Image Edit 2511 diffusion transformer."""

    def __init__(
        self,
        *,
        patch_size: int = 2,
        in_channels: int = 64,
        out_channels: int = 16,
        num_layers: int = 60,
        attention_head_dim: int = 128,
        num_attention_heads: int = 24,
        joint_attention_dim: int = 3584,
        axes_dims_rope: tuple[int, int, int] = (16, 56, 56),
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.patch_size = patch_size
        self.inner_dim = num_attention_heads * attention_head_dim
        self.pos_embed = QwenEmbedLayer3DRope(axes_dims_rope)
        self.time_text_embed = QwenTimestepProjEmbeddings(self.inner_dim)
        self.txt_norm = RMSNorm(joint_attention_dim)
        self.img_in = nn.Linear(in_channels, self.inner_dim)
        self.txt_in = nn.Linear(joint_attention_dim, self.inner_dim)
        self.transformer_blocks = nn.ModuleList(
            [
                QwenImageTransformerBlock(
                    self.inner_dim, num_attention_heads, attention_head_dim
                )
                for _ in range(num_layers)
            ]
        )
        self.norm_out = AdaLayerNormContinuous(self.inner_dim)
        self.proj_out = nn.Linear(
            self.inner_dim, patch_size * patch_size * out_channels
        )

    def forward(
        self,
        hidden_states: Tensor,
        encoder_hidden_states: Tensor,
        timestep: Tensor,
        image_shapes: list[tuple[int, int, int]],
        encoder_hidden_states_mask: Tensor | None = None,
    ) -> Tensor:
        """Predict packed latent flow for an output plus reference image."""
        image = self.img_in(hidden_states)
        text = self.txt_in(self.txt_norm(encoder_hidden_states))
        timestep = timestep.to(image.dtype)
        doubled_timestep = torch.cat([timestep, torch.zeros_like(timestep)])
        conditioning = self.time_text_embed(doubled_timestep, image)
        token_kind = torch.tensor(
            [
                [0] * prod(image_shapes[0])
                + [1] * sum(prod(s) for s in image_shapes[1:])
            ],
            device=image.device,
            dtype=torch.int32,
        )
        rotary = self.pos_embed(image_shapes, text.shape[1])
        attention_mask = None
        if encoder_hidden_states_mask is not None:
            image_mask = torch.ones(
                (image.shape[0], image.shape[1]),
                device=image.device,
                dtype=torch.bool,
            )
            joint_mask = torch.cat(
                [encoder_hidden_states_mask.bool(), image_mask], dim=1
            )
            attention_mask = joint_mask[:, None, None]
        for block in self.transformer_blocks:
            text, image = block(
                image,
                text,
                conditioning,
                rotary,
                token_kind,
                attention_mask,
            )
        image = self.norm_out(image, conditioning.chunk(2, dim=0)[0])
        return self.proj_out(image)


def true_cfg(cond: Tensor, uncond: Tensor, scale: float) -> Tensor:
    """Combine conditional and unconditional predictions with norm rescaling."""
    combined = uncond + scale * (cond - uncond)
    denominator = torch.linalg.vector_norm(combined, dim=-1, keepdim=True).clamp_min(
        torch.finfo(combined.dtype).eps
    )
    numerator = torch.linalg.vector_norm(cond, dim=-1, keepdim=True)
    return combined * (numerator / denominator)


__all__ = ["QwenImageTransformer", "true_cfg"]
