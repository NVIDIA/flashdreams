# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FlashDreams-native MiniMax H3 joint video/audio transformer."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Literal, cast

import torch
from flashdreams.core.attention import NativeAttention, apply_rope_freqs
from flashdreams.core.checkpoint.load import load_checkpoint
from flashdreams.infra.diffusion.transformer import (
    Transformer,
    TransformerAutoregressiveCache,
    TransformerConfig,
)
from torch import Tensor, nn

H3_TRANSFORMER_CHECKPOINT = (
    "https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/"
    "42ed227ee7df40d41602854ae760620d6eb651fe/transformer/"
    "diffusion_pytorch_model.safetensors.index.json"
)
H3_REF_TRANSFORMER_CHECKPOINT = (
    "https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/"
    "42ed227ee7df40d41602854ae760620d6eb651fe/transformer_ref/"
    "diffusion_pytorch_model.safetensors.index.json"
)
MODALITY_COUNT = 3


@dataclass(kw_only=True)
class MiniMaxH3TransformerCache(TransformerAutoregressiveCache):
    """Non-autoregressive request state required by ``predict_flow``."""

    audio_hidden_states: Tensor
    encoder_hidden_states: Tensor
    timestep: Tensor
    timestep_indices: Tensor
    token_tags: Tensor
    position_ids: Tensor
    video_indices: Tensor
    audio_indices: Tensor
    text_indices: Tensor
    last_audio_flow: Tensor | None = None


@dataclass(kw_only=True)
class MiniMaxH3TransformerConfig(TransformerConfig):
    """Architecture and checkpoint configuration for MiniMax H3 FL2VA."""

    _target: type[MiniMaxH3Transformer] = field(
        default_factory=lambda: MiniMaxH3Transformer
    )
    checkpoint_path: str | None = H3_TRANSFORMER_CHECKPOINT
    checkpoint_min_free_gb: float | None = None
    device: str = "cpu"
    execution_device: str = "cuda"
    sequential_cpu_offload: bool = True
    dtype: torch.dtype = torch.bfloat16
    attention_backend: Literal["flash", "cudnn", "efficient", "math"] = "flash"
    num_attention_heads: int = 56
    attention_head_dim: int = 128
    hidden_size: int = 5376
    num_layers: int = 50
    num_refiner_layers: int = 2
    ffn_dim: int = 14336
    in_channels: int = 24
    audio_in_channels: int = 32
    patch_size: tuple[int, int, int] = (1, 2, 2)
    text_dim: int = 5120
    freq_dim: int = 256
    time_embed_hidden_dim: int = 5376
    time_embed_dim: int = 2688
    rope_freq_dim: int = 16
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5
    qk_norm_eps: float = 1e-5
    final_norm_eps: float = 1e-5


def _module_dtype(module: nn.Module) -> torch.dtype:
    return next(module.parameters()).dtype


class _SwiGLU(nn.Module):
    def __init__(self, dim: int, inner_dim: int, **factory: Any) -> None:
        super().__init__()
        self.proj = nn.Linear(dim, inner_dim * 2, bias=False, **factory)

    def forward(self, hidden_states: Tensor) -> Tensor:
        value, gate = self.proj(hidden_states).chunk(2, dim=-1)
        return value * nn.functional.silu(gate)


class _FeedForward(nn.Module):
    def __init__(self, dim: int, inner_dim: int, **factory: Any) -> None:
        super().__init__()
        self.net = nn.ModuleList(
            [
                _SwiGLU(dim, inner_dim, **factory),
                nn.Dropout(0.0),
                nn.Linear(inner_dim, dim, bias=False, **factory),
            ]
        )

    def forward(self, hidden_states: Tensor) -> Tensor:
        for layer in self.net:
            hidden_states = layer(hidden_states)
        return hidden_states


class _TimestepEmbedding(nn.Module):
    def __init__(
        self, in_dim: int, hidden_dim: int, out_dim: int, **factory: Any
    ) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(in_dim, hidden_dim, **factory)
        self.act = nn.SiLU()
        self.linear_2 = nn.Linear(hidden_dim, out_dim, **factory)

    def forward(self, sample: Tensor) -> Tensor:
        return self.linear_2(self.act(self.linear_1(sample)))


class _RotaryEmbedding(nn.Module):
    def __init__(
        self,
        freq_dim: int,
        theta: float,
        *,
        device: torch.device,
    ) -> None:
        super().__init__()
        inv_freq = 1.0 / (
            theta
            ** (
                torch.arange(0, 2 * freq_dim, 2, dtype=torch.float32, device=device)
                / (2 * freq_dim)
            )
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, position_ids: Tensor) -> Tensor:
        inv_freq = cast(Tensor, self.inv_freq)
        frequencies = position_ids.float().unsqueeze(-1) * inv_freq.view(1, 1, -1)
        frequencies = torch.cat(frequencies.unbind(dim=1), dim=-1)
        return torch.cat((frequencies, frequencies), dim=-1)


def _apply_rotary(hidden_states: Tensor, frequencies: Tensor) -> Tensor:
    """Apply H3's partial RoPE in place with FlashDreams' CUDA kernel."""
    rotary_dim = frequencies.shape[-1]
    apply_rope_freqs(
        hidden_states[..., :rotary_dim],
        frequencies[:, None, None, :],
    )
    return hidden_states


class _Attention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        heads: int,
        head_dim: int,
        eps: float,
        backend: Literal["flash", "cudnn", "efficient", "math"],
        **factory: Any,
    ) -> None:
        super().__init__()
        inner_dim = heads * head_dim
        self.heads = heads
        self.to_q = nn.Linear(hidden_size, inner_dim, bias=False, **factory)
        self.to_k = nn.Linear(hidden_size, inner_dim, bias=False, **factory)
        self.to_v = nn.Linear(hidden_size, inner_dim, bias=False, **factory)
        self.norm_q = nn.RMSNorm(head_dim, eps=eps, **factory)
        self.norm_k = nn.RMSNorm(head_dim, eps=eps, **factory)
        self.to_out = nn.ModuleList(
            [nn.Linear(inner_dim, hidden_size, bias=False, **factory), nn.Dropout(0.0)]
        )
        self.attn_op = NativeAttention(qkv_format="bshd", backend=backend)

    def forward(self, hidden_states: Tensor, rotary: Tensor | None = None) -> Tensor:
        query = self.norm_q(self.to_q(hidden_states).unflatten(-1, (self.heads, -1)))
        key = self.norm_k(self.to_k(hidden_states).unflatten(-1, (self.heads, -1)))
        value = self.to_v(hidden_states).unflatten(-1, (self.heads, -1))
        if rotary is not None:
            query = _apply_rotary(query, rotary)
            key = _apply_rotary(key, rotary)
        output = self.attn_op(query, key, value).flatten(2, 3).type_as(query)
        return self.to_out[1](self.to_out[0](output))


class _RefinerBlock(nn.Module):
    def __init__(self, config: MiniMaxH3TransformerConfig, **factory: Any) -> None:
        super().__init__()
        self.norm1 = nn.RMSNorm(config.hidden_size, eps=config.norm_eps, **factory)
        self.attn = _Attention(
            config.hidden_size,
            config.num_attention_heads,
            config.attention_head_dim,
            config.qk_norm_eps,
            config.attention_backend,
            **factory,
        )
        self.norm2 = nn.RMSNorm(config.hidden_size, eps=config.norm_eps, **factory)
        self.ff = _FeedForward(config.hidden_size, config.ffn_dim, **factory)

    def forward(self, hidden_states: Tensor) -> Tensor:
        hidden_states = hidden_states + self.attn(self.norm1(hidden_states))
        return hidden_states + self.ff(self.norm2(hidden_states))


class _TokenRefiner(nn.Module):
    def __init__(self, config: MiniMaxH3TransformerConfig, **factory: Any) -> None:
        super().__init__()
        self.refiner_blocks = nn.ModuleList(
            [_RefinerBlock(config, **factory) for _ in range(config.num_refiner_layers)]
        )
        self.final_norm = nn.RMSNorm(
            config.hidden_size, eps=config.final_norm_eps, **factory
        )

    def forward(self, hidden_states: Tensor) -> Tensor:
        for block in self.refiner_blocks:
            hidden_states = block(hidden_states)
        return self.final_norm(hidden_states)


class _AdaLNProjection(nn.Module):
    def __init__(self, config: MiniMaxH3TransformerConfig, **factory: Any) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.linear = nn.Linear(
            config.time_embed_dim,
            6 * config.hidden_size * MODALITY_COUNT,
            **factory,
        )

    def forward(self, temb: Tensor) -> tuple[Tensor, ...]:
        output = self.linear(nn.functional.silu(temb).to(_module_dtype(self.linear)))
        return output.view(-1, 6 * self.hidden_size).chunk(6, dim=-1)


class _TransformerBlock(nn.Module):
    def __init__(self, config: MiniMaxH3TransformerConfig, **factory: Any) -> None:
        super().__init__()
        self.norm1 = nn.RMSNorm(config.hidden_size, eps=config.norm_eps, **factory)
        self.attn = _Attention(
            config.hidden_size,
            config.num_attention_heads,
            config.attention_head_dim,
            config.qk_norm_eps,
            config.attention_backend,
            **factory,
        )
        self.norm2 = nn.RMSNorm(config.hidden_size, eps=config.norm_eps, **factory)
        self.ff = _FeedForward(config.hidden_size, config.ffn_dim, **factory)
        self.adaln_proj = _AdaLNProjection(config, **factory)

    def forward(
        self,
        hidden_states: Tensor,
        temb: Tensor,
        adaln_indices: Tensor,
        rotary: Tensor,
    ) -> Tensor:
        shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = self.adaln_proj(temb)
        normalized = self.norm1(hidden_states)
        normalized = normalized * (1 + scale_a[adaln_indices]) + shift_a[adaln_indices]
        hidden_states = hidden_states + gate_a[adaln_indices] * self.attn(
            normalized, rotary
        )
        normalized = self.norm2(hidden_states)
        normalized = normalized * (1 + scale_m[adaln_indices]) + shift_m[adaln_indices]
        return hidden_states + gate_m[adaln_indices] * self.ff(normalized)


class _OutputNorm(nn.Module):
    def __init__(self, config: MiniMaxH3TransformerConfig, **factory: Any) -> None:
        super().__init__()
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.final_norm_eps, **factory)
        self.linear = nn.Linear(
            config.time_embed_dim, 2 * config.hidden_size, **factory
        )

    def forward(
        self, hidden_states: Tensor, temb: Tensor, timestep_indices: Tensor
    ) -> Tensor:
        shift, scale = self.linear(
            nn.functional.silu(temb).to(_module_dtype(self.linear))
        ).chunk(2, dim=-1)
        hidden_states = self.norm(hidden_states)
        return hidden_states * (1 + scale[timestep_indices]) + shift[timestep_indices]


class MiniMaxH3Transformer(Transformer[MiniMaxH3TransformerCache]):
    """Native FlashDreams transformer for H3's packed multimodal sequence."""

    config: MiniMaxH3TransformerConfig

    def __init__(self, config: MiniMaxH3TransformerConfig) -> None:
        super().__init__(config)
        self.config = config
        device = torch.device(config.device)
        low_precision: dict[str, Any] = {"device": device, "dtype": config.dtype}
        full_precision: dict[str, Any] = {
            "device": device,
            "dtype": torch.float32,
        }
        video_dim = config.in_channels * math.prod(config.patch_size)

        self.proj_in = nn.Linear(video_dim, config.hidden_size, **full_precision)
        self.audio_proj_in = nn.Linear(
            config.audio_in_channels, config.hidden_size, **full_precision
        )
        self.context_embedder = nn.Linear(
            config.text_dim, config.hidden_size, **low_precision
        )
        self.time_embedder = _TimestepEmbedding(
            config.freq_dim,
            config.time_embed_hidden_dim,
            config.time_embed_dim,
            **full_precision,
        )
        self.rope = _RotaryEmbedding(
            config.rope_freq_dim, config.rope_theta, device=device
        )
        self.token_refiner = _TokenRefiner(config, **low_precision)
        self.transformer_blocks = nn.ModuleList(
            [
                _TransformerBlock(config, **low_precision)
                for _ in range(config.num_layers)
            ]
        )
        self.norm_out = _OutputNorm(config, **low_precision)
        self.proj_out = nn.Linear(config.hidden_size, video_dim, **full_precision)
        self.audio_proj_out = nn.Linear(
            config.hidden_size, config.audio_in_channels, **full_precision
        )
        if config.checkpoint_path is not None:
            load_checkpoint(
                config.checkpoint_path,
                model=self,
                checkpoint_min_free_gb=config.checkpoint_min_free_gb,
            )
        self.eval()

    @property
    def latent_shape(self) -> tuple[int, ...]:
        return ()

    @staticmethod
    def _time_projection(timestep: Tensor, dim: int) -> Tensor:
        half = dim // 2
        exponent = (
            -math.log(10000)
            * torch.arange(half, dtype=torch.float32, device=timestep.device)
            / half
        )
        angles = timestep[:, None].float() * torch.exp(exponent)[None]
        embedding = torch.cat((angles.cos(), angles.sin()), dim=-1)
        if dim % 2:
            embedding = nn.functional.pad(embedding, (0, 1))
        return embedding

    def _run_on_execution_device(self, module: nn.Module, *args: Any) -> Any:
        """Run one weight group on the accelerator, then return it to CPU."""
        execution_device = torch.device(self.config.execution_device)
        if not self.config.sequential_cpu_offload:
            return module(*args)
        module.to(execution_device)
        try:
            return module(*args)
        finally:
            torch.cuda.synchronize(execution_device)
            module.to("cpu")
            torch.cuda.empty_cache()

    def forward_joint(
        self,
        hidden_states: Tensor,
        audio_hidden_states: Tensor,
        encoder_hidden_states: Tensor,
        timestep: Tensor,
        timestep_indices: Tensor,
        token_tags: Tensor,
        position_ids: Tensor,
        video_indices: Tensor,
        audio_indices: Tensor,
        text_indices: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Predict video and audio velocity for one packed H3 denoising step."""
        execution_device = torch.device(self.config.execution_device)
        if self.config.sequential_cpu_offload:
            hidden_states = hidden_states.to(execution_device)
            audio_hidden_states = audio_hidden_states.to(execution_device)
            encoder_hidden_states = encoder_hidden_states.to(execution_device)
            timestep = timestep.to(execution_device)
            timestep_indices = timestep_indices.to(execution_device)
            token_tags = token_tags.to(execution_device)
            position_ids = position_ids.to(execution_device)
            video_indices = video_indices.to(execution_device)
            audio_indices = audio_indices.to(execution_device)
            text_indices = text_indices.to(execution_device)
        sequence_length = position_ids.shape[0]
        if position_ids.shape != (sequence_length, 3):
            raise ValueError("position_ids must have shape [sequence_length, 3]")
        self.rope.to(execution_device)
        rotary = self.rope(position_ids)
        video = self._run_on_execution_device(
            self.proj_in, hidden_states.to(_module_dtype(self.proj_in))
        )
        audio = self._run_on_execution_device(
            self.audio_proj_in,
            audio_hidden_states.to(_module_dtype(self.audio_proj_in)),
        )
        text = self._run_on_execution_device(
            self.context_embedder,
            encoder_hidden_states.to(_module_dtype(self.context_embedder)),
        )
        text = self._run_on_execution_device(self.token_refiner, text)
        packed = text.new_zeros((text.shape[0], sequence_length, text.shape[-1]))
        packed = packed.index_copy(1, text_indices, text)
        packed = packed.index_copy(1, video_indices, video.to(text.dtype))
        packed = packed.index_copy(1, audio_indices, audio.to(text.dtype))

        temb = self._run_on_execution_device(
            self.time_embedder,
            self._time_projection(timestep, self.config.freq_dim).to(
                _module_dtype(self.time_embedder)
            ),
        )
        adaln_indices = timestep_indices * MODALITY_COUNT + token_tags
        for block in self.transformer_blocks:
            packed = self._run_on_execution_device(
                block, packed, temb, adaln_indices, rotary
            )
        packed = self._run_on_execution_device(
            self.norm_out, packed, temb, timestep_indices
        ).to(_module_dtype(self.proj_out))
        return (
            self._run_on_execution_device(self.proj_out, packed).index_select(
                1, video_indices
            ),
            self._run_on_execution_device(self.audio_proj_out, packed).index_select(
                1, audio_indices
            ),
        )

    def predict_flow(
        self,
        noisy_latent: Tensor,
        timestep: Tensor,
        cache: MiniMaxH3TransformerCache,
        input: object = None,
    ) -> Tensor:
        """Adapt the joint forward to FlashDreams' transformer interface."""
        del timestep, input
        video_flow, cache.last_audio_flow = self.forward_joint(
            noisy_latent,
            cache.audio_hidden_states,
            cache.encoder_hidden_states,
            cache.timestep,
            cache.timestep_indices,
            cache.token_tags,
            cache.position_ids,
            cache.video_indices,
            cache.audio_indices,
            cache.text_indices,
        )
        return video_flow

    def patchify_and_maybe_split_cp(self, x: object) -> object:
        """H3 inputs arrive already packed into sequence rows."""
        return x

    def unpatchify_and_maybe_gather_cp(self, x: Tensor) -> Tensor:
        """H3's custom pipeline unpacks rows after paired denoising."""
        return x


__all__ = [
    "H3_REF_TRANSFORMER_CHECKPOINT",
    "H3_TRANSFORMER_CHECKPOINT",
    "MiniMaxH3Transformer",
    "MiniMaxH3TransformerCache",
    "MiniMaxH3TransformerConfig",
]
