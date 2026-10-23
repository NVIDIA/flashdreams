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

"""AgiBot mixture-of-transformers network with action-conditioned causal video."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from flashdreams.core.attention import BlockKVCache
from flashdreams.infra.config import InstantiateConfig

from ..attention import (
    cached_joint_attention,
    valid_length_cached_attention,
)
from ..spec import SPEC, Cosmos3InteractiveSpec, LLMSpec
from .rope import mrope_cos_sin


def _rotate_half(x: Tensor) -> Tensor:
    """Rotate the two halves of the last dim (HF Qwen3 convention)."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rope(
    q: Tensor, k: Tensor, cos: Tensor, sin: Tensor
) -> tuple[Tensor, Tensor]:
    """Apply RoPE to ``[B, H, S, d]`` q/k using ``[B, S, d]`` cos/sin (HF convention)."""
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    q_out = (q * cos) + (_rotate_half(q) * sin)
    k_out = (k * cos) + (_rotate_half(k) * sin)
    return q_out, k_out


def _repeat_kv(x: Tensor, n_rep: int) -> Tensor:
    """Expand GQA KV heads ``[B, Hkv, S, d]`` -> ``[B, Hkv*n_rep, S, d]``."""
    if n_rep == 1:
        return x
    b, h, s, d = x.shape
    return x[:, :, None, :, :].expand(b, h, n_rep, s, d).reshape(b, h * n_rep, s, d)


def _kv_for_attention(
    k: Tensor,
    v: Tensor,
    n_rep: int,
    *,
    native_gqa_attention: bool,
) -> tuple[Tensor, Tensor]:
    if native_gqa_attention:
        return k, v
    return _repeat_kv(k, n_rep), _repeat_kv(v, n_rep)


class RMSNorm(nn.Module):
    """Weight-only RMSNorm (matches the text-tower norms; no bias in checkpoint)."""

    def __init__(self, dim: int, *, eps: float, dtype: torch.dtype) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim, dtype=dtype))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.to(dtype)) * self.weight


class _Attention(nn.Module):
    """Qwen3 grouped-query attention with per-head QK-norm; projections unbiased.

    Matches ``transformers`` ``Qwen3Attention``: QK-norm on the head-dim view
    before RoPE, ``rotate_half`` RoPE, GQA ``repeat_kv``, scale ``head_dim**-0.5``.
    """

    def __init__(self, llm: LLMSpec, dtype: torch.dtype) -> None:
        super().__init__()
        self.num_heads = llm.num_attention_heads
        self.num_kv_heads = llm.num_key_value_heads
        self.head_dim = llm.head_dim
        self.num_kv_groups = llm.num_attention_heads // llm.num_key_value_heads
        q_dim = llm.num_attention_heads * llm.head_dim
        kv_dim = llm.num_key_value_heads * llm.head_dim
        self.q_proj = nn.Linear(
            llm.hidden_size, q_dim, bias=llm.attention_bias, dtype=dtype
        )
        self.k_proj = nn.Linear(
            llm.hidden_size, kv_dim, bias=llm.attention_bias, dtype=dtype
        )
        self.v_proj = nn.Linear(
            llm.hidden_size, kv_dim, bias=llm.attention_bias, dtype=dtype
        )
        self.o_proj = nn.Linear(
            q_dim, llm.hidden_size, bias=llm.attention_bias, dtype=dtype
        )
        self.q_norm = RMSNorm(llm.head_dim, eps=llm.rms_norm_eps, dtype=dtype)
        self.k_norm = RMSNorm(llm.head_dim, eps=llm.rms_norm_eps, dtype=dtype)

    def forward(
        self, x: Tensor, cos: Tensor, sin: Tensor, *, is_causal: bool = False
    ) -> Tensor:
        b, s, _ = x.shape
        shape = (b, s, -1, self.head_dim)
        q = self.q_norm(self.q_proj(x).view(shape)).transpose(1, 2)  # [B, Hq, S, d]
        k = self.k_norm(self.k_proj(x).view(shape)).transpose(1, 2)  # [B, Hkv, S, d]
        v = self.v_proj(x).view(shape).transpose(1, 2)
        q, k = _apply_rope(q, k, cos, sin)
        k = _repeat_kv(k, self.num_kv_groups)
        v = _repeat_kv(v, self.num_kv_groups)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)
        out = out.transpose(1, 2).reshape(b, s, -1)
        return self.o_proj(out)


class _SwiGLUMLP(nn.Module):
    """SwiGLU FFN (gate/up/down), unbiased — matches the text tower."""

    def __init__(self, llm: LLMSpec, dtype: torch.dtype) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(
            llm.hidden_size, llm.intermediate_size, bias=False, dtype=dtype
        )
        self.up_proj = nn.Linear(
            llm.hidden_size, llm.intermediate_size, bias=False, dtype=dtype
        )
        self.down_proj = nn.Linear(
            llm.intermediate_size, llm.hidden_size, bias=False, dtype=dtype
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class _DecoderHalf(nn.Module):
    """One MoT pathway (understanding *or* generation) of a decoder layer.

    Each pathway carries its own normalization, attention, and MLP parameters.
    The MoT layer combines the understanding and generation attention paths.
    """

    def __init__(self, llm: LLMSpec, dtype: torch.dtype) -> None:
        super().__init__()
        self.input_layernorm = RMSNorm(
            llm.hidden_size, eps=llm.rms_norm_eps, dtype=dtype
        )
        self.self_attn = _Attention(llm, dtype)
        self.post_attention_layernorm = RMSNorm(
            llm.hidden_size, eps=llm.rms_norm_eps, dtype=dtype
        )
        self.mlp = _SwiGLUMLP(llm, dtype)

    def forward(
        self, x: Tensor, cos: Tensor, sin: Tensor, *, is_causal: bool = False
    ) -> Tensor:
        x = x + self.self_attn(self.input_layernorm(x), cos, sin, is_causal=is_causal)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


def _route(x: Tensor, is_gen: Tensor, und_out: Tensor, gen_out: Tensor) -> Tensor:
    """Select per-token between und/gen results using ``is_gen`` ``[S]``.

    Correctness-first: both pathways are computed on the full tensor and then
    selected (2x projection compute). The production path will gather tokens by
    type instead; the math is identical. ``is_gen`` broadcasts over the leading
    batch dim and any trailing dims.
    """
    shape = [1, -1] + [1] * (x.ndim - 2)
    return torch.where(is_gen.view(shape), gen_out, und_out)


class _MoTLayer(nn.Module):
    """A Mixture-of-Transformers layer: parallel understanding + generation halves.

    The understanding and generation token streams use separate weights (the und
    half vs the ``_moe_gen`` half) but share **one joint attention** (und causal
    self; gen full-cross-to-und + temporal-causal self). ``forward`` operates on
    a packed ``[B, S, D]`` sequence with a per-token ``is_gen`` tag.
    """

    def __init__(self, llm: LLMSpec, dtype: torch.dtype) -> None:
        super().__init__()
        self.und = _DecoderHalf(llm, dtype)
        self.gen = _DecoderHalf(llm, dtype)

    def step(
        self,
        h: Tensor,
        is_gen: Tensor,
        cos: Tensor,
        sin: Tensor,
        cache: dict[str, Tensor],
        *,
        append: bool,
        within_mask: Tensor | None,
        zero_value_mask: Tensor | None = None,
        native_gqa_attention: bool = False,
    ) -> Tensor:
        """Incremental (KV-cache) MoT layer over the *current* tokens only.

        The current tokens are a single uniform stream (all text, or all one
        chunk's vision), so ``is_gen`` is uniform and ``_route`` selects one half.
        Queries attend to this layer's cached K/V (``cache['k']``/``['v']``,
        post-RoPE, un-repeated) plus the current tokens. When ``append`` is set,
        the current K/V are written into the cache (text prefill + clean-frame
        seeding); denoise steps pass ``append=False`` so the noisy current frame
        never pollutes the cache.

        ``zero_value_mask`` is an optional ``[S]`` bool selecting current
        positions whose value projection is zeroed *in the cache only*; the
        AgiBot pathway uses it for null-action tokens.
        """
        ua, ga = self.und.self_attn, self.gen.self_attn
        b, s, _ = h.shape
        hd = ua.head_dim

        hn = _route(h, is_gen, self.und.input_layernorm(h), self.gen.input_layernorm(h))
        q = _route(hn, is_gen, ua.q_proj(hn), ga.q_proj(hn)).view(
            b, s, ua.num_heads, hd
        )
        k = _route(hn, is_gen, ua.k_proj(hn), ga.k_proj(hn)).view(
            b, s, ua.num_kv_heads, hd
        )
        v = _route(hn, is_gen, ua.v_proj(hn), ga.v_proj(hn)).view(
            b, s, ua.num_kv_heads, hd
        )
        q = _route(q, is_gen, ua.q_norm(q), ga.q_norm(q))
        k = _route(k, is_gen, ua.k_norm(k), ga.k_norm(k))

        q = q.transpose(1, 2)  # [B, Hkv-grouped later, s, d]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        q, k = _apply_rope(q, k, cos, sin)  # RoPE with the current tokens' positions
        cache_v = v
        if zero_value_mask is not None:
            zero_value_mask = zero_value_mask.to(device=v.device, dtype=torch.bool)
            if zero_value_mask.shape != (s,):
                raise ValueError(
                    f"zero_value_mask must have shape {(s,)}, got {tuple(zero_value_mask.shape)}"
                )
            cache_v = v.clone()
            cache_v[:, :, zero_value_mask] = 0

        ck, cv = cache.get("k"), cache.get("v")
        k_all = k if ck is None else torch.cat([ck, k], dim=2)
        # Current attention uses the raw value projection; the zeroing applies only
        # to the detached value tensor persisted for future frames.
        v_all = v if cv is None else torch.cat([cv, v], dim=2)
        n_cached = 0 if ck is None else int(ck.shape[2])
        if append:
            cache_v_all = cache_v if cv is None else torch.cat([cv, cache_v], dim=2)
            cache["k"], cache["v"] = k_all.detach(), cache_v_all.detach()

        kr, vr = _kv_for_attention(
            k_all, v_all, ua.num_kv_groups, native_gqa_attention=native_gqa_attention
        )
        attn = cached_joint_attention(
            q, kr, vr, n_cached, within_mask, enable_gqa=native_gqa_attention
        )  # [B, Hq, s, d]
        attn = attn.transpose(1, 2).reshape(b, s, -1)

        o = _route(attn, is_gen, ua.o_proj(attn), ga.o_proj(attn))
        h = h + o
        hn2 = _route(
            h,
            is_gen,
            self.und.post_attention_layernorm(h),
            self.gen.post_attention_layernorm(h),
        )
        mlp = _route(hn2, is_gen, self.und.mlp(hn2), self.gen.mlp(hn2))
        return h + mlp

    # BlockKVCache: fixed text K/V with rolling generation history.
    # The MoT joint attention splits cleanly: understanding (text) tokens never
    # attend to generation, and generation attends to *all* text + temporal-causal
    # gen. So text K/V (from the und half) are computed once and pinned; gen K/V
    # (from the gen half) live in a rolling BlockKVCache. This avoids the ``_route``
    # double-compute *and* is CUDA-graph friendly (fixed buffers at steady state).

    def text_prefill_kv(
        self,
        h: Tensor,
        cos: Tensor,
        sin: Tensor,
        within_mask: Tensor,
        *,
        native_gqa_attention: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Understanding-half forward over text; returns ``(h_out, k_und, v_und)``.

        ``k_und``/``v_und`` are the RoPE'd, un-repeated ``[B, Hkv, n_text, d]`` K/V
        to pin as the always-visible cross-context for generation tokens.
        """
        ua = self.und.self_attn
        b, s, _ = h.shape
        hd = ua.head_dim
        hn = self.und.input_layernorm(h)
        q = ua.q_norm(ua.q_proj(hn).view(b, s, ua.num_heads, hd)).transpose(1, 2)
        k = ua.k_norm(ua.k_proj(hn).view(b, s, ua.num_kv_heads, hd)).transpose(1, 2)
        v = ua.v_proj(hn).view(b, s, ua.num_kv_heads, hd).transpose(1, 2)
        q, k = _apply_rope(q, k, cos, sin)
        kr, vr = _kv_for_attention(
            k, v, ua.num_kv_groups, native_gqa_attention=native_gqa_attention
        )
        attn = cached_joint_attention(
            q, kr, vr, 0, within_mask, enable_gqa=native_gqa_attention
        )  # text causal self
        attn = attn.transpose(1, 2).reshape(b, s, -1)
        h = h + ua.o_proj(attn)
        h = h + self.und.mlp(self.und.post_attention_layernorm(h))
        return h, k, v

    def gen_step(
        self,
        h: Tensor,
        cos: Tensor,
        sin: Tensor,
        text_k: Tensor | None,
        text_v: Tensor | None,
        gen_cache: BlockKVCache,
        within_mask: Tensor | None,
        *,
        native_gqa_attention: bool = False,
    ) -> Tensor:
        """Generation-half forward for the current chunk against the cache.

        Writes the current chunk's (RoPE'd) K/V into ``gen_cache`` (overwriting on
        each denoise step; the final clean write persists), then attends to
        ``[pinned text K/V | cached gen K/V (incl. current)]``. ``before_update`` /
        ``after_update`` are driven once per chunk by the rollout, not here.
        """
        ga = self.gen.self_attn
        b, s, _ = h.shape
        hd = ga.head_dim
        hn = self.gen.input_layernorm(h)
        q = ga.q_norm(ga.q_proj(hn).view(b, s, ga.num_heads, hd)).transpose(1, 2)
        k = ga.k_norm(ga.k_proj(hn).view(b, s, ga.num_kv_heads, hd)).transpose(1, 2)
        v = ga.v_proj(hn).view(b, s, ga.num_kv_heads, hd).transpose(1, 2)
        q, k = _apply_rope(q, k, cos, sin)

        gen_cache.update(k, v)  # [B, Hkv, chunk, d] into the rolling window
        gk, gv = (
            gen_cache.cached_k(),
            gen_cache.cached_v(),
        )  # incl. current chunk at tail
        if text_k is not None:
            assert text_v is not None
            k_all = torch.cat([text_k, gk], dim=2)
            v_all = torch.cat([text_v, gv], dim=2)
        else:
            k_all, v_all = gk, gv
        n_cached = (
            int(k_all.shape[2]) - s
        )  # everything but the current chunk is fully visible
        kr, vr = _kv_for_attention(
            k_all, v_all, ga.num_kv_groups, native_gqa_attention=native_gqa_attention
        )
        attn = cached_joint_attention(
            q, kr, vr, n_cached, within_mask, enable_gqa=native_gqa_attention
        )
        attn = attn.transpose(1, 2).reshape(b, s, -1)
        h = h + ga.o_proj(attn)
        h = h + self.gen.mlp(self.gen.post_attention_layernorm(h))
        return h

    def gen_compute(
        self,
        h: Tensor,
        cos: Tensor,
        sin: Tensor,
        text_k: Tensor | None,
        text_v: Tensor | None,
        hist_k: Tensor | None,
        hist_v: Tensor | None,
        within_mask: Tensor | None,
        *,
        cached_mask: Tensor | None = None,
        cached_valid_length: Tensor | None = None,
        native_gqa_attention: bool = False,
    ) -> Tensor:
        """Break-free generation-half forward (no cache object → ``torch.compile``-clean).

        Same math as :meth:`gen_step` but the K/V history is passed in as plain
        tensors (``hist_k``/``hist_v``, gathered eagerly by the caller) instead of
        read from a :class:`BlockKVCache`. This keeps ``torch.sym_min`` and the
        data-dependent cache slices out of the traced graph, so ``torch.compile``
        fuses the whole layer and CUDA-graph capture covers the real compute.
        """
        ga = self.gen.self_attn
        b, s, _ = h.shape
        hd = ga.head_dim
        hn = self.gen.input_layernorm(h)
        q = ga.q_norm(ga.q_proj(hn).view(b, s, ga.num_heads, hd)).transpose(1, 2)
        k = ga.k_norm(ga.k_proj(hn).view(b, s, ga.num_kv_heads, hd)).transpose(1, 2)
        v = ga.v_proj(hn).view(b, s, ga.num_kv_heads, hd).transpose(1, 2)
        q, k = _apply_rope(q, k, cos, sin)

        parts_k = [t for t in (text_k, hist_k) if t is not None] + [k]
        parts_v = [t for t in (text_v, hist_v) if t is not None] + [v]
        k_all = torch.cat(parts_k, dim=2) if len(parts_k) > 1 else k
        v_all = torch.cat(parts_v, dim=2) if len(parts_v) > 1 else v
        n_cached = int(k_all.shape[2]) - s
        if cached_valid_length is not None:
            if cached_mask is not None or within_mask is not None:
                raise ValueError(
                    "valid-length cache attention only supports an unmasked "
                    "current chunk"
                )
            # varlen attention supports GQA directly; keep the smaller native
            # K/V head count instead of materializing the legacy repeated heads.
            attn = valid_length_cached_attention(
                q, k_all, v_all, cached_valid_length, s
            )
        else:
            kr, vr = _kv_for_attention(
                k_all,
                v_all,
                ga.num_kv_groups,
                native_gqa_attention=native_gqa_attention,
            )
            attn = cached_joint_attention(
                q,
                kr,
                vr,
                n_cached,
                within_mask,
                cached_mask=cached_mask,
                enable_gqa=native_gqa_attention,
            )
        attn = attn.transpose(1, 2).reshape(b, s, -1)
        h = h + ga.o_proj(attn)
        h = h + self.gen.mlp(self.gen.post_attention_layernorm(h))
        return h

    def gen_compute_with_kv(
        self,
        h: Tensor,
        cos: Tensor,
        sin: Tensor,
        text_k: Tensor | None,
        text_v: Tensor | None,
        hist_k: Tensor | None,
        hist_v: Tensor | None,
        within_mask: Tensor | None,
        *,
        cached_valid_length: Tensor | None = None,
        causal_block_tokens: int | None = None,
        native_gqa_attention: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Generation-half forward and return the current chunk's RoPE'd K/V.

        This is used when the caller needs to seed a non-standard chunk into a
        fixed stream cache, such as the camera path's one-frame sink followed by
        four-frame vision+action chunks.
        """
        ga = self.gen.self_attn
        b, s, _ = h.shape
        hd = ga.head_dim
        hn = self.gen.input_layernorm(h)
        q = ga.q_norm(ga.q_proj(hn).view(b, s, ga.num_heads, hd)).transpose(1, 2)
        k = ga.k_norm(ga.k_proj(hn).view(b, s, ga.num_kv_heads, hd)).transpose(1, 2)
        v = ga.v_proj(hn).view(b, s, ga.num_kv_heads, hd).transpose(1, 2)
        q, k = _apply_rope(q, k, cos, sin)

        parts_k = [t for t in (text_k, hist_k) if t is not None] + [k]
        parts_v = [t for t in (text_v, hist_v) if t is not None] + [v]
        k_all = torch.cat(parts_k, dim=2) if len(parts_k) > 1 else k
        v_all = torch.cat(parts_v, dim=2) if len(parts_v) > 1 else v
        n_cached = int(k_all.shape[2]) - s
        if cached_valid_length is not None:
            if within_mask is not None:
                raise ValueError(
                    "valid-length cache attention expresses causality through "
                    "causal_block_tokens, not within_mask"
                )
            attn = valid_length_cached_attention(
                q,
                k_all,
                v_all,
                cached_valid_length,
                s,
                causal_block_tokens=causal_block_tokens,
            )
        else:
            kr, vr = _kv_for_attention(
                k_all,
                v_all,
                ga.num_kv_groups,
                native_gqa_attention=native_gqa_attention,
            )
            attn = cached_joint_attention(
                q, kr, vr, n_cached, within_mask, enable_gqa=native_gqa_attention
            )
        attn = attn.transpose(1, 2).reshape(b, s, -1)
        h = h + ga.o_proj(attn)
        h = h + self.gen.mlp(self.gen.post_attention_layernorm(h))
        return h, k.detach(), v.detach()


def _timestep_embedding(t: Tensor, dim: int, max_period: int = 10000) -> Tensor:
    """DiT-style sinusoidal timestep embedding ``[N] -> [N, dim]`` (cos then sin)."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(0, half, dtype=torch.float32, device=t.device)
        / half
    )  # [dim/2]
    args = t[:, None].float() * freqs[None]  # [N, dim/2]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)  # [N, dim]
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


class _TimeEmbedder(nn.Module):
    """Diffusion timestep MLP: sinusoidal(freq) -> Linear -> SiLU -> Linear (DiT)."""

    def __init__(self, spec: Cosmos3InteractiveSpec, dtype: torch.dtype) -> None:
        super().__init__()
        self.freq_dim = spec.vfm.time_embed_freq_dim
        self.mlp = nn.Sequential(
            nn.Linear(spec.vfm.time_embed_freq_dim, spec.llm.hidden_size, dtype=dtype),
            nn.SiLU(),
            nn.Linear(spec.llm.hidden_size, spec.llm.hidden_size, dtype=dtype),
        )

    def forward(self, t: Tensor) -> Tensor:
        """``t`` ``[N]`` (already scaled by ``timestep_scale``) -> ``[N, hidden]``."""
        t_freq = _timestep_embedding(t, self.freq_dim).to(
            cast(nn.Linear, self.mlp[0]).weight.dtype
        )
        return self.mlp(t_freq)


class _VisionAttention(nn.Module):
    def __init__(self, dim: int, dtype: torch.dtype) -> None:
        super().__init__()
        self.qkv = nn.Linear(dim, dim * 3, bias=True, dtype=dtype)
        self.proj = nn.Linear(dim, dim, bias=True, dtype=dtype)


class _VisionMLP(nn.Module):
    def __init__(self, dim: int, hidden: int, dtype: torch.dtype) -> None:
        super().__init__()
        self.linear_fc1 = nn.Linear(dim, hidden, bias=True, dtype=dtype)
        self.linear_fc2 = nn.Linear(hidden, dim, bias=True, dtype=dtype)


class _VisionBlock(nn.Module):
    def __init__(self, spec: Cosmos3InteractiveSpec, dtype: torch.dtype) -> None:
        super().__init__()
        v = spec.vision
        self.norm1 = nn.LayerNorm(v.hidden_size, dtype=dtype)
        self.attn = _VisionAttention(v.hidden_size, dtype)
        self.norm2 = nn.LayerNorm(v.hidden_size, dtype=dtype)
        self.mlp = _VisionMLP(v.hidden_size, v.intermediate_size, dtype)


class _PatchEmbed(nn.Module):
    """ViT patch embed: Conv3d over (temporal_patch, patch, patch) voxels."""

    def __init__(self, spec: Cosmos3InteractiveSpec, dtype: torch.dtype) -> None:
        super().__init__()
        v = spec.vision
        self.proj = nn.Conv3d(
            v.in_channels,
            v.hidden_size,
            kernel_size=(v.temporal_patch_size, v.patch_size, v.patch_size),
            stride=(v.temporal_patch_size, v.patch_size, v.patch_size),
            dtype=dtype,
        )


class _PatchMerger(nn.Module):
    """Merger head: LayerNorm(norm_dim) then Linear(merged -> merged -> out)."""

    def __init__(
        self, *, norm_dim: int, merged_dim: int, out_dim: int, dtype: torch.dtype
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(norm_dim, dtype=dtype)
        self.linear_fc1 = nn.Linear(merged_dim, merged_dim, bias=True, dtype=dtype)
        self.linear_fc2 = nn.Linear(merged_dim, out_dim, bias=True, dtype=dtype)


class _VisionTower(nn.Module):
    """Qwen3-VL ViT: Conv3d patch embed + learned pos-embed + blocks + mergers."""

    def __init__(self, spec: Cosmos3InteractiveSpec, dtype: torch.dtype) -> None:
        super().__init__()
        v = spec.vision
        merged_dim = v.hidden_size * v.spatial_merge_size**2  # 1152 * 4 = 4608

        self.patch_embed = _PatchEmbed(spec, dtype)
        self.pos_embed = nn.Embedding(
            v.num_position_embeddings, v.hidden_size, dtype=dtype
        )
        self.blocks = nn.ModuleList([_VisionBlock(spec, dtype) for _ in range(v.depth)])
        # Main merger normalizes at the per-patch hidden size (1152), then merges.
        self.merger = _PatchMerger(
            norm_dim=v.hidden_size,
            merged_dim=merged_dim,
            out_dim=v.out_hidden_size,
            dtype=dtype,
        )
        # Deepstack mergers normalize at the merged size (4608).
        self.deepstack_merger_list = nn.ModuleList(
            _PatchMerger(
                norm_dim=merged_dim,
                merged_dim=merged_dim,
                out_dim=v.out_hidden_size,
                dtype=dtype,
            )
            for _ in range(len(v.deepstack_visual_indexes))
        )


@dataclass
class _MoTStreamCache:
    """Per-network streaming cache for the BlockKVCache rollout.

    ``text_k``/``text_v`` are the pinned, always-visible understanding K/V (one
    entry per layer, or ``None`` when there is no text). ``gen`` is one rolling
    :class:`BlockKVCache` per layer holding the generation stream's clean K/V.
    """

    text_k: list[Tensor | None]
    text_v: list[Tensor | None]
    gen: list[BlockKVCache]

    def reset(self) -> None:
        for c in self.gen:
            c.reset()


class DomainAwareLinear(nn.Module):
    """Per-domain linear stored as embeddings (X-VLA style).

    ``fc`` holds the flattened per-domain weight ``[num_domains, out*in]`` and
    ``bias`` holds ``[num_domains, out]``; a per-call ``domain_id`` (one per batch
    row) selects the embodiment's weight. Camera conditioning uses a single fixed
    domain, so at inference ``domain_id`` is a length-1 tensor.
    """

    def __init__(
        self, input_size: int, output_size: int, num_domains: int, dtype: torch.dtype
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.num_domains = num_domains
        self.fc = nn.Embedding(num_domains, output_size * input_size, dtype=dtype)
        self.bias = nn.Embedding(num_domains, output_size, dtype=dtype)

    def forward(self, x: Tensor, domain_id: Tensor) -> Tensor:
        """``x``: ``[B, I]`` or ``[B, T, I]``; ``domain_id``: ``[B]`` → same rank out."""
        b = int(domain_id.shape[0])
        w = self.fc(domain_id).view(b, self.input_size, self.output_size)  # [B,I,O]
        bias = self.bias(domain_id).view(b, self.output_size)  # [B,O]
        if x.dim() == 2:
            return torch.bmm(x.unsqueeze(1), w).squeeze(1) + bias  # [B,O]
        return torch.bmm(x, w) + bias.unsqueeze(1)  # [B,T,O]


@dataclass(kw_only=True)
class CosmosMoTNetworkConfig(InstantiateConfig):
    """Config for :class:`CosmosMoTNetwork`.

    Defaults reproduce the exported stage-3 8B no-action checkpoint; the frozen
    :class:`Cosmos3InteractiveSpec` carries every architectural dim.
    """

    _target: type["CosmosMoTNetwork"] = field(default_factory=lambda: CosmosMoTNetwork)
    spec: Cosmos3InteractiveSpec = field(default_factory=lambda: SPEC)
    dtype: torch.dtype = torch.bfloat16
    include_vision_tower: bool = True
    """Build the ViT tower (needed for I2V and for a full-checkpoint load).
    Text-to-video rollouts do not exercise it, but the exported checkpoint
    contains its weights, so it must exist for a strict ``load_state_dict``."""

    action_gen: bool = False
    """Build the action-conditioning pathway (``action2llm`` / ``llm2action`` /
    ``action_modality_embed``). Required to strict-load the camera-conditioned
    checkpoint (``causal_8b_sf_dmd_cam_*``); the no-action checkpoint omits these."""
    action_dim: int = 64
    """Action-vector width (camera checkpoint = 64)."""
    num_embodiment_domains: int = 32
    """Number of domain rows in the ``DomainAwareLinear`` embeddings (camera ckpt = 32)."""


class CosmosMoTNetwork(nn.Module):
    """AgiBot action-conditioned MoT network with incremental KV-cache inference.

    Reproduces the exported checkpoint's ``state_dict`` after ``checkpoint_remap``
    so ``load_state_dict`` is a strict, unambiguous load.
    """

    def __init__(self, config: CosmosMoTNetworkConfig) -> None:
        super().__init__()
        self.config = config
        spec = config.spec
        dtype = config.dtype
        llm = spec.llm

        self.embed_tokens = nn.Embedding(llm.vocab_size, llm.hidden_size, dtype=dtype)
        self.layers = nn.ModuleList(
            _MoTLayer(llm, dtype) for _ in range(llm.num_layers)
        )
        self.norm = RMSNorm(llm.hidden_size, eps=llm.rms_norm_eps, dtype=dtype)
        self.norm_moe_gen = RMSNorm(llm.hidden_size, eps=llm.rms_norm_eps, dtype=dtype)
        self.lm_head = nn.Linear(
            llm.hidden_size, llm.vocab_size, bias=False, dtype=dtype
        )

        # VFM adapters (video latent <-> LLM space) + diffusion timestep MLP.
        self.vae2llm = nn.Linear(
            spec.vfm.patch_latent_dim, llm.hidden_size, bias=True, dtype=dtype
        )
        self.llm2vae = nn.Linear(
            llm.hidden_size, spec.vfm.patch_latent_dim, bias=True, dtype=dtype
        )
        self.time_embedder = _TimeEmbedder(spec, dtype)

        if config.include_vision_tower:
            self.visual = _VisionTower(spec, dtype)

        # Action-conditioning pathway (camera checkpoint). ``action2llm`` encodes
        # the (camera-pose-derived) action vector into an LLM token; the clean
        # action tokens are packed alongside vision and attended by generation.
        # ``llm2action`` is present in the checkpoint (action-gen training) but
        # unused for forward-dynamics inference, where actions are the condition.
        if config.action_gen:
            self.action2llm = DomainAwareLinear(
                config.action_dim, llm.hidden_size, config.num_embodiment_domains, dtype
            )
            self.llm2action = DomainAwareLinear(
                llm.hidden_size, config.action_dim, config.num_embodiment_domains, dtype
            )
            self.action_modality_embed = nn.Parameter(
                torch.zeros(llm.hidden_size, dtype=dtype)
            )

    def encode_action(
        self, action_tokens: Tensor, domain_id: int | Tensor = 2
    ) -> Tensor:
        """Encode raw action vectors ``[N_act, action_dim]`` -> LLM tokens ``[N_act, hidden]``.

        Uses domain-aware ``action2llm`` and ``action_modality_embed`` for
        clean forward-dynamics conditioning (no timestep embed).
        ``domain_id`` selects one embodiment for all actions or one per row.
        """
        device = self.embed_tokens.weight.device
        dtype = self.config.dtype
        n = int(action_tokens.shape[0])
        if isinstance(domain_id, Tensor):
            if domain_id.dtype != torch.long:
                raise TypeError("action domain IDs must be an int64 Tensor")
            if domain_id.shape != (n,):
                raise ValueError("action domain IDs must have one entry per action row")
            did = domain_id.to(device=device)
        else:
            did = torch.full((n,), int(domain_id), dtype=torch.long, device=device)
        a = self.action2llm(
            action_tokens.to(device=device, dtype=dtype), did
        )  # [N_act, hidden]
        return a + self.action_modality_embed.view(1, -1)

    def embed_uniform_timestep(self, timestep: Tensor, count: int) -> Tensor:
        """Embed one scalar timestep, then expand it across uniform vision tokens.

        A chunk denoised at a single sigma shares one timestep across all of its
        vision tokens. Embedding the scalar once and expanding keeps that exact,
        rather than relying on a per-token tensor that must stay uniform.
        """
        count = int(count)
        if count < 0:
            raise ValueError(
                f"timestep expansion count must be non-negative, got {count}"
            )
        scalar = torch.as_tensor(
            timestep, device=self.embed_tokens.weight.device, dtype=torch.float32
        ).reshape(-1)
        if scalar.numel() != 1:
            raise ValueError(
                f"uniform timestep must contain one value, got {scalar.numel()}"
            )
        return self.time_embedder(scalar * self.config.spec.vfm.timestep_scale).expand(
            count, -1
        )

    def _assemble_input(
        self,
        text_ids: Tensor | None,
        vision_patches: Tensor,
        vision_timestep: Tensor,
        is_gen: Tensor,
        vision_noisy_mask: Tensor | None = None,
        *,
        action_tokens: Tensor | None = None,
        is_action: Tensor | None = None,
        action_domain_id: int | Tensor = 2,
        uniform_vision_timestep: bool = False,
    ) -> Tensor:
        """Build the packed ``[1, S, hidden]`` token stream fed to the layers.

        ``embed_tokens`` fills understanding positions; ``vae2llm(vision_patches)
        + time_embedder(t * timestep_scale)`` fills generation positions. This is
        the transformer input shared by incremental cache and denoising calls.
        """
        llm = self.config.spec.llm
        vfm = self.config.spec.vfm
        device = self.embed_tokens.weight.device
        dtype = self.config.dtype
        s = int(is_gen.shape[0])

        is_gen = is_gen.to(device)
        h = torch.zeros(1, s, llm.hidden_size, device=device, dtype=dtype)
        und = ~is_gen
        if text_ids is not None and bool(und.any()):
            h[:, und] = self.embed_tokens(text_ids.to(device)).to(dtype)

        # Vision fills gen∧¬action positions; action fills gen∧action positions.
        vis_gen = is_gen if is_action is None else (is_gen & ~is_action.to(device))
        vis = self.vae2llm(
            vision_patches.to(device=device, dtype=dtype)
        )  # [N_vis, hidden]
        te = (
            self.embed_uniform_timestep(
                vision_timestep.reshape(-1)[0], int(vision_patches.shape[0])
            )
            if uniform_vision_timestep
            else self.time_embedder(vision_timestep.to(device) * vfm.timestep_scale)
        ).to(dtype)
        if vision_noisy_mask is not None:
            te = te * vision_noisy_mask.to(device=device, dtype=dtype)[:, None]
        vis = vis + te
        h[:, vis_gen] = vis.unsqueeze(0)

        if action_tokens is not None and is_action is not None:
            a = self.encode_action(
                action_tokens, domain_id=action_domain_id
            )  # [N_act, hidden]
            h[:, is_action.to(device)] = a.unsqueeze(0)
        return h

    # Cache clean K/V for text and past frames; only the current chunk generates
    # new queries. Temporal causality keeps cached context stable across chunks.

    def new_kv_cache(self) -> list[dict[str, Tensor]]:
        """A fresh per-layer KV cache (one empty dict per MoT layer)."""
        return [{} for _ in self.layers]

    def _layers_step(
        self,
        h: Tensor,
        is_gen: Tensor,
        position_ids: Tensor,
        cache: list[dict[str, Tensor]],
        *,
        append: bool,
        within_mask: Tensor | None,
        zero_value_mask: Tensor | None = None,
        native_gqa_attention: bool = False,
    ) -> Tensor:
        llm = self.config.spec.llm
        device = self.embed_tokens.weight.device
        dtype = self.config.dtype
        is_gen = is_gen.to(device)
        h = h.to(device=device, dtype=dtype)
        cos, sin = mrope_cos_sin(
            position_ids.to(device), llm.head_dim, llm.rope_theta, llm.mrope_section
        )
        cos = cos.to(dtype)[None]
        sin = sin.to(dtype)[None]
        wm = within_mask.to(device) if within_mask is not None else None
        for layer, layer_cache in zip(cast(list[_MoTLayer], self.layers), cache):
            h = layer.step(
                h,
                is_gen,
                cos,
                sin,
                layer_cache,
                append=append,
                within_mask=wm,
                zero_value_mask=zero_value_mask,
                native_gqa_attention=native_gqa_attention,
            )
        return h

    def prefill_text_cache(
        self,
        text_ids: Tensor,
        position_ids: Tensor,
        cache: list[dict[str, Tensor]],
        *,
        native_gqa_attention: bool = False,
    ) -> None:
        """Seed the text (understanding) K/V into the cache (causal self-attention)."""
        device = self.embed_tokens.weight.device
        n = int(text_ids.shape[0])
        h = self.embed_tokens(text_ids.to(device)).to(self.config.dtype)[None]
        is_gen = torch.zeros(n, dtype=torch.bool)
        order = torch.arange(n)
        within = order[None, :] <= order[:, None]  # causal
        self._layers_step(
            h,
            is_gen,
            position_ids,
            cache,
            append=True,
            within_mask=within,
            native_gqa_attention=native_gqa_attention,
        )

    # Fixed-buffer streaming: graph-ready attention with bounded history.

    def init_stream_cache(
        self,
        *,
        chunk_grid: tuple[int, int, int],
        window_frames: int,
        sink_frames: int = 0,
        tokens_per_frame: int | None = None,
    ) -> _MoTStreamCache:
        """Allocate one rolling gen :class:`BlockKVCache` per layer.

        ``window_frames`` is the number of latent frames of generation history kept
        (the attention-sink window); ``window_frames % len_t == 0`` is required.
        ``tokens_per_frame`` defaults to ``gh*gw`` for video-only generation, but
        camera rollouts pass ``gh*gw + tcf`` so action tokens share the same cache
        block as their frame.
        Text K/V are filled later by :meth:`prefill_text_stream`.
        """
        llm = self.config.spec.llm
        device = self.embed_tokens.weight.device
        dtype = self.config.dtype
        len_t, gh, gw = chunk_grid
        hw = gh * gw
        frame_tokens = int(tokens_per_frame or hw)
        chunk_tokens = len_t * frame_tokens
        window_tokens = window_frames * frame_tokens
        sink_tokens = sink_frames * frame_tokens
        total = sink_tokens + window_tokens
        gen = [
            BlockKVCache(
                k_shape=(1, llm.num_key_value_heads, total, llm.head_dim),
                v_shape=(1, llm.num_key_value_heads, total, llm.head_dim),
                seq_dim=2,
                chunk_size=chunk_tokens,
                window_size=window_tokens,
                sink_size=sink_tokens,
                device=device,
                dtype=dtype,
            )
            for _ in self.layers
        ]
        n = len(self.layers)
        return _MoTStreamCache(text_k=[None] * n, text_v=[None] * n, gen=gen)

    def _mrope(self, position_ids: Tensor) -> tuple[Tensor, Tensor]:
        llm = self.config.spec.llm
        device = self.embed_tokens.weight.device
        cos, sin = mrope_cos_sin(
            position_ids.to(device), llm.head_dim, llm.rope_theta, llm.mrope_section
        )
        dtype = self.config.dtype
        return cos.to(dtype)[None], sin.to(dtype)[None]

    def prefill_text_stream(
        self,
        text_ids: Tensor | None,
        position_ids: Tensor,
        cache: _MoTStreamCache,
        *,
        native_gqa_attention: bool = False,
    ) -> None:
        """Pin the understanding K/V for ``text_ids`` into ``cache`` (one-time)."""
        if text_ids is None or int(text_ids.shape[0]) == 0:
            return
        device = self.embed_tokens.weight.device
        n = int(text_ids.shape[0])
        h = self.embed_tokens(text_ids.to(device)).to(self.config.dtype)[None]
        cos, sin = self._mrope(position_ids)
        order = torch.arange(n, device=device)
        within = order[None, :] <= order[:, None]  # causal
        for i, layer in enumerate(cast(list[_MoTLayer], self.layers)):
            h, k, v = layer.text_prefill_kv(
                h, cos, sin, within, native_gqa_attention=native_gqa_attention
            )
            cache.text_k[i] = k.detach()
            cache.text_v[i] = v.detach()

    # Gather inputs eagerly, then run compile/graph-friendly tensor computation.

    def gather_stream_history(
        self, cache: _MoTStreamCache, write_start: int
    ) -> tuple[
        list[Tensor | None],
        list[Tensor | None],
        list[Tensor | None],
        list[Tensor | None],
    ]:
        """Pull the pinned text K/V + the clean gen history ``[..., :write_start, ...]``.

        Done eagerly (outside any compiled/captured region). The gen history slices
        are views into each cache's fixed buffer, so they stay pointer-stable for
        CUDA-graph replay; ``write_start`` (the slot the current chunk will occupy)
        is constant at steady state, so the slices are static.
        """
        hist_k: list[Tensor | None] = []
        hist_v: list[Tensor | None] = []
        for gc in cache.gen:
            if write_start > 0:
                hist_k.append(gc._k[:, :, :write_start])
                hist_v.append(gc._v[:, :, :write_start])
            else:
                hist_k.append(None)
                hist_v.append(None)
        return cache.text_k, cache.text_v, hist_k, hist_v

    def gather_stream_history_masked(
        self, cache: _MoTStreamCache, write_start: int
    ) -> tuple[
        list[Tensor | None],
        list[Tensor | None],
        list[Tensor | None],
        list[Tensor | None],
        Tensor,
    ]:
        """Gather a fixed-size history plus a mask for unwritten fill slots.

        The fixed history ends immediately before the steady-state write block,
        so its shape and storage address match the normal saturated-cache path.
        During fill, only the prefix before ``write_start`` is valid. Text K/V
        are pinned and therefore always marked valid.

        Args:
            cache: Stream cache whose generation buffers have equal geometry.
            write_start: First physical slot reserved for the current chunk.

        Returns:
            Text K/V, fixed generation-history K/V, and a boolean validity mask
            over the concatenated cached text and generation positions.
        """
        hist_k: list[Tensor | None] = []
        hist_v: list[Tensor | None] = []
        history_size = 0
        for gc in cache.gen:
            layer_history_size = int(gc._k.shape[gc.seq_dim]) - int(gc.chunk_size)
            if hist_k and layer_history_size != history_size:
                raise ValueError("all stream-cache layers must have equal history size")
            history_size = layer_history_size
            hist_k.append(gc._k[:, :, :history_size])
            hist_v.append(gc._v[:, :, :history_size])
        if not 0 <= write_start <= history_size:
            raise ValueError(
                f"write_start must be in [0, {history_size}], got {write_start}"
            )
        device = self.embed_tokens.weight.device
        text_size = 0
        if cache.text_k and cache.text_k[0] is not None:
            text_size = int(cache.text_k[0].shape[2])
        text_valid = torch.ones(text_size, dtype=torch.bool, device=device)
        history_valid = torch.arange(history_size, device=device) < write_start
        cached_mask = torch.cat((text_valid, history_valid))
        return cache.text_k, cache.text_v, hist_k, hist_v, cached_mask

    def gather_stream_history_valid_length(
        self, cache: _MoTStreamCache, write_start: int
    ) -> tuple[
        list[Tensor | None],
        list[Tensor | None],
        list[Tensor | None],
        list[Tensor | None],
        Tensor,
    ]:
        """Gather fixed K/V buffers and the exact logical cached-token count.

        Unlike :meth:`gather_stream_history_masked`, this path does not build an
        attention mask. Fused variable-length attention receives the full fixed
        allocation plus this device-side scalar and ignores its unwritten tail.
        The tensors therefore retain identical shapes and storage throughout
        fill, saturation, and ring-buffer rollover.
        """
        hist_k: list[Tensor | None] = []
        hist_v: list[Tensor | None] = []
        history_size = 0
        for gc in cache.gen:
            layer_history_size = int(gc._k.shape[gc.seq_dim]) - int(gc.chunk_size)
            if hist_k and layer_history_size != history_size:
                raise ValueError("all stream-cache layers must have equal history size")
            history_size = layer_history_size
            hist_k.append(gc._k[:, :, :history_size])
            hist_v.append(gc._v[:, :, :history_size])
        if not 0 <= write_start <= history_size:
            raise ValueError(
                f"write_start must be in [0, {history_size}], got {write_start}"
            )
        text_size = 0
        if cache.text_k and cache.text_k[0] is not None:
            text_size = int(cache.text_k[0].shape[2])
        valid_cached = torch.tensor(
            text_size + write_start,
            device=self.embed_tokens.weight.device,
            dtype=torch.int32,
        )
        return cache.text_k, cache.text_v, hist_k, hist_v, valid_cached
