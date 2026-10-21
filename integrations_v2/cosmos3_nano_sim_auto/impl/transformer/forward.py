# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run MoT layers over one autoregressive chunk.

Text uses the shared prefill path. Generation orders keys as
``[text | current | memory]`` and supplies the query/key visibility mask
explicitly for the current cache contents.
"""

from __future__ import annotations

import functools
from collections.abc import Callable, Sequence
from typing import cast

import torch
import torch.nn.functional as F
from cosmos_interactive._network import (
    RMSNorm,
    _apply_rope,
    _kv_for_attention,
    _MoTLayer,
    _SwiGLUMLP,
)
from cosmos_interactive._rope import mrope_cos_sin
from cosmos_interactive._spec import LLMSpec
from torch import Tensor, nn

from cosmos3_nano_sim_auto.impl.transformer.attention import (
    ChunkMask,
    chunk_attention,
)
from cosmos3_nano_sim_auto.impl.transformer.fold import FoldPlan
from flashdreams.accelerated.common.rms_norm import rms_norm
from flashdreams.accelerated.multi_head_attention.kv_arena import (
    KVSegmentLayout,
    stage_current_kv_,
    validate_kv_arena_segments,
)
from flashdreams.accelerated.multi_head_attention.optimized import OptimizedImplConfig
from flashdreams.core.distributed.context_parallel import (
    TokenShard,
    build_shard,
    gather_tokens,
)
from flashdreams.core.distributed.parallel import ParallelContext

# One layer's cached keys and values, ``[B, Hkv, N, d]`` each: RoPE already
# applied, and key/value heads not yet repeated to match the query heads.
LayerKV = tuple[Tensor, Tensor]


def _rms_norm(module: nn.Module, input: Tensor, *, accelerated: bool) -> Tensor:
    """Use the shared fused primitive or the checkpoint module unchanged."""
    if not accelerated:
        return module(input)
    norm = cast(RMSNorm, module)
    return rms_norm(input, norm.weight, norm.eps)


@functools.cache
def _compiled_rope() -> Callable[..., tuple[Tensor, Tensor]]:
    return torch.compile(_apply_rope, dynamic=False)


def _rope(q: Tensor, k: Tensor, cos: Tensor, sin: Tensor) -> tuple[Tensor, Tensor]:
    """RoPE as one compiled kernel per tensor on CUDA instead of eight elementwise ones."""
    if q.is_cuda:
        return _compiled_rope()(q, k, cos, sin)
    return _apply_rope(q, k, cos, sin)


def _silu_mul(gate: Tensor, up: Tensor) -> Tensor:
    return F.silu(gate) * up


@functools.cache
def _compiled_silu_mul() -> Callable[[Tensor, Tensor], Tensor]:
    return torch.compile(_silu_mul, dynamic=False)


def _mlp(mlp: _SwiGLUMLP, x: Tensor) -> Tensor:
    """The checkpoint's MLP with ``silu(gate) * up`` as one compiled kernel on CUDA."""
    if x.is_cuda:
        return mlp.down_proj(_compiled_silu_mul()(mlp.gate_proj(x), mlp.up_proj(x)))
    return mlp(x)


def chunk_layer_forward(
    layer: _MoTLayer,
    h: Tensor,
    cos: Tensor,
    sin: Tensor,
    *,
    text_kv: LayerKV,
    memory_kv: LayerKV,
    mask: ChunkMask,
    attention_implementation: OptimizedImplConfig | None = None,
    native_gqa_attention: bool = False,
    accelerated_rms_norm: bool = False,
    attention_arena: LayerKV | None = None,
    attention_arena_layout: KVSegmentLayout | None = None,
    ctx: ParallelContext | None = None,
    shard: TokenShard | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """One MoT layer over the current chunk, generation half only.

    Args:
        layer: The layer to borrow weights from.
        h: ``[B, s, D]`` hidden states for the chunk's tokens -- this rank's
            token range when ``shard`` is given, the whole chunk otherwise.
        cos, sin: ``[B, s, d]`` RoPE tables for those same positions. Text and
            memory keys already carry the RoPE they were built with.
        text_kv, memory_kv: This layer's cached keys and values. Pass
            zero-length tensors rather than ``None`` for an empty side.
        mask: ``[s, n_text + N + n_memory]`` bool, from
            :meth:`~cosmos3_nano_sim_auto.impl.packing.ChunkMetadata.mask`. Its key axis has to
            be laid out in that order, which is what the packer produces. Under a
            shard its rows are this rank's and its keys are still everyone's, so
            ``s`` and ``N`` differ.
        ctx, shard: the mesh and this pass's token split. Both ``None`` runs the
            single-rank path, which is what the tests and any un-launched run do.

    Returns:
        ``(h_out, k, v)`` -- this rank's updated hidden states, and the *whole*
        chunk's post-RoPE keys and values for the caller to commit into memory.
        The keys are gathered rather than local because the cache is the same on
        every rank: they all attend it, so they all hold it.
    """
    attn_module = layer.gen.self_attn
    batch, tokens, _ = h.shape
    head_dim = attn_module.head_dim

    hn = _rms_norm(layer.gen.input_layernorm, h, accelerated=accelerated_rms_norm)
    projected_q = attn_module.q_proj(hn)
    projected_k = attn_module.k_proj(hn)
    projected_v = attn_module.v_proj(hn)
    q = _rms_norm(
        attn_module.q_norm,
        projected_q.view(batch, tokens, attn_module.num_heads, head_dim),
        accelerated=accelerated_rms_norm,
    ).transpose(1, 2)
    k = _rms_norm(
        attn_module.k_norm,
        projected_k.view(batch, tokens, attn_module.num_kv_heads, head_dim),
        accelerated=accelerated_rms_norm,
    ).transpose(1, 2)
    v = projected_v.view(batch, tokens, attn_module.num_kv_heads, head_dim).transpose(
        1, 2
    )
    q, k = _rope(q, k, cos, sin)

    # The one collective on the context axis, and it happens here rather than
    # before the projection on purpose: keys and values are 128 channels a rank
    # under eight-way tensor parallelism where the hidden state is 4096, so
    # gathering them moves a sixteenth of the bytes and saves every rank
    # recomputing every other rank's projection.
    #
    # Keys and values go in *one* gather rather than two. They are the same
    # shape, dtype and group, so concatenating them on the head axis and
    # splitting after halves the collective count -- 36 layers x 5 passes is 360
    # launches a chunk against 180. That matters where the axis is weakest: at
    # sixteen ranks over two nodes a rank holds a thousand tokens and launch
    # latency, not bandwidth, is what the gather costs.
    if ctx is not None and shard is not None:
        heads = k.shape[1]
        both = gather_tokens(torch.cat([k, v], dim=1), shard, ctx, dim=2)
        k, v = both[:, :heads], both[:, heads:]

    text_k, text_v = text_kv
    memory_k, memory_v = memory_kv
    if attention_arena is None and attention_arena_layout is None:
        k_all = torch.cat([text_k, k, memory_k], dim=2)
        v_all = torch.cat([text_v, v, memory_v], dim=2)
    elif attention_arena is not None and attention_arena_layout is not None:
        validate_kv_arena_segments(
            attention_arena,
            text_kv,
            memory_kv,
            attention_arena_layout,
        )
        k_all, v_all = stage_current_kv_(
            attention_arena, (k, v), attention_arena_layout
        )
    else:
        raise ValueError("attention arena and layout must be supplied together")
    kr, vr = _kv_for_attention(
        k_all,
        v_all,
        attn_module.num_kv_groups,
        native_gqa_attention=native_gqa_attention,
    )

    attn = chunk_attention(
        q,
        kr,
        vr,
        mask,
        enable_gqa=native_gqa_attention,
        implementation=attention_implementation,
    )
    attn = attn.transpose(1, 2).reshape(batch, tokens, -1)
    h = h + attn_module.o_proj(attn)
    h = h + _mlp(
        layer.gen.mlp,
        _rms_norm(
            layer.gen.post_attention_layernorm,
            h,
            accelerated=accelerated_rms_norm,
        ),
    )
    return h, k, v


def run_text_layers(
    layers: Sequence[_MoTLayer],
    llm: LLMSpec,
    h: Tensor,
    position_ids: Tensor,
    *,
    view_text_tokens: Sequence[int] | None = None,
    native_gqa_attention: bool = False,
) -> tuple[Tensor, list[LayerKV]]:
    """Run the prompt through the understanding half and keep its keys and values.

    Plain causal self-attention, and that is not a simplification: the prompt
    never attends to a vision token in this model, so there is nothing
    multiview about this half and ``text_prefill_kv`` is used as it stands.

    ``view_text_tokens`` gives the length of each camera's own caption for a
    checkpoint trained one caption per camera. Each is then its own document:
    causal within itself and blind to the others, since no camera's caption was
    written knowing what another's says. Left at ``None`` the whole prompt is
    one document, which is one caption for the rig.

    Done once per rollout, since the prompt does not change and nothing it can
    see does either, so the result stays valid for every chunk that follows.
    """
    device, dtype = h.device, h.dtype
    cos, sin = mrope_cos_sin(
        position_ids.to(device), llm.head_dim, llm.rope_theta, llm.mrope_section
    )
    cos = cos.to(dtype)[None]
    sin = sin.to(dtype)[None]

    order = torch.arange(h.shape[1], device=device)
    causal = order[None, :] <= order[:, None]
    if view_text_tokens is not None:
        caption = torch.repeat_interleave(
            torch.arange(len(view_text_tokens), device=device),
            torch.tensor(list(view_text_tokens), device=device),
        )
        causal = causal & (caption[:, None] == caption[None, :])

    text_kv: list[LayerKV] = []
    for layer in layers:
        h, k, v = layer.text_prefill_kv(
            h, cos, sin, causal, native_gqa_attention=native_gqa_attention
        )
        text_kv.append((k, v))
    return h, text_kv


def run_chunk_layers(
    layers: Sequence[_MoTLayer],
    llm: LLMSpec,
    h: Tensor,
    position_ids: Tensor,
    *,
    text_kv: Sequence[LayerKV],
    memory_kv: Sequence[LayerKV],
    mask: ChunkMask,
    final_norm: nn.Module | None = None,
    attention_implementation: OptimizedImplConfig | None = None,
    native_gqa_attention: bool = False,
    accelerated_rms_norm: bool = False,
    attention_arenas: Sequence[LayerKV] | None = None,
    attention_arena_layout: KVSegmentLayout | None = None,
    collect_kv: bool = True,
    ctx: ParallelContext | None = None,
) -> tuple[Tensor, list[LayerKV]]:
    """Run every MoT layer over one chunk and collect its keys and values.

    Takes the layer list and the spec rather than the network, so that the loop
    can be exercised without a vocabulary-sized embedding table or a vision tower.

    Args:
        h: ``[B, N, D]`` for the *whole* pass, on every rank. The split happens
            here rather than at the call site so that a caller hands over the
            same tensor whatever the mesh is; embedding N tokens on every rank
            costs one small projection and buys that.
        position_ids: ``[3, N]`` mRoPE ids for the chunk, from
            :func:`cosmos3_nano_sim_auto.impl.transformer.rope.chunk_mrope_ids`.
        text_kv, memory_kv: one entry per layer, in layer order.
        final_norm: the generation half's final norm (``norm_moe_gen``). A chunk
            is all vision tokens, so there is no choice of norm to make here.
        ctx: the mesh. ``None``, or a context whose axes are both inert, runs
            unsharded. When the context axis is live, ``mask`` must already have
            been built for this rank's rows -- pass
            :meth:`~flashdreams.core.distributed.context_parallel.TokenShard.local` as
            ``query_range`` to whichever builder made it.

    Returns:
        ``(h_out, chunk_kv)`` with one keys-and-values pair per layer, or an
        empty list when ``collect_kv`` is false. Noisy denoise passes do not
        commit their K/V and avoid retaining them this way.

        ``h_out`` is this rank's token range. The keys and values are the *whole*
        pass's however the mesh is shaped -- they are gathered before they are
        returned, because every rank attends the whole chunk -- so the cache
        written from them is identical on every rank.
    """
    if len(text_kv) != len(layers) or len(memory_kv) != len(layers):
        raise ValueError(
            f"Expected one cache entry per layer; got {len(layers)} layers, "
            f"{len(text_kv)} text and {len(memory_kv)} memory entries."
        )
    if (attention_arenas is None) != (attention_arena_layout is None):
        raise ValueError("attention arenas and layout must be supplied together")
    if attention_arenas is not None and len(attention_arenas) != len(layers):
        raise ValueError(
            f"Expected one attention arena per layer; got {len(layers)} layers "
            f"and {len(attention_arenas)} arenas."
        )

    device, dtype = h.device, h.dtype
    cos, sin = mrope_cos_sin(
        position_ids.to(device), llm.head_dim, llm.rope_theta, llm.mrope_section
    )
    cos = cos.to(dtype)[None]
    sin = sin.to(dtype)[None]
    mask = mask.to(device)

    shard: TokenShard | None = None
    if ctx is not None and ctx.context_parallel:
        shard = build_shard(int(h.shape[1]), ctx)
        h = shard.select(h, dim=1)
        cos = shard.select(cos, dim=1)
        sin = shard.select(sin, dim=1)
        rows = mask.query_length if isinstance(mask, FoldPlan) else mask.shape[-2]
        if rows != h.shape[1]:
            raise ValueError(
                f"the mask asks about {rows} queries but this rank owns "
                f"{h.shape[1]} tokens of the pass; build it with "
                f"query_range={shard.local}."
            )

    chunk_kv: list[LayerKV] = []
    arenas = [None] * len(layers) if attention_arenas is None else attention_arenas
    for layer, layer_text_kv, layer_memory_kv, layer_arena in zip(
        layers, text_kv, memory_kv, arenas, strict=True
    ):
        h, k, v = chunk_layer_forward(
            layer,
            h,
            cos,
            sin,
            text_kv=layer_text_kv,
            memory_kv=layer_memory_kv,
            mask=mask,
            attention_implementation=attention_implementation,
            native_gqa_attention=native_gqa_attention,
            accelerated_rms_norm=accelerated_rms_norm,
            attention_arena=layer_arena,
            attention_arena_layout=attention_arena_layout,
            ctx=ctx,
            shard=shard,
        )
        if collect_kv:
            chunk_kv.append((k, v))

    if final_norm is not None:
        h = _rms_norm(final_norm, h, accelerated=accelerated_rms_norm)
    return h, chunk_kv
