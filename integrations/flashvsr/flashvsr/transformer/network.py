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

"""FlashVSR DiT network on top of the flashdreams Wan 2.1 transformer stack.

Mirrors the ``template/transformer/network.py`` role: the raw network +
block + attention + sparse-attn helpers live here; the
:class:`Transformer` subclass that wires this network into the streaming
inference cache lives in :mod:`flashvsr.transformer`
(``__init__.py``).

``integrations/flashvsr/tests/parity_check/test_dit_parity.py`` asserts
bit-for-bit-modulo-tolerance parity against upstream's
``diffsynth.models.wan_video_dit.WanModel`` plus the streaming
forward wrapper
``diffsynth.pipelines.flashvsr_tiny_long.model_fn_wan_video`` (loaded
out of the cloned ``parity_check/FlashVSR/`` sibling).

Layering (top -> bottom):

- ``FlashVSRDiTNetwork`` (subclass of ``WanDiTNetwork``):
    DiT backbone. Reuses patch embedding, time embedding, head and parameter
    bookkeeping. Overrides ``_build_block`` to use ``FlashVSRBlock`` and
    ``forward`` to thread per-block ``lq_latents`` next to ``block_extra_kwargs``.

- ``FlashVSRBlock`` (subclass of ``Block``):
    Transformer block. Replaces the dense ``SelfAttention`` with
    ``SparseSelfAttention``; keeps ``CrossAttention`` (FlashVSR's static
    text-prompt cache is exactly what ``CrossAttnCache.text`` carries).
    Adds the additive ``lq_latent`` injection before AdaLN modulation.

- ``SparseSelfAttention`` (subclass of ``MultiHeadAttention``):
    Block-sparse self-attention with windowed KV streaming. Inherits the
    ``q/k/v/o`` linears, ``norm_q/norm_k`` and the per-head reshape contract
    from ``MultiHeadAttention``; replaces the dense ring-attention forward
    with FlashVSR's ``WindowPartition3D`` + topk-block draft mask +
    ``block_sparse_attn_func`` path. Streaming KV state is held in a
    ``BlockKVCache`` (chunk-rolling, sink-free).

Parameter naming is identical to ``WanDiTNetwork`` so the existing FlashVSR
checkpoint at ``flashvsr_tiny_long/dit_state_dict.pt`` loads drop-in.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple

import torch
import torch.nn as nn

# We deliberately import the underlying ``BlockSparseAttnFunc`` autograd
# function instead of the public ``block_sparse_attn_func`` wrapper. The
# wrapper unconditionally calls ``replace_ones_with_count(head_mask_type)``
# which uses ``Tensor.masked_scatter`` -- a pybind C++ method dynamo cannot
# trace -- forcing a graph break and a recompile cascade on every
# fill-state shape transition (eventually hitting ``recompile_limit (8)``).
# We pre-compute the renumbered head-mask in ``initialize_cache`` and call
# the autograd function directly with all the static knobs spelled out.
# Tracks ``block_sparse_attn==0.0.2``; bump the pin in ``uv.lock`` if the
# ``BlockSparseAttnFunc.apply`` signature changes.
from block_sparse_attn.block_sparse_attn_interface import BlockSparseAttnFunc
from einops import rearrange
from torch import Tensor

from flashdreams.core.attention import BlockKVCache
from flashdreams.core.attention.rope import apply_rope_freqs
from flashdreams.recipes.wan.transformer.impl.modules import (
    Block,
    BlockCache,
    MultiHeadAttention,
    sinusoidal_embedding_1d,
)
from flashdreams.recipes.wan.transformer.impl.network import (
    WanDiTNetwork,
    WanDiTNetworkConfig,
)

__all__ = [
    "FlashVSRDiTNetwork",
    "FlashVSRDiTNetworkConfig",
    "FlashVSRBlock",
    "SparseSelfAttention",
    "WindowPartition3D",
    "build_local_block_mask_shifted_vec_normal_slide",
    "generate_draft_block_mask",
    "_SELF_ATTN_WINDOW",
    "_SELF_ATTN_WINDOW_TOKENS",
]


## FlashVSR sparse-attention helpers

# These three helpers have no flashdreams equivalent; they implement the
# block-sparse path that distinguishes FlashVSR's DiT from the dense
# Wan 2.1 baseline. The parity check at
# ``integrations/flashvsr/tests/parity_check/test_dit_parity.py`` compares
# their outputs against upstream's ``diffsynth.models.wan_video_dit``
# implementation; the live import path uses the copies below.


class WindowPartition3D:
    """Partition / reverse-partition helpers for 5-D tensors (B,F,H,W,C)."""

    @staticmethod
    def partition(x: torch.Tensor, win: Tuple[int, int, int]):
        """Reshape ``[B, F, H, W, C]`` into ``[B*block_n, wf*wh*ww, C]`` windows."""
        B, F, H, W, C = x.shape
        wf, wh, ww = win
        assert F % wf == 0 and H % wh == 0 and W % ww == 0, (
            "Dims must divide by window size."
        )
        x = x.view(B, F // wf, wf, H // wh, wh, W // ww, ww, C)
        x = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous()
        return x.view(-1, wf * wh * ww, C)

    @staticmethod
    def reverse(
        windows: torch.Tensor, win: Tuple[int, int, int], orig: Tuple[int, int, int]
    ):
        """Invert :meth:`partition`: ``[B*block_n, wf*wh*ww, C]`` -> ``[B, F, H, W, C]``."""
        F, H, W = orig
        wf, wh, ww = win
        nf, nh, nw = F // wf, H // wh, W // ww
        B = windows.size(0) // (nf * nh * nw)
        x = windows.view(B, nf, nh, nw, wf, wh, ww, -1)
        x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous()
        return x.view(B, F, H, W, -1)


@torch.no_grad()
def build_local_block_mask_shifted_vec_normal_slide(
    block_h: int,
    block_w: int,
    win_h: int = 6,
    win_w: int = 6,
    include_self: bool = True,
    device=None,
) -> torch.Tensor:
    """Build the boolean ``[block_h*block_w, block_h*block_w]`` local-window mask.

    Each row corresponds to one query block ``(r, c)`` on the
    ``block_h x block_w`` grid; entry ``(i, j)`` is ``True`` if key
    block ``j`` falls inside the ``win_h x win_w`` window centred on
    query block ``i``. Consumed by :func:`generate_draft_block_mask` as
    the constant boolean prior added to the average-pooled attention
    scores before the top-k selection.

    Args:
        block_h, block_w: Block-grid dimensions (post-window-partition).
        win_h, win_w: Local-window radius in blocks; centred on each query.
        include_self: If ``False``, zero the diagonal so the query block
            doesn't attend to itself.
        device: Output device; defaults to CPU.
    """
    device = device or torch.device("cpu")
    H, W = block_h, block_w
    r = torch.arange(H, device=device)
    c = torch.arange(W, device=device)
    YY, XX = torch.meshgrid(r, c, indexing="ij")
    r_all = YY.reshape(-1)
    c_all = XX.reshape(-1)
    r_half = win_h // 2
    c_half = win_w // 2
    start_r = r_all - r_half
    end_r = start_r + win_h - 1
    start_c = c_all - c_half
    end_c = start_c + win_w - 1
    in_row = (r_all[None, :] >= start_r[:, None]) & (r_all[None, :] <= end_r[:, None])
    in_col = (c_all[None, :] >= start_c[:, None]) & (c_all[None, :] <= end_c[:, None])
    mask = in_row & in_col
    if not include_self:
        mask.fill_diagonal_(False)
    return mask


@torch.no_grad()
def generate_draft_block_mask(
    batch_size, nheads, seqlen, q_w, k_w, topk=10, local_attn_mask=None
):
    """Compute the per-head top-k block-sparse attention mask.

    Average-pools the per-window Q / K tensors to one vector per window,
    scores them ``softmax(QK^T / sqrt(D))`` with the boolean
    ``local_attn_mask`` added as a ``-inf`` / ``0`` prior, then keeps
    the top ``topk + 1`` entries per row as the draft sparsity pattern
    fed to ``BlockSparseAttnFunc``.

    Args:
        batch_size: Effective batch size; must be ``1`` (``cu_seqlens``
            is sized for a single sequence).
        nheads: Number of attention heads.
        seqlen: Number of latent-frame chunks per AR step
            (``f // win_f``, where ``win_f == 2`` for FlashVSR).
        q_w: Per-window queries ``[block_n, win_size, n_heads * head_dim]``.
        k_w: Per-window keys; same layout as ``q_w`` but with the cached
            KV stretch (``block_n_kv * win_size`` tokens).
        topk: Per-row top-k budget for the draft mask.
        local_attn_mask: Boolean ``[block_h*block_w, block_h*block_w]``
            local-window mask from
            :func:`build_local_block_mask_shifted_vec_normal_slide`.
    """
    assert batch_size == 1, "Only batch_size=1 supported for now"
    assert local_attn_mask is not None, "local_attn_mask must be provided"
    avgpool_q = torch.mean(q_w, dim=1)
    avgpool_k = torch.mean(k_w, dim=1)
    avgpool_q = rearrange(avgpool_q, "s (h d) -> s h d", h=nheads)
    avgpool_k = rearrange(avgpool_k, "s (h d) -> s h d", h=nheads)
    q_heads = avgpool_q.permute(1, 0, 2)
    k_heads = avgpool_k.permute(1, 0, 2)
    D = avgpool_q.shape[-1]
    scores = torch.einsum("hld,hmd->hlm", q_heads, k_heads) / math.sqrt(D)

    repeat_head = scores.shape[0]
    repeat_len = scores.shape[1] // local_attn_mask.shape[0]
    repeat_num = scores.shape[2] // local_attn_mask.shape[1]
    local_attn_mask = (
        local_attn_mask.unsqueeze(1).unsqueeze(0).repeat(repeat_len, 1, repeat_num, 1)
    )
    local_attn_mask = rearrange(local_attn_mask, "x a y b -> (x a) (y b)")
    local_attn_mask = local_attn_mask.unsqueeze(0).repeat(repeat_head, 1, 1)
    local_attn_mask = local_attn_mask.to(torch.float32)
    local_attn_mask = local_attn_mask.masked_fill(
        local_attn_mask == False, -float("inf")
    )
    local_attn_mask = local_attn_mask.masked_fill(local_attn_mask == True, 0)
    scores = scores + local_attn_mask

    attn_map = torch.softmax(scores, dim=-1)
    attn_map = rearrange(attn_map, "h (it s1) s2 -> (h it) s1 s2", it=seqlen)
    loop_num, s1, s2 = attn_map.shape
    flat = attn_map.reshape(loop_num, -1)
    apply_topk = min(flat.shape[1] - 1, topk)
    thresholds = torch.topk(flat, k=apply_topk + 1, dim=1, largest=True).values[:, -1]
    thresholds = thresholds.unsqueeze(1)
    mask_new = (flat > thresholds).reshape(loop_num, s1, s2)
    mask_new = rearrange(mask_new, "(h it) s1 s2 -> h (it s1) s2", it=seqlen)
    mask = mask_new.unsqueeze(0).repeat(batch_size, 1, 1, 1)
    return mask


_SELF_ATTN_WINDOW = (2, 8, 8)
"""Self-attention window ``(latent_frames, latent_h, latent_w)``. Kept
as a module-level constant rather than a config knob because the topk /
local-range math elsewhere in this file assumes ``win[0] == 2`` and
``win[1] == win[2] == 8``."""

_SELF_ATTN_WINDOW_TOKENS = (
    _SELF_ATTN_WINDOW[0] * _SELF_ATTN_WINDOW[1] * _SELF_ATTN_WINDOW[2]
)
"""Tokens per self-attention window (``win_f * win_h * win_w = 128``)."""


class SparseSelfAttention(MultiHeadAttention):
    """Block-sparse self-attention with windowed KV streaming.

    Subclass of :class:`MultiHeadAttention` so the ``q/k/v/o`` and
    ``norm_q/norm_k`` parameters live at the same names as the dense flashdreams
    ``SelfAttention`` -- checkpoints transfer drop-in.

    The dense ring-attention forward is replaced with FlashVSR's path:
    project Q/K/V -> 3D RoPE -> ``WindowPartition3D.partition`` along
    (T, H, W) -> write the partitioned K/V into a rolling
    :class:`BlockKVCache` -> average-pool topk draft mask over the local
    block window -> ``block_sparse_attn_func`` against the cached K/V ->
    reverse the partition -> apply ``o``.

    The cache holds the last ``window_size`` tokens of partitioned K/V; one
    AR chunk contributes ``chunk_size = block_n * win_size`` tokens (i.e.
    the chunk's full token count). ``cached_k() / cached_v()`` returns the
    just-written chunk plus all retained earlier chunks, matching the
    ``cat([pre_cache, new])`` view the legacy FlashVSR DiT used.
    """

    def __init__(
        self,
        query_dim: int,
        n_heads: int,
        head_dim: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__(
            query_dim=query_dim,
            context_dim=None,
            n_heads=n_heads,
            head_dim=head_dim,
            eps=eps,
        )
        self._local_attn_mask: Optional[Tensor] = None
        self._local_attn_mask_h: Optional[int] = None
        self._local_attn_mask_w: Optional[int] = None
        self._local_range: Optional[int] = None
        # Pre-declare the static int32 helper buffers required by
        # ``block_sparse_attn_func``. ``initialize_cache`` materialises
        # them once per rollout so ``forward`` doesn't allocate or sync to
        # device on every call (which previously broke CUDA-graph capture
        # and tripped dynamo recompiles inside ``block_sparse_attn``).
        self.register_buffer("_cu_seqlens_q", None, persistent=False)
        self.register_buffer("_head_mask_type", None, persistent=False)
        self.register_buffer("_cu_seqlens_k_table", None, persistent=False)
        self._chunk_tokens: Optional[int] = None

    def initialize_cache(
        self,
        batch_size: int,
        chunk_size: int,
        window_size: int,
        sink_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> BlockKVCache:
        """Build a fresh streaming KV cache for one rollout.

        Args:
            batch_size: Effective batch size for cache tensors. Must be ``1``
                because ``block_sparse_attn_func`` is invoked with
                ``cu_seqlens_q = [0, seqlen_q]`` (batch=1).
            chunk_size: Tokens added per AR step. For FlashVSR this is
                ``block_n * win_size`` which numerically equals the chunk's
                total token count ``f * h * w``.
            window_size: Rolling-window capacity in tokens (excluding
                ``sink_size``). Set this to ``(kv_ratio + 1) * chunk_size``
                so that ``cached_k()`` returns the just-written chunk plus
                ``kv_ratio`` retained earlier ones, matching the legacy
                ``cat([pre_cache, new])`` semantics.
            sink_size: Sink-token capacity. FlashVSR has none; pass ``0``.
            device: Device for cache tensors.
            dtype: Data type for cache tensors.
        """
        # Pre-allocate the int32 helper tensors that ``BlockSparseAttnFunc.apply``
        # consumes on every call. Sized once per rollout from the same
        # ``chunk_size`` / ``window_size`` the cache is being built with so
        # ``forward`` never re-issues ``torch.tensor([...], device=...)``
        # (which would force a host->device sync, break CUDA-graph capture
        # and -- with the wrapped ``block_sparse_attn_func`` -- invalidate
        # the dynamo cache around the un-traceable ``masked_scatter`` inside
        # ``replace_ones_with_count``).
        #
        # ``_head_mask_type`` is **already in the post-renumbering form**
        # (``[1, 2, 3, ..., n_heads]``) that ``replace_ones_with_count``
        # would have produced from the all-ones marker form. By computing
        # it here we let ``forward`` skip the wrapper and call
        # ``BlockSparseAttnFunc.apply`` directly, sidestepping the dynamo
        # graph break entirely.
        #
        # The cu_seqlens_k table holds one row per ``(kv_ratio + 1)`` fill
        # state; at attention time ``forward`` indexes it by the effective
        # number of cached chunks.
        assert sink_size == 0, (
            "Phase 1 cu_seqlens table assumes sink_size == 0 (FlashVSR's "
            f"only configuration); got sink_size={sink_size}."
        )
        kv_ratio_plus_one = window_size // chunk_size
        self._cu_seqlens_q = torch.tensor(
            [0, chunk_size], device=device, dtype=torch.int32
        )
        self._head_mask_type = torch.arange(
            1, self.n_heads + 1, device=device, dtype=torch.int32
        )
        self._cu_seqlens_k_table = torch.stack(
            [
                torch.tensor(
                    [0, (i + 1) * chunk_size], device=device, dtype=torch.int32
                )
                for i in range(kv_ratio_plus_one)
            ]
        )
        self._chunk_tokens = chunk_size

        total_size = sink_size + window_size
        return BlockKVCache(
            k_shape=(batch_size, total_size, self.n_heads, self.head_dim),
            v_shape=(batch_size, total_size, self.n_heads, self.head_dim),
            seq_dim=-3,
            chunk_size=chunk_size,
            window_size=window_size,
            sink_size=sink_size,
            device=device,
            dtype=dtype,
        )

    def _build_or_get_local_mask(
        self, h: int, w: int, local_range: int, device: torch.device
    ) -> Tensor:
        """Lazily (re)build the local-block mask used by the topk draft.

        ``h`` and ``w`` are the post-patchify spatial dims; the block grid is
        ``(h // 8, w // 8)``. Cached across forwards because every block
        reuses the same mask for a given (h, w, local_range) tuple.
        """
        block_h = h // _SELF_ATTN_WINDOW[1]
        block_w = w // _SELF_ATTN_WINDOW[2]
        if (
            self._local_attn_mask is None
            or self._local_attn_mask_h != block_h
            or self._local_attn_mask_w != block_w
            or self._local_range != local_range
        ):
            self._local_attn_mask = build_local_block_mask_shifted_vec_normal_slide(
                block_h,
                block_w,
                local_range,
                local_range,
                include_self=True,
                device=device,
            )
            self._local_attn_mask_h = block_h
            self._local_attn_mask_w = block_w
            self._local_range = local_range
        return self._local_attn_mask

    def forward(  # type: ignore[override]
        self,
        x: Tensor,
        kv_cache: BlockKVCache,
        rope_freqs: Tensor,
        *,
        f: int,
        h: int,
        w: int,
        topk: int,
        local_range: int,
    ) -> Tensor:
        """Run block-sparse self-attention and update ``kv_cache``.

        Args:
            x: Input tensor with shape ``[B, L, D]`` where ``L = f * h * w``
                and ``D = n_heads * head_dim``.
            kv_cache: Streaming KV cache returned by ``initialize_cache``.
            rope_freqs: 3D RoPE frequencies of shape ``[L, 1, 1, head_dim // 2]``.
            f, h, w: Patchify-space (T, H, W) dimensions; used for the window
                partition.
            topk: Top-k block budget for the draft block-sparse mask.
            local_range: Window radius for the local-block mask.

        Returns:
            Tensor of shape ``[B, L, D]``.
        """
        B, L, D = x.shape
        assert B == 1, "block_sparse_attn currently only supports batch size 1"
        assert L == f * h * w, f"Sequence length {L} != f*h*w ({f * h * w})"
        n, d = self.n_heads, self.head_dim

        q = self.norm_q(self.q(x)).reshape(B, L, n, d)
        k = self.norm_k(self.k(x)).reshape(B, L, n, d)
        v = self.v(x).reshape(B, L, n, d)
        q = apply_rope_freqs(q, rope_freqs, interleaved=True)
        k = apply_rope_freqs(k, rope_freqs, interleaved=True)

        win = _SELF_ATTN_WINDOW
        win_size = _SELF_ATTN_WINDOW_TOKENS
        # ``WindowPartition3D.partition`` expects [B, F, H, W, C] with the
        # head dim flattened into C.
        q5 = q.reshape(B, f, h, w, n * d)
        k5 = k.reshape(B, f, h, w, n * d)
        v5 = v.reshape(B, f, h, w, n * d)
        q_w_flat = WindowPartition3D.partition(q5, win)  # [B*block_n, win_size, n*d]
        k_w_flat = WindowPartition3D.partition(k5, win)
        v_w_flat = WindowPartition3D.partition(v5, win)
        block_n = q_w_flat.shape[0] // B

        # Stash the per-chunk K/V into the rolling cache. ``cached_k()`` then
        # yields the latest ``kv_ratio + 1`` chunks worth of tokens including
        # the just-written one.
        chunk_tokens = block_n * win_size
        k_chunk = k_w_flat.reshape(B, chunk_tokens, n, d)
        v_chunk = v_w_flat.reshape(B, chunk_tokens, n, d)
        kv_cache.update(k_chunk, v_chunk)

        cached_k = kv_cache.cached_k()  # [B, total_tokens_kv, n, d]
        cached_v = kv_cache.cached_v()
        total_tokens_kv = cached_k.shape[-3]
        block_n_kv = total_tokens_kv // win_size

        local_mask = self._build_or_get_local_mask(h, w, local_range, device=x.device)
        k_w_for_mask = cached_k.reshape(B * block_n_kv, win_size, n * d)
        seqlen_chunks = f // win[0]
        attention_mask = generate_draft_block_mask(
            B,
            n,
            seqlen_chunks,
            q_w_flat,
            k_w_for_mask,
            topk=topk,
            local_attn_mask=local_mask,
        )

        # Run block-sparse attention. The kernel wants [total_seq, n, d] (no
        # batch dim) plus cumulative seqlens for batch=1.
        seqlen_q = chunk_tokens
        seqlen_kv = block_n_kv * win_size
        q_in = q_w_flat.reshape(B * seqlen_q, n, d).contiguous()
        k_in = cached_k.reshape(B * seqlen_kv, n, d).contiguous()
        v_in = cached_v.reshape(B * seqlen_kv, n, d).contiguous()

        # Reuse the int32 helpers populated once by ``initialize_cache``.
        # The ``cu_seqlens_k`` table is keyed by the effective number of
        # cached chunks (1..kv_ratio + 1); ``total_tokens_kv`` is always an
        # integer multiple of ``chunk_tokens`` because the rolling cache
        # appends whole chunks.
        assert self._cu_seqlens_q is not None and self._chunk_tokens is not None, (
            "SparseSelfAttention.initialize_cache must be called before forward."
        )
        assert seqlen_q == self._chunk_tokens, (
            f"seqlen_q={seqlen_q} disagrees with chunk_tokens={self._chunk_tokens} "
            "registered at initialize_cache time."
        )
        cu_seqlens_q = self._cu_seqlens_q
        head_mask_type = self._head_mask_type
        cu_seqlens_k = self._cu_seqlens_k_table[
            total_tokens_kv // self._chunk_tokens - 1
        ]

        # Direct ``BlockSparseAttnFunc.apply`` instead of ``block_sparse_attn_func``.
        # The wrapper's only added behaviour is ``replace_ones_with_count``,
        # which we already pre-computed into ``head_mask_type`` at
        # ``initialize_cache`` time. Bypassing the wrapper removes the
        # ``Tensor.masked_scatter`` graph break (and the resulting dynamo
        # recompile-cache thrash) from the per-layer hot path. All knobs
        # below are static for FlashVSR; spelled out here so a future
        # upstream signature shift fails loudly rather than silently.
        out = BlockSparseAttnFunc.apply(
            q_in,
            k_in,
            v_in,
            cu_seqlens_q,
            cu_seqlens_k,
            128,  # m_block_dim
            128,  # n_block_dim
            head_mask_type,  # already in [1, 2, ..., n_heads] post-renumber form
            None,  # streaming_info
            attention_mask,  # base_blockmask
            seqlen_q,  # max_seqlen_q_
            seqlen_kv,  # max_seqlen_k_
            0.0,  # p_dropout
            None,  # softmax_scale (forces 1 / sqrt(head_dim) inside .forward)
            False,  # is_causal
            False,  # exact_streaming
            False,  # return_softmax / return_attn_probs
            -1,  # window_size_left
            -1,  # window_size_right
            False,  # deterministic
            torch.is_grad_enabled(),  # is_grad_enabled
        )  # [seqlen_q, n, d]

        out = out.reshape(B * block_n, win_size, n * d)
        out = WindowPartition3D.reverse(out, win, (f, h, w))
        out = out.reshape(B, L, n * d)
        return self.o(out)


class FlashVSRBlock(Block):
    """Wan 2.1 transformer block + FlashVSR sparse self-attention + LR-latent injection."""

    def __init__(
        self,
        dim: int,
        ffn_dim: int,
        num_heads: int,
        cross_attn_norm: bool = True,
        eps: float = 1e-6,
        i2v: bool = False,
    ) -> None:
        super().__init__(
            dim=dim,
            ffn_dim=ffn_dim,
            num_heads=num_heads,
            cross_attn_norm=cross_attn_norm,
            eps=eps,
            i2v=i2v,
        )
        # Replace the dense ``SelfAttention`` allocated by ``Block.__init__``
        # with the FlashVSR sparse one. Parameter naming
        # (``self_attn.{q,k,v,o,norm_q,norm_k}``) is preserved.
        self.self_attn = SparseSelfAttention(
            query_dim=dim,
            n_heads=num_heads,
            head_dim=dim // num_heads,
            eps=eps,
        )

    def forward(  # type: ignore[override]
        self,
        x: Tensor,
        e: Tensor,
        cache: BlockCache,
        rope_freqs: Tensor,
        *,
        f: int,
        h: int,
        w: int,
        topk: int,
        local_range: int,
        lq_latent: Optional[Tensor] = None,
    ) -> Tensor:
        """Run one FlashVSR block update.

        Args:
            x: Input hidden states ``[B, L, D]``.
            e: Modulation tensor ``[..., 6, D]``.
            cache: Per-block KV cache.
            rope_freqs: 3D RoPE frequencies for this AR step.
            f, h, w: Patchify-space (T, H, W) dimensions.
            topk: Block-sparse top-k budget.
            local_range: Local-block window radius.
            lq_latent: Optional per-block low-resolution latent contribution.
                Added to ``x`` before the AdaLN modulation, matching the
                legacy FlashVSR injection point.
        """
        assert self._parameters_updated_after_loading_checkpoint, (
            "Call ``update_parameters_after_loading_checkpoint`` after loading the checkpoint."
        )
        if lq_latent is not None:
            x = x + lq_latent

        e_chunks = (self.modulation + e).chunk(6, dim=-2)

        y = self.norm1(x) * (1 + e_chunks[1]) + e_chunks[0]
        y = self.self_attn(
            y,
            kv_cache=cache.self_attn,
            rope_freqs=rope_freqs,
            f=f,
            h=h,
            w=w,
            topk=topk,
            local_range=local_range,
        )
        x = x + (y * e_chunks[2])

        x = x + self.cross_attn(
            self.norm3(x),
            kv_cache=cache.cross_attn,
        )
        y = self.norm2(x) * (1 + e_chunks[4]) + e_chunks[3]
        y = self.ffn(y)
        x = x + (y * e_chunks[5])
        return x


@dataclass
class FlashVSRDiTNetworkConfig(WanDiTNetworkConfig):
    """Network config for the FlashVSR DiT.

    The default values match the ``flashvsr_tiny_long`` checkpoint shipped
    with FlashVSR-v1.1 (see ``stage_flashvsr_weights.py``).
    """

    _target: type["FlashVSRDiTNetwork"] = field(  # type: ignore[assignment]
        default_factory=lambda: FlashVSRDiTNetwork
    )

    dim: int = 1536
    ffn_dim: int = 8960
    num_heads: int = 12
    num_layers: int = 30
    in_dim: int = 16
    out_dim: int = 16
    text_dim: int = 4096
    freq_dim: int = 256
    eps: float = 1e-6
    patch_size: tuple[int, int, int] = (1, 2, 2)
    cross_attn_norm: bool = True
    cross_attn_enable_img: bool = False
    text_len: int = 512
    """Token length of the FlashVSR positive-prompt tensor."""
    patch_embedding_type: str = "conv3d"


class FlashVSRDiTNetwork(WanDiTNetwork):
    """Wan DiT backbone wired to use ``FlashVSRBlock``.

    Reuses :class:`WanDiTNetwork` for patch embedding, time embedding, the
    head, ``initialize_cache``, ``update_parameters_after_loading_checkpoint``
    and patchify / unpatchify. Overrides ``_build_block`` to instantiate
    ``FlashVSRBlock`` and ``forward`` to thread per-block ``lq_latents``
    through the block loop.
    """

    def _build_block(self, layer_idx: int) -> Block:
        return FlashVSRBlock(
            dim=self.dim,
            ffn_dim=self.ffn_dim,
            num_heads=self.num_heads,
            cross_attn_norm=self.cross_attn_norm,
            eps=self.eps,
            i2v=self.cross_attn_enable_img,
        )

    def forward(  # type: ignore[override]
        self,
        x: Tensor,
        timesteps: Tensor,
        cache,
        rope_freqs: Tensor,
        current_chunk_idx: int = 0,
        eager_mode: bool = True,
        block_extra_kwargs: Optional[dict[str, Any]] = None,
        lq_latents: Optional[List[Tensor] | Tensor] = None,
    ) -> Tensor:
        """Run one denoising forward.

        Args:
            x: Patchified noisy latent ``[..., L, D_in]``.
            timesteps: Diffusion timesteps tensor.
            cache: Per-network KV cache (``WanDiTNetworkCache``).
            rope_freqs: 3D RoPE frequencies for this AR step.
            current_chunk_idx: Chunk index for the streaming cache.
            eager_mode: If True, run cache before / after update hooks.
            block_extra_kwargs: Per-call kwargs uniformly forwarded to every
                block (``f``, ``h``, ``w``, ``topk``, ``local_range``).
            lq_latents: Optional per-block additive low-resolution latent
                contribution; ``lq_latents[i]`` (when set) is added to ``x``
                before block ``i``'s AdaLN modulation. Accepts either a
                ``List[Tensor]`` of length ``num_layers`` or a single
                leading-dim ``Tensor`` of shape
                ``[num_layers, ..., L, dim]`` -- both are indexable as
                ``lq_latents[i]`` and report the layer count via
                ``len(lq_latents)``. The Tensor form is used by
                ``FlashVSRTransformer._capturable_dit_forward`` so
                :class:`flashdreams.infra.cuda_graph.CUDAGraphWrapper` can
                stage the LR contribution as a single static input buffer
                (a list would be forwarded verbatim, leaving captured
                kernels referencing the wrong addresses on replay).

        Returns:
            Tensor with shape ``[..., L, prod(patch_size) * out_dim]``.
        """
        assert self._parameters_updated_after_loading_checkpoint, (
            "Call ``update_parameters_after_loading_checkpoint`` after loading the checkpoint."
        )
        block_extra_kwargs = block_extra_kwargs or {}
        batch_shape = x.shape[:-2]

        if self.patch_embedding_type == "linear":
            x = self.patch_embedding(x)
        elif self.patch_embedding_type == "conv3d":
            _weight = self.patch_embedding.weight.reshape(self.dim, -1)
            _bias = self.patch_embedding.bias
            x = torch.nn.functional.linear(x, _weight, _bias)
        else:
            raise ValueError(
                f"Invalid patch embedding type: {self.patch_embedding_type}"
            )

        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, timesteps).type_as(x)
        )
        e0 = self.time_projection(e).unflatten(-1, (6, self.dim))

        if eager_mode:
            cache.before_update(current_chunk_idx)
        for block_idx, block in enumerate(self.blocks):
            assert isinstance(block, FlashVSRBlock)
            lq = (
                lq_latents[block_idx]
                if lq_latents is not None and block_idx < len(lq_latents)
                else None
            )
            x = block(
                x=x,
                e=torch.broadcast_to(e0, batch_shape + e0.shape[-2:]),
                rope_freqs=rope_freqs,
                cache=cache[block_idx],
                lq_latent=lq,
                **block_extra_kwargs,
            )
        if eager_mode:
            cache.after_update(current_chunk_idx)

        x = self.head(x, torch.broadcast_to(e, batch_shape + (1, e.shape[-1])))
        return x
