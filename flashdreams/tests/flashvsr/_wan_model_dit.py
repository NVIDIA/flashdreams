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

"""Frozen parity reference for the FlashVSR DiT migration.

The live import path goes through
``flashdreams/recipes/flashvsr/transformer/network.py`` (FlashVSR on
``Wan21Transformer``); this module is dynamically loaded only by
``flashdreams/tests/flashvsr/test_dit_replacement.py`` to verify bit-for-bit
numerical agreement, mirroring the ``_tcdecoder.py`` / ``decoder/network.py``
convention. Do not modify.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from block_sparse_attn import block_sparse_attn_func
from einops import rearrange

from flashdreams.recipes.wan.transformer.impl.modules import (
    Head,
    sinusoidal_embedding_1d,
)
from flashdreams.recipes.wan.transformer.impl.rope import (
    RotaryPositionEmbedding3D,
    apply_rope_freqs,
)


def torch_cudnn_attention(
    query, key, value, return_lse: bool = True
) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
    (
        out,
        lse,
        cum_seq_q,
        cum_seq_k,
        max_q,
        max_k,
        philox_seed,
        philox_offset,
        debug_attn_mask,
        # ``torch.ops.aten._scaled_dot_product_cudnn_attention`` is dispatched
        # through ``OpOverloadPacket`` whose stub uses ``EllipsisType``-typed
        # variadic args; ty can't see the named-kwarg overload. Runtime works.
    ) = torch.ops.aten._scaled_dot_product_cudnn_attention(
        query=query,
        key=key,
        value=value,
        attn_bias=None,  # ty: ignore[invalid-argument-type]
        compute_log_sumexp=True,  # ty: ignore[invalid-argument-type]
    )
    if return_lse:
        return out, lse
    else:
        return out


@torch.no_grad()
def build_local_block_mask_shifted_vec_normal_slide(
    block_h: int,
    block_w: int,
    win_h: int = 6,
    win_w: int = 6,
    include_self: bool = True,
    device=None,
) -> torch.Tensor:
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


class WindowPartition3D:
    """Partition / reverse-partition helpers for 5-D tensors (B,F,H,W,C)."""

    @staticmethod
    def partition(x: torch.Tensor, win: Tuple[int, int, int]):
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
        F, H, W = orig
        wf, wh, ww = win
        nf, nh, nw = F // wf, H // wh, W // ww
        B = windows.size(0) // (nf * nh * nw)
        x = windows.view(B, nf, nh, nw, wf, wh, ww, -1)
        x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous()
        return x.view(B, F, H, W, -1)


@torch.no_grad()
def generate_draft_block_mask(
    batch_size, nheads, seqlen, q_w, k_w, topk=10, local_attn_mask=None
):
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


def flash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    num_heads: int,
    attention_mask=None,
):
    if attention_mask is not None:
        seqlen = q.shape[1]
        seqlen_kv = k.shape[1]
        q = rearrange(q, "b s (n d) -> (b s) n d", n=num_heads)
        k = rearrange(k, "b s (n d) -> (b s) n d", n=num_heads)
        v = rearrange(v, "b s (n d) -> (b s) n d", n=num_heads)
        cu_seqlens_q = torch.tensor([0, seqlen], device=q.device, dtype=torch.int32)
        cu_seqlens_k = torch.tensor([0, seqlen_kv], device=q.device, dtype=torch.int32)
        head_mask_type = torch.tensor(
            [1] * num_heads, device=q.device, dtype=torch.int32
        )
        streaming_info = None
        base_blockmask = attention_mask
        max_seqlen_q_ = seqlen
        max_seqlen_k_ = seqlen_kv
        p_dropout = 0.0
        x = block_sparse_attn_func(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_k,
            head_mask_type,
            streaming_info,
            base_blockmask,
            max_seqlen_q_,
            max_seqlen_k_,
            p_dropout,
            deterministic=False,
            softmax_scale=None,
            is_causal=False,
            exact_streaming=False,
            return_attn_probs=False,
        ).unsqueeze(0)
        x = rearrange(x, "b s n d -> b s (n d)", n=num_heads)
    else:
        q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
        x = torch_cudnn_attention(q, k, v, return_lse=False)
        x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)
    return x


def _apply_rope(x: torch.Tensor, freqs: torch.Tensor, num_heads: int) -> torch.Tensor:
    """Apply 3D RoPE to a flat ``[B, S, n*d]`` tensor via the flashdreams primitive."""
    assert x.ndim == 3 and freqs.ndim == 4
    b, s, D = x.shape
    x = x.reshape(b, s, num_heads, D // num_heads)
    x_out = apply_rope_freqs(x, freqs, interleaved=True)
    return x_out.reshape(b, s, D)


class AttentionModule(nn.Module):
    def __init__(self, num_heads):
        super().__init__()
        self.num_heads = num_heads

    def forward(self, q, k, v, attention_mask=None):
        x = flash_attention(
            q=q, k=k, v=v, num_heads=self.num_heads, attention_mask=attention_mask
        )
        return x


class SelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = torch.nn.RMSNorm(dim, eps=eps)
        self.norm_k = torch.nn.RMSNorm(dim, eps=eps)

        self.attn = AttentionModule(self.num_heads)
        self.local_attn_mask = None

    def forward(
        self,
        x,
        freqs,
        f=None,
        h=None,
        w=None,
        topk=None,
        kv_len=None,
        is_stream=False,
        pre_cache_k=None,
        pre_cache_v=None,
        local_range=9,
    ):
        B, L, D = x.shape
        # ``f``/``h``/``w`` are untyped kwargs with default ``None`` but are
        # always passed by callers; ty's ``Unknown | None`` types here are
        # spurious. Ignore the operator/view errors that follow.
        assert L == f * h * w, "Sequence length mismatch with provided (f,h,w)."  # ty: ignore[unsupported-operator]

        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(x))
        v = self.v(x)
        q = _apply_rope(q, freqs, self.num_heads)
        k = _apply_rope(k, freqs, self.num_heads)

        win = (2, 8, 8)
        q = q.view(B, f, h, w, D)  # ty: ignore[invalid-argument-type]
        k = k.view(B, f, h, w, D)  # ty: ignore[invalid-argument-type]
        v = v.view(B, f, h, w, D)  # ty: ignore[invalid-argument-type]

        q_w = WindowPartition3D.partition(q, win)
        k_w = WindowPartition3D.partition(k, win)
        v_w = WindowPartition3D.partition(v, win)

        seqlen = f // win[0]  # ty: ignore[unsupported-operator]
        one_len = k_w.shape[0] // B // seqlen
        if pre_cache_k is not None and pre_cache_v is not None:
            k_w = torch.cat([pre_cache_k, k_w], dim=0)
            v_w = torch.cat([pre_cache_v, v_w], dim=0)

        block_n = q_w.shape[0] // B
        block_s = q_w.shape[1]
        block_n_kv = k_w.shape[0] // B

        reorder_q = rearrange(
            q_w,
            "(b block_n) (block_s) d -> b (block_n block_s) d",
            block_n=block_n,
            block_s=block_s,
        )
        reorder_k = rearrange(
            k_w,
            "(b block_n) (block_s) d -> b (block_n block_s) d",
            block_n=block_n_kv,
            block_s=block_s,
        )
        reorder_v = rearrange(
            v_w,
            "(b block_n) (block_s) d -> b (block_n block_s) d",
            block_n=block_n_kv,
            block_s=block_s,
        )

        if (
            self.local_attn_mask is None
            or self.local_attn_mask_h != h // 8  # ty: ignore[unsupported-operator]
            or self.local_attn_mask_w != w // 8  # ty: ignore[unsupported-operator]
            or self.local_range != local_range
        ):
            self.local_attn_mask = build_local_block_mask_shifted_vec_normal_slide(
                h // 8,  # ty: ignore[unsupported-operator]
                w // 8,  # ty: ignore[unsupported-operator]
                local_range,
                local_range,
                include_self=True,
                device=k_w.device,
            )
            self.local_attn_mask_h = h // 8  # ty: ignore[unsupported-operator]
            self.local_attn_mask_w = w // 8  # ty: ignore[unsupported-operator]
            self.local_range = local_range
        attention_mask = generate_draft_block_mask(
            B,
            self.num_heads,
            seqlen,
            q_w,
            k_w,
            topk=topk,
            local_attn_mask=self.local_attn_mask,
        )

        x = self.attn(reorder_q, reorder_k, reorder_v, attention_mask)

        cur_block_n, cur_block_s, _ = k_w.shape
        cache_num = cur_block_n // one_len
        if cache_num > kv_len:
            cache_k = k_w[one_len:, :, :]
            cache_v = v_w[one_len:, :, :]
        else:
            cache_k = k_w
            cache_v = v_w

        x = rearrange(
            x,
            "b (block_n block_s) d -> (b block_n) (block_s) d",
            block_n=block_n,
            block_s=block_s,
        )
        x = WindowPartition3D.reverse(x, win, (f, h, w))  # ty: ignore[invalid-argument-type]
        x = x.view(B, f * h * w, D)  # ty: ignore[unsupported-operator]

        if is_stream:
            return self.o(x), cache_k, cache_v
        return self.o(x)


class CrossAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)

        self.norm_q = torch.nn.RMSNorm(dim, eps=eps)
        self.norm_k = torch.nn.RMSNorm(dim, eps=eps)

        self.attn = AttentionModule(self.num_heads)

        self.cache_k = None
        self.cache_v = None

    @torch.no_grad()
    def init_cache(self, ctx: torch.Tensor):
        self.cache_k = self.norm_k(self.k(ctx))
        self.cache_v = self.v(ctx)

    def clear_cache(self):
        self.cache_k = None
        self.cache_v = None

    def forward(self, x: torch.Tensor, y: torch.Tensor, is_stream: bool = False):
        q = self.norm_q(self.q(x))
        assert self.cache_k is not None and self.cache_v is not None
        k = self.cache_k
        v = self.cache_v

        x = self.attn(q, k, v)
        return self.o(x)


class GateModule(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()

    def forward(self, x, gate, residual):
        return x + gate * residual


class DiTBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, ffn_dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim

        self.self_attn = SelfAttention(dim, num_heads, eps)
        self.cross_attn = CrossAttention(dim, num_heads, eps)

        self.norm1 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm3 = nn.LayerNorm(dim, eps=eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(ffn_dim, dim),
        )
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)
        self.gate = GateModule()

    def forward(
        self,
        x,
        context,
        t_mod,
        freqs,
        f,
        h,
        w,
        topk=None,
        kv_len=None,
        is_stream=False,
        pre_cache_k=None,
        pre_cache_v=None,
        local_range=9,
    ):
        # ``self.modulation`` is ``(1, 6, D)`` pre-fuse and ``(6, D)`` after
        # ``WanModel.update_parameters_after_loading_checkpoint()``; ``dim=-2``
        # works for both layouts since ``t_mod`` is ``(..., 6, D)``.
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod
        ).chunk(6, dim=-2)
        input_x = self.norm1(x) * (1 + scale_msa) + shift_msa
        self_attn_output, self_attn_cache_k, self_attn_cache_v = self.self_attn(
            input_x,
            freqs,
            f,
            h,
            w,
            topk,
            kv_len=kv_len,
            is_stream=is_stream,
            pre_cache_k=pre_cache_k,
            pre_cache_v=pre_cache_v,
            local_range=local_range,
        )

        x = self.gate(x, gate_msa, self_attn_output)
        x = x + self.cross_attn(self.norm3(x), context, is_stream=is_stream)
        input_x = self.norm2(x) * (1 + scale_mlp) + shift_mlp
        x = self.gate(x, gate_mlp, self.ffn(input_x))
        if is_stream:
            return x, self_attn_cache_k, self_attn_cache_v
        return x


class WanModel(torch.nn.Module):
    def __init__(
        self,
        dim: int,
        in_dim: int,
        ffn_dim: int,
        out_dim: int,
        text_dim: int,
        freq_dim: int,
        eps: float,
        patch_size: Tuple[int, int, int],
        num_heads: int,
        num_layers: int,
        has_image_input: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.freq_dim = freq_dim
        self.patch_size = patch_size
        self.num_heads = num_heads

        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size
        )

        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim), nn.GELU(approximate="tanh"), nn.Linear(dim, dim)
        )
        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim)
        )
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))

        self.out_dim = out_dim

        self.blocks = nn.ModuleList(
            [DiTBlock(dim, num_heads, ffn_dim, eps) for _ in range(num_layers)]
        )
        self.head = Head(dim, out_dim, patch_size, eps)

        self.rope_freq_first: RotaryPositionEmbedding3D | None = None
        self.rope_freq_other: RotaryPositionEmbedding3D | None = None

        self._cross_kv_initialized = False
        self._parameters_updated_after_loading_checkpoint = False

    @torch.no_grad()
    def update_parameters_after_loading_checkpoint(self) -> None:
        """Fold load-time-known ops into weights; idempotent.

        Mirrors ``WanDiTNetwork.update_parameters_after_loading_checkpoint``:

        - Folds the ``(kt kh kw c) -> (c kt kh kw)`` channel-shuffle into
          ``head.head`` so the unpatchify pattern can match the flashdreams
          convention with no runtime rearrange beyond the standard one.
        - Squeezes block / head modulation from ``(1, 6, D)`` / ``(1, 2, D)``
          to ``(6, D)`` / ``(2, D)`` so broadcasting against arbitrary
          leading batch dims falls out of the pattern automatically.

        Must be called after ``load_state_dict`` and before the first
        ``forward``.
        """
        if self._parameters_updated_after_loading_checkpoint:
            return

        # Fold the (kt kh kw c) -> (c kt kh kw) shuffle into head.head's
        # weights/bias rows so the unpatchify rearrange can use the
        # flashdreams "(c kt kh kw)" pattern directly.
        kt, kh, kw = self.patch_size
        self.head.head.weight.data = rearrange(
            self.head.head.weight,
            "(kt kh kw c) in_dim -> (c kt kh kw) in_dim",
            kt=kt,
            kh=kh,
            kw=kw,
            c=self.out_dim,
        ).contiguous()
        if self.head.head.bias is not None:
            self.head.head.bias.data = rearrange(
                self.head.head.bias,
                "(kt kh kw c) -> (c kt kh kw)",
                kt=kt,
                kh=kh,
                kw=kw,
                c=self.out_dim,
            ).contiguous()

        # ``nn.Module.__getattr__`` returns ``Tensor | Module`` for items
        # accessed off a ``ModuleList``; ty cannot see ``modulation`` is
        # always a parameter on these blocks. Runtime guarantees it.
        for blk in self.blocks:
            if blk.modulation.dim() == 3:  # ty: ignore[call-non-callable]
                blk.modulation.data = blk.modulation.data.squeeze(0)  # ty: ignore[call-non-callable]
        if self.head.modulation.dim() == 3:
            self.head.modulation.data = self.head.modulation.data.squeeze(0)
        # ``flashdreams.recipes.wan.transformer.impl.modules.Head`` asserts
        # this flag in its ``forward``.
        self.head._parameters_updated_after_loading_checkpoint = True

        self._parameters_updated_after_loading_checkpoint = True

    def clear_cross_kv(self):
        for blk in self.blocks:
            blk.cross_attn.clear_cache()  # ty: ignore[call-non-callable, unresolved-attribute]
        self._cross_kv_initialized = False

    @torch.no_grad()
    def reinit_cross_kv(self, new_context: torch.Tensor):
        ctx_txt = self.text_embedding(new_context)
        for blk in self.blocks:
            blk.cross_attn.init_cache(ctx_txt)  # ty: ignore[call-non-callable, unresolved-attribute]
        self._cross_kv_initialized = True

    def patchify(self, x: torch.Tensor):
        # Equivalent to Conv3d(in_dim -> dim, kernel=stride=patch_size) but
        # implemented as a fused rearrange + linear: cheaper at inference
        # time and matches ``WanDiTNetwork.forward``'s
        # ``patch_embedding_type="conv3d"`` fast path.
        kt, kh, kw = self.patch_size
        B, _, T, H, W = x.shape
        f, h, w = T // kt, H // kh, W // kw
        x = rearrange(
            x,
            "b c (f kt) (h kh) (w kw) -> b (f h w) (c kt kh kw)",
            kt=kt,
            kh=kh,
            kw=kw,
        )
        weight = self.patch_embedding.weight.reshape(self.dim, -1)
        bias = self.patch_embedding.bias
        x = F.linear(x, weight, bias)
        return x, (f, h, w)

    def unpatchify(self, x: torch.Tensor, grid_size: torch.Tensor):
        # Pattern matches the post-fuse layout: ``head.head`` rows are
        # ``(c kt kh kw)`` after ``update_parameters_after_loading_checkpoint``.
        return rearrange(
            x,
            "b (f h w) (c x y z) -> b c (f x) (h y) (w z)",
            f=grid_size[0],
            h=grid_size[1],
            w=grid_size[2],
            x=self.patch_size[0],
            y=self.patch_size[1],
            z=self.patch_size[2],
        )

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        LQ_latents: Optional[torch.Tensor] = None,
        is_stream: bool = False,
        pre_cache_k: Optional[list[torch.Tensor]] = None,
        pre_cache_v: Optional[list[torch.Tensor]] = None,
        topk_ratio: float = 2.0,
        kv_ratio: float = 3.0,
        cur_process_idx: int = 0,
        # Annotation is wrong (should be ``Optional[Tensor]``); preserved
        # verbatim to keep this file byte-identical to the upstream reference.
        t_mod: torch.Tensor = None,  # ty: ignore[invalid-parameter-default]
        t: torch.Tensor = None,  # ty: ignore[invalid-parameter-default]
        local_range: int = 9,
    ):
        assert self._parameters_updated_after_loading_checkpoint, (
            "Call WanModel.update_parameters_after_loading_checkpoint() "
            "after loading the checkpoint and before the first forward."
        )
        x, (f, h, w) = self.patchify(x)

        win = (2, 8, 8)
        window_size = win[0] * h * w // 128
        square_num = window_size * window_size
        topk = int(square_num * topk_ratio) - 1
        kv_len = int(kv_ratio)

        if cur_process_idx == 0:
            if self.rope_freq_first is None:
                self.rope_freq_first = RotaryPositionEmbedding3D(
                    head_dim=self.dim // self.num_heads,
                    len_h=h,
                    len_w=w,
                    len_t=f,
                    interleaved=True,
                    device=x.device,
                )
            freqs = self.rope_freq_first.shift_t(0)
        else:
            if self.rope_freq_other is None:
                self.rope_freq_other = RotaryPositionEmbedding3D(
                    head_dim=self.dim // self.num_heads,
                    len_h=h,
                    len_w=w,
                    len_t=f,
                    interleaved=True,
                    device=x.device,
                )
            freqs = self.rope_freq_other.shift_t(2 + cur_process_idx * 2)

        for block_id, block in enumerate(self.blocks):
            if LQ_latents is not None and block_id < len(LQ_latents):
                x = x + LQ_latents[block_id]
            x, last_pre_cache_k, last_pre_cache_v = block(
                x,
                context,
                t_mod,
                freqs,
                f,
                h,
                w,
                topk,
                kv_len=kv_len,
                is_stream=is_stream,
                pre_cache_k=pre_cache_k[block_id] if pre_cache_k is not None else None,
                pre_cache_v=pre_cache_v[block_id] if pre_cache_v is not None else None,
                local_range=local_range,
            )
            if pre_cache_k is not None:
                pre_cache_k[block_id] = last_pre_cache_k
            if pre_cache_v is not None:
                pre_cache_v[block_id] = last_pre_cache_v

        # The flashdreams ``Head`` expects ``e`` of shape ``[..., 1, D]``; the
        # upsampler precomputes ``t`` as ``[B, D]`` (post ``time_embedding``),
        # so we add a singleton ``L=1`` axis here.
        if t.dim() == 2:
            t = t.unsqueeze(-2)
        x = self.head(x, t)
        # ``unpatchify``'s ``grid_size`` annotation says ``Tensor`` but the
        # legacy implementation accepts a ``tuple[int, int, int]`` shape spec.
        x = self.unpatchify(x, (f, h, w))  # ty: ignore[invalid-argument-type]
        return x, pre_cache_k, pre_cache_v
