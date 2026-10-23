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

"""MoT joint attention for hand inference."""

from __future__ import annotations

from collections.abc import Callable
from inspect import signature
from typing import cast

import torch
import torch.nn.functional as F
from torch import Tensor

varlen_attn: Callable[..., Tensor | tuple[Tensor, Tensor]] | None
try:
    from torch.nn.attention.varlen import varlen_attn as _torch_varlen_attn
except ImportError:  # pragma: no cover - exercised only on older PyTorch builds
    varlen_attn = None
else:
    varlen_attn = _torch_varlen_attn

_VARLEN_ACCEPTS_GQA = (
    varlen_attn is not None and "enable_gqa" in signature(varlen_attn).parameters
)
"""Whether the installed varlen API accepts an explicit GQA opt-in."""


def varlen_attention_available() -> bool:
    """Whether this PyTorch build exposes fused variable-length attention."""
    return varlen_attn is not None


def cached_joint_attention(
    q: Tensor,
    k_all: Tensor,
    v_all: Tensor,
    n_cached: int,
    within_mask: Tensor | None,
    *,
    cached_mask: Tensor | None = None,
    enable_gqa: bool = False,
) -> Tensor:
    """Incremental (KV-cache) attention for the current chunk's query tokens.

    The current chunk attends to **all** cached keys (clean text + past clean
    frames are always visible under the temporal-causal rule, since they precede
    the current frame) plus its own tokens, gated within the current block by
    ``within_mask`` (temporal-causal among current-chunk frames; ``None`` = full).

    Args:
        q: ``[B, Hq, s_cur, d]`` current-chunk queries.
        k_all, v_all: ``[B, Hq, n_cached + s_cur, d]`` cached ++ current K/V
            when pre-expanded, or ``[B, Hkv, n_cached + s_cur, d]`` when
            ``enable_gqa=True``.
        n_cached: number of cached (visible) key positions.
        within_mask: ``[s_cur, s_cur]`` bool over the current block, or ``None``.
        cached_mask: Optional ``[n_cached]`` bool validity mask. This lets a
            filling cache expose a fixed-size, right-padded history without
            allowing attention to read the unwritten slots.
        enable_gqa: pass unexpanded grouped K/V into PyTorch SDPA.

    Returns:
        ``[B, Hq, s_cur, d]`` attention output for the current tokens.
    """
    if cached_mask is not None:
        cached_mask = cached_mask.to(device=q.device, dtype=torch.bool)
        if cached_mask.shape != (n_cached,):
            raise ValueError(
                f"cached_mask must have shape {(n_cached,)}, "
                f"got {tuple(cached_mask.shape)}"
            )
        s_cur = q.shape[-2]
        if within_mask is None:
            current_mask = torch.ones(s_cur, dtype=torch.bool, device=q.device)
            mask = torch.cat((cached_mask, current_mask))[None, None, None]
        else:
            cached = cached_mask.unsqueeze(0).expand(s_cur, -1)
            mask = torch.cat((cached, within_mask), dim=1)[None, None]
        return F.scaled_dot_product_attention(
            q,
            k_all,
            v_all,
            attn_mask=mask,
            enable_gqa=enable_gqa,
        )
    if within_mask is None:
        # All cached keys are visible and the current chunk is a single frame
        # (full self): the mask would be all-True, i.e. no mask. Passing
        # ``attn_mask=None`` lets SDPA dispatch to FlashAttention / cuDNN instead
        # of the much slower dense-mask (math/efficient) kernel — the main
        # per-step win on the len_t=1 hot path.
        return F.scaled_dot_product_attention(
            q,
            k_all,
            v_all,
            enable_gqa=enable_gqa,
        )
    s_cur = q.shape[-2]
    s_all = k_all.shape[-2]
    mask = torch.ones((s_cur, s_all), dtype=torch.bool, device=q.device)
    mask[:, n_cached:] = within_mask
    return F.scaled_dot_product_attention(
        q,
        k_all,
        v_all,
        attn_mask=mask[None, None],
        enable_gqa=enable_gqa,
    )


def valid_length_cached_attention(
    q: Tensor,
    k_all: Tensor,
    v_all: Tensor,
    valid_cached: Tensor,
    current_tokens: int,
    *,
    causal_block_tokens: int | None = None,
) -> Tensor:
    """FlashAttention over a fixed K/V allocation with a dynamic valid prefix.

    ``k_all``/``v_all`` are laid out as ``[fixed cache | current chunk]``. During
    cache fill, unused cache slots sit between the valid prefix and the current
    chunk. Copy the current K/V into the first unused slots, then tell varlen
    FlashAttention the logical prefix length. Trailing physical slots remain
    allocated for pointer/shape stability but are never visible to attention.

    The cumulative sequence length is a device tensor, so one captured CUDA
    graph can replay while ``valid_cached`` grows from frame zero to saturation.
    """
    if varlen_attn is None:
        raise RuntimeError(
            "valid-length cache attention requires PyTorch varlen FlashAttention"
        )
    attention_fn = varlen_attn
    if q.shape[0] != 1 or k_all.shape[0] != 1 or v_all.shape[0] != 1:
        raise ValueError("valid-length cache attention currently requires batch=1")
    if valid_cached.numel() != 1:
        raise ValueError("valid_cached must be a scalar tensor")
    current = int(current_tokens)
    if current <= 0 or current > int(k_all.shape[-2]):
        raise ValueError(f"invalid current token count {current}")

    physical_cached = int(k_all.shape[-2]) - current
    valid_cached = valid_cached.to(device=q.device).reshape(())
    destination = valid_cached.to(dtype=torch.long) + torch.arange(
        current, device=q.device
    )
    packed_k = k_all.clone().index_copy_(
        2, destination, k_all[..., physical_cached:, :]
    )
    packed_v = v_all.clone().index_copy_(
        2, destination, v_all[..., physical_cached:, :]
    )
    packed_k = packed_k[0].transpose(0, 1)
    packed_v = packed_v[0].transpose(0, 1)

    def attend(query: Tensor, valid_total: Tensor) -> Tensor:
        query_tokens = int(query.shape[-2])
        cu_seq_q = torch.arange(2, device=q.device, dtype=torch.int32) * query_tokens
        cu_seq_k = torch.stack((torch.zeros_like(valid_total), valid_total))
        gqa_kwargs = (
            {"enable_gqa": query.shape[1] != packed_k.shape[1]}
            if _VARLEN_ACCEPTS_GQA
            else {}
        )
        out = attention_fn(
            query[0].transpose(0, 1),
            packed_k,
            packed_v,
            cu_seq_q,
            cu_seq_k,
            query_tokens,
            int(k_all.shape[-2]),
            **gqa_kwargs,
        )
        return cast(Tensor, out).transpose(0, 1).unsqueeze(0)

    valid_cached = valid_cached.to(dtype=torch.int32)
    if causal_block_tokens is None:
        return attend(q, valid_cached + current)

    block = int(causal_block_tokens)
    if block <= 0 or current % block:
        raise ValueError(
            "causal_block_tokens must be positive and divide current_tokens"
        )
    outputs = []
    for start in range(0, current, block):
        end = start + block
        outputs.append(attend(q[..., start:end, :], valid_cached + end))
    return torch.cat(outputs, dim=2)
