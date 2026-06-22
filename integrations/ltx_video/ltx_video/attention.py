# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""KV-aware LTX self-attention processors (attn1 only)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from ltx_video.kv_context import get_kv_context

if TYPE_CHECKING:
    from diffusers.models.transformers.transformer_ltx import LTXAttention


def build_causal_mask_bshd(
    chunk_len: int,
    past_len: int,
    n_heads: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """SDPA mask ``[1, n_heads, chunk_len, past_len + chunk_len]``."""
    total = past_len + chunk_len
    mask = torch.zeros(chunk_len, total, device=device, dtype=dtype)
    if chunk_len > 1:
        causal = torch.triu(
            torch.full((chunk_len, chunk_len), float("-inf"), device=device, dtype=dtype),
            diagonal=1,
        )
        mask[:, past_len:] = causal
    return mask.unsqueeze(0).unsqueeze(0).expand(1, n_heads, chunk_len, total)


def install_kv_attention_processors(transformer: nn.Module) -> int:
    """Replace attn1 processors with :class:`LTXKVAttnProcessor` (28 layers)."""
    from diffusers.models.transformers.transformer_ltx import LTXVideoAttnProcessor

    count = 0
    blocks = getattr(transformer, "transformer_blocks", None)
    if blocks is None:
        return 0

    for idx, block in enumerate(blocks):
        attn1 = getattr(block, "attn1", None)
        if attn1 is None:
            continue
        attn1.set_processor(LTXKVAttnProcessor(layer_idx=idx))
        count += 1

    print(f"[LTX KV] Installed KV processors on {count} attn1 layers")
    return count


def restore_default_attention_processors(transformer: nn.Module) -> None:
    from diffusers.models.transformers.transformer_ltx import LTXVideoAttnProcessor

    for block in getattr(transformer, "transformer_blocks", []):
        attn1 = getattr(block, "attn1", None)
        if attn1 is not None:
            attn1.set_processor(LTXVideoAttnProcessor())


class LTXKVAttnProcessor:
    """Self-attention processor with rolling KV-cache across AR chunks."""

    def __init__(self, layer_idx: int) -> None:
        self.layer_idx = layer_idx

    def __call__(
        self,
        attn: LTXAttention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        image_rotary_emb: torch.Tensor | None = None,
    ) -> torch.Tensor:
        from diffusers.models.attention_dispatch import dispatch_attention_fn
        from diffusers.models.transformers.transformer_ltx import apply_rotary_emb

        # Cross-attention path — never inject KV.
        if encoder_hidden_states is not None:
            enc = encoder_hidden_states
            batch_size, sequence_length, _ = enc.shape
            if attention_mask is not None:
                attention_mask = attn.prepare_attention_mask(
                    attention_mask, sequence_length, batch_size
                )
                attention_mask = attention_mask.view(
                    batch_size, attn.heads, -1, attention_mask.shape[-1]
                )
            query = attn.to_q(hidden_states)
            key = attn.to_k(enc)
            value = attn.to_v(enc)
            query = attn.norm_q(query)
            key = attn.norm_k(key)
            if image_rotary_emb is not None:
                query = apply_rotary_emb(query, image_rotary_emb)
                key = apply_rotary_emb(key, image_rotary_emb)
            query = query.unflatten(2, (attn.heads, -1))
            key = key.unflatten(2, (attn.heads, -1))
            value = value.unflatten(2, (attn.heads, -1))
            out = dispatch_attention_fn(
                query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
            )
            out = out.flatten(2, 3).to(query.dtype)
            out = attn.to_out[0](out)
            out = attn.to_out[1](out)
            return out

        ctx = get_kv_context()
        batch_size, sequence_length, _ = hidden_states.shape

        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)
        query = attn.norm_q(query)
        key = attn.norm_k(key)

        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb)
            key = apply_rotary_emb(key, image_rotary_emb)

        # Clone before reshape/cat so torch.compile cudagraph buffers are not aliased.
        query = query.clone()
        key = key.clone()
        value = value.clone()

        query = query.unflatten(2, (attn.heads, -1))
        key = key.unflatten(2, (attn.heads, -1))
        value = value.unflatten(2, (attn.heads, -1))

        past_len = 0
        past_layer_kv = None
        if ctx.past_kv is not None and self.layer_idx < len(ctx.past_kv):
            past_layer_kv = ctx.past_kv[self.layer_idx]

        if past_layer_kv is not None:
            past_k, past_v = past_layer_kv
            past_len = past_k.shape[1]
            key = torch.cat([past_k, key], dim=1)
            value = torch.cat([past_v, value], dim=1)

        if ctx.collect:
            ctx.set_layer_kv(
                self.layer_idx,
                (key.detach().clone(), value.detach().clone()),
            )

        attn_mask = None
        if past_len > 0:
            attn_mask = build_causal_mask_bshd(
                sequence_length, past_len, attn.heads, query.device, query.dtype
            )

        out = dispatch_attention_fn(
            query,
            key,
            value,
            attn_mask=attn_mask,
            dropout_p=0.0,
            # Match default LTX self-attention: full bidirectional within a chunk.
            # AR causality (past chunks vs current) is handled by attn_mask when past_len > 0.
            is_causal=False,
        )
        out = out.flatten(2, 3).to(query.dtype)
        out = attn.to_out[0](out)
        out = attn.to_out[1](out)
        return out


# Backward-compatible alias used by older code paths.
def patch_transformer_for_kv_cache(transformer: nn.Module) -> int:
    return install_kv_attention_processors(transformer)
