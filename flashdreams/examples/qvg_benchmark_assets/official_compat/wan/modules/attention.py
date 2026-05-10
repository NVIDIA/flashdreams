"""SDPA fallback for official QVG when flash-attn is unavailable."""

from __future__ import annotations

import torch
import torch.nn.functional as F

FLASH_ATTN_2_AVAILABLE = False
FLASH_ATTN_3_AVAILABLE = False

__all__ = ["flash_attention", "attention"]


def flash_attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.0,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    version=None,
):
    out_dtype = q.dtype
    batch_size, q_len_total, q_heads, _ = q.shape
    k_heads = k.shape[2]
    out = q.new_zeros((batch_size, q_len_total, q_heads, v.shape[-1]), dtype=out_dtype)

    for batch_idx in range(batch_size):
        q_len = int(q_lens[batch_idx].item()) if q_lens is not None else q_len_total
        k_len = int(k_lens[batch_idx].item()) if k_lens is not None else k.shape[1]
        qi = q[batch_idx : batch_idx + 1, :q_len].transpose(1, 2).to(dtype)
        ki = k[batch_idx : batch_idx + 1, :k_len].transpose(1, 2).to(dtype)
        vi = v[batch_idx : batch_idx + 1, :k_len].transpose(1, 2).to(dtype)

        if q_scale is not None:
            qi = qi * q_scale
        if q_heads != k_heads:
            assert q_heads % k_heads == 0, (
                f"Expected query heads divisible by KV heads, got {q_heads=} {k_heads=}"
            )
            repeat = q_heads // k_heads
            ki = ki.repeat_interleave(repeat, dim=1)
            vi = vi.repeat_interleave(repeat, dim=1)

        attn_mask = None
        if window_size != (-1, -1):
            left, right = window_size
            q_pos = torch.arange(q_len, device=q.device)[:, None]
            k_pos = torch.arange(k_len, device=q.device)[None, :]
            keep = torch.ones((q_len, k_len), device=q.device, dtype=torch.bool)
            if left >= 0:
                keep &= k_pos >= q_pos - left
            if right >= 0:
                keep &= k_pos <= q_pos + right
            attn_mask = keep[None, None, :, :]

        oi = F.scaled_dot_product_attention(
            qi,
            ki,
            vi,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=causal,
            scale=softmax_scale,
        )
        out[batch_idx : batch_idx + 1, :q_len] = oi.transpose(1, 2).to(out_dtype)

    return out


def attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.0,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    fa_version=None,
):
    return flash_attention(
        q=q,
        k=k,
        v=v,
        q_lens=q_lens,
        k_lens=k_lens,
        dropout_p=dropout_p,
        softmax_scale=softmax_scale,
        q_scale=q_scale,
        causal=causal,
        window_size=window_size,
        deterministic=deterministic,
        dtype=dtype,
        version=fa_version,
    )
