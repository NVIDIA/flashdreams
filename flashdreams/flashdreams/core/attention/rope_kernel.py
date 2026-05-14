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

"""Fused, inference-only Triton kernel for 3D rotary position embedding.

Byte-for-byte drop-in for
:func:`transformer_engine.pytorch.attention.rope.apply_rotary_pos_emb`
(``tensor_format="bshd", fused=True``) on the diffusion-transformer
forward path. Forward-only (no autograd graph), writes back in
place, and accepts the full-width ``[S, 1, 1, D]`` freqs layout
that :meth:`RotaryPositionEmbedding3D.shift_t` emits.

For each ``s`` the rotation is, in fp32::

    out[a] = x[a] * cos(f) - x[b] * sin(f)
    out[b] = x[b] * cos(f) + x[a] * sin(f)

where ``(a, b) = (d, d + D/2)`` when ``interleaved=False`` and
``(2k, 2k+1)`` when ``interleaved=True``.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch import Tensor


@triton.jit
def _rope_inference_kernel(
    x_ptr,
    freqs_ptr,
    stride_xb,
    stride_xs,
    stride_xh,
    stride_xd,
    stride_fs,
    stride_fd,
    S,
    H,
    D_HALF: tl.constexpr,
    INTERLEAVED: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Apply RoPE in place to a ``(BLOCK_S, BLOCK_H, D)`` tile of one batch row.

    Grid: ``(B, ceil(S / BLOCK_S), ceil(H / BLOCK_H))``. Each program
    loads its ``BLOCK_S`` rows of freqs once and broadcasts the
    resulting ``cos`` / ``sin`` across every head in the tile.

    The interleaved path issues one contiguous full-``D`` load and
    de-/re-interleaves in registers via ``tl.split`` / ``tl.join``;
    the non-interleaved path issues two strided half-``D`` loads,
    one for the ``a`` lanes and one for the ``b`` lanes.
    """
    pid_b = tl.program_id(0)
    pid_s = tl.program_id(1)
    pid_ht = tl.program_id(2)

    s_off = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    s_mask = s_off < S

    d_idx = tl.arange(0, BLOCK_D)
    d_mask = d_idx < D_HALF

    # Freqs are ``(BLOCK_S, D_HALF)`` and shared across heads; load
    # once, do cos / sin in fp32, then cast down before the multiply.
    freq_off = s_off[:, None] * stride_fs + d_idx[None, :] * stride_fd
    freq_mask = s_mask[:, None] & d_mask[None, :]
    freq = tl.load(freqs_ptr + freq_off, mask=freq_mask, other=0.0).to(tl.float32)
    cos_f = tl.cos(freq)
    sin_f = tl.sin(freq)

    h_offset = pid_ht * BLOCK_H
    h_idx = h_offset + tl.arange(0, BLOCK_H)
    h_mask = h_idx < H
    row_base = x_ptr + pid_b * stride_xb

    if INTERLEAVED:
        # One contiguous full-``D`` load, then deinterleave in
        # registers via ``reshape`` + ``tl.split``; rebuild the
        # interleaved layout on store via ``tl.join`` + ``reshape``.
        d_full = tl.arange(0, 2 * BLOCK_D)
        full_mask_d = d_full < (2 * D_HALF)
        offsets = (
            s_off[:, None, None] * stride_xs
            + h_idx[None, :, None] * stride_xh
            + d_full[None, None, :] * stride_xd
        )
        full_mask = (
            s_mask[:, None, None] & h_mask[None, :, None] & full_mask_d[None, None, :]
        )
        x_flat = tl.load(row_base + offsets, mask=full_mask, other=0.0)
        x_pairs = tl.reshape(x_flat, (BLOCK_S, BLOCK_H, BLOCK_D, 2))
        a, b = tl.split(x_pairs)

        cos_t = cos_f[:, None, :].to(a.dtype)
        sin_t = sin_f[:, None, :].to(a.dtype)
        out_a = a * cos_t - b * sin_t
        out_b = b * cos_t + a * sin_t

        out_pairs = tl.join(out_a, out_b)
        out_flat = tl.reshape(out_pairs, (BLOCK_S, BLOCK_H, 2 * BLOCK_D))
        tl.store(row_base + offsets, out_flat, mask=full_mask)
    else:
        a_off = (
            s_off[:, None, None] * stride_xs
            + h_idx[None, :, None] * stride_xh
            + d_idx[None, None, :] * stride_xd
        )
        b_off = a_off + D_HALF * stride_xd
        mask = s_mask[:, None, None] & h_mask[None, :, None] & d_mask[None, None, :]

        a = tl.load(row_base + a_off, mask=mask, other=0.0)
        b = tl.load(row_base + b_off, mask=mask, other=0.0)

        cos_t = cos_f[:, None, :].to(a.dtype)
        sin_t = sin_f[:, None, :].to(a.dtype)
        out_a = a * cos_t - b * sin_t
        out_b = b * cos_t + a * sin_t

        tl.store(row_base + a_off, out_a, mask=mask)
        tl.store(row_base + b_off, out_b, mask=mask)


def _pick_block_s(s: int) -> int:
    """Return the per-program ``BLOCK_S`` tile size for sequence length ``s``.

    Falls back to ``BLOCK_S=1`` for short ``s`` so the grid still
    saturates the GPU's SMs, and uses ``BLOCK_S=4`` once ``s`` is
    large enough that fatter tiles amortise dispatch latency.
    """
    return 4 if s >= 512 else 1


def apply_rotary_pos_emb(
    x: Tensor,
    freqs: Tensor,
    interleaved: bool = False,
    inplace: bool = True,
    block_s: int | None = None,
    num_warps: int | None = None,
    num_stages: int | None = None,
) -> Tensor:
    """Apply 3D RoPE to ``x`` using a fused inference-only Triton kernel.

    Args:
        x: Activations of shape ``[B, S, H, D]``. ``D`` must be even.
            ``fp32`` / ``fp16`` / ``bf16`` supported; raw ``fp8`` is
            rejected, matching TE's policy.
        freqs: RoPE frequencies of shape ``[S, 1, 1, D]`` — the
            full-width layout emitted by
            :meth:`RotaryPositionEmbedding3D.shift_t`. ``fp32`` /
            ``fp16`` / ``bf16`` supported; promoted to fp32 inside
            the kernel for the ``cos`` / ``sin`` pass.
        interleaved: If ``True``, rotate the pair ``(2k, 2k+1)``; else
            rotate ``(d, d + D/2)``.
        inplace: If ``True`` (default) write back into ``x`` storage and
            return ``x``; if ``False`` clone first.
        block_s: Override ``BLOCK_S``. ``None`` picks via
            :func:`_pick_block_s`; exposed for benchmarking.
        num_warps: Override Triton's ``num_warps``. ``None`` picks
            based on per-program element count.
        num_stages: Override Triton's ``num_stages``. ``None`` uses 2.

    Returns:
        Rotated tensor; shares storage with ``x`` when ``inplace`` is ``True``.

    Raises:
        NotImplementedError: ``x.dtype`` is one of the fp8 dtypes.
        ValueError: ``x`` / ``freqs`` rank, ``D`` parity, or shape
            consistency invariants are violated, or ``x`` is not on CUDA.
    """
    if x.dim() != 4:
        raise ValueError(f"x must be 4D [B, S, H, D]; got {tuple(x.shape)}.")
    if freqs.dim() != 4:
        raise ValueError(f"freqs must be 4D [S, 1, 1, D]; got {tuple(freqs.shape)}.")
    if not x.is_cuda:
        raise ValueError("apply_rotary_pos_emb requires a CUDA tensor.")
    if x.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        raise NotImplementedError(
            f"apply_rotary_pos_emb does not support fp8 inputs (got {x.dtype}); "
            "dequantise to bf16/fp16 first, matching TE's policy."
        )
    B, S, H, D = x.shape
    if D % 2 != 0:
        raise ValueError(f"head_dim must be even; got {D}.")
    head_dim_half = D // 2
    if freqs.shape[0] != S:
        raise ValueError(f"freqs[0]={freqs.shape[0]} must match x.shape[1]={S}.")

    out = x if inplace else x.clone()

    if freqs.shape[-1] != D:
        raise ValueError(
            f"freqs last-dim must equal head_dim ({D}); got "
            f"{freqs.shape[-1]}. Pass the full-width [S, 1, 1, head_dim] "
            "layout that RotaryPositionEmbedding3D.shift_t emits."
        )
    # Skip the redundant half of the full-width ``shift_t`` layout
    # via plain strides on the original storage; avoids a Python-side
    # ``reshape`` + slice (and a possible silent ``contiguous()`` copy
    # if ``freqs`` is ever non-contig).
    stride_fs = freqs.stride(0)
    stride_fd = freqs.stride(3) * (2 if interleaved else 1)

    if B == 0 or S == 0 or H == 0:
        return out

    # Triton tile sizes must be powers of two. Cap ``BLOCK_H`` at 32
    # so the head axis collapses to a single tile for the H = 16 / 24
    # / 32 configs we run in production.
    BLOCK_D = max(int(triton.next_power_of_2(head_dim_half)), 16)
    _MAX_BLOCK_H = 32
    BLOCK_H = min(int(triton.next_power_of_2(H)), _MAX_BLOCK_H)
    H_TILES = triton.cdiv(H, BLOCK_H)

    BLOCK_S = _pick_block_s(S) if block_s is None else int(block_s)
    BLOCK_S = max(int(triton.next_power_of_2(BLOCK_S)), 1)
    S_TILES = triton.cdiv(S, BLOCK_S)

    # Aim for ~16 elements per thread: enough work to amortise issue
    # latency, few enough threads to keep register pressure modest.
    elements_per_program = BLOCK_S * BLOCK_H * BLOCK_D
    if num_warps is None:
        if elements_per_program <= 256:
            num_warps_eff = 1
        elif elements_per_program <= 1024:
            num_warps_eff = 2
        elif elements_per_program <= 2048:
            num_warps_eff = 4
        else:
            num_warps_eff = 8
    else:
        num_warps_eff = int(num_warps)
    num_stages_eff = 2 if num_stages is None else int(num_stages)

    _rope_inference_kernel[(B, S_TILES, H_TILES)](
        out,
        freqs,
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out.stride(3),
        stride_fs,
        stride_fd,
        S,
        H,
        D_HALF=head_dim_half,
        INTERLEAVED=interleaved,
        BLOCK_S=BLOCK_S,
        BLOCK_H=BLOCK_H,
        BLOCK_D=BLOCK_D,
        num_warps=num_warps_eff,  # type: ignore[call-arg]
        num_stages=num_stages_eff,  # type: ignore[call-arg]
    )
    return out


__all__ = ["apply_rotary_pos_emb"]
