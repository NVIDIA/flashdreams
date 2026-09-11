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

"""Triton forward implementation for FlashVSR block-sparse attention."""

from dataclasses import dataclass
from typing import Optional, Tuple, overload

import torch

try:
    import triton
    import triton.language as tl
except ImportError as exc:  # pragma: no cover
    raise RuntimeError(
        "triton is required for FlashVSR sparse attention. Install triton or "
        "run FlashVSR with attention_mode='full'."
    ) from exc

try:
    from flash_attn import flash_attn_varlen_func as _flash_attn_varlen_func
except ImportError:  # pragma: no cover
    _flash_attn_varlen_func = None


BLOCK_DIM = 128
SHORT_SEQ_MAX = 2048
LONG_SEQ_MIN = 8192
VERY_LONG_SINGLE_SEQ_MIN = 32768
RCP_LN2 = tl.constexpr(1.4426950408889634)
LN2 = tl.constexpr(0.6931471805599453)


@dataclass(frozen=True)
class _KernelOptions:
    dense_block_m: int
    mode_block_m: int
    mixed_block_m: int
    blocksparse_block_m: int
    block_n_mode: int
    block_n_stream: int
    block_n_mixed: int
    block_n_sparse: int
    num_warps_dense: int
    num_warps_stream: int
    num_warps_mixed: int
    num_warps_sparse: int
    num_stages_stream: int
    num_stages_sparse: int
    use_row_list_sparse: bool
    use_row_list_mixed: bool
    use_single_mixed: bool
    use_uniform_block_stream: bool


@dataclass(frozen=True)
class _DispatchKey:
    sm: int
    dtype: torch.dtype
    headdim: int
    mode: str
    exact_streaming: bool
    batch_size: int
    max_seqlen_k: int
    seq_bucket: str


@dataclass
class _KernelOptionBuilder:
    dense_block_m: int
    mode_block_m: int
    mixed_block_m: int
    blocksparse_block_m: int
    block_n_mode: int
    block_n_stream: int
    block_n_mixed: int
    block_n_sparse: int
    num_warps_dense: int
    num_warps_stream: int
    num_warps_mixed: int
    num_warps_sparse: int
    num_stages_stream: int
    num_stages_sparse: int

    def build(
        self,
        *,
        use_row_list_sparse: bool,
        use_row_list_mixed: bool,
        use_single_mixed: bool,
        use_uniform_block_stream: bool,
    ) -> _KernelOptions:
        return _KernelOptions(
            dense_block_m=self.dense_block_m,
            mode_block_m=self.mode_block_m,
            mixed_block_m=self.mixed_block_m,
            blocksparse_block_m=self.blocksparse_block_m,
            block_n_mode=self.block_n_mode,
            block_n_stream=self.block_n_stream,
            block_n_mixed=self.block_n_mixed,
            block_n_sparse=self.block_n_sparse,
            num_warps_dense=self.num_warps_dense,
            num_warps_stream=self.num_warps_stream,
            num_warps_mixed=self.num_warps_mixed,
            num_warps_sparse=self.num_warps_sparse,
            num_stages_stream=self.num_stages_stream,
            num_stages_sparse=self.num_stages_sparse,
            use_row_list_sparse=use_row_list_sparse,
            use_row_list_mixed=use_row_list_mixed,
            use_single_mixed=use_single_mixed,
            use_uniform_block_stream=use_uniform_block_stream,
        )


def _device_sm(device: torch.device) -> int:
    if not torch.cuda.is_available() or device.type != "cuda":
        return 0
    major, minor = torch.cuda.get_device_capability(device)
    return major * 10 + minor


def _seq_bucket(batch_size: int, max_seqlen_k: int) -> str:
    if max_seqlen_k <= SHORT_SEQ_MAX:
        return "short"
    if max_seqlen_k >= VERY_LONG_SINGLE_SEQ_MIN and batch_size == 1:
        return "very_long_single"
    if max_seqlen_k >= LONG_SEQ_MIN and batch_size > 1:
        return "long_batch"
    if max_seqlen_k >= LONG_SEQ_MIN:
        return "long_single"
    return "medium"


def _make_dispatch_key(
    q: torch.Tensor,
    batch_size: int,
    max_seqlen_k: int,
    exact_streaming: bool,
    mode: str,
) -> _DispatchKey:
    return _DispatchKey(
        sm=_device_sm(q.device),
        dtype=q.dtype,
        headdim=q.shape[-1],
        mode=mode,
        exact_streaming=exact_streaming,
        batch_size=batch_size,
        max_seqlen_k=max_seqlen_k,
        seq_bucket=_seq_bucket(batch_size, max_seqlen_k),
    )


def _base_kernel_options(key: _DispatchKey) -> _KernelOptionBuilder:
    headdim = key.headdim
    long_k = key.max_seqlen_k >= LONG_SEQ_MIN
    long_batch = key.seq_bucket == "long_batch"

    block_n_mode = 64
    block_n_stream = 64
    block_n_mixed = 64
    block_n_sparse = 64
    if headdim == 128:
        block_n_mode = 32
        block_n_stream = 64
        block_n_mixed = 32
        block_n_sparse = 64
    elif long_k:
        block_n_stream = 128
        block_n_mixed = 128
        block_n_sparse = 128

    num_warps_mode = 4 if headdim <= 64 else 8
    return _KernelOptionBuilder(
        dense_block_m=128,
        mode_block_m=64 if headdim <= 64 else 128,
        mixed_block_m=128,
        blocksparse_block_m=64 if long_batch and headdim <= 64 else 128,
        block_n_mode=block_n_mode,
        block_n_stream=block_n_stream,
        block_n_mixed=block_n_mixed,
        block_n_sparse=block_n_sparse,
        num_warps_dense=8,
        num_warps_stream=num_warps_mode,
        num_warps_mixed=8 if block_n_mixed >= 128 else num_warps_mode,
        num_warps_sparse=num_warps_mode,
        num_stages_stream=3,
        num_stages_sparse=3,
    )


def _apply_long_batch_profile(opts: _KernelOptionBuilder, key: _DispatchKey) -> None:
    headdim = key.headdim
    if headdim <= 64:
        opts.block_n_stream = 16 if headdim == 32 else 64
        opts.block_n_mixed = 64
        opts.blocksparse_block_m = 128
        if headdim == 32:
            if not key.exact_streaming:
                opts.mode_block_m = 128
                opts.block_n_stream = 128
            opts.block_n_sparse = 128
        else:
            if not key.exact_streaming:
                opts.mode_block_m = 128
            opts.block_n_sparse = 64
    elif headdim == 128:
        if not key.exact_streaming:
            opts.mode_block_m = 128
            opts.block_n_stream = 64
        opts.block_n_mixed = 64
        opts.blocksparse_block_m = 128
        opts.block_n_sparse = 64

    opts.num_warps_mixed = 4
    if headdim == 128:
        if not key.exact_streaming:
            opts.num_warps_stream = 4
        opts.num_warps_sparse = 4


def _apply_very_long_single_profile(
    opts: _KernelOptionBuilder, key: _DispatchKey
) -> None:
    headdim = key.headdim
    if headdim <= 64:
        if headdim == 32:
            if not key.exact_streaming:
                opts.mode_block_m = 128
            else:
                opts.block_n_stream = 64
    elif headdim == 128:
        if not key.exact_streaming:
            opts.block_n_stream = 128
        opts.block_n_sparse = 64
        opts.num_warps_sparse = 4

    if not key.exact_streaming and headdim >= 64:
        opts.num_stages_stream = 2


def _select_kernel_options(
    key: _DispatchKey, has_base_blockmask: bool
) -> _KernelOptions:
    headdim = key.headdim
    opts = _base_kernel_options(key)

    if key.seq_bucket == "long_batch":
        _apply_long_batch_profile(opts, key)
    elif key.seq_bucket == "very_long_single":
        _apply_very_long_single_profile(opts, key)

    if key.sm != 103:
        # Keep the SM103 choices as a conservative default until per-arch profiles are added.
        pass

    use_bool_blockmask = key.seq_bucket in (
        "long_batch",
        "very_long_single",
    ) and headdim in (32, 64, 128)
    if (
        key.mode == "blocksparse"
        and has_base_blockmask
        and key.batch_size == 1
        and key.seq_bucket == "long_single"
        and headdim == 128
    ):
        opts.num_warps_sparse = 4
        use_bool_blockmask = True
    use_row_list_sparse = not use_bool_blockmask
    use_row_list_mixed = not use_bool_blockmask
    use_single_mixed = (
        key.mode == "auto"
        and has_base_blockmask
        and (
            key.seq_bucket in ("short", "long_batch")
            or (key.batch_size <= 1 and key.seq_bucket != "very_long_single")
        )
    )
    use_uniform_block_stream = (
        key.mode == "streaming"
        and not key.exact_streaming
        and key.seq_bucket == "very_long_single"
    )

    return opts.build(
        use_row_list_sparse=use_row_list_sparse,
        use_row_list_mixed=use_row_list_mixed,
        use_single_mixed=use_single_mixed,
        use_uniform_block_stream=use_uniform_block_stream,
    )


def _dispatch_kernel_options(
    q: torch.Tensor,
    batch_size: int,
    max_seqlen_k: int,
    exact_streaming: bool,
    mode: str,
    has_base_blockmask: bool,
) -> Tuple[_DispatchKey, _KernelOptions]:
    key = _make_dispatch_key(
        q=q,
        batch_size=batch_size,
        max_seqlen_k=max_seqlen_k,
        exact_streaming=exact_streaming,
        mode=mode,
    )
    return key, _select_kernel_options(key, has_base_blockmask=has_base_blockmask)


def _round_multiple(x: int, m: int) -> int:
    return (x + m - 1) // m * m


@overload
def _maybe_contiguous(x: torch.Tensor) -> torch.Tensor: ...


@overload
def _maybe_contiguous(x: None) -> None: ...


def _maybe_contiguous(x: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    return x.contiguous() if x is not None and x.stride(-1) != 1 else x


def _replace_ones_with_count(tensor: torch.Tensor) -> Tuple[torch.Tensor, int]:
    ones_mask = tensor == 1
    running_sparse_idx = torch.cumsum(ones_mask.to(torch.int32), dim=-1).to(
        tensor.dtype
    )
    remapped = torch.where(ones_mask, running_sparse_idx, tensor)
    return remapped, 0


def _convert_blockmask_row_reverse(blockmask: torch.Tensor) -> torch.Tensor:
    blockmask = blockmask.to(dtype=torch.uint8)
    nonzero_val, nonzero_colidx = blockmask.sort(dim=-1, stable=True, descending=True)
    nonzero_idx = nonzero_colidx.to(torch.int32)
    nonzero_idx[nonzero_val == 0] = -1
    return nonzero_idx.contiguous()


def _validate_forward_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    head_mask_type: torch.Tensor,
    streaming_info: Optional[torch.Tensor],
    base_blockmask: Optional[torch.Tensor],
    max_seqlen_q_: int,
    max_seqlen_k_: int,
    exact_streaming: bool,
) -> None:
    if q.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError("Triton fwd supports fp16/bf16 only")
    if k.dtype != q.dtype or v.dtype != q.dtype:
        raise TypeError("q/k/v must share dtype")
    if not (q.is_cuda and k.is_cuda and v.is_cuda):
        raise ValueError("q/k/v must be CUDA tensors")
    if q.ndim != 3 or k.ndim != 3 or v.ndim != 3:
        raise ValueError("q/k/v must be rank-3 varlen tensors")
    if q.shape[2] not in (32, 64, 128):
        raise ValueError("head_dim must be one of {32, 64, 128}")
    if cu_seqlens_q.dtype != torch.int32 or cu_seqlens_k.dtype != torch.int32:
        raise TypeError("cu_seqlens_q/cu_seqlens_k must be int32")
    if head_mask_type.dtype != torch.int32:
        raise TypeError("head_mask_type must be int32")
    if streaming_info is not None and streaming_info.dtype != torch.int32:
        raise TypeError("streaming_info must be int32")
    if cu_seqlens_q.numel() < 2 or cu_seqlens_k.numel() < 2:
        raise ValueError("batch size must be > 0")
    if q.shape[1] % k.shape[1] != 0:
        raise ValueError("nheads must be divisible by nheads_k (GQA/MQA)")
    if head_mask_type.numel() != q.shape[1]:
        raise ValueError("head_mask_type length must equal nheads")
    if streaming_info is not None and streaming_info.numel() != q.shape[1] * 2:
        raise ValueError("streaming_info must have length 2 * nheads")
    if max_seqlen_q_ <= 0 or max_seqlen_k_ < 0:
        raise ValueError("max seqlens must be positive")
    if exact_streaming and base_blockmask is not None:
        pass


@triton.jit
def _fwd_varlen_mixed_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    out_ptr,
    lse_ptr,
    cu_q_ptr,
    cu_k_ptr,
    head_mask_ptr,
    streaming_ptr,
    base_blockmask_ptr,
    stride_qt,
    stride_qh,
    stride_qd,
    stride_kt,
    stride_kh,
    stride_kd,
    stride_vt,
    stride_vh,
    stride_vd,
    stride_ot,
    stride_oh,
    stride_od,
    stride_lseb,
    stride_lseh,
    stride_lsem,
    stride_bmb,
    stride_bmh,
    stride_bmr,
    stride_bmc,
    nheads,
    gqa_group_size,
    nrow_max,
    ncol_max,
    max_seqlen_k,
    softmax_scale,
    IS_CAUSAL: tl.constexpr,
    EXACT_STREAMING: tl.constexpr,
    HAS_BASE_BLOCKMASK: tl.constexpr,
    USE_ROW_LIST: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_DIM_CONST: tl.constexpr,
    NCOL_MAX: tl.constexpr,
):
    pid_h = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_b = tl.program_id(2)
    if pid_h >= nheads:
        return

    q_start = tl.load(cu_q_ptr + pid_b).to(tl.int32)
    q_end = tl.load(cu_q_ptr + pid_b + 1).to(tl.int32)
    k_start = tl.load(cu_k_ptr + pid_b).to(tl.int32)
    k_end = tl.load(cu_k_ptr + pid_b + 1).to(tl.int32)
    q_len = q_end - q_start
    k_len = k_end - k_start

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < q_len

    hk = pid_h // gqa_group_size
    offs_d = tl.arange(0, BLOCK_D)
    q_ptrs = (
        q_ptr
        + (q_start + offs_m[:, None]) * stride_qt
        + pid_h * stride_qh
        + offs_d[None, :] * stride_qd
    )
    q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)
    tok_r = offs_m[:, None]
    block_r = (pid_m * BLOCK_M) // BLOCK_DIM_CONST

    m_i = tl.full([BLOCK_M], -float("inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

    head_type = tl.load(head_mask_ptr + pid_h).to(tl.int32)
    k_align_offset = k_len - q_len
    n_loop_end = k_len
    if IS_CAUSAL:
        max_tok_c = pid_m * BLOCK_M + (BLOCK_M - 1) + k_align_offset + 1
        n_loop_end = tl.minimum(k_len, tl.maximum(max_tok_c, 0))

    if head_type < 0:
        sink = tl.load(streaming_ptr + pid_h * 2).to(tl.int32)
        local = tl.load(streaming_ptr + pid_h * 2 + 1).to(tl.int32)
        ncol = (k_len + BLOCK_DIM_CONST - 1) // BLOCK_DIM_CONST
        start_row_idx = 0
        if IS_CAUSAL:
            start_row_idx = tl.maximum((q_len - k_len) // BLOCK_DIM_CONST, 0)
        causal_shift_blocks = tl.maximum(
            (k_len - q_len + BLOCK_DIM_CONST - 1) // BLOCK_DIM_CONST, 0
        )
        start_col = 0
        end_col = ncol
        if not EXACT_STREAMING:
            max_row_block_num = ncol
            if IS_CAUSAL:
                max_row_block_num = causal_shift_blocks + 1 + block_r - start_row_idx
            max_row_block_num = tl.maximum(max_row_block_num, 0)
            start_col = tl.minimum(tl.maximum(max_row_block_num - local, 0), ncol)
            end_col = tl.minimum(max_row_block_num, ncol)

        sink_end_tok = n_loop_end
        local_start_tok = n_loop_end
        local_end_tok = n_loop_end
        if EXACT_STREAMING:
            sink_end_tok = tl.minimum(n_loop_end, tl.maximum(sink, 0))
            tok_r_min = pid_m * BLOCK_M
            tok_r_max = pid_m * BLOCK_M + (BLOCK_M - 1)
            local_start_tok = tok_r_min + k_align_offset - (local - 1)
            local_end_tok = tok_r_max + k_align_offset + 1
            local_start_tok = tl.maximum(local_start_tok, 0)
            local_end_tok = tl.minimum(local_end_tok, n_loop_end)
        else:
            sink_end_tok = tl.minimum(n_loop_end, tl.maximum(sink, 0) * BLOCK_DIM_CONST)
            local_start_tok = tl.maximum(start_col * BLOCK_DIM_CONST, 0)
            local_end_tok = tl.minimum(end_col * BLOCK_DIM_CONST, n_loop_end)
        local_start_tok = tl.minimum(
            tl.maximum(local_start_tok, sink_end_tok), n_loop_end
        )
        local_end_tok = tl.maximum(local_end_tok, local_start_tok)

        for seg_id in range(0, 2):
            seg_start = tl.where(seg_id == 0, 0, local_start_tok)
            seg_end = tl.where(seg_id == 0, sink_end_tok, local_end_tok)
            seg_len = tl.maximum(seg_end - seg_start, 0)
            for seg_off in tl.range(0, seg_len, BLOCK_N, loop_unroll_factor=1):
                n_start = seg_start + seg_off
                offs_n = n_start + tl.arange(0, BLOCK_N)
                mask_n = offs_n < k_len

                k_ptrs = (
                    k_ptr
                    + (k_start + offs_n[:, None]) * stride_kt
                    + hk * stride_kh
                    + offs_d[None, :] * stride_kd
                )
                v_ptrs = (
                    v_ptr
                    + (k_start + offs_n[:, None]) * stride_vt
                    + hk * stride_vh
                    + offs_d[None, :] * stride_vd
                )
                k = tl.load(
                    k_ptrs,
                    mask=mask_n[:, None],
                    other=0.0,
                    eviction_policy="evict_last",
                )
                v = tl.load(
                    v_ptrs,
                    mask=mask_n[:, None],
                    other=0.0,
                    eviction_policy="evict_last",
                )

                qk = (tl.dot(q, tl.trans(k)) * (softmax_scale * RCP_LN2)).to(tl.float32)
                tok_c = offs_n[None, :]
                causal_keep = tok_c <= (tok_r + k_align_offset)

                full_tile = (
                    (not EXACT_STREAMING)
                    & (n_start + BLOCK_N <= seg_end)
                    & (pid_m * BLOCK_M + BLOCK_M <= q_len)
                )
                if IS_CAUSAL:
                    full_tile = full_tile & (
                        n_start + BLOCK_N - 1 <= pid_m * BLOCK_M + k_align_offset
                    )
                if full_tile:
                    m_ij = tl.max(qk, axis=1)
                    m_new = tl.maximum(m_i, m_ij)
                    alpha = tl.exp2(m_i - m_new)
                    p = tl.exp2(qk - m_new[:, None])
                    l_i = l_i * alpha + tl.sum(p, axis=1)
                    acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
                    m_i = m_new
                else:
                    if EXACT_STREAMING:
                        keep = (tok_c < sink) | (
                            tok_c >= (tok_r + k_align_offset - (local - 1))
                        )
                        keep = keep & (tok_c <= (tok_r + k_align_offset))
                    else:
                        block_c = n_start // BLOCK_DIM_CONST
                        block_keep = (block_c >= start_col) & (block_c < end_col)
                        block_keep = block_keep | (block_c < sink)
                        keep = tl.full([BLOCK_M, BLOCK_N], block_keep, dtype=tl.int1)
                        if IS_CAUSAL:
                            keep = keep & causal_keep

                    keep = keep & mask_m[:, None] & mask_n[None, :]
                    qk = tl.where(keep, qk, -float("inf"))

                    m_ij = tl.max(qk, axis=1)
                    m_new = tl.maximum(m_i, m_ij)
                    m_new_inf = m_new == -float("inf")
                    alpha = tl.exp2(tl.where(m_new_inf, 0.0, m_i - m_new))
                    p = tl.where(
                        keep,
                        tl.exp2(qk - tl.where(m_new_inf[:, None], 0.0, m_new[:, None])),
                        0.0,
                    )
                    l_i = l_i * alpha + tl.sum(p, axis=1)
                    acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
                    m_i = m_new
    else:
        if head_type == 0:
            for n_start in range(0, n_loop_end, BLOCK_N):
                offs_n = n_start + tl.arange(0, BLOCK_N)
                mask_n = offs_n < k_len

                k_ptrs = (
                    k_ptr
                    + (k_start + offs_n[:, None]) * stride_kt
                    + hk * stride_kh
                    + offs_d[None, :] * stride_kd
                )
                v_ptrs = (
                    v_ptr
                    + (k_start + offs_n[:, None]) * stride_vt
                    + hk * stride_vh
                    + offs_d[None, :] * stride_vd
                )
                k = tl.load(
                    k_ptrs,
                    mask=mask_n[:, None],
                    other=0.0,
                    eviction_policy="evict_last",
                )
                v = tl.load(
                    v_ptrs,
                    mask=mask_n[:, None],
                    other=0.0,
                    eviction_policy="evict_last",
                )

                qk = (tl.dot(q, tl.trans(k)) * (softmax_scale * RCP_LN2)).to(tl.float32)
                full_tile = (n_start + BLOCK_N <= k_len) & (
                    pid_m * BLOCK_M + BLOCK_M <= q_len
                )
                if IS_CAUSAL:
                    full_tile = full_tile & (
                        n_start + BLOCK_N - 1 <= pid_m * BLOCK_M + k_align_offset
                    )
                if full_tile:
                    m_ij = tl.max(qk, axis=1)
                    m_new = tl.maximum(m_i, m_ij)
                    alpha = tl.exp2(m_i - m_new)
                    p = tl.exp2(qk - m_new[:, None])
                    l_i = l_i * alpha + tl.sum(p, axis=1)
                    acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
                    m_i = m_new
                else:
                    keep = mask_n[None, :]
                    if IS_CAUSAL:
                        tok_c = offs_n[None, :]
                        keep = keep & (tok_c <= (tok_r + k_align_offset))
                    keep = keep & mask_m[:, None]
                    qk = tl.where(keep, qk, -float("inf"))

                    m_ij = tl.max(qk, axis=1)
                    m_new = tl.maximum(m_i, m_ij)
                    m_new_inf = m_new == -float("inf")
                    alpha = tl.exp2(tl.where(m_new_inf, 0.0, m_i - m_new))
                    p = tl.where(
                        keep,
                        tl.exp2(qk - tl.where(m_new_inf[:, None], 0.0, m_new[:, None])),
                        0.0,
                    )
                    l_i = l_i * alpha + tl.sum(p, axis=1)
                    acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
                    m_i = m_new
        else:
            sparse_head_idx = head_type - 1
            if USE_ROW_LIST:
                for nz_idx in tl.range(0, NCOL_MAX, 1, loop_unroll_factor=1):
                    block_c = tl.load(
                        base_blockmask_ptr
                        + pid_b * stride_bmb
                        + sparse_head_idx * stride_bmh
                        + block_r * stride_bmr
                        + nz_idx * stride_bmc,
                        mask=block_r < nrow_max,
                        other=-1,
                    ).to(tl.int32)
                    valid_block = block_c >= 0
                    if valid_block:
                        n_start = block_c * BLOCK_DIM_CONST
                        if n_start < n_loop_end:
                            for block_off in range(0, BLOCK_DIM_CONST, BLOCK_N):
                                n_block_start = n_start + block_off
                                offs_n = n_block_start + tl.arange(0, BLOCK_N)
                                mask_n = offs_n < k_len

                                k_ptrs = (
                                    k_ptr
                                    + (k_start + offs_n[:, None]) * stride_kt
                                    + hk * stride_kh
                                    + offs_d[None, :] * stride_kd
                                )
                                v_ptrs = (
                                    v_ptr
                                    + (k_start + offs_n[:, None]) * stride_vt
                                    + hk * stride_vh
                                    + offs_d[None, :] * stride_vd
                                )
                                k = tl.load(
                                    k_ptrs,
                                    mask=mask_n[:, None],
                                    other=0.0,
                                    eviction_policy="evict_last",
                                )
                                v = tl.load(
                                    v_ptrs,
                                    mask=mask_n[:, None],
                                    other=0.0,
                                    eviction_policy="evict_last",
                                )

                                qk = (
                                    tl.dot(q, tl.trans(k)) * (softmax_scale * RCP_LN2)
                                ).to(tl.float32)
                                keep = mask_m[:, None] & mask_n[None, :]
                                if IS_CAUSAL:
                                    tok_c = offs_n[None, :]
                                    keep = keep & (tok_c <= (tok_r + k_align_offset))
                                qk = tl.where(keep, qk, -float("inf"))

                                m_ij = tl.max(qk, axis=1)
                                m_new = tl.maximum(m_i, m_ij)
                                m_new_inf = m_new == -float("inf")
                                alpha = tl.exp2(tl.where(m_new_inf, 0.0, m_i - m_new))
                                p = tl.where(
                                    keep,
                                    tl.exp2(
                                        qk
                                        - tl.where(
                                            m_new_inf[:, None], 0.0, m_new[:, None]
                                        )
                                    ),
                                    0.0,
                                )
                                l_i = l_i * alpha + tl.sum(p, axis=1)
                                acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
                                m_i = m_new
            else:
                for n_start in range(0, n_loop_end, BLOCK_N):
                    block_c = n_start // BLOCK_DIM_CONST
                    block_keep = tl.full((), False, dtype=tl.int1)
                    if HAS_BASE_BLOCKMASK:
                        bm_ptrs = (
                            base_blockmask_ptr
                            + pid_b * stride_bmb
                            + sparse_head_idx * stride_bmh
                            + block_r * stride_bmr
                            + block_c * stride_bmc
                        )
                        bm_mask = (block_r < nrow_max) & (block_c < ncol_max)
                        block_keep = tl.load(bm_ptrs, mask=bm_mask, other=0).to(tl.int1)
                    if block_keep:
                        offs_n = n_start + tl.arange(0, BLOCK_N)
                        mask_n = offs_n < k_len

                        k_ptrs = (
                            k_ptr
                            + (k_start + offs_n[:, None]) * stride_kt
                            + hk * stride_kh
                            + offs_d[None, :] * stride_kd
                        )
                        v_ptrs = (
                            v_ptr
                            + (k_start + offs_n[:, None]) * stride_vt
                            + hk * stride_vh
                            + offs_d[None, :] * stride_vd
                        )
                        k = tl.load(
                            k_ptrs,
                            mask=mask_n[:, None],
                            other=0.0,
                            eviction_policy="evict_last",
                        )
                        v = tl.load(
                            v_ptrs,
                            mask=mask_n[:, None],
                            other=0.0,
                            eviction_policy="evict_last",
                        )

                        qk = (tl.dot(q, tl.trans(k)) * (softmax_scale * RCP_LN2)).to(
                            tl.float32
                        )
                        full_tile = (n_start + BLOCK_N <= k_len) & (
                            pid_m * BLOCK_M + BLOCK_M <= q_len
                        )
                        if IS_CAUSAL:
                            full_tile = full_tile & (
                                n_start + BLOCK_N - 1
                                <= pid_m * BLOCK_M + k_align_offset
                            )
                        if full_tile:
                            m_ij = tl.max(qk, axis=1)
                            m_new = tl.maximum(m_i, m_ij)
                            alpha = tl.exp2(m_i - m_new)
                            p = tl.exp2(qk - m_new[:, None])
                            l_i = l_i * alpha + tl.sum(p, axis=1)
                            acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
                            m_i = m_new
                        else:
                            keep = mask_m[:, None] & mask_n[None, :]
                            if IS_CAUSAL:
                                tok_c = offs_n[None, :]
                                keep = keep & (tok_c <= (tok_r + k_align_offset))
                            qk = tl.where(keep, qk, -float("inf"))

                            m_ij = tl.max(qk, axis=1)
                            m_new = tl.maximum(m_i, m_ij)
                            m_new_inf = m_new == -float("inf")
                            alpha = tl.exp2(tl.where(m_new_inf, 0.0, m_i - m_new))
                            p = tl.where(
                                keep,
                                tl.exp2(
                                    qk
                                    - tl.where(m_new_inf[:, None], 0.0, m_new[:, None])
                                ),
                                0.0,
                            )
                            l_i = l_i * alpha + tl.sum(p, axis=1)
                            acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
                            m_i = m_new

    row_has_any = l_i > 0.0
    out_vals = tl.where(row_has_any[:, None], acc / l_i[:, None], 0.0)
    out_ptrs = (
        out_ptr
        + (q_start + offs_m[:, None]) * stride_ot
        + pid_h * stride_oh
        + offs_d[None, :] * stride_od
    )
    tl.store(out_ptrs, out_vals.to(q.dtype), mask=mask_m[:, None])

    lse_vals = tl.where(row_has_any, m_i * LN2 + tl.log(l_i), float("inf"))
    lse_ptrs = (
        lse_ptr + pid_b * stride_lseb + pid_h * stride_lseh + offs_m * stride_lsem
    )
    tl.store(lse_ptrs, lse_vals, mask=mask_m)


@triton.jit
def _fwd_varlen_dense_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    out_ptr,
    lse_ptr,
    cu_q_ptr,
    cu_k_ptr,
    head_index_ptr,
    stride_qt,
    stride_qh,
    stride_qd,
    stride_kt,
    stride_kh,
    stride_kd,
    stride_vt,
    stride_vh,
    stride_vd,
    stride_ot,
    stride_oh,
    stride_od,
    stride_lseb,
    stride_lseh,
    stride_lsem,
    gqa_group_size,
    max_seqlen_k,
    softmax_scale,
    IS_CAUSAL: tl.constexpr,
    USE_HEAD_INDEX: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_h = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_b = tl.program_id(2)
    h_idx = tl.load(head_index_ptr + pid_h).to(tl.int32) if USE_HEAD_INDEX else pid_h

    q_start = tl.load(cu_q_ptr + pid_b).to(tl.int32)
    q_end = tl.load(cu_q_ptr + pid_b + 1).to(tl.int32)
    k_start = tl.load(cu_k_ptr + pid_b).to(tl.int32)
    k_end = tl.load(cu_k_ptr + pid_b + 1).to(tl.int32)
    q_len = q_end - q_start
    k_len = k_end - k_start

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < q_len
    hk = h_idx // gqa_group_size

    offs_d = tl.arange(0, BLOCK_D)
    q_ptrs = (
        q_ptr
        + (q_start + offs_m[:, None]) * stride_qt
        + h_idx * stride_qh
        + offs_d[None, :] * stride_qd
    )
    q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)
    tok_r = offs_m[:, None]

    m_i = tl.full([BLOCK_M], -float("inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
    k_align_offset = k_len - q_len

    n_loop_end = k_len
    if IS_CAUSAL:
        max_tok_c = pid_m * BLOCK_M + (BLOCK_M - 1) + k_align_offset + 1
        n_loop_end = tl.minimum(k_len, tl.maximum(max_tok_c, 0))
    for n_start in range(0, n_loop_end, BLOCK_N):
        offs_n = n_start + tl.arange(0, BLOCK_N)
        mask_n = offs_n < k_len

        k_ptrs = (
            k_ptr
            + (k_start + offs_n[:, None]) * stride_kt
            + hk * stride_kh
            + offs_d[None, :] * stride_kd
        )
        v_ptrs = (
            v_ptr
            + (k_start + offs_n[:, None]) * stride_vt
            + hk * stride_vh
            + offs_d[None, :] * stride_vd
        )
        k = tl.load(
            k_ptrs, mask=mask_n[:, None], other=0.0, eviction_policy="evict_last"
        )
        v = tl.load(
            v_ptrs, mask=mask_n[:, None], other=0.0, eviction_policy="evict_last"
        )

        qk = (tl.dot(q, tl.trans(k)) * (softmax_scale * RCP_LN2)).to(tl.float32)
        keep = mask_n[None, :]
        if IS_CAUSAL:
            tok_c = offs_n[None, :]
            keep = keep & (tok_c <= (tok_r + k_align_offset))
        keep = keep & mask_m[:, None]

        qk = tl.where(keep, qk, -float("inf"))

        m_ij = tl.max(qk, axis=1)
        m_new = tl.maximum(m_i, m_ij)
        m_new_inf = m_new == -float("inf")
        alpha = tl.exp2(tl.where(m_new_inf, 0.0, m_i - m_new))
        p = tl.where(
            keep, tl.exp2(qk - tl.where(m_new_inf[:, None], 0.0, m_new[:, None])), 0.0
        )
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_new

    row_has_any = l_i > 0.0
    out_vals = tl.where(row_has_any[:, None], acc / l_i[:, None], 0.0)
    out_ptrs = (
        out_ptr
        + (q_start + offs_m[:, None]) * stride_ot
        + h_idx * stride_oh
        + offs_d[None, :] * stride_od
    )
    tl.store(out_ptrs, out_vals.to(q.dtype), mask=mask_m[:, None])

    lse_vals = tl.where(row_has_any, m_i * LN2 + tl.log(l_i), float("inf"))
    lse_ptrs = (
        lse_ptr + pid_b * stride_lseb + h_idx * stride_lseh + offs_m * stride_lsem
    )
    tl.store(lse_ptrs, lse_vals, mask=mask_m)


@triton.jit
def _fwd_varlen_streaming_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    out_ptr,
    lse_ptr,
    cu_q_ptr,
    cu_k_ptr,
    head_index_ptr,
    streaming_ptr,
    stride_qt,
    stride_qh,
    stride_qd,
    stride_kt,
    stride_kh,
    stride_kd,
    stride_vt,
    stride_vh,
    stride_vd,
    stride_ot,
    stride_oh,
    stride_od,
    stride_lseb,
    stride_lseh,
    stride_lsem,
    gqa_group_size,
    max_seqlen_k,
    softmax_scale,
    IS_CAUSAL: tl.constexpr,
    EXACT_STREAMING: tl.constexpr,
    USE_HEAD_INDEX: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_DIM_CONST: tl.constexpr,
):
    pid_h = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_b = tl.program_id(2)
    h_idx = tl.load(head_index_ptr + pid_h).to(tl.int32) if USE_HEAD_INDEX else pid_h

    q_start = tl.load(cu_q_ptr + pid_b).to(tl.int32)
    q_end = tl.load(cu_q_ptr + pid_b + 1).to(tl.int32)
    k_start = tl.load(cu_k_ptr + pid_b).to(tl.int32)
    k_end = tl.load(cu_k_ptr + pid_b + 1).to(tl.int32)
    q_len = q_end - q_start
    k_len = k_end - k_start

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < q_len
    hk = h_idx // gqa_group_size

    offs_d = tl.arange(0, BLOCK_D)
    q_ptrs = (
        q_ptr
        + (q_start + offs_m[:, None]) * stride_qt
        + h_idx * stride_qh
        + offs_d[None, :] * stride_qd
    )
    q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)
    tok_r = offs_m[:, None]
    block_r = (pid_m * BLOCK_M) // BLOCK_DIM_CONST

    m_i = tl.full([BLOCK_M], -float("inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

    sink = tl.load(streaming_ptr + h_idx * 2).to(tl.int32)
    local = tl.load(streaming_ptr + h_idx * 2 + 1).to(tl.int32)
    ncol = (k_len + BLOCK_DIM_CONST - 1) // BLOCK_DIM_CONST
    k_align_offset = k_len - q_len
    start_row_idx = tl.maximum((q_len - k_len) // BLOCK_DIM_CONST, 0)
    causal_shift_blocks = tl.maximum(
        (k_len - q_len + BLOCK_DIM_CONST - 1) // BLOCK_DIM_CONST, 0
    )
    start_col = 0
    end_col = ncol
    if not EXACT_STREAMING:
        max_row_block_num = ncol
        if IS_CAUSAL:
            max_row_block_num = causal_shift_blocks + 1 + block_r - start_row_idx
        max_row_block_num = tl.maximum(max_row_block_num, 0)
        start_col = tl.minimum(tl.maximum(max_row_block_num - local, 0), ncol)
        end_col = tl.minimum(max_row_block_num, ncol)

    n_loop_end = k_len
    if IS_CAUSAL:
        max_tok_c = pid_m * BLOCK_M + (BLOCK_M - 1) + k_align_offset + 1
        n_loop_end = tl.minimum(k_len, tl.maximum(max_tok_c, 0))
    sink_end_tok = n_loop_end
    local_start_tok = n_loop_end
    local_end_tok = n_loop_end
    if EXACT_STREAMING:
        sink_end_tok = tl.minimum(n_loop_end, tl.maximum(sink, 0))
        tok_r_min = pid_m * BLOCK_M
        tok_r_max = pid_m * BLOCK_M + (BLOCK_M - 1)
        local_start_tok = tok_r_min + k_align_offset - (local - 1)
        local_end_tok = tok_r_max + k_align_offset + 1
        local_start_tok = tl.maximum(local_start_tok, 0)
        local_end_tok = tl.minimum(local_end_tok, n_loop_end)
    else:
        sink_end_tok = tl.minimum(n_loop_end, tl.maximum(sink, 0) * BLOCK_DIM_CONST)
        local_start_tok = tl.maximum(start_col * BLOCK_DIM_CONST, 0)
        local_end_tok = tl.minimum(end_col * BLOCK_DIM_CONST, n_loop_end)
    local_start_tok = tl.minimum(tl.maximum(local_start_tok, sink_end_tok), n_loop_end)
    local_end_tok = tl.maximum(local_end_tok, local_start_tok)

    for seg_id in range(0, 2):
        seg_start = tl.where(seg_id == 0, 0, local_start_tok)
        seg_end = tl.where(seg_id == 0, sink_end_tok, local_end_tok)
        seg_len = tl.maximum(seg_end - seg_start, 0)
        for seg_off in tl.range(0, seg_len, BLOCK_N, loop_unroll_factor=1):
            n_start = seg_start + seg_off
            offs_n = n_start + tl.arange(0, BLOCK_N)
            mask_n = offs_n < k_len

            k_ptrs = (
                k_ptr
                + (k_start + offs_n[:, None]) * stride_kt
                + hk * stride_kh
                + offs_d[None, :] * stride_kd
            )
            v_ptrs = (
                v_ptr
                + (k_start + offs_n[:, None]) * stride_vt
                + hk * stride_vh
                + offs_d[None, :] * stride_vd
            )
            k = tl.load(
                k_ptrs, mask=mask_n[:, None], other=0.0, eviction_policy="evict_last"
            )
            v = tl.load(
                v_ptrs, mask=mask_n[:, None], other=0.0, eviction_policy="evict_last"
            )

            qk = (tl.dot(q, tl.trans(k)) * (softmax_scale * RCP_LN2)).to(tl.float32)
            tok_c = offs_n[None, :]
            causal_keep = tok_c <= (tok_r + k_align_offset)

            full_tile = (
                (not EXACT_STREAMING)
                & (n_start + BLOCK_N <= k_len)
                & (pid_m * BLOCK_M + BLOCK_M <= q_len)
            )
            if IS_CAUSAL:
                full_tile = full_tile & (
                    n_start + BLOCK_N - 1 <= pid_m * BLOCK_M + k_align_offset
                )
            if full_tile:
                m_ij = tl.max(qk, axis=1)
                m_new = tl.maximum(m_i, m_ij)
                alpha = tl.exp2(m_i - m_new)
                p = tl.exp2(qk - m_new[:, None])
                l_i = l_i * alpha + tl.sum(p, axis=1)
                acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
                m_i = m_new
            else:
                if EXACT_STREAMING:
                    keep = (tok_c < sink) | (
                        tok_c >= (tok_r + k_align_offset - (local - 1))
                    )
                    keep = keep & (tok_c <= (tok_r + k_align_offset))
                else:
                    block_c = n_start // BLOCK_DIM_CONST
                    block_keep = (block_c >= start_col) & (block_c < end_col)
                    block_keep = block_keep | (block_c < sink)
                    keep = tl.full([BLOCK_M, BLOCK_N], block_keep, dtype=tl.int1)
                    if IS_CAUSAL:
                        keep = keep & causal_keep

                keep = keep & mask_m[:, None] & mask_n[None, :]
                qk = tl.where(keep, qk, -float("inf"))

                m_ij = tl.max(qk, axis=1)
                m_new = tl.maximum(m_i, m_ij)
                m_new_inf = m_new == -float("inf")
                alpha = tl.exp2(tl.where(m_new_inf, 0.0, m_i - m_new))
                p = tl.where(
                    keep,
                    tl.exp2(qk - tl.where(m_new_inf[:, None], 0.0, m_new[:, None])),
                    0.0,
                )
                l_i = l_i * alpha + tl.sum(p, axis=1)
                acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
                m_i = m_new

    row_has_any = l_i > 0.0
    out_vals = tl.where(row_has_any[:, None], acc / l_i[:, None], 0.0)
    out_ptrs = (
        out_ptr
        + (q_start + offs_m[:, None]) * stride_ot
        + h_idx * stride_oh
        + offs_d[None, :] * stride_od
    )
    tl.store(out_ptrs, out_vals.to(q.dtype), mask=mask_m[:, None])

    lse_vals = tl.where(row_has_any, m_i * LN2 + tl.log(l_i), float("inf"))
    lse_ptrs = (
        lse_ptr + pid_b * stride_lseb + h_idx * stride_lseh + offs_m * stride_lsem
    )
    tl.store(lse_ptrs, lse_vals, mask=mask_m)


@triton.jit
def _fwd_varlen_blocksparse_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    out_ptr,
    lse_ptr,
    cu_q_ptr,
    cu_k_ptr,
    head_index_ptr,
    head_mask_ptr,
    row_blockmask_ptr,
    stride_qt,
    stride_qh,
    stride_qd,
    stride_kt,
    stride_kh,
    stride_kd,
    stride_vt,
    stride_vh,
    stride_vd,
    stride_ot,
    stride_oh,
    stride_od,
    stride_lseb,
    stride_lseh,
    stride_lsem,
    stride_bmb,
    stride_bmh,
    stride_bmr,
    stride_bmc,
    gqa_group_size,
    nrow_max,
    ncol_max,
    max_seqlen_k,
    softmax_scale,
    IS_CAUSAL: tl.constexpr,
    USE_HEAD_INDEX: tl.constexpr,
    USE_ROW_LIST: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_DIM_CONST: tl.constexpr,
    NCOL_MAX: tl.constexpr,
):
    pid_h = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_b = tl.program_id(2)
    h_idx = tl.load(head_index_ptr + pid_h).to(tl.int32) if USE_HEAD_INDEX else pid_h

    q_start = tl.load(cu_q_ptr + pid_b).to(tl.int32)
    q_end = tl.load(cu_q_ptr + pid_b + 1).to(tl.int32)
    k_start = tl.load(cu_k_ptr + pid_b).to(tl.int32)
    k_end = tl.load(cu_k_ptr + pid_b + 1).to(tl.int32)
    q_len = q_end - q_start
    k_len = k_end - k_start

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < q_len
    hk = h_idx // gqa_group_size

    offs_d = tl.arange(0, BLOCK_D)
    q_ptrs = (
        q_ptr
        + (q_start + offs_m[:, None]) * stride_qt
        + h_idx * stride_qh
        + offs_d[None, :] * stride_qd
    )
    q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)
    tok_r = offs_m[:, None]
    block_r = (pid_m * BLOCK_M) // BLOCK_DIM_CONST

    m_i = tl.full([BLOCK_M], -float("inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
    k_align_offset = k_len - q_len
    sparse_head_idx = tl.load(head_mask_ptr + h_idx).to(tl.int32) - 1

    n_loop_end = k_len
    if IS_CAUSAL:
        max_tok_c = pid_m * BLOCK_M + (BLOCK_M - 1) + k_align_offset + 1
        n_loop_end = tl.minimum(k_len, tl.maximum(max_tok_c, 0))
    if USE_ROW_LIST:
        for nz_idx in tl.range(0, NCOL_MAX, 1, loop_unroll_factor=1):
            block_c = tl.load(
                row_blockmask_ptr
                + pid_b * stride_bmb
                + sparse_head_idx * stride_bmh
                + block_r * stride_bmr
                + nz_idx * stride_bmc,
                mask=block_r < nrow_max,
                other=-1,
            ).to(tl.int32)
            valid_block = block_c >= 0
            if valid_block:
                n_start = block_c * BLOCK_DIM_CONST
                if n_start < n_loop_end:
                    for block_off in range(0, BLOCK_DIM_CONST, BLOCK_N):
                        n_block_start = n_start + block_off
                        offs_n = n_block_start + tl.arange(0, BLOCK_N)
                        mask_n = offs_n < k_len

                        k_ptrs = (
                            k_ptr
                            + (k_start + offs_n[:, None]) * stride_kt
                            + hk * stride_kh
                            + offs_d[None, :] * stride_kd
                        )
                        v_ptrs = (
                            v_ptr
                            + (k_start + offs_n[:, None]) * stride_vt
                            + hk * stride_vh
                            + offs_d[None, :] * stride_vd
                        )
                        k = tl.load(
                            k_ptrs,
                            mask=mask_n[:, None],
                            other=0.0,
                            eviction_policy="evict_last",
                        )
                        v = tl.load(
                            v_ptrs,
                            mask=mask_n[:, None],
                            other=0.0,
                            eviction_policy="evict_last",
                        )

                        qk = (tl.dot(q, tl.trans(k)) * (softmax_scale * RCP_LN2)).to(
                            tl.float32
                        )

                        keep = mask_m[:, None] & mask_n[None, :]
                        if IS_CAUSAL:
                            tok_c = offs_n[None, :]
                            keep = keep & (tok_c <= (tok_r + k_align_offset))

                        qk = tl.where(keep, qk, -float("inf"))

                        m_ij = tl.max(qk, axis=1)
                        m_new = tl.maximum(m_i, m_ij)
                        m_new_inf = m_new == -float("inf")
                        alpha = tl.exp2(tl.where(m_new_inf, 0.0, m_i - m_new))
                        p = tl.where(
                            keep,
                            tl.exp2(
                                qk - tl.where(m_new_inf[:, None], 0.0, m_new[:, None])
                            ),
                            0.0,
                        )
                        l_i = l_i * alpha + tl.sum(p, axis=1)
                        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
                        m_i = m_new
    else:
        for n_start in range(0, n_loop_end, BLOCK_N):
            block_c = n_start // BLOCK_DIM_CONST
            bm_ptrs = (
                row_blockmask_ptr
                + pid_b * stride_bmb
                + sparse_head_idx * stride_bmh
                + block_r * stride_bmr
                + block_c * stride_bmc
            )
            bm_mask = (block_r < nrow_max) & (block_c < ncol_max)
            block_keep = tl.load(bm_ptrs, mask=bm_mask, other=0).to(tl.int1)
            if block_keep:
                offs_n = n_start + tl.arange(0, BLOCK_N)
                mask_n = offs_n < k_len

                k_ptrs = (
                    k_ptr
                    + (k_start + offs_n[:, None]) * stride_kt
                    + hk * stride_kh
                    + offs_d[None, :] * stride_kd
                )
                v_ptrs = (
                    v_ptr
                    + (k_start + offs_n[:, None]) * stride_vt
                    + hk * stride_vh
                    + offs_d[None, :] * stride_vd
                )
                k = tl.load(
                    k_ptrs,
                    mask=mask_n[:, None],
                    other=0.0,
                    eviction_policy="evict_last",
                )
                v = tl.load(
                    v_ptrs,
                    mask=mask_n[:, None],
                    other=0.0,
                    eviction_policy="evict_last",
                )

                qk = (tl.dot(q, tl.trans(k)) * (softmax_scale * RCP_LN2)).to(tl.float32)
                full_tile = (n_start + BLOCK_N <= k_len) & (
                    pid_m * BLOCK_M + BLOCK_M <= q_len
                )
                if IS_CAUSAL:
                    full_tile = full_tile & (
                        n_start + BLOCK_N - 1 <= pid_m * BLOCK_M + k_align_offset
                    )
                if full_tile:
                    m_ij = tl.max(qk, axis=1)
                    m_new = tl.maximum(m_i, m_ij)
                    alpha = tl.exp2(m_i - m_new)
                    p = tl.exp2(qk - m_new[:, None])
                    l_i = l_i * alpha + tl.sum(p, axis=1)
                    acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
                    m_i = m_new
                else:
                    keep = mask_m[:, None] & mask_n[None, :]
                    if IS_CAUSAL:
                        tok_c = offs_n[None, :]
                        keep = keep & (tok_c <= (tok_r + k_align_offset))

                    qk = tl.where(keep, qk, -float("inf"))

                    m_ij = tl.max(qk, axis=1)
                    m_new = tl.maximum(m_i, m_ij)
                    m_new_inf = m_new == -float("inf")
                    alpha = tl.exp2(tl.where(m_new_inf, 0.0, m_i - m_new))
                    p = tl.where(
                        keep,
                        tl.exp2(qk - tl.where(m_new_inf[:, None], 0.0, m_new[:, None])),
                        0.0,
                    )
                    l_i = l_i * alpha + tl.sum(p, axis=1)
                    acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
                    m_i = m_new

    row_has_any = l_i > 0.0
    out_vals = tl.where(row_has_any[:, None], acc / l_i[:, None], 0.0)
    out_ptrs = (
        out_ptr
        + (q_start + offs_m[:, None]) * stride_ot
        + h_idx * stride_oh
        + offs_d[None, :] * stride_od
    )
    tl.store(out_ptrs, out_vals.to(q.dtype), mask=mask_m[:, None])

    lse_vals = tl.where(row_has_any, m_i * LN2 + tl.log(l_i), float("inf"))
    lse_ptrs = (
        lse_ptr + pid_b * stride_lseb + h_idx * stride_lseh + offs_m * stride_lsem
    )
    tl.store(lse_ptrs, lse_vals, mask=mask_m)


@triton.jit
def _fwd_varlen_streaming_uniform_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    out_ptr,
    lse_ptr,
    cu_q_ptr,
    cu_k_ptr,
    head_index_ptr,
    stride_qt,
    stride_qh,
    stride_qd,
    stride_kt,
    stride_kh,
    stride_kd,
    stride_vt,
    stride_vh,
    stride_vd,
    stride_ot,
    stride_oh,
    stride_od,
    stride_lseb,
    stride_lseh,
    stride_lsem,
    gqa_group_size,
    max_seqlen_k,
    softmax_scale,
    IS_CAUSAL: tl.constexpr,
    USE_HEAD_INDEX: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_DIM_CONST: tl.constexpr,
    SINK_BLOCKS: tl.constexpr,
    LOCAL_BLOCKS: tl.constexpr,
):
    pid_h = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_b = tl.program_id(2)
    h_idx = tl.load(head_index_ptr + pid_h).to(tl.int32) if USE_HEAD_INDEX else pid_h

    q_start = tl.load(cu_q_ptr + pid_b).to(tl.int32)
    q_end = tl.load(cu_q_ptr + pid_b + 1).to(tl.int32)
    k_start = tl.load(cu_k_ptr + pid_b).to(tl.int32)
    k_end = tl.load(cu_k_ptr + pid_b + 1).to(tl.int32)
    q_len = q_end - q_start
    k_len = k_end - k_start

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < q_len
    hk = h_idx // gqa_group_size

    offs_d = tl.arange(0, BLOCK_D)
    q_ptrs = (
        q_ptr
        + (q_start + offs_m[:, None]) * stride_qt
        + h_idx * stride_qh
        + offs_d[None, :] * stride_qd
    )
    q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)
    tok_r = offs_m[:, None]
    block_r = (pid_m * BLOCK_M) // BLOCK_DIM_CONST

    m_i = tl.full([BLOCK_M], -float("inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

    ncol = (k_len + BLOCK_DIM_CONST - 1) // BLOCK_DIM_CONST
    k_align_offset = k_len - q_len
    start_row_idx = tl.maximum((q_len - k_len) // BLOCK_DIM_CONST, 0)
    causal_shift_blocks = tl.maximum(
        (k_len - q_len + BLOCK_DIM_CONST - 1) // BLOCK_DIM_CONST, 0
    )

    max_row_block_num = ncol
    if IS_CAUSAL:
        max_row_block_num = causal_shift_blocks + 1 + block_r - start_row_idx
    max_row_block_num = tl.maximum(max_row_block_num, 0)
    start_col = tl.minimum(tl.maximum(max_row_block_num - LOCAL_BLOCKS, 0), ncol)
    end_col = tl.minimum(max_row_block_num, ncol)

    n_loop_end = k_len
    if IS_CAUSAL:
        max_tok_c = pid_m * BLOCK_M + (BLOCK_M - 1) + k_align_offset + 1
        n_loop_end = tl.minimum(k_len, tl.maximum(max_tok_c, 0))

    sink_end_tok = tl.minimum(n_loop_end, tl.maximum(SINK_BLOCKS, 0) * BLOCK_DIM_CONST)
    local_start_tok = tl.maximum(start_col * BLOCK_DIM_CONST, 0)
    local_end_tok = tl.minimum(end_col * BLOCK_DIM_CONST, n_loop_end)
    local_start_tok = tl.minimum(tl.maximum(local_start_tok, sink_end_tok), n_loop_end)
    local_end_tok = tl.maximum(local_end_tok, local_start_tok)

    for n_start in range(0, SINK_BLOCKS * BLOCK_DIM_CONST, BLOCK_N):
        offs_n = n_start + tl.arange(0, BLOCK_N)
        mask_n = offs_n < sink_end_tok

        k_ptrs = (
            k_ptr
            + (k_start + offs_n[:, None]) * stride_kt
            + hk * stride_kh
            + offs_d[None, :] * stride_kd
        )
        v_ptrs = (
            v_ptr
            + (k_start + offs_n[:, None]) * stride_vt
            + hk * stride_vh
            + offs_d[None, :] * stride_vd
        )
        k = tl.load(
            k_ptrs, mask=mask_n[:, None], other=0.0, eviction_policy="evict_last"
        )
        v = tl.load(
            v_ptrs, mask=mask_n[:, None], other=0.0, eviction_policy="evict_last"
        )

        qk = (tl.dot(q, tl.trans(k)) * (softmax_scale * RCP_LN2)).to(tl.float32)
        full_tile = (n_start + BLOCK_N <= sink_end_tok) & (
            pid_m * BLOCK_M + BLOCK_M <= q_len
        )
        if IS_CAUSAL:
            full_tile = full_tile & (
                n_start + BLOCK_N - 1 <= pid_m * BLOCK_M + k_align_offset
            )
        if full_tile:
            m_ij = tl.max(qk, axis=1)
            m_new = tl.maximum(m_i, m_ij)
            alpha = tl.exp2(m_i - m_new)
            p = tl.exp2(qk - m_new[:, None])
            l_i = l_i * alpha + tl.sum(p, axis=1)
            acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
            m_i = m_new
        else:
            keep = mask_m[:, None] & mask_n[None, :]
            if IS_CAUSAL:
                tok_c = offs_n[None, :]
                keep = keep & (tok_c <= (tok_r + k_align_offset))
            qk = tl.where(keep, qk, -float("inf"))

            m_ij = tl.max(qk, axis=1)
            m_new = tl.maximum(m_i, m_ij)
            m_new_inf = m_new == -float("inf")
            alpha = tl.exp2(tl.where(m_new_inf, 0.0, m_i - m_new))
            p = tl.where(
                keep,
                tl.exp2(qk - tl.where(m_new_inf[:, None], 0.0, m_new[:, None])),
                0.0,
            )
            l_i = l_i * alpha + tl.sum(p, axis=1)
            acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
            m_i = m_new

    for local_off in range(0, LOCAL_BLOCKS * BLOCK_DIM_CONST, BLOCK_N):
        n_start = local_start_tok + local_off
        offs_n = n_start + tl.arange(0, BLOCK_N)
        mask_n = offs_n < local_end_tok

        k_ptrs = (
            k_ptr
            + (k_start + offs_n[:, None]) * stride_kt
            + hk * stride_kh
            + offs_d[None, :] * stride_kd
        )
        v_ptrs = (
            v_ptr
            + (k_start + offs_n[:, None]) * stride_vt
            + hk * stride_vh
            + offs_d[None, :] * stride_vd
        )
        k = tl.load(
            k_ptrs, mask=mask_n[:, None], other=0.0, eviction_policy="evict_last"
        )
        v = tl.load(
            v_ptrs, mask=mask_n[:, None], other=0.0, eviction_policy="evict_last"
        )

        qk = (tl.dot(q, tl.trans(k)) * (softmax_scale * RCP_LN2)).to(tl.float32)
        full_tile = (n_start + BLOCK_N <= local_end_tok) & (
            pid_m * BLOCK_M + BLOCK_M <= q_len
        )
        if IS_CAUSAL:
            full_tile = full_tile & (
                n_start + BLOCK_N - 1 <= pid_m * BLOCK_M + k_align_offset
            )
        if full_tile:
            m_ij = tl.max(qk, axis=1)
            m_new = tl.maximum(m_i, m_ij)
            alpha = tl.exp2(m_i - m_new)
            p = tl.exp2(qk - m_new[:, None])
            l_i = l_i * alpha + tl.sum(p, axis=1)
            acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
            m_i = m_new
        else:
            keep = mask_m[:, None] & mask_n[None, :]
            if IS_CAUSAL:
                tok_c = offs_n[None, :]
                keep = keep & (tok_c <= (tok_r + k_align_offset))
            qk = tl.where(keep, qk, -float("inf"))

            m_ij = tl.max(qk, axis=1)
            m_new = tl.maximum(m_i, m_ij)
            m_new_inf = m_new == -float("inf")
            alpha = tl.exp2(tl.where(m_new_inf, 0.0, m_i - m_new))
            p = tl.where(
                keep,
                tl.exp2(qk - tl.where(m_new_inf[:, None], 0.0, m_new[:, None])),
                0.0,
            )
            l_i = l_i * alpha + tl.sum(p, axis=1)
            acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
            m_i = m_new

    row_has_any = l_i > 0.0
    out_vals = tl.where(row_has_any[:, None], acc / l_i[:, None], 0.0)
    out_ptrs = (
        out_ptr
        + (q_start + offs_m[:, None]) * stride_ot
        + h_idx * stride_oh
        + offs_d[None, :] * stride_od
    )
    tl.store(out_ptrs, out_vals.to(q.dtype), mask=mask_m[:, None])

    lse_vals = tl.where(row_has_any, m_i * LN2 + tl.log(l_i), float("inf"))
    lse_ptrs = (
        lse_ptr + pid_b * stride_lseb + h_idx * stride_lseh + offs_m * stride_lsem
    )
    tl.store(lse_ptrs, lse_vals, mask=mask_m)


def _block_sparse_attn_fwd_core(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    head_mask_type: torch.Tensor,
    streaming_info: torch.Tensor,
    base_blockmask: Optional[torch.Tensor],
    max_seqlen_q_: int,
    max_seqlen_k_: int,
    scale: float,
    is_causal: bool,
    exact_streaming: bool,
    return_attn_probs: bool,
    mode: str,
    dispatch_key: _DispatchKey,
    opts: _KernelOptions,
):
    batch_size = dispatch_key.batch_size
    nheads = q.shape[1]
    gqa_group_size = q.shape[1] // k.shape[1]

    dense_only = mode == "dense"
    stream_only = mode == "streaming"
    blocksparse_only = mode == "blocksparse"

    max_q_round = _round_multiple(max_seqlen_q_, BLOCK_DIM)
    out = torch.empty_like(q)
    softmax_lse = torch.full(
        (batch_size, nheads, max_q_round),
        float("inf"),
        dtype=torch.float32,
        device=q.device,
    )
    s_dmask = torch.empty((0,), dtype=q.dtype, device=q.device)

    if max_seqlen_k_ == 0:
        out.zero_()
        return out if not return_attn_probs else (out, softmax_lse, s_dmask)

    dense_block_m = opts.dense_block_m
    mode_block_m = opts.mode_block_m
    mixed_block_m = opts.mixed_block_m
    blocksparse_block_m = opts.blocksparse_block_m
    block_n_mode = opts.block_n_mode
    block_n_stream = opts.block_n_stream
    block_n_mixed = opts.block_n_mixed
    block_n_sparse = opts.block_n_sparse
    num_warps_dense = opts.num_warps_dense
    num_warps_stream = opts.num_warps_stream
    num_warps_mixed = opts.num_warps_mixed
    num_warps_sparse = opts.num_warps_sparse
    num_stages_sparse = opts.num_stages_sparse
    num_stages_stream = opts.num_stages_stream
    use_row_list_sparse = opts.use_row_list_sparse
    use_row_list_mixed = opts.use_row_list_mixed
    use_single_mixed = opts.use_single_mixed
    very_long_single_seq = dispatch_key.seq_bucket == "very_long_single"
    stream_sink_blocks = 0
    stream_local_blocks = 0
    use_uniform_block_stream = opts.use_uniform_block_stream
    if use_uniform_block_stream and streaming_info.numel() >= 2:
        stream_sink_blocks = int(streaming_info[0].item())
        stream_local_blocks = int(streaming_info[1].item())
    else:
        use_uniform_block_stream = False
    # For USE_HEAD_INDEX=False kernels, this pointer is ignored.
    dummy_head_index = head_mask_type
    if dense_only and not exact_streaming:
        grid = (nheads, triton.cdiv(max_q_round, dense_block_m), batch_size)
        _fwd_varlen_dense_kernel[grid](
            q,
            k,
            v,
            out,
            softmax_lse,
            cu_seqlens_q,
            cu_seqlens_k,
            dummy_head_index,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            v.stride(0),
            v.stride(1),
            v.stride(2),
            out.stride(0),
            out.stride(1),
            out.stride(2),
            softmax_lse.stride(0),
            softmax_lse.stride(1),
            softmax_lse.stride(2),
            gqa_group_size,
            max_seqlen_k_,
            scale,
            IS_CAUSAL=is_causal,
            USE_HEAD_INDEX=False,
            BLOCK_M=dense_block_m,
            BLOCK_N=block_n_mode,
            BLOCK_D=q.shape[-1],
            num_warps=num_warps_dense,
            num_stages=3,
        )
    elif stream_only:
        grid = (nheads, triton.cdiv(max_q_round, mode_block_m), batch_size)
        if use_uniform_block_stream:
            _fwd_varlen_streaming_uniform_kernel[grid](
                q,
                k,
                v,
                out,
                softmax_lse,
                cu_seqlens_q,
                cu_seqlens_k,
                dummy_head_index,
                q.stride(0),
                q.stride(1),
                q.stride(2),
                k.stride(0),
                k.stride(1),
                k.stride(2),
                v.stride(0),
                v.stride(1),
                v.stride(2),
                out.stride(0),
                out.stride(1),
                out.stride(2),
                softmax_lse.stride(0),
                softmax_lse.stride(1),
                softmax_lse.stride(2),
                gqa_group_size,
                max_seqlen_k_,
                scale,
                IS_CAUSAL=is_causal,
                USE_HEAD_INDEX=False,
                BLOCK_M=mode_block_m,
                BLOCK_N=block_n_stream,
                BLOCK_D=q.shape[-1],
                BLOCK_DIM_CONST=BLOCK_DIM,
                SINK_BLOCKS=stream_sink_blocks,
                LOCAL_BLOCKS=stream_local_blocks,
                num_warps=num_warps_stream,
                num_stages=num_stages_stream,
            )
        else:
            _fwd_varlen_streaming_kernel[grid](
                q,
                k,
                v,
                out,
                softmax_lse,
                cu_seqlens_q,
                cu_seqlens_k,
                dummy_head_index,
                streaming_info,
                q.stride(0),
                q.stride(1),
                q.stride(2),
                k.stride(0),
                k.stride(1),
                k.stride(2),
                v.stride(0),
                v.stride(1),
                v.stride(2),
                out.stride(0),
                out.stride(1),
                out.stride(2),
                softmax_lse.stride(0),
                softmax_lse.stride(1),
                softmax_lse.stride(2),
                gqa_group_size,
                max_seqlen_k_,
                scale,
                IS_CAUSAL=is_causal,
                EXACT_STREAMING=exact_streaming,
                USE_HEAD_INDEX=False,
                BLOCK_M=mode_block_m,
                BLOCK_N=block_n_stream,
                BLOCK_D=q.shape[-1],
                BLOCK_DIM_CONST=BLOCK_DIM,
                num_warps=num_warps_stream,
                num_stages=num_stages_stream,
            )
    elif blocksparse_only and base_blockmask is not None:
        sparse_blockmask = (
            _convert_blockmask_row_reverse(base_blockmask)
            if use_row_list_sparse
            else base_blockmask
        )
        grid = (nheads, triton.cdiv(max_q_round, blocksparse_block_m), batch_size)
        _fwd_varlen_blocksparse_kernel[grid](
            q,
            k,
            v,
            out,
            softmax_lse,
            cu_seqlens_q,
            cu_seqlens_k,
            dummy_head_index,
            head_mask_type,
            sparse_blockmask,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            v.stride(0),
            v.stride(1),
            v.stride(2),
            out.stride(0),
            out.stride(1),
            out.stride(2),
            softmax_lse.stride(0),
            softmax_lse.stride(1),
            softmax_lse.stride(2),
            sparse_blockmask.stride(0),
            sparse_blockmask.stride(1),
            sparse_blockmask.stride(2),
            sparse_blockmask.stride(3),
            gqa_group_size,
            int(sparse_blockmask.shape[-2]),
            int(sparse_blockmask.shape[-1]),
            max_seqlen_k_,
            scale,
            IS_CAUSAL=is_causal,
            USE_HEAD_INDEX=False,
            USE_ROW_LIST=use_row_list_sparse,
            BLOCK_M=blocksparse_block_m,
            BLOCK_N=block_n_sparse,
            BLOCK_D=q.shape[-1],
            BLOCK_DIM_CONST=BLOCK_DIM,
            NCOL_MAX=int(sparse_blockmask.shape[-1]),
            num_warps=num_warps_sparse,
            num_stages=num_stages_sparse,
        )
    elif use_single_mixed and base_blockmask is not None:
        mixed_blockmask = (
            _convert_blockmask_row_reverse(base_blockmask)
            if use_row_list_mixed
            else base_blockmask
        )
        grid = (nheads, triton.cdiv(max_q_round, mixed_block_m), batch_size)
        _fwd_varlen_mixed_kernel[grid](
            q,
            k,
            v,
            out,
            softmax_lse,
            cu_seqlens_q,
            cu_seqlens_k,
            head_mask_type,
            streaming_info,
            mixed_blockmask,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            v.stride(0),
            v.stride(1),
            v.stride(2),
            out.stride(0),
            out.stride(1),
            out.stride(2),
            softmax_lse.stride(0),
            softmax_lse.stride(1),
            softmax_lse.stride(2),
            mixed_blockmask.stride(0),
            mixed_blockmask.stride(1),
            mixed_blockmask.stride(2),
            mixed_blockmask.stride(3),
            nheads,
            gqa_group_size,
            int(mixed_blockmask.shape[-2]),
            int(mixed_blockmask.shape[-1]),
            max_seqlen_k_,
            scale,
            IS_CAUSAL=is_causal,
            EXACT_STREAMING=exact_streaming,
            HAS_BASE_BLOCKMASK=True,
            USE_ROW_LIST=use_row_list_mixed,
            BLOCK_M=mixed_block_m,
            BLOCK_N=block_n_mixed,
            BLOCK_D=q.shape[-1],
            BLOCK_DIM_CONST=BLOCK_DIM,
            NCOL_MAX=int(mixed_blockmask.shape[-1]),
            num_warps=num_warps_mixed,
            num_stages=3,
        )
    else:
        dense_idx = (
            torch.nonzero(head_mask_type == 0, as_tuple=False).flatten().to(torch.int32)
        )
        stream_idx = (
            torch.nonzero(head_mask_type < 0, as_tuple=False).flatten().to(torch.int32)
        )
        sparse_idx = (
            torch.nonzero(head_mask_type > 0, as_tuple=False).flatten().to(torch.int32)
        )

        dense_done = False
        if (
            dense_idx.numel() > 0
            and _flash_attn_varlen_func is not None
            and not return_attn_probs
            and not exact_streaming
            and not (very_long_single_seq and q.shape[-1] == 128)
        ):
            dense_heads = [int(x) for x in dense_idx.detach().cpu().tolist()]
            for hk in sorted({h // gqa_group_size for h in dense_heads}):
                heads = [h for h in dense_heads if h // gqa_group_size == hk]
                head_tensor = torch.tensor(heads, dtype=torch.int64, device=q.device)
                q_dense = q.index_select(1, head_tensor)
                dense_out = _flash_attn_varlen_func(
                    q_dense,
                    k[:, hk : hk + 1, :],
                    v[:, hk : hk + 1, :],
                    cu_seqlens_q,
                    cu_seqlens_k,
                    max_seqlen_q_,
                    max_seqlen_k_,
                    dropout_p=0.0,
                    softmax_scale=scale,
                    causal=is_causal,
                    deterministic=False,
                )
                out.index_copy_(1, head_tensor, dense_out)
            dense_done = True

        if dense_idx.numel() > 0 and not dense_done:
            grid = (
                dense_idx.numel(),
                triton.cdiv(max_q_round, dense_block_m),
                batch_size,
            )
            _fwd_varlen_dense_kernel[grid](
                q,
                k,
                v,
                out,
                softmax_lse,
                cu_seqlens_q,
                cu_seqlens_k,
                dense_idx,
                q.stride(0),
                q.stride(1),
                q.stride(2),
                k.stride(0),
                k.stride(1),
                k.stride(2),
                v.stride(0),
                v.stride(1),
                v.stride(2),
                out.stride(0),
                out.stride(1),
                out.stride(2),
                softmax_lse.stride(0),
                softmax_lse.stride(1),
                softmax_lse.stride(2),
                gqa_group_size,
                max_seqlen_k_,
                scale,
                IS_CAUSAL=is_causal,
                USE_HEAD_INDEX=True,
                BLOCK_M=dense_block_m,
                BLOCK_N=block_n_mode,
                BLOCK_D=q.shape[-1],
                num_warps=num_warps_dense,
                num_stages=3,
            )

        if stream_idx.numel() > 0:
            grid = (
                stream_idx.numel(),
                triton.cdiv(max_q_round, mode_block_m),
                batch_size,
            )
            if use_uniform_block_stream:
                _fwd_varlen_streaming_uniform_kernel[grid](
                    q,
                    k,
                    v,
                    out,
                    softmax_lse,
                    cu_seqlens_q,
                    cu_seqlens_k,
                    stream_idx,
                    q.stride(0),
                    q.stride(1),
                    q.stride(2),
                    k.stride(0),
                    k.stride(1),
                    k.stride(2),
                    v.stride(0),
                    v.stride(1),
                    v.stride(2),
                    out.stride(0),
                    out.stride(1),
                    out.stride(2),
                    softmax_lse.stride(0),
                    softmax_lse.stride(1),
                    softmax_lse.stride(2),
                    gqa_group_size,
                    max_seqlen_k_,
                    scale,
                    IS_CAUSAL=is_causal,
                    USE_HEAD_INDEX=True,
                    BLOCK_M=mode_block_m,
                    BLOCK_N=block_n_stream,
                    BLOCK_D=q.shape[-1],
                    BLOCK_DIM_CONST=BLOCK_DIM,
                    SINK_BLOCKS=stream_sink_blocks,
                    LOCAL_BLOCKS=stream_local_blocks,
                    num_warps=num_warps_stream,
                    num_stages=num_stages_stream,
                )
            else:
                _fwd_varlen_streaming_kernel[grid](
                    q,
                    k,
                    v,
                    out,
                    softmax_lse,
                    cu_seqlens_q,
                    cu_seqlens_k,
                    stream_idx,
                    streaming_info,
                    q.stride(0),
                    q.stride(1),
                    q.stride(2),
                    k.stride(0),
                    k.stride(1),
                    k.stride(2),
                    v.stride(0),
                    v.stride(1),
                    v.stride(2),
                    out.stride(0),
                    out.stride(1),
                    out.stride(2),
                    softmax_lse.stride(0),
                    softmax_lse.stride(1),
                    softmax_lse.stride(2),
                    gqa_group_size,
                    max_seqlen_k_,
                    scale,
                    IS_CAUSAL=is_causal,
                    EXACT_STREAMING=exact_streaming,
                    USE_HEAD_INDEX=True,
                    BLOCK_M=mode_block_m,
                    BLOCK_N=block_n_stream,
                    BLOCK_D=q.shape[-1],
                    BLOCK_DIM_CONST=BLOCK_DIM,
                    num_warps=num_warps_stream,
                    num_stages=num_stages_stream,
                )

        if sparse_idx.numel() > 0:
            if base_blockmask is None:
                raise ValueError(
                    "base_blockmask must be provided for blocksparse heads"
                )
            sparse_blockmask = (
                _convert_blockmask_row_reverse(base_blockmask)
                if use_row_list_mixed
                else base_blockmask
            )
            split_sparse_block_m = (
                128
                if max_seqlen_k_ >= LONG_SEQ_MIN
                and batch_size > 1
                and q.shape[-1] == 32
                else blocksparse_block_m
            )
            grid = (
                sparse_idx.numel(),
                triton.cdiv(max_q_round, split_sparse_block_m),
                batch_size,
            )
            _fwd_varlen_blocksparse_kernel[grid](
                q,
                k,
                v,
                out,
                softmax_lse,
                cu_seqlens_q,
                cu_seqlens_k,
                sparse_idx,
                head_mask_type,
                sparse_blockmask,
                q.stride(0),
                q.stride(1),
                q.stride(2),
                k.stride(0),
                k.stride(1),
                k.stride(2),
                v.stride(0),
                v.stride(1),
                v.stride(2),
                out.stride(0),
                out.stride(1),
                out.stride(2),
                softmax_lse.stride(0),
                softmax_lse.stride(1),
                softmax_lse.stride(2),
                sparse_blockmask.stride(0),
                sparse_blockmask.stride(1),
                sparse_blockmask.stride(2),
                sparse_blockmask.stride(3),
                gqa_group_size,
                int(sparse_blockmask.shape[-2]),
                int(sparse_blockmask.shape[-1]),
                max_seqlen_k_,
                scale,
                IS_CAUSAL=is_causal,
                USE_HEAD_INDEX=True,
                USE_ROW_LIST=use_row_list_mixed,
                BLOCK_M=split_sparse_block_m,
                BLOCK_N=block_n_sparse,
                BLOCK_D=q.shape[-1],
                BLOCK_DIM_CONST=BLOCK_DIM,
                NCOL_MAX=int(sparse_blockmask.shape[-1]),
                num_warps=num_warps_sparse,
                num_stages=num_stages_sparse,
            )

    return out if not return_attn_probs else (out, softmax_lse, s_dmask)


@torch.library.custom_op(
    "flashvsr::_triton_block_sparse_attn_forward",
    mutates_args=(),
    device_types="cuda",
)
def _triton_block_sparse_attn_forward_opaque(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    head_mask_type: torch.Tensor,
    streaming_info: torch.Tensor,
    base_blockmask: torch.Tensor,
    max_seqlen_q_: int,
    max_seqlen_k_: int,
    scale: float,
    is_causal: bool,
    exact_streaming: bool,
) -> torch.Tensor:
    batch_size = cu_seqlens_q.numel() - 1
    dispatch_key, opts = _dispatch_kernel_options(
        q=q,
        batch_size=batch_size,
        max_seqlen_k=max_seqlen_k_,
        exact_streaming=exact_streaming,
        mode="blocksparse",
        has_base_blockmask=True,
    )
    return _block_sparse_attn_fwd_core(
        q=q,
        k=k,
        v=v,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        head_mask_type=head_mask_type,
        streaming_info=streaming_info,
        base_blockmask=base_blockmask,
        max_seqlen_q_=max_seqlen_q_,
        max_seqlen_k_=max_seqlen_k_,
        scale=scale,
        is_causal=is_causal,
        exact_streaming=exact_streaming,
        return_attn_probs=False,
        mode="blocksparse",
        dispatch_key=dispatch_key,
        opts=opts,
    )


@torch.library.register_fake("flashvsr::_triton_block_sparse_attn_forward")
def _triton_block_sparse_attn_forward_opaque_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    head_mask_type: torch.Tensor,
    streaming_info: torch.Tensor,
    base_blockmask: torch.Tensor,
    max_seqlen_q_: int,
    max_seqlen_k_: int,
    scale: float,
    is_causal: bool,
    exact_streaming: bool,
) -> torch.Tensor:
    del (
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        head_mask_type,
        streaming_info,
        base_blockmask,
        max_seqlen_q_,
        max_seqlen_k_,
        scale,
        is_causal,
        exact_streaming,
    )
    return torch.empty_like(q)


_wrapped_triton_block_sparse_attn_forward = (
    torch.ops.flashvsr._triton_block_sparse_attn_forward
)


def _should_use_opaque_blocksparse_forward(
    *,
    mode: str,
    base_blockmask: Optional[torch.Tensor],
    return_attn_probs: bool,
) -> bool:
    return (
        mode == "blocksparse"
        and base_blockmask is not None
        and not return_attn_probs
        and not torch.is_grad_enabled()
    )


def block_sparse_attn_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    head_mask_type: torch.Tensor,
    streaming_info: Optional[torch.Tensor],
    base_blockmask: Optional[torch.Tensor],
    max_seqlen_q_: int,
    max_seqlen_k_: int,
    p_dropout: float,
    deterministic: bool = False,
    softmax_scale: Optional[float] = None,
    is_causal: bool = False,
    exact_streaming: bool = False,
    return_attn_probs: bool = False,
    mode_hint: Optional[str] = None,
    head_mask_type_is_renumbered: bool = False,
):
    if deterministic:
        raise NotImplementedError(
            "Triton block-sparse attention does not support deterministic=True"
        )

    mode = "auto" if mode_hint is None else mode_hint
    if mode not in ("auto", "dense", "streaming", "blocksparse"):
        raise ValueError(
            "mode_hint must be one of {'auto','dense','streaming','blocksparse'}"
        )

    q, k, v = [_maybe_contiguous(x) for x in (q, k, v)]
    head_mask_type = _maybe_contiguous(head_mask_type)
    if mode in ("blocksparse", "auto") and not head_mask_type_is_renumbered:
        head_mask_type, _ = _replace_ones_with_count(head_mask_type)

    _validate_forward_inputs(
        q=q,
        k=k,
        v=v,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        head_mask_type=head_mask_type,
        streaming_info=streaming_info,
        base_blockmask=base_blockmask,
        max_seqlen_q_=max_seqlen_q_,
        max_seqlen_k_=max_seqlen_k_,
        exact_streaming=exact_streaming,
    )
    if exact_streaming and not is_causal:
        raise ValueError("exact_streaming requires is_causal=True")
    if streaming_info is None:
        streaming_info = torch.zeros(
            (q.shape[1] * 2,), dtype=torch.int32, device=q.device
        )
    if p_dropout > 0.0:
        raise NotImplementedError(
            "Triton-only forward currently supports p_dropout == 0.0"
        )

    batch_size = cu_seqlens_q.numel() - 1
    scale = float(softmax_scale if softmax_scale is not None else q.shape[-1] ** (-0.5))

    if _should_use_opaque_blocksparse_forward(
        mode=mode,
        base_blockmask=base_blockmask,
        return_attn_probs=return_attn_probs,
    ):
        assert base_blockmask is not None and streaming_info is not None
        return _wrapped_triton_block_sparse_attn_forward(
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
            scale,
            is_causal,
            exact_streaming,
        )

    if (
        mode == "dense"
        and not exact_streaming
        and not return_attn_probs
        and max_seqlen_k_ > 0
        and _flash_attn_varlen_func is not None
    ):
        return _flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q_,
            max_seqlen_k_,
            dropout_p=0.0,
            softmax_scale=scale,
            causal=is_causal,
            deterministic=False,
        )

    dispatch_key, opts = _dispatch_kernel_options(
        q=q,
        batch_size=batch_size,
        max_seqlen_k=max_seqlen_k_,
        exact_streaming=exact_streaming,
        mode=mode,
        has_base_blockmask=base_blockmask is not None,
    )
    return _block_sparse_attn_fwd_core(
        q=q,
        k=k,
        v=v,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        head_mask_type=head_mask_type,
        streaming_info=streaming_info,
        base_blockmask=base_blockmask,
        max_seqlen_q_=max_seqlen_q_,
        max_seqlen_k_=max_seqlen_k_,
        scale=scale,
        is_causal=is_causal,
        exact_streaming=exact_streaming,
        return_attn_probs=return_attn_probs,
        mode=mode,
        dispatch_key=dispatch_key,
        opts=opts,
    )


block_sparse_attn_triton_fwd_func = block_sparse_attn_func
