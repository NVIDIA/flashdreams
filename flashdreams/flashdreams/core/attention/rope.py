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

"""3D rotary position embeddings with CP-aware shifting.

Used by DiTs (e.g. Wan, AlpaDreams) that patchify into a (T, H, W) sequence.
"""

from typing import TypeVar

import torch
from einops import repeat
from torch import Tensor
from torch.distributed import ProcessGroup
from torch.distributed.tensor.device_mesh import DeviceMesh

from flashdreams.core.attention.rope_kernel import apply_rotary_pos_emb
from flashdreams.core.distributed.context_parallel import split_inputs_cp

T = TypeVar("T")


def unpack_optional(maybe_object: T | None) -> T:
    if maybe_object is None:
        raise ValueError("Expected a non-None object")
    return maybe_object


def _compute_freqs(
    dim: int,
    extrapolation_ratio: float = 1.0,
    device: torch.device = torch.device("cuda"),
) -> Tensor:
    """Compute base frequencies for one RoPE dimension with NTK extrapolation.

    Args:
        dim: Number of frequency components (typically dim // 2 of head_dim).
        extrapolation_ratio: Scale factor for extrapolation; > 1 extends context length.

    Returns:
        Base frequencies of shape ``[dim // 2]``.
    """
    dim_range = (
        torch.arange(0, dim, 2, dtype=torch.float32, device=device)[: (dim // 2)] / dim
    )
    ntk_factor = extrapolation_ratio ** (dim / (dim - 2))
    theta = 10000.0 * ntk_factor
    freqs = 1.0 / (theta**dim_range)
    return freqs


class RotaryPositionEmbedding3D:
    """3D rotary position embedding for (t, h, w) sequences.

    Splits head_dim into three parts for time, height, and width. Supports
    context parallelism and time-shift for causal / streaming use.
    """

    raw_freqs_h: Tensor
    raw_freqs_w: Tensor
    raw_freqs_t: Tensor
    freqs_h: Tensor
    freqs_w: Tensor
    freqs_t: Tensor

    def __init__(
        self,
        head_dim: int,
        len_h: int,
        len_w: int,
        len_t: int,
        h_extrapolation_ratio: float = 1.0,
        w_extrapolation_ratio: float = 1.0,
        t_extrapolation_ratio: float = 1.0,
        interleaved: bool = False,
        device: torch.device = torch.device("cuda"),
    ) -> None:
        """Build 3D RoPE for the given sequence lengths and head dimension.

        Args:
            head_dim: Attention head dimension; split into h/w/t sub-dims (2:2:2 ratio).
            len_h: Sequence length along height.
            len_w: Sequence length along width.
            len_t: Sequence length along time.
            h_extrapolation_ratio: NTK extrapolation ratio for height.
            w_extrapolation_ratio: NTK extrapolation ratio for width.
            t_extrapolation_ratio: NTK extrapolation ratio for time.
            interleaved: Whether to interleave the frequency components.
        """
        self.len_h = len_h
        self.len_w = len_w
        self.len_t = len_t
        self.device = device
        self.interleaved = interleaved

        dim_w = dim_h = head_dim // 6 * 2
        dim_t = head_dim - (dim_h + dim_w)

        self.raw_freqs_h = _compute_freqs(dim_h, h_extrapolation_ratio, device)
        self.raw_freqs_w = _compute_freqs(dim_w, w_extrapolation_ratio, device)
        self.raw_freqs_t = _compute_freqs(dim_t, t_extrapolation_ratio, device)

        seq_t = torch.arange(len_t, dtype=torch.float32, device=device)
        seq_h = torch.arange(len_h, dtype=torch.float32, device=device)
        seq_w = torch.arange(len_w, dtype=torch.float32, device=device)

        # Align with the patchify pattern (t, h, w).
        self.freqs_t = repeat(
            torch.outer(seq_t, self.raw_freqs_t),
            "t d -> (t h w) 1 1 d",
            h=len_h,
            w=len_w,
        )
        self.freqs_h = repeat(
            torch.outer(seq_h, self.raw_freqs_h),
            "h d -> (t h w) 1 1 d",
            t=len_t,
            w=len_w,
        )
        self.freqs_w = repeat(
            torch.outer(seq_w, self.raw_freqs_w),
            "w d -> (t h w) 1 1 d",
            t=len_t,
            h=len_h,
        )

        self.device_mesh: DeviceMesh | None = None
        self.freqs_t_cp: Tensor | None = None
        self.freqs_h_cp: Tensor | None = None
        self.freqs_w_cp: Tensor | None = None

    def set_context_parallel_group(self, cp_group: ProcessGroup | None) -> None:
        """Enable or disable context parallelism by splitting frequency buffers along seq dim.

        Currently we assume the sequence length is L = T * H * W. The memory layout is (T, H, W).

        Args:
            cp_group: Process group for context parallel; use None to disable CP.
        """
        if cp_group is None:
            self.device_mesh = None
            self.freqs_t_cp = None
            self.freqs_h_cp = None
            self.freqs_w_cp = None
        else:
            self.device_mesh = DeviceMesh.from_group(cp_group, device_type="cuda")
            self.freqs_t_cp = split_inputs_cp(
                self.freqs_t, seq_dim=0, cp_group=cp_group
            )
            self.freqs_h_cp = split_inputs_cp(
                self.freqs_h, seq_dim=0, cp_group=cp_group
            )
            self.freqs_w_cp = split_inputs_cp(
                self.freqs_w, seq_dim=0, cp_group=cp_group
            )

    def is_context_parallel_enabled(self) -> bool:
        """Return True if context parallelism is active."""
        return self.device_mesh is not None

    def context_parallel_size(self) -> int:
        """Return the context parallel world size, or 1 if CP is disabled."""
        return self.device_mesh.size() if self.device_mesh is not None else 1

    def shift_t(self, autoregressive_index: int) -> Tensor:
        """Shift the time dimension by ``autoregressive_index`` chunks.

        The internal offset is ``autoregressive_index * len_t`` so callers
        only need to track the AR step, not the per-chunk temporal length.

        Args:
            autoregressive_index: AR step index for the chunk being processed.
                Step 0 returns the unshifted frequencies.

        Returns:
            Concatenated RoPE frequencies of shape ``[L, 1, 1, head_dim // 2]``,
            where L is the sequence length T * H * W. The memory layout is (T, H, W).
        """
        offset = autoregressive_index * self.len_t
        if self.is_context_parallel_enabled():
            freqs_t = unpack_optional(self.freqs_t_cp) + offset * self.raw_freqs_t
            freqs_h = unpack_optional(self.freqs_h_cp)
            freqs_w = unpack_optional(self.freqs_w_cp)
        else:
            freqs_t = self.freqs_t + offset * self.raw_freqs_t
            freqs_h = self.freqs_h
            freqs_w = self.freqs_w

        if self.interleaved:
            freqs = torch.cat(
                [
                    freqs_t.repeat_interleave(2, dim=-1),
                    freqs_h.repeat_interleave(2, dim=-1),
                    freqs_w.repeat_interleave(2, dim=-1),
                ],
                dim=-1,
            )
        else:
            freqs = torch.cat([freqs_t, freqs_h, freqs_w] * 2, dim=-1)
        return freqs


def apply_rope_freqs(x: Tensor, freqs: Tensor, interleaved: bool = False) -> Tensor:
    """Apply RoPE frequencies to ``x`` in place via the fused Triton kernel.

    Writes back in place because every call site passes a freshly
    materialised Q or K — there is no autograd graph to preserve.

    Args:
        x: Input tensor of shape ``[B, S, H, D]``; rotated in place.
        freqs: RoPE frequencies of shape ``[S, 1, 1, D]`` as emitted by
            :meth:`RotaryPositionEmbedding3D.shift_t`.
        interleaved: If ``True``, rotate the pair ``(2k, 2k+1)``; else
            rotate ``(d, d + D/2)``.

    Returns:
        Rotated tensor of shape ``[B, S, H, D]``, sharing storage with ``x``.
    """
    return apply_rotary_pos_emb(x, freqs, interleaved=interleaved, inplace=True)
