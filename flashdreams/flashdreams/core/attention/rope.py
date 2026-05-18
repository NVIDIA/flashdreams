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


class _RotaryPositionEmbedding3DBase:
    """Shared 3D RoPE frequency construction."""

    raw_freqs_h: Tensor
    raw_freqs_w: Tensor
    raw_freqs_t: Tensor

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
            device: Device to use for the frequency calculations.
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

        self.device_mesh: DeviceMesh | None = None
        self.cp_group: ProcessGroup | None = None

    def _freq_components_for_len(self, len_t: int) -> tuple[Tensor, Tensor, Tensor]:
        seq_t = torch.arange(len_t, dtype=torch.float32, device=self.device)
        return self._freq_components(seq_t)

    def _freq_components(self, seq_t: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        seq_h = torch.arange(self.len_h, dtype=torch.float32, device=self.device)
        seq_w = torch.arange(self.len_w, dtype=torch.float32, device=self.device)
        len_t = seq_t.shape[0]
        freqs_t = repeat(
            torch.outer(seq_t, self.raw_freqs_t),
            "t d -> (t h w) 1 1 d",
            h=self.len_h,
            w=self.len_w,
        )
        freqs_h = repeat(
            torch.outer(seq_h, self.raw_freqs_h),
            "h d -> (t h w) 1 1 d",
            t=len_t,
            w=self.len_w,
        )
        freqs_w = repeat(
            torch.outer(seq_w, self.raw_freqs_w),
            "w d -> (t h w) 1 1 d",
            t=len_t,
            h=self.len_h,
        )
        return freqs_t, freqs_h, freqs_w

    def set_context_parallel_group(self, cp_group: ProcessGroup | None) -> None:
        """Enable or disable context parallelism by splitting frequency buffers along seq dim.

        Args:
            cp_group: Process group for context parallel; use None to disable CP.
        """
        if cp_group is None:
            self.cp_group = None
            self.device_mesh = None
        else:
            self.cp_group = cp_group
            device_type = (
                self.device.type
                if isinstance(self.device, torch.device)
                else str(self.device)
            )
            self.device_mesh = DeviceMesh.from_group(cp_group, device_type=device_type)

    def is_context_parallel_enabled(self) -> bool:
        """Return True if context parallelism is active."""
        return self.device_mesh is not None

    def context_parallel_size(self) -> int:
        """Return the context parallel world size, or 1 if CP is disabled."""
        return self.device_mesh.size() if self.device_mesh is not None else 1

    def _cat_freqs(self, freqs_t: Tensor, freqs_h: Tensor, freqs_w: Tensor) -> Tensor:
        if self.interleaved:
            return torch.cat(
                [
                    freqs_t.repeat_interleave(2, dim=-1),
                    freqs_h.repeat_interleave(2, dim=-1),
                    freqs_w.repeat_interleave(2, dim=-1),
                ],
                dim=-1,
            )
        return torch.cat([freqs_t, freqs_h, freqs_w] * 2, dim=-1)


class RotaryPositionEmbedding3D(_RotaryPositionEmbedding3DBase):
    """Standard 3D RoPE with unbounded autoregressive time positions.

    Each AR step emits only the current chunk's monotonically increasing
    positions. This is the default RoPE used by existing transformer recipes.
    Use it when keys are rotated before they are written into the KV cache, so
    the cached K tensor already carries its original global position.

    The head dimension is split across temporal, height, and width components
    in a 2:2:2 ratio. ``shift_t()`` concatenates those components into a
    full-width RoPE tensor of shape ``[L, 1, 1, head_dim]`` that can be passed
    directly to :func:`apply_rope_freqs`.

    Args:
        head_dim: Attention head dimension. Must be compatible with the fused
            RoPE kernel and is split across time, height, and width.
        len_h: Number of patch tokens along height in one chunk.
        len_w: Number of patch tokens along width in one chunk.
        len_t: Number of temporal patch tokens in one autoregressive chunk.
        h_extrapolation_ratio: NTK extrapolation ratio for height frequencies.
        w_extrapolation_ratio: NTK extrapolation ratio for width frequencies.
        t_extrapolation_ratio: NTK extrapolation ratio for time frequencies.
        interleaved: Whether RoPE pairs are stored as ``(2k, 2k+1)`` instead
            of ``(k, k + D/2)``.
        device: Device where frequency buffers are allocated.

    Attributes:
        raw_freqs_t: Base temporal RoPE frequency components before expansion,
            shape ``[dim_t // 2]``.
        raw_freqs_h: Base height RoPE frequency components before expansion,
            shape ``[dim_h // 2]``.
        raw_freqs_w: Base width RoPE frequency components before expansion,
            shape ``[dim_w // 2]``.
        freqs_t: Expanded temporal frequency components for one chunk, shape
            ``[L, 1, 1, dim_t // 2]``.
        freqs_h: Expanded height frequency components for one chunk, shape
            ``[L, 1, 1, dim_h // 2]``.
        freqs_w: Expanded width frequency components for one chunk, shape
            ``[L, 1, 1, dim_w // 2]``.

    Examples:

        Apply standard RoPE to the current query and key chunk before writing
        K into the KV cache:

        >>> rope = RotaryPositionEmbedding3D(
        ...     head_dim=128,
        ...     len_t=3,
        ...     len_h=60,
        ...     len_w=104,
        ...     interleaved=True,
        ... )
        >>> freqs = rope.shift_t(autoregressive_index=2)
        >>> freqs.shape
        torch.Size([18720, 1, 1, 128])
        >>> q = apply_rope_freqs(q, freqs, interleaved=rope.interleaved)
        >>> k = apply_rope_freqs(k, freqs, interleaved=rope.interleaved)
    """

    raw_freqs_t: Tensor
    raw_freqs_h: Tensor
    raw_freqs_w: Tensor
    freqs_t: Tensor
    freqs_h: Tensor
    freqs_w: Tensor

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
        super().__init__(
            head_dim=head_dim,
            len_h=len_h,
            len_w=len_w,
            len_t=len_t,
            h_extrapolation_ratio=h_extrapolation_ratio,
            w_extrapolation_ratio=w_extrapolation_ratio,
            t_extrapolation_ratio=t_extrapolation_ratio,
            interleaved=interleaved,
            device=device,
        )
        self.freqs_t, self.freqs_h, self.freqs_w = self._freq_components_for_len(len_t)

    def shift_t(self, autoregressive_index: int) -> Tensor:
        """Shift the time dimension by ``autoregressive_index`` chunks.

        The internal offset is ``autoregressive_index * len_t`` so callers
        only need to track the AR step, not the per-chunk temporal length.
        If context parallelism is enabled with :meth:`set_context_parallel_group`,
        the returned frequencies are the local CP shard along sequence dim 0.

        Args:
            autoregressive_index: AR step index for the chunk being processed.
                Step 0 returns the unshifted frequencies.

        Returns:
            Concatenated RoPE frequencies of shape ``[L, 1, 1, head_dim]``,
            where L is the sequence length T * H * W. The memory layout is (T, H, W).
        """
        offset = autoregressive_index * self.len_t
        freqs_t = self.freqs_t + offset * self.raw_freqs_t
        freqs = self._cat_freqs(freqs_t, self.freqs_h, self.freqs_w)
        if self.is_context_parallel_enabled():
            freqs = split_inputs_cp(
                freqs,
                seq_dim=0,
                cp_group=unpack_optional(self.cp_group),
            )
        return freqs


class KVCacheRelativeRotaryPositionEmbedding3D(_RotaryPositionEmbedding3DBase):
    """3D RoPE with bounded KV-cache-relative positions.

    Positions are reassigned every step to where each token currently sits in
    the KV cache. This must be paired with storing K without standard RoPE
    before the cache write and rotating cached K on read in the attention module.
    Use it for bounded sink/window caches where older tokens move through cache
    slots instead of retaining monotonically increasing global positions.

    ``shift_t()`` intentionally ignores ``autoregressive_index``: the returned
    frequencies describe KV-cache slots, not global AR time. The frequency
    tensor length is based on ``sink_size_t + window_size_t`` and therefore
    remains bounded even as generation continues.

    Args:
        head_dim: Attention head dimension. Must be compatible with the fused
            RoPE kernel and is split across time, height, and width.
        len_h: Number of patch tokens along height in one chunk.
        len_w: Number of patch tokens along width in one chunk.
        len_t: Number of temporal patch tokens in one autoregressive chunk.
        sink_size_t: Number of temporal cache positions kept as fixed sink
            tokens.
        window_size_t: Number of temporal cache positions kept as the rolling
            window. ``sink_size_t + window_size_t`` must be divisible by
            ``len_t`` so CP can split cache chunks consistently.
        h_extrapolation_ratio: NTK extrapolation ratio for height frequencies.
        w_extrapolation_ratio: NTK extrapolation ratio for width frequencies.
        t_extrapolation_ratio: NTK extrapolation ratio for time frequencies.
        interleaved: Whether RoPE pairs are stored as ``(2k, 2k+1)`` instead
            of ``(k, k + D/2)``.
        device: Device where frequency buffers are allocated.

    Attributes:
        raw_freqs_t: Base temporal RoPE frequency components before expansion,
            shape ``[dim_t // 2]``.
        raw_freqs_h: Base height RoPE frequency components before expansion,
            shape ``[dim_h // 2]``.
        raw_freqs_w: Base width RoPE frequency components before expansion,
            shape ``[dim_w // 2]``.
        freqs_t: Expanded temporal frequency components for all KV-cache slots,
            shape ``[S, 1, 1, dim_t // 2]``.
        freqs_h: Expanded height frequency components for all KV-cache slots,
            shape ``[S, 1, 1, dim_h // 2]``.
        freqs_w: Expanded width frequency components for all KV-cache slots,
            shape ``[S, 1, 1, dim_w // 2]``.

    Examples:

        Store unrotated K in the KV cache, then rotate a materialized cache
        read with cache-slot positions before attention:

        >>> rope = KVCacheRelativeRotaryPositionEmbedding3D(
        ...     head_dim=128,
        ...     len_t=3,
        ...     len_h=60,
        ...     len_w=104,
        ...     sink_size_t=5,
        ...     window_size_t=7,
        ...     interleaved=True,
        ... )
        >>> freqs = rope.shift_t(autoregressive_index=25)
        >>> freqs.shape
        torch.Size([74880, 1, 1, 128])
        >>> cached_k = apply_rope_freqs(
        ...     cached_k,
        ...     freqs,
        ...     interleaved=rope.interleaved,
        ... )
    """

    raw_freqs_t: Tensor
    raw_freqs_h: Tensor
    raw_freqs_w: Tensor
    freqs_t: Tensor
    freqs_h: Tensor
    freqs_w: Tensor

    def __init__(
        self,
        head_dim: int,
        len_h: int,
        len_w: int,
        len_t: int,
        sink_size_t: int,
        window_size_t: int,
        h_extrapolation_ratio: float = 1.0,
        w_extrapolation_ratio: float = 1.0,
        t_extrapolation_ratio: float = 1.0,
        interleaved: bool = False,
        device: torch.device = torch.device("cuda"),
    ) -> None:
        assert sink_size_t >= 0, "sink_size_t must be non-negative"
        assert window_size_t > 0, "window_size_t must be positive"
        self.sink_size_t = sink_size_t
        self.window_size_t = window_size_t
        self.kvcache_total_size_t = self.sink_size_t + self.window_size_t
        super().__init__(
            head_dim=head_dim,
            len_h=len_h,
            len_w=len_w,
            len_t=len_t,
            h_extrapolation_ratio=h_extrapolation_ratio,
            w_extrapolation_ratio=w_extrapolation_ratio,
            t_extrapolation_ratio=t_extrapolation_ratio,
            interleaved=interleaved,
            device=device,
        )
        assert self.kvcache_total_size_t % self.len_t == 0, (
            "sink_size_t + window_size_t "
            f"({self.kvcache_total_size_t}) must be divisible by len_t ({self.len_t})"
        )
        self.freqs_t, self.freqs_h, self.freqs_w = self._freq_components_for_len(
            self.kvcache_total_size_t
        )
        self._rope_freqs = self._cat_freqs(self.freqs_t, self.freqs_h, self.freqs_w)
        self._rope_freqs_cp: Tensor | None = None

    def _split_cache_freqs_cp(self, freqs: Tensor, valid_len_t: int) -> Tensor:
        """Split cache-relative RoPE frequencies chunk-by-chunk for CP."""
        cp_group = unpack_optional(self.cp_group)
        tokens_per_chunk = self.len_t * self.len_h * self.len_w
        assert valid_len_t % self.len_t == 0
        valid_tokens = valid_len_t * self.len_h * self.len_w
        freqs = freqs[:valid_tokens]
        freq_shape = freqs.shape[1:]
        freqs = freqs.reshape(valid_len_t // self.len_t, tokens_per_chunk, *freq_shape)
        freqs = split_inputs_cp(freqs, seq_dim=1, cp_group=cp_group)
        return freqs.reshape(-1, *freq_shape)

    def set_context_parallel_group(self, cp_group: ProcessGroup | None) -> None:
        super().set_context_parallel_group(cp_group)
        self._rope_freqs_cp = (
            None
            if cp_group is None
            else self._split_cache_freqs_cp(self._rope_freqs, self.kvcache_total_size_t)
        )

    def shift_t(self, autoregressive_index: int) -> Tensor:
        """Return fixed KV-cache-relative RoPE frequencies.

        The returned tensor covers all valid KV-cache slots, ordered as
        ``(T, H, W)``. If context parallelism is enabled with
        :meth:`set_context_parallel_group`, cache chunks are split independently
        so each rank receives the local sequence shard for every cache chunk.

        Args:
            autoregressive_index: Accepted for API parity with standard RoPE.
                Ignored because cache-relative positions are fixed by KV-cache
                position, not by global AR step.

        Returns:
            RoPE frequencies of shape ``[S, 1, 1, head_dim]``, where ``S`` is
            the valid KV-cache token count after optional CP splitting.
        """
        if self.is_context_parallel_enabled():
            if self._rope_freqs_cp is not None:
                return self._rope_freqs_cp
            return self._split_cache_freqs_cp(
                self._rope_freqs, self.kvcache_total_size_t
            )
        return self._rope_freqs


def apply_rope_freqs(x: Tensor, freqs: Tensor, interleaved: bool = False) -> Tensor:
    """Apply RoPE frequencies to ``x`` in place via the fused Triton kernel.

    Writes back in place because every call site passes a freshly
    materialised Q or K — there is no autograd graph to preserve.

    Args:
        x: Input tensor of shape ``[B, S, H, D]``; rotated in place.
        freqs: RoPE frequencies of shape ``[S, 1, 1, D]`` as emitted by
            :meth:`RotaryPositionEmbedding3D.shift_t` or
            :meth:`KVCacheRelativeRotaryPositionEmbedding3D.shift_t`.
        interleaved: If ``True``, rotate the pair ``(2k, 2k+1)``; else
            rotate ``(d, d + D/2)``.

    Returns:
        Rotated tensor of shape ``[B, S, H, D]``, sharing storage with ``x``.
    """
    return apply_rotary_pos_emb(x, freqs, interleaved=interleaved, inplace=True)
