# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""MemRoPE KV cache built on top of the base block KV cache."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from flashdreams.core.attention.kvcache import BlockKVCache


@dataclass
class MemRoPEKVCache(BlockKVCache):
    """
    KV cache with MemRoPE-style memory slots.

    The cache stores raw, unrotated keys. Its steady-state token layout is:
    [sink | long EMA memory | short EMA memory | recent tokens | current chunk].
    The two memory slots are one frame each; ``recent_size`` and
    ``chunk_size`` are token counts.
    """

    frame_size: int = 0
    """Number of tokens per latent frame after patchify and CP split."""

    recent_size: int = 0
    """Number of non-memory recent tokens preserved before the current chunk."""

    memory_frames: int = 2
    """Number of one-frame EMA memory slots."""

    ema_alpha_long: float = 0.01
    """EMA update rate for the long-term memory slot."""

    ema_alpha_short: float = 0.1
    """EMA update rate for the short-term memory slot."""

    _memory_initialized: bool = False
    """Whether the EMA memory slots have received their first compressed update."""

    def __post_init__(self) -> None:
        super().__post_init__()

        assert self.frame_size > 0, "frame_size must be positive"
        assert self.recent_size >= 0, "recent_size must be non-negative"
        assert self.chunk_size % self.frame_size == 0, (
            "chunk_size must contain a whole number of latent frames"
        )
        assert self.sink_size % self.frame_size == 0, (
            "sink_size must contain a whole number of latent frames"
        )
        assert self.recent_size % self.frame_size == 0, (
            "recent_size must contain a whole number of latent frames"
        )
        assert self.memory_frames in (0, 2), (
            "memory_frames must be 0 or 2 for the current MemRoPE layout"
        )

        memory_size = self.memory_size
        assert self.window_size == memory_size + self.recent_size + self.chunk_size, (
            "MemRoPE window_size must equal memory_size + recent_size + chunk_size"
        )

    @property
    def memory_size(self) -> int:
        """Total token count for memory slots."""
        return self.memory_frames * self.frame_size

    @property
    def total_size(self) -> int:
        """Total fixed cache capacity in tokens."""
        return self.sink_size + self.window_size

    @property
    def chunk_frames(self) -> int:
        """Number of latent frames in the current AR chunk."""
        return self.chunk_size // self.frame_size

    def _slice_tensor(self, x: Tensor, start: int | None, end: int | None) -> Tensor:
        idx: list[slice | int] = [slice(None)] * x.ndim
        idx[self.seq_dim] = slice(start, end)
        return x[tuple(idx)]

    def _assign_cache(self, target: Tensor, start: int, end: int, source: Tensor) -> None:
        target[self._seq_slice(start, end)] = source

    def _memory_frame_mean(self, x: Tensor) -> Tensor:
        summary = x.mean(dim=self.seq_dim, keepdim=True)
        shape = list(summary.shape)
        shape[self.seq_dim] = self.frame_size
        return summary.expand(tuple(shape)).clone()

    def _ema_update(self, old: Tensor, new: Tensor, alpha: float) -> Tensor:
        return alpha * new + (1.0 - alpha) * old

    def _compress_and_append(self, k: Tensor, v: Tensor) -> None:
        old_k = self._slice_tensor(self._k, 0, self._n_cached).clone()
        old_v = self._slice_tensor(self._v, 0, self._n_cached).clone()

        new_k = torch.empty_like(self._k)
        new_v = torch.empty_like(self._v)

        if self.sink_size > 0:
            self._assign_cache(
                new_k,
                0,
                self.sink_size,
                self._slice_tensor(old_k, 0, self.sink_size),
            )
            self._assign_cache(
                new_v,
                0,
                self.sink_size,
                self._slice_tensor(old_v, 0, self.sink_size),
            )

        # MemRoPE reserves the two slots after sink for EMA memory even before
        # the first compression; they are overwritten when EMA initializes.
        old_tail_start = self.sink_size + self.memory_size
        old_tail_len = self._n_cached - old_tail_start
        retain_len = min(self.recent_size, old_tail_len)
        evict_end = self._n_cached - retain_len

        evicted_k = self._slice_tensor(old_k, old_tail_start, evict_end)
        evicted_v = self._slice_tensor(old_v, old_tail_start, evict_end)
        retained_k = self._slice_tensor(old_k, evict_end, self._n_cached)
        retained_v = self._slice_tensor(old_v, evict_end, self._n_cached)

        recent_start = self.sink_size
        if self.memory_frames == 2:
            long_start = self.sink_size
            long_end = long_start + self.frame_size
            short_start = long_end
            short_end = short_start + self.frame_size

            if evicted_k.shape[self.seq_dim] > 0:
                summary_k = self._memory_frame_mean(evicted_k)
                summary_v = self._memory_frame_mean(evicted_v)
            else:
                summary_k = self._slice_tensor(
                    old_k, old_tail_start, old_tail_start + self.frame_size
                ).clone()
                summary_v = self._slice_tensor(
                    old_v, old_tail_start, old_tail_start + self.frame_size
                ).clone()

            if self._memory_initialized:
                old_long_k = self._slice_tensor(old_k, long_start, long_end).clone()
                old_long_v = self._slice_tensor(old_v, long_start, long_end).clone()
                old_short_k = self._slice_tensor(old_k, short_start, short_end).clone()
                old_short_v = self._slice_tensor(old_v, short_start, short_end).clone()
                long_k = self._ema_update(old_long_k, summary_k, self.ema_alpha_long)
                long_v = self._ema_update(old_long_v, summary_v, self.ema_alpha_long)
                short_k = self._ema_update(
                    old_short_k, summary_k, self.ema_alpha_short
                )
                short_v = self._ema_update(
                    old_short_v, summary_v, self.ema_alpha_short
                )
            else:
                long_k = summary_k
                long_v = summary_v
                short_k = summary_k
                short_v = summary_v
                self._memory_initialized = True

            self._assign_cache(new_k, long_start, long_end, long_k)
            self._assign_cache(new_v, long_start, long_end, long_v)
            self._assign_cache(new_k, short_start, short_end, short_k)
            self._assign_cache(new_v, short_start, short_end, short_v)
            recent_start = short_end

        recent_end = recent_start + retain_len
        if retain_len > 0:
            self._assign_cache(new_k, recent_start, recent_end, retained_k)
            self._assign_cache(new_v, recent_start, recent_end, retained_v)

        write_start = recent_end
        write_end = write_start + self.chunk_size
        self._assign_cache(new_k, write_start, write_end, k)
        self._assign_cache(new_v, write_start, write_end, v)

        self._k = new_k
        self._v = new_v
        self._n_cached = write_end

    def before_update(self, chunk_idx: int) -> None:
        """Mark the current chunk before writing K/V."""
        assert self._curr_chunk_idx is None, (
            "Must call after_update() before before_update()"
        )
        self._curr_chunk_idx = chunk_idx
        if chunk_idx == self._prev_chunk_idx:
            return
        assert chunk_idx == self._prev_chunk_idx + 1, (
            "Expected the new chunk_idx to be +1 from the previous chunk_idx, "
            f"got {chunk_idx} != {self._prev_chunk_idx} + 1"
        )

    def update(self, k: Tensor, v: Tensor) -> None:
        """Write raw K/V for the current chunk, compressing old tokens if needed."""
        assert self._curr_chunk_idx is not None, (
            "Must call before_update() before update()"
        )
        assert k.shape[self.seq_dim] == self.chunk_size, (
            f"Expected input k to have chunk_size {self.chunk_size}"
        )
        assert v.shape[self.seq_dim] == self.chunk_size, (
            f"Expected input v to have chunk_size {self.chunk_size}"
        )

        if self._curr_chunk_idx == self._prev_chunk_idx:
            self._overwrite_rightmost_filling(k, v)
        elif self._curr_chunk_idx == self._prev_chunk_idx + 1:
            if self._n_cached + self.chunk_size <= self.total_size:
                self._append_to_end(k, v)
                self._n_cached += self.chunk_size
            else:
                self._compress_and_append(k, v)
        else:
            raise ValueError(
                f"{self._curr_chunk_idx=} should be either "
                f"{self._prev_chunk_idx + 1} or {self._prev_chunk_idx}."
            )

    def after_update(self, chunk_idx: int) -> None:
        """Finalize bookkeeping after a MemRoPE cache update."""
        assert chunk_idx == self._curr_chunk_idx, (
            f"Expected chunk_idx to be {self._curr_chunk_idx}, got {chunk_idx}"
        )
        if self._curr_chunk_idx == self._prev_chunk_idx + 1:
            self._prev_chunk_idx += 1
        elif self._curr_chunk_idx == self._prev_chunk_idx:
            pass
        else:
            raise ValueError(
                f"{self._curr_chunk_idx=} should be either "
                f"{self._prev_chunk_idx + 1} or {self._prev_chunk_idx}."
            )
        self._curr_chunk_idx = None

    def cached_k(self) -> Tensor:
        """Return valid raw keys."""
        return self._k[self._seq_slice(0, self._n_cached)]

    def cached_v(self) -> Tensor:
        """Return valid values."""
        return self._v[self._seq_slice(0, self._n_cached)]

    def cached_frame_indices(self) -> Tensor:
        """Block-relative frame index for each cached latent frame."""
        assert self._n_cached % self.frame_size == 0, (
            "cached token count must contain whole frames"
        )
        num_frames = self._n_cached // self.frame_size
        return torch.arange(num_frames, device=self._k.device, dtype=torch.long)

    def query_frame_indices(self) -> Tensor:
        """Block-relative frame indices for the current query chunk."""
        assert self._n_cached >= self.chunk_size, "cache must include current chunk"
        start = (self._n_cached - self.chunk_size) // self.frame_size
        end = self._n_cached // self.frame_size
        return torch.arange(start, end, device=self._k.device, dtype=torch.long)

    def reset(self) -> None:
        """Reset the cache to its initial empty state."""
        super().reset()
        self._curr_chunk_idx = None
        self._memory_initialized = False
