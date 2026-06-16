# SPDX-FileCopyrightText: Copyright (c) 2026 Hongyu Zhou
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
"""DualChunkKVCache — rolling-window KV cache for video-action transformers.

Supports two sequential writes per AR step (video tokens then action tokens)
with fully static shapes suitable for ``torch.compile``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor


@dataclass
class DualChunkKVCache:
    """Rolling-window KV cache with two write slots per AR step.

    Physical layout per slot: ``[primary_tokens | secondary_tokens]``
    Full buffer: ``[slot_0 | slot_1 | ... | slot_{window-1}]``

    All operations use fixed-offset slicing — no dynamic indexing,
    ``nonzero``, or data-dependent control flow.
    """

    primary_chunk: int
    secondary_chunk: int
    window_slots: int
    batch_size: int
    num_heads: int
    head_dim: int
    device: torch.device
    dtype: torch.dtype

    _k: Tensor = field(init=False)
    _v: Tensor = field(init=False)
    _n_committed: int = field(init=False, default=0)
    _primary_written: bool = field(init=False, default=False)

    def __post_init__(self) -> None:
        total = self.window_slots * self.slot_size
        shape = (self.batch_size, total, self.num_heads, self.head_dim)
        self._k = torch.zeros(shape, device=self.device, dtype=self.dtype)
        self._v = torch.zeros(shape, device=self.device, dtype=self.dtype)

    @property
    def slot_size(self) -> int:
        return self.primary_chunk + self.secondary_chunk

    @property
    def total_capacity(self) -> int:
        return self.window_slots * self.slot_size

    @property
    def n_cached_tokens(self) -> int:
        """Number of committed tokens visible to attention."""
        return self._n_committed * self.slot_size

    @property
    def _is_full(self) -> bool:
        """Whether the buffer has reached capacity."""
        return self._n_committed >= self.window_slots

    def _write_offset(self) -> int:
        """Start position for the current (uncommitted) slot.

        In filling phase, appends after the last committed slot.
        In steady-state (full), always writes to the last slot position
        (after the left-shift in ``commit_slot`` made room).
        """
        if self._is_full:
            return (self.window_slots - 1) * self.slot_size
        return self._n_committed * self.slot_size

    def _roll_left(self) -> None:
        """Shift the buffer left by one slot, discarding the oldest."""
        tokens_to_keep = (self.window_slots - 1) * self.slot_size
        self._k[:, :tokens_to_keep] = self._k[:, self.slot_size : self.slot_size + tokens_to_keep].clone()
        self._v[:, :tokens_to_keep] = self._v[:, self.slot_size : self.slot_size + tokens_to_keep].clone()

    def write_primary(self, k: Tensor, v: Tensor) -> None:
        """Write primary (video) KV into the current slot.

        If the buffer is full, shifts left first to evict the oldest slot.

        Args:
            k: Shape ``[batch, primary_chunk, heads, head_dim]``.
            v: Same shape as ``k``.
        """
        if self._is_full:
            self._roll_left()
        pos = self._write_offset()
        self._k[:, pos : pos + self.primary_chunk] = k
        self._v[:, pos : pos + self.primary_chunk] = v
        self._primary_written = True

    def write_secondary(self, k: Tensor, v: Tensor) -> None:
        """Write secondary (action) KV into the current slot.

        Must be called after ``write_primary`` within the same slot.

        Args:
            k: Shape ``[batch, secondary_chunk, heads, head_dim]``.
            v: Same shape as ``k``.
        """
        assert self._primary_written
        pos = self._write_offset() + self.primary_chunk
        self._k[:, pos : pos + self.secondary_chunk] = k
        self._v[:, pos : pos + self.secondary_chunk] = v

    def commit_slot(self) -> None:
        """Finalize the current slot and advance the write pointer.

        In filling phase, increments the committed count. In steady-state
        the count stays at ``window_slots`` (the oldest slot was already
        evicted by ``_roll_left`` in ``write_primary``).
        """
        assert self._primary_written
        if not self._is_full:
            self._n_committed += 1
        self._primary_written = False

    def cached_k(self) -> Tensor:
        """Return all committed K tokens in temporal order."""
        n = self.n_cached_tokens
        return self._k[:, :n]

    def cached_v(self) -> Tensor:
        """Return all committed V tokens in temporal order."""
        n = self.n_cached_tokens
        return self._v[:, :n]

    def cached_kv_plus_fresh(
        self, k_fresh: Tensor, v_fresh: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Concatenate committed cache with fresh (uncommitted) tokens.

        Used for read-only attention during intermediate denoising steps.

        Args:
            k_fresh: Shape ``[batch, L_fresh, heads, head_dim]``.
            v_fresh: Same shape as ``k_fresh``.

        Returns:
            ``(full_k, full_v)`` each of shape
            ``[batch, n_cached + L_fresh, heads, head_dim]``.
        """
        return (
            torch.cat([self.cached_k(), k_fresh], dim=1),
            torch.cat([self.cached_v(), v_fresh], dim=1),
        )

    def reset(self) -> None:
        """Release GPU memory and reset state."""
        self._k = torch.empty(0)
        self._v = torch.empty(0)
        self._n_committed = 0
        self._primary_written = False
