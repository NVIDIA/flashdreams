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

"""Block KV cache for causal attention with a fixed-size local window."""

from dataclasses import dataclass, field
from typing import NamedTuple

import torch
from torch import Tensor
from typing_extensions import Self


class KVRange(NamedTuple):
    """Branchless KV-cache write/read pair for a single AR step.

    ``write_start`` is the per-AR-step in-place write offset for
    :meth:`RollingBlockKVCache.update_at`; ``valid_len`` is the post-update
    read length for :meth:`BlockKVCache.cached_k_at` /
    :meth:`BlockKVCache.cached_v_at`. Both are precomputed in
    :meth:`RollingBlockKVCache.before_update` and travel together through every
    cross-layer forward (network -> block -> self-attn). Bundled into a
    NamedTuple to keep cross-layer signatures concise; unpacked at the leaf
    APIs, which each consume only one of the two ints.
    """

    write_start: int
    valid_len: int


@dataclass(kw_only=True)
class BlockKVCache:
    """Storage + branchless-read backbone shared by the two concrete caches.

    Holds the K/V buffers (allocated in :meth:`__post_init__`) and the read API
    (:meth:`cached_k_at` / :meth:`cached_v_at`) common to:

    - :class:`RollingBlockKVCache` -- a self-contained fixed-size local-window
      cache that rolls left as new chunks arrive (self-attention).
    - :class:`PrefixBlockKVCache` -- an immutable one-shot prefix cache
      (cross-attention text / image embeddings).

    Keys and values can have arbitrary shape ``[..., total_size, ...]``; the
    sequence dimension is :attr:`seq_dim` (a normalized non-negative index). The
    base never mutates the buffers -- all writes live on
    :class:`RollingBlockKVCache`. This class is abstract; do not instantiate it
    directly.
    """

    k_shape: tuple[int, ...]
    """Shape of the keys. Must be the same as the values shape except for the last dimension."""

    v_shape: tuple[int, ...]
    """Shape of the values. Must be the same as the keys shape except for the last dimension."""

    seq_dim: int
    """Sequence-dimension index in the K/V tensor; may be negative (normalized
    to a non-negative index in :meth:`__post_init__`).

    A rollout-constant read on the hot path (:meth:`_seq_slice` /
    :meth:`cached_k_at`), so Dynamo constant-folds it in the traced region."""

    device: torch.device | str = torch.device("cuda")
    """Device to store the cache on."""

    dtype: torch.dtype = torch.float16
    """Data type to store the cache in."""

    _k: Tensor = field(init=False)
    """Cached keys. shape ``[..., total_size, ..., Dk]`` at ``seq_dim``."""

    _v: Tensor = field(init=False)
    """Cached values. shape ``[..., total_size, ..., Dv]`` at ``seq_dim``."""

    def __post_init__(self) -> None:
        assert self.k_shape[:-1] == self.v_shape[:-1], (
            "k and v must have the same shape except for the last dimension"
        )
        tensor_dim = len(self.k_shape)
        assert -tensor_dim <= self.seq_dim < tensor_dim, (
            f"seq_dim must be in [-{tensor_dim}, {tensor_dim}), got {self.seq_dim}"
        )
        # Normalize seq_dim to a non-negative index so downstream
        # indexing math doesn't have to special-case negatives.
        self.seq_dim = self.seq_dim if self.seq_dim >= 0 else self.seq_dim + tensor_dim
        self._k = torch.empty(self.k_shape, device=self.device, dtype=self.dtype)
        self._v = torch.empty(self.v_shape, device=self.device, dtype=self.dtype)

    def _seq_slice(self, start: int | None, end: int | None) -> tuple[slice | int, ...]:
        """Return an index tuple selecting ``[start:end]`` on ``seq_dim`` and all elements elsewhere."""
        idx: list[slice | int] = [slice(None)] * len(self.k_shape)
        idx[self.seq_dim] = slice(start, end)
        return tuple(idx)

    def cached_k_at(self, valid_len: int) -> Tensor:
        """Return cached keys sliced to ``[:valid_len]`` on ``seq_dim``."""
        return self._k[self._seq_slice(0, valid_len)]

    def cached_v_at(self, valid_len: int) -> Tensor:
        """Return cached values sliced to ``[:valid_len]`` on ``seq_dim``."""
        return self._v[self._seq_slice(0, valid_len)]


@dataclass(kw_only=True)
class RollingBlockKVCache(BlockKVCache):
    """Self-contained fixed-size local-window KV cache with CUDA-graph support.

    Owns both its layout (``chunk_size`` / ``window_size`` / ``sink_size``) and
    its rolling cursor. Layout along ``seq_dim``: [sink tokens | local window
    tokens]; sink tokens are never evicted and the local window rolls left by
    ``chunk_size`` once full. Chunks are non-overlapping: each update writes one
    chunk at the next logical position. ``total_size``
    (``sink_size + window_size``) must be divisible by ``chunk_size``.

    Per-step usage (driven by the owning ``*TransformerCache`` cascade):
        1. ``cache.before_update(chunk_idx)`` -- advance the cursor (precompute
           :attr:`write_start` / :attr:`valid_len`) and roll the buffer left on a
           steady-state advance. Runs **outside** the compiled forward.
        2. ``cache.update_at(k, v, cache.write_start)`` -- branchless in-graph write.
        3. ``cache.cached_k_at(cache.valid_len)`` / ``cached_v_at(...)`` -- read.
        4. ``cache.after_update(chunk_idx)`` -- finalize the cursor.

    Every self-attn cache in a DiT (and the cond/uncond CFG twins) shares the
    same geometry and chunk sequence, so they advance in lock-step. There is no
    shared lifecycle object; the owner reads the threaded :class:`KVRange` from
    the first cache (``block_caches[0].self_attn``). The branchless leaf
    (:meth:`update_at` / :meth:`cached_k_at`) only ever sees the two precomputed
    ints, keeping the traced graph ``_n_cached``-agnostic.
    """

    chunk_size: int
    """Number of tokens written per AR step."""

    window_size: int
    """Size of the rolling local-attention window (excludes sink tokens)."""

    sink_size: int = 0
    """Number of leading sink tokens that are never evicted."""

    # ---- rolling cursor (advanced outside the compiled forward) ----
    _n_cached: int = field(init=False, default=0)
    _curr_chunk_idx: int | None = field(init=False, default=None)
    _prev_chunk_idx: int = field(init=False, default=-1)
    _needs_buffer_roll: bool = field(init=False, default=False)

    write_start: int = field(init=False, default=0)
    """Branchless write offset for the current AR step (set by :meth:`before_update`)."""

    valid_len: int = field(init=False, default=0)
    """Post-update read length for the current AR step (set by :meth:`before_update`)."""

    @property
    def total_size(self) -> int:
        """``sink_size + window_size``; the physical length of the cache buffer."""
        return self.sink_size + self.window_size

    @property
    def range(self) -> KVRange:
        """Branchless write/read pair for the current AR step.

        Cross-layer plumbing (TransformerCache -> network -> block -> self-attn)
        takes a single :class:`KVRange`; the leaf APIs consume the unpacked
        fields. Valid after :meth:`before_update`.
        """
        return KVRange(write_start=self.write_start, valid_len=self.valid_len)

    def __post_init__(self) -> None:
        super().__post_init__()
        assert self.chunk_size > 0, "chunk_size must be positive"
        assert self.window_size > 0, "window_size must be positive"
        assert self.sink_size >= 0, "sink_size must be non-negative"
        assert self.total_size % self.chunk_size == 0, (
            f"total_size ({self.total_size}) must be divisible by "
            f"chunk_size ({self.chunk_size})"
        )
        # Branchless steady-state write requires the rightmost slot
        # (``total_size - chunk_size``) to fall outside the immutable sink
        # prefix; equivalently ``chunk_size <= window_size``.
        assert self.chunk_size <= self.window_size, (
            f"chunk_size ({self.chunk_size}) must be <= window_size "
            f"({self.window_size}) for branchless steady-state write."
        )
        assert self.k_shape[self.seq_dim] == self.total_size, (
            f"k_shape[{self.seq_dim}] ({self.k_shape[self.seq_dim]}) must equal "
            f"total_size ({self.total_size})"
        )

    # ---- cursor advance (Python, outside the compiled forward) ----

    def is_steady_state(self) -> bool:
        """True if the rolling window is full (post-fill phase).

        Must be called after :meth:`before_update`, i.e. with ``_curr_chunk_idx``
        set.
        """
        assert self._curr_chunk_idx is not None, (
            "Must call before_update() before is_steady_state()"
        )
        is_full = self.total_size == self._n_cached
        is_overlapping_with_sink = (
            self.sink_size > 0
            and self._curr_chunk_idx * self.chunk_size < self.sink_size
        )
        return is_full and not is_overlapping_with_sink

    def before_update(self, chunk_idx: int) -> None:
        """Advance the cursor for ``chunk_idx`` and roll the buffer if steady-state.

        Idempotent within an AR step. Precomputes :attr:`write_start` /
        :attr:`valid_len` and, on a steady-state advance, rolls the local window
        left. Runs outside the compiled forward (the in-graph leaf only reads the
        precomputed ints).

        - First call with a new ``chunk_idx``: advances ``_curr_chunk_idx``,
          precomputes the branchless params, and rolls on a steady-state advance.
        - Subsequent calls with the same ``chunk_idx`` (a residual double-call):
          no-op (the early return preserves "advance once").
        - Repeating the previous ``chunk_idx`` (multi-step scheduler reuse):
          recomputes the params for the reuse slot, no roll.
        """
        if self._curr_chunk_idx == chunk_idx:
            return
        assert self._curr_chunk_idx is None, (
            "Must call after_update() before before_update() for a new chunk_idx; "
            f"got chunk_idx={chunk_idx}, _curr_chunk_idx={self._curr_chunk_idx}"
        )
        self._curr_chunk_idx = chunk_idx
        self._needs_buffer_roll = False

        if chunk_idx == self._prev_chunk_idx + 1:
            if self.is_steady_state():
                self._needs_buffer_roll = True
        elif chunk_idx == self._prev_chunk_idx:
            # Multi-step scheduler reuses the same chunk: no buffer roll,
            # ``_n_cached`` will not advance in ``after_update``, but write_start
            # / valid_len need recomputing for the reuse-slot semantics
            # (write_start rewinds onto the just-written slot rather than
            # appending).
            pass
        else:
            raise AssertionError(
                "Expected the new chunk_idx to be either +1 from the previous "
                "chunk_idx (advance) or equal to it (multi-step scheduler reuse), "
                f"got {chunk_idx} vs _prev_chunk_idx={self._prev_chunk_idx}"
            )

        self.write_start = self._compute_write_start_now()
        self.valid_len = self._compute_valid_len_now()
        if self._needs_buffer_roll:
            self._roll_local_window_left()

    def after_update(self, chunk_idx: int) -> None:
        """Finalize cursor bookkeeping for ``chunk_idx``. Idempotent within an AR step.

        - First call with ``chunk_idx == _curr_chunk_idx``: advances
          ``_n_cached`` (if filling) and ``_prev_chunk_idx``, clears
          ``_curr_chunk_idx``.
        - Subsequent calls: no-op (asserts the cursor was already finalized).
        """
        if self._curr_chunk_idx is None:
            assert chunk_idx == self._prev_chunk_idx, (
                f"after_update({chunk_idx}) re-entry expected _prev_chunk_idx="
                f"{chunk_idx}, got {self._prev_chunk_idx}"
            )
            return

        assert chunk_idx == self._curr_chunk_idx, (
            f"Expected chunk_idx to be {self._curr_chunk_idx}, got {chunk_idx}"
        )

        if self._curr_chunk_idx == self._prev_chunk_idx + 1:
            if not self.is_steady_state():
                self._n_cached += self.chunk_size
            self._prev_chunk_idx += 1
        elif self._curr_chunk_idx == self._prev_chunk_idx:
            pass
        else:
            raise ValueError(
                f"{self._curr_chunk_idx=} should be either "
                f"{self._prev_chunk_idx + 1} or {self._prev_chunk_idx}."
            )

        self._curr_chunk_idx = None

    def size(self) -> int:
        """Number of valid cached tokens visible to attention.

        Before any ``before_update`` (``_curr_chunk_idx is None``): returns the
        count from the previous AR step. After ``before_update``: returns the
        precomputed :attr:`valid_len`.
        """
        if self._curr_chunk_idx is None:
            return self._n_cached
        return self.valid_len

    def _compute_write_start_now(self) -> int:
        """Branchless write offset from current cursor state.

        * steady state: rightmost slot, ``total_size - chunk_size``.
        * filling, advancing chunk (``_curr == _prev + 1``): append at ``_n_cached``.
        * filling, same chunk (``_curr == _prev``): overwrite the just-written
          rightmost slot at ``_n_cached - chunk_size``.
        """
        if self.is_steady_state():
            return int(self.total_size - self.chunk_size)
        if self._curr_chunk_idx == self._prev_chunk_idx + 1:
            return int(self._n_cached)
        if self._curr_chunk_idx == self._prev_chunk_idx:
            return int(self._n_cached - self.chunk_size)
        raise RuntimeError(
            f"Unexpected cache state: _curr_chunk_idx={self._curr_chunk_idx} "
            f"vs _prev_chunk_idx={self._prev_chunk_idx}"
        )

    def _compute_valid_len_now(self) -> int:
        """Post-update valid-token count from current cursor state."""
        if self.is_steady_state():
            return int(self.total_size)
        if self._curr_chunk_idx == self._prev_chunk_idx + 1:
            return int(self._n_cached + self.chunk_size)
        if self._curr_chunk_idx == self._prev_chunk_idx:
            return int(self._n_cached)
        raise RuntimeError(
            f"Unexpected cache state: _curr_chunk_idx={self._curr_chunk_idx} "
            f"vs _prev_chunk_idx={self._prev_chunk_idx}"
        )

    def _roll_local_window_left(self) -> None:
        """Shift the local window left by chunk_size tokens (steady-state only).

        Called by :meth:`before_update` on a steady-state advance; rolls only
        this cache's private K/V buffer.
        """
        total_size = self._k.shape[self.seq_dim]
        assert total_size == self._n_cached, (
            f"Expected full cache: {total_size=} != {self._n_cached=}"
        )
        tokens_to_keep = self.window_size - self.chunk_size

        if tokens_to_keep > 0:
            src_start = self.sink_size + self.chunk_size
            src_end = total_size
            dst_start = self.sink_size
            dst_end = self.sink_size + tokens_to_keep

            dst_slice = self._seq_slice(dst_start, dst_end)
            src_slice = self._seq_slice(src_start, src_end)
            self._k[dst_slice] = self._k[src_slice].clone()
            self._v[dst_slice] = self._v[src_slice].clone()

    # ---- branchless write API ----
    #
    # All data-dependent control flow lives in :meth:`before_update` (Python,
    # outside the compiled forward). The traced region only sees the
    # ``write_start`` int, keeping the graph ``_n_cached``-agnostic so Dynamo
    # never guards on (or recompiles because of) the filling/steady-state phase.

    def update_at(self, k: Tensor, v: Tensor, write_start: int) -> None:
        """Branchless in-place write at ``[write_start, write_start + chunk_size)``.

        ``write_start`` is precomputed by :meth:`before_update` (read via
        :attr:`write_start`). No asserts and no ``is_steady_state`` branches, so
        Dynamo never derives an ``_n_cached`` upper bound from this path.
        ``seq_dim`` / ``chunk_size`` are rollout-constants.
        """
        seq_dim = self.seq_dim
        chunk_size = self.chunk_size
        self._k.narrow(seq_dim, write_start, chunk_size).copy_(k)
        self._v.narrow(seq_dim, write_start, chunk_size).copy_(v)


@dataclass(kw_only=True)
class PrefixBlockKVCache(BlockKVCache):
    """Immutable one-shot prefix cache (cross-attention text / image embeddings).

    Filled once from a tensor via :meth:`from_tensor`; the whole input is the
    valid region. There is no cursor and no rolling: readers use :attr:`valid_len`
    (== the input length along ``seq_dim``) or the constant :attr:`range`.
    """

    @property
    def valid_len(self) -> int:
        """Number of valid cached tokens: the full prefix length along ``seq_dim``."""
        return self.k_shape[self.seq_dim]

    @property
    def range(self) -> KVRange:
        """Constant branchless read pair for the whole prefix (``write_start=0``).

        Lets cross-attention reuse the same ``apply_kv(..., kv_range=...)`` API
        as self-attention; only ``valid_len`` is consumed there.
        """
        return KVRange(write_start=0, valid_len=self.valid_len)

    @classmethod
    def from_tensor(cls, k: Tensor, v: Tensor, seq_dim: int) -> Self:
        """Build a prefix cache filled with ``k`` / ``v`` along ``seq_dim``.

        Used for immutable cross-attention prefix caches (text / image
        embeddings). ``seq_dim`` is normalized against ``k.ndim`` here (the cache
        stores a non-negative index). After construction, callers read via
        :meth:`cached_k_at` / :meth:`cached_v_at` using :attr:`valid_len` (which
        equals the input length).
        """
        tensor_dim = k.ndim
        assert -tensor_dim <= seq_dim < tensor_dim, (
            f"seq_dim must be in [-{tensor_dim}, {tensor_dim}), got {seq_dim}"
        )
        norm_seq_dim = seq_dim if seq_dim >= 0 else seq_dim + tensor_dim
        cache = cls(
            k_shape=tuple(k.shape),
            v_shape=tuple(v.shape),
            seq_dim=norm_seq_dim,
            device=k.device,
            dtype=k.dtype,
        )
        cache._k.copy_(k)
        cache._v.copy_(v)
        return cache
