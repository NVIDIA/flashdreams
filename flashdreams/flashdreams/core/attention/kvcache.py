from dataclasses import dataclass, field
from typing import Self

import torch
from torch import Tensor


@dataclass
class BlockKVCache:
    """
    KV cache for causal attention with a fixed-size local window and CUDA-graph support.

    Keys and values can have arbitrary shape ``[..., total_size, ...]``; the sequence
    (rolling) dimension is given by ``seq_dim`` (dimension index, can be negative).
    Layout along that dimension: [sink tokens | local window tokens]. Sink tokens are
    never evicted; the local window rolls left as new chunks are added if full.

    There are two write paths internally:

    - **Sink-aware chunk-aligned** (when ``sink_size > 0``): preserves the
      original behavior where each update writes exactly ``chunk_size``
      tokens, the rolling window is rolled left by ``chunk_size`` whenever
      the buffer is full, and ``(window_size + sink_size)`` must be a
      multiple of ``chunk_size``. New chunks that would overlap the sink
      region drop their leading tokens.
    - **Roll-and-append** (when ``sink_size == 0``): unifies filling and
      steady-state writes. Before each new chunk, evict exactly
      ``max(0, n_cached + chunk_size - window_size)`` of the oldest
      rolling tokens, then append the new chunk at the right edge.
      ``window_size`` is no longer required to be a multiple of
      ``chunk_size``; ``chunk_size <= window_size`` is required.

    Phases (descriptive only; not enforced as a state machine):
        - Filling: cache not yet full; tokens are written contiguously;
          ``cached_k()`` / ``cached_v()`` return only the valid prefix.
        - Steady-state: cache full; each new chunk triggers a left-roll of
          the local window and overwrites the rightmost positions;
          ``cached_k()`` / ``cached_v()`` return the full buffer.

    The argument ``chunk_idx`` (0, 1, 2, ...) is the index of the new chunk in the full
    sequence (not an index into the cache). If ``chunk_idx`` is greater than
    the previous one, the chunk is appended (or, in steady-state, written after
    the roll). If ``chunk_idx`` equals the previous one, the same cache positions
    are overwritten.

    Per-step usage:
        1. before_update(chunk_idx) — prepare (roll local window if steady-state).
        2. update(k, v) — write the new chunk's keys/values into the cache.
        3. cached_k() / cached_v() — get cached keys/values for attention.
        4. after_update(chunk_idx) — update internal bookkeeping.
    """

    k_shape: tuple[int, ...]
    """Shape of the keys. Must be the same as the values shape except for the last dimension."""

    v_shape: tuple[int, ...]
    """Shape of the values. Must be the same as the keys shape except for the last dimension."""

    seq_dim: int
    """Sequence dimension that will be rolled. Can be negative."""

    chunk_size: int
    """Number of tokens processed each time."""

    window_size: int
    """Size of the local attention window (excluding sink tokens)."""

    sink_size: int = 0
    """Number of sink tokens at the start of the cache that are never evicted. Defaults to 0."""

    device: torch.device | str = torch.device("cuda")
    """Device to store the cache on."""

    dtype: torch.dtype = torch.float16
    """Data type to store the cache in."""

    _prev_chunk_idx: int = -1
    """Chunk index of the last written chunk; -1 when empty."""

    _curr_chunk_idx: int | None = None
    """The index of the current chunk that is being processed. None when empty."""

    _n_cached: int = 0
    """Number of valid tokens currently in the cache."""

    _k: Tensor = field(init=False)
    """Cached keys. shape ``[..., total_size, ..., Dk]``, where the ``total_size`` is the length of the cache buffer at ``seq_dim`` dimension."""

    _v: Tensor = field(init=False)
    """Cached values. shape ``[..., total_size, ..., Dv]``, where the ``total_size`` is the length of the cache buffer at ``seq_dim`` dimension."""

    _last_write_start: int = 0
    """Start index of the most recent successful write along ``seq_dim``.
    Re-used by same-``chunk_idx`` overwrites in the roll-and-append path."""

    _last_write_end: int = 0
    """End index of the most recent successful write along ``seq_dim``."""

    @classmethod
    def from_tensor(cls, k: Tensor, v: Tensor, seq_dim: int) -> Self:
        cache = cls(
            k_shape=k.shape,
            v_shape=v.shape,
            seq_dim=seq_dim,
            chunk_size=k.shape[seq_dim],
            window_size=k.shape[seq_dim],
            device=k.device,
            dtype=k.dtype,
        )
        cache.before_update(0)
        cache.update(k, v)
        cache.after_update(0)
        cache._curr_chunk_idx = 0
        return cache

    def __post_init__(self) -> None:
        # k and v should have the same shape except for the last dimension
        assert self.k_shape[:-1] == self.v_shape[:-1], (
            "k and v must have the same shape except for the last dimension"
        )

        # update seq_dim to be positive
        tensor_dim = len(self.k_shape)
        assert -tensor_dim <= self.seq_dim < tensor_dim, (
            f"seq_dim must be in [-{tensor_dim}, {tensor_dim}), got {self.seq_dim}"
        )
        self.seq_dim = self.seq_dim if self.seq_dim >= 0 else self.seq_dim + tensor_dim

        # check non-negative sink size
        assert self.sink_size >= 0, "sink_size must be non-negative"

        # buffer length at seq_dim must equal sink_size + window_size
        expected_length = self.sink_size + self.window_size
        assert self.k_shape[self.seq_dim] == expected_length, (
            f"k_shape[seq_dim] ({self.k_shape[self.seq_dim]}) must equal sink_size + window_size ({expected_length})"
        )

        if self.sink_size > 0:
            # Sink-aware chunk-aligned path: keep the original divisibility
            # requirement so the rolling window stays chunk-aligned and
            # incoming chunks that overlap the sink region can drop their
            # leading tokens deterministically.
            #
            # FIXME: extend the roll-and-append path to support a non-zero
            # sink (e.g. by always preserving slots [0:sink_size] and
            # rolling only the [sink_size:] subrange). Not needed for the
            # alpadreams ``kv_drop_t`` experiment because alpadreams sets
            # ``sink_size_t == 0``.
            assert (
                self.window_size + self.sink_size
            ) % self.chunk_size == 0, (
                f"window_size + sink_size ({self.window_size + self.sink_size}) "
                f"must be divisible by chunk_size ({self.chunk_size}) when "
                f"sink_size > 0; non-divisible rolling windows with non-zero "
                f"sink are not implemented yet."
            )
        else:
            # Roll-and-append path: ``window_size`` does not need to be a
            # multiple of ``chunk_size``, but each chunk must fit inside
            # the rolling window so we never have to drop leading tokens
            # from the incoming chunk itself.
            assert self.chunk_size <= self.window_size, (
                f"chunk_size ({self.chunk_size}) must be <= window_size "
                f"({self.window_size}) in the roll-and-append path "
                f"(sink_size == 0)."
            )

        # initialize k and v
        self._k = torch.empty(self.k_shape, device=self.device, dtype=self.dtype)
        self._v = torch.empty(self.v_shape, device=self.device, dtype=self.dtype)

    def _seq_slice(self, start: int | None, end: int | None) -> tuple[slice | int, ...]:
        """Return an index tuple that slices the sequence dimension to [start:end]; other dims are :."""
        idx: list[slice | int] = [slice(None)] * len(self.k_shape)
        idx[self.seq_dim] = slice(start, end)
        return tuple(idx)

    def _roll_local_window_left(self) -> None:
        """Shift the local window left by chunk_size tokens (steady-state only)."""
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

    def _overwrite_rightmost_steady(self, k: Tensor, v: Tensor) -> None:
        """Write the new chunk into the rightmost positions (steady-state, after roll)."""
        total_size = self._k.shape[self.seq_dim]
        assert total_size == self._n_cached, (
            f"Expected full cache: {total_size=} != {self._n_cached=}"
        )
        write_end = total_size
        write_start = write_end - self.chunk_size
        if write_start > self.sink_size:
            sl_write = self._seq_slice(write_start, write_end)
            self._k[sl_write] = k
            self._v[sl_write] = v
        else:
            # The input token overlaps with the sink tokens, so we only keep partial of it.
            # Note: here we assume the sink tokens have already been written to the cache.
            # It is safe to assume this because this function will never be called for the
            # first chunk, and we assume the first chunk should be enough to cover the sink tokens.
            write_start = self.sink_size
            keep_size = write_end - write_start
            read_end = self.chunk_size
            read_start = read_end - keep_size
            sl_read = self._seq_slice(read_start, read_end)
            sl_write = self._seq_slice(write_start, write_end)
            self._k[sl_write] = k[sl_read]
            self._v[sl_write] = v[sl_read]

    def _overwrite_rightmost_filling(self, k: Tensor, v: Tensor) -> None:
        """Write the new chunk into the rightmost positions (filling phase)."""
        write_end = self._n_cached
        write_start = write_end - self.chunk_size
        assert write_start >= 0, (
            f"write [{write_start}:{write_end}) out of bounds for buffer size {self.sink_size + self.window_size}"
        )
        sl = self._seq_slice(write_start, write_end)
        self._k[sl] = k
        self._v[sl] = v

    def _append_to_end(self, k: Tensor, v: Tensor) -> None:
        """Append the new chunk to the end of the cache (filling phase)."""
        write_start = self._n_cached
        write_end = write_start + self.chunk_size
        assert write_end <= self.sink_size + self.window_size, (
            f"write [{write_start}:{write_end}) out of bounds for buffer size {self.sink_size + self.window_size}"
        )
        sl = self._seq_slice(write_start, write_end)
        self._k[sl] = k
        self._v[sl] = v

    def _roll_and_append(self, k: Tensor, v: Tensor) -> None:
        """Unified roll-and-append for ``sink_size == 0`` (handles non-divisible windows).

        Evicts exactly ``max(0, _n_cached + chunk_size - window_size)`` of the
        oldest rolling tokens, shifts the remaining valid prefix left to cover
        the eviction, and writes the incoming chunk at the right edge of the
        valid region. ``_n_cached`` itself is bumped in :meth:`after_update`.
        """
        chunk_size = self.chunk_size
        window = self.window_size

        excess = self._n_cached + chunk_size - window
        if excess > 0:
            # Shift the (n_cached - excess) trailing rolling tokens to the
            # front, freeing space at the right edge for the new chunk.
            keep = self._n_cached - excess
            if keep > 0:
                src = self._seq_slice(excess, self._n_cached)
                dst = self._seq_slice(0, keep)
                self._k[dst] = self._k[src].clone()
                self._v[dst] = self._v[src].clone()
            write_start = keep
        else:
            write_start = self._n_cached

        write_end = write_start + chunk_size
        sl = self._seq_slice(write_start, write_end)
        self._k[sl] = k
        self._v[sl] = v

        self._last_write_start = write_start
        self._last_write_end = write_end

    def _overwrite_last_write(self, k: Tensor, v: Tensor) -> None:
        """Re-write the slice of the most recent successful write.

        Used by same-``chunk_idx`` updates in the roll-and-append path,
        which can be triggered by repeated ``predict_flow`` calls inside
        one AR step (denoising loop iterations + the finalize pass).
        """
        sl = self._seq_slice(self._last_write_start, self._last_write_end)
        self._k[sl] = k
        self._v[sl] = v

    def is_steady_state(self) -> bool:
        """Return True if the cache is full (steady-state phase)."""
        assert self._curr_chunk_idx is not None, (
            "Must call before_update() before is_steady_state()"
        )
        total_size = self._k.shape[self.seq_dim]
        is_full = total_size == self._n_cached
        is_overlapping_with_sink = (
            self.sink_size > 0
            and self._curr_chunk_idx * self.chunk_size
            < self.sink_size  # start < sink_size
        )
        return is_full and not is_overlapping_with_sink

    def before_update(self, chunk_idx: int) -> None:
        """
        Prepare the cache before writing new tokens.

        If ``chunk_idx`` equals the previous chunk index, this is a no-op. Otherwise,
        we expect the ``chunk_idx`` to be +1 from the previous chunk index. In this case,
        we will roll the local window left if the cache is in steady-state, or no op
        if the cache is in filling phase.

        Args:
            chunk_idx: Chunk index of the new chunk in the full sequence.
        """
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
        # The roll-and-append path (sink_size == 0) folds the eviction into
        # ``update`` itself so the eviction count can depend on the actual
        # incoming chunk size and the buffer can be non-divisible. The
        # sink-aware chunk-aligned path keeps the original split of
        # before_update-rolls / update-writes.
        if self.sink_size > 0 and self.is_steady_state():
            self._roll_local_window_left()

    def update(self, k: Tensor, v: Tensor) -> None:
        """
        Write the new chunk's keys and values into the cache.

        Must be called after ``before_update()`` and before ``after_update()``.

        Args:
            k: Keys; shape must match cached keys except at seq_dim, where length must be chunk_size.
            v: Values; shape must match cached values except at seq_dim, where length must be chunk_size.
        """
        assert self._curr_chunk_idx is not None, (
            "Must call before_update() before update()"
        )

        chunk_size_k = k.shape[self.seq_dim]
        chunk_size_v = v.shape[self.seq_dim]
        assert chunk_size_k == self.chunk_size, (
            f"Expected input k to have chunk_size ({chunk_size_k}) at seq_dim ({self.seq_dim}), "
            f"got {chunk_size_k} != {self.chunk_size}"
        )
        assert chunk_size_v == self.chunk_size, (
            f"Expected input v to have chunk_size ({chunk_size_v}) at seq_dim ({self.seq_dim}), "
            f"got {chunk_size_v} != {self.chunk_size}"
        )
        if self.sink_size == 0:
            # Roll-and-append path: support non-divisible windows.
            if self._curr_chunk_idx == self._prev_chunk_idx + 1:
                self._roll_and_append(k, v)
            elif self._curr_chunk_idx == self._prev_chunk_idx:
                # Repeated predict_flow within the same AR step (denoising
                # loop iterations + finalize pass) write to the same slice.
                self._overwrite_last_write(k, v)
            else:
                raise ValueError(
                    f"{self._curr_chunk_idx=} should be either {self._prev_chunk_idx + 1} or {self._prev_chunk_idx}."
                )
        elif self.is_steady_state():
            self._overwrite_rightmost_steady(k, v)
            # Track the slice we just overwrote so external observers can
            # introspect it; same-chunk_idx overwrites take the same path
            # and produce the same result either way.
            total = self._k.shape[self.seq_dim]
            self._last_write_end = total
            self._last_write_start = max(self.sink_size, total - self.chunk_size)
        else:
            if self._curr_chunk_idx == self._prev_chunk_idx + 1:
                self._append_to_end(k, v)
                self._last_write_start = self._n_cached
                self._last_write_end = self._n_cached + self.chunk_size
            elif self._curr_chunk_idx == self._prev_chunk_idx:
                self._overwrite_rightmost_filling(k, v)
            else:
                raise ValueError(
                    f"{self._curr_chunk_idx=} should be either {self._prev_chunk_idx + 1} or {self._prev_chunk_idx}."
                )

    def after_update(self, chunk_idx: int) -> None:
        """
        Finalize bookkeeping after writing new tokens.

        Updates ``_prev_chunk_idx`` and, in filling phase, ``_n_cached``.

        Args:
            chunk_idx: The index of the new chunk in the full sequence.
        """
        assert chunk_idx == self._curr_chunk_idx, (
            f"Expected chunk_idx to be {self._curr_chunk_idx}, got {chunk_idx}"
        )

        if self._curr_chunk_idx == self._prev_chunk_idx + 1:
            total = self._k.shape[self.seq_dim]
            # ``_n_cached`` is the post-update count of valid tokens.
            # For both write paths, the right edge of the last write is
            # exactly the right edge of the valid region (and is bounded
            # by the buffer size).
            self._n_cached = min(self._n_cached + self.chunk_size, total)
            self._prev_chunk_idx += 1
        elif self._curr_chunk_idx == self._prev_chunk_idx:
            pass
        else:
            raise ValueError(
                f"{self._curr_chunk_idx=} should be either {self._prev_chunk_idx + 1} or {self._prev_chunk_idx}."
            )

        # reset the current chunk index as the last step.
        self._curr_chunk_idx = None

    def cached_k(self) -> Tensor:
        """
        Return cached keys for attention (valid prefix in filling phase, full buffer in steady-state).
        """
        if self.is_steady_state():
            return self._k
        total = self._k.shape[self.seq_dim]
        if self._curr_chunk_idx == self._prev_chunk_idx + 1:
            end = min(self._n_cached + self.chunk_size, total)
            return self._k[self._seq_slice(0, end)]
        elif self._curr_chunk_idx == self._prev_chunk_idx:
            return self._k[self._seq_slice(0, self._n_cached)]
        else:
            raise ValueError(
                f"{self._curr_chunk_idx=} should be either {self._prev_chunk_idx + 1} or {self._prev_chunk_idx}."
            )

    def cached_v(self) -> Tensor:
        """
        Return cached values for attention (valid prefix in filling phase, full buffer in steady-state).
        """
        if self.is_steady_state():
            return self._v
        total = self._v.shape[self.seq_dim]
        if self._curr_chunk_idx == self._prev_chunk_idx + 1:
            end = min(self._n_cached + self.chunk_size, total)
            return self._v[self._seq_slice(0, end)]
        elif self._curr_chunk_idx == self._prev_chunk_idx:
            return self._v[self._seq_slice(0, self._n_cached)]
        else:
            raise ValueError(
                f"{self._curr_chunk_idx=} should be either {self._prev_chunk_idx + 1} or {self._prev_chunk_idx}."
            )

    def reset(self) -> None:
        """Reset the cache to its initial empty state."""
        self._prev_chunk_idx = -1
        self._n_cached = 0
        self._last_write_start = 0
        self._last_write_end = 0
