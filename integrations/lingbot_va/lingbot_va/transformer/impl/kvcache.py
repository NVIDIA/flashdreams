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
        return min(self._n_committed, self.window_slots) * self.slot_size

    def _write_offset(self) -> int:
        """Start position for the current (uncommitted) slot."""
        idx = self._n_committed % self.window_slots
        return idx * self.slot_size

    def write_primary(self, k: Tensor, v: Tensor) -> None:
        """Write primary (video) KV into the current slot.

        Args:
            k: Shape ``[batch, primary_chunk, heads, head_dim]``.
            v: Same shape as ``k``.
        """
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

        If the buffer is full, the oldest slot is overwritten on the
        next write (circular buffer via modular indexing).
        """
        assert self._primary_written
        self._n_committed += 1
        self._primary_written = False

    def cached_k(self) -> Tensor:
        """Return all committed K tokens."""
        n = self.n_cached_tokens
        return self._k[:, :n]

    def cached_v(self) -> Tensor:
        """Return all committed V tokens."""
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
