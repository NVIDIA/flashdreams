# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Block-style KV cache with optional QVG-compressed old chunks."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor

from flashdreams.core.attention.kv_compress.base import (
    KVCacheStats,
    KVCompressionConfig,
    KVSpan,
    KVStorageBackend,
    KVStoragePayload,
    RuntimePhase,
    estimate_tensor_tree_bytes,
)


EntryKind = Literal["bf16", "quantized"]


@dataclass
class _CacheEntry:
    kind: EntryKind
    start_chunk: int
    end_chunk: int
    k: Tensor | None = None
    v: Tensor | None = None
    payload: KVStoragePayload | None = None
    payload_start_chunk: int = 0
    rope_freqs_by_chunk: tuple[Tensor, ...] | None = None


class QuantizedKVCache:
    """KV cache compatible with ``BlockKVCache`` but able to compress old chunks.

    The cache stores new/current chunks as dense tensors and compresses older
    finalized spans after the clean KV update. It currently assumes the Wan
    self-attention layout `[B, S, H, D]` (`seq_dim=1`).
    """

    def __init__(
        self,
        *,
        k_shape: tuple[int, ...],
        v_shape: tuple[int, ...],
        seq_dim: int,
        chunk_size: int,
        window_size: int,
        sink_size: int = 0,
        device: torch.device | str = torch.device("cuda"),
        dtype: torch.dtype = torch.float16,
        backend: KVStorageBackend,
        compression_config: KVCompressionConfig,
    ) -> None:
        assert len(k_shape) == 4, "QuantizedKVCache currently expects [B,S,H,D]"
        assert len(v_shape) == 4, "QuantizedKVCache currently expects [B,S,H,D]"
        tensor_dim = len(k_shape)
        seq_dim = seq_dim if seq_dim >= 0 else seq_dim + tensor_dim
        assert seq_dim == 1, "QuantizedKVCache currently supports seq_dim=1 only"
        assert k_shape[:-1] == v_shape[:-1], (
            "k and v must have the same shape except last dim"
        )
        assert (sink_size + window_size) == k_shape[seq_dim], (
            "cache capacity must match sink_size + window_size"
        )
        assert (sink_size + window_size) % chunk_size == 0, (
            "cache capacity must be divisible by chunk_size"
        )
        assert sink_size % chunk_size == 0, (
            "QuantizedKVCache requires sink_size divisible by chunk_size"
        )

        self.k_shape = k_shape
        self.v_shape = v_shape
        self.seq_dim = seq_dim
        self.chunk_size = chunk_size
        self.window_size = window_size
        self.sink_size = sink_size
        self.device = torch.device(device)
        self.dtype = dtype
        self.backend = backend
        self.compression_config = compression_config
        self.stores_prerope_keys = bool(
            compression_config.backend_config.get("store_prerope_keys", False)
        )

        self._prev_chunk_idx = -1
        self._curr_chunk_idx: int | None = None
        self._n_cached = 0
        self._entries: list[_CacheEntry | None] = [
            None for _ in range(self.max_chunks)
        ]
        self._dense_read_cache: tuple[int, int, Tensor, Tensor] | None = None
        self._rope_read_cache: tuple[int, int, Tensor] | None = None
        self._pending_rope_freqs: Tensor | None = None
        self.stats = KVCacheStats()

    @property
    def max_chunks(self) -> int:
        return (self.sink_size + self.window_size) // self.chunk_size

    @property
    def sink_chunks(self) -> int:
        return self.sink_size // self.chunk_size

    @property
    def n_cached_chunks(self) -> int:
        return self._n_cached // self.chunk_size

    def is_steady_state(self) -> bool:
        """Return True if the fixed-capacity cache is full."""
        assert self._curr_chunk_idx is not None, (
            "Must call before_update() before is_steady_state()"
        )
        return self._n_cached == self.sink_size + self.window_size

    def before_update(self, chunk_idx: int) -> None:
        """Prepare the cache before writing a chunk."""
        assert self._curr_chunk_idx is None, (
            "Must call after_update() before before_update()"
        )
        self._invalidate_dense_read_cache()
        self._curr_chunk_idx = chunk_idx
        if chunk_idx == self._prev_chunk_idx:
            return
        assert chunk_idx == self._prev_chunk_idx + 1, (
            "Expected the new chunk_idx to be +1 from previous chunk_idx, "
            f"got {chunk_idx} != {self._prev_chunk_idx} + 1"
        )
        if self.is_steady_state():
            self._roll_local_window_left()

    def update(self, k: Tensor, v: Tensor) -> None:
        """Write K/V for the current chunk."""
        assert self._curr_chunk_idx is not None, "Must call before_update() first"
        assert k.shape[self.seq_dim] == self.chunk_size
        assert v.shape[self.seq_dim] == self.chunk_size

        if self.is_steady_state():
            self._write_chunk(self.max_chunks - 1, k, v)
        elif self._curr_chunk_idx == self._prev_chunk_idx + 1:
            self._write_chunk(self.n_cached_chunks, k, v)
        elif self._curr_chunk_idx == self._prev_chunk_idx:
            self._write_chunk(max(self.n_cached_chunks - 1, 0), k, v)
        else:
            raise ValueError(
                f"{self._curr_chunk_idx=} should be either "
                f"{self._prev_chunk_idx + 1} or {self._prev_chunk_idx}."
            )
        self._pending_rope_freqs = None

    def after_update(self, chunk_idx: int) -> None:
        """Finalize bookkeeping after writing a chunk."""
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

    def cached_k(self) -> Tensor:
        """Return dense cached keys for attention."""
        return self._cached_dense_read()[0]

    def cached_k_rope_freqs(self) -> Tensor:
        """Return RoPE frequencies aligned with ``cached_k()``."""
        if not self.stores_prerope_keys:
            raise ValueError("This cache stores post-RoPE keys")
        return self._cached_rope_read()

    def cached_v(self) -> Tensor:
        """Return dense cached values for attention."""
        v = self._cached_dense_read()[1]
        # Wan attention reads K then V immediately before calling attention.
        # Drop the cache's extra reference so dequantized full-window buffers
        # do not survive until the next chunk.
        self._invalidate_dense_read_cache()
        return v

    def finalize_clean_chunk(self, chunk_idx: int) -> None:
        """Compress eligible old chunks after clean KV update."""
        assert chunk_idx == self._prev_chunk_idx, (
            f"Expected finalized chunk {self._prev_chunk_idx}, got {chunk_idx}"
        )
        interval = int(
            self.compression_config.schedule.get("compress_every_n_chunks", 1)
        )
        if interval > 1 and (chunk_idx + 1) % interval != 0:
            self._refresh_stats()
            return

        protected_recent = self.compression_config.protected_recent_chunks
        end_chunk = self.n_cached_chunks - protected_recent
        start_chunk = self.sink_chunks
        if end_chunk <= start_chunk:
            self._refresh_stats()
            return

        span_start: int | None = None
        for ci in range(start_chunk, end_chunk):
            entry = self._entries[ci]
            if entry is not None and entry.kind == "bf16":
                if span_start is None:
                    span_start = ci
                continue
            if span_start is not None:
                self._compress_chunk_span(span_start, ci)
                span_start = None
        if span_start is not None:
            self._compress_chunk_span(span_start, end_chunk)
        self._refresh_stats()

    def reset(self) -> None:
        """Reset cache to empty state."""
        self._prev_chunk_idx = -1
        self._curr_chunk_idx = None
        self._n_cached = 0
        self._entries = [None for _ in range(self.max_chunks)]
        self._dense_read_cache = None
        self._rope_read_cache = None
        self._pending_rope_freqs = None
        self.stats = KVCacheStats()

    def set_pending_rope_freqs(self, rope_freqs: Tensor | None) -> None:
        """Attach the current chunk's RoPE frequencies before ``update()``."""
        if not self.stores_prerope_keys:
            return
        if rope_freqs is None:
            raise ValueError("QVG pre-RoPE key storage requires rope_freqs")
        if rope_freqs.shape[0] != self.chunk_size:
            raise ValueError(
                f"Expected rope_freqs length {self.chunk_size}, "
                f"got {rope_freqs.shape[0]}"
            )
        self._pending_rope_freqs = rope_freqs.detach()

    def _invalidate_dense_read_cache(self) -> None:
        self._dense_read_cache = None
        self._rope_read_cache = None

    def _cached_dense_read(self) -> tuple[Tensor, Tensor]:
        start_token = 0
        end_token = self._visible_tokens()
        if self._dense_read_cache is not None:
            cached_start, cached_end, cached_k, cached_v = self._dense_read_cache
            if cached_start == start_token and cached_end == end_token:
                return cached_k, cached_v
        k, v = self._read_dense(start_token, end_token)
        self._dense_read_cache = (start_token, end_token, k, v)
        return k, v

    def _cached_rope_read(self) -> Tensor:
        start_token = 0
        end_token = self._visible_tokens()
        if self._rope_read_cache is not None:
            cached_start, cached_end, cached_rope = self._rope_read_cache
            if cached_start == start_token and cached_end == end_token:
                return cached_rope
        rope_freqs = self._read_rope_freqs(start_token, end_token)
        self._rope_read_cache = (start_token, end_token, rope_freqs)
        return rope_freqs

    def _visible_tokens(self) -> int:
        if self._curr_chunk_idx == self._prev_chunk_idx + 1:
            return min(
                self._n_cached + self.chunk_size,
                self.sink_size + self.window_size,
            )
        return self._n_cached

    def _roll_local_window_left(self) -> None:
        """Drop the oldest local-window chunk and keep sink chunks."""
        assert self._n_cached == self.sink_size + self.window_size
        drop_chunk = self.sink_chunks
        new_entries: list[_CacheEntry | None] = [None for _ in range(self.max_chunks)]
        for ci in range(drop_chunk):
            new_entries[ci] = self._entries[ci]
        old_ci = drop_chunk + 1
        while old_ci < self.max_chunks:
            entry = self._entries[old_ci]
            new_ci = old_ci - 1
            if entry is None:
                old_ci += 1
                continue
            if entry.kind == "bf16":
                new_entries[new_ci] = _CacheEntry(
                    kind="bf16",
                    start_chunk=new_ci,
                    end_chunk=new_ci + 1,
                    k=entry.k,
                    v=entry.v,
                    rope_freqs_by_chunk=entry.rope_freqs_by_chunk,
                )
                old_ci += 1
                continue

            assert entry.payload is not None
            payload = entry.payload
            payload_start_chunk = entry.payload_start_chunk + (
                old_ci - entry.start_chunk
            )
            old_end = old_ci + 1
            while old_end < self.max_chunks:
                next_entry = self._entries[old_end]
                if (
                    next_entry is None
                    or next_entry.kind != "quantized"
                    or next_entry.payload is not payload
                ):
                    break
                expected_payload_chunk = payload_start_chunk + (old_end - old_ci)
                next_payload_chunk = next_entry.payload_start_chunk + (
                    old_end - next_entry.start_chunk
                )
                if next_payload_chunk != expected_payload_chunk:
                    break
                old_end += 1

            new_start = new_ci
            new_end = new_ci + (old_end - old_ci)
            shifted_entry = _CacheEntry(
                kind="quantized",
                start_chunk=new_start,
                end_chunk=new_end,
                payload=payload,
                payload_start_chunk=payload_start_chunk,
                rope_freqs_by_chunk=entry.rope_freqs_by_chunk,
            )
            for ci in range(new_start, new_end):
                new_entries[ci] = shifted_entry
            old_ci = old_end
        self._entries = new_entries
        self._invalidate_dense_read_cache()

    def _write_chunk(self, chunk_idx: int, k: Tensor, v: Tensor) -> None:
        self._restore_entry_if_quantized(chunk_idx)
        rope_freqs_by_chunk = self._consume_pending_rope_freqs()
        self._entries[chunk_idx] = _CacheEntry(
            kind="bf16",
            start_chunk=chunk_idx,
            end_chunk=chunk_idx + 1,
            k=k.detach().to(device=self.device, dtype=self.dtype).clone(),
            v=v.detach().to(device=self.device, dtype=self.dtype).clone(),
            rope_freqs_by_chunk=rope_freqs_by_chunk,
        )

    def _consume_pending_rope_freqs(self) -> tuple[Tensor, ...] | None:
        if not self.stores_prerope_keys:
            return None
        if self._pending_rope_freqs is None:
            raise ValueError("Missing pending RoPE frequencies for pre-RoPE cache")
        rope_freqs = self._pending_rope_freqs
        if rope_freqs.device != self.device:
            rope_freqs = rope_freqs.to(device=self.device)
        return (rope_freqs,)

    def _restore_entry_if_quantized(self, chunk_idx: int) -> None:
        entry = self._entries[chunk_idx]
        if entry is None or entry.kind != "quantized":
            return
        assert entry.payload is not None
        start_chunk = entry.start_chunk
        end_chunk = entry.end_chunk
        k, v = self.backend.decompress_span(
            entry.payload, phase=RuntimePhase.DENOISE, device=self.device
        )
        k = k.permute(0, 2, 1, 3).contiguous()
        v = v.permute(0, 2, 1, 3).contiguous()
        for ci in range(start_chunk, end_chunk):
            s = (entry.payload_start_chunk + ci - start_chunk) * self.chunk_size
            e = s + self.chunk_size
            self._entries[ci] = _CacheEntry(
                kind="bf16",
                start_chunk=ci,
                end_chunk=ci + 1,
                k=k[:, s:e].to(dtype=self.dtype).clone(),
                v=v[:, s:e].to(dtype=self.dtype).clone(),
                rope_freqs_by_chunk=self._slice_entry_rope_chunks(
                    entry, ci, ci + 1
                ),
            )

    def _restore_all_quantized(self) -> None:
        seen: set[int] = set()
        for ci, entry in enumerate(list(self._entries)):
            if entry is None or entry.kind != "quantized":
                continue
            assert entry.payload is not None
            payload_id = id(entry.payload)
            if payload_id in seen:
                continue
            seen.add(payload_id)
            self._restore_entry_if_quantized(ci)

    def _read_dense(self, start_token: int, end_token: int) -> tuple[Tensor, Tensor]:
        assert start_token % self.chunk_size == 0
        assert end_token % self.chunk_size == 0
        start_chunk = start_token // self.chunk_size
        end_chunk = end_token // self.chunk_size
        k_parts: list[Tensor] = []
        v_parts: list[Tensor] = []
        ci = start_chunk
        dequant_ms = 0.0
        while ci < end_chunk:
            entry = self._entries[ci]
            if entry is None:
                raise ValueError(f"cache chunk {ci} is empty")
            if entry.kind == "bf16":
                assert entry.k is not None and entry.v is not None
                k_parts.append(entry.k)
                v_parts.append(entry.v)
                ci += 1
                continue
            assert entry.payload is not None
            k, v = self.backend.decompress_span(
                entry.payload, phase=RuntimePhase.DENOISE, device=self.device
            )
            dequant_ms += float(entry.payload.metadata.get("dequantize_ms", 0.0))
            k = k.permute(0, 2, 1, 3).contiguous()
            v = v.permute(0, 2, 1, 3).contiguous()
            read_start = max(ci, entry.start_chunk)
            read_end = min(end_chunk, entry.end_chunk)
            s = (
                entry.payload_start_chunk + read_start - entry.start_chunk
            ) * self.chunk_size
            e = (
                entry.payload_start_chunk + read_end - entry.start_chunk
            ) * self.chunk_size
            k_parts.append(k[:, s:e])
            v_parts.append(v[:, s:e])
            ci = read_end
        if dequant_ms:
            self.stats.dequantize_ms += dequant_ms
        return torch.cat(k_parts, dim=1), torch.cat(v_parts, dim=1)

    def _read_rope_freqs(self, start_token: int, end_token: int) -> Tensor:
        assert start_token % self.chunk_size == 0
        assert end_token % self.chunk_size == 0
        start_chunk = start_token // self.chunk_size
        end_chunk = end_token // self.chunk_size
        rope_parts: list[Tensor] = []
        ci = start_chunk
        while ci < end_chunk:
            entry = self._entries[ci]
            if entry is None:
                raise ValueError(f"cache chunk {ci} is empty")
            read_start = max(ci, entry.start_chunk)
            read_end = min(end_chunk, entry.end_chunk)
            rope_parts.extend(self._slice_entry_rope_chunks(entry, read_start, read_end))
            ci = read_end
        if len(rope_parts) == 1:
            return rope_parts[0]
        return torch.cat(rope_parts, dim=0)

    def _slice_entry_rope_chunks(
        self,
        entry: _CacheEntry,
        read_start: int,
        read_end: int,
    ) -> tuple[Tensor, ...]:
        if not self.stores_prerope_keys:
            return ()
        if entry.rope_freqs_by_chunk is None:
            raise ValueError("Missing RoPE frequencies for pre-RoPE cache entry")
        start = entry.payload_start_chunk + read_start - entry.start_chunk
        end = entry.payload_start_chunk + read_end - entry.start_chunk
        return entry.rope_freqs_by_chunk[start:end]

    def _compress_chunk_span(self, start_chunk: int, end_chunk: int) -> None:
        if end_chunk <= start_chunk:
            return
        k, v = self._read_dense(
            start_chunk * self.chunk_size, end_chunk * self.chunk_size
        )
        rope_freqs_by_chunk: tuple[Tensor, ...] | None = None
        if self.stores_prerope_keys:
            rope_parts: list[Tensor] = []
            ci = start_chunk
            while ci < end_chunk:
                entry = self._entries[ci]
                if entry is None:
                    raise ValueError(f"cache chunk {ci} is empty")
                read_start = max(ci, entry.start_chunk)
                read_end = min(end_chunk, entry.end_chunk)
                rope_parts.extend(
                    self._slice_entry_rope_chunks(entry, read_start, read_end)
                )
                ci = read_end
            rope_freqs_by_chunk = tuple(rope_parts)
        k_bhsd = k.permute(0, 2, 1, 3).contiguous()
        v_bhsd = v.permute(0, 2, 1, 3).contiguous()
        span = KVSpan(
            start=start_chunk * self.chunk_size,
            end=end_chunk * self.chunk_size,
        )
        start = time.perf_counter()
        payload = self.backend.compress_span(
            k_bhsd,
            v_bhsd,
            span=span,
            phase=RuntimePhase.FINALIZE_CLEAN_KV,
            config=self.compression_config,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        self.stats.quantize_ms += float(
            payload.metadata.get("quantize_ms", elapsed_ms)
        )
        entry = _CacheEntry(
            kind="quantized",
            start_chunk=start_chunk,
            end_chunk=end_chunk,
            payload=payload,
            rope_freqs_by_chunk=rope_freqs_by_chunk,
        )
        for ci in range(start_chunk, end_chunk):
            self._entries[ci] = entry
        self._invalidate_dense_read_cache()

    def _refresh_stats(self) -> None:
        stored = 0
        quantized_spans = 0
        seen_payloads: set[int] = set()
        for entry in self._entries[: self.n_cached_chunks]:
            if entry is None:
                continue
            if entry.kind == "bf16":
                stored += estimate_tensor_tree_bytes(entry.k)
                stored += estimate_tensor_tree_bytes(entry.v)
            else:
                assert entry.payload is not None
                payload_id = id(entry.payload)
                if payload_id not in seen_payloads:
                    stored += self.backend.estimate_bytes(entry.payload)
                    seen_payloads.add(payload_id)
                    quantized_spans += 1
        bf16_bytes = 0
        for shape in (self.k_shape, self.v_shape):
            elems_per_token = 1
            for dim_idx, dim in enumerate(shape):
                if dim_idx != self.seq_dim:
                    elems_per_token *= dim
            dtype_bytes = torch.empty((), dtype=self.dtype).element_size()
            bf16_bytes += elems_per_token * self._n_cached * dtype_bytes
        self.stats.bf16_equivalent_bytes = bf16_bytes
        self.stats.stored_bytes = stored
        self.stats.num_quantized_spans = quantized_spans
