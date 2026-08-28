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
"""VAKVCache — rolling-window KV cache for video-action transformers.

Wraps a ``BlockKVCache`` with a compile-friendly read path: intermediate
denoising steps read committed cache tensors (no writes), while the final step
writes the full [video|action] chunk via ``BlockKVCache.update()``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from flashdreams.core.attention.kvcache import BlockKVCache


@dataclass
class VAKVCache:
    """Rolling-window KV cache for video-action models.

    Lifecycle per AR step:
        cache.before_update(chunk_idx)
        ... intermediate denoising: read via committed_kv (no cache mutation)
        ... final video step: write_video(k, v)
        ... final action step: write_action(k, v) → commits full chunk
        cache.after_update(chunk_idx)
    """

    kv_cache: BlockKVCache
    video_chunk: int
    action_chunk: int

    @staticmethod
    def create(
        *,
        video_chunk: int,
        action_chunk: int,
        window_slots: int,
        batch_size: int,
        num_heads: int,
        head_dim: int,
        sink_size: int = 0,
        device: torch.device | str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ) -> "VAKVCache":
        """Construct a VAKVCache with the given dimensions."""
        slot_size = video_chunk + action_chunk
        window_size = window_slots * slot_size
        total_size = sink_size + window_size
        kv_shape = (batch_size, total_size, num_heads, head_dim)

        kv_cache = BlockKVCache(
            k_shape=kv_shape,
            v_shape=kv_shape,
            seq_dim=1,
            chunk_size=slot_size,
            window_size=window_size,
            sink_size=sink_size,
            device=device,
            dtype=dtype,
        )

        return VAKVCache(
            kv_cache=kv_cache,
            video_chunk=video_chunk,
            action_chunk=action_chunk,
        )

    @property
    def n_committed_tokens(self) -> int:
        """Number of valid prior-step tokens after opening the update window."""
        return self.kv_cache.write_end - self.kv_cache.chunk_size

    def before_update(self, chunk_idx: int) -> None:
        """Open the update window for a new AR step."""
        self.kv_cache.before_update(chunk_idx)

    def after_update(self, chunk_idx: int) -> None:
        """Close the update window and commit."""
        self.kv_cache.after_update(chunk_idx)

    def committed_kv(self) -> tuple[Tensor, Tensor]:
        """Return the valid prior-step KV prefix after any window roll.

        ``BlockKVCache.before_update`` rolls a full cache before the current
        chunk is written. Its trailing chunk-sized region is therefore stale
        until ``write_video``/``write_action`` replace it and must not be
        exposed as committed context.
        """
        n = self.n_committed_tokens
        return self.kv_cache.cached_k()[:, :n], self.kv_cache.cached_v()[:, :n]

    def write_video(self, k: Tensor, v: Tensor) -> None:
        """Write video KV to the current chunk (pads action with zeros).

        Called once on the final video denoising step.

        Args:
            k: Shape ``[batch, video_chunk, heads, head_dim]``.
            v: Same shape as ``k``.
        """
        batch, _, heads, head_dim = k.shape
        action_k = torch.zeros(
            batch,
            self.action_chunk,
            heads,
            head_dim,
            device=k.device,
            dtype=k.dtype,
        )
        action_v = torch.zeros_like(action_k)
        full_k = torch.cat([k, action_k], dim=1)
        full_v = torch.cat([v, action_v], dim=1)
        self.kv_cache.update(full_k, full_v)

    def write_action(
        self, k: Tensor, v: Tensor, video_k: Tensor, video_v: Tensor
    ) -> None:
        """Write full [video|action] KV to the current chunk (overwrite).

        Called once on the final action denoising step.

        Args:
            k: Action K, shape ``[batch, action_chunk, heads, head_dim]``.
            v: Action V, same shape.
            video_k: Video K from the final video step.
            video_v: Video V from the final video step.
        """
        full_k = torch.cat([video_k, k], dim=1)
        full_v = torch.cat([video_v, v], dim=1)
        self.kv_cache.update(full_k, full_v)

    def reset(self) -> None:
        """Reset to empty state."""
        self.kv_cache.reset()
