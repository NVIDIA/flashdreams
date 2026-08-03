# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Optional

import torch

PastKV = list[tuple[torch.Tensor, torch.Tensor]]


class LTXKVCache:
    """Rolling key-value cache for LTX transformer attention layers.

    Tensors are stored in LTX layout ``[batch, seq, heads, head_dim]``.
    """

    def __init__(self, window_size: Optional[int] = None) -> None:
        self._cache: Optional[PastKV] = None
        self.step: int = 0
        self.window_size = window_size

    def get(self) -> Optional[PastKV]:
        return self._cache

    def update(self, new_kv: PastKV) -> None:
        if self._cache is None:
            self._cache = new_kv
        else:
            merged: PastKV = []
            for (k_old, v_old), (k_new, v_new) in zip(
                self._cache, new_kv, strict=True
            ):
                k = torch.cat([k_old, k_new], dim=1)
                v = torch.cat([v_old, v_new], dim=1)
                if self.window_size is not None:
                    k = k[:, -self.window_size :, :, :]
                    v = v[:, -self.window_size :, :, :]
                merged.append((k, v))
            self._cache = merged
        self.step += 1

    def clear(self) -> None:
        self._cache = None
        self.step = 0

    @property
    def seq_len(self) -> int:
        if self._cache is None:
            return 0
        return self._cache[0][0].shape[1]
