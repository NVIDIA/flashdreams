# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Thread-local KV state for LTX self-attention processors."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

PastKV = list[tuple[torch.Tensor, torch.Tensor]]


@dataclass
class KVContext:
    """Mutable KV state for one transformer forward / denoise step."""

    past_kv: PastKV | None = None
    present_kv: PastKV = field(default_factory=list)
    collect: bool = False
    num_layers: int = 0

    def begin_forward(self) -> None:
        if self.collect:
            self.present_kv = [None] * self.num_layers  # type: ignore[list-item]

    def set_layer_kv(self, layer_idx: int, kv: tuple[torch.Tensor, torch.Tensor]) -> None:
        if self.collect and layer_idx < len(self.present_kv):
            self.present_kv[layer_idx] = kv

    def collected_kv(self) -> PastKV | None:
        if not self.present_kv or any(x is None for x in self.present_kv):
            return None
        return self.present_kv  # type: ignore[return-value]


_CTX = KVContext()


def get_kv_context() -> KVContext:
    return _CTX


def configure_kv_context(
    *,
    past_kv: PastKV | None,
    collect: bool,
    num_layers: int,
) -> None:
    _CTX.past_kv = past_kv
    _CTX.collect = collect
    _CTX.num_layers = num_layers
    _CTX.present_kv = []


def reset_kv_context() -> None:
    _CTX.past_kv = None
    _CTX.collect = False
    _CTX.num_layers = 0
    _CTX.present_kv = []
