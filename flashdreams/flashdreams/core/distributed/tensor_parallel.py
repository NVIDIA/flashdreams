# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Inference linear projections sharded along their output or input axis."""

from __future__ import annotations

import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn


class ColumnParallelLinear(nn.Module):
    """A contiguous output-row shard with no collective."""

    def __init__(self, source: nn.Linear, start: int, end: int) -> None:
        super().__init__()
        if not 0 <= start < end <= source.out_features:
            raise ValueError("output shard must lie within the source projection")
        self.in_features = source.in_features
        self.out_features = end - start
        self.weight = nn.Parameter(
            source.weight.data[start:end].clone(), requires_grad=False
        )
        bias = None if source.bias is None else source.bias.data[start:end].clone()
        self.bias = None if bias is None else nn.Parameter(bias, requires_grad=False)

    def forward(self, x: Tensor) -> Tensor:
        return F.linear(x, self.weight, self.bias)


class RowParallelLinear(nn.Module):
    """An input-column shard whose partial output is summed over ``group``.

    Exactly one rank must retain the source bias via ``keeps_bias``.
    """

    def __init__(
        self,
        source: nn.Linear,
        start: int,
        end: int,
        *,
        group: dist.ProcessGroup | None,
        keeps_bias: bool,
    ) -> None:
        super().__init__()
        if group is None:
            raise ValueError(
                "a row-parallel projection requires its tensor process group"
            )
        if not 0 <= start < end <= source.in_features:
            raise ValueError("input shard must lie within the source projection")
        self.in_features = end - start
        self.out_features = source.out_features
        self.weight = nn.Parameter(
            source.weight.data[:, start:end].clone(), requires_grad=False
        )
        bias = (
            source.bias.data.clone()
            if (source.bias is not None and keeps_bias)
            else None
        )
        self.bias = None if bias is None else nn.Parameter(bias, requires_grad=False)
        self._group = group

    def forward(self, x: Tensor) -> Tensor:
        out = F.linear(x, self.weight, self.bias)
        dist.all_reduce(out, group=self._group)
        return out
