# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU regression tests for context-parallel subgroup rank handling."""

from __future__ import annotations

from typing import cast

import pytest
import torch
from torch import Tensor
from torch.distributed.tensor.device_mesh import DeviceMesh

import flashdreams.core.attention.cp as cp_module
from flashdreams.core.attention.cp import ContextParallelAttention

pytestmark = pytest.mark.ci_cpu


class _FakeSubgroupMesh:
    def get_rank(self) -> int:
        raise AssertionError("ring rotation must not use the global rank")

    def get_local_rank(self) -> int:
        return 1

    def size(self) -> int:
        return 3

    def get_group(self) -> object:
        return object()


def test_ring_rotation_indexes_all_gather_with_subgroup_local_rank(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    visited_keys: list[float] = []

    def fake_all_gather(
        _local: Tensor,
        *,
        gather_dim: int,
        group: object,
    ) -> Tensor:
        assert gather_dim == 0
        assert group is not None
        return torch.tensor([0.0, 100.0, 1.0, 101.0, 2.0, 102.0])

    def fake_attention(
        query: Tensor,
        key: Tensor,
        value: Tensor,
        *,
        return_lse: bool,
    ) -> tuple[Tensor, Tensor]:
        assert return_lse
        visited_keys.append(float(key.item()))
        return torch.zeros_like(query), torch.zeros_like(query)

    monkeypatch.setattr(cp_module.funcol, "all_gather_tensor", fake_all_gather)
    monkeypatch.setattr(cp_module, "torch_sdpa_cudnn", fake_attention)

    attention = ContextParallelAttention(
        backend="cudnn",
        method="ring",
        convert_to_fp32=False,
    )
    attention.device_mesh = cast(DeviceMesh, _FakeSubgroupMesh())
    query = torch.zeros(1, 1, 1, 1)
    key = torch.ones(1, 1, 1, 1)
    value = torch.full((1, 1, 1, 1), 101.0)

    output = attention._impl_ring(query, key, value)

    assert output.shape == query.shape
    assert visited_keys == [1.0, 2.0, 0.0]
