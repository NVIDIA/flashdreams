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

"""CPU contract tests for the optional SageAttention 2 backend."""

from __future__ import annotations

from typing import Any, cast

import pytest
import torch
from torch import Tensor

import flashdreams.core.attention.cp as cp_module
import flashdreams.core.attention.native as native_module
from flashdreams.core.attention import ContextParallelAttention, NativeAttention

pytestmark = pytest.mark.ci_cpu


class _FakeDeviceMesh:
    """Minimal two-rank device mesh used to exercise collective routing."""

    def get_rank(self) -> int:
        """Return the local rank."""
        return 0

    def size(self) -> int:
        """Return the simulated CP size."""
        return 2

    def get_group(self) -> object:
        """Return an opaque simulated process group."""
        return object()


@pytest.fixture
def fake_sage2(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Install a CPU fake for SageAttention and return its call records."""
    calls: list[dict[str, Any]] = []

    def fake_sageattn(
        query: Tensor,
        key: Tensor,
        value: Tensor,
        *,
        tensor_layout: str,
        is_causal: bool,
        return_lse: bool,
    ) -> Tensor | tuple[Tensor, Tensor]:
        calls.append(
            {
                "query": query,
                "key": key,
                "value": value,
                "tensor_layout": tensor_layout,
                "is_causal": is_causal,
                "return_lse": return_lse,
            }
        )
        output = query.clone()
        if not return_lse:
            return output
        if tensor_layout == "HND":
            batch, heads, sequence, _ = query.shape
        else:
            batch, sequence, heads, _ = query.shape
        lse = torch.zeros((batch, heads, sequence), dtype=torch.float32)
        return output, lse

    monkeypatch.setattr(native_module, "_load_sage2_op", lambda: fake_sageattn)
    monkeypatch.setattr(
        native_module, "_validate_sage2_inputs", lambda *args, **kwargs: None
    )
    return calls


def test_default_backend_does_not_import_sage2(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_load() -> None:
        raise AssertionError("default backend attempted to import SageAttention")

    monkeypatch.setattr(native_module, "_load_sage2_op", fail_load)

    NativeAttention()
    ContextParallelAttention()


def test_missing_sage2_dependency_has_actionable_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing_sageattention(name: str) -> Any:
        raise ModuleNotFoundError(f"No module named '{name}'")

    native_module._load_sage2_op.cache_clear()
    monkeypatch.setattr(native_module, "import_module", missing_sageattention)

    with pytest.raises(RuntimeError, match="requires SageAttention 2.x"):
        NativeAttention(backend="sage2")


def test_native_sage2_uses_nhd_without_transpose(
    fake_sage2: list[dict[str, Any]],
) -> None:
    attention = NativeAttention(qkv_format="bshd", backend="sage2")
    query = torch.randn(1, 5, 4, 16)
    key = torch.randn_like(query)
    value = torch.randn_like(query)

    output = attention(query, key, value)

    assert output.shape == query.shape
    assert len(fake_sage2) == 1
    assert fake_sage2[0]["query"] is query
    assert fake_sage2[0]["tensor_layout"] == "NHD"
    assert fake_sage2[0]["return_lse"] is False


def test_context_parallel_sage2_single_rank_uses_nhd(
    fake_sage2: list[dict[str, Any]],
) -> None:
    attention = ContextParallelAttention(
        qkv_format="bshd", backend="sage2", method="ring"
    )
    query = torch.randn(1, 5, 4, 16)

    output = attention(query, query, query)

    assert output.shape == query.shape
    assert len(fake_sage2) == 1
    assert fake_sage2[0]["query"] is query
    assert fake_sage2[0]["tensor_layout"] == "NHD"
    assert fake_sage2[0]["return_lse"] is False


def test_context_parallel_sage2_ring_waits_for_kv_and_requests_lse(
    fake_sage2: list[dict[str, Any]], monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[str] = []
    fake_sage_op = native_module._load_sage2_op()

    def ordered_sage_op(*args: Any, **kwargs: Any) -> Tensor | tuple[Tensor, Tensor]:
        events.append("sage")
        return fake_sage_op(*args, **kwargs)

    class _PendingGather:
        """Simulated pending functional-collective result."""

        def __init__(self, tensor: Tensor) -> None:
            self.tensor = tensor

        def wait(self) -> Tensor:
            """Resolve the pending result to its gathered tensor."""
            events.append("wait")
            return self.tensor

    def fake_all_gather(tensor: Tensor, **kwargs: Any) -> _PendingGather:
        events.append("gather")
        return _PendingGather(torch.cat([tensor, tensor]))

    monkeypatch.setattr(native_module, "_load_sage2_op", lambda: ordered_sage_op)
    monkeypatch.setattr(cp_module.funcol, "all_gather_tensor", fake_all_gather)
    attention = ContextParallelAttention(
        qkv_format="bshd",
        backend="sage2",
        method="ring",
        convert_to_fp32=False,
    )
    attention.device_mesh = cast(Any, _FakeDeviceMesh())
    query = torch.randn(1, 5, 4, 16)

    output = attention(query, query, query)

    assert output.shape == query.shape
    assert events == ["gather", "sage", "wait", "sage"]
    assert len(fake_sage2) == 2
    assert all(call["tensor_layout"] == "NHD" for call in fake_sage2)
    assert all(call["return_lse"] is True for call in fake_sage2)


def test_context_parallel_sage2_ulysses_preserves_nhd(
    fake_sage2: list[dict[str, Any]], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        cp_module.funcol, "all_to_all_single", lambda tensor, *args, **kwargs: tensor
    )
    attention = ContextParallelAttention(
        qkv_format="bshd", backend="sage2", method="ulysses"
    )
    attention.device_mesh = cast(Any, _FakeDeviceMesh())
    query = torch.randn(1, 5, 4, 16)

    output = attention(query, query, query)

    assert output.shape == query.shape
    assert len(fake_sage2) == 1
    assert fake_sage2[0]["query"].shape == (1, 10, 2, 16)
    assert fake_sage2[0]["tensor_layout"] == "NHD"
    assert fake_sage2[0]["return_lse"] is False


def test_context_parallel_sage2_ulysses_rejects_grouped_query_heads(
    fake_sage2: list[dict[str, Any]],
) -> None:
    attention = ContextParallelAttention(
        qkv_format="bshd", backend="sage2", method="ulysses"
    )
    attention.device_mesh = cast(Any, _FakeDeviceMesh())
    query = torch.randn(1, 5, 4, 16)
    key_value = torch.randn(1, 5, 2, 16)

    with pytest.raises(ValueError, match="equal query, key, and value head counts"):
        attention(query, key_value, key_value)

    assert not fake_sage2


def test_sage2_validates_dtype_head_dimension_and_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        native_module, "_load_sage2_op", lambda: lambda *args, **kwargs: args[0]
    )
    attention = NativeAttention(qkv_format="bshd", backend="sage2")

    float32_qkv = torch.randn(1, 2, 2, 64)
    with pytest.raises(ValueError, match="float16 or bfloat16"):
        attention(float32_qkv, float32_qkv, float32_qkv)

    oversized_qkv = torch.randn(1, 2, 2, 129, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="1 through 128"):
        attention(oversized_qkv, oversized_qkv, oversized_qkv)

    query = torch.randn(1, 2, 4, 64, dtype=torch.bfloat16)
    key_value = torch.randn(1, 2, 2, 64, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="grouped-query attention is not supported"):
        attention(query, key_value, key_value)

    cpu_qkv = torch.randn(1, 2, 2, 64, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="on a CUDA device"):
        attention(cpu_qkv, cpu_qkv, cpu_qkv)
