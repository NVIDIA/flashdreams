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

"""Real-kernel numerical parity tests for the optional SageAttention 2 backend.

Run the single-rank test with plain pytest and the context-parallel tests with
``torchrun``::

    uv run --extra dev pytest \
        flashdreams/tests/test_sage2_attention_gpu.py -v
    uv run --extra dev torchrun --nproc_per_node=2 -m pytest \
        flashdreams/tests/test_sage2_attention_gpu.py -v

The tests skip when CUDA, a compatible SageAttention installation, or the
requested distributed launch is unavailable. They never replace SageAttention
or distributed collectives with fakes.
"""

from __future__ import annotations

import math
import os
from collections.abc import Iterator
from typing import Literal

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor
from torch.distributed import ProcessGroup

from flashdreams.core.attention import ContextParallelAttention, NativeAttention

pytestmark = pytest.mark.ci_gpu


@pytest.fixture(scope="module")
def cuda_device() -> torch.device:
    """Select this worker's CUDA device or skip on a CPU-only host."""
    if not torch.cuda.is_available():
        pytest.skip("SageAttention numerical parity requires CUDA.")

    device_count = torch.cuda.device_count()
    if device_count == 0:
        pytest.skip("No visible CUDA device is available.")

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device("cuda", local_rank % device_count)
    torch.cuda.set_device(device)
    return device


@pytest.fixture(scope="module")
def sage2_attention(cuda_device: torch.device) -> NativeAttention:
    """Load SageAttention and verify that its kernel is usable on this GPU."""
    try:
        attention = NativeAttention(qkv_format="bshd", backend="sage2").to(cuda_device)
    except RuntimeError as exc:
        pytest.skip(str(exc))

    # A package can expose ``sageattn`` while omitting the extension for the
    # active architecture. Probe one real launch so that such environments skip
    # before entering NCCL collectives. Only known availability errors are
    # converted to skips; numerical or integration failures remain test failures.
    generator = torch.Generator(device=cuda_device).manual_seed(11)
    probe = torch.randn(
        (1, 128, 1, 64),
        generator=generator,
        device=cuda_device,
        dtype=torch.bfloat16,
    )
    try:
        attention(probe, probe, probe)
        torch.cuda.synchronize(cuda_device)
    except (AssertionError, RuntimeError, ValueError) as exc:
        message = str(exc).lower()
        unavailable_markers = (
            "unsupported cuda architecture",
            "kernel is not available",
            "kernel is unavailable",
            "compute capability",
            "not compiled",
            "not built",
            "no kernel image is available",
        )
        if any(marker in message for marker in unavailable_markers):
            pytest.skip(f"SageAttention is unavailable on {cuda_device}: {exc}")
        raise

    return attention


@pytest.fixture(scope="module")
def distributed_group(
    cuda_device: torch.device,
    sage2_attention: NativeAttention,
) -> Iterator[ProcessGroup]:
    """Initialize the torchrun NCCL group used by CP parity tests."""
    requested_world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if dist.is_initialized():
        requested_world_size = dist.get_world_size()
    if requested_world_size < 2:
        pytest.skip(
            "Context-parallel SageAttention parity requires a torchrun launch "
            "with at least two ranks."
        )
    if not dist.is_nccl_available():
        pytest.skip("Context-parallel SageAttention parity requires NCCL.")

    owns_process_group = not dist.is_initialized()
    if owns_process_group:
        dist.init_process_group(backend="nccl", init_method="env://")

    group = dist.group.WORLD
    if group is None:
        raise RuntimeError("The default distributed process group was not created.")
    if dist.get_backend(group) != "nccl":
        if owns_process_group:
            dist.destroy_process_group()
        pytest.skip("Context-parallel SageAttention parity requires an NCCL group.")
    try:
        yield group
    finally:
        if owns_process_group and dist.is_initialized():
            dist.destroy_process_group()


def _make_qkv(
    *,
    sequence_length: int,
    num_heads: int,
    device: torch.device,
) -> tuple[Tensor, Tensor, Tensor]:
    """Create deterministic, non-identical Q/K/V inputs in BSHD layout."""
    generator = torch.Generator(device="cpu").manual_seed(2026)
    shape = (1, sequence_length, num_heads, 64)

    def sample() -> Tensor:
        return torch.randn(shape, generator=generator, dtype=torch.float32).to(
            device=device, dtype=torch.bfloat16
        )

    return sample(), sample(), sample()


def _sdpa_reference(query: Tensor, key: Tensor, value: Tensor) -> Tensor:
    """Compute an unquantized float32 SDPA reference in BSHD layout."""
    query_hnd = query.transpose(1, 2).float()
    key_hnd = key.transpose(1, 2).float()
    value_hnd = value.transpose(1, 2).float()
    scores = torch.matmul(query_hnd, key_hnd.transpose(-2, -1)) / math.sqrt(
        query.shape[-1]
    )
    return torch.matmul(scores.softmax(dim=-1), value_hnd).transpose(1, 2)


def _assert_numerical_parity(
    actual: Tensor,
    expected: Tensor,
    *,
    max_relative_rmse: float,
    min_cosine_similarity: float,
) -> None:
    """Check aggregate error without overfitting to quantized pointwise noise."""
    actual_float = actual.float()
    expected_float = expected.float()
    assert torch.isfinite(actual_float).all()

    error = actual_float - expected_float
    reference_rms = expected_float.square().mean().sqrt().clamp_min(1e-8)
    relative_rmse = error.square().mean().sqrt() / reference_rms
    cosine_similarity = F.cosine_similarity(
        actual_float.flatten(), expected_float.flatten(), dim=0
    )

    assert relative_rmse.item() <= max_relative_rmse, (
        f"relative RMSE {relative_rmse.item():.6f} exceeds {max_relative_rmse:.6f}"
    )
    assert cosine_similarity.item() >= min_cosine_similarity, (
        f"cosine similarity {cosine_similarity.item():.6f} is below "
        f"{min_cosine_similarity:.6f}"
    )


def test_sage2_single_rank_matches_float32_sdpa(
    cuda_device: torch.device,
    sage2_attention: NativeAttention,
) -> None:
    """The real Sage2 kernel remains numerically close to unquantized SDPA."""
    query, key, value = _make_qkv(
        sequence_length=256,
        num_heads=4,
        device=cuda_device,
    )

    expected = _sdpa_reference(query, key, value)
    actual = sage2_attention(query, key, value)

    assert actual.shape == expected.shape
    _assert_numerical_parity(
        actual,
        expected,
        max_relative_rmse=0.15,
        min_cosine_similarity=0.99,
    )


@pytest.mark.parametrize("method", ["ring", "ulysses"])
def test_context_parallel_sage2_matches_full_sequence_reference(
    method: Literal["ring", "ulysses"],
    cuda_device: torch.device,
    sage2_attention: NativeAttention,
    distributed_group: ProcessGroup,
) -> None:
    """Real Ring/Ulysses collectives match full-sequence Sage2 and SDPA."""
    world_size = dist.get_world_size(distributed_group)
    rank = dist.get_rank(distributed_group)
    local_sequence_length = 128
    global_sequence_length = local_sequence_length * world_size
    num_heads = 2 * world_size
    query, key, value = _make_qkv(
        sequence_length=global_sequence_length,
        num_heads=num_heads,
        device=cuda_device,
    )
    if method == "ring":
        # Make the first KV shard substantially less likely than the others.
        # Equal-weight output averaging can look deceptively accurate for IID
        # shards; this bias makes a broken/missing LSE merge fail decisively.
        query[..., 0] = 4.0
        key[..., 0] = 2.0
        key[:, :local_sequence_length, :, 0] = -2.0
    shard = slice(
        rank * local_sequence_length,
        (rank + 1) * local_sequence_length,
    )
    local_qkv = tuple(tensor[:, shard].contiguous() for tensor in (query, key, value))

    # The direct Sage2 reference isolates CP redistribution/LSE merge behavior;
    # the float32 reference independently bounds Sage2 quantization error.
    expected_sage2 = sage2_attention(query, key, value)[:, shard]
    expected_sdpa = _sdpa_reference(query, key, value)[:, shard]

    attention = ContextParallelAttention(
        qkv_format="bshd",
        backend="sage2",
        method=method,
    ).to(cuda_device)
    attention.set_context_parallel_group(distributed_group)
    actual = attention(*local_qkv)

    assert actual.shape == expected_sage2.shape
    _assert_numerical_parity(
        actual,
        expected_sage2,
        max_relative_rmse=0.08,
        min_cosine_similarity=0.997,
    )
    _assert_numerical_parity(
        actual,
        expected_sdpa,
        max_relative_rmse=0.18,
        min_cosine_similarity=0.985,
    )
