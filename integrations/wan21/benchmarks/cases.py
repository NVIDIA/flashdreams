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

"""Shared Wan 2.1 attention benchmark cases."""

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch

from flashdreams.accelerated.multi_head_attention_triton import SDPABackend
from flashdreams.recipes.wan.transformer.impl.modules import AttentionBackend


@dataclass(frozen=True)
class AttentionBenchmarkCase:
    """Configuration and metadata for one attention benchmark implementation."""

    implementation: str
    """Stable implementation name stored in benchmark metadata."""

    attention_backend: AttentionBackend
    """DiT block implementation configured for this case."""

    sdpa_backend: SDPABackend
    """SDPA implementation configured for Triton self-attention."""

    self_attention_operator: str
    """Self-attention operator reported in benchmark metadata."""

    cross_attention_operator: str
    """Cross-attention operator reported in benchmark metadata."""

    minimum_compute_capability: tuple[int, int] | None = None
    """Minimum CUDA compute capability; ``None`` accepts any CUDA device."""

    @property
    def pytest_id(self) -> str:
        """Return the readable pytest parameter identifier."""
        return self.implementation.replace("_", "-")


WAN_TORCH_CASE = AttentionBenchmarkCase(
    implementation="wan_torch",
    attention_backend=AttentionBackend.WAN,
    sdpa_backend=SDPABackend.CUDNN,
    self_attention_operator="cudnn",
    cross_attention_operator="cudnn",
)

TRITON_CUDNN_CASE = AttentionBenchmarkCase(
    implementation="triton_cudnn",
    attention_backend=AttentionBackend.TRITON,
    sdpa_backend=SDPABackend.CUDNN,
    self_attention_operator="torch_cudnn_sdpa",
    cross_attention_operator="triton_fa2",
    minimum_compute_capability=(9, 0),
)

TRITON_FA2_CASE = AttentionBenchmarkCase(
    implementation="triton_fa2",
    attention_backend=AttentionBackend.TRITON,
    sdpa_backend=SDPABackend.TRITON,
    self_attention_operator="triton_fa2",
    cross_attention_operator="triton_fa2",
    minimum_compute_capability=(9, 0),
)

ATTENTION_CASES = (WAN_TORCH_CASE, TRITON_CUDNN_CASE, TRITON_FA2_CASE)
"""Attention cases exercised by each Wan 2.1 benchmark layer."""

assert {case.attention_backend for case in ATTENTION_CASES} == set(AttentionBackend)
assert {
    case.sdpa_backend
    for case in ATTENTION_CASES
    if case.attention_backend is AttentionBackend.TRITON
} == set(SDPABackend)


def skip_unsupported_device(
    case: AttentionBenchmarkCase,
    device: torch.device,
) -> None:
    """Skip a benchmark case when device is older than its minimum capability."""
    minimum = case.minimum_compute_capability
    if minimum is None:
        return
    if torch.cuda.get_device_capability(device) < minimum:
        pytest.skip(
            f"{case.pytest_id} attention requires compute capability "
            f"{minimum[0]}.{minimum[1]}+"
        )
