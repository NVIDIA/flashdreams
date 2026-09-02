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

"""Shared LingBot attention benchmark cases."""

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch

from flashdreams.accelerated.multi_head_attention.optimized import (
    OptimizedImplConfig,
    QKVFusionOption,
    QuantizationOption,
    SDPABackend,
)
from flashdreams.recipes.wan.transformer.impl.modules import AttentionBackend


@dataclass(frozen=True)
class AttentionBenchmarkCase:
    """Configuration for one attention benchmark implementation."""

    implementation: str
    """Stable implementation name used by pytest and pipeline setup."""

    self_attention_backend: AttentionBackend
    """Self-attention implementation configured for this case."""

    cross_attention_backend: AttentionBackend
    """Cross-attention implementation configured for this case."""

    self_attn_optimized_impl_config: OptimizedImplConfig = OptimizedImplConfig(
        qkv_fusion_option=QKVFusionOption.FULL,
        sdpa_backend=SDPABackend.FA2,
    )
    """Optimized implementation policy used by self-attention."""

    cross_attn_optimized_impl_config: OptimizedImplConfig = OptimizedImplConfig(
        qkv_fusion_option=QKVFusionOption.FUSE_KV,
        sdpa_backend=SDPABackend.FA2,
    )
    """Optimized implementation policy used by cross-attention."""

    minimum_compute_capability: tuple[int, int] | None = None
    """Minimum CUDA compute capability; ``None`` accepts any CUDA device."""

    @property
    def pytest_id(self) -> str:
        """Return the readable pytest parameter identifier."""
        return self.implementation.replace("_", "-")


BENCHMARK_CASES = (
    AttentionBenchmarkCase(
        implementation="torch",
        self_attention_backend=AttentionBackend.TORCH,
        cross_attention_backend=AttentionBackend.TORCH,
    ),
    # Manually tuned performance config for RTX PRO 6000.
    AttentionBenchmarkCase(
        implementation="optimized-rtx-pro-6000",
        self_attention_backend=AttentionBackend.OPTIMIZED,
        cross_attention_backend=AttentionBackend.OPTIMIZED,
        self_attn_optimized_impl_config=OptimizedImplConfig(
            qkv_fusion_option=QKVFusionOption.NONE,
            sdpa_backend=SDPABackend.FA2,
            use_tma=True,
            quantization=QuantizationOption(
                projection=torch.float8_e4m3fn, quantized_sdpa=True
            ),
        ),
        cross_attn_optimized_impl_config=OptimizedImplConfig(
            qkv_fusion_option=QKVFusionOption.NONE,
            sdpa_backend=SDPABackend.FA2,
            use_tma=True,
            quantization=QuantizationOption(
                projection=torch.float8_e4m3fn, quantized_sdpa=True
            ),
        ),
        minimum_compute_capability=(9, 0),
    ),
    # Manually tuned performance config for GB300.
    AttentionBenchmarkCase(
        implementation="optimized-gb300",
        self_attention_backend=AttentionBackend.OPTIMIZED,
        cross_attention_backend=AttentionBackend.TORCH,
        self_attn_optimized_impl_config=OptimizedImplConfig(
            qkv_fusion_option=QKVFusionOption.NONE,
            sdpa_backend=SDPABackend.CUDNN,
            use_tma=False,
            quantization=QuantizationOption(projection=None, quantized_sdpa=True),
        ),
        minimum_compute_capability=(9, 0),
    ),
)
"""Torch baseline and isolated/combined optimized attention cases."""


def skip_unsupported_device(
    case: AttentionBenchmarkCase,
    device: torch.device,
) -> None:
    """Skip a benchmark case when the device is older than its minimum capability."""
    minimum = case.minimum_compute_capability
    if minimum is None:
        return
    if torch.cuda.get_device_capability(device) < minimum:
        pytest.skip(
            f"{case.implementation} attention requires compute capability "
            f"{minimum[0]}.{minimum[1]}+"
        )
