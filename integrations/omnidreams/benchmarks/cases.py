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

"""Shared OmniDreams attention benchmark cases."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pytest
import torch
from omnidreams.transformer.impl.modules import AttentionBackend

from flashdreams.accelerated.multi_head_attention.optimized import (
    OptimizedImplConfig,
    QKVFusionOption,
    QuantizationOption,
    SDPABackend,
)


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
    """Optimized implementation policy used by accelerated self-attention."""

    cross_attn_optimized_impl_config: OptimizedImplConfig = OptimizedImplConfig(
        qkv_fusion_option=QKVFusionOption.FUSE_KV,
        sdpa_backend=SDPABackend.FA2,
    )
    """Optimized implementation policy used by accelerated cross-attention."""

    native_dit: bool = False
    """Whether the full-pipeline case bypasses the PyTorch network."""

    native_dit_backend: Literal["fp8_kvcache_cudnn", "bf16"] = "fp8_kvcache_cudnn"
    """Native DiT compute backend used when ``native_dit`` is enabled."""

    native_attention_backend: Literal["cudnn", "sparge", "sage3", "sage3_fp8"] = "cudnn"
    """Native attention backend used when ``native_dit`` is enabled."""

    minimum_compute_capability: tuple[int, int] | None = None
    """Minimum CUDA compute capability; ``None`` accepts any CUDA device."""

    @property
    def pytest_id(self) -> str:
        """Return the readable pytest parameter identifier."""
        return self.implementation.replace("_", "-")


BENCHMARK_CASES = [
    # Pytorch reference implementation.
    AttentionBenchmarkCase(
        implementation="omnidreams_torch",
        self_attention_backend=AttentionBackend.OMNIDREAMS,
        cross_attention_backend=AttentionBackend.OMNIDREAMS,
    ),
    # Selected production-shaped pair from the recorded GB300 MHA benchmark.
    AttentionBenchmarkCase(
        implementation="optimized_cudnn_fp8_self_full_no_tma_cross_none_tma",
        self_attention_backend=AttentionBackend.OPTIMIZED,
        cross_attention_backend=AttentionBackend.OPTIMIZED,
        self_attn_optimized_impl_config=OptimizedImplConfig(
            qkv_fusion_option=QKVFusionOption.FULL,
            sdpa_backend=SDPABackend.CUDNN,
            use_tma=False,
            quantization=QuantizationOption(
                projection=None,
                quantized_sdpa=True,
            ),
        ),
        cross_attn_optimized_impl_config=OptimizedImplConfig(
            qkv_fusion_option=QKVFusionOption.FUSE_KV,
            sdpa_backend=SDPABackend.FA2,
            use_tma=True,
            quantization=QuantizationOption(projection=None, quantized_sdpa=False),
        ),
        minimum_compute_capability=(9, 0),
    ),
    # RTX PRO 6000 quantized SDPA pair with e4m3 projections enabled.
    AttentionBenchmarkCase(
        implementation="optimized_fa2_quantized_sdpa_self_full_tma_cross_none_tma",
        self_attention_backend=AttentionBackend.OPTIMIZED,
        cross_attention_backend=AttentionBackend.OMNIDREAMS,
        self_attn_optimized_impl_config=OptimizedImplConfig(
            qkv_fusion_option=QKVFusionOption.FULL,
            sdpa_backend=SDPABackend.FA2,
            use_tma=True,
            quantization=QuantizationOption(
                projection=torch.float8_e4m3fn,
                quantized_sdpa=True,
            ),
        ),
        minimum_compute_capability=(9, 0),
    ),
    # Native CUDA implementation.
    AttentionBenchmarkCase(
        implementation="cuda",
        self_attention_backend=AttentionBackend.OMNIDREAMS,
        cross_attention_backend=AttentionBackend.OMNIDREAMS,
        native_dit=True,
    ),
    AttentionBenchmarkCase(
        implementation="cuda_sparge",
        self_attention_backend=AttentionBackend.OMNIDREAMS,
        cross_attention_backend=AttentionBackend.OMNIDREAMS,
        native_dit=True,
        native_attention_backend="sparge",
        minimum_compute_capability=(12, 0),
    ),
    AttentionBenchmarkCase(
        implementation="cuda_sage3",
        self_attention_backend=AttentionBackend.OMNIDREAMS,
        cross_attention_backend=AttentionBackend.OMNIDREAMS,
        native_dit=True,
        native_dit_backend="bf16",
        native_attention_backend="sage3",
        minimum_compute_capability=(12, 0),
    ),
    AttentionBenchmarkCase(
        implementation="cuda_sage3_fp8",
        self_attention_backend=AttentionBackend.OMNIDREAMS,
        cross_attention_backend=AttentionBackend.OMNIDREAMS,
        native_dit=True,
        native_attention_backend="sage3_fp8",
        minimum_compute_capability=(12, 0),
    ),
]
"""Attention cases exercised by the OmniDreams benchmarks.

Full QKV fusion only applies to self-attention because production text
cross-attention has unequal query and context widths.
"""


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
            f"{case.implementation} attention requires compute capability "
            f"{minimum[0]}.{minimum[1]}+"
        )
