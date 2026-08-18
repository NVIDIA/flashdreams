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

from flashdreams.accelerated.multi_head_attention_triton import (
    QKVFusionOption,
    SDPABackend,
)


@dataclass(frozen=True)
class AttentionBenchmarkCase:
    """Configuration and metadata for one attention benchmark implementation."""

    implementation: str
    """Stable implementation name stored in benchmark metadata."""

    self_attention_backend: AttentionBackend
    """Self-attention implementation configured for this case."""

    cross_attention_backend: AttentionBackend
    """Cross-attention implementation configured for this case."""

    sdpa_backend: SDPABackend
    """SDPA backend configured for accelerated self-attention."""

    self_attention_operator: str
    """Self-attention operator reported in benchmark metadata."""

    cross_attention_operator: str
    """Cross-attention operator reported in benchmark metadata."""

    use_fp8: bool = True
    """Whether accelerated projections and supported attention storage use FP8."""

    self_attn_qkv_fusion_option: QKVFusionOption = QKVFusionOption.FULL
    """Projection fusion policy used by accelerated self-attention."""

    cross_attn_qkv_fusion_option: QKVFusionOption = QKVFusionOption.FUSE_KV
    """Projection fusion policy used by accelerated cross-attention."""

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
    AttentionBenchmarkCase(
        implementation="omnidreams_torch",
        self_attention_backend=AttentionBackend.OMNIDREAMS,
        cross_attention_backend=AttentionBackend.OMNIDREAMS,
        sdpa_backend=SDPABackend.CUDNN,
        self_attention_operator="cudnn",
        cross_attention_operator="cudnn",
        use_fp8=False,
        self_attn_qkv_fusion_option=QKVFusionOption.NONE,
        cross_attn_qkv_fusion_option=QKVFusionOption.NONE,
    ),
    AttentionBenchmarkCase(
        implementation="triton_fa2_fp8_full",
        self_attention_backend=AttentionBackend.TRITON,
        cross_attention_backend=AttentionBackend.TRITON,
        sdpa_backend=SDPABackend.TRITON,
        self_attention_operator="triton_fa2",
        cross_attention_operator="triton_fa2",
        minimum_compute_capability=(9, 0),
    ),
    AttentionBenchmarkCase(
        implementation="triton_cudnn_bf16_full",
        self_attention_backend=AttentionBackend.TRITON,
        cross_attention_backend=AttentionBackend.TRITON,
        sdpa_backend=SDPABackend.CUDNN,
        self_attention_operator="torch_cudnn_sdpa",
        cross_attention_operator="torch_cudnn_sdpa",
        use_fp8=False,
    ),
    AttentionBenchmarkCase(
        implementation="triton_cudnn_bf16_full_omnidreams_cross",
        self_attention_backend=AttentionBackend.TRITON,
        cross_attention_backend=AttentionBackend.OMNIDREAMS,
        sdpa_backend=SDPABackend.CUDNN,
        self_attention_operator="torch_cudnn_sdpa",
        cross_attention_operator="cudnn",
        use_fp8=False,
        cross_attn_qkv_fusion_option=QKVFusionOption.NONE,
    ),
    AttentionBenchmarkCase(
        implementation="cuda",
        self_attention_backend=AttentionBackend.OMNIDREAMS,
        cross_attention_backend=AttentionBackend.OMNIDREAMS,
        sdpa_backend=SDPABackend.CUDNN,
        self_attention_operator="cudnn",
        cross_attention_operator="cudnn",
        use_fp8=False,
        self_attn_qkv_fusion_option=QKVFusionOption.NONE,
        cross_attn_qkv_fusion_option=QKVFusionOption.NONE,
        native_dit=True,
    ),
    AttentionBenchmarkCase(
        implementation="cuda_sparge",
        self_attention_backend=AttentionBackend.OMNIDREAMS,
        cross_attention_backend=AttentionBackend.OMNIDREAMS,
        sdpa_backend=SDPABackend.CUDNN,
        self_attention_operator="sparge",
        cross_attention_operator="sparge",
        use_fp8=False,
        self_attn_qkv_fusion_option=QKVFusionOption.NONE,
        cross_attn_qkv_fusion_option=QKVFusionOption.NONE,
        native_dit=True,
        native_attention_backend="sparge",
        minimum_compute_capability=(12, 0),
    ),
    AttentionBenchmarkCase(
        implementation="cuda_sage3",
        self_attention_backend=AttentionBackend.OMNIDREAMS,
        cross_attention_backend=AttentionBackend.OMNIDREAMS,
        sdpa_backend=SDPABackend.CUDNN,
        self_attention_operator="sage3",
        cross_attention_operator="sage3",
        use_fp8=False,
        self_attn_qkv_fusion_option=QKVFusionOption.NONE,
        cross_attn_qkv_fusion_option=QKVFusionOption.NONE,
        native_dit=True,
        native_dit_backend="bf16",
        native_attention_backend="sage3",
        minimum_compute_capability=(12, 0),
    ),
    AttentionBenchmarkCase(
        implementation="cuda_sage3_fp8",
        self_attention_backend=AttentionBackend.OMNIDREAMS,
        cross_attention_backend=AttentionBackend.OMNIDREAMS,
        sdpa_backend=SDPABackend.CUDNN,
        self_attention_operator="sage3_fp8",
        cross_attention_operator="sage3_fp8",
        use_fp8=False,
        self_attn_qkv_fusion_option=QKVFusionOption.NONE,
        cross_attn_qkv_fusion_option=QKVFusionOption.NONE,
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
            f"{case.pytest_id} attention requires compute capability "
            f"{minimum[0]}.{minimum[1]}+"
        )
