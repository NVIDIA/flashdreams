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

"""CPU coverage for Omnidreams DiT attention backend selection."""

import pytest
import torch
from omnidreams.transformer.impl import modules as transformer_modules
from omnidreams.transformer.impl.modules import AttentionBackend, Block
from omnidreams.transformer.impl.network import CosmosDiTNetwork, CosmosDiTNetworkConfig

from flashdreams.accelerated import multi_head_attention_triton as triton_attention
from flashdreams.accelerated.multi_head_attention import (
    AttentionType,
)
from flashdreams.accelerated.multi_head_attention_triton import (
    QKVFusionOption,
    SDPABackend,
    TritonMultiHeadAttention,
)
from integrations.omnidreams.benchmarks.cases import BENCHMARK_CASES
from integrations.omnidreams.benchmarks.test_modules import (
    _MODULE_BENCHMARK_CASES,
    _MODULE_CROSS_ATTENTION_CASES,
    _MODULE_SELF_ATTENTION_CASES,
)

pytestmark = pytest.mark.ci_cpu


def test_dit_attention_backend_defaults_to_omnidreams() -> None:
    """Keep existing Omnidreams attention as the default."""
    default_block = Block(
        x_dim=12,
        context_dim=8,
        num_heads=1,
        enable_cross_view_attn=True,
    )

    assert default_block.self_attention_backend is AttentionBackend.OMNIDREAMS
    assert default_block.cross_attention_backend is AttentionBackend.OMNIDREAMS
    assert transformer_modules.MultiHeadAttention.__base__ is torch.nn.Module
    assert isinstance(default_block.self_attn, transformer_modules.SelfAttention)
    assert isinstance(default_block.cross_attn, transformer_modules.CrossAttention)


@pytest.mark.parametrize(
    ("self_attention_backend", "cross_attention_backend"),
    (
        (AttentionBackend.TRITON, AttentionBackend.OMNIDREAMS),
        (AttentionBackend.OMNIDREAMS, AttentionBackend.TRITON),
    ),
    ids=("triton-self", "triton-cross"),
)
def test_network_config_selects_attention_backends_independently(
    self_attention_backend: AttentionBackend,
    cross_attention_backend: AttentionBackend,
) -> None:
    """Select self- and cross-attention implementations independently."""
    config = CosmosDiTNetworkConfig(
        model_channels=32,
        num_blocks=1,
        num_heads=2,
        crossattn_emb_channels=16,
        use_crossattn_projection=False,
        enable_cross_view_attn=True,
        self_attention_backend=self_attention_backend,
        cross_attention_backend=cross_attention_backend,
        sdpa_backend=SDPABackend.CUDNN,
        use_fp8=False,
    )

    block = CosmosDiTNetwork(config).blocks[0]

    expected_self_type = (
        transformer_modules.TritonSelfAttention
        if self_attention_backend is AttentionBackend.TRITON
        else transformer_modules.SelfAttention
    )
    expected_cross_type = (
        transformer_modules.TritonCrossAttention
        if cross_attention_backend is AttentionBackend.TRITON
        else transformer_modules.CrossAttention
    )
    assert block.self_attention_backend is self_attention_backend
    assert block.cross_attention_backend is cross_attention_backend
    assert isinstance(block.self_attn, expected_self_type)
    assert isinstance(block.cross_attn, expected_cross_type)
    assert isinstance(block.cross_view_attn, expected_cross_type)


def test_omnidreams_attention_preserves_cache_lifecycles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Update self-attention cache while keeping cross-attention cache static."""

    def cpu_sdpa(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        return torch.nn.functional.scaled_dot_product_attention(query, key, value)

    self_attention = transformer_modules.SelfAttention(
        query_dim=16,
        n_heads=1,
        head_dim=16,
    )
    monkeypatch.setattr(
        transformer_modules, "apply_rope_freqs", lambda tensor, _: tensor
    )
    monkeypatch.setattr(self_attention.attn_op, "_impl", cpu_sdpa)
    self_cache = self_attention.allocate_kv_cache(
        batch_size=2,
        chunk_size=3,
        window_size=6,
        sink_size=0,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    query = torch.randn(1, 2, 3, 16)
    self_cache.before_update(0)
    output = self_attention(
        query,
        self_cache,
        rope_freqs=torch.zeros(3, 1, 1, 16),
    )
    assert self_cache.cached_k().shape == (2, 3, 1, 16)
    self_cache.after_update(0)
    assert output.shape == query.shape

    cross_attention = transformer_modules.CrossAttention(
        query_dim=16,
        context_dim=16,
        n_heads=1,
        head_dim=16,
    )
    monkeypatch.setattr(cross_attention.attn_op, "_impl", cpu_sdpa)
    cross_cache = cross_attention.compute_kv(torch.randn(1, 2, 5, 16))
    cached_key = cross_cache.cached_k().clone()
    cached_value = cross_cache.cached_v().clone()
    output = cross_attention(query, cross_cache)
    assert output.shape == query.shape
    torch.testing.assert_close(cross_cache.cached_k(), cached_key)
    torch.testing.assert_close(cross_cache.cached_v(), cached_value)


@pytest.mark.parametrize(
    "sdpa_backend", tuple(SDPABackend), ids=lambda backend: backend.value
)
def test_network_config_selects_triton_attention(
    sdpa_backend: SDPABackend,
) -> None:
    """Propagate the configured self-attention SDPA implementation."""
    config = CosmosDiTNetworkConfig(
        model_channels=32,
        num_blocks=1,
        num_heads=2,
        crossattn_emb_channels=16,
        use_crossattn_projection=False,
        enable_cross_view_attn=True,
        self_attention_backend=AttentionBackend.TRITON,
        cross_attention_backend=AttentionBackend.TRITON,
        sdpa_backend=sdpa_backend,
    )

    network = CosmosDiTNetwork(config)
    block = network.blocks[0]

    assert config.sdpa_backend is sdpa_backend
    assert network.sdpa_backend is sdpa_backend
    assert block.sdpa_backend is sdpa_backend
    self_attention = block.self_attn
    assert isinstance(self_attention, TritonMultiHeadAttention)
    assert self_attention.use_fp8 is True
    assert self_attention.attention_type is AttentionType.SELF_ATTENTION
    assert self_attention.qkv_fusion_option is QKVFusionOption.FULL
    assert self_attention.sdpa_backend is sdpa_backend
    assert self_attention._derived_weights.fused_qkv_weight is not None
    assert self_attention._derived_weights.fused_qkv_weight.dtype == torch.float8_e4m3fn
    assert self_attention._derived_weights.output_weight_fp8 is not None
    assert (
        self_attention._derived_weights.output_weight_fp8.dtype == torch.float8_e4m3fn
    )
    cache = self_attention.allocate_kv_cache(
        batch_size=1,
        chunk_size=2,
        window_size=4,
        sink_size=0,
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
    )
    assert cache.dtype is (
        torch.float8_e4m3fn if sdpa_backend is SDPABackend.TRITON else torch.bfloat16
    )
    with pytest.raises(TypeError, match="FP8 projections require"):
        self_attention.allocate_kv_cache(
            batch_size=1,
            chunk_size=2,
            window_size=4,
            sink_size=0,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
    assert isinstance(block.cross_attn, transformer_modules.TritonCrossAttention)
    assert isinstance(block.cross_view_attn, transformer_modules.TritonCrossAttention)
    assert block.cross_attn.context_dim == 16
    assert block.cross_attn.attention_type is AttentionType.CROSS_ATTENTION
    assert block.cross_attn.qkv_fusion_option is QKVFusionOption.FUSE_KV
    assert block.cross_attn.use_fp8 is True
    assert block.cross_attn.sdpa_backend is SDPABackend.TRITON
    assert block.cross_view_attn.context_dim == 32
    assert block.cross_view_attn.attention_type is AttentionType.CROSS_ATTENTION
    assert block.cross_view_attn.qkv_fusion_option is QKVFusionOption.FUSE_KV
    assert block.cross_view_attn.use_fp8 is True
    assert block.cross_view_attn.sdpa_backend is SDPABackend.TRITON


def test_network_config_selects_triton_attention_policies() -> None:
    """Propagate cross-attention SDPA, QKV fusion, and FP8 policies."""
    config = CosmosDiTNetworkConfig(
        model_channels=32,
        num_blocks=1,
        num_heads=2,
        crossattn_emb_channels=16,
        use_crossattn_projection=False,
        enable_cross_view_attn=True,
        self_attention_backend=AttentionBackend.TRITON,
        cross_attention_backend=AttentionBackend.TRITON,
        cross_attn_sdpa_backend=SDPABackend.CUDNN,
        self_attn_qkv_fusion_option=QKVFusionOption.NONE,
        cross_attn_qkv_fusion_option=QKVFusionOption.NONE,
        use_fp8=False,
    )

    network = CosmosDiTNetwork(config)
    block = network.blocks[0]
    assert isinstance(block, Block)
    self_attention = block.self_attn
    cross_attention = block.cross_attn
    cross_view_attention = block.cross_view_attn
    assert isinstance(self_attention, TritonMultiHeadAttention)
    assert isinstance(cross_attention, TritonMultiHeadAttention)
    assert isinstance(cross_view_attention, TritonMultiHeadAttention)

    assert network.cross_attn_sdpa_backend is SDPABackend.CUDNN
    assert network.self_attn_qkv_fusion_option is QKVFusionOption.NONE
    assert network.cross_attn_qkv_fusion_option is QKVFusionOption.NONE
    assert network.use_fp8 is False
    assert self_attention.sdpa_backend is SDPABackend.TRITON
    assert cross_attention.sdpa_backend is SDPABackend.CUDNN
    assert cross_view_attention.sdpa_backend is SDPABackend.CUDNN
    assert self_attention.qkv_fusion_option is QKVFusionOption.NONE
    assert cross_attention.qkv_fusion_option is QKVFusionOption.NONE
    assert cross_view_attention.qkv_fusion_option is QKVFusionOption.NONE
    assert self_attention.use_fp8 is False
    assert cross_attention.use_fp8 is False
    assert cross_view_attention.use_fp8 is False


def test_benchmark_cases_match_selected_matrix() -> None:
    """Keep the end-to-end benchmark matrix limited to selected configurations."""
    pytorch_cases = [case for case in BENCHMARK_CASES if not case.native_dit]
    assert tuple(
        (
            case.implementation,
            case.self_attention_backend,
            case.cross_attention_backend,
            case.sdpa_backend,
            case.use_fp8,
            case.self_attn_qkv_fusion_option,
            case.cross_attn_qkv_fusion_option,
        )
        for case in pytorch_cases
    ) == (
        (
            "omnidreams_torch",
            AttentionBackend.OMNIDREAMS,
            AttentionBackend.OMNIDREAMS,
            SDPABackend.CUDNN,
            False,
            QKVFusionOption.NONE,
            QKVFusionOption.NONE,
        ),
        (
            "triton_fa2_fp8_full",
            AttentionBackend.TRITON,
            AttentionBackend.TRITON,
            SDPABackend.TRITON,
            True,
            QKVFusionOption.FULL,
            QKVFusionOption.FUSE_KV,
        ),
        (
            "triton_cudnn_bf16_full",
            AttentionBackend.TRITON,
            AttentionBackend.TRITON,
            SDPABackend.CUDNN,
            False,
            QKVFusionOption.FULL,
            QKVFusionOption.FUSE_KV,
        ),
        (
            "triton_cudnn_bf16_full_omnidreams_cross",
            AttentionBackend.TRITON,
            AttentionBackend.OMNIDREAMS,
            SDPABackend.CUDNN,
            False,
            QKVFusionOption.FULL,
            QKVFusionOption.NONE,
        ),
    )

    native_cases = [case for case in BENCHMARK_CASES if case.native_dit]
    assert tuple(
        (
            case.implementation,
            case.native_dit_backend,
            case.native_attention_backend,
            case.minimum_compute_capability,
        )
        for case in native_cases
    ) == (
        ("cuda", "fp8_kvcache_cudnn", "cudnn", None),
        ("cuda_sparge", "fp8_kvcache_cudnn", "sparge", None),
        ("cuda_sage3", "bf16", "sage3", (12, 0)),
        ("cuda_sage3_fp8", "fp8_kvcache_cudnn", "sage3_fp8", (12, 0)),
    )
    assert len({case.pytest_id for case in BENCHMARK_CASES}) == len(BENCHMARK_CASES)


def test_module_benchmark_cases_cover_attention_policy_matrix() -> None:
    """Cover every module SDPA, precision, and QKV fusion combination."""
    triton_cases = [
        case
        for case in _MODULE_BENCHMARK_CASES
        if case.self_attention_backend is AttentionBackend.TRITON
        and case.cross_attention_backend is AttentionBackend.TRITON
    ]
    expected_cases = {
        (sdpa_backend, use_fp8, qkv_fusion_option)
        for sdpa_backend in SDPABackend
        for use_fp8 in (False, True)
        for qkv_fusion_option in QKVFusionOption
    }

    assert {
        (
            case.sdpa_backend,
            case.use_fp8,
            case.self_attn_qkv_fusion_option,
        )
        for case in triton_cases
    } == expected_cases
    assert len(_MODULE_BENCHMARK_CASES) == 14
    assert len(_MODULE_SELF_ATTENTION_CASES) == 13
    assert len(_MODULE_CROSS_ATTENTION_CASES) == 9


def test_triton_backend_preserves_checkpoint_keys() -> None:
    """Load Omnidreams weights into the Triton block strictly."""
    omnidreams_block = Block(x_dim=32, context_dim=16, num_heads=2)
    triton_block = Block(
        x_dim=32,
        context_dim=16,
        num_heads=2,
        self_attention_backend=AttentionBackend.TRITON,
        cross_attention_backend=AttentionBackend.TRITON,
    )

    triton_block.load_state_dict(omnidreams_block.state_dict(), strict=True)


def test_triton_cross_attention_dispatches_tma(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep the cross-attention adapter on Triton's backend-owned forward."""
    torch.manual_seed(0)
    triton_block = Block(
        x_dim=32,
        context_dim=16,
        num_heads=2,
        self_attention_backend=AttentionBackend.TRITON,
        cross_attention_backend=AttentionBackend.TRITON,
    )
    triton_cross_attention = triton_block.cross_attn
    assert isinstance(triton_cross_attention, transformer_modules.TritonCrossAttention)
    assert triton_cross_attention.attention_type is AttentionType.CROSS_ATTENTION
    assert type(triton_cross_attention).forward is TritonMultiHeadAttention.forward
    assert (
        type(triton_cross_attention).compute_kv is TritonMultiHeadAttention.compute_kv
    )
    assert (
        type(triton_cross_attention)._attention is TritonMultiHeadAttention._attention
    )

    calls: list[tuple[torch.Size, torch.Size, torch.Size]] = []

    def record_tma_attention(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        calls.append((query.shape, key.shape, value.shape))
        return torch.nn.functional.scaled_dot_product_attention(
            query.transpose(1, 2),
            key.transpose(1, 2),
            value.transpose(1, 2),
        ).transpose(1, 2)

    monkeypatch.setattr(
        triton_attention,
        "flash_attention_2_tma",
        record_tma_attention,
    )
    query = torch.randn(2, 3, 2, 16)
    key = torch.randn(2, 5, 2, 16)
    value = torch.randn(2, 5, 2, 16)
    triton_output = triton_cross_attention._attention(query, key, value)

    assert calls == [
        (
            torch.Size([2, 3, 2, 16]),
            torch.Size([2, 5, 2, 16]),
            torch.Size([2, 5, 2, 16]),
        )
    ]
    assert triton_output.shape == query.shape
    assert torch.isfinite(triton_output).all()
