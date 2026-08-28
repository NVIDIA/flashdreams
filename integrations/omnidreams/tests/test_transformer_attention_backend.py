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
from omnidreams.transformer import CosmosTransformerConfig
from omnidreams.transformer.impl import modules as transformer_modules
from omnidreams.transformer.impl.modules import AttentionBackend, Block
from omnidreams.transformer.impl.network import CosmosDiTNetwork, CosmosDiTNetworkConfig

from flashdreams.accelerated.multi_head_attention import (
    AttentionType,
)
from flashdreams.accelerated.multi_head_attention import (
    optimized as optimized_attention,
)
from flashdreams.accelerated.multi_head_attention.optimized import (
    OptimizedImplConfig,
    OptimizedMultiHeadAttention,
    QKVFusionOption,
    QuantizationOption,
    SDPABackend,
)
from integrations.omnidreams.benchmarks.cases import BENCHMARK_CASES
from integrations.omnidreams.benchmarks.test_modules import (
    _CUDNN_OPTIMIZED_IMPL_CONFIGS,
    _FA2_OPTIMIZED_IMPL_CONFIGS,
    _FULL_POLICY_SEARCH,
    _MODULE_BLOCK_CASES,
    _MODULE_CROSS_ATTENTION_CONFIGS,
    _MODULE_SELF_ATTENTION_CONFIGS,
    _OPTIMIZED_IMPL_CONFIGS,
    _implementation_id,
    _module_block_cases,
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
        (AttentionBackend.OPTIMIZED, AttentionBackend.OMNIDREAMS),
        (AttentionBackend.OMNIDREAMS, AttentionBackend.OPTIMIZED),
    ),
    ids=("optimized-self", "optimized-cross"),
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
    )

    block = CosmosDiTNetwork(config).blocks[0]

    expected_self_type = (
        transformer_modules.OptimizedSelfAttention
        if self_attention_backend is AttentionBackend.OPTIMIZED
        else transformer_modules.SelfAttention
    )
    expected_cross_type = (
        transformer_modules.OptimizedCrossAttention
        if cross_attention_backend is AttentionBackend.OPTIMIZED
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


def test_optimized_attention_caches_cuda_capability(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cache CUDA capability validation per device."""
    attention = transformer_modules.OptimizedSelfAttention(
        query_dim=32,
        n_heads=2,
        head_dim=16,
    )
    validated_devices: list[int] = []

    monkeypatch.setattr(torch.cuda, "current_device", lambda: 3)

    def get_device_capability(device: int) -> tuple[int, int]:
        validated_devices.append(device)
        return (9, 0)

    monkeypatch.setattr(torch.cuda, "get_device_capability", get_device_capability)

    attention._validate_cuda_device("cuda")
    attention._validate_cuda_device("cuda:3")

    assert validated_devices == [3]


@pytest.mark.parametrize(
    "sdpa_backend", tuple(SDPABackend), ids=lambda backend: backend.value
)
def test_network_config_selects_optimized_attention(
    sdpa_backend: SDPABackend,
) -> None:
    """Propagate the configured self-attention SDPA implementation."""
    optimized_impl_config = OptimizedImplConfig(
        qkv_fusion_option=QKVFusionOption.FULL,
        sdpa_backend=sdpa_backend,
        use_tma=False,
    )
    config = CosmosDiTNetworkConfig(
        model_channels=32,
        num_blocks=1,
        num_heads=2,
        crossattn_emb_channels=16,
        use_crossattn_projection=False,
        enable_cross_view_attn=True,
        self_attention_backend=AttentionBackend.OPTIMIZED,
        cross_attention_backend=AttentionBackend.OPTIMIZED,
        self_attn_optimized_impl_config=optimized_impl_config,
    )

    network = CosmosDiTNetwork(config)
    block = network.blocks[0]

    assert config.self_attn_optimized_impl_config is optimized_impl_config
    assert network.self_attn_optimized_impl_config is optimized_impl_config
    assert block.self_attn_optimized_impl_config is optimized_impl_config
    self_attention = block.self_attn
    assert isinstance(self_attention, OptimizedMultiHeadAttention)
    assert self_attention.attention_type is AttentionType.SELF_ATTENTION
    assert self_attention.optimized_impl_config is optimized_impl_config
    assert self_attention.qkv_fusion_option is QKVFusionOption.FULL
    assert self_attention.sdpa_backend is sdpa_backend
    assert self_attention.use_tma is False
    assert isinstance(self_attention.fused_qkv, torch.nn.Linear)
    assert isinstance(self_attention.fused_kv, torch.nn.Linear)
    cache = self_attention.allocate_kv_cache(
        batch_size=1,
        chunk_size=2,
        window_size=4,
        sink_size=0,
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
    )
    assert cache.dtype is torch.bfloat16
    assert cache._k.is_contiguous()
    assert cache._v.is_contiguous()
    assert isinstance(block.cross_attn, transformer_modules.OptimizedCrossAttention)
    assert isinstance(
        block.cross_view_attn, transformer_modules.OptimizedCrossAttention
    )
    assert block.cross_attn.attention_config.context_dim == 16
    assert block.cross_attn.attention_type is AttentionType.CROSS_ATTENTION
    assert block.cross_attn.qkv_fusion_option is QKVFusionOption.FUSE_KV
    assert block.cross_attn.sdpa_backend is SDPABackend.FA2
    assert block.cross_view_attn.attention_config.context_dim == 32
    assert block.cross_view_attn.attention_type is AttentionType.CROSS_ATTENTION
    assert block.cross_view_attn.qkv_fusion_option is QKVFusionOption.FUSE_KV
    assert block.cross_view_attn.sdpa_backend is SDPABackend.FA2


def test_network_config_selects_optimized_attention_policies() -> None:
    """Propagate cross-attention SDPA and QKV fusion policies."""
    self_optimized_impl_config = OptimizedImplConfig(
        qkv_fusion_option=QKVFusionOption.NONE,
        sdpa_backend=SDPABackend.FA2,
        use_tma=False,
    )
    cross_optimized_impl_config = OptimizedImplConfig(
        qkv_fusion_option=QKVFusionOption.NONE,
        sdpa_backend=SDPABackend.CUDNN,
        use_tma=False,
    )

    config = CosmosDiTNetworkConfig(
        model_channels=32,
        num_blocks=1,
        num_heads=2,
        crossattn_emb_channels=16,
        use_crossattn_projection=False,
        enable_cross_view_attn=True,
        self_attention_backend=AttentionBackend.OPTIMIZED,
        cross_attention_backend=AttentionBackend.OPTIMIZED,
        self_attn_optimized_impl_config=self_optimized_impl_config,
        cross_attn_optimized_impl_config=cross_optimized_impl_config,
    )

    network = CosmosDiTNetwork(config)
    block = network.blocks[0]
    assert isinstance(block, Block)
    self_attention = block.self_attn
    cross_attention = block.cross_attn
    cross_view_attention = block.cross_view_attn
    assert isinstance(self_attention, OptimizedMultiHeadAttention)
    assert isinstance(cross_attention, OptimizedMultiHeadAttention)
    assert isinstance(cross_view_attention, OptimizedMultiHeadAttention)

    assert network.self_attn_optimized_impl_config is self_optimized_impl_config
    assert network.cross_attn_optimized_impl_config is cross_optimized_impl_config
    assert block.self_attn_optimized_impl_config is self_optimized_impl_config
    assert block.cross_attn_optimized_impl_config is cross_optimized_impl_config
    assert self_attention.optimized_impl_config is self_optimized_impl_config
    assert cross_attention.optimized_impl_config is cross_optimized_impl_config
    assert cross_view_attention.optimized_impl_config is cross_optimized_impl_config
    assert not self_attention.use_tma and not cross_attention.use_tma
    assert self_attention.sdpa_backend is SDPABackend.FA2
    assert cross_attention.sdpa_backend is SDPABackend.CUDNN
    assert cross_view_attention.sdpa_backend is SDPABackend.CUDNN
    assert self_attention.qkv_fusion_option is QKVFusionOption.NONE
    assert cross_attention.qkv_fusion_option is QKVFusionOption.NONE
    assert cross_view_attention.qkv_fusion_option is QKVFusionOption.NONE


def test_benchmark_cases_match_selected_matrix() -> None:
    """Keep the end-to-end benchmark matrix limited to selected configurations."""
    pytorch_cases = [case for case in BENCHMARK_CASES if not case.native_dit]
    assert tuple(
        (
            case.implementation,
            case.self_attention_backend,
            case.cross_attention_backend,
            case.self_attn_optimized_impl_config.sdpa_backend,
            case.self_attn_optimized_impl_config.qkv_fusion_option,
            case.cross_attn_optimized_impl_config.qkv_fusion_option,
        )
        for case in pytorch_cases
    ) == (
        (
            "omnidreams_torch",
            AttentionBackend.OMNIDREAMS,
            AttentionBackend.OMNIDREAMS,
            SDPABackend.FA2,
            QKVFusionOption.FULL,
            QKVFusionOption.FUSE_KV,
        ),
        (
            "optimized_cudnn_fp8_self_full_no_tma_cross_none_tma",
            AttentionBackend.OPTIMIZED,
            AttentionBackend.OPTIMIZED,
            SDPABackend.CUDNN,
            QKVFusionOption.FULL,
            QKVFusionOption.FUSE_KV,
        ),
        (
            "optimized_fa2_quantized_sdpa_self_full_tma_cross_none_tma",
            AttentionBackend.OPTIMIZED,
            AttentionBackend.OMNIDREAMS,
            SDPABackend.FA2,
            QKVFusionOption.FULL,
            QKVFusionOption.FUSE_KV,
        ),
    )
    gb300_best = pytorch_cases[1]
    assert gb300_best.self_attn_optimized_impl_config == OptimizedImplConfig(
        qkv_fusion_option=QKVFusionOption.FULL,
        sdpa_backend=SDPABackend.CUDNN,
        use_tma=False,
        quantization=QuantizationOption(
            projection=None,
            quantized_sdpa=True,
        ),
    )
    assert gb300_best.cross_attn_optimized_impl_config == OptimizedImplConfig(
        qkv_fusion_option=QKVFusionOption.FUSE_KV,
        sdpa_backend=SDPABackend.FA2,
        use_tma=True,
        quantization=QuantizationOption(
            projection=None,
            quantized_sdpa=False,
        ),
    )
    assert gb300_best.minimum_compute_capability == (9, 0)

    rtx_pro_6000_best = pytorch_cases[2]
    assert rtx_pro_6000_best.self_attn_optimized_impl_config == OptimizedImplConfig(
        qkv_fusion_option=QKVFusionOption.FULL,
        sdpa_backend=SDPABackend.FA2,
        use_tma=True,
        quantization=QuantizationOption(
            projection=torch.float8_e4m3fn,
            quantized_sdpa=True,
        ),
    )
    assert rtx_pro_6000_best.cross_attn_optimized_impl_config == OptimizedImplConfig(
        qkv_fusion_option=QKVFusionOption.FUSE_KV,
        sdpa_backend=SDPABackend.FA2,
        use_tma=True,
        quantization=QuantizationOption(
            projection=None,
            quantized_sdpa=False,
        ),
    )
    assert rtx_pro_6000_best.minimum_compute_capability == (9, 0)

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
        ("cuda_sparge", "fp8_kvcache_cudnn", "sparge", (12, 0)),
        ("cuda_sage3", "bf16", "sage3", (12, 0)),
        ("cuda_sage3_fp8", "fp8_kvcache_cudnn", "sage3_fp8", (12, 0)),
    )
    assert len({case.pytest_id for case in BENCHMARK_CASES}) == len(BENCHMARK_CASES)


def test_module_benchmark_configs_cover_attention_policy_matrix() -> None:
    """Keep exhaustive policies isolated and full-block pairs representative."""
    expected_cudnn_configs = {
        OptimizedImplConfig(
            sdpa_backend=SDPABackend.CUDNN,
            qkv_fusion_option=qkv_fusion_option,
            use_tma=False,
            quantization=QuantizationOption(
                projection=projection_dtype,
                quantized_sdpa=quantized_sdpa,
            ),
        )
        for qkv_fusion_option in QKVFusionOption
        for projection_dtype in (None, torch.float8_e4m3fn)
        for quantized_sdpa in (False, True)
    }
    expected_fa2_configs = {
        OptimizedImplConfig(
            sdpa_backend=SDPABackend.FA2,
            qkv_fusion_option=qkv_fusion_option,
            use_tma=use_tma,
            quantization=QuantizationOption(
                projection=projection_dtype,
                quantized_sdpa=quantized_sdpa,
            ),
        )
        for qkv_fusion_option in QKVFusionOption
        for use_tma in (False, True)
        for projection_dtype in (None, torch.float8_e4m3fn)
        for quantized_sdpa in (False, True)
    }
    expected_configs = expected_cudnn_configs | expected_fa2_configs

    assert set(_CUDNN_OPTIMIZED_IMPL_CONFIGS) == expected_cudnn_configs
    assert set(_FA2_OPTIMIZED_IMPL_CONFIGS) == expected_fa2_configs
    assert set(_OPTIMIZED_IMPL_CONFIGS) == expected_configs
    assert _MODULE_SELF_ATTENTION_CONFIGS[0] is None
    assert set(_MODULE_SELF_ATTENTION_CONFIGS[1:]) == expected_configs

    expected_cross_configs = {
        config
        for config in expected_configs
        if config.qkv_fusion_option is not QKVFusionOption.FULL
    }
    assert _MODULE_CROSS_ATTENTION_CONFIGS[0] is None
    assert set(_MODULE_CROSS_ATTENTION_CONFIGS[1:]) == expected_cross_configs

    representative_block_cases = _module_block_cases(False)
    assert representative_block_cases == tuple(
        case for case in BENCHMARK_CASES if not case.native_dit
    )
    assert tuple(
        case.minimum_compute_capability for case in representative_block_cases
    ) == (None, (9, 0), (9, 0))

    full_block_cases = _module_block_cases(True)
    assert len(full_block_cases) == 37 * 25
    assert len({case.pytest_id for case in full_block_cases}) == len(full_block_cases)
    assert all(
        case.minimum_compute_capability
        == (
            (9, 0)
            if AttentionBackend.OPTIMIZED
            in (case.self_attention_backend, case.cross_attention_backend)
            else None
        )
        for case in full_block_cases
    )
    assert _MODULE_BLOCK_CASES == (
        full_block_cases if _FULL_POLICY_SEARCH else representative_block_cases
    )
    assert len(_MODULE_SELF_ATTENTION_CONFIGS) == 37
    assert len(_MODULE_CROSS_ATTENTION_CONFIGS) == 25
    assert len(
        {_implementation_id(config) for config in _MODULE_SELF_ATTENTION_CONFIGS}
    ) == len(_MODULE_SELF_ATTENTION_CONFIGS)
    assert len(
        {_implementation_id(config) for config in _MODULE_CROSS_ATTENTION_CONFIGS}
    ) == len(_MODULE_CROSS_ATTENTION_CONFIGS)


def test_optimized_backend_preserves_checkpoint_keys() -> None:
    """Load Omnidreams weights into the optimized block strictly."""
    omnidreams_block = Block(x_dim=32, context_dim=16, num_heads=2)
    optimized_block = Block(
        x_dim=32,
        context_dim=16,
        num_heads=2,
        self_attention_backend=AttentionBackend.OPTIMIZED,
        cross_attention_backend=AttentionBackend.OPTIMIZED,
    )

    optimized_block.load_state_dict(omnidreams_block.state_dict(), strict=True)


def test_optimized_cross_attention_dispatches_tma(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep the cross-attention adapter on the optimized backend-owned forward."""
    torch.manual_seed(0)
    optimized_block = Block(
        x_dim=32,
        context_dim=16,
        num_heads=2,
        self_attention_backend=AttentionBackend.OPTIMIZED,
        cross_attention_backend=AttentionBackend.OPTIMIZED,
    )
    optimized_cross_attention = optimized_block.cross_attn
    assert isinstance(
        optimized_cross_attention, transformer_modules.OptimizedCrossAttention
    )
    assert optimized_cross_attention.attention_type is AttentionType.CROSS_ATTENTION
    assert (
        type(optimized_cross_attention).forward is OptimizedMultiHeadAttention.forward
    )
    assert (
        type(optimized_cross_attention).compute_kv
        is OptimizedMultiHeadAttention.compute_kv
    )
    assert (
        type(optimized_cross_attention)._attention
        is OptimizedMultiHeadAttention._attention
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
        optimized_attention,
        "is_tma_flash_attention_supported",
        lambda *_: True,
    )
    monkeypatch.setattr(
        optimized_attention,
        "flash_attention_2_tma",
        record_tma_attention,
    )
    query = torch.randn(2, 3, 2, 16)
    key = torch.randn(2, 5, 2, 16)
    value = torch.randn(2, 5, 2, 16)
    optimized_output = optimized_cross_attention._attention(query, key, value)

    assert calls == [
        (
            torch.Size([2, 3, 2, 16]),
            torch.Size([2, 5, 2, 16]),
            torch.Size([2, 5, 2, 16]),
        )
    ]
    assert optimized_output.shape == query.shape
    assert torch.isfinite(optimized_output).all()
