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

"""Numerical parity tests for optimized and Torch multi-head attention."""

from __future__ import annotations

import pytest
import torch
from torch import Tensor

from flashdreams.accelerated.multi_head_attention import (
    AttentionConfig,
    AttentionType,
    QKNormScope,
    RoPEConfig,
    RoPEScope,
    RoPEStyle,
)
from flashdreams.accelerated.multi_head_attention.cudnn import native_cudnn_fp8_sdpa
from flashdreams.accelerated.multi_head_attention.optimized import (
    OptimizedImplConfig,
    OptimizedMultiHeadAttention,
    QKVFusionOption,
    QuantizationOption,
    SDPABackend,
)
from flashdreams.accelerated.multi_head_attention.torch import TorchMultiHeadAttention
from flashdreams.accelerated.quantization.linear import QuantizedNonPersistentLinear
from flashdreams.accelerated.quantization.quantizer import (
    DTYPE_MAX,
)
from flashdreams.core.attention import BlockKVCache

pytestmark = pytest.mark.ci_gpu

_QUERY_DIM = 128
_N_HEADS = 2
_HEAD_DIM = _QUERY_DIM // _N_HEADS
_CHUNK_SIZE = 16
_WINDOW_SIZE = 32
_SINK_SIZE = 4

_ROPE_CASES = (
    (None, "none"),
    *(
        (
            RoPEConfig(style=style, scope=scope),
            f"{style.value}-{scope.value}",
        )
        for style in RoPEStyle
        for scope in RoPEScope
    ),
)
"""Supported rotary policies and their pytest identifiers."""

_ATTENTION_CONFIGS = tuple(
    pytest.param(
        AttentionConfig(
            query_dim=_QUERY_DIM,
            n_heads=_N_HEADS,
            head_dim=_HEAD_DIM,
            qk_norm_scope=qk_norm_scope,
            rope_config=rope_config,
        ),
        id=f"norm-{qk_norm_scope.value}-rope-{rope_id}",
    )
    for qk_norm_scope in QKNormScope
    for rope_config, rope_id in _ROPE_CASES
)
"""Every supported normalization and rotary policy combination."""

_OPTIMIZED_IMPL_CONFIGS = tuple(
    pytest.param(
        OptimizedImplConfig(
            qkv_fusion_option=qkv_fusion_option,
            sdpa_backend=sdpa_backend,
            use_tma=use_tma,
        ),
        id=f"{sdpa_backend.value}-{qkv_fusion_option.value}-{'tma' if use_tma else 'no-tma'}",
    )
    for sdpa_backend in SDPABackend
    for qkv_fusion_option in QKVFusionOption
    for use_tma in (False, True)
)
"""Every supported optimized backend, fusion, and TMA preference combination."""


class _AttentionModules:
    """Provide checkpoint-compatible projection and normalization modules."""

    attention_config: AttentionConfig
    """Attention geometry and policies supplied by the concrete backend."""

    @property
    def query_projection(self) -> torch.nn.Linear:
        """Return the query projection."""
        return self.q_proj

    @property
    def key_projection(self) -> torch.nn.Linear:
        """Return the key projection."""
        return self.k_proj

    @property
    def value_projection(self) -> torch.nn.Linear:
        """Return the value projection."""
        return self.v_proj

    @property
    def output_projection(self) -> torch.nn.Linear:
        """Return the output projection."""
        return self.output_proj

    @property
    def query_norm(self) -> torch.nn.Module:
        """Return the query normalization module."""
        return self.q_norm

    @property
    def key_norm(self) -> torch.nn.Module:
        """Return the key normalization module."""
        return self.k_norm

    def _initialize_modules(self) -> None:
        """Initialize the shared checkpoint-compatible modules."""
        assert self.attention_config.context_dim is not None
        self.q_proj = torch.nn.Linear(
            self.attention_config.query_dim, self.attention_config.inner_dim, bias=True
        )
        self.k_proj = torch.nn.Linear(
            self.attention_config.context_dim,
            self.attention_config.inner_dim,
            bias=True,
        )
        self.v_proj = torch.nn.Linear(
            self.attention_config.context_dim,
            self.attention_config.inner_dim,
            bias=True,
        )
        self.output_proj = torch.nn.Linear(
            self.attention_config.inner_dim, self.attention_config.query_dim, bias=True
        )
        if self.attention_config.qk_norm_scope is QKNormScope.NONE:
            self.q_norm = torch.nn.Identity()
            self.k_norm = torch.nn.Identity()
            return
        norm_dim = (
            self.attention_config.head_dim
            if self.attention_config.qk_norm_scope is QKNormScope.HEAD
            else self.attention_config.inner_dim
        )
        self.q_norm = torch.nn.RMSNorm(norm_dim, eps=self.attention_config.qk_norm_eps)
        self.k_norm = torch.nn.RMSNorm(norm_dim, eps=self.attention_config.qk_norm_eps)


class _TorchMHA(_AttentionModules, TorchMultiHeadAttention):
    """Checkpoint-compatible Torch reference attention."""

    def __init__(
        self,
        attention_type: AttentionType,
        attention_config: AttentionConfig,
    ) -> None:
        """Initialize the Torch reference.

        Args:
            attention_type: Relationship between query and context tokens.
            attention_config: Shared attention geometry and policies.
        """
        super().__init__(attention_type, attention_config)
        self._initialize_modules()


class _OptimizedMHA(_AttentionModules, OptimizedMultiHeadAttention):
    """Checkpoint-compatible Optimized attention under test."""

    def __init__(
        self,
        attention_type: AttentionType,
        attention_config: AttentionConfig,
        optimized_impl_config: OptimizedImplConfig,
    ) -> None:
        """Initialize the Optimized implementation.

        Args:
            attention_type: Relationship between query and context tokens.
            attention_config: Shared attention geometry and policies.
            optimized_impl_config: optimized backend and projection-fusion policies.
        """
        super().__init__(attention_type, attention_config, optimized_impl_config)
        self._initialize_modules()
        self._initialize_derived_weights()


def _rope_freqs(
    length: int,
    attention_config: AttentionConfig,
    generator: torch.Generator,
    device: torch.device,
) -> Tensor | None:
    """Generate rotary frequencies when the attention policy enables RoPE.

    Args:
        length: Token sequence length.
        attention_config: Shared attention policy.
        generator: Seeded CUDA random generator.
        device: CUDA device on which to allocate the frequencies.

    Returns:
        Rotation angles shaped ``[L, 1, 1, D]``, or ``None`` when disabled.
    """
    if attention_config.rope_config is None:
        return None
    half_freqs = torch.randn(
        length,
        1,
        1,
        attention_config.head_dim // 2,
        generator=generator,
        device=device,
        dtype=torch.float32,
    )
    if attention_config.rope_config.style is RoPEStyle.INTERLEAVED:
        return half_freqs.repeat_interleave(2, dim=-1)
    return torch.cat((half_freqs, half_freqs), dim=-1)


def _assert_close(
    actual: Tensor,
    expected: Tensor,
    tolerance: float = 2e-2,
) -> None:
    """Compare optimized output with the Torch reference."""
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)


def _assert_cache_close(
    actual: BlockKVCache,
    expected: BlockKVCache,
    tolerance: float = 2e-2,
) -> None:
    """Compare visible optimized and Torch cache contents."""
    _assert_close(
        actual.cached_k().to(expected.cached_k().dtype),
        expected.cached_k(),
        tolerance,
    )
    _assert_close(
        actual.cached_v().to(expected.cached_v().dtype),
        expected.cached_v(),
        tolerance,
    )


def _check_self_attention(
    reference: _TorchMHA,
    actual: _OptimizedMHA,
    attention_config: AttentionConfig,
    generator: torch.Generator,
    device: torch.device,
    tolerance: float = 2e-2,
) -> None:
    """Compare streaming self-attention through cache fill and rolling.

    Args:
        reference: Torch attention with weights shared by ``actual``.
        actual: Optimized attention under test.
        attention_config: Shared attention geometry and policies.
        generator: Seeded CUDA random generator.
        device: CUDA device on which to run the comparison.
        tolerance: Absolute and relative comparison tolerance.
    """
    reference_cache = reference.allocate_kv_cache(
        batch_size=1,
        chunk_size=_CHUNK_SIZE,
        window_size=_WINDOW_SIZE,
        sink_size=_SINK_SIZE,
        device=device,
        dtype=torch.bfloat16,
    )
    actual_cache = actual.allocate_kv_cache(
        batch_size=1,
        chunk_size=_CHUNK_SIZE,
        window_size=_WINDOW_SIZE,
        sink_size=_SINK_SIZE,
        device=device,
        dtype=torch.bfloat16,
    )
    expected_cache_dtype = (
        torch.float8_e4m3fn
        if actual.optimized_impl_config.quantization.quantized_sdpa
        else torch.bfloat16
    )
    assert actual_cache._k.dtype is expected_cache_dtype
    assert actual_cache._v.dtype is expected_cache_dtype
    for chunk_idx in range(3):
        x = torch.randn(
            1,
            _CHUNK_SIZE,
            attention_config.query_dim,
            generator=generator,
            device=device,
            dtype=torch.bfloat16,
        )
        rope_length = (
            _SINK_SIZE + _WINDOW_SIZE
            if attention_config.rope_config is not None
            and attention_config.rope_config.scope is RoPEScope.AFTER_KV_CACHE
            else _CHUNK_SIZE
        )
        rope_freqs = _rope_freqs(rope_length, attention_config, generator, device)

        reference_cache.before_update(chunk_idx)
        expected = reference(x, reference_cache, rope_freqs)

        actual_cache.before_update(chunk_idx)
        output = actual(x, actual_cache, rope_freqs)

        _assert_close(output, expected, tolerance)
        _assert_cache_close(actual_cache, reference_cache, tolerance)
        reference_cache.after_update(chunk_idx)
        actual_cache.after_update(chunk_idx)


def _check_cross_attention(
    reference: _TorchMHA,
    actual: _OptimizedMHA,
    attention_config: AttentionConfig,
    generator: torch.Generator,
    device: torch.device,
    tolerance: float = 2e-2,
) -> None:
    """Compare static cross-attention with the Torch reference.

    Args:
        reference: Torch attention with weights shared by ``actual``.
        actual: Optimized attention under test.
        attention_config: Shared attention geometry and policies.
        generator: Seeded CUDA random generator.
        device: CUDA device on which to run the comparison.
        tolerance: Absolute and relative comparison tolerance.
    """
    assert attention_config.context_dim is not None
    context = torch.randn(
        1,
        24,
        attention_config.context_dim,
        generator=generator,
        device=device,
        dtype=torch.bfloat16,
    )
    query_length = 8
    query = torch.randn(
        1,
        query_length,
        attention_config.query_dim,
        generator=generator,
        device=device,
        dtype=torch.bfloat16,
    )
    assert query.shape[-2] != context.shape[-2]
    context_rope = _rope_freqs(24, attention_config, generator, device)
    query_rope_length = (
        max(query_length, context.shape[-2])
        if attention_config.rope_config is not None
        and attention_config.rope_config.scope is RoPEScope.AFTER_KV_CACHE
        else query_length
    )
    query_rope = _rope_freqs(query_rope_length, attention_config, generator, device)
    reference_cache = reference.compute_kv(context, context_rope)
    actual_cache = actual.compute_kv(context, context_rope)
    expected_cache_dtype = (
        torch.float8_e4m3fn
        if actual.optimized_impl_config.quantization.quantized_sdpa
        else torch.bfloat16
    )
    assert actual_cache._k.dtype is expected_cache_dtype
    assert actual_cache._v.dtype is expected_cache_dtype

    expected = reference(query, reference_cache, query_rope)
    output = actual(query, actual_cache, query_rope)

    _assert_close(output, expected, tolerance)
    _assert_cache_close(actual_cache, reference_cache, tolerance)


@pytest.mark.parametrize(
    "projection_biases",
    [
        pytest.param((True, False, False), id="query-only"),
        pytest.param((False, True, False), id="key-only"),
        pytest.param((False, False, True), id="value-only"),
        pytest.param((True, True, False), id="query-key"),
        pytest.param((True, False, True), id="query-value"),
        pytest.param((False, True, True), id="key-value"),
    ],
)
@pytest.mark.parametrize(
    "qkv_fusion_option", tuple(QKVFusionOption), ids=lambda value: value.value
)
def test_refresh_derived_weights_rejects_mixed_projection_biases(
    projection_biases: tuple[bool, bool, bool],
    qkv_fusion_option: QKVFusionOption,
) -> None:
    """Reject mixed Q/K/V bias policies for every projection-fusion mode."""
    attention = _OptimizedMHA(
        AttentionType.SELF_ATTENTION,
        AttentionConfig(
            query_dim=_QUERY_DIM,
            n_heads=_N_HEADS,
            head_dim=_HEAD_DIM,
        ),
        OptimizedImplConfig(qkv_fusion_option=qkv_fusion_option),
    )
    projections = (
        attention.query_projection,
        attention.key_projection,
        attention.value_projection,
    )
    for projection, has_bias in zip(projections, projection_biases, strict=True):
        if not has_bias:
            projection.register_parameter("bias", None)

    with pytest.raises(
        ValueError,
        match="must either all have biases or all omit biases",
    ):
        attention._refresh_derived_weights()


@pytest.mark.parametrize(
    "attention_type", tuple(AttentionType), ids=lambda value: value.value
)
@pytest.mark.parametrize("attention_config", _ATTENTION_CONFIGS)
@pytest.mark.parametrize("optimized_impl_config", _OPTIMIZED_IMPL_CONFIGS)
@torch.inference_mode()
def test_mha_optimized_matches_torch(
    cuda_device: torch.device,
    attention_type: AttentionType,
    attention_config: AttentionConfig,
    optimized_impl_config: OptimizedImplConfig,
) -> None:
    """Match Optimized attention with Torch for every supported policy."""
    torch.manual_seed(7)
    reference = _TorchMHA(attention_type, attention_config)
    actual = _OptimizedMHA(attention_type, attention_config, optimized_impl_config)
    actual.load_state_dict(reference.state_dict(), strict=True)
    reference.to(device=cuda_device, dtype=torch.bfloat16).eval()
    actual.to(device=cuda_device, dtype=torch.bfloat16).eval()

    generator = torch.Generator(device=cuda_device).manual_seed(11)
    if attention_type is AttentionType.SELF_ATTENTION:
        _check_self_attention(
            reference, actual, attention_config, generator, cuda_device
        )
    else:
        _check_cross_attention(
            reference, actual, attention_config, generator, cuda_device
        )


@pytest.mark.parametrize(
    "attention_type", tuple(AttentionType), ids=lambda value: value.value
)
@pytest.mark.parametrize(
    "qkv_fusion_option", tuple(QKVFusionOption), ids=lambda value: value.value
)
@pytest.mark.parametrize(
    "projection_dtype",
    tuple(DTYPE_MAX),
    ids=lambda dtype: str(dtype).removeprefix("torch."),
)
@torch.inference_mode()
def test_mha_optimized_quantized_projections_match_torch(
    cuda_device: torch.device,
    attention_type: AttentionType,
    qkv_fusion_option: QKVFusionOption,
    projection_dtype: torch.dtype,
) -> None:
    """Match quantized Q/K/V projections against native-precision attention."""
    attention_config = AttentionConfig(
        query_dim=_QUERY_DIM,
        n_heads=_N_HEADS,
        head_dim=_HEAD_DIM,
        qk_norm_scope=QKNormScope.HEAD,
    )
    optimized_impl_config = OptimizedImplConfig(
        qkv_fusion_option=qkv_fusion_option,
        quantization=QuantizationOption(projection=projection_dtype),
        sdpa_backend=SDPABackend.CUDNN,
        use_tma=False,
    )
    torch.manual_seed(17)
    reference = _TorchMHA(attention_type, attention_config)
    actual = _OptimizedMHA(attention_type, attention_config, optimized_impl_config)
    actual.load_state_dict(reference.state_dict(), strict=True)
    reference.to(device=cuda_device, dtype=torch.bfloat16).eval()
    actual.to(device=cuda_device, dtype=torch.bfloat16).eval()

    assert set(actual.state_dict()) == set(reference.state_dict())
    assert isinstance(actual.quantized_query_projection, QuantizedNonPersistentLinear)
    assert actual.quantized_query_projection.dtype is projection_dtype
    if qkv_fusion_option is QKVFusionOption.NONE:
        assert isinstance(actual.quantized_key_projection, QuantizedNonPersistentLinear)
        assert isinstance(
            actual.quantized_value_projection, QuantizedNonPersistentLinear
        )
        assert actual.fused_qkv is None
        assert actual.fused_kv is None
    else:
        assert actual.quantized_key_projection is None
        assert actual.quantized_value_projection is None
        assert isinstance(actual.fused_kv, QuantizedNonPersistentLinear)
        assert actual.fused_kv.dtype is projection_dtype
        if qkv_fusion_option is QKVFusionOption.FULL:
            assert isinstance(actual.fused_qkv, QuantizedNonPersistentLinear)
            assert actual.fused_qkv.dtype is projection_dtype
        else:
            assert actual.fused_qkv is None

    quantization_tolerance = (
        torch.finfo(projection_dtype).eps
        if projection_dtype.is_floating_point
        else 5 / DTYPE_MAX[projection_dtype]
    )
    tolerance = max(2e-2, quantization_tolerance)
    generator = torch.Generator(device=cuda_device).manual_seed(19)
    if attention_type is AttentionType.SELF_ATTENTION:
        _check_self_attention(
            reference,
            actual,
            attention_config,
            generator,
            cuda_device,
            tolerance,
        )
    else:
        _check_cross_attention(
            reference,
            actual,
            attention_config,
            generator,
            cuda_device,
            tolerance,
        )


@pytest.mark.parametrize(
    "attention_type", tuple(AttentionType), ids=lambda value: value.value
)
@pytest.mark.parametrize("rope_scope", tuple(RoPEScope), ids=lambda value: value.value)
@pytest.mark.parametrize(
    "sdpa_backend", tuple(SDPABackend), ids=lambda value: value.value
)
@pytest.mark.parametrize("use_tma", (False, True), ids=("no-tma", "tma"))
@torch.inference_mode()
def test_mha_optimized_quantized_sdpa_matches_torch(
    cuda_device: torch.device,
    attention_type: AttentionType,
    rope_scope: RoPEScope,
    sdpa_backend: SDPABackend,
    use_tma: bool,
) -> None:
    """Exercise the unscaled e4m3 SDPA/cache contract across backends."""
    attention_config = AttentionConfig(
        query_dim=_QUERY_DIM,
        n_heads=_N_HEADS,
        head_dim=_HEAD_DIM,
        qk_norm_scope=QKNormScope.HEAD,
        rope_config=RoPEConfig(style=RoPEStyle.SPLIT, scope=rope_scope),
    )
    optimized_impl_config = OptimizedImplConfig(
        qkv_fusion_option=QKVFusionOption.FULL,
        quantization=QuantizationOption(quantized_sdpa=True),
        sdpa_backend=sdpa_backend,
        use_tma=use_tma,
    )
    torch.manual_seed(23)
    reference = _TorchMHA(attention_type, attention_config)
    actual = _OptimizedMHA(attention_type, attention_config, optimized_impl_config)
    actual.load_state_dict(reference.state_dict(), strict=True)
    reference.to(device=cuda_device, dtype=torch.bfloat16).eval()
    actual.to(device=cuda_device, dtype=torch.bfloat16).eval()

    generator = torch.Generator(device=cuda_device).manual_seed(29)
    tolerance = 2 * torch.finfo(torch.float8_e4m3fn).eps
    if attention_type is AttentionType.SELF_ATTENTION:
        _check_self_attention(
            reference,
            actual,
            attention_config,
            generator,
            cuda_device,
            tolerance,
        )
    else:
        _check_cross_attention(
            reference,
            actual,
            attention_config,
            generator,
            cuda_device,
            tolerance,
        )


@torch.inference_mode()
def test_native_cudnn_fp8_sdpa_returns_independent_outputs(
    cuda_device: torch.device,
) -> None:
    """Keep retained FP8 outputs unchanged across cached graph executions."""
    shape = (1, 1, 16, 16)
    query = torch.zeros(shape, device=cuda_device, dtype=torch.float8_e4m3fn)
    key = torch.zeros_like(query)
    first = native_cudnn_fp8_sdpa(query, key, torch.ones_like(query))
    retained = first.float().clone()

    second = native_cudnn_fp8_sdpa(query, key, torch.zeros_like(query))

    assert first.data_ptr() != second.data_ptr()
    torch.testing.assert_close(first.float(), retained)


@torch.inference_mode()
def test_native_cudnn_fp8_sdpa_replays_inside_cuda_graph(
    cuda_device: torch.device,
) -> None:
    """Replay cuDNN FP8 attention with ordered input and output operations."""
    shape = (1, 1, 16, 16)
    query = torch.zeros(shape, device=cuda_device, dtype=torch.float8_e4m3fn)
    key = torch.zeros_like(query)
    value = torch.empty_like(query)
    source = torch.ones_like(query)
    native_cudnn_fp8_sdpa(query, key, source)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, capture_error_mode="thread_local"):
        value.copy_(source)
        output = native_cudnn_fp8_sdpa(query, key, value).float()

    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(output, torch.ones_like(output))

    source.zero_()
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(output, torch.zeros_like(output))
