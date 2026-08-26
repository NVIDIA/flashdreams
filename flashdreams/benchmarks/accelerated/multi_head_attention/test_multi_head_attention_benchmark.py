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

"""Matched microbenchmarks for Torch and optimized attention multi-head attention.

Both implementations use identical geometry and random weights.

Run the manual GPU benchmarks with::

    uv run --package flashdreams --group test pytest \
        flashdreams/benchmarks/accelerated/multi_head_attention/test_multi_head_attention_benchmark.py \
        -p no:manual_marker -m manual --benchmark-only -v
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import pytest
import torch

if TYPE_CHECKING:
    from pytest_benchmark.fixture import BenchmarkFixture
from torch import Tensor

from flashdreams.accelerated.multi_head_attention import (
    AttentionConfig,
    AttentionType,
    QKNormScope,
    RoPEConfig,
    RoPEScope,
    RoPEStyle,
)
from flashdreams.accelerated.multi_head_attention.optimized import (
    OptimizedImplConfig,
    OptimizedMultiHeadAttention,
    QKVFusionOption,
    QuantizationOption,
    SDPABackend,
    is_tma_flash_attention_supported,
)
from flashdreams.accelerated.multi_head_attention.torch import TorchMultiHeadAttention

pytestmark = [
    pytest.mark.manual,
    pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="Accelerated multi-head attention benchmarks require CUDA.",
    ),
]

_WARMUP_ROUNDS = 5
"""Warmup calls used to absorb kernel compilation and autotuning."""

_BENCHMARK_ROUNDS = 50
"""Measured calls used for each implementation comparison."""


_IMPLEMENTATION_CONFIGS: tuple[OptimizedImplConfig | None, ...] = (
    None,
    *(
        OptimizedImplConfig(
            sdpa_backend=SDPABackend.CUDNN,
            qkv_fusion_option=qkv_fusion_option,
            use_tma=False,
            quantization=QuantizationOption(
                projection=projection_quantization,
                quantized_sdpa=quantized_sdpa,
            ),
        )
        for qkv_fusion_option in QKVFusionOption
        for projection_quantization in (None, torch.float8_e4m3fn)
        for quantized_sdpa in (False, True)
    ),
    *(
        OptimizedImplConfig(
            sdpa_backend=SDPABackend.FA2,
            qkv_fusion_option=qkv_fusion_option,
            use_tma=use_tma,
            quantization=QuantizationOption(
                projection=projection_quantization,
                quantized_sdpa=quantized_sdpa,
            ),
        )
        for qkv_fusion_option in QKVFusionOption
        for use_tma in (False, True)
        for projection_quantization in (None, torch.float8_e4m3fn)
        for quantized_sdpa in (False, True)
    ),
)
"""Torch reference, optimized policies, FP8 projections, and quantized SDPA rows."""


def _implementation_id(config: OptimizedImplConfig | None) -> str:
    """Return a stable pytest identifier for an implementation config."""
    if config is None:
        return "reference-torch"
    backend = config.sdpa_backend.value
    fusion = config.qkv_fusion_option.value.replace("_", "-")
    tma = "tma" if config.use_tma else "no-tma"
    projection = (
        ""
        if config.quantization.projection is None
        else (
            f"-projection-{str(config.quantization.projection).removeprefix('torch.')}"
        )
    )
    quantized_sdpa = "-quantized-sdpa" if config.quantization.quantized_sdpa else ""
    return f"optimized-{backend}-{fusion}-{tma}{projection}{quantized_sdpa}"


_SHARED_CONFIGS = tuple(
    pytest.param(
        qk_norm_scope,
        rope_scope,
        rope_interleaved,
        bias,
        id=(
            f"norm-{qk_norm_scope.value}-"
            f"rope-{'interleaved' if rope_interleaved else 'split'}-"
            f"scope-{rope_scope.value.replace('_', '-')}-"
            f"bias-{'on' if bias else 'off'}"
        ),
    )
    for qk_norm_scope in QKNormScope
    for rope_scope in RoPEScope
    for rope_interleaved in (False, True)
    for bias in (False, True)
)
"""Policies shared by Torch and optimized attention, each mapped to its own benchmark group."""

_REPRESENTATIVE_CONFIGS = tuple(
    config
    for config in _SHARED_CONFIGS
    if config.values
    in (
        (
            QKNormScope.HEAD,
            RoPEScope.BEFORE_KV_CACHE,
            False,
            False,
        ),
        (
            QKNormScope.INNER,
            RoPEScope.BEFORE_KV_CACHE,
            True,
            True,
        ),
    )
)
"""Cosmos- and Wan-style policies used by the default benchmark sweep."""

_FULL_POLICY_SEARCH = os.environ.get("FLASHDREAMS_RUN_FULL_BENCHMARK") == "1"
"""Whether to benchmark every shared attention policy."""

_BENCHMARK_CONFIGS = _SHARED_CONFIGS if _FULL_POLICY_SEARCH else _REPRESENTATIVE_CONFIGS
"""Selected policies; set ``FLASHDREAMS_RUN_FULL_BENCHMARK=1`` for all."""

_CROSS_ATTENTION_CONFIGS = tuple(
    config
    for config in _BENCHMARK_CONFIGS
    if config.values[1] is RoPEScope.BEFORE_KV_CACHE
)
"""Cross-attention policies representable with distinct query/context lengths."""


class _TorchMultiHeadAttention(TorchMultiHeadAttention):
    """PyTorch reference attention implementation used by benchmarks."""

    @property
    def query_projection(self) -> torch.nn.Linear:
        """Return the reference query projection."""
        return self.q_proj

    @property
    def key_projection(self) -> torch.nn.Linear:
        """Return the reference key projection."""
        return self.k_proj

    @property
    def value_projection(self) -> torch.nn.Linear:
        """Return the reference value projection."""
        return self.v_proj

    @property
    def output_projection(self) -> torch.nn.Linear:
        """Return the reference output projection."""
        return self.output_proj

    @property
    def query_norm(self) -> torch.nn.Module:
        """Return the reference query normalization."""
        return self.q_norm

    @property
    def key_norm(self) -> torch.nn.Module:
        """Return the reference key normalization."""
        return self.k_norm

    def __init__(
        self,
        query_dim: int,
        n_heads: int = 8,
        head_dim: int = 64,
        *,
        context_dim: int | None = None,
        attention_type: AttentionType = AttentionType.SELF_ATTENTION,
        qkv_bias: bool = False,
        output_bias: bool = False,
        qk_norm_scope: QKNormScope = QKNormScope.HEAD,
        rope_scope: RoPEScope = RoPEScope.BEFORE_KV_CACHE,
        rope_interleaved: bool = False,
    ) -> None:
        """Initialize reference projections and normalization modules."""
        super().__init__(
            attention_type=attention_type,
            attention_config=AttentionConfig(
                query_dim=query_dim,
                context_dim=context_dim,
                n_heads=n_heads,
                head_dim=head_dim,
                qk_norm_scope=qk_norm_scope,
                rope_config=RoPEConfig(
                    scope=rope_scope,
                    style=(
                        RoPEStyle.INTERLEAVED if rope_interleaved else RoPEStyle.SPLIT
                    ),
                ),
            ),
        )
        assert self.attention_config.context_dim is not None
        self.q_proj = torch.nn.Linear(
            self.attention_config.query_dim,
            self.attention_config.inner_dim,
            bias=qkv_bias,
        )
        self.k_proj = torch.nn.Linear(
            self.attention_config.context_dim,
            self.attention_config.inner_dim,
            bias=qkv_bias,
        )
        self.v_proj = torch.nn.Linear(
            self.attention_config.context_dim,
            self.attention_config.inner_dim,
            bias=qkv_bias,
        )
        self.output_proj = torch.nn.Linear(
            self.attention_config.inner_dim,
            self.attention_config.query_dim,
            bias=output_bias,
        )
        if self.attention_config.qk_norm_scope is QKNormScope.NONE:
            self.q_norm = torch.nn.Identity()
            self.k_norm = torch.nn.Identity()
        else:
            norm_dim = (
                self.attention_config.head_dim
                if self.attention_config.qk_norm_scope is QKNormScope.HEAD
                else self.attention_config.inner_dim
            )
            self.q_norm = torch.nn.RMSNorm(
                norm_dim, eps=self.attention_config.qk_norm_eps
            )
            self.k_norm = torch.nn.RMSNorm(
                norm_dim, eps=self.attention_config.qk_norm_eps
            )


class _OptimizedMultiHeadAttention(OptimizedMultiHeadAttention):
    """Optimized attention implementation used by benchmarks."""

    effective_use_tma: bool | None
    """Whether the first attention call selected TMA; ``None`` before execution."""

    @property
    def query_projection(self) -> torch.nn.Linear:
        """Return the optimized query projection."""
        return self.q_proj

    @property
    def key_projection(self) -> torch.nn.Linear:
        """Return the optimized key projection."""
        return self.k_proj

    @property
    def value_projection(self) -> torch.nn.Linear:
        """Return the optimized value projection."""
        return self.v_proj

    @property
    def output_projection(self) -> torch.nn.Linear:
        """Return the optimized output projection."""
        return self.output_proj

    @property
    def query_norm(self) -> torch.nn.Module:
        """Return the optimized query normalization."""
        return self.q_norm

    @property
    def key_norm(self) -> torch.nn.Module:
        """Return the optimized key normalization."""
        return self.k_norm

    def __init__(
        self,
        query_dim: int,
        n_heads: int = 8,
        head_dim: int = 64,
        *,
        context_dim: int | None = None,
        attention_type: AttentionType = AttentionType.SELF_ATTENTION,
        optimized_impl_config: OptimizedImplConfig,
        qkv_bias: bool = False,
        output_bias: bool = False,
        qk_norm_scope: QKNormScope = QKNormScope.HEAD,
        rope_scope: RoPEScope = RoPEScope.BEFORE_KV_CACHE,
        rope_interleaved: bool = False,
    ) -> None:
        """Initialize optimized projections and normalization modules."""
        super().__init__(
            attention_type=attention_type,
            attention_config=AttentionConfig(
                query_dim=query_dim,
                context_dim=context_dim,
                n_heads=n_heads,
                head_dim=head_dim,
                qk_norm_scope=qk_norm_scope,
                rope_config=RoPEConfig(
                    scope=rope_scope,
                    style=(
                        RoPEStyle.INTERLEAVED if rope_interleaved else RoPEStyle.SPLIT
                    ),
                ),
            ),
            optimized_impl_config=optimized_impl_config,
        )
        assert self.attention_config.context_dim is not None
        self.q_proj = torch.nn.Linear(
            self.attention_config.query_dim,
            self.attention_config.inner_dim,
            bias=qkv_bias,
        )
        self.k_proj = torch.nn.Linear(
            self.attention_config.context_dim,
            self.attention_config.inner_dim,
            bias=qkv_bias,
        )
        self.v_proj = torch.nn.Linear(
            self.attention_config.context_dim,
            self.attention_config.inner_dim,
            bias=qkv_bias,
        )
        self.output_proj = torch.nn.Linear(
            self.attention_config.inner_dim,
            self.attention_config.query_dim,
            bias=output_bias,
        )
        if self.attention_config.qk_norm_scope is QKNormScope.NONE:
            self.q_norm = torch.nn.Identity()
            self.k_norm = torch.nn.Identity()
        else:
            norm_dim = (
                self.attention_config.head_dim
                if self.attention_config.qk_norm_scope is QKNormScope.HEAD
                else self.attention_config.inner_dim
            )
            self.q_norm = torch.nn.RMSNorm(
                norm_dim, eps=self.attention_config.qk_norm_eps
            )
            self.k_norm = torch.nn.RMSNorm(
                norm_dim, eps=self.attention_config.qk_norm_eps
            )
        self.effective_use_tma = None
        self._initialize_derived_weights()

    def _attention(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        *,
        output_dtype: torch.dtype | None = None,
    ) -> Tensor:
        """Record the effective backend selected by the first attention call."""
        if self.effective_use_tma is None:
            self.effective_use_tma = (
                self.sdpa_backend is SDPABackend.FA2
                and self.use_tma
                and is_tma_flash_attention_supported(query, key, value)
            )
        return super()._attention(query, key, value, output_dtype=output_dtype)


_Attention = _TorchMultiHeadAttention | _OptimizedMultiHeadAttention

_BATCH_SIZE = 1
_DTYPE = torch.bfloat16
_SEED = 42
_SINK_SIZE = 0


_QUERY_DIM = 2048
"""Input and output feature width shared by both implementations."""

_N_HEADS = 16
"""Number of attention heads shared by both implementations."""

_HEAD_DIM = _QUERY_DIM // _N_HEADS

_CHUNK_SIZE = 80 * 60
"""Number of query tokens processed by each benchmark call."""

_WINDOW_CHUNKS = 6
"""Number of chunks retained in the full rolling cache."""

_WINDOW_SIZE = _WINDOW_CHUNKS * _CHUNK_SIZE


def _make_attention(
    optimized_impl_config: OptimizedImplConfig | None,
    *,
    attention_type: AttentionType,
    qk_norm_scope: QKNormScope,
    rope_scope: RoPEScope,
    rope_interleaved: bool,
    bias: bool,
) -> _Attention:
    """Build one weight-matched Torch reference or Optimized implementation.

    Args:
        optimized_impl_config: Optimized policy; ``None`` uses the Torch reference.
        attention_type: Whether to configure streaming self-attention or static
            cross-attention.
        qk_norm_scope: Shared Q/K normalization policy.
        rope_scope: Whether keys are rotated before or after cache storage.
        rope_interleaved: Shared rotary-pair layout.
        bias: Whether every Q/K/V and output projection uses a bias.

    Returns:
        Configured attention module with deterministic random weights.
    """
    reference = _TorchMultiHeadAttention(
        query_dim=_QUERY_DIM,
        context_dim=_QUERY_DIM,
        n_heads=_N_HEADS,
        head_dim=_HEAD_DIM,
        attention_type=attention_type,
        qkv_bias=bias,
        output_bias=bias,
        qk_norm_scope=qk_norm_scope,
        rope_scope=rope_scope,
        rope_interleaved=rope_interleaved,
    )
    if optimized_impl_config is None:
        return reference

    optimized_attention = _OptimizedMultiHeadAttention(
        query_dim=_QUERY_DIM,
        context_dim=_QUERY_DIM,
        n_heads=_N_HEADS,
        head_dim=_HEAD_DIM,
        attention_type=attention_type,
        optimized_impl_config=optimized_impl_config,
        qkv_bias=bias,
        output_bias=bias,
        qk_norm_scope=qk_norm_scope,
        rope_scope=rope_scope,
        rope_interleaved=rope_interleaved,
    )
    optimized_attention.load_state_dict(reference.state_dict(), strict=True)
    return optimized_attention


@torch.inference_mode()
def _benchmark_multi_head_attention(
    benchmark: BenchmarkFixture,
    optimized_impl_config: OptimizedImplConfig | None,
    *,
    attention_type: AttentionType,
    qk_norm_scope: QKNormScope,
    rope_scope: RoPEScope,
    rope_interleaved: bool,
    bias: bool,
    cross_attention_query_only: bool = False,
) -> None:
    """Run one synchronized attention benchmark within a shared-policy group.

    Streaming self-attention times forward over a prefilled rolling cache.
    Cross-attention either times static K/V preparation and forward together or
    reuses K/V prepared before timing to isolate repeated query latency.

    Args:
        benchmark: Pytest benchmark fixture used to record synchronized timings.
        optimized_impl_config: Optimized policy; ``None`` uses the Torch reference.
        attention_type: Self- or cross-attention benchmark family.
        qk_norm_scope: Shared Q/K normalization policy.
        rope_scope: Whether keys are rotated before or after cache storage.
        rope_interleaved: Shared rotary-pair layout.
        bias: Whether every Q/K/V and output projection uses a bias.
        cross_attention_query_only: Whether to prepare cross-attention K/V before
            timing.
    """
    if not torch.cuda.is_bf16_supported():
        pytest.skip("Multi-head attention benchmark requires bfloat16 support.")

    device = torch.device("cuda")
    is_optimized = optimized_impl_config is not None
    if is_optimized and torch.cuda.get_device_capability(device)[0] < 9:
        pytest.skip("Optimized attention requires compute capability 9.0 or newer.")

    torch.manual_seed(_SEED)
    attention = _make_attention(
        optimized_impl_config,
        attention_type=attention_type,
        qk_norm_scope=qk_norm_scope,
        rope_scope=rope_scope,
        rope_interleaved=rope_interleaved,
        bias=bias,
    )
    attention.to(device=device, dtype=_DTYPE).eval()

    generator = torch.Generator(device=device).manual_seed(_SEED)
    inputs = [
        torch.randn(
            _BATCH_SIZE,
            _CHUNK_SIZE,
            _QUERY_DIM,
            generator=generator,
            device=device,
            dtype=_DTYPE,
        )
        for _ in range(_WINDOW_CHUNKS + 1)
    ]
    rope_freq_count = (
        1 if rope_scope is RoPEScope.AFTER_KV_CACHE else _WINDOW_CHUNKS + 1
    )
    rope_freq_length = (
        _WINDOW_SIZE if rope_scope is RoPEScope.AFTER_KV_CACHE else _CHUNK_SIZE
    )
    half_rope_freqs = [
        torch.randn(
            rope_freq_length,
            1,
            1,
            _HEAD_DIM // 2,
            generator=generator,
            device=device,
            dtype=torch.float32,
        )
        for _ in range(rope_freq_count)
    ]
    rope_freqs = [
        (
            freqs.repeat_interleave(2, dim=-1)
            if rope_interleaved
            else torch.cat((freqs, freqs), dim=-1)
        )
        for freqs in half_rope_freqs
    ]
    if rope_scope is RoPEScope.AFTER_KV_CACHE:
        rope_freqs *= _WINDOW_CHUNKS + 1

    attention_label = (
        "self" if attention_type is AttentionType.SELF_ATTENTION else "cross"
    )
    if cross_attention_query_only:
        attention_label += "-query-only"
    projection_dtype = (
        _DTYPE
        if optimized_impl_config is None
        or optimized_impl_config.quantization.projection is None
        else optimized_impl_config.quantization.projection
    )
    quantized_sdpa = (
        optimized_impl_config is not None
        and optimized_impl_config.quantization.quantized_sdpa
    )
    sdpa_backend = (
        "torch-sdpa"
        if optimized_impl_config is None
        else optimized_impl_config.sdpa_backend.value
    )
    qkv_fusion_option = (
        QKVFusionOption.NONE.value
        if optimized_impl_config is None
        else optimized_impl_config.qkv_fusion_option.value
    )
    use_tma = optimized_impl_config is not None and optimized_impl_config.use_tma
    timing_scope = (
        "steady-state"
        if attention_type is AttentionType.SELF_ATTENTION
        else "query-only"
        if cross_attention_query_only
        else "end-to-end"
    )
    rope_label = "interleaved" if rope_interleaved else "split"
    bias_label = "on" if bias else "off"
    benchmark.group = "-".join(
        (
            "multi-head-attention",
            attention_label,
            "norm",
            qk_norm_scope.value,
            "rope",
            rope_label,
            "scope",
            rope_scope.value.replace("_", "-"),
            "bias",
            bias_label,
        )
    )
    benchmark.extra_info.update(
        {
            "implementation": _implementation_id(optimized_impl_config),
            "attention_type": attention_type.value,
            "timing_scope": timing_scope,
            "policy_search": "full" if _FULL_POLICY_SEARCH else "representative",
            "input_dtype": str(_DTYPE),
            "projection_dtype": str(projection_dtype),
            "sdpa_dtype": str(torch.float8_e4m3fn if quantized_sdpa else _DTYPE),
            "query_dim": _QUERY_DIM,
            "context_dim": _QUERY_DIM,
            "n_heads": _N_HEADS,
            "head_dim": _HEAD_DIM,
            "qk_norm_scope": qk_norm_scope.value,
            "rope_scope": rope_scope.value,
            "rope_interleaved": rope_interleaved,
            "bias": bias,
            "quantized_sdpa": quantized_sdpa,
            "sdpa_backend": sdpa_backend,
            "qkv_fusion_option": qkv_fusion_option,
            "use_tma": use_tma,
            "seed": _SEED,
            "batch_size": _BATCH_SIZE,
            "chunk_size": _CHUNK_SIZE,
            "window_chunks": _WINDOW_CHUNKS,
            "window_size": _WINDOW_SIZE,
            "sink_size": _SINK_SIZE,
            "warmup_rounds": _WARMUP_ROUNDS,
            "benchmark_rounds": _BENCHMARK_ROUNDS,
        }
    )

    query = inputs[_WINDOW_CHUNKS]
    query_rope = rope_freqs[_WINDOW_CHUNKS]
    if attention_type is AttentionType.SELF_ATTENTION:
        cache = attention.allocate_kv_cache(
            batch_size=_BATCH_SIZE,
            chunk_size=_CHUNK_SIZE,
            window_size=_WINDOW_SIZE,
            sink_size=_SINK_SIZE,
            device=device,
            dtype=_DTYPE,
        )

        # Prefill and roll outside the timer. Every measured forward sees the
        # same full context and overwrites the same final cache slot.
        for chunk_idx in range(_WINDOW_CHUNKS):
            cache.before_update(chunk_idx)
            attention(inputs[chunk_idx], cache, rope_freqs[chunk_idx])
            cache.after_update(chunk_idx)
        cache.before_update(_WINDOW_CHUNKS)
        torch.cuda.synchronize()

        def synchronized_self_forward() -> Tensor:
            result = attention(query, cache, query_rope)
            torch.cuda.synchronize()
            return result

        output = benchmark.pedantic(
            synchronized_self_forward,
            iterations=1,
            rounds=_BENCHMARK_ROUNDS,
            warmup_rounds=_WARMUP_ROUNDS,
        )
        cache.after_update(_WINDOW_CHUNKS)
    else:
        context = torch.cat(inputs[:_WINDOW_CHUNKS], dim=1)
        context_rope = torch.cat(rope_freqs[:_WINDOW_CHUNKS], dim=0)
        if cross_attention_query_only:
            # Reuse prompt K/V across every timed query, matching repeated
            # cross-attention calls against static context.
            cache = attention.compute_kv(context, context_rope)
            torch.cuda.synchronize()

            def synchronized_cross_forward() -> Tensor:
                result = attention(query, cache, query_rope)
                torch.cuda.synchronize()
                return result

        else:
            torch.cuda.synchronize()

            # Static K/V projection is part of this end-to-end cross-attention
            # measurement because fusion changes that stage.
            def synchronized_cross_forward() -> Tensor:
                cache = attention.compute_kv(context, context_rope)
                result = attention(query, cache, query_rope)
                torch.cuda.synchronize()
                return result

        output = benchmark.pedantic(
            synchronized_cross_forward,
            iterations=1,
            rounds=_BENCHMARK_ROUNDS,
            warmup_rounds=_WARMUP_ROUNDS,
        )

    if isinstance(attention, _OptimizedMultiHeadAttention):
        assert attention.effective_use_tma is not None
        effective_use_tma = attention.effective_use_tma
    else:
        effective_use_tma = False
    benchmark.extra_info.update(
        {
            "effective_sdpa_backend": (
                "fa2-tma" if effective_use_tma else sdpa_backend
            ),
            "effective_use_tma": effective_use_tma,
        }
    )
    assert output.shape == query.shape
    assert torch.isfinite(output).all()


@pytest.mark.parametrize(
    "optimized_impl_config",
    _IMPLEMENTATION_CONFIGS,
    ids=_implementation_id,
)
@pytest.mark.parametrize(
    "qk_norm_scope,rope_scope,rope_interleaved,bias",
    _BENCHMARK_CONFIGS,
)
def test_self_attention_benchmark(
    benchmark: BenchmarkFixture,
    optimized_impl_config: OptimizedImplConfig | None,
    qk_norm_scope: QKNormScope,
    rope_scope: RoPEScope,
    rope_interleaved: bool,
    bias: bool,
) -> None:
    """Compare streaming self-attention within one shared-policy group.

    Args:
        benchmark: Pytest benchmark fixture used to record synchronized timings.
        optimized_impl_config: Optimized policy; ``None`` uses the Torch reference.
        qk_norm_scope: Shared Q/K normalization policy defining the group.
        rope_scope: Shared cache-relative rotation policy defining the group.
        rope_interleaved: Shared rotary-pair layout defining the group.
        bias: Shared projection-bias policy defining the group.
    """
    _benchmark_multi_head_attention(
        benchmark,
        optimized_impl_config,
        attention_type=AttentionType.SELF_ATTENTION,
        qk_norm_scope=qk_norm_scope,
        rope_scope=rope_scope,
        rope_interleaved=rope_interleaved,
        bias=bias,
    )


@pytest.mark.parametrize(
    "optimized_impl_config",
    _IMPLEMENTATION_CONFIGS,
    ids=_implementation_id,
)
@pytest.mark.parametrize(
    "qk_norm_scope,rope_scope,rope_interleaved,bias",
    _CROSS_ATTENTION_CONFIGS,
)
def test_cross_attention_benchmark(
    benchmark: BenchmarkFixture,
    optimized_impl_config: OptimizedImplConfig | None,
    qk_norm_scope: QKNormScope,
    rope_scope: RoPEScope,
    rope_interleaved: bool,
    bias: bool,
) -> None:
    """Compare end-to-end cross-attention within one shared-policy group.

    Args:
        benchmark: Pytest benchmark fixture used to record synchronized timings.
        optimized_impl_config: Optimized policy; ``None`` uses the Torch reference.
        qk_norm_scope: Shared Q/K normalization policy defining the group.
        rope_scope: Shared cache-relative rotation policy defining the group.
        rope_interleaved: Shared rotary-pair layout defining the group.
        bias: Shared projection-bias policy defining the group.
    """
    _benchmark_multi_head_attention(
        benchmark,
        optimized_impl_config,
        attention_type=AttentionType.CROSS_ATTENTION,
        qk_norm_scope=qk_norm_scope,
        rope_scope=rope_scope,
        rope_interleaved=rope_interleaved,
        bias=bias,
    )


@pytest.mark.parametrize(
    "optimized_impl_config",
    _IMPLEMENTATION_CONFIGS,
    ids=_implementation_id,
)
@pytest.mark.parametrize(
    "qk_norm_scope,rope_scope,rope_interleaved,bias",
    _CROSS_ATTENTION_CONFIGS,
)
def test_cross_attention_query_only_benchmark(
    benchmark: BenchmarkFixture,
    optimized_impl_config: OptimizedImplConfig | None,
    qk_norm_scope: QKNormScope,
    rope_scope: RoPEScope,
    rope_interleaved: bool,
    bias: bool,
) -> None:
    """Compare cross-attention query latency with K/V prepared before timing.

    Args:
        benchmark: Pytest benchmark fixture used to record synchronized timings.
        optimized_impl_config: Optimized policy; ``None`` uses the Torch reference.
        qk_norm_scope: Shared Q/K normalization policy defining the group.
        rope_scope: Shared cache-relative rotation policy defining the group.
        rope_interleaved: Shared rotary-pair layout defining the group.
        bias: Shared projection-bias policy defining the group.
    """
    _benchmark_multi_head_attention(
        benchmark,
        optimized_impl_config,
        attention_type=AttentionType.CROSS_ATTENTION,
        qk_norm_scope=qk_norm_scope,
        rope_scope=rope_scope,
        rope_interleaved=rope_interleaved,
        bias=bias,
        cross_attention_query_only=True,
    )
