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

"""Matched microbenchmarks for Torch and Triton multi-head attention.

Both implementations use identical geometry and random weights.

Run the manual GPU benchmarks with::

    uv run --package flashdreams --group test pytest \
        flashdreams/benchmarks/accelerated/test_multi_head_attention_benchmark.py \
        -p no:manual_marker -m manual --benchmark-only -v
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import pytest
import torch
from pytest_benchmark.fixture import BenchmarkFixture
from torch import Tensor

from flashdreams.accelerated.multi_head_attention import AttentionType, QKNormScope
from flashdreams.accelerated.multi_head_attention_torch import TorchMultiHeadAttention
from flashdreams.accelerated.multi_head_attention_triton import (
    QKVFusionOption,
    SDPABackend,
    TritonMultiHeadAttention,
)

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


class _Implementation(str, Enum):
    """Multi-head attention implementations covered by the benchmark."""

    REFERENCE_TORCH = "reference_torch"
    TRITON = "triton"


@dataclass(frozen=True)
class _ImplementationCase:
    """One implementation-specific row within a shared benchmark group."""

    implementation: _Implementation
    """Attention implementation exercised by this row."""

    sdpa_backend: SDPABackend | None = None
    """Triton attention backend, or ``None`` for the Torch reference."""

    use_fp8: bool = False
    """Whether the Triton row enables FP8 projections and supported storage."""

    qkv_fusion_option: QKVFusionOption | None = None
    """Triton projection fusion policy, or ``None`` for the Torch reference."""

    @property
    def id(self) -> str:
        """Return a stable pytest identifier for this implementation row."""
        if self.implementation is _Implementation.REFERENCE_TORCH:
            return "reference-torch"
        assert self.sdpa_backend is not None
        assert self.qkv_fusion_option is not None
        backend = "fa2" if self.sdpa_backend is SDPABackend.TRITON else "cudnn"
        precision = "fp8" if self.use_fp8 else "bf16"
        fusion = self.qkv_fusion_option.value.replace("_", "-")
        return f"triton-{backend}-{precision}-{fusion}"


_IMPLEMENTATION_CASES = (
    _ImplementationCase(_Implementation.REFERENCE_TORCH),
    *(
        _ImplementationCase(
            implementation=_Implementation.TRITON,
            sdpa_backend=sdpa_backend,
            use_fp8=use_fp8,
            qkv_fusion_option=qkv_fusion_option,
        )
        for sdpa_backend in SDPABackend
        for use_fp8 in (False, True)
        for qkv_fusion_option in QKVFusionOption
    ),
)
"""Torch reference followed by every Triton backend, precision, and fusion row."""

_SHARED_CONFIGS = tuple(
    pytest.param(
        qk_norm_scope,
        rope_interleaved,
        bias,
        id=(
            f"norm-{qk_norm_scope.value}-"
            f"rope-{'interleaved' if rope_interleaved else 'split'}-"
            f"bias-{'on' if bias else 'off'}"
        ),
    )
    for qk_norm_scope in QKNormScope
    for rope_interleaved in (False, True)
    for bias in (False, True)
)
"""Policies shared by Torch and Triton, each mapped to its own benchmark group."""


class _TorchMultiHeadAttention(TorchMultiHeadAttention):
    """Canonical Torch attention implementation used by benchmarks."""

    @property
    def query_projection(self) -> torch.nn.Linear:
        """Return the canonical query projection."""
        return self.q_proj

    @property
    def key_projection(self) -> torch.nn.Linear:
        """Return the canonical key projection."""
        return self.k_proj

    @property
    def value_projection(self) -> torch.nn.Linear:
        """Return the canonical value projection."""
        return self.v_proj

    @property
    def output_projection(self) -> torch.nn.Linear:
        """Return the canonical output projection."""
        return self.output_proj

    @property
    def query_norm(self) -> torch.nn.Module:
        """Return the canonical query normalization."""
        return self.q_norm

    @property
    def key_norm(self) -> torch.nn.Module:
        """Return the canonical key normalization."""
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
        rope_interleaved: bool = False,
    ) -> None:
        """Initialize canonical projections and normalization modules."""
        super().__init__(
            query_dim=query_dim,
            n_heads=n_heads,
            head_dim=head_dim,
            context_dim=context_dim,
            attention_type=attention_type,
            qk_norm_scope=qk_norm_scope,
            rope_interleaved=rope_interleaved,
        )
        self.q_proj = torch.nn.Linear(self.query_dim, self.inner_dim, bias=qkv_bias)
        self.k_proj = torch.nn.Linear(self.context_dim, self.inner_dim, bias=qkv_bias)
        self.v_proj = torch.nn.Linear(self.context_dim, self.inner_dim, bias=qkv_bias)
        self.output_proj = torch.nn.Linear(
            self.inner_dim, self.query_dim, bias=output_bias
        )
        if self.qk_norm_scope is QKNormScope.NONE:
            self.q_norm = torch.nn.Identity()
            self.k_norm = torch.nn.Identity()
        else:
            norm_dim = (
                self.head_dim
                if self.qk_norm_scope is QKNormScope.HEAD
                else self.inner_dim
            )
            self.q_norm = torch.nn.RMSNorm(norm_dim, eps=self.qk_norm_eps)
            self.k_norm = torch.nn.RMSNorm(norm_dim, eps=self.qk_norm_eps)


class _TritonMultiHeadAttention(TritonMultiHeadAttention):
    """Canonical Triton attention implementation used by benchmarks."""

    @property
    def query_projection(self) -> torch.nn.Linear:
        """Return the canonical query projection."""
        return self.q_proj

    @property
    def key_projection(self) -> torch.nn.Linear:
        """Return the canonical key projection."""
        return self.k_proj

    @property
    def value_projection(self) -> torch.nn.Linear:
        """Return the canonical value projection."""
        return self.v_proj

    @property
    def output_projection(self) -> torch.nn.Linear:
        """Return the canonical output projection."""
        return self.output_proj

    @property
    def query_norm(self) -> torch.nn.Module:
        """Return the canonical query normalization."""
        return self.q_norm

    @property
    def key_norm(self) -> torch.nn.Module:
        """Return the canonical key normalization."""
        return self.k_norm

    def __init__(
        self,
        query_dim: int,
        n_heads: int = 8,
        head_dim: int = 64,
        *,
        context_dim: int | None = None,
        attention_type: AttentionType = AttentionType.SELF_ATTENTION,
        qkv_fusion_option: QKVFusionOption = QKVFusionOption.FULL,
        qkv_bias: bool = False,
        output_bias: bool = False,
        qk_norm_scope: QKNormScope = QKNormScope.HEAD,
        rope_interleaved: bool = False,
        use_fp8: bool = False,
        sdpa_backend: SDPABackend = SDPABackend.CUDNN,
    ) -> None:
        """Initialize canonical projections and normalization modules."""
        super().__init__(
            query_dim=query_dim,
            n_heads=n_heads,
            head_dim=head_dim,
            context_dim=context_dim,
            attention_type=attention_type,
            qkv_fusion_option=qkv_fusion_option,
            qk_norm_scope=qk_norm_scope,
            rope_interleaved=rope_interleaved,
            use_fp8=use_fp8,
            sdpa_backend=sdpa_backend,
        )
        self.q_proj = torch.nn.Linear(self.query_dim, self.inner_dim, bias=qkv_bias)
        self.k_proj = torch.nn.Linear(self.context_dim, self.inner_dim, bias=qkv_bias)
        self.v_proj = torch.nn.Linear(self.context_dim, self.inner_dim, bias=qkv_bias)
        self.output_proj = torch.nn.Linear(
            self.inner_dim, self.query_dim, bias=output_bias
        )
        if self.qk_norm_scope is QKNormScope.NONE:
            self.q_norm = torch.nn.Identity()
            self.k_norm = torch.nn.Identity()
        else:
            norm_dim = (
                self.head_dim
                if self.qk_norm_scope is QKNormScope.HEAD
                else self.inner_dim
            )
            self.q_norm = torch.nn.RMSNorm(norm_dim, eps=self.qk_norm_eps)
            self.k_norm = torch.nn.RMSNorm(norm_dim, eps=self.qk_norm_eps)
        self._initialize_derived_weights()


_Attention = _TorchMultiHeadAttention | _TritonMultiHeadAttention

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
    case: _ImplementationCase,
    *,
    attention_type: AttentionType,
    qk_norm_scope: QKNormScope,
    rope_interleaved: bool,
    bias: bool,
) -> _Attention:
    """Build one weight-matched Torch reference or Triton implementation.

    Args:
        case: Implementation-specific backend, precision, and fusion settings.
        attention_type: Whether to configure streaming self-attention or static
            cross-attention.
        qk_norm_scope: Shared Q/K normalization policy.
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
        rope_interleaved=rope_interleaved,
    )
    if case.implementation is _Implementation.REFERENCE_TORCH:
        return reference

    assert case.sdpa_backend is not None
    assert case.qkv_fusion_option is not None
    triton_attention = _TritonMultiHeadAttention(
        query_dim=_QUERY_DIM,
        context_dim=_QUERY_DIM,
        n_heads=_N_HEADS,
        head_dim=_HEAD_DIM,
        attention_type=attention_type,
        qkv_fusion_option=case.qkv_fusion_option,
        qkv_bias=bias,
        output_bias=bias,
        qk_norm_scope=qk_norm_scope,
        rope_interleaved=rope_interleaved,
        use_fp8=case.use_fp8,
        sdpa_backend=case.sdpa_backend,
    )
    triton_attention.load_state_dict(reference.state_dict(), strict=True)
    return triton_attention


@torch.inference_mode()
def _benchmark_multi_head_attention(
    benchmark: BenchmarkFixture,
    case: _ImplementationCase,
    *,
    attention_type: AttentionType,
    qk_norm_scope: QKNormScope,
    rope_interleaved: bool,
    bias: bool,
) -> None:
    """Run one synchronized attention benchmark within a shared-policy group.

    Streaming self-attention times forward over a prefilled rolling cache.
    Cross-attention times static K/V preparation and forward together so the
    requested fusion variants exercise the work they actually change.

    Args:
        benchmark: Pytest benchmark fixture used to record synchronized timings.
        case: Implementation-specific backend, precision, and fusion settings.
        attention_type: Self- or cross-attention benchmark family.
        qk_norm_scope: Shared Q/K normalization policy.
        rope_interleaved: Shared rotary-pair layout.
        bias: Whether every Q/K/V and output projection uses a bias.
    """
    if not torch.cuda.is_bf16_supported():
        pytest.skip("Multi-head attention benchmark requires bfloat16 support.")

    device = torch.device("cuda")
    is_triton = case.implementation is _Implementation.TRITON
    if is_triton and torch.cuda.get_device_capability(device)[0] < 9:
        pytest.skip(
            "Triton accelerated attention requires compute capability 9.0 or newer."
        )

    torch.manual_seed(_SEED)
    attention = _make_attention(
        case,
        attention_type=attention_type,
        qk_norm_scope=qk_norm_scope,
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
    rope_freqs = [
        torch.randn(
            _CHUNK_SIZE,
            1,
            1,
            _HEAD_DIM,
            generator=generator,
            device=device,
            dtype=torch.float32,
        )
        for _ in range(_WINDOW_CHUNKS + 1)
    ]

    attention_label = (
        "self" if attention_type is AttentionType.SELF_ATTENTION else "cross"
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
            "bias",
            bias_label,
        )
    )

    sdpa_backend = (
        case.sdpa_backend.value if case.sdpa_backend is not None else "torch_auto"
    )
    qkv_fusion_option = (
        case.qkv_fusion_option.value
        if case.qkv_fusion_option is not None
        else "not_applicable"
    )
    cache_dtype = (
        torch.float8_e4m3fn
        if (is_triton and case.use_fp8 and case.sdpa_backend is SDPABackend.TRITON)
        else _DTYPE
    )
    benchmark.extra_info.update(
        {
            "implementation": case.implementation.value,
            "implementation_case": case.id,
            "attention_type": attention_type.value,
            "timed_region": (
                "forward"
                if attention_type is AttentionType.SELF_ATTENTION
                else "compute_kv_and_forward"
            ),
            "batch_size": _BATCH_SIZE,
            "query_tokens": _CHUNK_SIZE,
            "context_tokens": _WINDOW_SIZE,
            "query_dim": _QUERY_DIM,
            "context_dim": _QUERY_DIM,
            "num_heads": _N_HEADS,
            "head_dim": _HEAD_DIM,
            "parameter_count": sum(
                parameter.numel() for parameter in attention.parameters()
            ),
            "checkpoint": "random_init_shared_weights",
            "dtype": str(_DTYPE),
            "cache_dtype": str(cache_dtype),
            "bias": bias,
            "qkv_bias": bias,
            "output_bias": bias,
            "qk_norm_scope": qk_norm_scope.value,
            "rope_interleaved": rope_interleaved,
            "sdpa_backend": sdpa_backend,
            "qkv_fusion_option": qkv_fusion_option,
            "use_fp8": case.use_fp8 if is_triton else False,
            "cache_state": (
                "full_rolling_window"
                if attention_type is AttentionType.SELF_ATTENTION
                else "rebuilt_static_context"
            ),
            "cache_prefill_chunks": (
                _WINDOW_CHUNKS if attention_type is AttentionType.SELF_ATTENTION else 0
            ),
            "gpu": torch.cuda.get_device_name(device),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "warmup_rounds": _WARMUP_ROUNDS,
            "benchmark_rounds": _BENCHMARK_ROUNDS,
            "seed": _SEED,
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
        torch.cuda.synchronize()

        # Static K/V projection is part of this end-to-end cross-attention
        # measurement because fusion changes that stage, not query-only forward.
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

    assert output.shape == query.shape
    assert torch.isfinite(output).all()


@pytest.mark.parametrize(
    "case",
    _IMPLEMENTATION_CASES,
    ids=lambda case: case.id,
)
@pytest.mark.parametrize(
    "qk_norm_scope,rope_interleaved,bias",
    _SHARED_CONFIGS,
)
def test_self_attention_benchmark(
    benchmark: BenchmarkFixture,
    case: _ImplementationCase,
    qk_norm_scope: QKNormScope,
    rope_interleaved: bool,
    bias: bool,
) -> None:
    """Compare streaming self-attention within one shared-policy group.

    Args:
        benchmark: Pytest benchmark fixture used to record synchronized timings.
        case: Torch reference or one Triton backend, precision, and fusion row.
        qk_norm_scope: Shared Q/K normalization policy defining the group.
        rope_interleaved: Shared rotary-pair layout defining the group.
        bias: Shared projection-bias policy defining the group.
    """
    _benchmark_multi_head_attention(
        benchmark,
        case,
        attention_type=AttentionType.SELF_ATTENTION,
        qk_norm_scope=qk_norm_scope,
        rope_interleaved=rope_interleaved,
        bias=bias,
    )


@pytest.mark.parametrize(
    "case",
    _IMPLEMENTATION_CASES,
    ids=lambda case: case.id,
)
@pytest.mark.parametrize(
    "qk_norm_scope,rope_interleaved,bias",
    _SHARED_CONFIGS,
)
def test_cross_attention_benchmark(
    benchmark: BenchmarkFixture,
    case: _ImplementationCase,
    qk_norm_scope: QKNormScope,
    rope_interleaved: bool,
    bias: bool,
) -> None:
    """Compare end-to-end cross-attention within one shared-policy group.

    Args:
        benchmark: Pytest benchmark fixture used to record synchronized timings.
        case: Torch reference or one Triton backend, precision, and fusion row.
        qk_norm_scope: Shared Q/K normalization policy defining the group.
        rope_interleaved: Shared rotary-pair layout defining the group.
        bias: Shared projection-bias policy defining the group.
    """
    _benchmark_multi_head_attention(
        benchmark,
        case,
        attention_type=AttentionType.CROSS_ATTENTION,
        qk_norm_scope=qk_norm_scope,
        rope_interleaved=rope_interleaved,
        bias=bias,
    )
