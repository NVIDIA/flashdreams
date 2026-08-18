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

"""Parity, cache-lifecycle, and policy tests for Triton multi-head attention."""

from __future__ import annotations

from typing import cast
from unittest.mock import MagicMock

import pytest
import torch

import flashdreams.accelerated.multi_head_attention_triton as triton_mha
from flashdreams.accelerated.multi_head_attention import (
    AttentionType,
    MultiHeadAttention,
    QKNormScope,
)
from flashdreams.accelerated.multi_head_attention_torch import TorchMultiHeadAttention
from flashdreams.accelerated.multi_head_attention_triton import (
    QKVFusionOption,
    SDPABackend,
    TritonMultiHeadAttention,
)
from flashdreams.core.attention import (
    BlockKVCache,
    RotaryPositionEmbedding3D,
)


class _TorchMultiHeadAttention(TorchMultiHeadAttention):
    """Weight-compatible PyTorch reference used for Triton parity tests."""

    @property
    def query_projection(self) -> torch.nn.Linear:
        """Return the checkpoint-compatible ``q_proj`` module."""
        return self.q_proj

    @property
    def key_projection(self) -> torch.nn.Linear:
        """Return the checkpoint-compatible ``k_proj`` module."""
        return self.k_proj

    @property
    def value_projection(self) -> torch.nn.Linear:
        """Return the checkpoint-compatible ``v_proj`` module."""
        return self.v_proj

    @property
    def output_projection(self) -> torch.nn.Linear:
        """Return the checkpoint-compatible ``output_proj`` module."""
        return self.output_proj

    @property
    def query_norm(self) -> torch.nn.Module:
        """Return the configured query normalization module."""
        return self.q_norm

    @property
    def key_norm(self) -> torch.nn.Module:
        """Return the configured key normalization module."""
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
        qk_norm_eps: float = 1e-6,
        qk_norm_scope: QKNormScope = QKNormScope.HEAD,
        rope_interleaved: bool = False,
    ) -> None:
        """Initialize the checkpoint-compatible PyTorch parity reference.

        Args:
            query_dim: Query input and output width ``Q``.
            n_heads: Number of attention heads ``H``.
            head_dim: Per-head feature width ``D``.
            context_dim: Context width ``C``; ``None`` uses ``query_dim``.
            attention_type: Select streaming self-attention or static
                cross-attention.
            qkv_bias: Add biases to the Q/K/V projections.
            output_bias: Add a bias to the output projection.
            qk_norm_eps: Epsilon for query and key RMS normalization.
            qk_norm_scope: Select per-head, full-inner-width, or disabled Q/K
                normalization.
            rope_interleaved: Rotate adjacent RoPE feature pairs when ``True``.
        """
        super().__init__(
            query_dim=query_dim,
            n_heads=n_heads,
            head_dim=head_dim,
            context_dim=context_dim,
            attention_type=attention_type,
            qk_norm_eps=qk_norm_eps,
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
    """Checkpoint-compatible Triton implementation used by backend tests."""

    @property
    def query_projection(self) -> torch.nn.Linear:
        """Return the checkpoint-compatible ``q_proj`` module."""
        return self.q_proj

    @property
    def key_projection(self) -> torch.nn.Linear:
        """Return the checkpoint-compatible ``k_proj`` module."""
        return self.k_proj

    @property
    def value_projection(self) -> torch.nn.Linear:
        """Return the checkpoint-compatible ``v_proj`` module."""
        return self.v_proj

    @property
    def output_projection(self) -> torch.nn.Linear:
        """Return the checkpoint-compatible ``output_proj`` module."""
        return self.output_proj

    @property
    def query_norm(self) -> torch.nn.Module:
        """Return the configured query normalization module."""
        return self.q_norm

    @property
    def key_norm(self) -> torch.nn.Module:
        """Return the configured key normalization module."""
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
        qk_norm_eps: float = 1e-6,
        qk_norm_scope: QKNormScope = QKNormScope.HEAD,
        rope_interleaved: bool = False,
        use_fp8: bool = False,
        sdpa_backend: SDPABackend = SDPABackend.CUDNN,
    ) -> None:
        """Initialize the checkpoint-compatible Triton test implementation.

        Args:
            query_dim: Query input and output width ``Q``.
            n_heads: Number of attention heads ``H``.
            head_dim: Per-head feature width ``D``.
            context_dim: Context width ``C``; ``None`` uses ``query_dim``.
            attention_type: Select streaming self-attention or static
                cross-attention.
            qkv_fusion_option: Select independent, K/V-fused, or fully fused
                projection weights.
            qkv_bias: Add biases to the Q/K/V projections.
            output_bias: Add a bias to the output projection.
            qk_norm_eps: Epsilon for query and key RMS normalization.
            qk_norm_scope: Select per-head, full-inner-width, or disabled Q/K
                normalization.
            rope_interleaved: Rotate adjacent RoPE feature pairs when ``True``.
            use_fp8: Quantize projection weights and supported attention storage.
            sdpa_backend: Select the cuDNN or Triton attention kernel.
        """
        super().__init__(
            query_dim=query_dim,
            n_heads=n_heads,
            head_dim=head_dim,
            context_dim=context_dim,
            attention_type=attention_type,
            qkv_fusion_option=qkv_fusion_option,
            qk_norm_eps=qk_norm_eps,
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


@pytest.fixture(scope="module")
def tma_device() -> torch.device:
    """Return a CUDA device that supports tensor-memory acceleration.

    Tests using this fixture skip when CUDA is unavailable or the device has a
    compute capability below 9.0.

    Returns:
        CUDA device used by the GPU attention tests.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA required.")
    device = torch.device("cuda")
    if torch.cuda.get_device_capability(device)[0] < 9:
        pytest.skip("TMA attention requires compute capability 9.0 or newer.")
    return device


@pytest.mark.ci_gpu
@pytest.mark.parametrize(
    "sdpa_backend", tuple(SDPABackend), ids=lambda backend: backend.value
)
def test_attention_dispatches_configured_backend(
    monkeypatch: pytest.MonkeyPatch,
    tma_device: torch.device,
    sdpa_backend: SDPABackend,
) -> None:
    """Route token-major Q/K/V through exactly the selected SDPA backend.

    Args:
        monkeypatch: Pytest patch helper used to replace both attention kernels.
        tma_device: CUDA device with tensor-memory acceleration support.
        sdpa_backend: Backend expected to receive ``[B, L, H, D]`` Q/K/V.
    """
    attention = _TritonMultiHeadAttention(
        128,
        n_heads=2,
        head_dim=64,
        sdpa_backend=sdpa_backend,
    )
    query = torch.empty((1, 2, 2, 64), device=tma_device, dtype=torch.bfloat16)
    calls: list[SDPABackend] = []

    def record_cudnn(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        """Record a cuDNN dispatch and return ``query`` unchanged.

        Args:
            query: Query tensor after conversion to cuDNN's head-major layout.
            key: Key tensor after conversion to cuDNN's head-major layout.
            value: Value tensor after conversion to cuDNN's head-major layout.

        Returns:
            Unmodified query tensor used as an identity kernel result.
        """
        del key, value
        calls.append(SDPABackend.CUDNN)
        return query

    def record_triton(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        """Record a Triton dispatch and return ``query`` unchanged.

        Args:
            query: Token-major query tensor ``[B, L, H, D]``.
            key: Token-major key tensor ``[B, S, H, D]``.
            value: Token-major value tensor ``[B, S, H, D]``.

        Returns:
            Unmodified query tensor used as an identity kernel result.
        """
        del key, value
        calls.append(SDPABackend.TRITON)
        return query

    # Replace both kernels with identity recorders so dispatch is observable
    # without coupling this test to their numerical implementations.
    cudnn_kernel = MagicMock()
    monkeypatch.setattr(triton_mha.F, "scaled_dot_product_attention", record_cudnn)
    monkeypatch.setattr(
        triton_mha.torch.nn.attention,
        "sdpa_kernel",
        cudnn_kernel,
    )
    monkeypatch.setattr(triton_mha, "flash_attention_2_tma", record_triton)

    # Identity output proves the selected stub ran; the call list and context
    # manager assertions exclude an accidental fallback to the other backend.
    assert torch.equal(attention._attention(query, query, query), query)
    assert calls == [sdpa_backend]
    if sdpa_backend is SDPABackend.CUDNN:
        cudnn_kernel.assert_called_once_with(
            torch.nn.attention.SDPBackend.CUDNN_ATTENTION
        )
    else:
        cudnn_kernel.assert_not_called()


def _make_cache(device: torch.device, n_heads: int, head_dim: int) -> BlockKVCache:
    """Build a BF16 rolling cache with an immutable sink prefix.

    Args:
        device: Device that owns the cache storage.
        n_heads: Head count ``H`` in the logical ``[B, S, H, D]`` layout.
        head_dim: Per-head feature width ``D``.

    Returns:
        Empty cache with a four-token sink and a two-chunk local window.
    """
    return BlockKVCache(
        k_shape=(1, 36, n_heads, head_dim),
        v_shape=(1, 36, n_heads, head_dim),
        seq_dim=1,
        chunk_size=16,
        window_size=32,
        sink_size=4,
        device=device,
        dtype=torch.bfloat16,
    )


@pytest.mark.ci_cpu
def test_full_self_attention_keeps_fused_query_local_to_forward(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pass fused Q directly from cache update to attention within ``forward``.

    The generic :class:`BlockKVCache` must contain only persistent K/V state;
    processed Q stays local to the Triton call instead of crossing stages through
    a backend-specific cache subtype.

    Args:
        monkeypatch: Pytest patch helper used to isolate forward-path routing.
    """
    # Triton owns the only runtime entry point rather than inheriting a staged
    # update/query implementation from the abstract interface.
    assert TritonMultiHeadAttention.forward is not MultiHeadAttention.forward
    assert "forward" in TritonMultiHeadAttention.__dict__

    attention = _TritonMultiHeadAttention(
        query_dim=32,
        n_heads=2,
        head_dim=16,
        qk_norm_scope=QKNormScope.NONE,
    )
    cache = attention.allocate_kv_cache(
        batch_size=1,
        chunk_size=2,
        window_size=2,
        sink_size=0,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    # Cache allocation must not introduce transient query fields.
    assert type(cache) is BlockKVCache
    assert not hasattr(cache, "_processed_query")
    assert not hasattr(cache, "_query_source")

    x = torch.randn(1, 2, 32)
    raw_query = torch.randn(1, 2, 2, 16)
    key = torch.randn_like(raw_query)
    value = torch.randn_like(raw_query)
    processed_query = raw_query + 1

    # Replace projection, preprocessing, and attention with CPU-safe identity
    # stubs that preserve the fused query object's identity.
    project_qkv = MagicMock(return_value=(raw_query, key, value))
    project_query = MagicMock()
    fused_update = MagicMock(return_value=processed_query)
    attention_kernel = MagicMock(side_effect=lambda query, key, value: query)
    output_projection = MagicMock(side_effect=lambda output, output_dtype: output)
    monkeypatch.setattr(attention, "_validate_fused_update_inputs", MagicMock())
    monkeypatch.setattr(attention, "_project_qkv", project_qkv)
    monkeypatch.setattr(attention, "_project_query", project_query)
    monkeypatch.setattr(attention, "_attention", attention_kernel)
    monkeypatch.setattr(attention, "_project_output", output_projection)
    monkeypatch.setattr(
        triton_mha,
        "fused_rms_rope_kv_cache_update",
        fused_update,
    )

    # Keep the caller-owned cache lifecycle around the single forward update.
    cache.before_update(0)
    output = attention(x, cache)

    # Full fusion must bypass the standalone query projection and feed the exact
    # query returned by fused preprocessing into attention.
    project_qkv.assert_called_once()
    project_query.assert_not_called()
    fused_update.assert_called_once()
    assert attention_kernel.call_args.args[0] is processed_query
    torch.testing.assert_close(output, processed_query.reshape_as(x))
    cache.after_update(0)


@pytest.mark.ci_gpu
@pytest.mark.parametrize(
    "sdpa_backend", tuple(SDPABackend), ids=lambda backend: backend.value
)
@pytest.mark.parametrize(
    (
        "qk_norm_scope",
        "rope_interleaved",
        "projection_bias",
        "n_heads",
        "head_dim",
    ),
    [
        pytest.param(QKNormScope.HEAD, False, False, 16, 128, id="cosmos"),
        pytest.param(QKNormScope.INNER, True, True, 12, 128, id="wan"),
        pytest.param(QKNormScope.NONE, False, False, 2, 64, id="no-qk-norm"),
    ],
)
def test_triton_attention_matches_reference_through_window_roll(
    tma_device: torch.device,
    qk_norm_scope: QKNormScope,
    rope_interleaved: bool,
    projection_bias: bool,
    n_heads: int,
    head_dim: int,
    sdpa_backend: SDPABackend,
) -> None:
    """Match streaming Triton attention through cache fill and rolling updates.

    Weight-identical PyTorch and Triton modules consume the same BF16 chunks and
    RoPE frequencies. Chunk indices ``(0, 1, 2, 2)`` cover initial filling,
    steady-state rolling, and replacement at a repeated logical step.

    Args:
        tma_device: CUDA device with tensor-memory acceleration support.
        qk_norm_scope: Query/key normalization layout under test.
        rope_interleaved: Whether RoPE rotates adjacent feature pairs.
        projection_bias: Whether all projections include checkpoint biases.
        n_heads: Attention head count ``H`` for the recipe-shaped case.
        head_dim: Per-head feature width ``D``.
        sdpa_backend: cuDNN or Triton attention kernel under test.
    """
    # Build weight-identical implementations so differences isolate the Triton
    # projection, cache-write, and attention paths.
    torch.manual_seed(7)
    inner_dim = n_heads * head_dim
    reference = _TorchMultiHeadAttention(
        query_dim=inner_dim,
        n_heads=n_heads,
        head_dim=head_dim,
        qkv_bias=projection_bias,
        output_bias=projection_bias,
        qk_norm_scope=qk_norm_scope,
        rope_interleaved=rope_interleaved,
    ).to(device=tma_device, dtype=torch.bfloat16)
    triton_attention = _TritonMultiHeadAttention(
        query_dim=inner_dim,
        n_heads=n_heads,
        head_dim=head_dim,
        qkv_fusion_option=QKVFusionOption.FULL,
        qkv_bias=projection_bias,
        output_bias=projection_bias,
        qk_norm_scope=qk_norm_scope,
        rope_interleaved=rope_interleaved,
        sdpa_backend=sdpa_backend,
    ).to(device=tma_device, dtype=torch.bfloat16)
    triton_attention.load_state_dict(reference.state_dict())
    reference.eval()
    triton_attention.eval()

    # Allocate caches with the same logical ``[B, S, H, D]`` capacity and drive
    # both with identical positional frequencies.
    reference_cache = _make_cache(tma_device, n_heads, head_dim)
    triton_cache = triton_attention.allocate_kv_cache(
        batch_size=1,
        chunk_size=16,
        window_size=32,
        sink_size=4,
        device=tma_device,
        dtype=torch.bfloat16,
    )
    rope = RotaryPositionEmbedding3D(
        head_dim=head_dim,
        len_t=1,
        len_h=1,
        len_w=16,
        interleaved=rope_interleaved,
        device=tma_device,
    )
    generator = torch.Generator(device=tma_device).manual_seed(11)

    with torch.inference_mode():
        # Follow the caller-owned before/forward/after lifecycle for every chunk;
        # the repeated final index exercises overwrite without advancing state.
        for chunk_idx in (0, 1, 2, 2):
            x = torch.randn(
                1,
                16,
                inner_dim,
                generator=generator,
                device=tma_device,
                dtype=torch.bfloat16,
            )
            rope_freqs = rope.shift_t(chunk_idx)
            reference_cache.before_update(chunk_idx)
            triton_cache.before_update(chunk_idx)

            expected = reference(x, reference_cache, rope_freqs)
            actual = triton_attention(x, triton_cache, rope_freqs)

            # Compare both output parity and the visible K/V state that feeds the
            # next streaming step.
            torch.testing.assert_close(actual, expected, atol=1e-2, rtol=1e-2)
            torch.testing.assert_close(
                triton_cache.cached_k(),
                reference_cache.cached_k(),
                atol=2e-2,
                rtol=2e-2,
            )
            torch.testing.assert_close(
                triton_cache.cached_v(),
                reference_cache.cached_v(),
                atol=2e-3,
                rtol=1e-2,
            )
            reference_cache.after_update(chunk_idx)
            triton_cache.after_update(chunk_idx)


@pytest.mark.ci_gpu
@pytest.mark.parametrize(
    "qkv_fusion_option",
    [QKVFusionOption.NONE, QKVFusionOption.FUSE_KV],
    ids=lambda option: option.value,
)
def test_unfused_self_attention_matches_reference(
    tma_device: torch.device,
    qkv_fusion_option: QKVFusionOption,
) -> None:
    """Match one non-fully-fused self-attention update to PyTorch.

    Args:
        tma_device: CUDA device with tensor-memory acceleration support.
        qkv_fusion_option: Independent or K/V-fused projection path under test.
    """
    # Share checkpoint weights and cache geometry so only projection fusion and
    # the Triton runtime path differ.
    torch.manual_seed(13)
    reference = _TorchMultiHeadAttention(
        query_dim=128,
        n_heads=2,
        head_dim=64,
        qkv_bias=True,
        qk_norm_scope=QKNormScope.NONE,
    ).to(device=tma_device, dtype=torch.bfloat16)
    attention = _TritonMultiHeadAttention(
        query_dim=128,
        n_heads=2,
        head_dim=64,
        qkv_fusion_option=qkv_fusion_option,
        qkv_bias=True,
        qk_norm_scope=QKNormScope.NONE,
        sdpa_backend=SDPABackend.CUDNN,
    ).to(device=tma_device, dtype=torch.bfloat16)
    attention.load_state_dict(reference.state_dict(), strict=True)
    reference_cache = _make_cache(tma_device, 2, 64)
    cache = attention.allocate_kv_cache(
        batch_size=1,
        chunk_size=16,
        window_size=32,
        sink_size=4,
        device=tma_device,
        dtype=torch.bfloat16,
    )
    x = torch.randn((1, 16, 128), device=tma_device, dtype=torch.bfloat16)
    # Enter the same first-chunk update phase before either implementation writes.
    reference_cache.before_update(0)
    cache.before_update(0)

    expected = reference(x, reference_cache)
    actual = attention(x, cache)

    # Output and visible K/V parity jointly verify the projection and cache write.
    torch.testing.assert_close(actual, expected, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(
        cache.cached_k(), reference_cache.cached_k(), atol=2e-2, rtol=2e-2
    )
    torch.testing.assert_close(
        cache.cached_v(), reference_cache.cached_v(), atol=0, rtol=0
    )
    reference_cache.after_update(0)
    cache.after_update(0)


@pytest.mark.ci_gpu
@pytest.mark.parametrize(
    "sdpa_backend", tuple(SDPABackend), ids=lambda backend: backend.value
)
@pytest.mark.parametrize(
    (
        "qk_norm_scope",
        "rope_interleaved",
        "projection_bias",
        "n_heads",
        "head_dim",
    ),
    [
        pytest.param(QKNormScope.HEAD, False, False, 16, 128, id="cosmos"),
        pytest.param(QKNormScope.INNER, True, True, 12, 128, id="wan"),
        pytest.param(QKNormScope.NONE, False, False, 2, 64, id="no-qk-norm"),
    ],
)
def test_fp8_triton_attention_matches_bf16_reference_through_window_roll(
    tma_device: torch.device,
    qk_norm_scope: QKNormScope,
    rope_interleaved: bool,
    projection_bias: bool,
    n_heads: int,
    head_dim: int,
    sdpa_backend: SDPABackend,
) -> None:
    """Bound FP8 error through cache fill and rolling self-attention updates.

    The BF16 reference and FP8-enabled Triton module share master weights. The
    test covers both native cuDNN cache storage and E4M3 Triton cache storage.

    Args:
        tma_device: CUDA device with tensor-memory acceleration support.
        qk_norm_scope: Query/key normalization layout under test.
        rope_interleaved: Whether RoPE rotates adjacent feature pairs.
        projection_bias: Whether all projections include checkpoint biases.
        n_heads: Attention head count ``H`` for the recipe-shaped case.
        head_dim: Per-head feature width ``D``.
        sdpa_backend: Backend determining attention and cache storage format.
    """
    # Keep BF16 master weights identical while enabling FP8 execution only on the
    # Triton implementation.
    torch.manual_seed(17)
    inner_dim = n_heads * head_dim
    reference = _TorchMultiHeadAttention(
        query_dim=inner_dim,
        n_heads=n_heads,
        head_dim=head_dim,
        qkv_bias=projection_bias,
        output_bias=projection_bias,
        qk_norm_scope=qk_norm_scope,
        rope_interleaved=rope_interleaved,
    ).to(device=tma_device, dtype=torch.bfloat16)
    triton_attention = _TritonMultiHeadAttention(
        query_dim=inner_dim,
        n_heads=n_heads,
        head_dim=head_dim,
        qkv_fusion_option=QKVFusionOption.FULL,
        qkv_bias=projection_bias,
        output_bias=projection_bias,
        qk_norm_scope=qk_norm_scope,
        rope_interleaved=rope_interleaved,
        use_fp8=True,
        sdpa_backend=sdpa_backend,
    ).to(device=tma_device, dtype=torch.bfloat16)
    triton_attention.load_state_dict(reference.state_dict())
    reference.eval()
    triton_attention.eval()

    reference_cache = _make_cache(tma_device, n_heads, head_dim)
    triton_cache = triton_attention.allocate_kv_cache(
        batch_size=1,
        chunk_size=16,
        window_size=32,
        sink_size=4,
        device=tma_device,
        dtype=torch.bfloat16,
    )
    # Verify logical ``[B, S, H, D]`` shape plus the physical layout required by
    # per-head normalization or token-wide normalization.
    assert isinstance(triton_cache, BlockKVCache)
    expected_cache_dtype = (
        torch.float8_e4m3fn if sdpa_backend is SDPABackend.TRITON else torch.bfloat16
    )
    assert triton_cache._k.dtype is expected_cache_dtype
    assert triton_cache._v.dtype is expected_cache_dtype
    assert triton_cache._k.shape == (1, 36, n_heads, head_dim)
    if qk_norm_scope is QKNormScope.HEAD:
        assert triton_cache._k.stride() == (
            36 * n_heads * head_dim,
            head_dim,
            36 * head_dim,
            1,
        )
        assert triton_cache._v.stride() == triton_cache._k.stride()
    else:
        assert triton_cache._k.is_contiguous()
        assert triton_cache._v.is_contiguous()
    rope = RotaryPositionEmbedding3D(
        head_dim=head_dim,
        len_t=1,
        len_h=1,
        len_w=16,
        interleaved=rope_interleaved,
        device=tma_device,
    )
    generator = torch.Generator(device=tma_device).manual_seed(19)

    with torch.inference_mode():
        # Follow the same before/forward/after lifecycle used by the BF16 parity
        # test while the cache transitions into steady-state storage.
        for chunk_idx in (0, 1, 2, 2):
            x = torch.randn(
                1,
                16,
                inner_dim,
                generator=generator,
                device=tma_device,
                dtype=torch.bfloat16,
            )
            rope_freqs = rope.shift_t(chunk_idx)
            reference_cache.before_update(chunk_idx)
            triton_cache.before_update(chunk_idx)

            expected = reference(x, reference_cache, rope_freqs)
            actual = triton_attention(x, triton_cache, rope_freqs)

            # Compare outputs and dequantized cache contents at the error bound
            # expected from row-scaled E4M3 projections and storage.
            torch.testing.assert_close(actual, expected, atol=5e-2, rtol=5e-2)
            torch.testing.assert_close(
                triton_cache.cached_k().to(torch.bfloat16),
                reference_cache.cached_k(),
                atol=1.5e-1,
                rtol=1.5e-1,
            )
            torch.testing.assert_close(
                triton_cache.cached_v().to(torch.bfloat16),
                reference_cache.cached_v(),
                atol=1.5e-1,
                rtol=1.5e-1,
            )
            reference_cache.after_update(chunk_idx)
            triton_cache.after_update(chunk_idx)


@pytest.mark.ci_gpu
@pytest.mark.parametrize(
    "sdpa_backend", tuple(SDPABackend), ids=lambda backend: backend.value
)
@pytest.mark.parametrize("use_fp8", [False, True], ids=["native", "fp8"])
def test_asymmetric_cross_attention_forward_matches_reference(
    tma_device: torch.device,
    sdpa_backend: SDPABackend,
    use_fp8: bool,
) -> None:
    """Match asymmetric cross-attention and preserve its static K/V cache.

    Context ``[2, 5, 128]`` and query ``[1, 2, 3, 256]`` flatten to the same
    effective batch ``B = 2`` while retaining different token and feature widths.

    Args:
        tma_device: CUDA device with tensor-memory acceleration support.
        sdpa_backend: cuDNN or Triton attention kernel under test.
        use_fp8: Whether projection and supported cache storage use E4M3.
    """
    # Load one checkpoint into both implementations so asymmetric projection and
    # backend precision are the only parity variables.
    torch.manual_seed(31)
    reference = _TorchMultiHeadAttention(
        query_dim=256,
        context_dim=128,
        n_heads=2,
        head_dim=64,
        attention_type=AttentionType.CROSS_ATTENTION,
        qkv_bias=True,
        output_bias=True,
    ).to(device=tma_device, dtype=torch.bfloat16)
    attention = _TritonMultiHeadAttention(
        query_dim=256,
        context_dim=128,
        n_heads=2,
        head_dim=64,
        attention_type=AttentionType.CROSS_ATTENTION,
        qkv_fusion_option=QKVFusionOption.FUSE_KV,
        qkv_bias=True,
        output_bias=True,
        use_fp8=use_fp8,
        sdpa_backend=sdpa_backend,
    ).to(device=tma_device, dtype=torch.bfloat16)
    attention.load_state_dict(reference.state_dict(), strict=True)

    # The leading query dimensions flatten to the context batch while output
    # reshaping must restore ``[1, 2, 3, 256]``.
    generator = torch.Generator(device=tma_device).manual_seed(37)
    context = torch.randn(
        2,
        5,
        128,
        generator=generator,
        device=tma_device,
        dtype=torch.bfloat16,
    )
    query = torch.randn(
        1,
        2,
        3,
        256,
        generator=generator,
        device=tma_device,
        dtype=torch.bfloat16,
    )
    context_rope = torch.randn(
        5,
        1,
        1,
        64,
        generator=generator,
        device=tma_device,
    )
    query_rope = torch.randn(
        3,
        1,
        1,
        64,
        generator=generator,
        device=tma_device,
    )

    # Materialize static context once, then snapshot Triton storage before two
    # read-only forward calls.
    reference_cache = reference.compute_kv(context, context_rope)
    cache = attention.compute_kv(context, context_rope)
    cached_key = cache.cached_k().clone()
    cached_value = cache.cached_v().clone()
    expected = reference(query, reference_cache, query_rope)
    actual = attention(query, cache, query_rope)
    repeated = attention(query, cache, query_rope)

    # Repeated output identity and byte-exact K/V snapshots enforce static-cache
    # reuse independently of numerical parity with the reference.
    tolerance = 5e-2 if use_fp8 else 1e-2
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
    torch.testing.assert_close(repeated, actual, atol=0, rtol=0)
    assert torch.equal(cache.cached_k(), cached_key)
    assert torch.equal(cache.cached_v(), cached_value)
    expected_cache_dtype = (
        torch.float8_e4m3fn
        if use_fp8 and sdpa_backend is SDPABackend.TRITON
        else torch.bfloat16
    )
    assert cache.cached_k().dtype is expected_cache_dtype
    assert cache.cached_v().dtype is expected_cache_dtype


@pytest.mark.parametrize(
    ("attention_type", "qkv_fusion_option", "context_dim"),
    [
        pytest.param(
            AttentionType.SELF_ATTENTION,
            QKVFusionOption.NONE,
            256,
            id="self-none",
        ),
        pytest.param(
            AttentionType.SELF_ATTENTION,
            QKVFusionOption.FULL,
            256,
            id="self-full",
        ),
        pytest.param(
            AttentionType.SELF_ATTENTION,
            QKVFusionOption.FUSE_KV,
            256,
            id="self-fuse-kv",
        ),
        pytest.param(
            AttentionType.CROSS_ATTENTION,
            QKVFusionOption.NONE,
            128,
            id="cross-none",
        ),
        pytest.param(
            AttentionType.CROSS_ATTENTION,
            QKVFusionOption.FULL,
            256,
            id="cross-full",
        ),
        pytest.param(
            AttentionType.CROSS_ATTENTION,
            QKVFusionOption.FUSE_KV,
            128,
            id="cross-fuse-kv",
        ),
    ],
)
@pytest.mark.ci_cpu
def test_qkv_fusion_option_controls_native_derived_weights(
    attention_type: AttentionType,
    qkv_fusion_option: QKVFusionOption,
    context_dim: int,
) -> None:
    """Build only the native derived buffers required by each fusion policy.

    ``FULL`` owns one QKV allocation whose K/V tail aliases the same storage;
    ``FUSE_KV`` owns only a K/V allocation; ``NONE`` owns no fused buffers.

    Args:
        attention_type: Self- or cross-attention configuration under test.
        qkv_fusion_option: Derived-weight topology expected from the module.
        context_dim: Context width compatible with the selected fusion policy.
    """
    attention = _TritonMultiHeadAttention(
        query_dim=256,
        context_dim=context_dim,
        attention_type=attention_type,
        n_heads=2,
        head_dim=64,
        qkv_fusion_option=qkv_fusion_option,
        qkv_bias=True,
        output_bias=True,
    )
    reference = _TorchMultiHeadAttention(
        query_dim=256,
        context_dim=context_dim,
        attention_type=attention_type,
        n_heads=2,
        head_dim=64,
        qkv_bias=True,
        output_bias=True,
    )

    # Verify the public policy and checkpoint-native projection parameters before
    # inspecting derived execution-only buffers.
    assert attention.qkv_fusion_option is qkv_fusion_option
    assert attention.q_proj.bias is not None
    assert attention.k_proj.bias is not None
    assert attention.v_proj.bias is not None
    # ``FULL`` materializes Q/K/V in row order for a single projection GEMM.
    if qkv_fusion_option is QKVFusionOption.FULL:
        assert attention._derived_weights.fused_qkv_weight is not None
        assert attention._derived_weights.fused_qkv_bias is not None
        assert torch.equal(
            attention._derived_weights.fused_qkv_weight,
            torch.cat(
                (
                    attention.q_proj.weight,
                    attention.k_proj.weight,
                    attention.v_proj.weight,
                )
            ),
        )
        assert torch.equal(
            attention._derived_weights.fused_qkv_bias,
            torch.cat(
                (
                    attention.q_proj.bias,
                    attention.k_proj.bias,
                    attention.v_proj.bias,
                )
            ),
        )
    else:
        assert attention._derived_weights.fused_qkv_weight is None
        assert attention._derived_weights.fused_qkv_bias is None

    # Both fused policies expose K/V rows; ``FULL`` must expose a zero-copy tail view
    # rather than allocating the same rows twice.
    if qkv_fusion_option is not QKVFusionOption.NONE:
        fused_kv_weight = attention._derived_weights.fused_kv_weight
        fused_kv_bias = attention._derived_weights.fused_kv_bias
        assert fused_kv_weight is not None and fused_kv_bias is not None
        assert torch.equal(
            fused_kv_weight,
            torch.cat((attention.k_proj.weight, attention.v_proj.weight)),
        )
        assert torch.equal(
            fused_kv_bias,
            torch.cat((attention.k_proj.bias, attention.v_proj.bias)),
        )
        if qkv_fusion_option is QKVFusionOption.FULL:
            fused_qkv_weight = attention._derived_weights.fused_qkv_weight
            fused_qkv_bias = attention._derived_weights.fused_qkv_bias
            assert fused_qkv_weight is not None and fused_qkv_bias is not None
            assert (
                fused_kv_weight.untyped_storage().data_ptr()
                == fused_qkv_weight.untyped_storage().data_ptr()
            )
            assert (
                fused_kv_weight.storage_offset()
                == attention.inner_dim * attention.context_dim
            )
            assert (
                fused_kv_bias.untyped_storage().data_ptr()
                == fused_qkv_bias.untyped_storage().data_ptr()
            )
            assert fused_kv_bias.storage_offset() == attention.inner_dim
    else:
        assert attention._derived_weights.fused_kv_weight is None
        assert attention._derived_weights.fused_kv_bias is None

    # Native derived tensors have no FP8 scales and remain absent from checkpoints.
    assert attention._derived_weights.fused_qkv_weight_scale is None
    assert attention._derived_weights.fused_kv_weight_scale is None
    assert attention.state_dict().keys() == reference.state_dict().keys()


@pytest.mark.ci_cpu
def test_qkv_fusion_option_constructor_policy() -> None:
    """Enforce enum-only fusion policies and full fusion's symmetric width."""
    # Keep the serialized enum surface and default stable for callers.
    assert {option.value for option in QKVFusionOption} == {
        "none",
        "full",
        "fuse_kv",
    }
    assert (
        _TritonMultiHeadAttention(128, head_dim=64).qkv_fusion_option
        is QKVFusionOption.FULL
    )

    # Reject string lookalikes and asymmetric inputs before building fused weights.
    string_option = cast(QKVFusionOption, "full")
    with pytest.raises(TypeError, match="QKVFusionOption"):
        _TritonMultiHeadAttention(
            128,
            head_dim=64,
            qkv_fusion_option=string_option,
        )
    with pytest.raises(ValueError, match="query_dim.*context_dim"):
        _TritonMultiHeadAttention(
            query_dim=256,
            context_dim=128,
            attention_type=AttentionType.CROSS_ATTENTION,
            head_dim=64,
            qkv_fusion_option=QKVFusionOption.FULL,
        )


@pytest.mark.parametrize(
    ("qkv_fusion_option", "context_dim", "expected_linear_calls"),
    [
        pytest.param(QKVFusionOption.NONE, 128, 2, id="none"),
        pytest.param(QKVFusionOption.FULL, 256, 1, id="full"),
        pytest.param(QKVFusionOption.FUSE_KV, 128, 1, id="fuse-kv"),
    ],
)
@pytest.mark.ci_cpu
def test_native_kv_projection_obeys_fusion_option(
    monkeypatch: pytest.MonkeyPatch,
    qkv_fusion_option: QKVFusionOption,
    context_dim: int,
    expected_linear_calls: int,
) -> None:
    """Match separate K/V results with the selected native GEMM topology.

    Args:
        monkeypatch: Pytest patch helper used to count linear projections.
        qkv_fusion_option: Independent, full-QKV, or K/V-fused policy under test.
        context_dim: Context width accepted by the selected policy.
        expected_linear_calls: Number of native K/V GEMMs the policy should issue.
    """
    attention = _TritonMultiHeadAttention(
        query_dim=256,
        context_dim=context_dim,
        attention_type=AttentionType.CROSS_ATTENTION,
        n_heads=2,
        head_dim=64,
        qkv_fusion_option=qkv_fusion_option,
        qkv_bias=True,
        qk_norm_scope=QKNormScope.NONE,
    )
    context = torch.randn(2, 5, context_dim)
    head_shape = (-1, 5, attention.n_heads, attention.head_dim)
    # Compute the unfused checkpoint projections before wrapping ``F.linear`` so
    # the mock counts only calls made by ``_project_kv``.
    expected_key = attention.k_proj(context).reshape(head_shape)
    expected_value = attention.v_proj(context).reshape(head_shape)
    validate_tokens = MagicMock()
    linear = MagicMock(wraps=triton_mha.F.linear)
    monkeypatch.setattr(attention, "_validate_tokens", validate_tokens)
    monkeypatch.setattr(triton_mha.F, "linear", linear)

    # Execute the implementation path once, then verify both dispatch topology
    # and numerical equivalence to the separate checkpoint projections.
    key, value = attention._project_kv(context)

    validate_tokens.assert_called_once_with(context, attention.context_dim, "context")
    assert linear.call_count == expected_linear_calls
    if qkv_fusion_option is not QKVFusionOption.NONE:
        assert linear.call_args.args[1] is attention._derived_weights.fused_kv_weight
        assert linear.call_args.args[2] is attention._derived_weights.fused_kv_bias
    torch.testing.assert_close(key, expected_key)
    torch.testing.assert_close(value, expected_value)


@pytest.mark.ci_cpu
def test_native_full_qkv_projection_matches_separate_projections(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Match separate Q/K/V results with one full-fusion native GEMM.

    Args:
        monkeypatch: Pytest patch helper used to count native linear projections.
    """
    attention = _TritonMultiHeadAttention(
        query_dim=128,
        context_dim=128,
        attention_type=AttentionType.CROSS_ATTENTION,
        n_heads=2,
        head_dim=64,
        qkv_fusion_option=QKVFusionOption.FULL,
        qkv_bias=True,
        qk_norm_scope=QKNormScope.NONE,
    )
    x = torch.randn(2, 5, 128)
    head_shape = (-1, 5, attention.n_heads, attention.head_dim)
    # Materialize the three checkpoint projections before counting the fused path.
    expected_query = attention.q_proj(x).reshape(head_shape)
    expected_key = attention.k_proj(x).reshape(head_shape)
    expected_value = attention.v_proj(x).reshape(head_shape)
    linear = MagicMock(wraps=triton_mha.F.linear)
    monkeypatch.setattr(triton_mha.F, "linear", linear)

    # One linear call must preserve all three ``[B, L, H, D]`` results.
    query, key, value = attention._project_qkv(x)

    assert linear.call_count == 1
    torch.testing.assert_close(query, expected_query)
    torch.testing.assert_close(key, expected_key)
    torch.testing.assert_close(value, expected_value)


@pytest.mark.parametrize(
    ("qkv_fusion_option", "context_dim", "expected_source_shapes"),
    [
        pytest.param(
            QKVFusionOption.FULL,
            256,
            ((384, 256), (256, 128)),
            id="full",
        ),
        pytest.param(
            QKVFusionOption.FUSE_KV,
            128,
            ((256, 128), (128, 256), (256, 128)),
            id="fuse-kv",
        ),
        pytest.param(
            QKVFusionOption.NONE,
            128,
            ((128, 256), (128, 128), (128, 128), (256, 128)),
            id="none",
        ),
    ],
)
@pytest.mark.ci_cpu
def test_fp8_refresh_quantizes_minimal_sources_and_clears_stale_fusions(
    monkeypatch: pytest.MonkeyPatch,
    qkv_fusion_option: QKVFusionOption,
    context_dim: int,
    expected_source_shapes: tuple[tuple[int, int], ...],
) -> None:
    """Quantize minimal policy sources and discard stale fused buffers.

    Args:
        monkeypatch: Pytest patch helper used to record quantization sources.
        qkv_fusion_option: FP8 derived-weight topology under test.
        context_dim: Context width compatible with the selected fusion policy.
        expected_source_shapes: Ordered quantizer inputs, including output weight.
    """
    attention = _TritonMultiHeadAttention(
        query_dim=256,
        context_dim=context_dim,
        attention_type=AttentionType.CROSS_ATTENTION,
        n_heads=2,
        head_dim=64,
        qkv_fusion_option=qkv_fusion_option,
        use_fp8=True,
    )
    derived = attention._derived_weights
    # Poison every fused buffer so successful refresh must replace active tensors
    # and clear inactive tensors instead of passing on initial ``None`` values.
    stale = torch.ones(1)
    for name in (
        "fused_qkv_weight",
        "fused_qkv_bias",
        "fused_qkv_weight_scale",
        "fused_kv_weight",
        "fused_kv_bias",
        "fused_kv_weight_scale",
    ):
        setattr(derived, name, stale)

    # Record each source while preserving real row-scaled E4M3 quantization.
    quantize = MagicMock(wraps=triton_mha.quantize_fp8_weight)
    monkeypatch.setattr(triton_mha, "quantize_fp8_weight", quantize)

    attention._refresh_derived_weights()

    # Ordered shapes enforce the minimal call count: ``FULL`` quantizes QKV once,
    # ``FUSE_KV`` quantizes KV and Q once each, and ``NONE`` quantizes Q/K/V separately.
    sources = [call.args[0] for call in quantize.call_args_list]
    assert tuple(tuple(source.shape) for source in sources) == expected_source_shapes
    assert sources[-1] is attention.output_projection.weight

    if qkv_fusion_option is QKVFusionOption.FULL:
        torch.testing.assert_close(
            sources[0],
            torch.cat(
                (
                    attention.query_projection.weight,
                    attention.key_projection.weight,
                    attention.value_projection.weight,
                )
            ),
        )
    elif qkv_fusion_option is QKVFusionOption.FUSE_KV:
        torch.testing.assert_close(
            sources[0],
            torch.cat(
                (
                    attention.key_projection.weight,
                    attention.value_projection.weight,
                )
            ),
        )
        assert sources[1] is attention.query_projection.weight
    else:
        for source, projection in zip(
            sources[:-1],
            (
                attention.query_projection,
                attention.key_projection,
                attention.value_projection,
            ),
            strict=True,
        ):
            assert source is projection.weight

    # Active-buffer presence and stale-object identity validate the refreshed
    # topology independently of quantizer call arguments.
    qkv_active = qkv_fusion_option is QKVFusionOption.FULL
    kv_active = qkv_fusion_option is not QKVFusionOption.NONE
    assert (derived.fused_qkv_weight is not None) is qkv_active
    assert (derived.fused_qkv_weight_scale is not None) is qkv_active
    assert (derived.fused_kv_weight is not None) is kv_active
    assert (derived.fused_kv_weight_scale is not None) is kv_active
    assert derived.fused_qkv_bias is None
    assert derived.fused_kv_bias is None
    for tensor in (
        derived.fused_qkv_weight,
        derived.fused_qkv_weight_scale,
        derived.fused_kv_weight,
        derived.fused_kv_weight_scale,
    ):
        assert tensor is not stale


@pytest.mark.parametrize(
    ("qkv_fusion_option", "context_dim"),
    [
        pytest.param(QKVFusionOption.NONE, 128, id="none"),
        pytest.param(QKVFusionOption.FULL, 256, id="full"),
        pytest.param(QKVFusionOption.FUSE_KV, 128, id="fuse-kv"),
    ],
)
@pytest.mark.ci_cpu
def test_fp8_fusion_storage_refreshes_after_state_load(
    qkv_fusion_option: QKVFusionOption,
    context_dim: int,
) -> None:
    """Refresh FP8 buffers after state load while preserving storage topology.

    Args:
        qkv_fusion_option: FP8 ownership and aliasing policy under test.
        context_dim: Context width compatible with the selected fusion policy.
    """
    attention = _TritonMultiHeadAttention(
        query_dim=256,
        context_dim=context_dim,
        attention_type=AttentionType.CROSS_ATTENTION,
        n_heads=2,
        head_dim=64,
        qkv_fusion_option=qkv_fusion_option,
        use_fp8=True,
    )
    # Snapshot derived Q/K/V values, then load zero master weights to prove the
    # post-load hook reconstructs rather than reuses nonpersistent buffers.
    original_q = attention._derived_weights.q_weight_fp8
    original_k = attention._derived_weights.k_weight_fp8
    original_v = attention._derived_weights.v_weight_fp8
    assert original_q is not None
    assert original_k is not None
    assert original_v is not None
    original_q = original_q.clone()
    original_k = original_k.clone()
    original_v = original_v.clone()
    state = attention.state_dict()
    state["q_proj.weight"] = torch.zeros_like(state["q_proj.weight"])
    state["k_proj.weight"] = torch.zeros_like(state["k_proj.weight"])
    state["v_proj.weight"] = torch.zeros_like(state["v_proj.weight"])

    attention.load_state_dict(state, strict=True)

    q_weight = attention._derived_weights.q_weight_fp8
    k_weight = attention._derived_weights.k_weight_fp8
    v_weight = attention._derived_weights.v_weight_fp8
    q_scale = attention._derived_weights.q_weight_scale
    k_scale = attention._derived_weights.k_weight_scale
    v_scale = attention._derived_weights.v_weight_scale
    assert q_weight is not None and q_scale is not None
    assert k_weight is not None and k_scale is not None
    assert v_weight is not None and v_scale is not None
    assert q_weight.shape == attention.q_proj.weight.shape
    assert k_weight.shape == attention.k_proj.weight.shape
    assert v_weight.shape == attention.v_proj.weight.shape
    assert q_scale.shape == (attention.inner_dim,)
    assert k_scale.shape == (attention.inner_dim,)
    assert v_scale.shape == (attention.inner_dim,)
    assert not torch.equal(q_weight, original_q)
    assert not torch.equal(k_weight, original_k)
    assert not torch.equal(v_weight, original_v)

    # Check policy-specific ownership: ``FULL`` shares one QKV allocation, ``FUSE_KV``
    # shares only K/V, and NONE owns three independent allocations.
    q_storage = q_weight.untyped_storage().data_ptr()
    k_storage = k_weight.untyped_storage().data_ptr()
    v_storage = v_weight.untyped_storage().data_ptr()
    if qkv_fusion_option is QKVFusionOption.FULL:
        fused_qkv = attention._derived_weights.fused_qkv_weight
        fused_qkv_scale = attention._derived_weights.fused_qkv_weight_scale
        fused_kv = attention._derived_weights.fused_kv_weight
        fused_kv_scale = attention._derived_weights.fused_kv_weight_scale
        assert fused_qkv is not None and fused_qkv_scale is not None
        assert fused_kv is not None and fused_kv_scale is not None
        fused_storage = fused_qkv.untyped_storage().data_ptr()
        assert q_storage == fused_storage
        assert k_storage == fused_storage
        assert v_storage == fused_storage
        assert fused_kv.untyped_storage().data_ptr() == fused_storage
        assert fused_kv_scale.untyped_storage().data_ptr() == (
            fused_qkv_scale.untyped_storage().data_ptr()
        )
        assert fused_qkv_scale.shape == (3 * attention.inner_dim,)
        assert fused_kv.shape == (2 * attention.inner_dim, attention.context_dim)
        assert fused_kv_scale.shape == (2 * attention.inner_dim,)
        assert fused_kv.storage_offset() == attention.inner_dim * attention.context_dim
        assert fused_kv_scale.storage_offset() == attention.inner_dim
        assert torch.equal(fused_kv_scale, fused_qkv_scale[attention.inner_dim :])
    elif qkv_fusion_option is QKVFusionOption.FUSE_KV:
        fused_kv = attention._derived_weights.fused_kv_weight
        fused_kv_scale = attention._derived_weights.fused_kv_weight_scale
        assert fused_kv is not None and fused_kv_scale is not None
        fused_storage = fused_kv.untyped_storage().data_ptr()
        assert q_storage != fused_storage
        assert k_storage == fused_storage
        assert v_storage == fused_storage
        assert fused_kv_scale.shape == (2 * attention.inner_dim,)
        assert attention._derived_weights.fused_qkv_weight is None
        assert attention._derived_weights.fused_qkv_weight_scale is None
    else:
        assert len({q_storage, k_storage, v_storage}) == 3
        assert attention._derived_weights.fused_qkv_weight is None
        assert attention._derived_weights.fused_qkv_weight_scale is None
        assert attention._derived_weights.fused_kv_weight is None
        assert attention._derived_weights.fused_kv_weight_scale is None

    # Derived buffers must stay out of checkpoints, and ``Module.to`` must rebuild
    # E4M3 weights after casting the master parameters.
    assert not any(
        "fused" in key or "weight_fp8" in key or "weight_scale" in key for key in state
    )
    attention.to(dtype=torch.float64)
    assert attention.q_proj.weight.dtype is torch.float64
    assert attention._derived_weights.q_weight_fp8 is not None
    assert attention._derived_weights.k_weight_fp8 is not None
    assert attention._derived_weights.v_weight_fp8 is not None
    assert attention._derived_weights.q_weight_fp8.dtype is torch.float8_e4m3fn
    assert attention._derived_weights.k_weight_fp8.dtype is torch.float8_e4m3fn
    assert attention._derived_weights.v_weight_fp8.dtype is torch.float8_e4m3fn


@pytest.mark.ci_cpu
def test_fp8_rejects_unaligned_query_or_context_width() -> None:
    """Reject FP8 query or context widths that violate GEMM alignment."""
    with pytest.raises(ValueError, match="query_dim and context_dim"):
        _TritonMultiHeadAttention(
            255,
            context_dim=128,
            head_dim=64,
            attention_type=AttentionType.CROSS_ATTENTION,
            qkv_fusion_option=QKVFusionOption.FUSE_KV,
            use_fp8=True,
        )
    with pytest.raises(ValueError, match="query_dim and context_dim"):
        _TritonMultiHeadAttention(
            256,
            context_dim=127,
            head_dim=64,
            attention_type=AttentionType.CROSS_ATTENTION,
            qkv_fusion_option=QKVFusionOption.FUSE_KV,
            use_fp8=True,
        )


@pytest.mark.parametrize(
    ("scope", "weight_shape"),
    [
        pytest.param(QKNormScope.NONE, None, id="none"),
        pytest.param(QKNormScope.HEAD, (64,), id="head"),
        pytest.param(QKNormScope.INNER, (128,), id="inner"),
    ],
)
@pytest.mark.ci_cpu
def test_qk_norm_scope_controls_modules_and_epsilon(
    scope: QKNormScope,
    weight_shape: tuple[int, ...] | None,
) -> None:
    """Configure Q/K normalization modules at the requested feature scope.

    Args:
        scope: Disabled, per-head, or full-inner-width normalization policy.
        weight_shape: Expected RMSNorm affine shape; ``None`` expects identity.
    """
    attention = _TritonMultiHeadAttention(
        256,
        n_heads=2,
        head_dim=64,
        qk_norm_scope=scope,
        qk_norm_eps=3e-5,
    )

    # NONE installs identities; active scopes retain both affine geometry and the
    # public epsilon on query and key modules.
    assert attention.qk_norm_eps == 3e-5
    if weight_shape is None:
        assert isinstance(attention.q_norm, torch.nn.Identity)
        assert isinstance(attention.k_norm, torch.nn.Identity)
    else:
        assert isinstance(attention.q_norm, torch.nn.RMSNorm)
        assert isinstance(attention.k_norm, torch.nn.RMSNorm)
        assert attention.q_norm.weight.shape == weight_shape
        assert attention.k_norm.weight.shape == weight_shape
        assert attention.q_norm.eps == 3e-5
        assert attention.k_norm.eps == 3e-5


@pytest.mark.ci_cpu
def test_triton_sdpa_backend_constructor_policy() -> None:
    """Enforce the cuDNN default and enum-only SDPA backend selection."""
    assert _TritonMultiHeadAttention(128, head_dim=64).sdpa_backend is SDPABackend.CUDNN
    assert (
        _TritonMultiHeadAttention(
            128,
            head_dim=64,
            sdpa_backend=SDPABackend.TRITON,
        ).sdpa_backend
        is SDPABackend.TRITON
    )
    string_backend = cast(SDPABackend, "cudnn")
    with pytest.raises(TypeError, match="SDPABackend"):
        _TritonMultiHeadAttention(128, head_dim=64, sdpa_backend=string_backend)
