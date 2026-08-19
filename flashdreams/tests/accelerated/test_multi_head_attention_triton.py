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

"""Numerical correctness tests for Triton multi-head attention."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from torch import Tensor

from flashdreams.accelerated.multi_head_attention import (
    AttentionType,
    QKNormScope,
)
from flashdreams.accelerated.multi_head_attention_triton import (
    QKVFusionOption,
    SDPABackend,
    TritonMultiHeadAttention,
)

pytestmark = pytest.mark.ci_gpu

_QUERY_DIM = 128
_N_HEADS = 2
_HEAD_DIM = _QUERY_DIM // _N_HEADS
_CHUNK_SIZE = 16
_WINDOW_SIZE = 32
_SINK_SIZE = 4


class _TritonMultiHeadAttention(TritonMultiHeadAttention):
    """Checkpoint-compatible Triton attention used by correctness tests."""

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

    def __init__(
        self,
        *,
        attention_type: AttentionType,
        qk_norm_scope: QKNormScope,
        rope_interleaved: bool,
        sdpa_backend: SDPABackend,
        qkv_fusion_option: QKVFusionOption,
        use_fp8: bool,
        bias: bool,
    ) -> None:
        """Initialize one correctness-test configuration."""
        super().__init__(
            query_dim=_QUERY_DIM,
            n_heads=_N_HEADS,
            head_dim=_HEAD_DIM,
            attention_type=attention_type,
            qkv_fusion_option=qkv_fusion_option,
            qk_norm_scope=qk_norm_scope,
            rope_interleaved=rope_interleaved,
            use_fp8=use_fp8,
            sdpa_backend=sdpa_backend,
        )
        self.q_proj = torch.nn.Linear(_QUERY_DIM, self.inner_dim, bias=bias)
        self.k_proj = torch.nn.Linear(_QUERY_DIM, self.inner_dim, bias=bias)
        self.v_proj = torch.nn.Linear(_QUERY_DIM, self.inner_dim, bias=bias)
        self.output_proj = torch.nn.Linear(self.inner_dim, _QUERY_DIM, bias=bias)
        if qk_norm_scope is QKNormScope.NONE:
            self.q_norm = torch.nn.Identity()
            self.k_norm = torch.nn.Identity()
        else:
            norm_dim = (
                self.head_dim if qk_norm_scope is QKNormScope.HEAD else self.inner_dim
            )
            self.q_norm = torch.nn.RMSNorm(norm_dim, eps=self.qk_norm_eps)
            self.k_norm = torch.nn.RMSNorm(norm_dim, eps=self.qk_norm_eps)
        self._initialize_derived_weights()


@pytest.fixture(scope="module")
def tma_device() -> torch.device:
    """Return a CUDA device with tensor-memory acceleration."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required.")
    device = torch.device("cuda")
    if torch.cuda.get_device_capability(device)[0] < 9:
        pytest.skip("TMA attention requires compute capability 9.0 or newer.")
    return device


def _normalize(
    x: Tensor,
    norm: torch.nn.Module,
    scope: QKNormScope,
) -> Tensor:
    """Apply the configured RMS normalization with native PyTorch math."""
    if scope is QKNormScope.NONE:
        return x
    assert isinstance(norm, torch.nn.RMSNorm)
    original_shape = x.shape
    if scope is QKNormScope.INNER:
        x = x.flatten(-2)
    return F.rms_norm(x, norm.normalized_shape, norm.weight, norm.eps).reshape(
        original_shape
    )


def _project_query(attention: _TritonMultiHeadAttention, x: Tensor) -> Tensor:
    """Project queries with independent PyTorch operators."""
    query = F.linear(x, attention.q_proj.weight, attention.q_proj.bias).reshape(
        -1, x.shape[-2], attention.n_heads, attention.head_dim
    )
    return _normalize(query, attention.q_norm, attention.qk_norm_scope)


def _project_kv(
    attention: _TritonMultiHeadAttention,
    x: Tensor,
) -> tuple[Tensor, Tensor]:
    """Project keys and values with independent PyTorch operators."""
    head_shape = (-1, x.shape[-2], attention.n_heads, attention.head_dim)
    key = F.linear(x, attention.k_proj.weight, attention.k_proj.bias).reshape(
        head_shape
    )
    value = F.linear(x, attention.v_proj.weight, attention.v_proj.bias).reshape(
        head_shape
    )
    return _normalize(key, attention.k_norm, attention.qk_norm_scope), value


def _apply_rope(x: Tensor, rope_freqs: Tensor, interleaved: bool) -> Tensor:
    """Apply rotary embeddings with independent PyTorch operators."""
    freqs = rope_freqs[:, 0, 0, :].reshape(1, x.shape[-3], 1, x.shape[-1])
    if interleaved:
        rotated = torch.stack((-x[..., 1::2], x[..., 0::2]), dim=-1).flatten(-2)
    else:
        first, second = x.chunk(2, dim=-1)
        rotated = torch.cat((-second, first), dim=-1)
    return x * freqs.cos().to(x.dtype) + rotated * freqs.sin().to(x.dtype)


def _reference_output(
    attention: _TritonMultiHeadAttention,
    query: Tensor,
    key: Tensor,
    value: Tensor,
    output_shape: torch.Size,
) -> Tensor:
    """Compute non-causal MHA with PyTorch's math SDPA backend."""
    with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
        output = F.scaled_dot_product_attention(
            query.transpose(1, 2),
            key.transpose(1, 2),
            value.transpose(1, 2),
            dropout_p=0.0,
            is_causal=False,
        ).transpose(1, 2)
    output = F.linear(
        output.flatten(-2),
        attention.output_proj.weight,
        attention.output_proj.bias,
    )
    return output.reshape(output_shape)


def _rope_freqs(
    length: int,
    generator: torch.Generator,
    device: torch.device,
) -> Tensor:
    """Generate deterministic rotary angles for one token sequence."""
    return torch.randn(
        length,
        1,
        1,
        _HEAD_DIM,
        generator=generator,
        device=device,
    )


def _assert_close(actual: Tensor, expected: Tensor, use_fp8: bool) -> None:
    """Compare native or FP8 attention at its numerical error bound."""
    tolerance = 8e-2 if use_fp8 else 2e-2
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)


def _check_self_attention(
    attention: _TritonMultiHeadAttention,
    generator: torch.Generator,
    device: torch.device,
    dtype: torch.dtype,
    use_rope: bool,
) -> None:
    """Check streaming self-attention through cache fill and rolling."""
    cache = attention.allocate_kv_cache(
        batch_size=1,
        chunk_size=_CHUNK_SIZE,
        window_size=_WINDOW_SIZE,
        sink_size=_SINK_SIZE,
        device=device,
        dtype=dtype,
    )
    all_keys: list[Tensor] = []
    all_values: list[Tensor] = []

    for chunk_idx in range(3):
        x = torch.randn(
            1,
            _CHUNK_SIZE,
            _QUERY_DIM,
            generator=generator,
            device=device,
            dtype=dtype,
        )
        rope = _rope_freqs(_CHUNK_SIZE, generator, device) if use_rope else None
        query = _project_query(attention, x)
        key, value = _project_kv(attention, x)
        if rope is not None:
            query = _apply_rope(query, rope, attention.rope_interleaved)
            key = _apply_rope(key, rope, attention.rope_interleaved)
        all_keys.append(key)
        all_values.append(value)

        visible_key = torch.cat(all_keys, dim=1)
        visible_value = torch.cat(all_values, dim=1)
        if visible_key.shape[1] > _SINK_SIZE + _WINDOW_SIZE:
            visible_key = torch.cat(
                (visible_key[:, :_SINK_SIZE], visible_key[:, -_WINDOW_SIZE:]),
                dim=1,
            )
            visible_value = torch.cat(
                (visible_value[:, :_SINK_SIZE], visible_value[:, -_WINDOW_SIZE:]),
                dim=1,
            )
        expected = _reference_output(
            attention, query, visible_key, visible_value, x.shape
        )

        cache.before_update(chunk_idx)
        actual = attention(x, cache, rope)
        cache.after_update(chunk_idx)
        _assert_close(actual, expected, attention.use_fp8)


def _check_cross_attention(
    attention: _TritonMultiHeadAttention,
    generator: torch.Generator,
    device: torch.device,
    dtype: torch.dtype,
    use_rope: bool,
) -> None:
    """Check static cross-attention against independent PyTorch math."""
    context = torch.randn(
        1,
        24,
        _QUERY_DIM,
        generator=generator,
        device=device,
        dtype=dtype,
    )
    query_tokens = torch.randn(
        1,
        8,
        _QUERY_DIM,
        generator=generator,
        device=device,
        dtype=dtype,
    )
    context_rope = _rope_freqs(24, generator, device) if use_rope else None
    query_rope = _rope_freqs(8, generator, device) if use_rope else None

    query = _project_query(attention, query_tokens)
    key, value = _project_kv(attention, context)
    if query_rope is not None and context_rope is not None:
        query = _apply_rope(query, query_rope, attention.rope_interleaved)
        key = _apply_rope(key, context_rope, attention.rope_interleaved)
    expected = _reference_output(attention, query, key, value, query_tokens.shape)

    cache = attention.compute_kv(context, context_rope)
    actual = attention(query_tokens, cache, query_rope)
    _assert_close(actual, expected, attention.use_fp8)


@pytest.mark.parametrize(
    "attention_type", tuple(AttentionType), ids=lambda value: value.value
)
@pytest.mark.parametrize(
    "qk_norm_scope", tuple(QKNormScope), ids=lambda value: value.value
)
@pytest.mark.parametrize(
    "rope_interleaved", [False, True], ids=["rope-split", "rope-interleaved"]
)
@pytest.mark.parametrize(
    "sdpa_backend", tuple(SDPABackend), ids=lambda value: value.value
)
@pytest.mark.parametrize(
    "qkv_fusion_option", tuple(QKVFusionOption), ids=lambda value: value.value
)
@pytest.mark.parametrize("use_fp8", [False, True], ids=["native", "fp8"])
@torch.inference_mode()
def test_multi_head_attention_matches_pytorch(
    tma_device: torch.device,
    attention_type: AttentionType,
    qk_norm_scope: QKNormScope,
    rope_interleaved: bool,
    sdpa_backend: SDPABackend,
    qkv_fusion_option: QKVFusionOption,
    use_fp8: bool,
) -> None:
    """Match every supported MHA policy combination with PyTorch math."""
    dtype = torch.float16 if rope_interleaved else torch.bfloat16
    torch.manual_seed(7)
    attention = _TritonMultiHeadAttention(
        attention_type=attention_type,
        qk_norm_scope=qk_norm_scope,
        rope_interleaved=rope_interleaved,
        sdpa_backend=sdpa_backend,
        qkv_fusion_option=qkv_fusion_option,
        use_fp8=use_fp8,
        bias=rope_interleaved,
    ).to(device=tma_device, dtype=dtype)
    attention.eval()

    generator = torch.Generator(device=tma_device).manual_seed(11)
    use_rope = qk_norm_scope is not QKNormScope.NONE
    if attention_type is AttentionType.SELF_ATTENTION:
        _check_self_attention(attention, generator, tma_device, dtype, use_rope)
    else:
        _check_cross_attention(attention, generator, tma_device, dtype, use_rope)
