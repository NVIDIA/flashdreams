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

"""CPU behavior and GPU recipe parity tests for Torch multi-head attention."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from flashdreams.accelerated.multi_head_attention import (
    AttentionType,
    QKNormScope,
)
from flashdreams.accelerated.multi_head_attention_torch import TorchMultiHeadAttention
from flashdreams.core.attention import BlockKVCache, RotaryPositionEmbedding3D
from flashdreams.recipes.cosmos.transformer.impl import modules as cosmos_modules
from flashdreams.recipes.wan.transformer.impl import modules as wan_modules


class _TorchMultiHeadAttention(TorchMultiHeadAttention):
    """Canonical Torch backend for behavior and parity tests.

    Generic projection and normalization names let tests load recipe parameters
    while retaining the shared accelerated-attention interface.
    """

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
        qk_norm_eps: float = 1e-6,
        qk_norm_scope: QKNormScope = QKNormScope.HEAD,
        rope_interleaved: bool = False,
    ) -> None:
        """Initialize canonical projections and normalization modules.

        Args:
            query_dim: Feature width ``Q`` of query and output tokens.
            n_heads: Number of attention heads ``H``.
            head_dim: Feature width ``D`` of each head.
            context_dim: Feature width ``C`` projected into keys and values;
                ``None`` uses ``query_dim``.
            attention_type: Select self-attention or static cross-attention.
            qkv_bias: Add bias to the query, key, and value projections.
            output_bias: Add bias to the output projection.
            qk_norm_eps: Epsilon shared by query and key RMSNorm modules.
            qk_norm_scope: Normalize each head, the full inner width, or neither.
            rope_interleaved: Pair adjacent RoPE features when ``True``; otherwise
                pair features across the two half-width blocks.
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
        # Register generic checkpoint fields after shared geometry is available;
        # model adapters can expose different names through logical properties.
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


@pytest.mark.ci_cpu
def test_qk_norm_constructor_policy() -> None:
    """Configure shared RMSNorm epsilon and the normalization-off policy."""

    # Compare active per-head normalization with the authoritative ``NONE``
    # policy used to disable both query and key normalization.
    normalized = _TorchMultiHeadAttention(
        query_dim=16,
        n_heads=2,
        head_dim=8,
        qk_norm_eps=1e-4,
    )
    disabled = _TorchMultiHeadAttention(
        query_dim=16,
        n_heads=2,
        head_dim=8,
        qk_norm_scope=QKNormScope.NONE,
    )

    # Both sides of Q/K normalization must share module type and epsilon.
    assert isinstance(normalized.q_norm, torch.nn.RMSNorm)
    assert isinstance(normalized.k_norm, torch.nn.RMSNorm)
    assert normalized.q_norm.eps == 1e-4
    assert normalized.k_norm.eps == 1e-4
    assert isinstance(disabled.q_norm, torch.nn.Identity)
    assert isinstance(disabled.k_norm, torch.nn.Identity)


@pytest.mark.ci_cpu
def test_torch_attention_runs_complete_streaming_forward_on_cpu() -> None:
    """Run one complete streaming self-attention update on CPU.

    The ``[B, L, Q]`` output must preserve input shape while the prepared cache
    exposes the current ``[B, L, H, D]`` keys and values.
    """
    # Use a compact ``[B=2, L=2, Q=24]`` input that still exercises multiple
    # heads and the complete reference attention path.
    batch_size = 2
    chunk_size = 2
    n_heads = 2
    head_dim = 12
    query_dim = n_heads * head_dim
    x = torch.randn(batch_size, chunk_size, query_dim)

    # Match cache and 3D RoPE geometry to the single input chunk.
    attention = _TorchMultiHeadAttention(
        query_dim=query_dim,
        n_heads=n_heads,
        head_dim=head_dim,
    )
    kv_cache = attention.allocate_kv_cache(
        batch_size=batch_size,
        chunk_size=chunk_size,
        window_size=chunk_size,
        sink_size=0,
        device=torch.device("cpu"),
        dtype=x.dtype,
    )
    rope = RotaryPositionEmbedding3D(
        head_dim=head_dim,
        len_t=1,
        len_h=1,
        len_w=chunk_size,
        device=torch.device("cpu"),
    )

    # Open the cache write phase before ``forward`` appends the current K/V.
    kv_cache.before_update(0)
    output = attention(x, kv_cache, rope.shift_t(0))

    # Validate both the public output contract and the visible cache layout.
    assert output.shape == x.shape
    assert torch.isfinite(output).all()
    assert kv_cache.cached_k().shape == (
        batch_size,
        chunk_size,
        n_heads,
        head_dim,
    )
    assert kv_cache.cached_v().shape == (
        batch_size,
        chunk_size,
        n_heads,
        head_dim,
    )
    # Commit cache bookkeeping only after inspecting the state consumed by SDPA.
    kv_cache.after_update(0)


@pytest.mark.ci_cpu
def test_torch_cross_attention_matches_manual_sdpa_and_reuses_static_cache() -> None:
    """Match asymmetric cross-attention to manual SDPA without cache mutation.

    Query and context leading dimensions flatten to the same cache batch ``B=6``
    despite their different public layouts and feature widths.
    """
    # Arrange query and context leading dimensions so both flatten to ``B=6``
    # while retaining distinct sequence and feature widths.
    query_dim, context_dim, n_heads, head_dim = 10, 6, 2, 4
    query_length, context_length = 4, 7
    attention = _TorchMultiHeadAttention(
        query_dim=query_dim,
        context_dim=context_dim,
        attention_type=AttentionType.CROSS_ATTENTION,
        n_heads=n_heads,
        head_dim=head_dim,
        qk_norm_scope=QKNormScope.NONE,
    )
    query = torch.randn(2, 3, query_length, query_dim)
    context = torch.randn(3, 2, context_length, context_dim)
    query_rope = torch.randn(query_length, 1, 1, head_dim)
    context_rope = torch.randn(context_length, 1, 1, head_dim)

    # Precompute static K/V once, then verify projection, reshape, and key RoPE
    # before exercising the query-only forward branch.
    cache = attention.compute_kv(context, context_rope)
    expected_key = attention._apply_rope(
        attention.k_proj(context).reshape(-1, context_length, n_heads, head_dim),
        context_rope,
    )
    expected_value = attention.v_proj(context).reshape(
        -1, context_length, n_heads, head_dim
    )
    torch.testing.assert_close(cache.cached_k(), expected_key)
    torch.testing.assert_close(cache.cached_v(), expected_value)

    # Snapshot tensors and lifecycle fields around two reads of the same static
    # context; cross-attention must not enter a cache update phase.
    cache_key = cache._k.clone()
    cache_value = cache._v.clone()
    cache_state = (cache._prev_chunk_idx, cache._curr_chunk_idx, cache._n_cached)
    output = attention(query, cache, query_rope)
    repeated_output = attention(query, cache, query_rope)

    # Reproduce the reference path explicitly in head-major SDPA layout, then
    # restore the original ``[2, 3, L, Q]`` public query geometry.
    expected_query = attention._apply_rope(
        attention.q_proj(query).reshape(-1, query_length, n_heads, head_dim),
        query_rope,
    )
    expected = F.scaled_dot_product_attention(
        expected_query.transpose(1, 2),
        expected_key.transpose(1, 2),
        expected_value.transpose(1, 2),
        dropout_p=0.0,
        is_causal=False,
    )
    expected = attention.output_proj(expected.transpose(1, 2).flatten(-2))
    expected = expected.reshape(2, 3, query_length, query_dim)

    # Check numerical parity and deterministic static reuse while exact cache
    # contents and lifecycle state remain unchanged across both forward calls.
    assert output.shape == query.shape
    torch.testing.assert_close(output, expected)
    torch.testing.assert_close(repeated_output, expected)
    assert torch.equal(cache._k, cache_key)
    assert torch.equal(cache._v, cache_value)
    assert cache_state == (
        cache._prev_chunk_idx,
        cache._curr_chunk_idx,
        cache._n_cached,
    )


@pytest.mark.ci_cpu
def test_torch_self_forward_matches_private_stages_with_rope() -> None:
    """Match self-attention ``forward`` to its private update and query stages.

    Independent caches ensure the public path and explicit private composition
    write identical rotated K/V rather than sharing mutation side effects.
    """
    # Extra leading query dimensions flatten to cache batch ``B=4``; both paths
    # receive identical ``[L, 1, 1, D]`` RoPE data.
    query_dim, n_heads, head_dim, chunk_size = 10, 2, 4, 3
    attention = _TorchMultiHeadAttention(
        query_dim=query_dim,
        n_heads=n_heads,
        head_dim=head_dim,
    )
    x = torch.randn(2, 2, chunk_size, query_dim)
    rope_freqs = torch.randn(chunk_size, 1, 1, head_dim)
    # Allocate independent caches so the public and private-stage executions
    # cannot observe one another's K/V writes.
    caches = [
        attention.allocate_kv_cache(
            batch_size=4,
            chunk_size=chunk_size,
            window_size=chunk_size,
            sink_size=0,
            device=x.device,
            dtype=x.dtype,
        )
        for _ in range(2)
    ]
    # Prepare the same first-chunk write interval for both executions.
    for cache in caches:
        cache.before_update(0)

    # Compare the public runtime entry point with the implementation's explicit
    # update-then-query composition.
    actual = attention(x, caches[0], rope_freqs)
    updated_cache = attention._update_kv(x, caches[1], rope_freqs)
    expected = attention._query_kv(x, updated_cache, rope_freqs)

    # Output and both cache planes must agree before lifecycle bookkeeping closes.
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(caches[0].cached_k(), caches[1].cached_k())
    torch.testing.assert_close(caches[0].cached_v(), caches[1].cached_v())
    for cache in caches:
        cache.after_update(0)


@pytest.mark.ci_cpu
def test_torch_attention_validates_boundaries() -> None:
    """Reject invalid token widths, flattened batches, and cache lifecycle use."""
    # Use asymmetric cross-attention to validate context width independently from
    # query width and flattened cache batch.
    attention = _TorchMultiHeadAttention(
        query_dim=10,
        context_dim=6,
        attention_type=AttentionType.CROSS_ATTENTION,
        n_heads=2,
        head_dim=4,
    )
    context = torch.randn(2, 5, 6)

    # Reject malformed context before allocating a reusable static cache.
    with pytest.raises(ValueError, match="context feature width"):
        attention.compute_kv(torch.randn(2, 5, 7))

    # Query validation covers both the feature axis and the product of leading
    # dimensions mapped onto cache batch ``B``.
    cache = attention.compute_kv(context)
    with pytest.raises(ValueError, match="query feature width"):
        attention(torch.randn(2, 3, 11), cache)
    with pytest.raises(ValueError, match="cache batch"):
        attention(torch.randn(3, 3, 10), cache)

    # A rolling self-attention cache requires ``before_update`` even when its
    # tensor geometry and dtype otherwise match the input.
    self_attention = _TorchMultiHeadAttention(
        query_dim=6,
        n_heads=2,
        head_dim=4,
    )
    rolling_cache = self_attention.allocate_kv_cache(
        batch_size=2,
        chunk_size=5,
        window_size=5,
        sink_size=0,
        device=context.device,
        dtype=context.dtype,
    )
    with pytest.raises(RuntimeError, match=r"before_update\(\)"):
        self_attention(context, rolling_cache)


@pytest.fixture(scope="module")
def cuda_device() -> torch.device:
    """Return the CUDA device used by recipe parity tests.

    Returns:
        CUDA device available to fused RoPE and attention implementations.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA required.")
    return torch.device("cuda")


def _make_cache(
    batch_size: int,
    chunk_size: int,
    n_heads: int,
    head_dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> BlockKVCache:
    """Build a two-chunk rolling cache for parity tests.

    Args:
        batch_size: Flattened cache batch ``B``.
        chunk_size: Tokens ``L`` written by each update.
        n_heads: Number of cached attention heads ``H``.
        head_dim: Feature width ``D`` of each cached head.
        device: Device holding cache storage.
        dtype: Data type of cached keys and values.

    Returns:
        Empty ``[B, 2 * L, H, D]`` cache that rolls on its third update.
    """
    window_size = 2 * chunk_size
    return BlockKVCache(
        k_shape=(batch_size, window_size, n_heads, head_dim),
        v_shape=(batch_size, window_size, n_heads, head_dim),
        seq_dim=1,
        chunk_size=chunk_size,
        window_size=window_size,
        device=device,
        dtype=dtype,
    )


def _assert_streaming_parity(
    recipe_attention: cosmos_modules.MultiHeadAttention
    | wan_modules.MultiHeadAttention,
    torch_attention: TorchMultiHeadAttention,
    *,
    rope_interleaved: bool,
    device: torch.device,
) -> None:
    """Compare recipe and reference attention through cache fill and roll.

    Args:
        recipe_attention: Model-specific Cosmos or Wan attention module.
        torch_attention: Generic Torch implementation with equivalent parameters.
        rope_interleaved: Use the recipe's RoPE feature-pairing convention.
        device: CUDA device used by modules, inputs, RoPE, and caches.
    """
    # BF16 exercises the recipes' production cuDNN attention and fused RoPE paths
    # against the Torch reference while keeping this parity test small.
    dtype = torch.bfloat16
    recipe_attention.to(device=device, dtype=dtype).eval()
    torch_attention.to(device=device, dtype=dtype).eval()

    # A ``1 x 8 x 8`` patch grid produces one 64-token chunk. The two-chunk
    # cache fills on steps 0 and 1, then rolls its local window on step 2.
    batch_size = 1
    len_t, len_h, len_w = 1, 8, 8
    chunk_size = len_t * len_h * len_w

    # Each implementation mutates its own cache so neither can mask an
    # incorrect update in the other implementation.
    recipe_cache = _make_cache(
        batch_size,
        chunk_size,
        torch_attention.n_heads,
        torch_attention.head_dim,
        device,
        dtype,
    )
    torch_cache = _make_cache(
        batch_size,
        chunk_size,
        torch_attention.n_heads,
        torch_attention.head_dim,
        device,
        dtype,
    )

    # Generate the same model-shaped 3D frequencies used by production. The
    # pairing convention is selected by the recipe-specific test case.
    rope = RotaryPositionEmbedding3D(
        head_dim=torch_attention.head_dim,
        len_t=len_t,
        len_h=len_h,
        len_w=len_w,
        interleaved=rope_interleaved,
        device=device,
    )
    generator = torch.Generator(device=device).manual_seed(1234)

    with torch.inference_mode():
        for chunk_idx in range(3):
            # Reuse identical tokens and shifted positions for both paths.
            x = torch.randn(
                batch_size,
                chunk_size,
                torch_attention.query_dim,
                generator=generator,
                device=device,
                dtype=dtype,
            )
            rope_freqs = rope.shift_t(chunk_idx)

            # Select the current write region, rolling the window first on the
            # third chunk, before either forward mutates cache storage.
            recipe_cache.before_update(chunk_idx)
            torch_cache.before_update(chunk_idx)

            expected = recipe_attention(x, recipe_cache, rope_freqs)
            actual = torch_attention(x, torch_cache, rope_freqs)

            # Output parity covers Q projection, attention, and output
            # projection. K/V parity localizes failures to their projections,
            # normalization, RoPE, or cache updates; the wider K tolerance covers
            # BF16 fused-RoPE rounding.
            torch.testing.assert_close(actual, expected, atol=5e-3, rtol=5e-3)
            torch.testing.assert_close(
                torch_cache.cached_k(),
                recipe_cache.cached_k(),
                atol=2e-2,
                rtol=2e-2,
            )
            torch.testing.assert_close(
                torch_cache.cached_v(),
                recipe_cache.cached_v(),
                atol=0,
                rtol=0,
            )

            # Commit bookkeeping only after comparing the visible state that
            # was consumed by this chunk's attention call.
            recipe_cache.after_update(chunk_idx)
            torch_cache.after_update(chunk_idx)


@pytest.mark.ci_gpu
def test_torch_multi_head_attention_matches_cosmos(
    cuda_device: torch.device,
) -> None:
    """Match the Torch reference to Cosmos streaming self-attention.

    Cosmos uses bias-free projections, per-head Q/K RMSNorm, and half-split
    RoPE pairs.

    Args:
        cuda_device: CUDA device used for the three-chunk parity rollout.
    """
    torch.manual_seed(7)

    # Configure the generic implementation with the exact Cosmos attention
    # geometry and projection policies.
    cosmos_attention = cosmos_modules.MultiHeadAttention(
        query_dim=256,
        n_heads=2,
        head_dim=128,
    )
    torch_attention = _TorchMultiHeadAttention(
        query_dim=256,
        n_heads=2,
        head_dim=128,
        qkv_bias=False,
        output_bias=False,
        qk_norm_scope=QKNormScope.HEAD,
        rope_interleaved=False,
    )

    # Cosmos and the Torch reference share parameter names, so a strict state
    # dict load also verifies that their trainable structures align.
    torch_attention.load_state_dict(cosmos_attention.state_dict())

    # Exercise cache filling and rolling with real half-split FlashDreams RoPE.
    _assert_streaming_parity(
        cosmos_attention,
        torch_attention,
        rope_interleaved=False,
        device=cuda_device,
    )


@pytest.mark.ci_gpu
def test_torch_multi_head_attention_matches_wan(
    cuda_device: torch.device,
) -> None:
    """Match the Torch reference to Wan streaming self-attention.

    Wan uses biased projections, inner-width Q/K RMSNorm, and interleaved RoPE
    pairs.

    Args:
        cuda_device: CUDA device used for the three-chunk parity rollout.
    """
    torch.manual_seed(8)

    # Configure the generic implementation with Wan's attention geometry and
    # its model-specific normalization and RoPE policies.
    wan_attention = wan_modules.MultiHeadAttention(
        query_dim=256,
        n_heads=2,
        head_dim=128,
    )
    torch_attention = _TorchMultiHeadAttention(
        query_dim=256,
        n_heads=2,
        head_dim=128,
        qkv_bias=True,
        output_bias=True,
        qk_norm_scope=QKNormScope.INNER,
        rope_interleaved=True,
    )

    # Wan uses q/k/v/o and norm_q/norm_k names, so copy each equivalent module
    # explicitly before comparing the forward paths.
    torch_attention.q_proj.load_state_dict(wan_attention.q.state_dict())
    torch_attention.k_proj.load_state_dict(wan_attention.k.state_dict())
    torch_attention.v_proj.load_state_dict(wan_attention.v.state_dict())
    torch_attention.output_proj.load_state_dict(wan_attention.o.state_dict())
    torch_attention.q_norm.load_state_dict(wan_attention.norm_q.state_dict())
    torch_attention.k_norm.load_state_dict(wan_attention.norm_k.state_dict())

    # Exercise cache filling and rolling with real interleaved FlashDreams RoPE.
    _assert_streaming_parity(
        wan_attention,
        torch_attention,
        rope_interleaved=True,
        device=cuda_device,
    )
