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

"""GPU correctness tests for accelerated Wan recipe components."""

from __future__ import annotations

import pytest
import torch

from flashdreams.accelerated.multi_head_attention_triton import (
    SDPABackend,
    TritonMultiHeadAttention,
)
from flashdreams.core.attention import RotaryPositionEmbedding3D
from flashdreams.recipes.wan.transformer.impl.modules import (
    AttentionBackend,
    Block,
    CrossAttention,
    SelfAttention,
    TritonCrossAttention,
)

pytestmark = pytest.mark.ci_gpu


@pytest.fixture(scope="module")
def tma_device() -> torch.device:
    """Return a CUDA device that supports tensor-memory acceleration."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required.")
    device = torch.device("cuda")
    if torch.cuda.get_device_capability(device)[0] < 9:
        pytest.skip("TMA attention requires compute capability 9.0 or newer.")
    return device


@pytest.mark.parametrize(
    "sdpa_backend", tuple(SDPABackend), ids=lambda backend: backend.value
)
def test_triton_self_attention_matches_default_through_window_roll(
    tma_device: torch.device,
    sdpa_backend: SDPABackend,
) -> None:
    """Compare Triton with Wan self-attention through cache fill and roll."""
    torch.manual_seed(7)
    default_block = Block(
        dim=256,
        ffn_dim=512,
        num_heads=2,
    )
    triton_block = Block(
        dim=256,
        ffn_dim=512,
        num_heads=2,
        attention_backend=AttentionBackend.TRITON,
        sdpa_backend=sdpa_backend,
    )
    triton_block.load_state_dict(default_block.state_dict(), strict=True)

    assert default_block.attention_backend is AttentionBackend.WAN
    reference = default_block.self_attn.to(
        device=tma_device, dtype=torch.bfloat16
    ).eval()
    actual_attention = triton_block.self_attn.to(
        device=tma_device, dtype=torch.bfloat16
    ).eval()
    assert isinstance(reference, SelfAttention)
    assert isinstance(actual_attention, TritonMultiHeadAttention)
    assert actual_attention.use_fp8 is True
    assert actual_attention.sdpa_backend is sdpa_backend

    batch_size = 1
    len_t, len_h, len_w = 1, 1, 16
    chunk_size = len_t * len_h * len_w
    window_size = 2 * chunk_size
    reference_cache = reference.allocate_kv_cache(
        batch_size=batch_size,
        chunk_size=chunk_size,
        window_size=window_size,
        sink_size=0,
        device=tma_device,
        dtype=torch.bfloat16,
    )
    actual_cache = actual_attention.allocate_kv_cache(
        batch_size=batch_size,
        chunk_size=chunk_size,
        window_size=window_size,
        sink_size=0,
        device=tma_device,
        dtype=torch.bfloat16,
    )
    assert actual_cache.dtype is (
        torch.float8_e4m3fn if sdpa_backend is SDPABackend.TRITON else torch.bfloat16
    )
    rope = RotaryPositionEmbedding3D(
        head_dim=128,
        len_t=len_t,
        len_h=len_h,
        len_w=len_w,
        interleaved=True,
        device=tma_device,
    )
    generator = torch.Generator(device=tma_device).manual_seed(11)

    with torch.inference_mode():
        for chunk_idx in range(3):
            x = torch.randn(
                batch_size,
                chunk_size,
                256,
                generator=generator,
                device=tma_device,
                dtype=torch.bfloat16,
            )
            rope_freqs = rope.shift_t(chunk_idx)
            reference_cache.before_update(chunk_idx)
            actual_cache.before_update(chunk_idx)

            expected = reference(x, reference_cache, rope_freqs)
            actual = actual_attention(x, actual_cache, rope_freqs)

            torch.testing.assert_close(actual, expected, atol=5e-2, rtol=5e-2)
            torch.testing.assert_close(
                actual_cache.cached_k().to(torch.bfloat16),
                reference_cache.cached_k(),
                atol=1.5e-1,
                rtol=1.25e-1,
            )
            torch.testing.assert_close(
                actual_cache.cached_v().to(torch.bfloat16),
                reference_cache.cached_v(),
                atol=1.5e-1,
                rtol=1.25e-1,
            )

            reference_cache.after_update(chunk_idx)
            actual_cache.after_update(chunk_idx)


def test_triton_cross_attention_matches_default_static_text_cache(
    tma_device: torch.device,
) -> None:
    """Match Wan T2V cross-attention while preserving static text caches.

    Args:
        tma_device: CUDA device with tensor-memory acceleration support.
    """
    torch.manual_seed(19)
    reference = (
        CrossAttention(
            query_dim=256,
            n_heads=2,
            head_dim=128,
        )
        .to(
            device=tma_device,
            dtype=torch.bfloat16,
        )
        .eval()
    )
    actual_attention = (
        TritonCrossAttention(
            query_dim=256,
            n_heads=2,
            head_dim=128,
        )
        .to(
            device=tma_device,
            dtype=torch.bfloat16,
        )
        .eval()
    )
    actual_attention.load_state_dict(reference.state_dict(), strict=True)

    generator = torch.Generator(device=tma_device).manual_seed(23)
    context = torch.randn(
        (1, 32, 256),
        generator=generator,
        device=tma_device,
        dtype=torch.bfloat16,
    )
    query = torch.randn(
        (1, 16, 256),
        generator=generator,
        device=tma_device,
        dtype=torch.bfloat16,
    )

    with torch.inference_mode():
        reference_cache = reference.initialize_cache(context)
        actual_cache = actual_attention.initialize_cache(context)
        reference_key = reference_cache.text.cached_k().clone()
        reference_value = reference_cache.text.cached_v().clone()
        actual_key = actual_cache.text.cached_k().clone()
        actual_value = actual_cache.text.cached_v().clone()

        expected = reference(query, reference_cache)
        actual = actual_attention(query, actual_cache)

    torch.testing.assert_close(actual, expected, atol=5e-2, rtol=5e-2)
    assert torch.equal(reference_cache.text.cached_k(), reference_key)
    assert torch.equal(reference_cache.text.cached_v(), reference_value)
    assert torch.equal(actual_cache.text.cached_k(), actual_key)
    assert torch.equal(actual_cache.text.cached_v(), actual_value)
    assert actual_cache.text.dtype is torch.float8_e4m3fn
