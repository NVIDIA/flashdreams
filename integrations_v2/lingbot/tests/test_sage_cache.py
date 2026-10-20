# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cache lifecycle checks for channel-major Sage values."""

import pytest
import torch

from flashdreams.core.attention.kvcache import BlockKVCache
from integrations_v2.lingbot.impl.transformer.impl.modules import CamCtrlBlock

pytestmark = pytest.mark.ci_cpu


@pytest.mark.parametrize("use_tma", [True, False])
@pytest.mark.parametrize(
    "chunk_size,window_size,sink_size", [(2, 6, 0), (2, 6, 3), (4, 3, 5)]
)
def test_sage_cache_layout_and_rolling(
    use_tma: bool, chunk_size: int, window_size: int, sink_size: int
) -> None:
    """Preserve sink tokens, repeated writes and storage addresses on reset."""
    block = CamCtrlBlock(
        dim=128,
        ffn_dim=128,
        num_heads=1,
        self_attention_backend="sage",
        self_attention_use_tma=use_tma,
    )
    context = torch.randn(2, 3, 128)
    block_cache = block.initialize_cache(chunk_size, window_size, sink_size, context)
    cache = block_cache.self_attn
    reference = BlockKVCache(
        k_shape=cache.k_shape,
        v_shape=cache.v_shape,
        seq_dim=cache.seq_dim,
        chunk_size=chunk_size,
        window_size=window_size,
        sink_size=sink_size,
        device="cpu",
        dtype=cache.dtype,
    )
    assert (cache._v.stride(1) == 1) == use_tma
    addresses = (cache._k.data_ptr(), cache._v.data_ptr())
    for _ in range(2):
        for index in range(8):
            previous_sum = None
            for repeat in range(2):
                values = torch.full(
                    (2, chunk_size, 1, 128), index * 2 + repeat, dtype=cache.dtype
                )
                block_cache.before_update(index)
                reference.before_update(index)
                history_sum = block_cache.key_history_sum
                for current in (cache, reference):
                    current.update(values + 1, values)
                if window_size >= chunk_size:
                    assert history_sum is not None
                    split_mean = (
                        history_sum
                        + cache.cached_k()[:, block_cache.key_history_length :].sum(
                            dim=1, keepdim=True, dtype=torch.float32
                        )
                    ) / cache.size
                    torch.testing.assert_close(
                        split_mean, cache.cached_k().mean(dim=1, keepdim=True)
                    )
                else:
                    assert history_sum is None
                if repeat:
                    assert history_sum is previous_sum
                previous_sum = history_sum
                torch.testing.assert_close(cache.cached_k(), reference.cached_k())
                torch.testing.assert_close(cache.cached_v(), reference.cached_v())
                for current in (cache, reference):
                    current.after_update(index)
        cache.reset()
        reference.reset()
        assert cache.size == 0
        assert (cache._k.data_ptr(), cache._v.data_ptr()) == addresses
