# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn.functional as F
from omnidreams.transformer.impl.modules import BlockCache
from omnidreams.transformer.impl.network import CosmosDiTNetworkCache
from omnidreams_singleview.python.optimized_dit import _CosmosNetworkShapeOps

from flashdreams.core.attention import BlockKVCache


def test_shape_ops_rebuilds_cross_attention_cache_from_context() -> None:
    config = SimpleNamespace(
        num_heads=2,
        patch_temporal=1,
        patch_spatial=1,
        use_crossattn_projection=False,
    )
    stale_context = torch.full((1, 1, 3, 4), -1.0)
    new_context = torch.arange(12, dtype=torch.float32).reshape(1, 1, 3, 4)
    k_weight = torch.eye(4)
    v_weight = 2.0 * torch.eye(4)
    k_norm_weight = torch.ones(2)

    stale_cross = _cross_cache(stale_context, k_weight, v_weight, k_norm_weight)
    template = CosmosDiTNetworkCache(
        block_caches=[
            BlockCache(
                self_attn=BlockKVCache(
                    k_shape=(1, 2, 2, 2),
                    v_shape=(1, 2, 2, 2),
                    seq_dim=-3,
                    chunk_size=1,
                    window_size=2,
                    sink_size=0,
                    device="cpu",
                    dtype=torch.float32,
                ),
                cross_attn=stale_cross,
            )
        ]
    )
    shape_ops = _CosmosNetworkShapeOps(
        config,
        device=torch.device("cpu"),
        dtype=torch.float32,
        cache_templates=(template,),
        cross_cache_weights={
            "blocks.0.cross_attn.k_proj.weight": k_weight,
            "blocks.0.cross_attn.v_proj.weight": v_weight,
            "blocks.0.cross_attn.k_norm.weight": k_norm_weight,
        },
    )

    cache = shape_ops.initialize_cache(
        chunk_size=1,
        window_size=2,
        sink_size=0,
        context=new_context,
    )

    expected_cross = _cross_cache(new_context, k_weight, v_weight, k_norm_weight)
    assert torch.allclose(
        cache.block_caches[0].cross_attn.cached_k(), expected_cross.cached_k()
    )
    assert torch.allclose(
        cache.block_caches[0].cross_attn.cached_v(), expected_cross.cached_v()
    )
    assert not torch.allclose(
        cache.block_caches[0].cross_attn.cached_k(), stale_cross.cached_k()
    )


def _cross_cache(
    context: torch.Tensor,
    k_weight: torch.Tensor,
    v_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
) -> BlockKVCache:
    batch_size = 1
    token_count = int(context.shape[-2])
    num_heads = 2
    head_dim = 2
    k = F.linear(context, k_weight).reshape(
        batch_size, token_count, num_heads, head_dim
    )
    k = F.rms_norm(k, (head_dim,), weight=k_norm_weight, eps=1e-6)
    v = F.linear(context, v_weight).reshape(
        batch_size, token_count, num_heads, head_dim
    )
    return BlockKVCache.from_tensor(k.contiguous(), v.contiguous(), seq_dim=-3)
