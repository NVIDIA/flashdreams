# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CUDA equivalence tests for Waypoint's fixed-capacity attention cache."""

from dataclasses import replace

import pytest
import torch
import torch.nn.functional as F
from waypoint import WAYPOINT_1_5
from waypoint.transformer import WaypointAttentionPolicy, WaypointKVCache
from waypoint.transformer.network import _compiled_fixed_attention

pytestmark = pytest.mark.ci_gpu


@pytest.mark.parametrize("layer_index", [0, 3], ids=["local", "global"])
def test_fixed_cuda_attention_matches_compact_reference(layer_index: int) -> None:
    """Fixed CUDA history produces the compact policy's attention results."""
    assert torch.cuda.is_available()
    spec = replace(
        WAYPOINT_1_5,
        n_layers=4,
        local_window=2,
        global_window=6,
        global_pinned_dilation=2,
    )
    policy = WaypointAttentionPolicy(spec=spec)
    compact = WaypointKVCache(policy=policy)
    fixed = WaypointKVCache(policy=policy, use_fixed_attention=True)
    generator = torch.Generator(device="cuda").manual_seed(464)

    for frame_index in range(8):
        key = torch.randn(
            1,
            1,
            128,
            16,
            device="cuda",
            dtype=torch.bfloat16,
            generator=generator,
        )
        value = torch.randn(
            key.shape,
            device="cuda",
            dtype=key.dtype,
            generator=generator,
        )
        query = torch.randn(
            1,
            2,
            128,
            16,
            device="cuda",
            dtype=torch.bfloat16,
            generator=generator,
        )
        reference_view = compact.update(
            layer_index=layer_index,
            frame_index=frame_index,
            key=key,
            value=value,
        )
        fixed.set_frozen(True)
        provisional_view = fixed.update(
            layer_index=layer_index,
            frame_index=frame_index,
            key=key,
            value=value,
        )
        repeated_view = fixed.update(
            layer_index=layer_index,
            frame_index=frame_index,
            key=key,
            value=value,
        )
        assert repeated_view.block_mask is provisional_view.block_mask
        fixed.set_frozen(False)
        fixed_view = fixed.update(
            layer_index=layer_index,
            frame_index=frame_index,
            key=key,
            value=value,
        )
        assert fixed_view.block_mask is provisional_view.block_mask
        fixed.set_frozen(True)

        reference = F.scaled_dot_product_attention(
            query,
            reference_view.key,
            reference_view.value,
            enable_gqa=True,
        )
        assert fixed_view.block_mask is not None
        actual = _compiled_fixed_attention(
            query,
            fixed_view.key,
            fixed_view.value,
            fixed_view.block_mask,
        )
        torch.testing.assert_close(actual, reference, atol=2e-2, rtol=2e-2)
