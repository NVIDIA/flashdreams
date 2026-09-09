# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU contracts for the FlashDreams-native SwiftVR transformer."""

from typing import cast

import pytest
import torch
from swiftvr.impl.attention import SwiftVRBlock, _axis_starts
from swiftvr.impl.model import SwiftVRDiTNetwork, SwiftVRPipeline

from flashdreams.recipes.wan import wan_dit_state_dict_from_diffusers
from flashdreams.recipes.wan.transformer.impl.network import (
    WanDiTNetwork,
    WanDiTNetworkTI2V5BConfig,
)

pytestmark = pytest.mark.ci_cpu


def _diffusers_checkpoint_key(native_key: str) -> str:
    replacements = (
        ("text_embedding.0", "condition_embedder.text_embedder.linear_1"),
        ("text_embedding.2", "condition_embedder.text_embedder.linear_2"),
        ("time_embedding.0", "condition_embedder.time_embedder.linear_1"),
        ("time_embedding.2", "condition_embedder.time_embedder.linear_2"),
        ("time_projection.1", "condition_embedder.time_proj"),
        ("head.modulation", "scale_shift_table"),
        ("head.head", "proj_out"),
        (".self_attn.q", ".attn1.to_q"),
        (".self_attn.k", ".attn1.to_k"),
        (".self_attn.v", ".attn1.to_v"),
        (".self_attn.o", ".attn1.to_out.0"),
        (".cross_attn.q", ".attn2.to_q"),
        (".cross_attn.k", ".attn2.to_k"),
        (".cross_attn.v", ".attn2.to_v"),
        (".cross_attn.o", ".attn2.to_out.0"),
        (".self_attn.norm_q", ".attn1.norm_q"),
        (".self_attn.norm_k", ".attn1.norm_k"),
        (".cross_attn.norm_q", ".attn2.norm_q"),
        (".cross_attn.norm_k", ".attn2.norm_k"),
        (".norm3", ".norm2"),
        (".modulation", ".scale_shift_table"),
        (".ffn.0", ".ffn.net.0.proj"),
        (".ffn.2", ".ffn.net.2"),
    )
    for native, diffusers in replacements:
        if native in native_key:
            return native_key.replace(native, diffusers, 1)
    return native_key


def test_swiftvr_preserves_native_wan_checkpoint_layout() -> None:
    config = WanDiTNetworkTI2V5BConfig()
    with torch.device("meta"):
        native = WanDiTNetwork(config)
        swiftvr = SwiftVRDiTNetwork(config, (16, 16))

    native_shapes = {
        key: tuple(value.shape) for key, value in native.state_dict().items()
    }
    swiftvr_shapes = {
        key: tuple(value.shape) for key, value in swiftvr.state_dict().items()
    }

    assert len(swiftvr_shapes) == 825
    assert swiftvr_shapes == native_shapes
    diffusers_state = {
        _diffusers_checkpoint_key(key): value
        for key, value in native.state_dict().items()
    }
    unchanged_keys = {
        key for key in native_shapes if _diffusers_checkpoint_key(key) == key
    }
    remapped_shapes = {
        key: tuple(value.shape)
        for key, value in wan_dit_state_dict_from_diffusers(diffusers_state).items()
    }
    assert unchanged_keys == {"patch_embedding.weight", "patch_embedding.bias"}
    assert len(diffusers_state) == 825
    assert remapped_shapes == native_shapes
    assert all(isinstance(block, SwiftVRBlock) for block in swiftvr.blocks)
    assert [
        cast(SwiftVRBlock, block).self_attn.shifted for block in swiftvr.blocks[:4]
    ] == [
        False,
        True,
        False,
        True,
    ]


def test_legacy_diffusers_ffn_keys_map_to_native_wan_components() -> None:
    source = {
        "blocks.7.ffn.fc_in.weight": torch.empty(1),
        "blocks.7.ffn.fc_out.bias": torch.empty(1),
    }

    assert set(wan_dit_state_dict_from_diffusers(source)) == {
        "blocks.7.ffn.0.weight",
        "blocks.7.ffn.2.bias",
    }


def test_shifted_blocks_use_distinct_windows() -> None:
    unshifted = _axis_starts(33, 16, shifted=False, device=torch.device("cpu")).tolist()
    shifted = _axis_starts(33, 16, shifted=True, device=torch.device("cpu")).tolist()

    assert unshifted == [0, 16, 17]
    assert shifted == [0, 8, 17]


def test_pipeline_rejects_cpu_before_checkpoint_resolution() -> None:
    with pytest.raises(ValueError, match="requires CUDA"):
        SwiftVRPipeline.from_pretrained(
            "unused",
            revision=None,
            device="cpu",
            dtype=torch.bfloat16,
            attention_window=(16, 16),
            compile_blocks=False,
        )
