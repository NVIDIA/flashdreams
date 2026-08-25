# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU-safe configuration checks for the FlashVSR integration."""

import pytest

from flashvsr.config import (
    PIPELINE_FLASHVSR_V1_1_FULL_ATTN,
    PIPELINE_FLASHVSR_V1_1_SPARSE_1_5,
    PIPELINE_FLASHVSR_V1_1_SPARSE_2_0,
    build_flashvsr_v1_1,
)
from flashvsr.impl.encoder import FlashVSREncoderConfig
from flashvsr.impl.pipeline import FlashVSRPipelineConfig
from flashvsr.impl.transformer import FlashVSRTransformerConfig

pytestmark = pytest.mark.ci_cpu


def test_builder_wires_model_components() -> None:
    """Build the model pipeline without a runner or application shim."""
    config = build_flashvsr_v1_1(input_H=704, input_W=1280)

    assert isinstance(config, FlashVSRPipelineConfig)
    assert isinstance(config.encoder, FlashVSREncoderConfig)
    assert isinstance(config.diffusion_model.transformer, FlashVSRTransformerConfig)


def test_builder_scales_sparse_budget_with_cropped_resolution() -> None:
    """Use the post-crop target dimensions for the FlashVSR top-k budget."""
    config = build_flashvsr_v1_1(
        input_H=416,
        input_W=768,
        sparse_ratio=1.5,
    )
    transformer = config.diffusion_model.transformer

    assert isinstance(transformer, FlashVSRTransformerConfig)
    assert transformer.topk_ratio == pytest.approx(1.5 * 768 * 1280 / (768 * 1536))


def test_shipped_configs_cover_sparse_and_full_attention() -> None:
    """Keep all public variants as pipeline configs in the root config module."""
    sparse_2 = PIPELINE_FLASHVSR_V1_1_SPARSE_2_0.diffusion_model.transformer
    sparse_1_5 = PIPELINE_FLASHVSR_V1_1_SPARSE_1_5.diffusion_model.transformer
    full = PIPELINE_FLASHVSR_V1_1_FULL_ATTN.diffusion_model.transformer

    assert sparse_2.attention_mode == "sparse"
    assert sparse_1_5.attention_mode == "sparse"
    assert full.attention_mode == "full"


def test_builder_rejects_empty_post_crop_target() -> None:
    """Reject inputs too small to yield one 128-pixel target tile."""
    with pytest.raises(AssertionError, match="at least 128"):
        build_flashvsr_v1_1(input_H=10, input_W=10)
