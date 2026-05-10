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

"""Smoke tests for the FlashVSR recipe.

The config-wiring tests run on every CPU CI invocation; the pipeline
``.setup()`` smoke is opt-in (requires the FlashVSR-v1.1 weights staged
under ``$FLASHDREAMS_CACHE_DIR/upsampler/weights/FlashVSR-v1.1``) and
gated behind ``pytest.mark.manual``.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from flashdreams.recipes.flashvsr.config import (
    AVAILABLE_FLASHVSR_DIT_CHECKPOINT_PATHS,
    AVAILABLE_FLASHVSR_PROJECTOR_PATHS,
    AVAILABLE_FLASHVSR_PROMPT_PATHS,
    AVAILABLE_FLASHVSR_TCDECODER_PATHS,
    build_flashvsr_v1_1,
)
from flashdreams.recipes.flashvsr.encoder import FlashVSREncoderConfig
from flashdreams.recipes.flashvsr.pipeline import FlashVSRPipelineConfig
from flashdreams.recipes.flashvsr.transformer import FlashVSRTransformerConfig


def test_build_flashvsr_v1_1_wires_default_resolution() -> None:
    """Default 704x1280 input wires through the encoder/transformer cleanly."""
    config = build_flashvsr_v1_1(input_H=704, input_W=1280)

    assert isinstance(config, FlashVSRPipelineConfig)
    assert isinstance(config.encoder, FlashVSREncoderConfig)
    assert config.encoder.input_H == 704
    assert config.encoder.input_W == 1280
    assert config.encoder.scale == 2

    transformer_config = config.diffusion_model.transformer
    assert isinstance(transformer_config, FlashVSRTransformerConfig)
    # 2x upscale of 704x1280, then /8 patchify -> 176 latent rows, 320 cols.
    assert transformer_config.height == 176
    assert transformer_config.width == 320
    assert transformer_config.len_t == 2
    assert transformer_config.kv_ratio == 3
    # Inherited Wan21 sizing: KV cache holds (kv_ratio + 1) * len_t pre-patchify frames.
    assert transformer_config.window_size_t == (3 + 1) * 2

    # The 1.1 prompt + projector + tcdecoder + dit checkpoints all flow in.
    assert config.prompt_path == AVAILABLE_FLASHVSR_PROMPT_PATHS["v1.1"]
    assert (
        config.encoder.projector_checkpoint_path
        == AVAILABLE_FLASHVSR_PROJECTOR_PATHS["v1.1"]
    )
    assert (
        config.decoder.tcdecoder_checkpoint_path
        == AVAILABLE_FLASHVSR_TCDECODER_PATHS["v1.1"]
    )
    assert (
        transformer_config.checkpoint_path
        == AVAILABLE_FLASHVSR_DIT_CHECKPOINT_PATHS["v1.1-tiny-long"]
    )


def test_build_flashvsr_v1_1_rejects_misaligned_resolution() -> None:
    """Target resolution must be divisible by 128 for the FlashVSR DiT."""
    config = build_flashvsr_v1_1(input_H=540, input_W=960)
    # 540 * 2 = 1080, which is not divisible by 128; the encoder asserts at setup.
    with pytest.raises(AssertionError, match="divisible by 128"):
        config.encoder.setup()


def test_build_flashvsr_v1_1_scales_topk_with_resolution() -> None:
    """``topk_ratio`` follows the legacy 768 * 1280 / (target_H * target_W) formula."""
    # Reference resolution at which the top-k budget matches sparse_ratio
    # exactly (the FlashVSR-tiny "base" target). Mirrors the literal in
    # ``flashdreams.recipes.flashvsr.config._transformer_config``.
    REF_H, REF_W = 768, 1280

    def expected_topk(
        *, input_H: int, input_W: int, scale: int, sparse_ratio: float
    ) -> float:
        target_H, target_W = input_H * scale, input_W * scale
        return sparse_ratio * REF_H * REF_W / (target_H * target_W)

    base = build_flashvsr_v1_1(input_H=384, input_W=640, sparse_ratio=2.0)
    # target = 768 x 1280 = REF_H x REF_W -> ratio is exactly sparse_ratio (2.0).
    base_xfm = base.diffusion_model.transformer
    assert isinstance(base_xfm, FlashVSRTransformerConfig)
    assert base_xfm.topk_ratio == pytest.approx(
        expected_topk(input_H=384, input_W=640, scale=2, sparse_ratio=2.0)
    )
    assert base_xfm.topk_ratio == pytest.approx(2.0)

    # target = 1408 x 2560 = 3.667 x base (not 4x: 1408/768 = 1.833,
    # 2560/1280 = 2.0). topk_ratio scales 1/3.667 -> ~0.5455.
    larger = build_flashvsr_v1_1(input_H=704, input_W=1280, sparse_ratio=2.0)
    larger_xfm = larger.diffusion_model.transformer
    assert isinstance(larger_xfm, FlashVSRTransformerConfig)
    assert larger_xfm.topk_ratio == pytest.approx(
        expected_topk(input_H=704, input_W=1280, scale=2, sparse_ratio=2.0)
    )


_WEIGHTS_ROOT = Path(AVAILABLE_FLASHVSR_PROJECTOR_PATHS["v1.1"]).expanduser().parent
_WEIGHTS_REASON = (
    f"FlashVSR-v1.1 weights not found under {_WEIGHTS_ROOT}; "
    "stage with internal/upsampler/scripts/download_flashvsr_weights.sh."
)


@pytest.mark.skipif(not _WEIGHTS_ROOT.exists(), reason=_WEIGHTS_REASON)
def test_flashvsr_pipeline_setup() -> None:
    """``build_flashvsr_v1_1(...).setup()`` instantiates the full pipeline.

    Stays on CPU (no ``.to('cuda')``) so it can exercise the import +
    checkpoint-load + module-graph paths on a CPU CI runner. Weight files
    must already be staged; the ``skipif`` makes this auto-skip on CI
    runners without weights staged.
    """
    config = build_flashvsr_v1_1(
        input_H=384,
        input_W=640,
        dtype=torch.float32,
    )
    pipeline = config.setup()
    assert pipeline.encoder is not None
    assert pipeline.decoder is not None
    assert pipeline.diffusion_model is not None
