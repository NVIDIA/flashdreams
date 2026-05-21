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

The config-wiring tests run on every CPU CI invocation. The pipeline
``.setup()`` smoke (``test_flashvsr_pipeline_setup``) resolves the
FlashVSR-v1.1 checkpoints through
:func:`flashdreams.core.checkpoint.load.load_checkpoint` against the HF
URLs in :data:`AVAILABLE_FLASHVSR_CHECKPOINT_PATHS` (i.e. the standard
``~/.cache/huggingface/hub/`` cache populated by ``hf_hub_download``)
-- a previously cached run or network access is required.
"""

from __future__ import annotations

import pytest
import torch
from flashvsr.config import (
    AVAILABLE_FLASHVSR_CHECKPOINT_PATHS,
    RUNNER_FLASHVSR_V1_1_FULL_ATTN,
    RUNNER_FLASHVSR_V1_1_SPARSE_1_5,
    RUNNER_FLASHVSR_V1_1_SPARSE_2_0,
    build_flashvsr_v1_1,
)
from flashvsr.encoder import FlashVSREncoderConfig
from flashvsr.pipeline import FlashVSRPipelineConfig
from flashvsr.runner import (
    FlashVSRRunnerConfig,
    _ensure_mgpu_config_supported,
    _resolve_target_and_topk_ratio,
)
from flashvsr.transformer import FlashVSRTransformerConfig
from flashvsr.transformer.network import FlashVSRDiTNetworkConfig, SparseSelfAttention

from flashdreams.infra.config import derive_config

pytestmark = pytest.mark.ci_gpu

_V1_1_PATHS = AVAILABLE_FLASHVSR_CHECKPOINT_PATHS["v1.1-tiny-long"]


def test_build_flashvsr_v1_1_wires_default_resolution() -> None:
    """Default 704x1280 input wires through the encoder/transformer cleanly."""
    config = build_flashvsr_v1_1(input_H=704, input_W=1280)

    assert isinstance(config, FlashVSRPipelineConfig)
    assert isinstance(config.encoder, FlashVSREncoderConfig)
    assert config.encoder.input_H == 704
    assert config.encoder.input_W == 1280
    assert config.encoder.scale == 2
    # 2x upscale of 704x1280, then /8 patchify -> 176 latent rows, 320 cols.
    # ``height``/``width`` were removed from ``FlashVSRTransformerConfig`` in
    # PR #47; the per-rollout latent dims are now derived from the encoder
    # target inside ``FlashVSRPipeline.initialize_cache`` and stashed on the
    # transformer instance. This stays a CPU-only check.
    assert config.encoder.input_H * config.encoder.scale // 8 == 176
    assert config.encoder.input_W * config.encoder.scale // 8 == 320

    transformer_config = config.diffusion_model.transformer
    assert isinstance(transformer_config, FlashVSRTransformerConfig)
    assert transformer_config.len_t == 2
    assert transformer_config.kv_ratio == 3
    assert transformer_config.attention_mode == "sparse"
    # Inherited Wan21 sizing: KV cache holds (kv_ratio + 1) * len_t pre-patchify frames.
    assert transformer_config.window_size_t == (3 + 1) * 2

    # The 1.1 prompt + projector + tcdecoder + dit checkpoints all flow in.
    assert config.prompt_path == _V1_1_PATHS["prompt"]
    assert config.encoder.projector_checkpoint_path == _V1_1_PATHS["encoder"]
    assert config.decoder.tcdecoder_checkpoint_path == _V1_1_PATHS["decoder"]
    assert transformer_config.checkpoint_path == _V1_1_PATHS["dit"]


def test_build_flashvsr_v1_1_wires_full_attention_mode() -> None:
    """Full attention is an opt-in mode that also reaches the network config."""
    config = build_flashvsr_v1_1(
        input_H=384,
        input_W=640,
        attention_mode="full",
        sparse_ratio=2.0,
    )

    transformer_config = config.diffusion_model.transformer
    assert isinstance(transformer_config, FlashVSRTransformerConfig)
    assert transformer_config.attention_mode == "full"
    assert isinstance(transformer_config.network, FlashVSRDiTNetworkConfig)
    assert transformer_config.network.attention_mode == "full"
    # The legacy top-k knob is still populated so runner derivation remains
    # stable, but the dense block ignores it at forward time.
    assert transformer_config.topk_ratio == pytest.approx(2.0)

    network = derive_config(
        transformer_config.network,
        dim=16,
        ffn_dim=32,
        num_heads=2,
        num_layers=1,
        in_dim=4,
        out_dim=4,
        text_dim=8,
        freq_dim=8,
        text_len=4,
    ).setup()
    block = network.blocks[0]
    assert block.attention_mode == "full"
    assert not isinstance(block.self_attn, SparseSelfAttention)


def test_build_flashvsr_v1_1_crops_misaligned_resolution_to_128_multiple() -> None:
    """Non-128-aligned upres dims are symmetric-cropped, not rejected.

    Mirrors upstream FlashVSR's ``compute_scaled_and_target_dims`` /
    ``upscale_then_center_crop`` helpers in
    ``examples/WanVSR/infer_flashvsr_v1.1_tiny.py``: the encoder
    bicubic-upsamples to ``(input * scale)``, then center-crops to the
    largest 128-multiple.

    ``projector_checkpoint_path=None`` keeps this CPU-only -- the random
    projector init is enough to exercise the dim math without an HF
    download.
    """
    # 540 * 2 = 1080 -> floor(1080 / 128) * 128 = 1024 (32 px top + 32 px
    # bottom symmetric trim); 960 * 2 = 1920 is already 128-aligned.
    config = derive_config(
        build_flashvsr_v1_1(input_H=540, input_W=960),
        encoder=dict(projector_checkpoint_path=None),
    )
    encoder = config.encoder.setup()
    assert encoder.scaled_H == 1080
    assert encoder.scaled_W == 1920
    assert encoder.target_H == 1024
    assert encoder.target_W == 1920

    # Width-only crop case: 416 * 2 = 832 -> 768 (32+32); 768 * 2 = 1536
    # stays 1536. Matches the ``outputs/example4.mp4`` size that
    # surfaced this code path.
    config_416 = derive_config(
        build_flashvsr_v1_1(input_H=416, input_W=768),
        encoder=dict(projector_checkpoint_path=None),
    )
    encoder_416 = config_416.encoder.setup()
    assert encoder_416.scaled_H == 832
    assert encoder_416.scaled_W == 1536
    assert encoder_416.target_H == 768
    assert encoder_416.target_W == 1536


def test_build_flashvsr_v1_1_rejects_too_small_resolution() -> None:
    """Inputs that don't cover one 128-multiple post-scale are rejected.

    Builder and encoder both assert; the builder is the user-facing
    entry point and saves a ``ZeroDivisionError`` in the ``topk_ratio``
    formula, while the encoder re-checks at ``setup()`` time so
    callers that construct ``FlashVSREncoderConfig`` directly still
    get a clean error.
    """
    # 10 * 2 = 20 -> floor(20 / 128) * 128 = 0 -> the post-crop target is empty.
    with pytest.raises(AssertionError, match="at least 128"):
        build_flashvsr_v1_1(input_H=10, input_W=10)

    # Direct EncoderConfig path (bypasses the builder).
    encoder_config = FlashVSREncoderConfig(input_H=10, input_W=10, scale=2)
    with pytest.raises(AssertionError, match="too small to crop"):
        encoder_config.setup()


def test_build_flashvsr_v1_1_scales_topk_with_resolution() -> None:
    """``topk_ratio`` follows the upstream 768 * 1280 / (target_H * target_W) formula.

    The target dims here are the **post-crop** 128-multiple target the
    encoder operates on, mirroring upstream's per-call
    ``topk_ratio = sparse_ratio * 768*1280 / (th*tw)`` in
    ``examples/WanVSR/infer_flashvsr_v1.1_tiny.py`` where ``(th, tw)``
    are the cropped target.
    """
    # Reference resolution at which the top-k budget matches sparse_ratio
    # exactly (the FlashVSR-tiny "base" target). Mirrors the literal in
    # ``flashvsr.config._transformer_config``.
    REF_H, REF_W = 768, 1280

    def expected_topk(
        *, input_H: int, input_W: int, scale: int, sparse_ratio: float
    ) -> float:
        target_H = ((input_H * scale) // 128) * 128
        target_W = ((input_W * scale) // 128) * 128
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

    # Non-128-aligned: 416 * 2 = 832 -> 768; 768 * 2 = 1536. topk_ratio
    # tracks the cropped (768, 1536), not the un-cropped (832, 1536),
    # matching upstream's per-input formula.
    misaligned = build_flashvsr_v1_1(input_H=416, input_W=768, sparse_ratio=1.5)
    misaligned_xfm = misaligned.diffusion_model.transformer
    assert isinstance(misaligned_xfm, FlashVSRTransformerConfig)
    assert misaligned_xfm.topk_ratio == pytest.approx(
        expected_topk(input_H=416, input_W=768, scale=2, sparse_ratio=1.5)
    )
    # Cross-check vs the un-cropped value: had we used target = 832 x 1536
    # we'd get a smaller ratio (768*1280 / (832*1536) ~= 0.7692 < 0.8333).
    uncropped = 1.5 * REF_H * REF_W / (832 * 1536)
    assert misaligned_xfm.topk_ratio > uncropped


def test_shipped_runners_carry_their_sparse_ratio() -> None:
    """``_build_sparse_ratio_variant`` populates ``runner.sparse_ratio``.

    Without this, ``FlashVSRRunner.run`` would re-derive ``topk_ratio``
    against the default ``sparse_ratio=2.0`` for every shipped slug,
    silently overriding the ``-sparse-ratio-1.5`` preset.
    """
    assert isinstance(RUNNER_FLASHVSR_V1_1_SPARSE_2_0, FlashVSRRunnerConfig)
    assert RUNNER_FLASHVSR_V1_1_SPARSE_2_0.sparse_ratio == pytest.approx(2.0)
    assert isinstance(RUNNER_FLASHVSR_V1_1_SPARSE_1_5, FlashVSRRunnerConfig)
    assert RUNNER_FLASHVSR_V1_1_SPARSE_1_5.sparse_ratio == pytest.approx(1.5)
    assert isinstance(RUNNER_FLASHVSR_V1_1_FULL_ATTN, FlashVSRRunnerConfig)
    full_transformer = (
        RUNNER_FLASHVSR_V1_1_FULL_ATTN.pipeline.diffusion_model.transformer
    )
    assert isinstance(full_transformer, FlashVSRTransformerConfig)
    assert full_transformer.attention_mode == "full"


def test_mgpu_guard_rejects_sparse_flashvsr_presets() -> None:
    """Only the full-attention FlashVSR preset supports multi-GPU."""
    for sparse_runner in (
        RUNNER_FLASHVSR_V1_1_SPARSE_2_0,
        RUNNER_FLASHVSR_V1_1_SPARSE_1_5,
    ):
        with pytest.raises(ValueError, match="block_sparse_attn"):
            _ensure_mgpu_config_supported(sparse_runner, world_size=2)
        _ensure_mgpu_config_supported(sparse_runner, world_size=1)
    _ensure_mgpu_config_supported(RUNNER_FLASHVSR_V1_1_FULL_ATTN, world_size=2)


@pytest.mark.parametrize(
    ("input_H", "input_W", "scale", "sparse_ratio", "expected_target", "expected_topk"),
    [
        # 128-aligned base case -> target = REF -> topk_ratio == sparse_ratio.
        (384, 640, 2, 2.0, (768, 1280), 2.0),
        # Encoder's "scaffold" placeholder. Target = 1408 x 2560; topk
        # shrinks by REF / target = 0.2727.
        (704, 1280, 2, 1.5, (1408, 2560), 1.5 * 768 * 1280 / (1408 * 2560)),
        # Non-128-aligned: height crops 832 -> 768; width stays 1536.
        # This is the ``outputs/example4.mp4`` shape that surfaced the
        # divisibility issue. topk_ratio scales against the **cropped**
        # target, not the un-cropped one.
        (416, 768, 2, 1.5, (768, 1536), 1.5 * 768 * 1280 / (768 * 1536)),
    ],
)
def test_resolve_target_and_topk_ratio_matches_upstream(
    input_H: int,
    input_W: int,
    scale: int,
    sparse_ratio: float,
    expected_target: tuple[int, int],
    expected_topk: float,
) -> None:
    """The runner helper mirrors upstream's per-input topk formula."""
    target_H, target_W, topk_ratio = _resolve_target_and_topk_ratio(
        input_H=input_H,
        input_W=input_W,
        scale=scale,
        sparse_ratio=sparse_ratio,
    )
    assert (target_H, target_W) == expected_target
    assert topk_ratio == pytest.approx(expected_topk)


def test_resolve_target_and_topk_ratio_rejects_too_small_input() -> None:
    """Inputs that don't fit one 128-multiple post-scale raise on the runner side."""
    with pytest.raises(AssertionError, match="too small to crop"):
        _resolve_target_and_topk_ratio(
            input_H=10, input_W=10, scale=2, sparse_ratio=2.0
        )


def test_runner_overrides_topk_ratio_via_derive_config() -> None:
    """End-to-end: shipped runner cfg + runner-style derive_config = correct topk.

    Reproduces the exact ``derive_config`` shape that
    :meth:`FlashVSRRunner.run` uses for a 416x768 input on the
    ``sparse-ratio-1.5`` recipe and verifies both the encoder dims and
    the transformer ``topk_ratio`` end up overridden.
    """
    runner_cfg = RUNNER_FLASHVSR_V1_1_SPARSE_1_5
    # ``RunnerConfig.pipeline`` is typed as the abstract
    # ``StreamInferencePipelineConfig`` whose ``encoder`` /
    # ``transformer`` slots widen to the base configs; narrow once so
    # FlashVSR-specific fields (``encoder.scale``,
    # ``transformer.topk_ratio``) are accessible below.
    runner_pipeline = runner_cfg.pipeline
    assert isinstance(runner_pipeline, FlashVSRPipelineConfig)
    runner_transformer = runner_pipeline.diffusion_model.transformer
    assert isinstance(runner_transformer, FlashVSRTransformerConfig)

    H, W = 416, 768
    target_H, target_W, topk_ratio = _resolve_target_and_topk_ratio(
        input_H=H,
        input_W=W,
        scale=runner_pipeline.encoder.scale,
        sparse_ratio=runner_cfg.sparse_ratio,
    )
    pipeline_config = derive_config(
        runner_pipeline,
        encoder=dict(input_H=H, input_W=W),
        diffusion_model=dict(transformer=dict(topk_ratio=topk_ratio)),
    )
    assert isinstance(pipeline_config, FlashVSRPipelineConfig)
    assert pipeline_config.encoder.input_H == H
    assert pipeline_config.encoder.input_W == W
    # Other encoder knobs (scale, dtype, ...) pass through unchanged.
    assert pipeline_config.encoder.scale == runner_pipeline.encoder.scale
    transformer = pipeline_config.diffusion_model.transformer
    assert isinstance(transformer, FlashVSRTransformerConfig)
    assert transformer.topk_ratio == pytest.approx(topk_ratio)
    # Sanity: the override actually changed something. Builder placeholder
    # was 704x1280 (-> 1408x2560 -> topk_ratio ~= 0.409) which differs
    # noticeably from the 416x768 (-> 768x1536 -> topk_ratio ~= 1.25)
    # we just derived.
    assert transformer.topk_ratio != pytest.approx(runner_transformer.topk_ratio)
    # Match expected post-crop target for the same (H, W).
    assert (target_H, target_W) == (768, 1536)


@pytest.mark.manual
def test_flashvsr_pipeline_setup() -> None:
    """``build_flashvsr_v1_1(...).setup()`` instantiates the full pipeline.

    Stays on CPU (no ``.to('cuda')``) so it can exercise the import +
    checkpoint-load + module-graph paths on a CPU CI runner. Checkpoint
    resolution flows through the production ``load_checkpoint(URL)`` path
    in ``flashvsr.encoder/decoder/transformer.setup()``, which routes
    HF URLs in :data:`AVAILABLE_FLASHVSR_CHECKPOINT_PATHS` through
    ``hf_hub_download`` (i.e. the standard
    ``~/.cache/huggingface/hub/`` cache) -- a previously cached run or
    network access is required. Marked ``manual`` to keep the
    HF-cache-dependent path opt-in.
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
