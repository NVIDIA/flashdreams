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

"""Smoke tests for the FlashVSR integration.

The config-wiring tests run on every CPU CI invocation. The pipeline
``.setup()`` smoke (``test_flashvsr_pipeline_setup``) resolves the
FlashVSR-v1.1 checkpoints through
:func:`flashdreams.core.checkpoint.load.load_checkpoint` against the HF
URLs in :data:`AVAILABLE_FLASHVSR_CHECKPOINT_PATHS` (i.e. the standard
``~/.cache/huggingface/hub/`` cache populated by ``hf_hub_download``)
-- a previously cached run or network access is required.
"""

from __future__ import annotations

import builtins

import pytest
import torch
from flashdreams.infra.config import derive_config
from flashvsr.config import (
    AVAILABLE_FLASHVSR_CHECKPOINT_PATHS,
    PIPELINE_FLASHVSR_V1_1_FULL_ATTN,
    PIPELINE_FLASHVSR_V1_1_SPARSE_1_5,
    PIPELINE_FLASHVSR_V1_1_SPARSE_2_0,
    build_flashvsr_v1_1,
)
from flashvsr.impl.encoder import FlashVSREncoderConfig
from flashvsr.impl.pipeline import FlashVSRPipelineConfig
from flashvsr.impl.transformer import FlashVSRTransformerConfig
from flashvsr.impl.transformer import network as flashvsr_network
from flashvsr.impl.transformer.network import (
    FlashVSRDiTNetworkConfig,
    SparseSelfAttention,
)

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


def test_sparse_self_attention_uses_in_tree_triton_backend(monkeypatch) -> None:
    """Sparse attention dispatches to the in-tree Triton backend."""

    monkeypatch.setattr(
        flashvsr_network, "apply_rope_freqs", lambda x, *_args, **_kwargs: x
    )

    original_import = builtins.__import__

    def fail_block_sparse_import(name, *args, **kwargs):
        if name.startswith("block_sparse_attn"):
            raise AssertionError(
                "external block_sparse_attn package should not be imported"
            )
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fail_block_sparse_import)

    calls = []

    def fake_triton_backend(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        head_mask_type,
        streaming_info,
        base_blockmask,
        max_seqlen_q_,
        max_seqlen_k_,
        p_dropout,
        *,
        deterministic=False,
        softmax_scale=None,
        is_causal=False,
        exact_streaming=False,
        return_attn_probs=False,
        mode_hint=None,
        head_mask_type_is_renumbered=False,
    ):
        calls.append(
            {
                "q_shape": tuple(q.shape),
                "k_shape": tuple(k.shape),
                "v_shape": tuple(v.shape),
                "cu_seqlens_q": cu_seqlens_q.tolist(),
                "cu_seqlens_k": cu_seqlens_k.tolist(),
                "head_mask_type": head_mask_type.tolist(),
                "streaming_info": streaming_info,
                "base_blockmask_shape": tuple(base_blockmask.shape),
                "max_seqlen_q": max_seqlen_q_,
                "max_seqlen_k": max_seqlen_k_,
                "p_dropout": p_dropout,
                "deterministic": deterministic,
                "softmax_scale": softmax_scale,
                "is_causal": is_causal,
                "exact_streaming": exact_streaming,
                "return_attn_probs": return_attn_probs,
                "mode_hint": mode_hint,
                "head_mask_type_is_renumbered": head_mask_type_is_renumbered,
            }
        )
        return torch.zeros_like(q)

    monkeypatch.setattr(
        flashvsr_network,
        "_get_block_sparse_attn_triton_func",
        lambda: fake_triton_backend,
    )

    attn = SparseSelfAttention(query_dim=32, n_heads=1, head_dim=32)
    cache = attn.initialize_cache(
        batch_size=1,
        chunk_size=128,
        window_size=128,
        sink_size=0,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    x = torch.randn(1, 128, 32)
    rope_freqs = torch.empty(128, 1, 1, 32)

    cache.before_update(0)
    out = attn(
        x,
        kv_cache=cache,
        rope_freqs=rope_freqs,
        f=2,
        h=8,
        w=8,
        topk=1,
        local_range=1,
    )
    cache.after_update(0)

    assert out.shape == x.shape
    assert len(calls) == 1
    call = calls[0]
    assert call["q_shape"] == (128, 1, 32)
    assert call["k_shape"] == (128, 1, 32)
    assert call["v_shape"] == (128, 1, 32)
    assert call["cu_seqlens_q"] == [0, 128]
    assert call["cu_seqlens_k"] == [0, 128]
    assert call["head_mask_type"] == [1]
    assert call["streaming_info"].tolist() == [0, 0]
    assert call["base_blockmask_shape"] == (1, 1, 1, 1)
    assert call["max_seqlen_q"] == 128
    assert call["max_seqlen_k"] == 128
    assert call["p_dropout"] == 0.0
    assert call["deterministic"] is False
    assert call["softmax_scale"] is None
    assert call["is_causal"] is False
    assert call["exact_streaming"] is False
    assert call["return_attn_probs"] is False
    assert call["mode_hint"] == "blocksparse"
    assert call["head_mask_type_is_renumbered"] is True


@pytest.mark.parametrize(
    ("max_seqlen_k", "seq_bucket"),
    [
        (27648, "long_single"),
        (36864, "very_long_single"),
    ],
)
def test_triton_blocksparse_flashvsr_single_batch_hdim128_uses_tuned_dispatch(
    max_seqlen_k: int,
    seq_bucket: str,
) -> None:
    """FlashVSR sparse inference uses the measured fast Triton path."""

    triton_sparse_attn = pytest.importorskip(
        "flashvsr.impl.transformer.triton_sparse_attn"
    )
    q = torch.empty((9216, 12, 128), dtype=torch.bfloat16)

    key, opts = triton_sparse_attn._dispatch_kernel_options(
        q=q,
        batch_size=1,
        max_seqlen_k=max_seqlen_k,
        exact_streaming=False,
        mode="blocksparse",
        has_base_blockmask=True,
    )

    assert key.seq_bucket == seq_bucket
    assert opts.block_n_sparse == 64
    assert opts.use_row_list_sparse is False
    assert opts.num_warps_sparse == 4


def test_triton_blocksparse_inference_uses_opaque_forward(monkeypatch) -> None:
    """Compiled FlashVSR inference keeps Triton sparse attention opaque to Inductor."""

    triton_sparse_attn = pytest.importorskip(
        "flashvsr.impl.transformer.triton_sparse_attn"
    )
    monkeypatch.setattr(
        triton_sparse_attn,
        "_validate_forward_inputs",
        lambda **_kwargs: None,
    )

    calls = []

    def fake_opaque_forward(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        head_mask_type,
        streaming_info,
        base_blockmask,
        max_seqlen_q_,
        max_seqlen_k_,
        scale,
        is_causal,
        exact_streaming,
    ):
        calls.append(
            {
                "q_shape": tuple(q.shape),
                "k_shape": tuple(k.shape),
                "v_shape": tuple(v.shape),
                "cu_seqlens_q": cu_seqlens_q.tolist(),
                "cu_seqlens_k": cu_seqlens_k.tolist(),
                "head_mask_type": head_mask_type.tolist(),
                "streaming_info": streaming_info.tolist(),
                "base_blockmask_shape": tuple(base_blockmask.shape),
                "max_seqlen_q": max_seqlen_q_,
                "max_seqlen_k": max_seqlen_k_,
                "scale": scale,
                "is_causal": is_causal,
                "exact_streaming": exact_streaming,
            }
        )
        return torch.zeros_like(q)

    monkeypatch.setattr(
        triton_sparse_attn,
        "_wrapped_triton_block_sparse_attn_forward",
        fake_opaque_forward,
    )

    q = torch.empty((128, 1, 32), dtype=torch.bfloat16)
    k = torch.empty_like(q)
    v = torch.empty_like(q)
    cu_seqlens = torch.tensor([0, 128], dtype=torch.int32)
    head_mask_type = torch.tensor([1], dtype=torch.int32)
    streaming_info = torch.zeros((2,), dtype=torch.int32)
    base_blockmask = torch.ones((1, 1, 1, 1), dtype=torch.bool)

    with torch.inference_mode():
        out = triton_sparse_attn.block_sparse_attn_func(
            q,
            k,
            v,
            cu_seqlens,
            cu_seqlens,
            head_mask_type,
            streaming_info,
            base_blockmask,
            128,
            128,
            0.0,
            softmax_scale=None,
            is_causal=False,
            exact_streaming=False,
            return_attn_probs=False,
            mode_hint="blocksparse",
            head_mask_type_is_renumbered=True,
        )

    assert out.shape == q.shape
    assert len(calls) == 1
    call = calls[0]
    assert call["q_shape"] == (128, 1, 32)
    assert call["k_shape"] == (128, 1, 32)
    assert call["v_shape"] == (128, 1, 32)
    assert call["cu_seqlens_q"] == [0, 128]
    assert call["cu_seqlens_k"] == [0, 128]
    assert call["head_mask_type"] == [1]
    assert call["streaming_info"] == [0, 0]
    assert call["base_blockmask_shape"] == (1, 1, 1, 1)
    assert call["max_seqlen_q"] == 128
    assert call["max_seqlen_k"] == 128
    assert call["scale"] == pytest.approx(32**-0.5)
    assert call["is_causal"] is False
    assert call["exact_streaming"] is False


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


def test_shipped_pipeline_presets_are_v2_pipeline_configs() -> None:
    """Expose model presets without wrapping them in runner configs."""
    for config in (
        PIPELINE_FLASHVSR_V1_1_SPARSE_2_0,
        PIPELINE_FLASHVSR_V1_1_SPARSE_1_5,
        PIPELINE_FLASHVSR_V1_1_FULL_ATTN,
    ):
        assert isinstance(config, FlashVSRPipelineConfig)

    stable_transformer = PIPELINE_FLASHVSR_V1_1_SPARSE_2_0.diffusion_model.transformer
    fast_transformer = PIPELINE_FLASHVSR_V1_1_SPARSE_1_5.diffusion_model.transformer
    full_transformer = PIPELINE_FLASHVSR_V1_1_FULL_ATTN.diffusion_model.transformer
    assert isinstance(stable_transformer, FlashVSRTransformerConfig)
    assert isinstance(fast_transformer, FlashVSRTransformerConfig)
    assert isinstance(full_transformer, FlashVSRTransformerConfig)
    assert stable_transformer.topk_ratio > fast_transformer.topk_ratio
    assert full_transformer.attention_mode == "full"


@pytest.mark.manual
def test_flashvsr_pipeline_setup() -> None:
    """``build_flashvsr_v1_1(...).setup()`` instantiates the full pipeline.

    Stays on CPU (no ``.to('cuda')``) so it can exercise the import +
    checkpoint-load + module-graph paths on a CPU CI runner. Checkpoint
    resolution flows through the production ``load_checkpoint(URL)`` path
    in ``flashvsr.impl.encoder/decoder/transformer.setup()``, which routes
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
