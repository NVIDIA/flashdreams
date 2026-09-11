# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU contracts for the FlashDreams-native SwiftVR pipeline."""

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
from swiftvr.config import build_swiftvr_pipeline
from swiftvr.impl.attention import SwiftVRBlock, _axis_starts
from swiftvr.impl.decoder import SwiftVRDecoderConfig
from swiftvr.impl.decoder.network import SwiftVRTAEHV, SwiftVRTemporalGrow
from swiftvr.impl.encoder import (
    SwiftVREncoder,
    SwiftVREncoderConfig,
    _encode_complete_groups,
)
from swiftvr.impl.pipeline import SwiftVRPipeline, SwiftVRPipelineConfig
from swiftvr.impl.transformer import SwiftVRTransformerConfig
from swiftvr.impl.transformer.network import (
    SwiftVRDiTNetwork,
    SwiftVRDiTNetworkConfig,
)

from flashdreams.infra.pipeline import StreamInferencePipeline
from flashdreams.recipes.taehv.checkpoint import legacy_to_blocks_keys
from flashdreams.recipes.taehv.impl import TAEHV, MemBlock, TGrow, TPool
from flashdreams.recipes.taehv.impl import Encoder as TAEHVEncoder
from flashdreams.recipes.wan import wan_dit_state_dict_from_diffusers
from flashdreams.recipes.wan.transformer.impl.network import (
    WanDiTNetwork,
    WanDiTNetworkTI2V5BConfig,
)

pytestmark = pytest.mark.ci_cpu


def test_pipeline_config_follows_stream_inference_component_contracts(
    tmp_path: Path,
) -> None:
    (tmp_path / "reae.safetensors").touch()
    (tmp_path / "prompt_embedding.safetensors").touch()
    transformer_dir = tmp_path / "transformer"
    transformer_dir.mkdir()
    (transformer_dir / "diffusion_pytorch_model.safetensors").touch()

    config = build_swiftvr_pipeline(
        checkpoint=str(tmp_path),
        revision=None,
        attention_window=(8, 12),
        chunk_size=24,
    )

    assert isinstance(config, SwiftVRPipelineConfig)
    assert config._target is SwiftVRPipeline
    assert issubclass(SwiftVRPipeline, StreamInferencePipeline)
    assert isinstance(config.encoder, SwiftVREncoderConfig)
    assert isinstance(config.diffusion_model.transformer, SwiftVRTransformerConfig)
    assert isinstance(config.decoder, SwiftVRDecoderConfig)
    assert config.diffusion_model.transformer.network.attention_window == (8, 12)
    assert config.diffusion_model.transformer.latent_frames == 6


def test_pipeline_cache_uses_reae_latent_resolution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def capture_initialize_cache(
        _self: StreamInferencePipeline,
        transformer_context: dict[str, Any] | None = None,
        encoder_context: dict[str, Any] | None = None,
        decoder_context: dict[str, Any] | None = None,
    ) -> Any:
        captured.update(
            transformer=transformer_context,
            encoder=encoder_context,
            decoder=decoder_context,
        )
        return None

    monkeypatch.setattr(
        StreamInferencePipeline,
        "initialize_cache",
        capture_initialize_cache,
    )
    pipeline = object.__new__(SwiftVRPipeline)
    torch.nn.Module.__init__(pipeline)
    pipeline.encoder = cast(Any, SimpleNamespace(spatial_compression_ratio=16))
    pipeline.register_buffer("prompt_embedding", torch.zeros(1), persistent=False)

    pipeline.initialize_cache(output_height=65, output_width=97, overlap=1)

    transformer_context = captured["transformer"]
    assert transformer_context["height"] == 6
    assert transformer_context["width"] == 8
    assert transformer_context["prompt_embedding"] is pipeline.prompt_embedding
    assert transformer_context["overlap"] == 1


def test_reae_checkpoint_maps_bijectively_to_shared_taehv_components() -> None:
    encoder = SwiftVREncoder(SwiftVREncoderConfig(dtype=torch.float32)).network
    decoder = SwiftVRTAEHV(None)
    stock_decoder = TAEHV(
        checkpoint_path=None,
        model_type="wan22",
        channels=(512, 256, 128, 64),
        use_cuda_graph=False,
    )
    model = {
        **{f"encoder.{key}": value for key, value in encoder.state_dict().items()},
        **decoder.state_dict(),
    }
    checkpoint = {
        key.replace(".blocks.", ".", 1): value for key, value in model.items()
    }
    transformed = legacy_to_blocks_keys(checkpoint)
    encoder_state = {
        key.removeprefix("encoder."): value
        for key, value in transformed.items()
        if key.startswith("encoder.")
    }

    assert len(checkpoint) == len(transformed) == 128
    assert len(encoder_state) == 64
    assert type(encoder) is TAEHVEncoder
    assert set(encoder_state) == set(encoder.state_dict())
    assert set(transformed) == set(model)
    assert all(model[key].shape == transformed[key].shape for key in transformed)
    assert sum(isinstance(block, MemBlock) for block in encoder.blocks) == 9
    assert sum(isinstance(block, TPool) for block in encoder.blocks) == 3
    assert (
        sum(isinstance(block, SwiftVRTemporalGrow) for block in decoder.decoder.blocks)
        == 3
    )
    assert not any(isinstance(block, TGrow) for block in decoder.decoder.blocks)
    assert [
        index
        for index, (stock, candidate) in enumerate(
            zip(stock_decoder.decoder.blocks, decoder.decoder.blocks, strict=True)
        )
        if type(stock) is not type(candidate)
    ] == [7, 13, 19]
    assert tuple(model["decoder.blocks.7.proj.weight"].shape) == (512, 512, 1, 1)
    assert tuple(model["decoder.blocks.13.conv3d.weight"].shape) == (
        256,
        256,
        3,
        1,
        1,
    )
    assert tuple(model["decoder.blocks.19.conv3d.weight"].shape) == (
        128,
        128,
        3,
        1,
        1,
    )


def test_reae_complete_group_adapter_preserves_stream_boundaries() -> None:
    torch.manual_seed(0)
    network = TAEHVEncoder(48, 3, 2, torch.nn.ReLU(inplace=True)).eval()
    video = torch.randn(1, 8, 12, 16, 16)

    whole = _encode_complete_groups(network, video, {})
    state: dict[int, torch.Tensor] = {}
    chunked = torch.cat(
        [
            _encode_complete_groups(network, video[:, :4], state),
            _encode_complete_groups(network, video[:, 4:], state),
        ],
        dim=1,
    )

    assert len(state) == 9
    torch.testing.assert_close(chunked, whole)


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
        swiftvr = SwiftVRDiTNetwork(SwiftVRDiTNetworkConfig())

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
