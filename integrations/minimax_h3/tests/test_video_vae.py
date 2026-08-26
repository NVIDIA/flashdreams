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

"""CPU contracts for the native MiniMax H3 video autoencoder."""

from __future__ import annotations

import hashlib
from dataclasses import replace

import pytest
import torch
from minimax_h3.video_vae import (
    H3_VIDEO_VAE_CHECKPOINT,
    MiniMaxH3VideoVAE,
    MiniMaxH3VideoVAEConfig,
)

pytestmark = pytest.mark.ci_cpu

_RELEASED_STATE_SCHEMA_DIGEST = (
    "4a5bd9d5f1357a4ffe51213327e1d23da8d6544d88aaa827ae33dd0b7e825460"
)


def _tiny_config(**changes: object) -> MiniMaxH3VideoVAEConfig:
    config = MiniMaxH3VideoVAEConfig(
        checkpoint_path=None,
        latent_channels=4,
        block_out_channels=(8, 16),
        layers_per_block=1,
        spatial_downsample_factors=(2, 2),
        temporal_downsample_factors=(2, 2),
        norm_num_groups=8,
        decoder_num_layers=2,
        decoder_num_attention_heads=2,
        decoder_attention_head_dim=8,
        decoder_num_register_tokens=2,
        decoder_ffn_mult=2,
        latents_mean=(0.0,) * 4,
        latents_std=(1.0,) * 4,
    )
    return replace(config, **changes)


@pytest.fixture(scope="module")
def tiny_video_vae() -> MiniMaxH3VideoVAE:
    """Construct the pinned oracle's reduced video architecture once."""
    torch.manual_seed(23)
    return _tiny_config().setup()


def test_released_video_vae_config_and_checkpoint_schema() -> None:
    """Match the immutable H3 shard index and every released tensor entry."""
    config = MiniMaxH3VideoVAEConfig(device="meta", checkpoint_path=None)
    model = config.setup()
    records = []
    parameter_count = 0
    for key, value in sorted(model.state_dict().items()):
        assert value.dtype == torch.float32
        shape = ",".join(str(dimension) for dimension in value.shape)
        records.append(f"{key}:float32:{shape}")
        parameter_count += value.numel()

    assert H3_VIDEO_VAE_CHECKPOINT.endswith(
        "42ed227ee7df40d41602854ae760620d6eb651fe/"
        "vae/diffusion_pytorch_model.safetensors.index.json"
    )
    assert model.spatial_compression_ratio == 16
    assert model.temporal_compression_ratio == 4
    assert (model.tokens_chunk_size, model.token_overlap, model.frame_overlap) == (
        5,
        2,
        5,
    )
    assert len(records) == 703
    assert parameter_count == 2_603_868_984
    assert hashlib.sha256("\n".join(records).encode()).hexdigest() == (
        _RELEASED_STATE_SCHEMA_DIGEST
    )


def test_video_vae_round_trip_preserves_released_temporal_geometry(
    tiny_video_vae: MiniMaxH3VideoVAE,
) -> None:
    """Map 22 pixel frames to seven latents and back to 22 frames."""
    sample = torch.linspace(-1.0, 1.0, 1 * 3 * 22 * 8 * 8, dtype=torch.float32).reshape(
        1, 3, 22, 8, 8
    )

    posterior = tiny_video_vae.encode(sample).latent_dist
    decoded = tiny_video_vae.decode(posterior.mode())

    assert posterior.mean.shape == posterior.logvar.shape == (1, 4, 7, 2, 2)
    assert decoded.shape == sample.shape
    assert decoded.dtype == torch.float32
    assert bool(torch.isfinite(decoded).all())


def test_single_conditioning_frame_encodes_to_one_latent_frame(
    tiny_video_vae: MiniMaxH3VideoVAE,
) -> None:
    """Keep keyframe conditioning on the encoder's single-frame path."""
    sample = torch.zeros(1, 3, 1, 8, 8)
    posterior = tiny_video_vae.encode(sample).latent_dist
    assert posterior.mode().shape == (1, 4, 1, 2, 2)


def test_tiled_video_round_trip_preserves_canvas_geometry() -> None:
    """Blend overlapping latent-aligned tiles back to the requested canvas."""
    model = _tiny_config(
        tile_sample_min_height=8,
        tile_sample_min_width=8,
        tile_sample_min_overlap_height=4,
        tile_sample_min_overlap_width=4,
    ).setup()
    sample = torch.linspace(
        -0.5, 0.5, 1 * 3 * 22 * 12 * 12, dtype=torch.float32
    ).reshape(1, 3, 22, 12, 12)

    posterior = model.encode(sample).latent_dist
    decoded = model.decode(posterior.mode())

    assert posterior.mode().shape == (1, 4, 7, 3, 3)
    assert decoded.shape == sample.shape
    assert bool(torch.isfinite(decoded).all())


def test_released_tile_layout_covers_canvas_on_latent_boundaries() -> None:
    """Distribute released tile slack without breaking 16-pixel alignment."""
    model = MiniMaxH3VideoVAEConfig(device="meta", checkpoint_path=None).setup()
    starts, lengths, overlaps = model._split_tiles(768, 256, 64)
    assert starts == [0, 160, 336, 512]
    assert lengths == [256, 256, 256, 256]
    assert overlaps == [96, 80, 80]
    assert all(value % 16 == 0 for value in (*starts, *overlaps))


def test_pixel_adapters_apply_released_normalization_and_base_range(
    tiny_video_vae: MiniMaxH3VideoVAE,
) -> None:
    """Encode base-range pixels and decode normalized latents back to RGB."""
    pixels = torch.linspace(0.0, 1.0, 1 * 3 * 22 * 8 * 8, dtype=torch.float32).reshape(
        1, 3, 22, 8, 8
    )

    normalized_latents = tiny_video_vae.encode_pixels(pixels)
    output = tiny_video_vae.decode_output(normalized_latents)

    assert normalized_latents.shape == (1, 4, 7, 2, 2)
    assert output.shape == pixels.shape
    assert output.dtype == torch.float32
    assert bool(torch.isfinite(output).all())
    assert bool((output >= 0.0).all())
    assert bool((output <= 1.0).all())


def test_conditioning_adapter_samples_seed_42_and_rounds_before_normalizing(
    tiny_video_vae: MiniMaxH3VideoVAE,
) -> None:
    """Match H3's independent seeded posterior and deliberate FP16 rounding."""
    pixels = torch.linspace(0.0, 1.0, 1 * 3 * 1 * 8 * 8, dtype=torch.float32).reshape(
        1, 3, 1, 8, 8
    )

    actual = tiny_video_vae.encode_condition_pixels(pixels)
    repeated = tiny_video_vae.encode_condition_pixels(pixels)
    posterior = tiny_video_vae._encode_pixel_posterior(pixels)
    expected = posterior.sample(generator=torch.Generator(device="cpu").manual_seed(42))
    expected = tiny_video_vae.normalize_latents(
        expected.to(torch.float16).float().cpu()
    )

    assert actual.device.type == "cpu"
    assert actual.dtype == torch.float32
    assert torch.equal(actual, repeated)
    assert torch.equal(actual, expected)
    with pytest.raises(ValueError, match="non-negative"):
        tiny_video_vae.encode_condition_pixels(pixels, seed=-1)


def test_video_latent_normalization_is_per_channel() -> None:
    """Round-trip distinct mean and scale values for every latent channel."""
    model = _tiny_config(
        latents_mean=(0.0, 1.0, 2.0, 3.0),
        latents_std=(1.0, 2.0, 3.0, 4.0),
    ).setup()
    latents = torch.arange(4, dtype=torch.float32).view(1, 4, 1, 1, 1)

    normalized = model.normalize_latents(latents)
    restored = model.denormalize_latents(normalized)

    torch.testing.assert_close(normalized, torch.zeros_like(normalized))
    torch.testing.assert_close(restored, latents)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"temporal_downsample_factors": (2,)}, "temporal downsample"),
        ({"decoder_rope_dim_ratio": 0.5}, "rotary width"),
        ({"token_drop": 5}, "latent chunk size"),
        ({"tile_sample_min_width": 255}, "align to the spatial ratio"),
        ({"tile_sample_min_overlap_width": 256}, "smaller than tile sizes"),
    ],
)
def test_video_vae_rejects_invalid_configuration(
    changes: dict[str, object], message: str
) -> None:
    """Reject geometry variants that cannot preserve released behavior."""
    with pytest.raises(ValueError, match=message):
        _tiny_config(**changes).setup()


def test_video_vae_rejects_invalid_pixels_and_downcast_weights() -> None:
    """Stop invalid input values and keep all VAE weights in FP32."""
    model = _tiny_config().setup()
    with pytest.raises(ValueError, match="within"):
        model.encode_pixels(torch.full((1, 3, 1, 8, 8), 1.1))
    with pytest.raises(ValueError, match="finite"):
        model.encode(torch.full((1, 3, 1, 8, 8), float("nan")))

    model.to(dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="must remain float32"):
        model.decode(torch.zeros(1, 4, 7, 2, 2))
