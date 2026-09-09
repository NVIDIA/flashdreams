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

"""Small CPU checks for native H3 codec geometry, precision and checkpoint keys."""

from dataclasses import asdict
import importlib
import json
import os
from pathlib import Path
import struct
import subprocess
import sys

import pytest
import torch

from minimax_h3.impl.audio_encoder import AudioEncoderConfig, MiniMaxH3AudioEncoder
from minimax_h3.impl.video_vae import (
    MiniMaxH3VideoDecoder,
    MiniMaxH3VideoEncoder,
    VideoVAEConfig,
)

pytestmark = pytest.mark.ci_cpu


def _video_config() -> VideoVAEConfig:
    return VideoVAEConfig(
        latent_channels=2,
        block_out_channels=(4, 4, 4, 4),
        layers_per_block=1,
        spatial_downsample_factors=(2, 2, 2, 2),
        temporal_downsample_factors=(1, 2, 2, 1),
        norm_num_groups=1,
        decoder_num_layers=1,
        decoder_num_attention_heads=2,
        decoder_attention_head_dim=8,
        decoder_rope_dim_ratio=0.75,
        decoder_ffn_mult=2,
        latents_mean=(0.0, 0.0),
        latents_std=(1.0, 1.0),
    )


def _audio_config() -> AudioEncoderConfig:
    return AudioEncoderConfig(
        encoder_dim=2,
        encoder_rates=(2, 2),
        latent_dim=8,
        latent_channels=2,
        num_attention_heads=2,
        latents_mean=(0.0, 0.0),
        latents_std=(1.0, 1.0),
    )


def test_independent_codec_subtrees_and_native_imports() -> None:
    """Keep unused weights and Diffusers outside the native codec imports."""
    video = _video_config()
    assert {key.split(".")[0] for key in MiniMaxH3VideoEncoder(video).state_dict()} == {
        "encoder",
        "quant_conv",
    }
    assert {key.split(".")[0] for key in MiniMaxH3VideoDecoder(video).state_dict()} == {
        "decoder",
        "post_quant_conv",
    }
    assert {
        key.split(".")[0] for key in MiniMaxH3AudioEncoder(_audio_config()).state_dict()
    } == {"encoder", "pre_block", "mean_proj"}
    code = """
import sys
class NoDiffusers:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'diffusers' or fullname.startswith('diffusers.'):
            raise AssertionError('native codec attempted to import ' + fullname)
sys.meta_path.insert(0, NoDiffusers())
import minimax_h3.impl.video_vae
import minimax_h3.impl.audio_encoder
"""
    subprocess.run([sys.executable, "-c", code], check=True, timeout=120)


def test_sampling_seed_and_single_frame_geometry() -> None:
    """Preserve single-image encoding and CPU-generator posterior arithmetic."""
    torch.set_num_threads(1)
    model = MiniMaxH3VideoEncoder(_video_config()).eval()
    pixels = torch.randn(1, 3, 1, 32, 32)
    with torch.no_grad():
        moments = model.encode(pixels)
        mean, logvar = moments.chunk(2, dim=1)
        expected = mean + torch.exp(0.5 * logvar.clamp(-30, 20)) * torch.randn(
            mean.shape, generator=torch.Generator().manual_seed(42)
        )
        actual = model.sample(pixels, generator=torch.Generator().manual_seed(42))
    assert actual.shape == (1, 2, 1, 2, 2)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    rounded = actual.half().float()
    assert rounded.dtype == torch.float32


@pytest.mark.parametrize("latent_frames,pixel_frames", [(7, 22), (8, 26), (12, 39)])
def test_temporal_decode_tail_and_tiling(latent_frames: int, pixel_frames: int) -> None:
    """Preserve overlapped temporal geometry, including padded partial tails."""
    torch.set_num_threads(1)
    model = MiniMaxH3VideoDecoder(_video_config()).eval()
    model.tile_sample_min_height = model.tile_sample_min_width = 32
    model.tile_sample_min_overlap_height = model.tile_sample_min_overlap_width = 16
    with torch.no_grad():
        pixels = model.decode(torch.randn(1, 2, latent_frames, 3, 3))
    assert pixels.shape == (1, 3, pixel_frames, 48, 48)
    assert pixels.dtype == torch.float32 and torch.isfinite(pixels).all()
    starts, sizes, overlaps = model._split_tiles(80, 32, 16)
    assert starts[-1] + sizes[-1] == 80
    assert all(overlap >= 16 and overlap % 16 == 0 for overlap in overlaps)


def test_audio_mono_batches_padding_and_precision() -> None:
    """Encode stereo-as-batch references and reject downcast checkpoint weights."""
    torch.set_num_threads(1)
    model = MiniMaxH3AudioEncoder(_audio_config()).eval()
    with torch.no_grad():
        result = model.encode(torch.randn(2, 1, 15))
    assert result.shape == (2, 2, 4)
    assert result.dtype == torch.float32 and torch.isfinite(result).all()
    with pytest.raises(ValueError, match="mono waveform"):
        model.encode(torch.zeros(1, 2, 10))
    with pytest.raises(ValueError, match="float32"):
        model.half().encode(torch.zeros(1, 1, 16))


def test_config_metadata_and_invalid_tiles() -> None:
    """Accept released metadata while rejecting architecture typos and bad tiles."""
    assert VideoVAEConfig.from_dict({"_class_name": "test"}) == VideoVAEConfig()
    assert AudioEncoderConfig.from_dict({"decoder_dim": 1024}) == AudioEncoderConfig()
    with pytest.raises(ValueError, match="Unknown"):
        VideoVAEConfig.from_dict({"latent_channel": 24})
    model = MiniMaxH3VideoDecoder(_video_config())
    with pytest.raises(ValueError, match="overlap"):
        model._split_tiles(64, 32, 32)
    with pytest.raises(ValueError, match="align"):
        model._split_tiles(65, 32, 16)


def test_optional_cached_checkpoint_bijection() -> None:
    """Compare native meta tensors with cached headers, without reading weights."""
    root = os.environ.get("MINIMAX_H3_MODEL_PATH")
    if root is None:
        pytest.skip("Set MINIMAX_H3_MODEL_PATH to an already cached model snapshot")
    for component, cls, config_cls, prefixes in (
        ("vae", MiniMaxH3VideoEncoder, VideoVAEConfig, ("encoder.", "quant_conv.")),
        (
            "vae",
            MiniMaxH3VideoDecoder,
            VideoVAEConfig,
            ("decoder.", "post_quant_conv."),
        ),
        (
            "audio_vae",
            MiniMaxH3AudioEncoder,
            AudioEncoderConfig,
            ("encoder.", "pre_block.", "mean_proj."),
        ),
    ):
        folder = Path(root) / component
        values = json.loads((folder / "config.json").read_text())
        with torch.device("meta"):
            model = cls(config_cls.from_dict(values))
        expected = {}
        for path in folder.glob("*.safetensors"):
            with path.open("rb") as stream:
                size = struct.unpack("<Q", stream.read(8))[0]
                header = json.loads(stream.read(size))
            expected.update(
                {
                    key: tuple(value["shape"])
                    for key, value in header.items()
                    if key.startswith(prefixes)
                }
            )
        assert expected, f"No cached safetensors found in {folder}"
        assert {
            key: tuple(value.shape) for key, value in model.state_dict().items()
        } == expected


def test_optional_video_reference_parity() -> None:
    """Compare reduced-size codec math to an optional installed reference."""
    pytest.importorskip("diffusers")
    try:
        module = importlib.import_module(
            "diffusers.models.autoencoders.autoencoder_kl_minimax_h3"
        )
    except ModuleNotFoundError:
        pytest.skip("Installed Diffusers has no MiniMax H3 reference codec")
    torch.set_num_threads(1)
    config = _video_config()
    reference = module.AutoencoderKLMiniMaxH3(**asdict(config)).eval()
    # Random init uses zero residual scales; exercise attention and feed-forward.
    with torch.no_grad():
        for block in reference.decoder.transformer_blocks:
            block.scale1.fill_(0.1)
            block.scale2.fill_(0.1)
    encoder, decoder = (
        MiniMaxH3VideoEncoder(config).eval(),
        MiniMaxH3VideoDecoder(config).eval(),
    )
    for model, prefixes in (
        (encoder, ("encoder.", "quant_conv.")),
        (decoder, ("decoder.", "post_quant_conv.")),
    ):
        model.load_state_dict(
            {
                key: value
                for key, value in reference.state_dict().items()
                if key.startswith(prefixes)
            },
            strict=True,
        )
    with torch.no_grad():
        for count in (1, 22):
            x = torch.randn(1, 3, count, 32, 32)
            posterior = reference.encode(x).latent_dist
            torch.testing.assert_close(
                encoder.encode(x), posterior.parameters, rtol=1e-5, atol=1e-6
            )
            torch.testing.assert_close(
                encoder.sample(x, generator=torch.Generator().manual_seed(42)),
                posterior.sample(generator=torch.Generator().manual_seed(42)),
                rtol=1e-5,
                atol=1e-6,
            )
        for model in (reference, decoder):
            model.tile_sample_min_height = model.tile_sample_min_width = 32
            model.tile_sample_min_overlap_height = (
                model.tile_sample_min_overlap_width
            ) = 16
        for count in (7, 8, 12):
            z = torch.randn(1, 2, count, 3, 3)
            torch.testing.assert_close(
                decoder.decode(z), reference.decode(z).sample, rtol=1e-5, atol=1e-6
            )


def test_optional_audio_reference_parity() -> None:
    """Compare the native encoder and causal projection with the reference mean."""
    pytest.importorskip("diffusers")
    try:
        module = importlib.import_module(
            "diffusers.models.autoencoders.autoencoder_kl_minimax_h3_audio"
        )
    except ModuleNotFoundError:
        pytest.skip("Installed Diffusers has no MiniMax H3 reference audio codec")
    torch.set_num_threads(1)
    config = _audio_config()
    reference = module.AutoencoderKLMiniMaxH3Audio(
        **asdict(config),
        decoder_dim=8,
        decoder_rates=(2, 2),
        decoder_kernel_sizes=(4, 4),
        resblock_kernel_sizes=(3,),
        resblock_dilation_sizes=((1, 3, 5),),
    ).eval()
    model = MiniMaxH3AudioEncoder(config).eval()
    prefixes = ("encoder.", "pre_block.", "mean_proj.")
    model.load_state_dict(
        {
            key: value
            for key, value in reference.state_dict().items()
            if key.startswith(prefixes)
        },
        strict=True,
    )
    with torch.no_grad():
        for length in (15, 16, 17):
            waveform = torch.randn(2, 1, length)
            torch.testing.assert_close(
                model.encode(waveform),
                reference.encode(waveform).latent_dist.mode(),
                rtol=1e-5,
                atol=1e-6,
            )
