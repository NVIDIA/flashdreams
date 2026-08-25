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

"""CPU contract tests for the native MiniMax H3 waveform autoencoder."""

from __future__ import annotations

import hashlib
from dataclasses import replace

import pytest
import torch
from minimax_h3.audio_vae import (
    H3_AUDIO_VAE_CHECKPOINT,
    MiniMaxH3AudioVAE,
    MiniMaxH3AudioVAEConfig,
)

pytestmark = pytest.mark.ci_cpu

_RELEASED_STATE_SCHEMA_DIGEST = (
    "085f7e6f63160281a619e73aa0da778faa5259a25cb5e23c458bb1dd48c5de3a"
)


def _tiny_config(**changes: object) -> MiniMaxH3AudioVAEConfig:
    config = MiniMaxH3AudioVAEConfig(
        checkpoint_path=None,
        encoder_dim=4,
        encoder_rates=(2, 2),
        latent_dim=32,
        latent_channels=8,
        num_attention_heads=2,
        decoder_dim=16,
        decoder_rates=(2, 2),
        decoder_kernel_sizes=(4, 4),
        resblock_kernel_sizes=(3, 7),
        resblock_dilation_sizes=((1, 3), (1, 3)),
        latents_mean=(0.0,) * 8,
        latents_std=(1.0,) * 8,
    )
    return replace(config, **changes)


@pytest.fixture(scope="module")
def tiny_audio_vae() -> MiniMaxH3AudioVAE:
    """Construct the pinned oracle's reduced audio architecture once."""
    torch.manual_seed(19)
    return _tiny_config().setup()


def test_released_audio_vae_config_and_checkpoint_schema() -> None:
    """Match the immutable H3 audio artifact and all released tensor entries."""
    config = MiniMaxH3AudioVAEConfig(device="meta", checkpoint_path=None)
    model = config.setup()
    records = []
    for key, value in sorted(model.state_dict().items()):
        assert value.dtype == torch.float32
        shape = ",".join(str(dimension) for dimension in value.shape)
        records.append(f"{key}:F32:{shape}")

    assert H3_AUDIO_VAE_CHECKPOINT.endswith(
        "42ed227ee7df40d41602854ae760620d6eb651fe/"
        "audio_vae/diffusion_pytorch_model.safetensors"
    )
    assert config.sampling_rate == 32_000
    assert model.hop_length == 800
    assert len(config.latents_mean) == len(config.latents_std) == 32
    assert len(records) == 1_087
    assert hashlib.sha256("\n".join(records).encode()).hexdigest() == (
        _RELEASED_STATE_SCHEMA_DIGEST
    )


def test_audio_vae_pads_encode_and_decodes_complete_hops(
    tiny_audio_vae: MiniMaxH3AudioVAE,
) -> None:
    """Right-pad an unaligned waveform and decode the resulting complete hops."""
    sample = torch.linspace(-0.9, 0.9, 66, dtype=torch.float32).reshape(2, 1, 33)
    posterior = tiny_audio_vae.encode(sample).latent_dist
    decoded = tiny_audio_vae.decode(posterior.mode())

    assert posterior.mean.shape == posterior.logs.shape == (2, 8, 9)
    assert decoded.shape == (2, 1, 36)
    assert decoded.dtype == torch.float32
    assert bool(torch.isfinite(decoded).all())
    assert bool((decoded >= -1.0).all())
    assert bool((decoded <= 1.0).all())


def test_reference_audio_adapter_preserves_channel_major_oracle_layout(
    tiny_audio_vae: MiniMaxH3AudioVAE,
) -> None:
    """Flatten two encoded channels in the exact conditioning row order."""
    samples = torch.linspace(-0.5, 0.5, 64, dtype=torch.float32).reshape(2, 32)
    mode = tiny_audio_vae.encode(samples[:, None]).latent_dist.mode()

    actual = tiny_audio_vae.encode_condition(samples)

    assert actual.shape == (16, 8)
    torch.testing.assert_close(actual, mode.cpu().transpose(1, 2).reshape(16, 8))
    assert actual.is_contiguous()
    assert actual.dtype == torch.float32
    assert actual.device.type == "cpu"


def test_generated_audio_adapter_denormalizes_and_returns_typed_stereo(
    tiny_audio_vae: MiniMaxH3AudioVAE,
) -> None:
    """Decode normalized channel-batch latents into typed stereo PCM."""
    latents = torch.linspace(-0.4, 0.4, 128, dtype=torch.float32).reshape(2, 8, 8)
    expected = tiny_audio_vae.decode(tiny_audio_vae.denormalize(latents))[:, 0]

    output = tiny_audio_vae.decode_output(latents)

    assert output.samples.shape == (2, 32)
    assert output.sample_rate == 32_000
    assert output.sample_offset == 0
    assert output.samples.is_contiguous()
    assert output.samples.dtype == torch.float32
    torch.testing.assert_close(output.samples, expected)


def test_audio_latent_denormalization_is_per_channel() -> None:
    """Apply distinct released-style mean and scale values per latent channel."""
    config = _tiny_config(
        latents_mean=tuple(float(index) for index in range(8)),
        latents_std=tuple(float(index + 1) for index in range(8)),
    )
    model = config.setup()
    latents = torch.ones(2, 8, 3)

    actual = model.denormalize(latents)

    expected_channels = torch.tensor([1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0]).view(1, 8, 1)
    torch.testing.assert_close(actual, expected_channels.expand(2, 8, 3))


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"decoder_rates": (2, 3)}, "hop length"),
        ({"sampling_rate": 44_100}, "32000 Hz"),
        ({"latents_std": (1.0,) * 7 + (0.0,)}, "standard deviations"),
        ({"decoder_kernel_sizes": (4,)}, "one kernel size"),
        ({"resblock_dilation_sizes": ((1, 3),)}, "one dilation schedule"),
    ],
)
def test_audio_vae_rejects_invalid_configuration(
    changes: dict[str, object], message: str
) -> None:
    """Reject architecture variants that cannot load or decode H3 weights."""
    with pytest.raises(ValueError, match=message):
        _tiny_config(**changes).setup()


def test_audio_vae_rejects_nonfinite_inputs_and_downcast_weights() -> None:
    """Keep the waveform decoder in FP32 and stop invalid values at its boundary."""
    model = _tiny_config().setup()
    with pytest.raises(ValueError, match="finite"):
        model.encode(torch.tensor([[[float("nan")]]]))
    with pytest.raises(ValueError, match="finite"):
        model.decode(torch.full((2, 8, 1), float("inf")))

    model.to(dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="must remain float32"):
        model.decode(torch.zeros(2, 8, 1))
