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

"""CUDA parity tests for the native MiniMax H3 waveform autoencoder."""

from __future__ import annotations

from typing import Any

import pytest
import torch
from minimax_h3.audio_vae import MiniMaxH3AudioVAEConfig

pytestmark = pytest.mark.ci_gpu


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_native_audio_vae_matches_official_h3_forward() -> None:
    """Match pinned Diffusers encode and decode with strict checkpoint keys."""
    oracle = pytest.importorskip(
        "diffusers.models.autoencoders.autoencoder_kl_minimax_h3_audio",
        reason="exact pinned Diffusers H3 oracle is not installed",
    )
    architecture: dict[str, Any] = {
        "encoder_dim": 4,
        "encoder_rates": (2, 2),
        "latent_dim": 32,
        "latent_channels": 8,
        "num_attention_heads": 2,
        "decoder_dim": 16,
        "decoder_rates": (2, 2),
        "decoder_kernel_sizes": (4, 4),
        "resblock_kernel_sizes": (3, 7),
        "resblock_dilation_sizes": ((1, 3), (1, 3)),
        "sampling_rate": 32_000,
        "latents_mean": (0.0,) * 8,
        "latents_std": (1.0,) * 8,
    }
    device = torch.device("cuda")
    torch.manual_seed(19)
    official: Any = oracle.AutoencoderKLMiniMaxH3Audio(**architecture).to(device).eval()
    native = MiniMaxH3AudioVAEConfig(
        checkpoint_path=None,
        device="cuda",
        **architecture,
    ).setup()
    native.load_state_dict(official.state_dict(), strict=True)
    sample = torch.randn(2, 1, 33, device=device)
    latents = torch.randn(2, 8, 9, device=device)

    with (
        torch.inference_mode(),
        torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH),
    ):
        expected_posterior = official.encode(sample).latent_dist
        actual_posterior = native.encode(sample).latent_dist
        expected_decoded = official.decode(latents).sample
        actual_decoded = native.decode(latents)

    torch.testing.assert_close(actual_posterior.mean, expected_posterior.mean)
    torch.testing.assert_close(actual_posterior.logs, expected_posterior.logs)
    torch.testing.assert_close(actual_decoded, expected_decoded)
