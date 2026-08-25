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

"""CUDA parity tests for the native MiniMax H3 video autoencoder."""

from __future__ import annotations

from typing import Any

import pytest
import torch
from minimax_h3.video_vae import MiniMaxH3VideoVAEConfig

pytestmark = pytest.mark.ci_gpu


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_native_video_vae_matches_official_h3_forward() -> None:
    """Match pinned Diffusers encode and decode with strict checkpoint keys."""
    oracle = pytest.importorskip(
        "diffusers.models.autoencoders.autoencoder_kl_minimax_h3",
        reason="exact pinned Diffusers H3 oracle is not installed",
    )
    architecture: dict[str, Any] = {
        "in_channels": 3,
        "out_channels": 3,
        "latent_channels": 4,
        "block_out_channels": (8, 16),
        "layers_per_block": 1,
        "spatial_downsample_factors": (2, 2),
        "temporal_downsample_factors": (2, 2),
        "norm_num_groups": 8,
        "decoder_num_layers": 2,
        "decoder_num_attention_heads": 2,
        "decoder_attention_head_dim": 8,
        "decoder_num_register_tokens": 2,
        "decoder_ffn_mult": 2,
        "clip_length": 17,
        "token_drop": 3,
        "latents_mean": (0.0,) * 4,
        "latents_std": (1.0,) * 4,
    }
    device = torch.device("cuda")
    torch.manual_seed(23)
    official: Any = oracle.AutoencoderKLMiniMaxH3(**architecture).to(device).eval()
    native = MiniMaxH3VideoVAEConfig(
        checkpoint_path=None,
        device="cuda",
        **architecture,
    ).setup()
    native.load_state_dict(official.state_dict(), strict=True)
    sample = torch.randn(1, 3, 22, 8, 8, device=device)
    latents = torch.randn(1, 4, 7, 2, 2, device=device)

    with (
        torch.inference_mode(),
        torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH),
    ):
        expected_posterior = official.encode(sample).latent_dist
        actual_posterior = native.encode(sample).latent_dist
        expected_decoded = official.decode(latents).sample
        actual_decoded = native.decode(latents)

    torch.testing.assert_close(actual_posterior.mean, expected_posterior.mean)
    torch.testing.assert_close(actual_posterior.logvar, expected_posterior.logvar)
    torch.testing.assert_close(actual_decoded, expected_decoded)
