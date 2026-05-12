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

"""Shape + invariance tests for the opacity / camera-ray patchification.

CPU-only, torch + einops are the only deps.
"""

from __future__ import annotations

import torch
from artifixer.network.dit import artifixer_embedding_dims
from artifixer.network.patches import patchify_camera_rays, patchify_opacity

VAE_T = 4
VAE_S = 8
PATCH = (1, 2, 2)


def _expected_L(t_latent: int, h: int, w: int, patch: tuple[int, int, int]) -> int:
    kt, kh, kw = patch
    assert t_latent % kt == 0 and h % (VAE_S * kh) == 0 and w % (VAE_S * kw) == 0
    return (t_latent // kt) * (h // (VAE_S * kh)) * (w // (VAE_S * kw))


def test_patchify_opacity_first_chunk_shape() -> None:
    """First AR chunk pads the first frame; output token count matches the
    expected post-patch latent token count.
    """
    # 1 latent frame for the first chunk: 1 + 3 padding -> 4 input frames
    # covers 1 latent frame (VAE_T = 4). Spatial: 32x32 input -> 2x2 latent
    # tokens after VAE (32 / 8 = 4) and patch (4 / 2 = 2).
    opacity = torch.randn(2, 1, 32, 32)  # (B, T, H, W) at INPUT resolution
    out = patchify_opacity(
        opacity,
        vae_scale_factor_temporal=VAE_T,
        vae_scale_factor_spatial=VAE_S,
        patch_size=PATCH,
        frame_offset=0,
    )
    opacity_dim, _ = artifixer_embedding_dims(PATCH)
    assert out.shape == (2, _expected_L(1, 32, 32, PATCH), opacity_dim)


def test_patchify_opacity_later_chunk_no_pad() -> None:
    """Non-zero frame_offset does not pad: input has already-paired frames."""
    # 4 input frames -> 1 latent frame (VAE_T = 4)
    opacity = torch.randn(2, 4, 32, 32)
    out = patchify_opacity(
        opacity,
        vae_scale_factor_temporal=VAE_T,
        vae_scale_factor_spatial=VAE_S,
        patch_size=PATCH,
        frame_offset=7,  # any non-zero value
    )
    opacity_dim, _ = artifixer_embedding_dims(PATCH)
    assert out.shape == (2, _expected_L(1, 32, 32, PATCH), opacity_dim)


def test_patchify_camera_rays_latent_rate_shape() -> None:
    """Plucker rays already at the latent temporal rate skip the pad/regroup."""
    # camera_rays.shape[1] == hidden_post_patch_t (1 latent frame)
    rays = torch.randn(2, 1, 32, 32, 6)
    out = patchify_camera_rays(
        rays,
        hidden_post_patch_t=1,
        vae_scale_factor_temporal=VAE_T,
        vae_scale_factor_spatial=VAE_S,
        patch_size=PATCH,
        frame_offset=0,
    )
    _, camera_dim = artifixer_embedding_dims(PATCH)
    assert out.shape == (2, _expected_L(1, 32, 32, PATCH), camera_dim)


def test_patchify_camera_rays_input_rate_first_chunk_shape() -> None:
    """Input-rate Plucker rays on the first AR chunk get the 1+3 left-pad.

    This branch in dreamfix (transformer.py L330-L340) groups t4 input frames
    per latent frame, so the per-token feature carries an extra ``vae_t``
    multiplier vs the latent-rate branch:

      per_token_dim_input_rate = vae_t * camera_dim_latent_rate

    At inference, camera_rays is always already at the latent rate (see
    ``model_training/pipeline/kv_cache_pipeline.py`` L213), so this branch
    is unused. We still cover it for parity with the dreamfix forward.
    """
    rays = torch.randn(2, 1, 32, 32, 6)  # 1 input frame
    out = patchify_camera_rays(
        rays,
        hidden_post_patch_t=999,  # mismatch -> not latent-rate
        vae_scale_factor_temporal=VAE_T,
        vae_scale_factor_spatial=VAE_S,
        patch_size=PATCH,
        frame_offset=0,
    )
    _, camera_dim = artifixer_embedding_dims(PATCH)
    expected_dim = VAE_T * camera_dim
    assert out.shape == (2, _expected_L(1, 32, 32, PATCH), expected_dim)


def test_patchify_opacity_constant_input_constant_output() -> None:
    """A constant opacity field produces a constant per-token feature.

    Invariant: any rearrangement of the same scalar must still be scalar
    along the token axis.
    """
    opacity = torch.full((1, 4, 16, 16), 0.42)
    out = patchify_opacity(
        opacity,
        vae_scale_factor_temporal=VAE_T,
        vae_scale_factor_spatial=VAE_S,
        patch_size=PATCH,
        frame_offset=7,
    )
    assert torch.allclose(out, torch.full_like(out, 0.42))
