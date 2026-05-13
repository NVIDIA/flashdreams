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

"""Shape + invariance tests for the opacity-weighted latent mix.

CPU-only; only ``torch`` is required.
"""

from __future__ import annotations

import torch
from artifixer.latent_mix import opacity_weighted_latent_mix

VAE_T = 4
VAE_S = 8


def _condition_and_noise(batch=1, in_dim=16, t_lat=2, hl=4, wl=4):
    """Build matching condition / noise tensors at the latent grid."""
    cond = torch.randn(batch, in_dim, t_lat, hl, wl)
    noise = torch.randn(batch, in_dim, t_lat, hl, wl)
    return cond, noise


def test_first_chunk_opacity_one_returns_condition_exactly() -> None:
    """opacity = 1 everywhere -> latent_mix = condition (noise is dropped)."""
    cond, noise = _condition_and_noise(t_lat=2, hl=4, wl=4)
    # First chunk uses ``1 + 4 * (t_lat - 1)`` input frames after the 1+3 pad
    # collapses to t_lat = 2 latent frames.
    t_input_first = 1 + VAE_T * (cond.shape[2] - 1)
    opacity = torch.ones(1, t_input_first, VAE_S * 4, VAE_S * 4)

    out = opacity_weighted_latent_mix(
        condition=cond,
        opacity=opacity,
        noise=noise,
        vae_scale_factor_temporal=VAE_T,
        vae_scale_factor_spatial=VAE_S,
        is_first_chunk=True,
    )
    torch.testing.assert_close(out, cond)


def test_first_chunk_opacity_zero_returns_noise_exactly() -> None:
    """opacity = 0 everywhere -> latent_mix = noise (condition is dropped)."""
    cond, noise = _condition_and_noise(t_lat=2, hl=4, wl=4)
    t_input_first = 1 + VAE_T * (cond.shape[2] - 1)
    opacity = torch.zeros(1, t_input_first, VAE_S * 4, VAE_S * 4)

    out = opacity_weighted_latent_mix(
        condition=cond,
        opacity=opacity,
        noise=noise,
        vae_scale_factor_temporal=VAE_T,
        vae_scale_factor_spatial=VAE_S,
        is_first_chunk=True,
    )
    torch.testing.assert_close(out, noise)


def test_later_chunk_shape_no_pad() -> None:
    """Non-first chunk has ``vae_t * t_lat`` input frames already."""
    cond, noise = _condition_and_noise(batch=2, t_lat=2, hl=4, wl=4)
    t_input = VAE_T * cond.shape[2]
    opacity = torch.full((2, t_input, VAE_S * 4, VAE_S * 4), 0.5)

    out = opacity_weighted_latent_mix(
        condition=cond,
        opacity=opacity,
        noise=noise,
        vae_scale_factor_temporal=VAE_T,
        vae_scale_factor_spatial=VAE_S,
        is_first_chunk=False,
    )
    # opacity_lat = 0.5 everywhere -> 0.5 * cond + 0.5 * noise.
    torch.testing.assert_close(out, 0.5 * cond + 0.5 * noise)


def test_first_chunk_handles_single_latent_frame() -> None:
    """Edge case: t_lat = 1, which requires the 1+3 left-pad for vae_t=4."""
    cond, noise = _condition_and_noise(t_lat=1, hl=4, wl=4)
    opacity = torch.full((1, 1, VAE_S * 4, VAE_S * 4), 0.7)

    out = opacity_weighted_latent_mix(
        condition=cond,
        opacity=opacity,
        noise=noise,
        vae_scale_factor_temporal=VAE_T,
        vae_scale_factor_spatial=VAE_S,
        is_first_chunk=True,
    )
    # After 1+3 pad: opacity is shape (1, 4, ...) all 0.7 (pad uses the
    # repeated first frame which is 0.7). max_pool3d -> (1, 1, 1, hl, wl)
    # all 0.7. Mix = 0.7 * cond + 0.3 * noise.
    torch.testing.assert_close(out, 0.7 * cond + 0.3 * noise)


def test_max_pool_picks_largest_in_window() -> None:
    """max_pool3d uses the MAX, not the average -- regression check."""
    cond, noise = _condition_and_noise(t_lat=1, hl=1, wl=1)
    # 1 + 4 * 0 = 1 input frame. Pad to 4. Make first 3 pad-frames 0.0 and
    # the last (original) frame 1.0 by overriding the source frame.
    opacity = torch.zeros(1, 1, VAE_S, VAE_S)
    opacity[..., 0, 0] = 1.0  # one pixel == 1 in the 8x8 input window

    out = opacity_weighted_latent_mix(
        condition=cond,
        opacity=opacity,
        noise=noise,
        vae_scale_factor_temporal=VAE_T,
        vae_scale_factor_spatial=VAE_S,
        is_first_chunk=True,
    )
    # The padded stack is 4 copies of the same 8x8 input. max_pool3d takes
    # the max -- a single 1.0 pixel survives, so opacity_lat = 1.0 at the
    # single latent pixel, mix = cond.
    torch.testing.assert_close(out, cond)
