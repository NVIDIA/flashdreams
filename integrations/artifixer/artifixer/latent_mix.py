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

"""Opacity-weighted latent mixing of noise + VAE-encoded reconstruction.

Mirrors ``ArtifixerPipelineBase.prepare_latents`` at
``dreamfix/model_training/pipeline/pipeline_base.py`` L98-L122. The pipeline
replaces the base ``DiffusionModel.generate`` initial-noise sample with a
per-AR-chunk mix::

    latents = condition * opacity_lat + noise * (1 - opacity_lat)

where ``opacity_lat`` is the per-pixel opacity max-pooled from the input
resolution down to the VAE latent grid. The mix is the *only* place
opacity influences the inference path before the network forward: the
per-block ``opacity_embedding`` MLPs (Phase 2.1) feed the opacity into
attention through a separate channel.

The helper is shape-checked and self-contained -- callers supply the noise
tensor so the pipeline can keep its existing RNG / CP-broadcast logic.
"""

from __future__ import annotations

import torch
from torch import Tensor


def opacity_weighted_latent_mix(
    *,
    condition: Tensor,
    opacity: Tensor,
    noise: Tensor,
    vae_scale_factor_temporal: int,
    vae_scale_factor_spatial: int,
    is_first_chunk: bool,
) -> Tensor:
    """Compute the dreamfix opacity-weighted latent mix for one AR chunk.

    Args:
        condition: VAE-encoded reconstruction-rendered RGB, shape
            ``(B, in_dim, T_lat, Hl, Wl)``. Same layout the network consumes.
        opacity: Per-pixel alpha at input resolution, shape
            ``(B, T_input, H_input, W_input)``. The pipeline slices this
            from the full-rollout opacity to the current chunk's frame range.
        noise: Random noise tensor with the same shape as ``condition``.
        vae_scale_factor_temporal: Wan VAE temporal stride (e.g. ``4``).
        vae_scale_factor_spatial: Wan VAE spatial stride (e.g. ``8``).
        is_first_chunk: When ``True``, the first input frame is left-padded
            by 3 copies to absorb the Wan VAE's ``1 + 4`` temporal layout
            (one latent frame covers one input frame on the boundary, every
            other latent frame covers four). Mirrors the
            ``is_first_chunk`` branch in dreamfix.

    Returns:
        Mixed latent, same shape as ``condition``.
    """
    assert condition.shape == noise.shape, (
        f"condition.shape {tuple(condition.shape)} != noise.shape "
        f"{tuple(noise.shape)}"
    )

    if is_first_chunk:
        opacity = torch.cat(
            [opacity[:, :1].repeat_interleave(3, dim=1), opacity], dim=1
        )

    # max_pool3d expects (N, C, D, H, W); opacity is (B, T, H, W) -> add C=1.
    opacity_5d = opacity.unsqueeze(1)
    opacity_lat = torch.nn.functional.max_pool3d(
        opacity_5d,
        kernel_size=(
            vae_scale_factor_temporal,
            vae_scale_factor_spatial,
            vae_scale_factor_spatial,
        ),
    )

    expected_lat_shape = (
        opacity_lat.shape[0],
        1,
        condition.shape[2],
        condition.shape[3],
        condition.shape[4],
    )
    assert opacity_lat.shape == expected_lat_shape, (
        f"opacity_lat.shape {tuple(opacity_lat.shape)} != expected "
        f"{expected_lat_shape}"
    )

    opacity_lat = opacity_lat.to(condition.dtype)
    return condition * opacity_lat + noise * (1.0 - opacity_lat)


__all__ = ["opacity_weighted_latent_mix"]
