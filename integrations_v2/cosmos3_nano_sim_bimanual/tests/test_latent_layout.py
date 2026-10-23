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

"""Patchify <-> unpatchify round-trip + layout/order checks."""

from __future__ import annotations

import pytest
import torch
from cosmos3_nano_sim_bimanual.patchify import (
    crop_latent,
    pad_latent,
    patchify_latent,
    unpatchify_latent,
)

pytestmark = pytest.mark.ci_cpu


def test_pad_and_crop_preserve_top_left_latent() -> None:
    latent = torch.arange(24, dtype=torch.float32).reshape(2, 1, 3, 4)

    padded = pad_latent(latent, bottom=2, right=3)

    assert padded.shape == (2, 1, 5, 7)
    assert padded.dtype == latent.dtype
    assert padded.device == latent.device
    assert padded.is_contiguous()
    assert torch.equal(padded[..., :3, :4], latent)
    assert not torch.count_nonzero(padded[..., 3:, :])
    assert not torch.count_nonzero(padded[..., :, 4:])
    assert torch.equal(crop_latent(padded, height=3, width=4), latent)


def test_patchify_preserves_hand_token_order() -> None:
    latent = torch.arange(16).reshape(2, 2, 2, 2)

    tokens = patchify_latent(latent, 2)

    assert torch.equal(
        tokens,
        torch.tensor(
            [
                [0, 8, 1, 9, 2, 10, 3, 11],
                [4, 12, 5, 13, 6, 14, 7, 15],
            ]
        ),
    )


def test_round_trip_identity() -> None:
    torch.manual_seed(0)
    c, t, h, w, p = 48, 3, 8, 6, 2
    latent = torch.randn(c, t, h, w)
    tokens = patchify_latent(latent, p)
    assert tokens.shape == (t * (h // p) * (w // p), p * p * c)
    back = unpatchify_latent(
        tokens, channels=c, grid_t=t, grid_h=h // p, grid_w=w // p, patch_spatial=p
    )
    assert back.shape == latent.shape
    assert torch.allclose(back, latent)


def test_token_count_matches_rope_vision_layout() -> None:
    # T-major: token count = grid_t * (H/p) * (W/p), same as vae_mrope_ids length.
    from cosmos3_nano_sim_bimanual.transformer.rope import vae_mrope_ids

    c, t, h, w, p = 16, 2, 4, 4, 2
    latent = torch.randn(c, t, h, w)
    tokens = patchify_latent(latent, p)
    ids, _ = vae_mrope_ids(t, h // p, w // p, 0.0, fps=None)
    assert tokens.shape[0] == ids.shape[1]


def test_patch_latent_dim_matches_spec() -> None:
    from cosmos3_nano_sim_bimanual.spec import SPEC

    c, t, h, w = SPEC.vfm.vae_latent_channels, 1, 4, 4
    p = SPEC.vfm.latent_patch_size
    tokens = patchify_latent(torch.randn(c, t, h, w), p)
    assert tokens.shape[1] == SPEC.vfm.patch_latent_dim  # 2^2 * 48 = 192
