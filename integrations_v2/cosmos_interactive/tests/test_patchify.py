# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Patchify <-> unpatchify round-trip + layout/order checks."""

from __future__ import annotations

import pytest
import torch
from cosmos_interactive._patchify import patchify_latent, unpatchify_latent

pytestmark = pytest.mark.ci_cpu


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
    from cosmos_interactive._rope import vae_mrope_ids

    c, t, h, w, p = 16, 2, 4, 4, 2
    latent = torch.randn(c, t, h, w)
    tokens = patchify_latent(latent, p)
    ids, _ = vae_mrope_ids(t, h // p, w // p, 0.0, fps=None)
    assert tokens.shape[0] == ids.shape[1]


def test_patch_latent_dim_matches_spec() -> None:
    from cosmos_interactive._spec import SPEC

    c, t, h, w = SPEC.vfm.vae_latent_channels, 1, 4, 4
    p = SPEC.vfm.latent_patch_size
    tokens = patchify_latent(torch.randn(c, t, h, w), p)
    assert tokens.shape[1] == SPEC.vfm.patch_latent_dim  # 2^2 * 48 = 192
