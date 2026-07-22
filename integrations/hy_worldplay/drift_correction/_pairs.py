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

"""Loading and counterfactual-history helpers for strafe-loop pair clips."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor

from hy_worldplay._action import HyWorldPlayCtrl

TOKENS_PER_FRAME = 880
"""Post-patchify tokens per latent frame at 704x1280 (44/2 * 80/2)."""

PATCH_DIM = 192
"""Patchified feature width (48 channels * 1 * 2 * 2)."""


def load_clip(path: Path, device: torch.device | str, dtype: torch.dtype) -> dict:
    """Load a ``build_pairs.py`` clip; latents land on ``device`` in ``dtype``."""
    d = torch.load(path, map_location="cpu", weights_only=False)
    d["latents"] = d["latents"].to(device, dtype)
    return d


def make_ctrl(
    d: dict, k: int, *, device: torch.device | str, dtype: torch.dtype
) -> HyWorldPlayCtrl:
    """Reconstruct the patchified per-AR-step ctrl payload for chunk ``k``.

    For chunks ``>= 1`` the captured ctrl carries an all-zero mask, so the
    image-latent stamp is a no-op and zero latent/mask reconstruct the
    rollout conditioning exactly.
    """
    assert k >= 1, "chunk 0 carries the stamped image latent; not reconstructable"
    sl = slice(k * 4, (k + 1) * 4)
    zeros = torch.zeros(1, 4 * TOKENS_PER_FRAME, PATCH_DIM, device=device, dtype=dtype)
    return HyWorldPlayCtrl(
        latent=zeros,
        mask=torch.zeros_like(zeros),
        _is_patchified=True,
        action=d["rollout_action"][..., sl].to(device),
        viewmats=d["rollout_viewmats"][..., sl, :, :].to(device, dtype),
        Ks=d["rollout_Ks"][..., sl, :, :].to(device, dtype),
        memory_frame_indices=d["memory_frame_indices"][k],
        rollout_viewmats=d["rollout_viewmats"].to(device, dtype),
        rollout_Ks=d["rollout_Ks"].to(device, dtype),
        rollout_action=d["rollout_action"].to(device),
    )


def history_of(d: dict, k: int) -> Tensor:
    """Patchified drifted history for chunk ``k`` (frames ``0 .. 4k-1``)."""
    return d["latents"][..., : k * 4 * TOKENS_PER_FRAME, :]


def chunk_x0(d: dict, k: int) -> Tensor:
    """Patchified clean latent the rollout produced for chunk ``k``."""
    return d["latents"][
        ..., k * 4 * TOKENS_PER_FRAME : (k + 1) * 4 * TOKENS_PER_FRAME, :
    ]


def clean_counterfactual(
    history: Tensor,
    *,
    selected: list[int],
    lap_latents: int,
    clean_lap: int = 1,
) -> Tensor:
    """Swap the memory-selected frames' content for lap-aligned clean frames.

    Frames already inside laps ``<= clean_lap`` map to themselves. The
    prefill only reads the selected frames, so this substitution changes the
    entire effective history while leaving positions and per-frame
    conditioning untouched (the strafe loop makes actions and viewmats
    identical across laps).
    """
    out = history.clone()
    horizon = (clean_lap + 1) * lap_latents
    for idx in selected:
        if idx < horizon:
            continue
        src = (idx % lap_latents) + clean_lap * lap_latents
        s = slice(idx * TOKENS_PER_FRAME, (idx + 1) * TOKENS_PER_FRAME)
        d = slice(src * TOKENS_PER_FRAME, (src + 1) * TOKENS_PER_FRAME)
        out[..., s, :] = history[..., d, :]
    return out
