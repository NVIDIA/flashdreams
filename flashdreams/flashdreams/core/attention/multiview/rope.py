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

"""Multi-view position IDs for multi-axis rotary position embeddings."""

from __future__ import annotations

import torch
from torch import Tensor

from flashdreams.core.attention.multiview.packing import ClipGeometry

BASE_FPS = 24.0
"""Default frame-rate scale for temporal position IDs."""

MODALITY_MARGIN = 15_000.0
"""Default temporal-axis gap between text and vision modalities."""


def vision_temporal_offset(
    num_text_tokens: int,
    *,
    modality_margin: float = MODALITY_MARGIN,
) -> float:
    """Where vision starts on the temporal axis, given the text ahead of it.

    Text tokens take one position each from zero, so vision would start at
    ``num_text_tokens``; the margin then pushes it far enough past that no
    vision position can collide with a text one. 15,000 is not a tuning
    parameter but a compatibility default for layouts using this convention.
    """
    if num_text_tokens < 0:
        raise ValueError(
            f"num_text_tokens must be non-negative, got {num_text_tokens}."
        )
    return float(num_text_tokens) + modality_margin


def temporal_positions(geometry: ClipGeometry, frames: Tensor) -> Tensor:
    """``[num_views * len(frames)]`` latent-time coordinates, view-major.

    ``frames`` are global frame indexes within the clip, so a chunk passes the
    range it actually covers rather than counting from zero.
    """
    views = torch.arange(geometry.num_views, device=frames.device, dtype=torch.float32)
    grid = views[:, None] * geometry.stride + frames[None, :].to(torch.float32)
    return grid.reshape(-1)


def _mrope_ids(
    geometry: ClipGeometry,
    frames: Tensor,
    *,
    fps: float,
    temporal_offset: float,
    base_fps: float = BASE_FPS,
) -> Tensor:
    """``[3, N]`` position ids -- temporal, height, width -- in packed token order."""
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}.")
    patch_h, patch_w = geometry.patch_h, geometry.patch_w
    spatial = patch_h * patch_w

    scaled = temporal_positions(geometry, frames) * (base_fps / fps) + temporal_offset
    t_index = scaled.repeat_interleave(spatial)

    grid_t = int(scaled.numel())
    h_index = (
        torch.arange(patch_h, device=frames.device, dtype=torch.float32)
        .repeat_interleave(patch_w)
        .repeat(grid_t)
    )
    w_index = torch.arange(patch_w, device=frames.device, dtype=torch.float32).repeat(
        grid_t * patch_h
    )

    return torch.stack([t_index, h_index, w_index], dim=0)


def clip_mrope_ids(
    geometry: ClipGeometry,
    *,
    fps: float,
    temporal_offset: float,
    base_fps: float = BASE_FPS,
    device: torch.device | None = None,
) -> Tensor:
    """Position ids for a whole clip, for one of the two vision tracks.

    Both tracks of the prefill get the same tensor, which is what it means for
    a control token and the frame it describes to share a position.
    """
    frames = torch.arange(geometry.frames_per_view, device=device)
    return _mrope_ids(
        geometry, frames, fps=fps, temporal_offset=temporal_offset, base_fps=base_fps
    )


def chunk_mrope_ids(
    geometry: ClipGeometry,
    *,
    chunk_start: int,
    chunk_frames: int,
    fps: float,
    temporal_offset: float,
    base_fps: float = BASE_FPS,
    device: torch.device | None = None,
) -> Tensor:
    """Position ids for the frames one autoregressive chunk covers.

    The clip's own positions restricted to those frames, never a fresh count
    from zero: the cache the chunk attends against numbers frames by where they
    fall in the clip, so the chunk has to as well.
    """
    if chunk_frames < 1:
        raise ValueError(f"chunk_frames must be >= 1, got {chunk_frames}.")
    if chunk_start < 0 or chunk_start + chunk_frames > geometry.frames_per_view:
        raise ValueError(
            f"Chunk [{chunk_start}, {chunk_start + chunk_frames}) runs outside the clip's "
            f"{geometry.frames_per_view} frames per view."
        )
    frames = torch.arange(chunk_start, chunk_start + chunk_frames, device=device)
    return _mrope_ids(
        geometry, frames, fps=fps, temporal_offset=temporal_offset, base_fps=base_fps
    )
