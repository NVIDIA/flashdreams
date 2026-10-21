# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Where the eleven cameras sit on the mRoPE grid.

A vision token's temporal position is its frame, scaled to the reference rate
and pushed past the text. Cameras at the same instant share it exactly: this
axis carries time and not identity. Which camera a token belongs to is carried
by the attention routing and by the caption scoped to that camera, neither of
which is a position.

Over one camera the expression is just ``arange``, which the stock path
produces. Over a chunk it is not, a chunk not knowing where its frames start, so
:func:`chunk_mrope_ids` passes positions explicitly.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor

from cosmos3_nano_sim_auto.impl.packing import ClipGeometry

# The checkpoint uses base_fps=30: a latent frame at 30 FPS advances
# temporal mRoPE by exactly 1.0.
BASE_FPS = 30.0
MODALITY_MARGIN = 15_000.0


def vision_temporal_offset(
    num_text_tokens: int,
    *,
    modality_margin: float = MODALITY_MARGIN,
) -> float:
    """Where vision starts on the temporal axis, given the text ahead of it.

    Text tokens take one position each from zero, so vision would start at
    ``num_text_tokens``; the margin then pushes it far enough past that no
    vision position can collide with a text one. 15,000 is not a tuning
    parameter but what the checkpoint was trained with, and vision has to land
    where training put it.
    """
    if num_text_tokens < 0:
        raise ValueError(
            f"num_text_tokens must be non-negative, got {num_text_tokens}."
        )
    return float(num_text_tokens) + modality_margin


def caption_mrope_ids(view_text_tokens: Sequence[int]) -> tuple[Tensor, int]:
    """``[3, sum(runs)]`` text positions for one caption per camera, and the
    positions they occupy between them.

    Each caption is laid from zero rather than after the one before it. The
    captions are alternative descriptions of a single instant of a single rig,
    not a sequence, so running them end to end would push vision further along
    the temporal axis with every extra camera and make a clip's vision
    positions depend on how many cameras it has. Vision instead follows the
    longest caption, which is the returned count, and that is what
    :func:`vision_temporal_offset` takes.
    """
    if not view_text_tokens:
        raise ValueError("per-view captions need at least one camera.")
    if any(tokens < 0 for tokens in view_text_tokens):
        raise ValueError(
            f"caption lengths must be non-negative, got {list(view_text_tokens)}."
        )
    ids = torch.cat(
        [torch.arange(tokens, dtype=torch.float32) for tokens in view_text_tokens]
    )
    return ids.unsqueeze(0).expand(3, -1).contiguous(), max(view_text_tokens)


def temporal_positions(geometry: ClipGeometry, frames: Tensor) -> Tensor:
    """``[num_views * len(frames)]`` latent-time coordinates, view-major.

    Every camera repeats the same frames, an instant of the drive being one
    coordinate however many cameras saw it.

    ``frames`` are global frame indexes within the clip, so a chunk passes the
    range it actually covers rather than counting from zero.
    """
    return frames.to(torch.float32).repeat(geometry.num_views)


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
) -> Tensor:
    """Position ids for a whole clip, for one of the two vision tracks.

    Both tracks of the prefill get the same tensor, which is what it means for
    a control token and the frame it describes to share a position.

    Raises:
        ValueError: the clip has no end, so neither would this tensor. Position
            a chunk at a time with :func:`chunk_mrope_ids`.
    """
    if geometry.frames_per_view is None:
        raise ValueError(
            "a clip with no end has no whole-clip positions. Use chunk_mrope_ids."
        )
    frames = torch.arange(geometry.frames_per_view)
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
) -> Tensor:
    """Position ids for the frames one autoregressive chunk covers.

    The clip's own positions restricted to those frames, never a fresh count
    from zero: the cache the chunk attends against numbers frames by where they
    fall in the clip, so the chunk has to as well.
    """
    if chunk_frames < 1:
        raise ValueError(f"chunk_frames must be >= 1, got {chunk_frames}.")
    last = geometry.frames_per_view
    if chunk_start < 0 or (last is not None and chunk_start + chunk_frames > last):
        raise ValueError(
            f"Chunk [{chunk_start}, {chunk_start + chunk_frames}) runs outside the clip's "
            f"{geometry.frames_per_view} frames per view."
        )
    frames = torch.arange(chunk_start, chunk_start + chunk_frames)
    return _mrope_ids(
        geometry, frames, fps=fps, temporal_offset=temporal_offset, base_fps=base_fps
    )
