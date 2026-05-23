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

"""HY-WorldPlay reconstituted-context memory frame-index selection.

At the start of denoising for a non-first AR chunk, picks which
historical frame indices the transformer attends to via KV cache.
The selection combines:

* **Temporal context** -- the most recent ``temporal_context_size``
  frames before the current chunk, kept unconditionally for
  short-range continuity.
* **FOV-overlap memory** -- of all older 4-frame clips, score each by
  the mean ``1 - FOV overlap`` (a Monte-Carlo similarity over a fixed
  point cloud around the current camera) between the clip's 1st / 3rd
  frame and the predicted clip's frames, then greedy-pick clips in
  ascending distance until ``memory_frames - temporal_context_size``
  frames are collected.

All routines accept an explicit ``device`` so GPU callers can
pre-allocate ``points_local`` alongside the rest of the pipeline.
``device=None`` keeps everything on CPU, which is slower but lets the
algorithm be unit-tested without a GPU.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import torch
from torch import Tensor

__all__ = [
    "DEFAULT_FOV_H_DEG",
    "DEFAULT_FOV_V_DEG",
    "DEFAULT_MEMORY_FRAMES",
    "DEFAULT_PREDICT_LATENT_SIZE",
    "DEFAULT_SPHERE_POINTS_COUNT",
    "DEFAULT_SPHERE_POINTS_RADIUS",
    "DEFAULT_TEMPORAL_CONTEXT_SIZE",
    "calculate_fov_overlap_similarity",
    "generate_points_in_sphere",
    "select_memory_frame_indices",
]


## Selection defaults

DEFAULT_MEMORY_FRAMES = 16
"""Total budget of memory frames per AR step (temporal context + FOV-selected)."""

DEFAULT_TEMPORAL_CONTEXT_SIZE = 12
"""Number of most-recent past frames kept unconditionally each AR step."""

DEFAULT_PREDICT_LATENT_SIZE = 4
"""Latent frames the current AR step predicts; sizes the query clip for FOV scoring."""

DEFAULT_FOV_H_DEG = 60.0
"""Horizontal FOV (degrees) for the selection-time overlap computation."""

DEFAULT_FOV_V_DEG = 35.0
"""Vertical FOV (degrees) for the selection-time overlap computation."""

DEFAULT_SPHERE_POINTS_COUNT = 50_000
"""Monte-Carlo sample count for the FOV-overlap point cloud."""

DEFAULT_SPHERE_POINTS_RADIUS = 8.0
"""Radius of the uniform-in-sphere Monte-Carlo cloud."""


## FOV-overlap utilities


def generate_points_in_sphere(
    n_points: int,
    radius: float,
    *,
    generator: torch.Generator | None = None,
    device: torch.device | str | None = None,
) -> Tensor:
    """Uniformly sample ``n_points`` 3D points inside a sphere of ``radius``.

    Args:
        n_points: Number of points to draw.
        radius: Sphere radius (world units).
        generator: Optional :class:`torch.Generator` for deterministic
            sampling; ``None`` draws from the global RNG.
        device: Target device for the returned tensor; ``None`` is CPU.

    Returns:
        ``[n_points, 3]`` tensor of ``(x, y, z)`` cartesian coordinates.
    """
    samples_r = torch.rand(n_points, generator=generator, device=device)
    samples_phi = torch.rand(n_points, generator=generator, device=device)
    samples_u = torch.rand(n_points, generator=generator, device=device)

    r = radius * torch.pow(samples_r, 1.0 / 3.0)
    phi = 2.0 * math.pi * samples_phi
    theta = torch.acos(1.0 - 2.0 * samples_u)

    x = r * torch.sin(theta) * torch.cos(phi)
    y = r * torch.sin(theta) * torch.sin(phi)
    z = r * torch.cos(theta)
    return torch.stack((x, y, z), dim=1)


def _rotation_to_pitch_yaw_deg(R: Tensor) -> tuple[Tensor, Tensor]:
    """Extract ``(pitch, yaw)`` in degrees from a 3x3 W2C rotation matrix.

    Conventions:

    * X = right, Y = up, Z = forward (computer-vision convention).
    * Yaw is in the XZ plane (``atan2(x, z)``).
    * Pitch is the elevation above horizontal
      (``atan2(y, sqrt(x**2 + z**2))``).

    The camera's forward vector in the world frame is the third column
    of the C2W rotation, which is the third row of the W2C rotation
    (``R.T[:, 2] == R[2, :]``).
    """
    R_c2w = R.T
    fwd = R_c2w[:, 2]
    x, y, z = fwd[0], fwd[1], fwd[2]
    yaw_deg = torch.atan2(x, z) * (180.0 / math.pi)
    pitch_deg = torch.atan2(y, torch.sqrt(x * x + z * z)) * (180.0 / math.pi)
    return pitch_deg, yaw_deg


def _is_inside_fov(
    points: Tensor,
    center: Tensor,
    center_pitch: Tensor,
    center_yaw: Tensor,
    fov_half_h: Tensor,
    fov_half_v: Tensor,
) -> Tensor:
    """Boolean mask for which 3D ``points`` lie inside the camera's view frustum.

    ``points`` is ``[N, 3]``, ``center`` is ``[3]``; returns a ``[N]`` bool tensor.
    """
    vectors = points - center[None, :]
    x = vectors[..., 0]
    y = vectors[..., 1]
    z = vectors[..., 2]

    azimuth = torch.atan2(x, z) * (180.0 / math.pi)
    elevation = torch.atan2(y, torch.sqrt(x * x + z * z)) * (180.0 / math.pi)

    diff_azimuth = torch.remainder(azimuth - center_yaw + 180.0, 360.0) - 180.0
    diff_elevation = torch.remainder(elevation - center_pitch + 180.0, 360.0) - 180.0
    return (diff_azimuth.abs() < fov_half_h) & (diff_elevation.abs() < fov_half_v)


def calculate_fov_overlap_similarity(
    w2c_curr: np.ndarray | Tensor,
    w2c_hist: np.ndarray | Tensor,
    *,
    points_local: Tensor,
    fov_h_deg: float = DEFAULT_FOV_H_DEG,
    fov_v_deg: float = DEFAULT_FOV_V_DEG,
    device: torch.device | str | None = None,
) -> float:
    """Monte-Carlo FOV-overlap similarity between two W2C poses.

    Estimates ``|Curr_FOV ∩ Hist_FOV| / |Curr_FOV|`` over a
    pre-sampled cloud of ``points_local`` (uniform-in-sphere; see
    :func:`generate_points_in_sphere`). Includes a ``hist < 8.0``
    distance gate so far-away viewpoints that share an angular bin
    with the current view don't inflate the score.

    Returns:
        ``float`` in ``[0.0, 1.0]``.
    """
    w2c_curr_t = torch.as_tensor(w2c_curr, device=device)
    w2c_hist_t = torch.as_tensor(w2c_hist, device=device)

    # Re-frame both poses into the current camera's coordinate system
    # so the point cloud stays in a consistent local frame across calls.
    c2w_curr = torch.linalg.inv(w2c_curr_t)
    c2w_hist = torch.linalg.inv(w2c_hist_t)
    C_inv = w2c_curr_t

    w2c_curr_loc = torch.linalg.inv(C_inv @ c2w_curr)
    w2c_hist_loc = torch.linalg.inv(C_inv @ c2w_hist)

    R_curr, t_curr = w2c_curr_loc[:3, :3], w2c_curr_loc[:3, 3]
    R_hist, t_hist = w2c_hist_loc[:3, :3], w2c_hist_loc[:3, 3]
    P_curr = -R_curr.T @ t_curr
    P_hist = -R_hist.T @ t_hist

    pitch_curr, yaw_curr = _rotation_to_pitch_yaw_deg(R_curr)
    pitch_hist, yaw_hist = _rotation_to_pitch_yaw_deg(R_hist)

    fov_half_h = torch.tensor(fov_h_deg / 2.0, device=device)
    fov_half_v = torch.tensor(fov_v_deg / 2.0, device=device)

    points_world = points_local + P_curr[None, :]

    in_fov_curr = _is_inside_fov(
        points_world, P_curr, pitch_curr, yaw_curr, fov_half_h, fov_half_v
    )
    in_fov_hist = _is_inside_fov(
        points_world, P_hist, pitch_hist, yaw_hist, fov_half_h, fov_half_v
    )

    # Distance gate: only count historical points within 8.0 units of the
    # historical camera, to prune far-away viewpoints that happen to
    # share an angular bin with the current view.
    dist_mask = torch.norm(points_world - P_hist[None, :], dim=1) < 8.0
    in_fov_hist = in_fov_hist & dist_mask

    fov_curr_count = in_fov_curr.sum()
    if fov_curr_count.item() == 0:
        return 0.0
    overlap_count = (in_fov_curr & in_fov_hist).sum()
    return float((overlap_count.float() / fov_curr_count.float()).item())


## Selection policy


def select_memory_frame_indices(
    w2c: np.ndarray | Tensor,
    current_frame_idx: int,
    *,
    points_local: Tensor,
    memory_frames: int = DEFAULT_MEMORY_FRAMES,
    temporal_context_size: int = DEFAULT_TEMPORAL_CONTEXT_SIZE,
    pred_latent_size: int = DEFAULT_PREDICT_LATENT_SIZE,
    fov_h_deg: float = DEFAULT_FOV_H_DEG,
    fov_v_deg: float = DEFAULT_FOV_V_DEG,
    device: torch.device | str | None = None,
) -> list[int]:
    """Pick the memory + temporal-context frame indices for the current AR step.

    Returns a sorted list of unique frame indices to read from history
    for the AR step about to denoise frames
    ``[current_frame_idx, current_frame_idx + pred_latent_size)``.

    Args:
        w2c: All per-frame world-to-camera matrices for the full
            rollout, shape ``[num_total_frames, 4, 4]``. May be a numpy
            array (cheap) or a torch tensor (avoids a copy on the GPU
            path).
        current_frame_idx: Index of the *first* frame the current AR
            step is about to generate. Must satisfy
            ``3 <= current_frame_idx < num_total_frames``.
        points_local: Pre-sampled Monte-Carlo sphere points, shape
            ``[N, 3]``. Build once per pipeline via
            :func:`generate_points_in_sphere` and reuse across AR steps.
        memory_frames: Total budget of historical frame indices to
            return (temporal context + FOV-selected).
        temporal_context_size: Number of most-recent past frames kept
            unconditionally.
        pred_latent_size: Latent frames the current AR step predicts;
            sizes the query clip against which historical clips are
            scored.
        fov_h_deg: Horizontal FOV (degrees) for the overlap metric.
        fov_v_deg: Vertical FOV (degrees) for the overlap metric.
        device: Optional torch device for the FOV-overlap computation;
            ``None`` (default) keeps everything on CPU.

    Returns:
        Sorted ``list[int]`` of length ``memory_frames``
        (= ``temporal_context_size`` recent frames +
        ``memory_frames - temporal_context_size`` FOV-selected older
        frames).

    Raises:
        ValueError: ``current_frame_idx`` is out of range, or
            ``memory_frames < temporal_context_size`` (negative
            FOV-selection budget).
    """
    w2c_t = w2c if isinstance(w2c, Tensor) else torch.as_tensor(np.asarray(w2c))
    num_total_frames = w2c_t.shape[0]
    if current_frame_idx >= num_total_frames or current_frame_idx < 3:
        raise ValueError(
            "current_frame_idx must lie in [3, num_total_frames); "
            f"got current_frame_idx={current_frame_idx}, "
            f"num_total_frames={num_total_frames}."
        )
    if memory_frames < temporal_context_size:
        raise ValueError(
            f"memory_frames ({memory_frames}) must be >= "
            f"temporal_context_size ({temporal_context_size}) so the "
            f"FOV-selection budget is non-negative."
        )

    start_context_idx = max(0, current_frame_idx - temporal_context_size)
    context_indices = list(range(start_context_idx, current_frame_idx))

    query_clip_end = min(current_frame_idx + pred_latent_size, num_total_frames)
    query_clip_indices = list(range(current_frame_idx, query_clip_end))

    # Historical 4-frame clips end at ``current_frame_idx - temporal_context_size``
    # (so they don't overlap the temporal-context window).
    historical_clip_starts = list(
        range(0, current_frame_idx - temporal_context_size, 4)
    )

    candidate_distances: list[tuple[int, float]] = []
    for hist_idx in historical_clip_starts:
        hist_w2c_1 = w2c_t[hist_idx]
        hist_w2c_2 = w2c_t[hist_idx + 2]
        total_dist = 0.0
        for q_idx in query_clip_indices:
            d1 = 1.0 - calculate_fov_overlap_similarity(
                w2c_t[q_idx],
                hist_w2c_1,
                points_local=points_local,
                fov_h_deg=fov_h_deg,
                fov_v_deg=fov_v_deg,
                device=device,
            )
            d2 = 1.0 - calculate_fov_overlap_similarity(
                w2c_t[q_idx],
                hist_w2c_2,
                points_local=points_local,
                fov_h_deg=fov_h_deg,
                fov_v_deg=fov_v_deg,
                device=device,
            )
            total_dist += (d1 + d2) / 2.0
        candidate_distances.append((hist_idx, total_dist / len(query_clip_indices)))

    candidate_distances.sort(key=lambda pair: pair[1])

    fov_budget = memory_frames - temporal_context_size
    memory_indices: list[int] = []
    for start_idx, _ in candidate_distances:
        if start_idx not in memory_indices:
            memory_indices.extend(range(start_idx, start_idx + 4))
        if len(memory_indices) >= fov_budget:
            break

    combined = set(context_indices) | set(memory_indices)
    final = sorted(combined)
    # Invariant: temporal context + FOV-selected sets must sum exactly
    # to ``memory_frames``. The greedy ``extend(range(start, start+4))``
    # loop above may over-shoot ``fov_budget`` by up to 3 (it breaks
    # *after* adding a clip), so misconfigured budgets surface here.
    assert len(final) == memory_frames, (
        f"memory selection produced {len(final)} frames; expected "
        f"memory_frames={memory_frames} "
        f"(temporal_context_size={temporal_context_size}, "
        f"fov_budget={fov_budget})."
    )
    return final


## Helpers


def coerce_indices(indices: Sequence[int] | Tensor) -> list[int]:
    """Normalise a frame-index container to a plain ``list[int]``.

    Keeps :attr:`HyWorldPlayCtrl.memory_frame_indices` type-stable across
    the encoder / transformer boundary regardless of whether the producer
    returned a torch tensor or a numpy array.
    """
    if isinstance(indices, Tensor):
        return [int(x) for x in indices.tolist()]
    return [int(x) for x in indices]
