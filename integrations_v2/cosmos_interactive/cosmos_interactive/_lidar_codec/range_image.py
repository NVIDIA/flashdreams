# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Metric range, intensity, validity and azimuth conventions for the LiDAR codec.

The 1800 calibrated ray columns remain distinct from the V1.2 network's
1808-column circular canvas.
"""

import torch
from torch import Tensor


def circular_pad_range_video_3ch(video: Tensor, *, model_width: int) -> Tensor:
    """Wrap the azimuth seam equally on both sides of a range image."""
    pad = model_width - video.shape[-1]
    if pad < 0 or pad % 2 or pad // 2 > video.shape[-1]:
        raise ValueError("Invalid symmetric LiDAR canvas padding")
    if pad == 0:
        return video
    half = pad // 2
    return torch.cat((video[..., -half:], video, video[..., :half]), dim=-1)


def crop_range_video_width(video: Tensor, *, semantic_width: int) -> Tensor:
    """Remove only the circular padding, without resizing calibrated rays."""
    pad = video.shape[-1] - semantic_width
    if semantic_width < 1 or pad < 0 or pad % 2:
        raise ValueError("Invalid symmetric LiDAR canvas crop")
    half = pad // 2
    return video[..., half : half + semantic_width] if pad else video


def metric_lidar_to_network(
    video: Tensor, *, min_range: float, max_range: float
) -> tuple[Tensor, Tensor]:
    """Normalize metric range and unit intensity, filling invalid rays with -1."""
    if video.ndim != 5 or video.shape[1] != 3:
        raise ValueError(f"Expected metric LiDAR [B,3,T,H,W], got {tuple(video.shape)}")
    range_m, intensity = video[:, :1], video[:, 1:2]
    lo = torch.as_tensor(min_range, dtype=video.dtype, device=video.device)
    hi = torch.as_tensor(max_range, dtype=video.dtype, device=video.device)
    if bool((hi <= lo).item()):
        raise ValueError("max_range must exceed min_range")
    valid = (video[:, 2:3] >= 0.5) & (range_m >= lo) & (range_m <= hi)
    normalized_range = (range_m.clamp(lo, hi) - lo) / (hi - lo) * 2.0 - 1.0
    normalized_intensity = intensity.clamp(0.0, 1.0) * 2.0 - 1.0
    invalid = torch.full_like(normalized_range, -1.0)
    valid_float = valid.to(dtype=video.dtype)
    return torch.cat(
        (
            torch.where(valid, normalized_range, invalid),
            torch.where(valid, normalized_intensity, invalid),
            valid_float,
        ),
        dim=1,
    ), valid_float


def network_lidar_to_metric_clip(
    normalized: Tensor,
    validity: Tensor,
    *,
    min_range: float | Tensor,
    max_range: float | Tensor,
    apply_validity_mask: bool = True,
    validity_threshold: float = 0.5,
) -> Tensor:
    """Decode clamped metric ranges, unit intensities, and resolved validity."""
    if normalized.ndim != 5 or normalized.shape[1] < 2:
        raise ValueError(
            f"Expected reconstruction [B,2+,T,H,W], got {tuple(normalized.shape)}"
        )
    if not 0.0 < validity_threshold < 1.0:
        raise ValueError("validity_threshold must lie in (0, 1)")
    lo = torch.as_tensor(min_range, dtype=normalized.dtype, device=normalized.device)
    hi = torch.as_tensor(max_range, dtype=normalized.dtype, device=normalized.device)
    range_m = (normalized[:, :1].clamp(-1.0, 1.0) + 1.0) * 0.5 * (hi - lo) + lo
    intensity = (normalized[:, 1:2].clamp(-1.0, 1.0) + 1.0) * 0.5
    metric = torch.cat((range_m, intensity), dim=1)
    if apply_validity_mask:
        valid = validity >= validity_threshold
        metric = torch.where(valid, metric, torch.zeros_like(metric))
        validity = valid.to(dtype=validity.dtype)
    return torch.cat((metric, validity), dim=1)
