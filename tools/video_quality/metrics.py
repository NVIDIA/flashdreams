# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compatibility imports for the installable video-quality metrics module."""

from flashdreams.quality.video_quality.metrics import (
    RGBVideo,
    VideoMetricsInput,
    compute_video_metrics,
    synthetic_video,
)

__all__ = [
    "RGBVideo",
    "VideoMetricsInput",
    "compute_video_metrics",
    "synthetic_video",
]
