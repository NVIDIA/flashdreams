# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Utilities for output-quality regression checks."""

from flashdreams.quality.clip_compare import (
    ClipComparisonResult,
    ClipComparisonThresholds,
    ClipFrameMetrics,
    assert_clip_within_thresholds,
    bottom_half,
    compare_video_arrays,
    format_clip_comparison,
    parse_frame_indices,
    read_video_rgb,
    select_frame_indices,
)

__all__ = [
    "ClipComparisonResult",
    "ClipComparisonThresholds",
    "ClipFrameMetrics",
    "assert_clip_within_thresholds",
    "bottom_half",
    "compare_video_arrays",
    "format_clip_comparison",
    "parse_frame_indices",
    "read_video_rgb",
    "select_frame_indices",
]
