# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compatibility imports for the installable video-quality manifest module."""

from flashdreams.quality.video_quality.manifest import (
    KNOWN_SEVERITIES,
    KNOWN_SUITES,
    KNOWN_THRESHOLD_OPS,
    CaseAssets,
    KnownBadClip,
    Threshold,
    VideoQualityCase,
    VideoQualityManifest,
    Window,
    load_manifest,
)

__all__ = [
    "KNOWN_SEVERITIES",
    "KNOWN_SUITES",
    "KNOWN_THRESHOLD_OPS",
    "CaseAssets",
    "KnownBadClip",
    "Threshold",
    "VideoQualityCase",
    "VideoQualityManifest",
    "Window",
    "load_manifest",
]
