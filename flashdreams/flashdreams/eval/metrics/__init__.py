# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Metrics module for FlashDreams evaluation.

Metric classes register themselves with MetricRegistry at import time.
Importing this package is side-effect-free; heavy model weights are loaded
lazily on first use.
"""

from .base import BaseMetric, MetricRegistry

# Trigger metric class registration.
from .image_metrics import CLIPIQA, LPIPS, MUSIQ, NIQE, PSNR, SSIM  # noqa: F401
from .video_metrics import DOVER  # noqa: F401

__all__ = [
    "BaseMetric",
    "MetricRegistry",
    "PSNR",
    "SSIM",
    "LPIPS",
    "NIQE",
    "MUSIQ",
    "CLIPIQA",
    "DOVER",
]
