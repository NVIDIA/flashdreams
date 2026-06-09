# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compatibility imports for optional video-quality evaluators."""

from flashdreams.quality.video_quality.evaluators import (
    Evaluator,
    get_evaluator,
    register_evaluator,
    registered_evaluators,
)

__all__ = [
    "Evaluator",
    "get_evaluator",
    "register_evaluator",
    "registered_evaluators",
]
