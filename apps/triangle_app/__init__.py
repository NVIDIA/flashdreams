# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Triangle application."""

from .application import (
    DEFAULT_TRIANGLE_COLOR,
    TRIANGLE_INPUT_MODES,
    TRIANGLE_INPUT_SCHEMA,
    TRIANGLE_OUTPUT_MODES,
    TriangleApp,
    TriangleInputProvider,
    TriangleOutputMode,
    TriangleScenario,
)

__all__ = [
    "DEFAULT_TRIANGLE_COLOR",
    "TRIANGLE_INPUT_MODES",
    "TRIANGLE_INPUT_SCHEMA",
    "TRIANGLE_OUTPUT_MODES",
    "TriangleApp",
    "TriangleInputProvider",
    "TriangleOutputMode",
    "TriangleScenario",
]
