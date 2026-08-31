# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OmniDreams binding for the reusable Crazy Robotaxi application."""

from __future__ import annotations

from crazy_robotaxi import CrazyRobotaxiApplication, CrazyRobotaxiApplicationDefaults
from omnidreams.config import (
    OMNIDREAMS_FAST_PERF_PIPELINE_CONFIG,
    OMNIDREAMS_PERF_PIPELINE_CONFIG,
    OMNIDREAMS_PIPELINE_CONFIG,
)

from flashdreams.api_v2.application import IApplication

OMNIDREAMS_CRAZY_ROBOTAXI_DEFAULTS = CrazyRobotaxiApplicationDefaults(
    title="Crazy Robotaxi",
    slug="crazy-robotaxi",
    width=1280,
    height=704,
    pipeline_config=OMNIDREAMS_PIPELINE_CONFIG,
)
OMNIDREAMS_CRAZY_ROBOTAXI_PERF_DEFAULTS = CrazyRobotaxiApplicationDefaults(
    title="Crazy Robotaxi (Perf)",
    slug="crazy-robotaxi-perf",
    width=1168,
    height=640,
    pipeline_config=OMNIDREAMS_PERF_PIPELINE_CONFIG,
)
OMNIDREAMS_CRAZY_ROBOTAXI_FAST_PERF_DEFAULTS = CrazyRobotaxiApplicationDefaults(
    title="Crazy Robotaxi (Fast Perf)",
    slug="crazy-robotaxi-fast-perf",
    width=1168,
    height=640,
    pipeline_config=OMNIDREAMS_FAST_PERF_PIPELINE_CONFIG,
)


def create_app() -> IApplication:
    """Create Crazy Robotaxi with the regular OmniDreams config."""
    return CrazyRobotaxiApplication(defaults=OMNIDREAMS_CRAZY_ROBOTAXI_DEFAULTS)


def create_perf_app() -> IApplication:
    """Create Crazy Robotaxi with the performance OmniDreams config."""
    return CrazyRobotaxiApplication(defaults=OMNIDREAMS_CRAZY_ROBOTAXI_PERF_DEFAULTS)


def create_fast_perf_app() -> IApplication:
    """Create Crazy Robotaxi with fast OmniDreams acceleration when available."""
    return CrazyRobotaxiApplication(
        defaults=OMNIDREAMS_CRAZY_ROBOTAXI_FAST_PERF_DEFAULTS
    )


__all__ = [
    "OMNIDREAMS_CRAZY_ROBOTAXI_DEFAULTS",
    "OMNIDREAMS_CRAZY_ROBOTAXI_FAST_PERF_DEFAULTS",
    "OMNIDREAMS_CRAZY_ROBOTAXI_PERF_DEFAULTS",
    "create_app",
    "create_fast_perf_app",
    "create_perf_app",
]
