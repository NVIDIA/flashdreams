# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OmniDreams binding for the reusable Crazy Robotaxi application."""

from __future__ import annotations

from crazy_robotaxi import CrazyRobotaxiApplication, CrazyRobotaxiApplicationDefaults
from omnidreams.config import (
    OMNIDREAMS_FAST_PERF_PIPELINE_CONFIG,
    OMNIDREAMS_FAST_PERF_RESPONSIVE_PIPELINE_CONFIG,
    OMNIDREAMS_OPTIMIZED_GB300_RESPONSIVE_PIPELINE_CONFIG,
    OMNIDREAMS_OPTIMIZED_RTX_PRO_6000_RESPONSIVE_PIPELINE_CONFIG,
    OMNIDREAMS_PERF_PIPELINE_CONFIG,
    OMNIDREAMS_PERF_RESPONSIVE_PIPELINE_CONFIG,
    OMNIDREAMS_PIPELINE_CONFIG,
    OMNIDREAMS_RESPONSIVE_PIPELINE_CONFIG,
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
OMNIDREAMS_CRAZY_ROBOTAXI_RESPONSIVE_DEFAULTS = CrazyRobotaxiApplicationDefaults(
    title="Crazy Robotaxi (Responsive)",
    slug="crazy-robotaxi-responsive",
    width=1280,
    height=704,
    pipeline_config=OMNIDREAMS_RESPONSIVE_PIPELINE_CONFIG,
)
OMNIDREAMS_CRAZY_ROBOTAXI_PERF_RESPONSIVE_DEFAULTS = CrazyRobotaxiApplicationDefaults(
    title="Crazy Robotaxi (Perf Responsive)",
    slug="crazy-robotaxi-perf-responsive",
    width=1168,
    height=640,
    pipeline_config=OMNIDREAMS_PERF_RESPONSIVE_PIPELINE_CONFIG,
)
OMNIDREAMS_CRAZY_ROBOTAXI_FAST_PERF_RESPONSIVE_DEFAULTS = (
    CrazyRobotaxiApplicationDefaults(
        title="Crazy Robotaxi (Fast Perf Responsive)",
        slug="crazy-robotaxi-fast-perf-responsive",
        width=1168,
        height=640,
        pipeline_config=OMNIDREAMS_FAST_PERF_RESPONSIVE_PIPELINE_CONFIG,
    )
)
OMNIDREAMS_CRAZY_ROBOTAXI_OPTIMIZED_GB300_RESPONSIVE_DEFAULTS = (
    CrazyRobotaxiApplicationDefaults(
        title="Crazy Robotaxi (Optimized GB300 Responsive)",
        slug="crazy-robotaxi-optimized-gb300-responsive",
        width=1280,
        height=704,
        pipeline_config=OMNIDREAMS_OPTIMIZED_GB300_RESPONSIVE_PIPELINE_CONFIG,
    )
)
OMNIDREAMS_CRAZY_ROBOTAXI_OPTIMIZED_RTX_PRO_6000_RESPONSIVE_DEFAULTS = (
    CrazyRobotaxiApplicationDefaults(
        title="Crazy Robotaxi (Optimized RTX PRO 6000 Responsive)",
        slug="crazy-robotaxi-optimized-rtx-pro-6000-responsive",
        width=1280,
        height=704,
        pipeline_config=(OMNIDREAMS_OPTIMIZED_RTX_PRO_6000_RESPONSIVE_PIPELINE_CONFIG),
    )
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


def create_responsive_app() -> IApplication:
    """Create Crazy Robotaxi with responsive early-block model history."""
    return CrazyRobotaxiApplication(
        defaults=OMNIDREAMS_CRAZY_ROBOTAXI_RESPONSIVE_DEFAULTS
    )


def create_perf_responsive_app() -> IApplication:
    """Create the performance app with responsive early-block model history."""
    return CrazyRobotaxiApplication(
        defaults=OMNIDREAMS_CRAZY_ROBOTAXI_PERF_RESPONSIVE_DEFAULTS
    )


def create_fast_perf_responsive_app() -> IApplication:
    """Create the native-VAE app with responsive early-block model history."""
    return CrazyRobotaxiApplication(
        defaults=OMNIDREAMS_CRAZY_ROBOTAXI_FAST_PERF_RESPONSIVE_DEFAULTS
    )


def create_optimized_gb300_responsive_app() -> IApplication:
    """Create the responsive app with the GB300-optimized attention policy."""
    return CrazyRobotaxiApplication(
        defaults=OMNIDREAMS_CRAZY_ROBOTAXI_OPTIMIZED_GB300_RESPONSIVE_DEFAULTS
    )


def create_optimized_rtx_pro_6000_responsive_app() -> IApplication:
    """Create the responsive app with the RTX PRO 6000 attention policy."""
    return CrazyRobotaxiApplication(
        defaults=(OMNIDREAMS_CRAZY_ROBOTAXI_OPTIMIZED_RTX_PRO_6000_RESPONSIVE_DEFAULTS)
    )


__all__ = [
    "OMNIDREAMS_CRAZY_ROBOTAXI_DEFAULTS",
    "OMNIDREAMS_CRAZY_ROBOTAXI_FAST_PERF_DEFAULTS",
    "OMNIDREAMS_CRAZY_ROBOTAXI_FAST_PERF_RESPONSIVE_DEFAULTS",
    "OMNIDREAMS_CRAZY_ROBOTAXI_OPTIMIZED_GB300_RESPONSIVE_DEFAULTS",
    "OMNIDREAMS_CRAZY_ROBOTAXI_OPTIMIZED_RTX_PRO_6000_RESPONSIVE_DEFAULTS",
    "OMNIDREAMS_CRAZY_ROBOTAXI_PERF_DEFAULTS",
    "OMNIDREAMS_CRAZY_ROBOTAXI_PERF_RESPONSIVE_DEFAULTS",
    "OMNIDREAMS_CRAZY_ROBOTAXI_RESPONSIVE_DEFAULTS",
    "create_app",
    "create_fast_perf_app",
    "create_fast_perf_responsive_app",
    "create_optimized_gb300_responsive_app",
    "create_optimized_rtx_pro_6000_responsive_app",
    "create_perf_app",
    "create_perf_responsive_app",
    "create_responsive_app",
]
