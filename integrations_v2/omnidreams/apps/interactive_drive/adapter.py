# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OmniDreams binding for the reusable interactive-drive application."""

from __future__ import annotations

from interactive_drive import (
    InteractiveDriveApplication,
    InteractiveDriveApplicationDefaults,
)
from omnidreams.config import (
    OMNIDREAMS_FAST_PERF_PIPELINE_CONFIG,
    OMNIDREAMS_PERF_PIPELINE_CONFIG,
    OMNIDREAMS_PIPELINE_CONFIG,
)

from flashdreams.api_v2.application import IApplication

OMNIDREAMS_INTERACTIVE_DRIVE_DEFAULTS = InteractiveDriveApplicationDefaults(
    title="Interactive Drive",
    slug="interactive-drive",
    total_blocks=0,
    fps=30,
    width=1280,
    height=704,
    pipeline_config=OMNIDREAMS_PIPELINE_CONFIG,
)
OMNIDREAMS_INTERACTIVE_DRIVE_PERF_DEFAULTS = InteractiveDriveApplicationDefaults(
    title="Interactive Drive (Perf)",
    slug="interactive-drive-perf",
    total_blocks=0,
    fps=30,
    width=1168,
    height=640,
    pipeline_config=OMNIDREAMS_PERF_PIPELINE_CONFIG,
)

OMNIDREAMS_INTERACTIVE_DRIVE_FAST_PERF_DEFAULTS = InteractiveDriveApplicationDefaults(
    title="Interactive Drive (Fast Perf)",
    slug="interactive-drive-fast-perf",
    total_blocks=0,
    fps=30,
    width=1168,
    height=640,
    pipeline_config=OMNIDREAMS_FAST_PERF_PIPELINE_CONFIG,
)


def create_app() -> IApplication:
    """Create Interactive Drive with the regular OmniDreams config."""
    return InteractiveDriveApplication(
        defaults=OMNIDREAMS_INTERACTIVE_DRIVE_DEFAULTS,
    )


def create_perf_app() -> IApplication:
    """Create Interactive Drive with the performance OmniDreams config."""
    return InteractiveDriveApplication(
        defaults=OMNIDREAMS_INTERACTIVE_DRIVE_PERF_DEFAULTS,
    )


def create_fast_perf_app() -> IApplication:
    """Create Interactive Drive with the fast performance OmniDreams config."""
    return InteractiveDriveApplication(
        defaults=OMNIDREAMS_INTERACTIVE_DRIVE_FAST_PERF_DEFAULTS,
    )


__all__ = ["create_app", "create_perf_app", "create_fast_perf_app"]
