# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Plug-compatible runtime entrypoint for the interactive driving demo."""

from __future__ import annotations

from flashdreams.runtime.demo import DemoAdapter, DemoSpec
from flashdreams.runtime.metrics import MetricsRecorder
from flashdreams.serving.presentation import HudOverlay

from interactive_drive_app.application import (
    DrivingSessionOutcome,
    InteractiveDriveApplication,
)


def run_driving_session(
    *,
    spec: DemoSpec,
    adapter: DemoAdapter,
    overlay: HudOverlay | None = None,
    metrics: MetricsRecorder | None = None,
) -> DrivingSessionOutcome:
    """Run one model adapter using the driving demo's keyboard semantics.

    The adapter remains responsible for global scenario inputs and for mapping
    the canonical ``driver_command`` modality into its model-facing step fields.
    """
    app = InteractiveDriveApplication(
        adapter=adapter,
        initial_spec=spec,
        overlay=overlay,
    )
    try:
        return app.run_session(
            spec=spec,
            session_id="session-0",
            metrics=metrics,
        )
    finally:
        app.close()


__all__ = ["run_driving_session"]
