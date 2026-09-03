# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lingbot specialization of the shared camera-to-video application."""

from __future__ import annotations

import dataclasses
from typing import Any

import torch
from cam2v import Cam2VApplication, Cam2VApplicationDefaults

from flashdreams.api_v2.application import IApplication
from flashdreams.infra.config import derive_config
from lingbot.config import (
    PIPELINE_LINGBOT_WORLD_FAST,
    PIPELINE_LINGBOT_WORLD_FAST_TAEHV_WINDOW15_SINK3,
    PIPELINE_LINGBOT_WORLD_V2_14B_CAUSAL_FAST,
    PIPELINE_LINGBOT_WORLD_V2_14B_CAUSAL_FAST_TAEHV_WINDOW15_SINK3,
)
from lingbot.impl.conditioning import resolve_lingbot_conditioning

LINGBOT_CAM2V_DEFAULTS = Cam2VApplicationDefaults(
    pipeline_config=derive_config(
        PIPELINE_LINGBOT_WORLD_FAST_TAEHV_WINDOW15_SINK3,
        enable_sync_and_profile=False,
    ),
    input_resolver=resolve_lingbot_conditioning,
    total_blocks=20,
    pixel_width=832,
    pixel_height=464,
    first_frame_dtype=torch.bfloat16,
    first_frame_interpolation="cubic",
    fps=16,
    log_model_timing=True,
    install_hint="Install the Lingbot integration: pip install flashdreams-lingbot.",
    input_defaults={"example_data": False, "example_idx": 0},
)
"""Lingbot defaults for the reusable Cam2V application."""


class LingbotCam2VApplication(Cam2VApplication):
    """Lingbot World specialization of the shared Cam2V application."""

    def __init__(self, pipeline_config: Any | None = None) -> None:
        """Select the pipeline config used by this Cam2V application.

        Args:
            pipeline_config: Model variant to run; ``None`` uses the application
                default.
        """
        defaults = LINGBOT_CAM2V_DEFAULTS
        if pipeline_config is not None:
            defaults = dataclasses.replace(
                defaults,
                pipeline_config=derive_config(
                    pipeline_config, enable_sync_and_profile=False
                ),
            )
        super().__init__(defaults=defaults)


def create_app() -> IApplication:
    """Return a Lingbot camera-to-video application."""
    return LingbotCam2VApplication()


def create_app_fast() -> IApplication:
    """Return the Lingbot World Fast application."""
    return LingbotCam2VApplication(pipeline_config=PIPELINE_LINGBOT_WORLD_FAST)


def create_app_fast_taehv_window15_sink3() -> IApplication:
    """Return the Lingbot World Fast bounded-window TAEHV application."""
    return LingbotCam2VApplication(
        pipeline_config=PIPELINE_LINGBOT_WORLD_FAST_TAEHV_WINDOW15_SINK3
    )


def create_app_v2_14b_causal_fast() -> IApplication:
    """Return the Lingbot World v2 14B Causal Fast application."""
    return LingbotCam2VApplication(
        pipeline_config=PIPELINE_LINGBOT_WORLD_V2_14B_CAUSAL_FAST
    )


def create_app_v2_14b_causal_fast_taehv_window15_sink3() -> IApplication:
    """Return the Lingbot World v2 bounded-window TAEHV application."""
    return LingbotCam2VApplication(
        pipeline_config=(PIPELINE_LINGBOT_WORLD_V2_14B_CAUSAL_FAST_TAEHV_WINDOW15_SINK3)
    )


__all__ = [
    "LingbotCam2VApplication",
    "create_app",
    "create_app_fast",
    "create_app_fast_taehv_window15_sink3",
    "create_app_v2_14b_causal_fast",
    "create_app_v2_14b_causal_fast_taehv_window15_sink3",
]
