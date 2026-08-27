# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lingbot specialization of the shared camera-to-video application."""

from __future__ import annotations

from dataclasses import replace

import torch
from cam2v import Cam2VApplication, Cam2VApplicationDefaults
from lingbot.config import RUNNER_LINGBOT_WORLD_FAST_TAEHV_WINDOW15_SINK3

from flashdreams.api_v2.application import IApplication
from flashdreams.infra.config import derive_config

from .conditioning import resolve_lingbot_conditioning

_INSTALL_HINT = (
    "Install the Lingbot Cam2V application: pip install flashdreams-cam2v-lingbot."
)


def _interactive_defaults() -> Cam2VApplicationDefaults:
    """Keep device-wide profiling barriers out of the interactive UI path."""
    runner = RUNNER_LINGBOT_WORLD_FAST_TAEHV_WINDOW15_SINK3
    defaults = Cam2VApplicationDefaults.from_runner_config(
        runner,
        input_resolver=resolve_lingbot_conditioning,
        first_frame_dtype=torch.bfloat16,
        first_frame_interpolation="cubic",
        install_hint=_INSTALL_HINT,
    )
    return replace(
        defaults,
        log_model_timing=True,
        pipeline_config=derive_config(
            runner.pipeline,
            enable_sync_and_profile=False,
        ),
    )


class LingbotCam2VApplication(Cam2VApplication):
    """Lingbot World configured through its existing interactive runner config."""

    def __init__(self) -> None:
        super().__init__(defaults=_interactive_defaults())


def create_app() -> IApplication:
    """Return a Lingbot camera-to-video application."""
    return LingbotCam2VApplication()


__all__ = ["LingbotCam2VApplication", "create_app"]
