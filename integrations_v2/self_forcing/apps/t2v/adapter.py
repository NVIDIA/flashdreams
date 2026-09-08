# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Self-Forcing text-to-video application, generating a clip from a prompt."""

import dataclasses
from typing import Any

from t2v import T2VApplication, T2VApplicationDefaults

from flashdreams.api_v2.application import IApplication
from self_forcing.config import (
    PIPELINE_WAN21_T2V_1PT3B,
    PIPELINE_WAN21_T2V_1PT3B_SINK5_WINDOW7_REROPE,
    PIPELINE_WAN21_T2V_1PT3B_TAEHV,
)

SELF_FORCING_T2V_DEFAULTS = T2VApplicationDefaults(
    pipeline_config=PIPELINE_WAN21_T2V_1PT3B,
    total_blocks=60,
    pixel_width=832,
    pixel_height=480,
    fps=16,
)


class SelfForcingT2VApplication(T2VApplication):
    """Self-Forcing distilled Wan 2.1 1.3B, generating video from text."""

    def __init__(self, pipeline_config: Any | None = None) -> None:
        """
        Args:
            pipeline_config: Model to run, in place of the distilled four-step
                checkpoint. A test passes a stand-in.
        """
        defaults = SELF_FORCING_T2V_DEFAULTS
        if pipeline_config is not None:
            defaults = dataclasses.replace(defaults, pipeline_config=pipeline_config)
        super().__init__(defaults=defaults)


def create_app() -> IApplication:
    """Return a new Self-Forcing text-to-video application."""
    return SelfForcingT2VApplication()


def create_app_taehv() -> IApplication:
    """Return Self-Forcing with the TAEHV decoder config."""
    return SelfForcingT2VApplication(PIPELINE_WAN21_T2V_1PT3B_TAEHV)


def create_app_sink5_window7_rerope() -> IApplication:
    """Return Self-Forcing with the long-rollout config."""
    return SelfForcingT2VApplication(PIPELINE_WAN21_T2V_1PT3B_SINK5_WINDOW7_REROPE)
