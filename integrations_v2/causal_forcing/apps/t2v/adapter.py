# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Causal-Forcing text-to-video application, generating a clip from a prompt."""

import dataclasses
from typing import Any

from causal_forcing.config import (
    PIPELINE_WAN21_T2V_1PT3B_CHUNKWISE,
    PIPELINE_WAN21_T2V_1PT3B_FRAMEWISE,
)

from flashdreams.api_v2.application import IApplication
from t2v import T2VApplication, T2VApplicationDefaults

CAUSAL_FORCING_T2V_DEFAULTS = T2VApplicationDefaults(
    pipeline_config=PIPELINE_WAN21_T2V_1PT3B_CHUNKWISE,
    total_blocks=60,
    pixel_width=832,
    pixel_height=480,
    fps=16,
)


class CausalForcingT2VApplication(T2VApplication):
    """Causal-Forcing Wan 2.1 1.3B, generating video from text.

    The chunkwise variant rather than the framewise one: it generates three
    latent frames a block instead of one, the same shape of rollout the other
    streaming models here have.
    """

    def __init__(self, pipeline_config: Any | None = None) -> None:
        """
        Args:
            pipeline_config: Model to run, in place of the chunkwise
                checkpoint. A test passes a stand-in.
        """
        defaults = CAUSAL_FORCING_T2V_DEFAULTS
        if pipeline_config is not None:
            defaults = dataclasses.replace(defaults, pipeline_config=pipeline_config)
        super().__init__(defaults=defaults)


def create_app() -> IApplication:
    """Return a new Causal-Forcing text-to-video application."""
    return CausalForcingT2VApplication()


def create_app_framewise() -> IApplication:
    """Return the framewise Causal-Forcing text-to-video application."""
    return CausalForcingT2VApplication(PIPELINE_WAN21_T2V_1PT3B_FRAMEWISE)
