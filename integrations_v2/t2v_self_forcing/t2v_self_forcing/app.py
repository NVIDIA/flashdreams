# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Self-Forcing text-to-video application, generating a clip from a prompt."""

import dataclasses
from typing import Any

from self_forcing.config import RUNNER_WAN21_T2V_1PT3B

from flashdreams.api_v2.application import IApplication
from flashdreams.t2v_v2.application import T2VApplication
from flashdreams.t2v_v2.defaults import T2VApplicationDefaults


class SelfForcingT2VApplication(T2VApplication):
    """Self-Forcing distilled Wan 2.1 1.3B, generating video from text.

    Everything about running a text-to-video model is
    :class:`~flashdreams.t2v_v2.application.T2VApplication`; what belongs here
    is which model, and that comes from the runner config this integration
    already ships, so the frame size and rate are not written down twice.
    """

    def __init__(self, pipeline_config: Any | None = None) -> None:
        """
        Args:
            pipeline_config: Model to run, replacing the one the runner config
                names. The default is the distilled four-step checkpoint; a test
                passes a stand-in.
        """
        defaults = T2VApplicationDefaults.from_runner_config(RUNNER_WAN21_T2V_1PT3B)
        if pipeline_config is not None:
            defaults = dataclasses.replace(defaults, pipeline_config=pipeline_config)
        super().__init__(defaults=defaults)


def create_app() -> IApplication:
    """Return a new Self-Forcing text-to-video application."""
    return SelfForcingT2VApplication()
