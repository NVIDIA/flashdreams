# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Causal-Forcing text-to-video application, generating a clip from a prompt."""

import dataclasses
from typing import Any

from causal_forcing.config import RUNNER_WAN21_T2V_1PT3B_CHUNKWISE

from flashdreams.api_v2.application import IApplication
from flashdreams.t2v_v2.application import T2VApplication
from flashdreams.t2v_v2.defaults import T2VApplicationDefaults


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
        defaults = T2VApplicationDefaults.from_runner_config(
            RUNNER_WAN21_T2V_1PT3B_CHUNKWISE
        )
        if pipeline_config is not None:
            defaults = dataclasses.replace(defaults, pipeline_config=pipeline_config)
        super().__init__(defaults=defaults)


def create_app() -> IApplication:
    """Return a new Causal-Forcing text-to-video application."""
    return CausalForcingT2VApplication()
