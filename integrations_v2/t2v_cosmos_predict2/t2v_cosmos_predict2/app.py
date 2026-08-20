# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cosmos Predict2 text-to-video application, generating a clip from a prompt."""

import dataclasses
from typing import Any

from cosmos_predict2.config import RUNNER_COSMOS2_T2V_2B_720P

from flashdreams.api_v2.application import IApplication
from flashdreams.t2v_v2.application import T2VApplication
from flashdreams.t2v_v2.defaults import T2VApplicationDefaults


class CosmosPredict2T2VApplication(T2VApplication):
    """Cosmos Predict2 2B at 720p, generating video from text in one rollout.

    Everything about running a text-to-video model is
    :class:`~flashdreams.t2v_v2.application.T2VApplication`; what belongs here
    is which model, and that comes from the runner config this integration
    already ships, so the frame size and rate are not written down twice.

    Like Wan 2.1 and unlike the streaming models here, this one generates its
    whole clip in a single block. Its runner config says so, with a block count
    of one, and this refuses to be asked for more.
    """

    def __init__(self, pipeline_config: Any | None = None) -> None:
        """
        Args:
            pipeline_config: Model to run, replacing the one the runner config
                names. The default is the 2B 720p checkpoint; a test passes a
                stand-in.
        """
        defaults = T2VApplicationDefaults.from_runner_config(RUNNER_COSMOS2_T2V_2B_720P)
        if pipeline_config is not None:
            defaults = dataclasses.replace(defaults, pipeline_config=pipeline_config)
        super().__init__(defaults=defaults)

    def _validate_total_blocks(self, total_blocks: int) -> None:
        """Reject a rollout, since this model generates its clip in one block.

        Raises:
            ValueError: More than one block was asked for. A second block would
                not continue the first, so generating one is not something this
                model can be asked for a little more of.
        """
        super()._validate_total_blocks(total_blocks)
        if total_blocks > 1:
            raise ValueError(
                "Cosmos Predict2 T2V generates its whole clip in one block; "
                f"--total-blocks must be 1, got {total_blocks}."
            )


def create_app() -> IApplication:
    """Return a new Cosmos Predict2 text-to-video application."""
    return CosmosPredict2T2VApplication()
