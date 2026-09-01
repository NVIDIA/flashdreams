# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cosmos Predict2 text-to-video application, generating a clip from a prompt."""

import dataclasses
from typing import Any

from cosmos_predict2.config import PIPELINE_COSMOS2_T2V_2B_720P

from flashdreams.api_v2.application import IApplication
from t2v import T2VApplication, T2VApplicationDefaults

COSMOS_PREDICT2_T2V_DEFAULTS = T2VApplicationDefaults(
    pipeline_config=PIPELINE_COSMOS2_T2V_2B_720P,
    total_blocks=1,
    pixel_width=1280,
    pixel_height=720,
    fps=16,
)


class CosmosPredict2T2VApplication(T2VApplication):
    """Cosmos Predict2 2B at 720p, generating video from text in one rollout.

    Bidirectional rather than streaming: it generates its whole clip in a
    single block.
    """

    def __init__(self, pipeline_config: Any | None = None) -> None:
        """
        Args:
            pipeline_config: Model to run, in place of the 2B 720p checkpoint.
                A test passes a stand-in.
        """
        defaults = COSMOS_PREDICT2_T2V_DEFAULTS
        if pipeline_config is not None:
            defaults = dataclasses.replace(defaults, pipeline_config=pipeline_config)
        super().__init__(defaults=defaults)

    def _validate_total_blocks(self, total_blocks: int) -> None:
        """Reject a rollout: a second block would not continue the first.

        Raises:
            ValueError: More than one block was asked for.
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
