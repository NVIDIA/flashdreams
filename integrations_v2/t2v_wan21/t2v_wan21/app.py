# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Wan 2.1 text-to-video application, generating a clip from a prompt."""

import dataclasses
from typing import Any

from wan21.config import RUNNER_WAN21_T2V_1PT3B_480P

from flashdreams.api_v2.application import IApplication
from flashdreams.t2v_v2.application import T2VApplication
from flashdreams.t2v_v2.defaults import T2VApplicationDefaults


class Wan21T2VApplication(T2VApplication):
    """Wan 2.1 1.3B at 480p, generating video from text in one rollout.

    Bidirectional rather than streaming: it attends over the whole clip and
    generates it in a single block, so a run is one step. Its runner config
    states no rollout length, there being no rollout, so the length is stated
    here.
    """

    def __init__(self, pipeline_config: Any | None = None) -> None:
        """
        Args:
            pipeline_config: Model to run, in place of the 480p checkpoint. A
                test passes a stand-in.
        """
        defaults = T2VApplicationDefaults.from_runner_config(
            RUNNER_WAN21_T2V_1PT3B_480P,
            total_blocks=1,
        )
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
                "Wan 2.1 T2V generates its whole clip in one block; "
                f"--total-blocks must be 1, got {total_blocks}."
            )


def create_app() -> IApplication:
    """Return a new Wan 2.1 text-to-video application."""
    return Wan21T2VApplication()
