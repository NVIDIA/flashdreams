# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FastVideo CausalWan 2.2 text-to-video application, generating from a prompt."""

import dataclasses
from typing import Any

from fastvideo_causal_wan22.config import RUNNER_WAN22_T2V_14B

from flashdreams.api_v2.application import IApplication
from flashdreams.infra.config import derive_config
from flashdreams.t2v_v2.application import T2VApplication
from flashdreams.t2v_v2.defaults import T2VApplicationDefaults


class FastvideoCausalWan22T2VApplication(T2VApplication):
    """FastVideo CausalWan 2.2 14B, generating video from text.

    Holds two transformers rather than one, a high-noise branch and a low-noise
    branch, which is what the compile override below is for.
    """

    def __init__(self, pipeline_config: Any | None = None) -> None:
        """
        Args:
            pipeline_config: Model to run, in place of the 14B checkpoint
                pair. A test passes a stand-in.
        """
        defaults = T2VApplicationDefaults.from_runner_config(RUNNER_WAN22_T2V_14B)
        if pipeline_config is not None:
            defaults = dataclasses.replace(defaults, pipeline_config=pipeline_config)
        super().__init__(defaults=defaults)

    def _apply_compile_override(self, pipeline_config: Any, enabled: bool) -> Any:
        """Turn compilation on or off for both noise-level transformers.

        The shared override reaches one, leaving the other branch out of step.
        """
        return derive_config(
            pipeline_config,
            diffusion_model={
                "transformer": {
                    "transformer_high_noise": {"compile_network": enabled},
                    "transformer_low_noise": {"compile_network": enabled},
                },
            },
        )


def create_app() -> IApplication:
    """Return a new FastVideo CausalWan 2.2 text-to-video application."""
    return FastvideoCausalWan22T2VApplication()
