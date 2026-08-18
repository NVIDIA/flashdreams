# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""FastVideo CausalWan 2.2 text-to-video application factory."""

from typing import Any

from t2v import (
    T2VApplication,
    T2VApplicationDefaults,
    T2VApplicationSession,
)

from fastvideo_causal_wan22.config import RUNNER_WAN22_T2V_14B
from flashdreams.demo import IFlashDreamsApplication
from flashdreams.infra.config import derive_config


class FastvideoCausalWan22T2VApplication(T2VApplication):
    """FastVideo CausalWan 2.2 text-to-video application."""

    session_type = T2VApplicationSession

    def __init__(self) -> None:
        super().__init__(
            defaults=T2VApplicationDefaults.from_runner_config(RUNNER_WAN22_T2V_14B)
        )

    def _apply_compile_override(
        self,
        pipeline_config: Any,
        enabled: bool,
    ) -> Any:
        """Apply compilation to both Wan 2.2 noise-level transformers."""
        return derive_config(
            pipeline_config,
            diffusion_model={
                "transformer": {
                    "transformer_high_noise": {"compile_network": enabled},
                    "transformer_low_noise": {"compile_network": enabled},
                },
            },
        )


def create_app() -> IFlashDreamsApplication:
    """Create the FastVideo CausalWan 2.2 text-to-video application."""
    return FastvideoCausalWan22T2VApplication()


__all__ = [
    "FastvideoCausalWan22T2VApplication",
    "create_app",
]
