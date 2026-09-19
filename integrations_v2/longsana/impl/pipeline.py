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

"""LongSana prompt-to-video orchestration on the shared Runtime V2 pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, cast

import torch
from sana_wm.impl.conditioning import (
    SanaWMTextPromptEncoder,
    SanaWMTextPromptEncoderConfig,
    SanaWMTextPromptRequest,
)

from flashdreams.infra.pipeline import (
    StreamInferencePipeline,
    StreamInferencePipelineCache,
    StreamInferencePipelineConfig,
)
from longsana.impl.constants import (
    LONGSANA_TEXT_CONFIG_PATH,
    MOTION_SCORE,
)
from longsana.impl.transformer import (
    LongSanaConditioning,
    LongSanaTransformer,
)


@dataclass(kw_only=True)
class LongSanaPipelineConfig(StreamInferencePipelineConfig):
    """Config for prompt encoding, LongSana diffusion, and Wan decoding."""

    _target: type["LongSanaPipeline"] = field(default_factory=lambda: LongSanaPipeline)

    prompt_encoder: SanaWMTextPromptEncoderConfig = field(
        default_factory=lambda: SanaWMTextPromptEncoderConfig(
            config_path=LONGSANA_TEXT_CONFIG_PATH,
            offload_text_encoder=True,
        )
    )
    """Shared SANA-family Gemma/CHI prompt encoder."""

    motion_score: int = MOTION_SCORE
    """Training-time motion-score suffix added to every user prompt."""


class LongSanaPipeline(StreamInferencePipeline):
    """Runtime V2 pipeline with one prompt and a recurrent LongSana cache."""

    config: LongSanaPipelineConfig

    def __init__(self, config: LongSanaPipelineConfig) -> None:
        super().__init__(config)
        self.config = config
        self.prompt_encoder = cast(
            SanaWMTextPromptEncoder,
            config.prompt_encoder.setup(),
        )

    @torch.no_grad()
    def initialize_cache(
        self,
        *,
        text: list[str],
        image: Any = None,
        height: int | None = None,
        width: int | None = None,
    ) -> StreamInferencePipelineCache:
        """Encode one prompt and construct a constant-memory rollout cache."""
        if image is not None:
            raise ValueError("The released LongSana checkpoint is text-to-video only.")
        if len(text) != 1 or not text[0].strip():
            raise ValueError("LongSana requires exactly one non-empty prompt.")

        transformer = self.diffusion_model.transformer
        if not isinstance(transformer, LongSanaTransformer):
            raise TypeError("LongSanaPipeline requires LongSanaTransformer.")
        latent_height = transformer.config.latent_height if height is None else height
        latent_width = transformer.config.latent_width if width is None else width
        expected = (
            transformer.config.latent_height,
            transformer.config.latent_width,
        )
        if (latent_height, latent_width) != expected:
            raise ValueError(
                "This LongSana release is configured for latent size "
                f"{expected}, got {(latent_height, latent_width)}."
            )

        prompt = f"{text[0].strip()} motion score: {self.config.motion_score}."
        encoded = self.prompt_encoder(
            SanaWMTextPromptRequest(prompt=prompt, negative_prompt="")
        )
        conditioning = LongSanaConditioning(
            condition=encoded.condition,
            mask=encoded.condition_mask,
        )
        return super().initialize_cache(
            transformer_context={"conditioning": conditioning},
        )

    def close(self) -> None:
        """Release prompt and generator runtimes held by the resident pipeline."""
        self.prompt_encoder.release_runtime()
        transformer = self.diffusion_model.transformer
        if isinstance(transformer, LongSanaTransformer):
            transformer.release_runtime()
