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

"""Raw and embedded global conditions for Omnidreams inference."""

from typing import Annotated, TypeAlias

import torch
from flashdreams.runtime.global_condition import (
    GlobalConditionHandler as BaseGlobalConditionHandler,
)
from flashdreams.runtime.global_condition import (
    RawGlobalCondition as BaseRawGlobalCondition,
)
from omnidreams.pipeline import OmnidreamsPipeline
from omnidreams.runtime.inference_session import InferenceGlobalCondition
from pydantic import AfterValidator, StringConstraints, validate_call
from torch import Tensor


def _validate_first_frame_image(tensor: Tensor) -> Tensor:
    """Validate a normalized single-view first-frame image tensor."""
    if tensor.ndim != 6:
        raise ValueError(
            "expected a rank-6 first-frame image in "
            "[B=1, V=1, T=1, C=3, H, W] layout; "
            f"got rank {tensor.ndim} with shape {tuple(tensor.shape)}"
        )
    if tuple(tensor.shape[:4]) != (1, 1, 1, 3):
        raise ValueError(
            "expected first-frame image shape [B=1, V=1, T=1, C=3, H, W]; "
            f"got {tuple(tensor.shape)}"
        )
    if tensor.shape[-2] <= 0 or tensor.shape[-1] <= 0:
        raise ValueError(
            "expected positive first-frame image spatial dimensions; "
            f"got {tuple(tensor.shape)}"
        )
    if not tensor.dtype.is_floating_point:
        raise ValueError(
            "first-frame image must use a floating-point dtype for normalized "
            f"[-1, 1] pixels; got {tensor.dtype}"
        )
    return tensor


_TextPrompt: TypeAlias = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1)
]
"""Validated non-empty positive text prompt."""

_NegativeTextPrompt: TypeAlias = Annotated[
    str, StringConstraints(strip_whitespace=True)
]
"""Validated negative text prompt; the empty prompt remains valid."""

_FirstFrameImage: TypeAlias = Annotated[
    Tensor, AfterValidator(_validate_first_frame_image)
]
"""Normalized first-frame pixels in ``[1, 1, 1, 3, H, W]`` layout."""


class RawGlobalCondition(BaseRawGlobalCondition):
    """Application-facing Omnidreams rollout conditions."""

    text_prompt: _TextPrompt
    """Positive prompt applied to the generated driving scene."""

    negative_text_prompt: _NegativeTextPrompt
    """Negative prompt embedded for classifier-free guidance."""

    first_frame_image: _FirstFrameImage
    """Normalized first-frame pixels in ``[1, 1, 1, 3, H, W]`` layout."""


class GlobalConditionHandler(BaseGlobalConditionHandler):
    """Embed raw Omnidreams prompts and a first-frame image."""

    _pipeline: OmnidreamsPipeline
    """Pipeline whose one-shot encoders produce the rollout embeddings."""

    def __init__(self, pipeline: OmnidreamsPipeline) -> None:
        """Initialize the handler with a pipeline containing one-shot encoders.

        Args:
            pipeline: Omnidreams pipeline used to validate and embed conditions.

        Raises:
            RuntimeError: The pipeline's text or image encoder is not loaded.
        """
        self._pipeline = pipeline
        self._require_encoders()

    @torch.no_grad()
    @validate_call
    def __call__(
        self, raw_global_condition: RawGlobalCondition
    ) -> InferenceGlobalCondition:
        """Embed raw prompts and first-frame pixels for an inference session.

        Args:
            raw_global_condition: Positive and negative prompts plus normalized
                first-frame pixels.

        Returns:
            Model-ready text, negative-text, and image embeddings.

        Raises:
            RuntimeError: The pipeline's text or image encoder is not loaded.
            ValidationError: The raw condition fails Pydantic validation.
            ValueError: The image resolution violates pipeline alignment.
        """
        text_encoder, image_encoder = self._require_encoders()
        first_frame_image = raw_global_condition["first_frame_image"]
        self._pipeline._validate_image_resolution(first_frame_image)

        text_embeddings = text_encoder([raw_global_condition["text_prompt"]]).unsqueeze(
            0
        )
        negative_text_embeddings = text_encoder(
            [raw_global_condition["negative_text_prompt"]]
        ).unsqueeze(0)
        image_embeddings = image_encoder(first_frame_image)
        return InferenceGlobalCondition(
            text_embeddings=text_embeddings,
            negative_text_embeddings=negative_text_embeddings,
            image_embeddings=image_embeddings,
        )

    def _require_encoders(self):
        """Return loaded one-shot encoders or fail with lifecycle guidance."""
        text_encoder = self._pipeline.text_encoder
        image_encoder = self._pipeline.image_encoder
        if text_encoder is None or image_encoder is None:
            raise RuntimeError(
                "GlobalConditionHandler requires loaded Omnidreams text and image "
                "encoders; construct the pipeline with both encoder configs and "
                "do not release one-shot encoders before conversion"
            )
        return text_encoder, image_encoder


__all__ = ["GlobalConditionHandler", "RawGlobalCondition"]
