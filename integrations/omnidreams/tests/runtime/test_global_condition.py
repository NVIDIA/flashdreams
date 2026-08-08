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

"""CPU tests for Omnidreams raw global-condition embedding."""

from typing import Any, cast

import pytest
import torch
from omnidreams.pipeline import OmnidreamsPipeline
from omnidreams.runtime.global_condition import (
    GlobalConditionHandler,
    RawGlobalCondition,
)
from omnidreams.runtime.inference_session import InferenceGlobalCondition
from pydantic import ValidationError
from torch import Tensor

pytestmark = pytest.mark.ci_cpu


class _TextEncoder:
    """Text-encoder test double that records prompts and returns embeddings."""

    def __init__(self) -> None:
        """Initialize the prompt record."""
        self.calls: list[list[str]] = []

    def __call__(self, prompts: list[str]) -> Tensor:
        """Return one deterministic embedding per prompt."""
        self.calls.append(prompts)
        value = float(len(self.calls))
        return torch.full((len(prompts), 2, 4), value)


class _ImageEncoder:
    """Image-encoder test double that records and spatially pools pixels."""

    def __init__(self) -> None:
        """Initialize the image record."""
        self.calls: list[Tensor] = []

    def __call__(self, image: Tensor) -> Tensor:
        """Return a small latent while preserving batch, view, and time axes."""
        self.calls.append(image)
        return image.mean(dim=(-2, -1), keepdim=True)


class _Pipeline:
    """Pipeline test double exposing the one-shot conditioning contract."""

    def __init__(self) -> None:
        """Initialize loaded encoders and image-validation records."""
        self.text_encoder: _TextEncoder | None = _TextEncoder()
        self.image_encoder: _ImageEncoder | None = _ImageEncoder()
        self.validated_images: list[Tensor] = []

    def _validate_image_resolution(self, image: Tensor) -> None:
        """Record the image passed through pipeline alignment validation."""
        self.validated_images.append(image)


def _handler() -> tuple[GlobalConditionHandler, _Pipeline]:
    """Build a handler backed by lightweight one-shot encoders."""
    pipeline = _Pipeline()
    return GlobalConditionHandler(cast(OmnidreamsPipeline, pipeline)), pipeline


def test_global_condition_handler_embeds_prompts_and_first_frame() -> None:
    """Verify conversion produces the inference session's embedding layouts."""
    handler, pipeline = _handler()
    first_frame_image = torch.full((1, 1, 1, 3, 8, 16), 0.5)

    condition = handler(
        RawGlobalCondition(
            text_prompt="  drive through a city  ",
            negative_text_prompt="  blurry  ",
            first_frame_image=first_frame_image,
        )
    )

    assert isinstance(condition, InferenceGlobalCondition)
    assert pipeline.validated_images == [first_frame_image]
    assert pipeline.text_encoder is not None
    assert pipeline.text_encoder.calls == [["drive through a city"], ["blurry"]]
    assert pipeline.image_encoder is not None
    assert pipeline.image_encoder.calls == [first_frame_image]
    assert condition.text_embeddings.shape == (1, 1, 2, 4)
    assert condition.negative_text_embeddings is not None
    assert condition.negative_text_embeddings.shape == (1, 1, 2, 4)
    assert condition.image_embeddings.shape == (1, 1, 1, 3, 1, 1)
    torch.testing.assert_close(condition.text_embeddings, torch.ones(1, 1, 2, 4))
    torch.testing.assert_close(
        condition.negative_text_embeddings,
        torch.full((1, 1, 2, 4), 2.0),
    )


@pytest.mark.parametrize(
    "first_frame_image",
    [
        torch.zeros(1, 1, 3, 8, 8),
        torch.zeros(1, 2, 1, 3, 8, 8),
        torch.zeros(1, 1, 1, 4, 8, 8),
        torch.zeros(1, 1, 1, 3, 0, 8),
        torch.zeros(1, 1, 1, 3, 8, 8, dtype=torch.uint8),
    ],
)
def test_global_condition_handler_rejects_invalid_first_frame(
    first_frame_image: Tensor,
) -> None:
    """Verify raw first-frame tensors satisfy the single-view image contract."""
    handler, _pipeline = _handler()

    with pytest.raises(ValidationError):
        handler(
            cast(
                Any,
                {
                    "text_prompt": "city",
                    "negative_text_prompt": "blur",
                    "first_frame_image": first_frame_image,
                },
            )
        )


def test_global_condition_handler_rejects_invalid_prompt_and_extra_fields() -> None:
    """Verify Pydantic validates raw prompt fields and rejects extras."""
    handler, _pipeline = _handler()
    first_frame_image = torch.zeros(1, 1, 1, 3, 8, 8)

    with pytest.raises(ValidationError):
        handler(
            cast(
                Any,
                {
                    "text_prompt": "   ",
                    "negative_text_prompt": "blur",
                    "first_frame_image": first_frame_image,
                },
            )
        )
    with pytest.raises(ValidationError):
        handler(
            cast(
                Any,
                {
                    "text_prompt": "city",
                    "negative_text_prompt": "blur",
                    "first_frame_image": first_frame_image,
                    "unexpected": True,
                },
            )
        )


def test_global_condition_handler_requires_loaded_encoders() -> None:
    """Verify construction fails after either one-shot encoder is released."""
    pipeline = _Pipeline()
    pipeline.text_encoder = None

    with pytest.raises(RuntimeError, match="requires loaded.*text and image"):
        GlobalConditionHandler(cast(OmnidreamsPipeline, pipeline))
