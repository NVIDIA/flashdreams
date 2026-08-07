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

"""Pydantic validation tests for inference sessions."""

from __future__ import annotations

from typing import Any, TypeAlias

import pytest
import torch
from flashdreams.infra.pipeline import (
    StreamInferencePipeline,
    StreamInferencePipelineCache,
)
from flashdreams.runtime.inference_session import (
    InferenceGlobalCondition,
    InferenceInput,
    InferenceOutput,
    InferenceSession,
    InferenceUserCondition,
)
from pydantic import ValidationError, validate_call
from torch import Tensor, nn

pytestmark = pytest.mark.ci_cpu


## Pipeline test doubles


class _MockStreamInferencePipelineCache(StreamInferencePipelineCache):
    """In-memory cache mock without model-specific state."""

    def __init__(self) -> None:
        """Initialize a cache without model-specific state."""


class _MockStreamInferencePipeline(StreamInferencePipeline):
    """Pipeline mock that creates an in-memory cache without model setup."""

    def __init__(self) -> None:
        nn.Module.__init__(self)

    def initialize_cache(
        self,
        transformer_context: object | None = None,
        encoder_context: object | None = None,
        decoder_context: object | None = None,
    ) -> _MockStreamInferencePipelineCache:
        """Return a fresh cache without constructing model components."""
        del transformer_context, encoder_context, decoder_context
        return _MockStreamInferencePipelineCache()


## Session condition and output contracts


class _MockUserCondition(InferenceUserCondition):
    """User-provided controls for the mock inference step."""

    movement: Tensor
    """Embedded latent tensor describing character movement."""

    camera: Tensor
    """Embedded latent tensor describing camera rotation."""


class _MockGlobalCondition(InferenceGlobalCondition):
    """Session-wide controls for the mock inference step."""

    frame: Tensor
    """Embedded latent tensor describing the global conditioning frame."""

    prompt: Tensor
    """Embedded latent tensor describing prompt conditioning."""


# Specialize both nested dictionaries so ``validate_call`` sees their fields.
_MockInferenceInput: TypeAlias = InferenceInput[
    _MockUserCondition, _MockGlobalCondition
]


class _MockInferenceOutput(InferenceOutput):
    """Output returned by the mock inference session."""

    frame_chunk: Tensor
    """Fully decoded frame chunk from the model latent; for WAN, its shape is
    ``[4, H, W, 3]``."""


class _MockInferenceSession(InferenceSession[_MockStreamInferencePipeline]):
    """Inference session with concrete condition dictionaries."""

    @validate_call
    def step(self, inference_input: _MockInferenceInput) -> _MockInferenceOutput:
        """Return a frame chunk from the validated inference input."""
        global_condition = inference_input.get("global_condition")
        frame_chunk = (
            global_condition["frame"]
            if global_condition is not None
            else inference_input["user_condition"]["camera"]
        )
        return _MockInferenceOutput(frame_chunk=frame_chunk)


## Fixtures and condition factories


@pytest.fixture
def session() -> _MockInferenceSession:
    """Create a session backed by the lightweight pipeline double."""
    return _MockInferenceSession(_MockStreamInferencePipeline())


def _user_condition() -> _MockUserCondition:
    """Build a complete per-step condition for validation tests."""
    return _MockUserCondition(
        movement=torch.tensor([1.0, 0.0, -1.0]),
        camera=torch.eye(4),
    )


def _global_condition() -> _MockGlobalCondition:
    """Build a complete rollout-wide condition for validation tests."""
    return _MockGlobalCondition(
        frame=torch.zeros(3, 8, 8),
        prompt=torch.ones(4, 16),
    )


## Accepted session inputs


def test_step_validates_nested_conditions(session: _MockInferenceSession) -> None:
    """Verify complete nested conditions pass step validation."""
    user_condition = _user_condition()
    global_condition = _global_condition()
    inference_input: Any = {
        "user_condition": user_condition,
        "global_condition": global_condition,
    }

    # Pass a raw mapping so ``step`` performs Pydantic validation and conversion.
    output = session.step(inference_input)

    assert torch.equal(output.frame_chunk, global_condition["frame"])


def test_step_accepts_missing_optional_global_condition(
    session: _MockInferenceSession,
) -> None:
    """Verify step accepts an omitted optional global condition."""
    user_condition = _user_condition()
    inference_input: Any = {"user_condition": user_condition}

    output = session.step(inference_input)

    assert torch.equal(output.frame_chunk, user_condition["camera"])


## Rejected session inputs


def test_step_rejects_missing_user_condition(
    session: _MockInferenceSession,
) -> None:
    """Verify step rejects an omitted required user condition."""
    inference_input: Any = {"global_condition": _global_condition()}

    with pytest.raises(ValidationError) as exc_info:
        session.step(inference_input)

    assert any(
        error["loc"][-1:] == ("user_condition",) for error in exc_info.value.errors()
    )


@pytest.mark.parametrize("missing_field", ["movement", "camera"])
def test_step_rejects_missing_user_field(
    session: _MockInferenceSession,
    missing_field: str,
) -> None:
    """Verify step rejects a user condition missing a required tensor field."""
    user_condition = dict(_user_condition())
    del user_condition[missing_field]
    inference_input: Any = {
        "user_condition": user_condition,
        "global_condition": _global_condition(),
    }

    with pytest.raises(ValidationError) as exc_info:
        session.step(inference_input)

    assert any(
        error["loc"][-2:] == ("user_condition", missing_field)
        for error in exc_info.value.errors()
    )


@pytest.mark.parametrize("missing_field", ["frame", "prompt"])
def test_step_rejects_missing_global_field(
    session: _MockInferenceSession,
    missing_field: str,
) -> None:
    """Verify step rejects a global condition missing a required tensor field."""
    global_condition = dict(_global_condition())
    del global_condition[missing_field]
    inference_input: Any = {
        "user_condition": _user_condition(),
        "global_condition": global_condition,
    }

    with pytest.raises(ValidationError) as exc_info:
        session.step(inference_input)

    assert any(
        error["loc"][-2:] == ("global_condition", missing_field)
        for error in exc_info.value.errors()
    )
