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

from typing import Any

import pytest
import torch
from pydantic import ValidationError

from .mocks import (
    MockGlobalCondition,
    MockStreamInferencePipeline,
    MockUserCondition,
    ValidatedInferenceSession,
)

pytestmark = pytest.mark.ci_cpu


## Fixtures and condition factories


@pytest.fixture
def session() -> ValidatedInferenceSession:
    """Create a session backed by the lightweight pipeline double."""
    return ValidatedInferenceSession(MockStreamInferencePipeline())


def _user_condition() -> MockUserCondition:
    """Build a complete per-step condition for validation tests."""
    return MockUserCondition(
        movement=torch.tensor([1.0, 0.0, -1.0]),
        camera=torch.eye(4),
    )


def _global_condition() -> MockGlobalCondition:
    """Build a complete rollout-wide condition for validation tests."""
    return MockGlobalCondition(
        frame=torch.zeros(3, 8, 8),
        prompt=torch.ones(4, 16),
    )


## Accepted session inputs


def test_step_validates_nested_conditions(session: ValidatedInferenceSession) -> None:
    """Verify complete nested conditions pass step validation."""
    user_condition = _user_condition()
    global_condition = _global_condition()
    inference_input: Any = {
        "user_condition": user_condition,
        "global_condition": global_condition,
    }

    # Pass a raw mapping so ``step`` performs Pydantic validation and conversion.
    output = session.step(inference_input)

    assert torch.equal(output.frame_chunk, global_condition.frame)


def test_step_accepts_missing_optional_global_condition(
    session: ValidatedInferenceSession,
) -> None:
    """Verify step accepts an omitted optional global condition."""
    user_condition = _user_condition()
    inference_input: Any = {"user_condition": user_condition}

    output = session.step(inference_input)

    assert torch.equal(output.frame_chunk, user_condition.camera)


## Rejected session inputs


def test_step_rejects_missing_user_condition(
    session: ValidatedInferenceSession,
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
    session: ValidatedInferenceSession,
    missing_field: str,
) -> None:
    """Verify step rejects a user condition missing a required tensor field."""
    user_condition = _user_condition().model_dump()
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
    session: ValidatedInferenceSession,
    missing_field: str,
) -> None:
    """Verify step rejects a global condition missing a required tensor field."""
    global_condition = _global_condition().model_dump()
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
