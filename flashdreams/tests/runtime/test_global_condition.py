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

"""CPU tests for raw global-condition conversion contracts."""

from typing import Any, cast

import pytest
from flashdreams.runtime.global_condition import (
    GlobalConditionHandler,
    RawGlobalCondition,
)
from flashdreams.runtime.inference_session import InferenceGlobalCondition
from pydantic import ValidationError, validate_call

pytestmark = pytest.mark.ci_cpu


class _RawPromptCondition(RawGlobalCondition):
    """Raw prompt supplied by an application boundary."""

    prompt: str
    """Unprocessed rollout prompt."""


class _PromptCondition(InferenceGlobalCondition):
    """Model-ready prompt condition used by the test handler."""

    prompt: str
    """Normalized rollout prompt."""


class _PromptConditionHandler(GlobalConditionHandler):
    """Normalize a raw prompt into an inference condition."""

    @validate_call
    def __call__(self, raw_global_condition: _RawPromptCondition) -> _PromptCondition:
        """Normalize whitespace around the rollout prompt."""
        return _PromptCondition(prompt=raw_global_condition["prompt"].strip())


def test_global_condition_handler_validates_and_converts_raw_condition() -> None:
    """Verify a concrete handler receives validated typed-dictionary data."""
    handler = _PromptConditionHandler()

    condition = handler(_RawPromptCondition(prompt="  drive forward  "))

    assert condition.prompt == "drive forward"


def test_global_condition_handler_rejects_invalid_raw_condition() -> None:
    """Verify Pydantic rejects invalid raw condition fields and extras."""
    handler = _PromptConditionHandler()

    with pytest.raises(ValidationError):
        handler(cast(Any, {"prompt": 42}))
    with pytest.raises(ValidationError):
        handler(cast(Any, {"prompt": "drive", "unexpected": True}))
