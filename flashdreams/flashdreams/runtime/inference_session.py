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

"""Inference session contracts with pipeline and cache ownership."""

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from flashdreams.infra.pipeline import (
    StreamInferencePipeline,
    StreamInferencePipelineCache,
)
from pydantic import BaseModel, ConfigDict, validate_call, with_config
from typing_extensions import NotRequired, TypedDict


@with_config(ConfigDict(arbitrary_types_allowed=True, extra="forbid"))
class InferenceUserCondition(TypedDict):
    """Base typed dictionary for per-step user conditions."""


@with_config(ConfigDict(arbitrary_types_allowed=True, extra="forbid"))
class InferenceGlobalCondition(TypedDict):
    """Base typed dictionary for rollout-wide conditions."""


UserConditionT = TypeVar("UserConditionT", bound=InferenceUserCondition)
"""User-condition type parameter for :class:`InferenceInput`."""

GlobalConditionT = TypeVar("GlobalConditionT", bound=InferenceGlobalCondition)
"""Global-condition type parameter for :class:`InferenceInput`."""


@with_config(ConfigDict(arbitrary_types_allowed=True, extra="forbid"))
class InferenceInput(TypedDict, Generic[UserConditionT, GlobalConditionT]):
    """Validated conditions consumed by one inference step."""

    user_condition: UserConditionT
    """Required per-step user condition."""

    global_condition: NotRequired[GlobalConditionT | None]
    """Optional rollout-wide condition."""


class InferenceOutput(BaseModel):
    """Base model for outputs produced by one inference step."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")


# TODO: Replace StreamInferencePipeline with the flashdreams.pipeline module.
PipelineT = TypeVar("PipelineT", bound=StreamInferencePipeline)
"""Pipeline type parameter for :class:`InferenceSession`."""


class InferenceSession(ABC, Generic[PipelineT]):
    """Stateful interface around an inference pipeline and session cache.

    Subclasses implement :meth:`step` for integration-specific rollout I/O.
    """

    pipeline: PipelineT
    """Pipeline owned and driven by the inference session."""

    cache: StreamInferencePipelineCache
    """Current per-session cache initialized by the pipeline."""

    def __init__(self, pipeline: PipelineT) -> None:
        """Initialize the session and reset its pipeline cache.

        Args:
            pipeline: Pipeline to drive.
        """
        self.pipeline = pipeline
        self.reset()

    def reset(self) -> None:
        """Reset the session with a fresh pipeline cache."""
        self.cache = self.pipeline.initialize_cache()

    @abstractmethod
    @validate_call
    def step(self, inference_input: InferenceInput) -> InferenceOutput:
        """Run one inference step.

        Args:
            inference_input: Input for the next inference step.

        Returns:
            Output produced by the inference step.

        Raises:
            ValidationError: ``inference_input`` fails Pydantic validation.
        """
