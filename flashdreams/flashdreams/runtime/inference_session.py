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

"""Inference session lifecycle and model-input envelope."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, TypedDict

from typing_extensions import Unpack

from flashdreams.infra.pipeline import (
    StreamInferencePipeline,
    StreamInferencePipelineConfig,
)
from flashdreams.runtime._utils import freeze_mapping
from flashdreams.runtime.inputs import InputPhase, validate_phase


@dataclass(frozen=True, kw_only=True, slots=True)
class InferenceInput:
    """Encoded inputs for one :class:`InferenceSession` call.

    Two conditioning slots:

    - ``global_conditioning``: values that condition the whole rollout, such as
      the conditioning frame or prompt. Normally supplied when the session
      starts.
    - ``step``: values needed to generate the next chunk or frame.

    A non-empty ``global_conditioning`` on a mid-rollout input is an *update
    request*, not a reset. The session should apply it when the model supports
    that; resetting rollout state is a separate, explicit
    :meth:`InferenceSession.reset` call. Whether a given value can be updated
    mid-rollout is declared by ``InputField.update_policy``; see
    ``InferenceInputSchema.unsupported_global_updates``.
    """

    __hash__ = None

    global_conditioning: Mapping[str, Any] = field(default_factory=dict)
    step: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "global_conditioning", freeze_mapping(self.global_conditioning)
        )
        object.__setattr__(self, "step", freeze_mapping(self.step))
        object.__setattr__(self, "metadata", freeze_mapping(self.metadata))

    @property
    def requests_global_update(self) -> bool:
        """Return whether this input asks the session to update conditioning."""
        return bool(self.global_conditioning)

    def with_step(self, step: Mapping[str, Any]) -> "InferenceInput":
        """Return a copy with replaced per-step payload.

        The global slot is carried through unchanged, so a mid-rollout input
        built this way keeps whatever update request it already had. Use
        :meth:`without_global_update` for the common steady-state case.
        """
        return InferenceInput(
            global_conditioning=self.global_conditioning,
            step=step,
            metadata=self.metadata,
        )

    def with_global_update(
        self, global_conditioning: Mapping[str, Any]
    ) -> "InferenceInput":
        """Return a copy requesting a mid-rollout conditioning update."""
        return InferenceInput(
            global_conditioning=global_conditioning,
            step=self.step,
            metadata=self.metadata,
        )

    def without_global_update(self) -> "InferenceInput":
        """Return a copy that requests no conditioning update."""
        return InferenceInput(step=self.step, metadata=self.metadata)

    def for_phase(self, phase: InputPhase) -> Mapping[str, Any]:
        """Return the payload mapping for ``phase``."""
        return (
            self.global_conditioning if validate_phase(phase) == "global" else self.step
        )


class InferenceSessionConfig(TypedDict):
    """Configuration for constructing an inference session."""

    pipeline: StreamInferencePipelineConfig
    """Pipeline configuration to instantiate."""


class InferenceSession:
    """Stateful inference pipeline session."""

    def __init__(self, **kwargs: Unpack[InferenceSessionConfig]) -> None:
        """Initialize the inference pipeline.

        Args:
            **kwargs: Session construction keyword arguments.
        """
        # Initialize the inference pipeline from the provided configuration.
        self.pipeline: StreamInferencePipeline = kwargs["pipeline"].setup()

    def __del__(self) -> None:
        """Release session resources."""
        if hasattr(self, "pipeline"):
            del self.pipeline

    def reset(self) -> None:
        """Reset the inference session."""

    def step(self, inference_input: InferenceInput) -> None:
        """Run one inference step.

        Args:
            inference_input: Model-ready inputs for the step.
        """
