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

"""Inference session lifecycle, model-input envelope, and schema."""

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from flashdreams.infra.pipeline import (
    StreamInferencePipeline,
    StreamInferencePipelineCache,
    StreamInferencePipelineConfig,
)
from flashdreams.runtime._utils import freeze_mapping
from flashdreams.runtime.inputs import InputField, TimeWindow, check_payload
from flashdreams.runtime.types import StepRequest


@dataclass(frozen=True, kw_only=True, slots=True)
class InferenceInput:
    """Global and per-step conditioning for one inference call."""

    __hash__ = None

    global_conditioning: Mapping[str, Any] = field(default_factory=dict)
    per_step_conditioning: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "global_conditioning", freeze_mapping(self.global_conditioning)
        )
        object.__setattr__(
            self,
            "per_step_conditioning",
            freeze_mapping(self.per_step_conditioning),
        )


@dataclass(frozen=True, kw_only=True, slots=True)
class InferenceOutput:
    """Generated output and metadata for one inference step."""

    __hash__ = None

    step_index: int
    """Zero-based index of the completed inference step."""

    output: Any = None
    """Generated payload for the step."""

    frame_count: int | None = None
    """Number of generated frames when the output is frame-based."""

    output_window: TimeWindow | None = None
    """Session time window represented by the generated output."""

    metadata: Mapping[str, Any] = field(default_factory=dict)
    """Output metadata supplied by the session or model adapter."""

    metrics: Mapping[str, float | int] = field(default_factory=dict)
    """Per-step numeric measurements."""

    def __post_init__(self) -> None:
        if self.step_index < 0:
            raise ValueError("InferenceOutput.step_index must be >= 0.")
        if self.frame_count is not None and self.frame_count < 0:
            raise ValueError("InferenceOutput.frame_count must be >= 0.")
        object.__setattr__(self, "metadata", freeze_mapping(self.metadata))
        object.__setattr__(self, "metrics", freeze_mapping(self.metrics))


@dataclass(frozen=True, kw_only=True, slots=True)
class InferenceInputSchema:
    """Required and optional fields in each inference input payload."""

    global_fields: tuple[InputField, ...] = ()
    """Model inputs required before starting the initial generation/session."""

    per_step_fields: tuple[InputField, ...] = ()
    """Per-step model inputs required after the session starts."""

    def check_global_payload(self, inputs: InferenceInput) -> None:
        """Check that required global fields are present in ``inputs``."""
        check_payload(self.global_fields, inputs.global_conditioning)

    def check_per_step_payload(self, inputs: InferenceInput) -> None:
        """Check that required per-step fields are present in ``inputs``."""
        check_payload(self.per_step_fields, inputs.per_step_conditioning)


@dataclass(frozen=True, kw_only=True, slots=True)
class InferenceSessionConfig:
    """Configuration for constructing an inference session."""

    __hash__ = None

    pipeline: StreamInferencePipelineConfig
    """Pipeline configuration to instantiate."""


class InferenceSession(ABC):
    """Stateful inference pipeline session."""

    _pipeline_cache: StreamInferencePipelineCache[Any, Any, Any] | None
    """Pipeline cache for the active rollout; ``None`` before its first step."""

    _step_index: int
    """Zero-based index assigned to the next generated output."""

    def __init__(self, config: InferenceSessionConfig) -> None:
        """Initialize the inference pipeline.

        Args:
            config: Session configuration.
        """
        self.config = config
        # Initialize the inference pipeline from the provided configuration.
        self.pipeline: StreamInferencePipeline = self.config.pipeline.setup()
        self._pipeline_cache = None
        self._step_index = 0

    def __del__(self) -> None:
        """Release session resources."""
        if hasattr(self, "pipeline"):
            del self.pipeline

    def next_step_request(self) -> StepRequest:
        """Return input requirements for the next pipeline step."""
        return StepRequest(step_index=self._step_index)

    @abstractmethod
    def reset(self) -> None:
        """Reset the pipeline and discard the active rollout state."""
        pipeline_reset = getattr(self.pipeline, "reset", None)
        if callable(pipeline_reset):
            pipeline_reset()
        self._pipeline_cache = None
        self._step_index = 0

    @abstractmethod
    def step(self, inference_input: InferenceInput) -> InferenceOutput:
        """Run one inference step.

        Args:
            inference_input: Model-ready inputs for the step.

        Returns:
            Generated output for the step.

        Raises:
            ValueError: Global conditioning is supplied after the rollout starts.
        """
        request = self.next_step_request()
        input_schema = request.inference_input_schema
        if self._pipeline_cache is None:
            if input_schema is not None:
                input_schema.check_global_payload(inference_input)
            self._pipeline_cache = self.pipeline.initialize_cache(
                **inference_input.global_conditioning
            )
        elif inference_input.global_conditioning:
            raise ValueError(
                "InferenceInput.global_conditioning can only be supplied on the "
                "first step after reset()."
            )

        if input_schema is not None:
            input_schema.check_per_step_payload(inference_input)
        pipeline_input = inference_input.per_step_conditioning or None
        output = self.pipeline.generate(
            autoregressive_index=request.step_index,
            cache=self._pipeline_cache,
            input=pipeline_input,
        )
        self.pipeline.finalize(
            autoregressive_index=request.step_index,
            cache=self._pipeline_cache,
        )
        inference_output = InferenceOutput(
            step_index=request.step_index,
            output=output,
            output_window=request.user_input_window,
            metadata=request.metadata,
            metrics={}, # No metrics for now, inject to pipeline later.
        )
        self._step_index = request.step_index + 1
        return inference_output
