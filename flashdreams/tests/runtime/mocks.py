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

"""Shared inference pipeline and session test doubles."""

from __future__ import annotations

from typing import TypeAlias

from flashdreams.infra.pipeline import (
    StreamInferencePipeline,
    StreamInferencePipelineCache,
    StreamInferencePipelineConfig,
)
from flashdreams.runtime.inference_session import (
    InferenceGlobalCondition,
    InferenceInput,
    InferenceOutput,
    InferenceSession,
    InferenceUserCondition,
)
from pydantic import validate_call
from torch import Tensor, nn


class MockStreamInferencePipelineCache(StreamInferencePipelineCache):
    """In-memory pipeline cache without model-specific state."""

    def __init__(self) -> None:
        """Initialize an empty cache."""


class MockStreamInferencePipeline(StreamInferencePipeline):
    """Pipeline test double that records cache initialization."""

    initialize_cache_calls: int
    """Number of caches initialized for inference sessions."""

    def __init__(self) -> None:
        """Initialize the pipeline without model components."""
        nn.Module.__init__(self)
        self.initialize_cache_calls = 0

    def initialize_cache(
        self,
        transformer_context: object | None = None,
        encoder_context: object | None = None,
        decoder_context: object | None = None,
    ) -> MockStreamInferencePipelineCache:
        """Create and record a fresh inference-session cache."""
        del transformer_context, encoder_context, decoder_context
        self.initialize_cache_calls += 1
        return MockStreamInferencePipelineCache()


class MockStreamInferencePipelineConfig(StreamInferencePipelineConfig):
    """Pipeline config test double that returns a configured pipeline."""

    pipeline: MockStreamInferencePipeline
    """Pipeline returned by :meth:`setup`."""

    setup_calls: int
    """Number of times :meth:`setup` has been called."""

    def __init__(self, pipeline: MockStreamInferencePipeline) -> None:
        """Initialize with the pipeline returned by :meth:`setup`."""
        self.pipeline = pipeline
        self.setup_calls = 0

    def setup(self) -> MockStreamInferencePipeline:
        """Return the configured pipeline and record the setup call."""
        self.setup_calls += 1
        return self.pipeline


class MockInferenceSession(InferenceSession[MockStreamInferencePipeline]):
    """Inference session test double that records inputs and outputs."""

    def __init__(
        self,
        pipeline: MockStreamInferencePipeline | None = None,
    ) -> None:
        """Initialize with a supplied or fresh mock pipeline."""
        self.inputs: list[InferenceInput] = []
        self.outputs: list[InferenceOutput] = []
        super().__init__(
            pipeline if pipeline is not None else MockStreamInferencePipeline()
        )

    def step(self, inference_input: InferenceInput) -> InferenceOutput:
        """Record an input and return a unique empty output."""
        inference_output = InferenceOutput()
        self.inputs.append(inference_input)
        self.outputs.append(inference_output)
        return inference_output


class MockUserCondition(InferenceUserCondition):
    """User-provided controls for validated inference steps."""

    movement: Tensor
    """Embedded latent tensor describing character movement."""

    camera: Tensor
    """Embedded latent tensor describing camera rotation."""


class MockGlobalCondition(InferenceGlobalCondition):
    """Session-wide controls for validated inference steps."""

    frame: Tensor
    """Embedded latent tensor describing the global conditioning frame."""

    prompt: Tensor
    """Embedded latent tensor describing prompt conditioning."""


MockInferenceInput: TypeAlias = InferenceInput[MockUserCondition, MockGlobalCondition]
"""Inference input with fully specialized nested condition models."""


class MockInferenceOutput(InferenceOutput):
    """Output returned by the validated inference session."""

    frame_chunk: Tensor
    """Fully decoded frame chunk from the model latent."""


class ValidatedInferenceSession(InferenceSession[MockStreamInferencePipeline]):
    """Inference session with Pydantic-validated condition models."""

    @validate_call
    def step(self, inference_input: MockInferenceInput) -> MockInferenceOutput:
        """Return a frame chunk from the validated inference input."""
        global_condition = inference_input.global_condition
        frame_chunk = (
            global_condition.frame
            if global_condition is not None
            else inference_input.user_condition.camera
        )
        return MockInferenceOutput(frame_chunk=frame_chunk)


__all__ = [
    "MockGlobalCondition",
    "MockInferenceInput",
    "MockInferenceOutput",
    "MockInferenceSession",
    "MockStreamInferencePipeline",
    "MockStreamInferencePipelineCache",
    "MockStreamInferencePipelineConfig",
    "MockUserCondition",
    "ValidatedInferenceSession",
]
