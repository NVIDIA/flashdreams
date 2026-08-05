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

"""Tests for inference-session pipeline orchestration."""

from typing import Any, cast

import pytest
import torch

from flashdreams.infra.pipeline import (
    StreamInferencePipeline,
    StreamInferencePipelineConfig,
)
from flashdreams.runtime.inference_session import (
    InferenceInput,
    InferenceInputSchema,
    InferenceOutput,
    InferenceSession,
    InferenceSessionConfig,
)
from flashdreams.runtime.inputs import TimeWindow
from flashdreams.runtime.types import StepRequest

pytestmark = pytest.mark.ci_cpu


class FakeStreamInferencePipeline(StreamInferencePipeline[Any, Any, Any]):
    """Record session orchestration without constructing model components."""

    def __init__(self, output: Any) -> None:
        torch.nn.Module.__init__(self)
        self.output = output
        self.cache = object()
        self.reset_calls = 0
        self.initialize_cache_calls: list[dict[str, Any]] = []
        self.generate_calls: list[dict[str, Any]] = []
        self.finalize_calls: list[dict[str, Any]] = []

    def reset(self) -> None:
        self.reset_calls += 1

    def initialize_cache(self, **global_conditioning: Any) -> object:
        self.initialize_cache_calls.append(global_conditioning)
        self.cache = object()
        return self.cache

    def generate(
        self, autoregressive_index: int, cache: object, input: Any = None
    ) -> Any:
        self.generate_calls.append(
            {
                "autoregressive_index": autoregressive_index,
                "cache": cache,
                "input": input,
            }
        )
        return self.output

    def finalize(self, autoregressive_index: int, cache: object) -> dict[str, float]:
        self.finalize_calls.append(
            {"autoregressive_index": autoregressive_index, "cache": cache}
        )
        return {"total_ms": 4.0}


class _FakePipelineConfig:
    def __init__(self, pipeline: FakeStreamInferencePipeline) -> None:
        self.pipeline = pipeline
        self.setup_calls = 0

    def setup(self) -> FakeStreamInferencePipeline:
        self.setup_calls += 1
        return self.pipeline


class _ConcreteInferenceSession(InferenceSession):
    def next_step_request(self) -> StepRequest:
        return StepRequest(
            step_index=self._step_index,
            inference_input_schema=InferenceInputSchema(),
            user_input_window=TimeWindow(
                start_s=float(self._step_index),
                end_s=float(self._step_index + 1),
            ),
            metadata={"request": "fake"},
        )

    def reset(self) -> None:
        super().reset()

    def step(self, inference_input: InferenceInput) -> InferenceOutput:
        return super().step(inference_input)


def _create_session(
    pipeline: FakeStreamInferencePipeline,
) -> tuple[_ConcreteInferenceSession, _FakePipelineConfig]:
    pipeline_config = _FakePipelineConfig(pipeline)
    config = InferenceSessionConfig(
        pipeline=cast(StreamInferencePipelineConfig, pipeline_config)
    )
    return _ConcreteInferenceSession(config), pipeline_config


def test_constructor_initializes_the_configured_pipeline() -> None:
    pipeline = FakeStreamInferencePipeline(output=object())

    session, pipeline_config = _create_session(pipeline)

    assert session.pipeline is pipeline
    assert pipeline_config.setup_calls == 1


def test_reset_resets_the_pipeline_and_rollout_state() -> None:
    pipeline = FakeStreamInferencePipeline(output=object())
    session, _ = _create_session(pipeline)
    session.step(
        InferenceInput(
            global_conditioning={"prompt": "first"},
            per_step_conditioning={"control": 1},
        )
    )

    session.reset()
    result = session.step(
        InferenceInput(
            global_conditioning={"prompt": "second"},
            per_step_conditioning={"control": 2},
        )
    )

    assert pipeline.reset_calls == 1
    assert pipeline.initialize_cache_calls == [
        {"prompt": "first"},
        {"prompt": "second"},
    ]
    assert result.step_index == 0


def test_step_converts_inference_input_and_wraps_pipeline_output() -> None:
    generated = object()
    pipeline = FakeStreamInferencePipeline(output=generated)
    session, _ = _create_session(pipeline)
    inference_input = InferenceInput(
        global_conditioning={"prompt": "drive"},
        per_step_conditioning={"steering": 0.25},
    )

    result = session.step(inference_input)

    assert pipeline.initialize_cache_calls == [{"prompt": "drive"}]
    assert pipeline.generate_calls == [
        {
            "autoregressive_index": 0,
            "cache": pipeline.cache,
            "input": {"steering": 0.25},
        }
    ]
    assert pipeline.finalize_calls == [
        {"autoregressive_index": 0, "cache": pipeline.cache}
    ]
    assert result == InferenceOutput(
        step_index=0,
        output=generated,
        output_window=TimeWindow(start_s=0.0, end_s=1.0),
        metadata={"request": "fake"},
        metrics={"total_ms": 4.0},
    )


def test_step_reuses_the_pipeline_cache_and_advances_the_index() -> None:
    pipeline = FakeStreamInferencePipeline(output=object())
    session, _ = _create_session(pipeline)
    session.step(InferenceInput(global_conditioning={"prompt": "drive"}))

    result = session.step(InferenceInput(per_step_conditioning={"steering": -0.5}))

    assert pipeline.initialize_cache_calls == [{"prompt": "drive"}]
    assert pipeline.generate_calls[-1] == {
        "autoregressive_index": 1,
        "cache": pipeline.cache,
        "input": {"steering": -0.5},
    }
    assert result.step_index == 1
    assert session.next_step_request().step_index == 2
