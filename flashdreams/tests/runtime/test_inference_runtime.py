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

"""CPU tests for inference runtime pipeline and session ownership."""

from __future__ import annotations

import pytest
import torch
from flashdreams.infra.pipeline import (
    StreamInferencePipeline,
    StreamInferencePipelineCache,
    StreamInferencePipelineConfig,
)
from flashdreams.runtime.inference_runtime import InferenceRuntime
from flashdreams.runtime.inference_session import (
    InferenceInput,
    InferenceOutput,
    InferenceSession,
)
from torch import nn

pytestmark = pytest.mark.ci_cpu


## Runtime test doubles


class _MockStreamInferencePipelineCache(StreamInferencePipelineCache):
    """Pipeline cache mock without model-specific state."""

    def __init__(self) -> None:
        """Initialize an empty cache."""


class _MockStreamInferencePipeline(StreamInferencePipeline):
    """Pipeline mock that records per-session cache initialization."""

    initialize_cache_calls: int
    """Number of caches initialized for created sessions."""

    def __init__(self) -> None:
        nn.Module.__init__(self)
        self.initialize_cache_calls = 0

    def initialize_cache(
        self,
        transformer_context: object | None = None,
        encoder_context: object | None = None,
        decoder_context: object | None = None,
    ) -> _MockStreamInferencePipelineCache:
        """Create and record a fresh mock session cache."""
        del transformer_context, encoder_context, decoder_context
        self.initialize_cache_calls += 1
        return _MockStreamInferencePipelineCache()


class _MockStreamInferencePipelineConfig(StreamInferencePipelineConfig):
    """Pipeline config mock that returns a preconstructed pipeline."""

    pipeline: _MockStreamInferencePipeline
    """Pipeline returned by the setup method."""

    setup_calls: int
    """Number of times the setup method has been called."""

    def __init__(self, pipeline: _MockStreamInferencePipeline) -> None:
        self.pipeline = pipeline
        self.setup_calls = 0

    def setup(self) -> _MockStreamInferencePipeline:
        """Return the configured mock pipeline and record the setup call."""
        self.setup_calls += 1
        return self.pipeline


class _MockInferenceSession(InferenceSession[_MockStreamInferencePipeline]):
    """Session mock that uses the runtime-owned pipeline."""

    def step(self, inference_input: InferenceInput) -> InferenceOutput:
        """Return an empty output without running the mock pipeline."""
        del inference_input
        return InferenceOutput()


class _MockInferenceRuntime(InferenceRuntime[_MockInferenceSession]):
    """Concrete runtime mock with a no-op warmup."""

    def warmup(self) -> None:
        """Complete warmup without running model computation."""


## Fixtures


@pytest.fixture
def runtime_bundle(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[
    _MockInferenceRuntime,
    _MockStreamInferencePipelineConfig,
    _MockStreamInferencePipeline,
]:
    """Build a single-process runtime with mocked pipeline setup."""
    # Keep the fixture on the deterministic non-distributed initialization path.
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)

    pipeline = _MockStreamInferencePipeline()
    pipeline_config = _MockStreamInferencePipelineConfig(pipeline)
    runtime = _MockInferenceRuntime(pipeline_config, _MockInferenceSession)
    return runtime, pipeline_config, pipeline


## Runtime ownership behavior


def test_runtime_sets_up_and_privately_holds_pipeline(
    runtime_bundle: tuple[
        _MockInferenceRuntime,
        _MockStreamInferencePipelineConfig,
        _MockStreamInferencePipeline,
    ],
) -> None:
    """Verify runtime construction sets up and retains one pipeline."""
    runtime, pipeline_config, pipeline = runtime_bundle

    assert pipeline_config.setup_calls == 1
    assert runtime._pipeline is pipeline
    assert runtime._session_type is _MockInferenceSession
    assert runtime._local_rank == 0
    assert runtime._global_rank == 0
    assert runtime._world_size == 1
    assert runtime._is_rank_zero


def test_create_session_privately_shares_pipeline_and_initializes_fresh_cache(
    runtime_bundle: tuple[
        _MockInferenceRuntime,
        _MockStreamInferencePipelineConfig,
        _MockStreamInferencePipeline,
    ],
) -> None:
    """Verify created sessions share the pipeline but own separate caches."""
    runtime, pipeline_config, pipeline = runtime_bundle

    first_session = runtime.create_session()
    second_session = runtime.create_session()

    assert pipeline_config.setup_calls == 1
    assert first_session is not second_session
    assert first_session._pipeline is pipeline
    assert second_session._pipeline is pipeline
    assert isinstance(first_session._cache, _MockStreamInferencePipelineCache)
    assert isinstance(second_session._cache, _MockStreamInferencePipelineCache)
    assert first_session._cache is not second_session._cache
    assert pipeline.initialize_cache_calls == 2
