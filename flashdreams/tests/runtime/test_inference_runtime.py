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
from flashdreams.runtime.inference_runtime import (
    InferenceRuntime,
    InferenceRuntimeConfig,
)

from .mocks import (
    MockInferenceSession,
    MockStreamInferencePipeline,
    MockStreamInferencePipelineCache,
    MockStreamInferencePipelineConfig,
)

pytestmark = pytest.mark.ci_cpu


## Runtime test doubles


class _MockInferenceRuntime(InferenceRuntime[MockInferenceSession]):
    """Concrete runtime mock with a no-op warmup."""

    def warmup(self) -> None:
        """Complete warmup without running model computation."""


## Fixtures


@pytest.fixture
def runtime_bundle(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[
    _MockInferenceRuntime,
    MockStreamInferencePipelineConfig,
    MockStreamInferencePipeline,
]:
    """Build a single-process runtime with mocked pipeline setup."""
    # Keep the fixture on the deterministic non-distributed initialization path.
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)

    pipeline = MockStreamInferencePipeline()
    pipeline_config = MockStreamInferencePipelineConfig(pipeline)
    runtime_config = InferenceRuntimeConfig(
        _target=_MockInferenceRuntime,
        pipeline=pipeline_config,
        session_type=MockInferenceSession,
    )
    runtime = runtime_config.setup()
    return runtime, pipeline_config, pipeline


## Runtime ownership behavior


def test_runtime_sets_up_and_privately_holds_pipeline(
    runtime_bundle: tuple[
        _MockInferenceRuntime,
        MockStreamInferencePipelineConfig,
        MockStreamInferencePipeline,
    ],
) -> None:
    """Verify runtime construction sets up and retains one pipeline."""
    runtime, pipeline_config, pipeline = runtime_bundle

    assert pipeline_config.setup_calls == 1
    assert runtime._pipeline is pipeline
    assert runtime._session_type is MockInferenceSession
    assert runtime._local_rank == 0
    assert runtime._global_rank == 0
    assert runtime._world_size == 1
    assert runtime._is_rank_zero


def test_create_session_privately_shares_pipeline_and_initializes_fresh_cache(
    runtime_bundle: tuple[
        _MockInferenceRuntime,
        MockStreamInferencePipelineConfig,
        MockStreamInferencePipeline,
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
    assert isinstance(first_session._cache, MockStreamInferencePipelineCache)
    assert isinstance(second_session._cache, MockStreamInferencePipelineCache)
    assert first_session._cache is not second_session._cache
    assert pipeline.initialize_cache_calls == 2
