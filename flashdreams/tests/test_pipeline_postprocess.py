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

"""CPU tests for pipeline-level streaming post-processing."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from flashdreams.infra.config import InstantiateConfig
from flashdreams.infra.pipeline import (
    StreamInferencePipeline,
    StreamInferencePipelineConfig,
)
from flashdreams.infra.postprocess import (
    VideoChunk,
    VideoPostprocessChainConfig,
    VideoPostProcessor,
    VideoPostProcessorConfig,
    VideoPostProcessorSession,
    VideoSpec,
)

pytestmark = pytest.mark.ci_cpu


class _FakeTransformer:
    def initialize_autoregressive_cache(self) -> SimpleNamespace:
        return SimpleNamespace()


@dataclass(kw_only=True)
class _FakeDiffusionModelConfig(InstantiateConfig):
    _target: type["_FakeDiffusionModel"] = field(
        default_factory=lambda: _FakeDiffusionModel
    )

    output: torch.Tensor = field(
        default_factory=lambda: torch.arange(
            2 * 3 * 4 * 5, dtype=torch.float32
        ).reshape(2, 3, 4, 5)
    )
    """Decoded output returned by the fake diffusion model."""


class _FakeDiffusionModel(torch.nn.Module):
    def __init__(self, config: _FakeDiffusionModelConfig) -> None:
        super().__init__()
        self.config = config
        self.transformer = _FakeTransformer()

    @property
    def device(self) -> torch.device:
        return torch.device("cpu")

    def generate(
        self,
        autoregressive_index: int,
        cache: SimpleNamespace,
        input: object = None,
    ) -> tuple[torch.Tensor, SimpleNamespace]:
        del autoregressive_index, cache, input
        return self.config.output.clone(), SimpleNamespace()

    def finalize(self, final_state: SimpleNamespace) -> None:
        del final_state


@dataclass(kw_only=True)
class _AddOnePostProcessorConfig(VideoPostProcessorConfig):
    _target: type["_AddOnePostProcessor"] = field(
        default_factory=lambda: _AddOnePostProcessor
    )


class _AddOnePostProcessor(VideoPostProcessor[_AddOnePostProcessorConfig]):
    def start(self, spec: VideoSpec) -> VideoPostProcessorSession:
        del spec
        return _AddOnePostProcessorSession()


class _AddOnePostProcessorSession(VideoPostProcessorSession):
    def process(self, chunk: VideoChunk) -> list[VideoChunk]:
        return [
            VideoChunk(
                tensor=chunk.tensor + 1,
                layout=chunk.layout,
                value_range=chunk.value_range,
                metadata=chunk.metadata,
            )
        ]

    def flush(self) -> list[VideoChunk]:
        return []


@dataclass(kw_only=True)
class _BufferUntilFlushPostProcessorConfig(VideoPostProcessorConfig):
    _target: type["_BufferUntilFlushPostProcessor"] = field(
        default_factory=lambda: _BufferUntilFlushPostProcessor
    )


class _BufferUntilFlushPostProcessor(
    VideoPostProcessor[_BufferUntilFlushPostProcessorConfig]
):
    def start(self, spec: VideoSpec) -> VideoPostProcessorSession:
        del spec
        return _BufferUntilFlushPostProcessorSession()


class _BufferUntilFlushPostProcessorSession(VideoPostProcessorSession):
    def __init__(self) -> None:
        self._chunk: VideoChunk | None = None

    def process(self, chunk: VideoChunk) -> list[VideoChunk]:
        self._chunk = chunk
        return []

    def flush(self) -> list[VideoChunk]:
        if self._chunk is None:
            return []
        chunk = self._chunk
        self._chunk = None
        return [
            VideoChunk(
                tensor=chunk.tensor + 2,
                layout=chunk.layout,
                value_range=chunk.value_range,
                metadata=chunk.metadata,
            )
        ]


def _pipeline(
    postprocessor: VideoPostProcessorConfig,
) -> tuple[StreamInferencePipeline[Any, Any, Any], torch.Tensor]:
    output = torch.arange(2 * 3 * 4 * 5, dtype=torch.float32).reshape(2, 3, 4, 5)
    config = StreamInferencePipelineConfig(
        name="test-postprocess-pipeline",
        diffusion_model=_FakeDiffusionModelConfig(output=output),
        postprocess=VideoPostprocessChainConfig(processors=(postprocessor,)),
        postprocess_output_layout="tchw",
    )
    return config.setup(), output


def test_generate_returns_postprocessed_output_and_keeps_raw_cache() -> None:
    pipeline, raw = _pipeline(_AddOnePostProcessorConfig())
    cache = pipeline.initialize_cache()

    output = pipeline.generate(0, cache)

    assert torch.equal(output, raw + 1)
    assert torch.equal(cache.last_raw_output, raw)
    assert torch.equal(cache.last_output, output)


def test_buffered_postprocess_flush_preserves_original_generate_output() -> None:
    pipeline, raw = _pipeline(_BufferUntilFlushPostProcessorConfig())
    cache = pipeline.initialize_cache()

    output = pipeline.generate(0, cache)
    tail = pipeline.flush_postprocess(cache)

    assert output.shape == (0, 3, 4, 5)
    assert torch.equal(cache.last_raw_output, raw)
    assert tail is not None
    assert torch.equal(tail, raw + 2)
