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

"""CPU tests for the T2V application runtime and session boundary."""

from __future__ import annotations

import argparse
from typing import Any, cast

import pytest
import torch
import torch.nn as nn
import t2v_app
from t2v_app import application
from t2v_app.presets import PipelinePreset, PresetCatalog, RuntimePresetOptions
from t2v_app.runtime import T2VRuntime
from t2v_app.session import T2VSession

from flashdreams.infra.pipeline import (
    StreamInferencePipeline,
    StreamInferencePipelineConfig,
)
from flashdreams.runtime import InferenceInput
from flashdreams_runner import (
    Application,
    ApplicationArguments,
    DriveSession,
    IOHandler,
    Runtime,
)

pytestmark = pytest.mark.ci_cpu


def _preset() -> tuple[
    str,
    PipelinePreset[RuntimePresetOptions],
    PresetCatalog[RuntimePresetOptions],
]:
    preset_id = "test-t2v"
    preset = PipelinePreset(
        provider="tests:provider",
        runtime=RuntimePresetOptions(
            prompt="default prompt",
            total_blocks=2,
            pixel_height=64,
            pixel_width=96,
            fps=12,
            output_layout="tchw",
        ),
        pipeline={},
    )
    return (
        preset_id,
        preset,
        PresetCatalog(default_preset_id=preset_id, presets={preset_id: preset}),
    )


def test_application_module_conforms_to_single_factory_abi() -> None:
    assert isinstance(t2v_app, Application)
    assert cast(Any, t2v_app).__all__ == ["create_runtime"]


def test_create_runtime_parses_options_without_constructing_pipeline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipeline_constructed = False

    class Pipeline(StreamInferencePipeline[Any, Any, Any]):
        def __init__(self, config: object) -> None:
            nn.Module.__init__(self)
            del config
            nonlocal pipeline_constructed
            pipeline_constructed = True

    preset_id, _, catalog = _preset()
    pipeline_config = StreamInferencePipelineConfig(
        _target=cast(Any, Pipeline),
        name=preset_id,
        diffusion_model=cast(Any, None),
    )

    class PipelineProvider:
        def create_pipeline_config(
            self, *, preset_id: str, options: object
        ) -> StreamInferencePipelineConfig:
            assert preset_id == "test-t2v"
            assert options == {}
            return pipeline_config

    monkeypatch.setattr(application, "load_preset_catalog", lambda _: catalog)
    monkeypatch.setattr(
        application, "load_pipeline_provider", lambda _: PipelineProvider()
    )
    arguments = ApplicationArguments(
        mode="webrtc",
        parser=argparse.ArgumentParser(),
        argv=("--prompt", "A waterfall"),
    )

    runtime = application.create_runtime(arguments)

    assert isinstance(runtime, T2VRuntime)
    assert runtime.config.video_width == 96
    assert runtime.config.fps == 12
    assert runtime.config.default_steps == 2
    assert arguments.options.prompt == "A waterfall"
    assert not pipeline_constructed


def test_runtime_owns_pipeline_and_session_owns_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    class Cache:
        def close(self) -> None:
            calls.append("cache.close")

    class Decoder:
        spatial_compression_ratio = 8

    monkeypatch.setattr("t2v_app.session.StreamingVideoDecoder", Decoder)

    class Pipeline(StreamInferencePipeline[Any, Any, Any]):
        def __init__(self, config: object) -> None:
            nn.Module.__init__(self)
            del config
            self.decoder = cast(Any, Decoder())
            calls.append("pipeline.init")

        def to(self, device: object) -> "Pipeline":
            assert str(device) == "cpu"
            calls.append("pipeline.to")
            return self

        def eval(self) -> "Pipeline":
            calls.append("pipeline.eval")
            return self

        def initialize_cache(self, **kwargs: object) -> Cache:
            assert kwargs == {
                "text": ["A waterfall"],
                "image": None,
                "height": 8,
                "width": 12,
            }
            calls.append("session.initialize_cache")
            return Cache()

        def get_num_output_frames(self, autoregressive_index: int) -> int:
            assert autoregressive_index == 1
            return 3

        def generate(self, autoregressive_index: int, cache: object) -> torch.Tensor:
            assert autoregressive_index == 0
            assert isinstance(cache, Cache)
            calls.append("session.pipeline.generate")
            return torch.zeros((3, 3, 2, 2))

        def finalize(
            self, autoregressive_index: int, cache: object
        ) -> dict[str, float]:
            assert autoregressive_index == 0
            assert isinstance(cache, Cache)
            calls.append("session.pipeline.finalize")
            return {"step_ms": 1.0}

        def close(self) -> None:
            calls.append("pipeline.close")

    preset_id, _, catalog = _preset()
    pipeline_config = StreamInferencePipelineConfig(
        _target=cast(Any, Pipeline),
        name=preset_id,
        diffusion_model=cast(Any, None),
    )

    class PipelineProvider:
        def create_pipeline_config(
            self, *, preset_id: str, options: object
        ) -> StreamInferencePipelineConfig:
            del preset_id, options
            return pipeline_config

    monkeypatch.setattr(application, "load_preset_catalog", lambda _: catalog)
    monkeypatch.setattr(
        application, "load_pipeline_provider", lambda _: PipelineProvider()
    )
    runtime = application.create_runtime(
        ApplicationArguments(
            mode="mp4",
            parser=argparse.ArgumentParser(),
            argv=("--prompt", "A waterfall"),
        )
    )

    class Mode:
        name = "test"

        def run(self, runtime: Runtime, drive_session: DriveSession) -> tuple[()]:
            del runtime, drive_session
            return ()

    mode = Mode()
    assert isinstance(mode, IOHandler)
    runtime.initialize(device="cpu", io_handler=mode)
    assert runtime.peek_steady_output_num_frames() == 3
    session = runtime.create_session(InferenceInput())
    assert isinstance(session, T2VSession)
    result = session.step(InferenceInput())
    assert result.step_index == 0
    assert result.frame_count == 3
    assert result.metrics["step_ms"] == 1.0
    session.destroy()
    runtime.destroy()

    assert calls == [
        "pipeline.init",
        "pipeline.to",
        "pipeline.eval",
        "session.initialize_cache",
        "session.pipeline.generate",
        "session.pipeline.finalize",
        "cache.close",
        "pipeline.close",
    ]


def test_total_blocks_is_only_a_finite_mode_default() -> None:
    defaults = RuntimePresetOptions(
        prompt="default prompt",
        pixel_height=64,
        pixel_width=96,
        fps=12,
        output_layout="tchw",
    )

    assert application._total_steps({"total_blocks": None}, defaults) is None
    assert application._total_steps({"total_blocks": 4}, defaults) == 4
