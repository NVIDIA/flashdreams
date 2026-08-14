# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch
from v2v import V2VApplication, V2VApplicationDefaults

from flashdreams.demo import IFlashDreamsApplication, NullInputHandler, NullOutputSink
from flashdreams.runtime.demo import application_runtime

pytestmark = pytest.mark.ci_cpu


class _FakePipeline:
    device = "cpu"

    class diffusion_model:
        dtype = torch.float32

    class encoder:
        target_H = 32
        target_W = 64

    def __init__(self) -> None:
        self.generated: list[torch.Tensor] = []
        self.finalized: list[int] = []
        self.closed = False

    def to(self, device: str) -> "_FakePipeline":
        self.device = device
        return self

    def eval(self) -> "_FakePipeline":
        return self

    def initialize_cache(self) -> object:
        return object()

    def generate(self, index: int, cache: object, source: torch.Tensor) -> torch.Tensor:
        del index, cache
        self.generated.append(source)
        return source

    def finalize(self, index: int, cache: object) -> None:
        del cache
        self.finalized.append(index)

    def close(self) -> None:
        self.closed = True


class _FakePipelineConfig:
    def __init__(self, pipeline: _FakePipeline) -> None:
        self.pipeline = pipeline

    def setup(self) -> _FakePipeline:
        return self.pipeline


def test_v2v_session_streams_cold_and_steady_chunks(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pipeline = _FakePipeline()
    source = tmp_path / "input.mp4"
    source.touch()
    video = torch.arange(29 * 4 * 8 * 3, dtype=torch.uint8).reshape(29, 4, 8, 3).numpy()
    monkeypatch.setattr("v2v.v2v.read_video_rgb", lambda path: video)
    monkeypatch.setattr("v2v.v2v.read_video_fps", lambda path: 24.0)
    application = V2VApplication(
        defaults=V2VApplicationDefaults(
            pipeline_config=_FakePipelineConfig(pipeline),
            first_chunk_frames=5,
            chunk_frames=8,
            default_input_height=4,
            default_input_width=8,
        )
    )
    application.init(["--input-path", str(source), "--device", "cpu"])
    sink = NullOutputSink(store_outputs=True)
    result = application_runtime.run_batch_application_session(
        application=application,
        input_handler=NullInputHandler(),
        input_schema=application.input_schema,
        output_sink=sink,
    )

    assert isinstance(application, IFlashDreamsApplication)
    assert result.status == "completed"
    assert [chunk.shape for chunk in pipeline.generated] == [
        (1, 3, 5, 4, 8),
        (1, 3, 8, 4, 8),
        (1, 3, 8, 4, 8),
        (1, 3, 8, 4, 8),
    ]
    assert pipeline.finalized == [0, 1, 2, 3]
    assert sink.session_info is not None
    assert sink.session_info.frames_per_second == 24.0
    assert sink.session_info.video_height == 32
    assert sink.session_info.video_width == 64


def test_v2v_webrtc_configuration_exposes_upload_ui() -> None:
    application = V2VApplication(
        defaults=V2VApplicationDefaults(
            pipeline_config=_FakePipelineConfig(_FakePipeline()),
            first_chunk_frames=5,
            chunk_frames=8,
            default_input_height=4,
            default_input_width=8,
        )
    )

    class _Factory:
        configuration: object | None = None

        def set_web_configuration(self, configuration: object) -> None:
            self.configuration = configuration

    factory = _Factory()
    application.configure_webrtc(factory)

    assert application.requires_pre_session_web
    assert factory.configuration is not None


def test_v2v_reuses_last_matching_resolution_pipeline() -> None:
    created: list[_FakePipeline] = []

    class _Config:
        def setup(self) -> _FakePipeline:
            pipeline = _FakePipeline()
            created.append(pipeline)
            return pipeline

    application = V2VApplication(
        defaults=V2VApplicationDefaults(
            pipeline_config=_Config(),
            first_chunk_frames=5,
            chunk_frames=8,
            default_input_height=4,
            default_input_width=8,
        )
    )

    first = application._pipeline_for_video(4, 8, "cpu")
    assert application._pipeline_for_video(4, 8, "cpu") is first
    second = application._pipeline_for_video(8, 8, "cpu")

    assert len(created) == 2
    assert second is created[1]
    assert first.closed
