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

"""CPU contract tests for the reusable upsample-video application."""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest
import torch
from flashdreams.api_v2.session import ISession
from flashdreams.infra.postprocess import (
    VideoChunk,
    VideoPostProcessorConfig,
    VideoSpec,
)
from flashdreams.runtime_v2.session_desc import SessionDesc
from flashdreams.runtime_v2.user_input_events import UserInputEvents
from flashdreams.runtime_v2.video_tensor import VideoTensorLayout
from upsample_video import (
    LoadedVideo,
    UpsampleVideoApplication,
    UpsampleVideoApplicationDefaults,
)

pytestmark = pytest.mark.ci_cpu

_INPUT_SPEC = VideoSpec(height=64, width=128, fps=24.0)
"""Smallest convenient 2x input whose output axes are both 128-aligned."""


@dataclass(slots=True)
class _FakeProcessorSession:
    """Record input chunks and emit a nearest-neighbor 2x stand-in."""

    inputs: list[VideoChunk]
    """Chunks consumed by the fake processor."""

    prepared: bool = False
    """Whether the application invoked the preparation hook."""

    flushed: bool = False
    """Whether end-of-stream flushing has run."""

    def prepare(self) -> None:
        self.prepared = True

    def process(self, chunk: VideoChunk) -> list[VideoChunk]:
        self.inputs.append(chunk)
        output = chunk.tensor.repeat_interleave(2, dim=-2).repeat_interleave(2, dim=-1)
        return [VideoChunk(tensor=output.unsqueeze(0), layout="btchw")]

    def flush(self) -> list[VideoChunk]:
        self.flushed = True
        return []


@dataclass(slots=True)
class _FakeProcessor:
    """Create fake processor sessions for an application test."""

    sessions: list[_FakeProcessorSession]
    """Sessions created across initialization and resets."""

    def start(self, spec: VideoSpec) -> _FakeProcessorSession:
        assert spec == _INPUT_SPEC
        session = _FakeProcessorSession(inputs=[])
        self.sessions.append(session)
        return session


@dataclass(kw_only=True)
class _FakeProcessorConfig(VideoPostProcessorConfig):
    """Configure the CPU-only upsampling stand-in."""

    sessions: list[_FakeProcessorSession] = field(default_factory=list)
    """Sessions created by this config."""

    def output_spec(self, input_spec: VideoSpec) -> VideoSpec:
        return VideoSpec(
            height=input_spec.height * 2,
            width=input_spec.width * 2,
            fps=input_spec.fps,
        )

    def setup(self) -> _FakeProcessor:
        return _FakeProcessor(self.sessions)


def _loaded_video(frame_count: int) -> LoadedVideo:
    frames = torch.linspace(
        -1.0,
        1.0,
        steps=frame_count * 3 * _INPUT_SPEC.height * _INPUT_SPEC.width,
    ).reshape(frame_count, 3, _INPUT_SPEC.height, _INPUT_SPEC.width)
    return LoadedVideo(frames=frames, spec=_INPUT_SPEC)


def _application(
    requested_frames: list[int],
    processor_sessions: list[_FakeProcessorSession] | None = None,
) -> UpsampleVideoApplication:
    def load(frame_count: int) -> LoadedVideo:
        requested_frames.append(frame_count)
        return _loaded_video(frame_count)

    return UpsampleVideoApplication(
        defaults=UpsampleVideoApplicationDefaults(
            processor=_FakeProcessorConfig(
                sessions=processor_sessions if processor_sessions is not None else []
            ),
            first_chunk_size=13,
            steady_chunk_size=16,
            model_name="fake-2x",
        ),
        input_loader=load,
        input_spec=_INPUT_SPEC,
    )


def _session() -> tuple[ISession, list[_FakeProcessorSession], list[int]]:
    processor_sessions: list[_FakeProcessorSession] = []
    requested_frames: list[int] = []
    application = _application(requested_frames, processor_sessions)
    application.init(["--max-chunks", "2"])
    session = application.create_session(application.session_desc())
    session.init()
    return session, processor_sessions, requested_frames


def test_application_advertises_the_upsampled_video_contract() -> None:
    application = _application([])

    desc = application.session_desc()

    assert desc.output_layout is VideoTensorLayout.bcthw
    assert (desc.video_width, desc.video_height) == (256, 128)
    assert desc.frames_per_second_for_ui == 24
    assert desc.frames_per_second_for_step == 24
    assert desc.metadata["application"] == "upsample-video"
    assert desc.metadata["model"] == "fake-2x"


def test_model_loop_upsamples_cold_and_steady_chunks() -> None:
    session, processor_sessions, requested_frames = _session()
    model_loop = session.model_loop

    cold = model_loop.step(0, UserInputEvents([]))[0]
    steady = model_loop.step(1, UserInputEvents([]))[0]

    assert requested_frames == [29]
    assert cold.read_output().shape == (1, 3, 13, 128, 256)
    assert cold.output_layout is VideoTensorLayout.bcthw
    assert cold.frame_count == 13
    assert steady.read_output().shape == (1, 3, 16, 128, 256)
    assert steady.frame_count == 16
    assert model_loop.is_finished()
    assert processor_sessions[0].prepared
    assert processor_sessions[0].flushed
    assert [chunk.tensor.shape[0] for chunk in processor_sessions[0].inputs] == [
        13,
        16,
    ]


def test_reset_restarts_from_the_cold_chunk() -> None:
    session, processor_sessions, _ = _session()
    model_loop = session.model_loop
    first = model_loop.step(0, UserInputEvents([]))[0].read_output()

    model_loop.reset()
    repeated = model_loop.step(0, UserInputEvents([]))[0].read_output()

    assert torch.equal(repeated, first)
    assert len(processor_sessions) == 2


def test_init_rejects_nonpositive_chunk_count() -> None:
    with pytest.raises(ValueError, match="--max-chunks"):
        _application([]).init(["--max-chunks", "0"])


def test_create_session_before_init_raises() -> None:
    with pytest.raises(RuntimeError, match="init"):
        _application([]).create_session(_application([]).session_desc())


def test_create_session_rejects_an_incompatible_layout() -> None:
    application = _application([])
    application.init(["--max-chunks", "1"])
    desc = application.session_desc()
    incompatible = SessionDesc(
        output_layout=VideoTensorLayout.tchw,
        video_width=desc.video_width,
        video_height=desc.video_height,
    )

    with pytest.raises(ValueError, match="bcthw"):
        application.create_session(incompatible)
