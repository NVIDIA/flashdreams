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

"""CPU contract tests for SwiftVR post-processing."""

from __future__ import annotations

from typing import Any

import pytest
import swiftvr.impl.postprocess as postprocess
import torch
from swiftvr.impl.postprocess import SwiftVRPostProcessorConfig

from flashdreams.infra.postprocess import (
    VideoChunk,
    VideoSpec,
)
from flashdreams.infra.postprocess.base import concatenate_video_chunks

pytestmark = pytest.mark.ci_cpu


class _FakeStream:
    """Mimic SwiftVR's three-frame cold-start trim and four-frame tail."""

    def __init__(self, output_height: int, output_width: int) -> None:
        self.output_height = output_height
        self.output_width = output_width
        self.first = True
        self.buffered = 0

    def step(self, frames: torch.Tensor) -> torch.Tensor | None:
        self.buffered += frames.shape[0]
        ready = self.buffered // 4 * 4
        self.buffered -= ready
        if ready == 0:
            return None
        output_frames = ready - 3 if self.first else ready
        self.first = False
        return torch.zeros((1, output_frames, 3, self.output_height, self.output_width))

    def flush(self) -> torch.Tensor | None:
        if self.buffered == 0:
            return None
        self.buffered = 0
        return torch.zeros((1, 4, 3, self.output_height, self.output_width))


class _FakePipeline:
    def __init__(self) -> None:
        self.device = torch.device("cpu")
        self.starts: list[dict[str, int]] = []

    def initialize_cache(self, **kwargs: int) -> _FakeStream:
        self.starts.append(kwargs)
        return _FakeStream(kwargs["output_height"], kwargs["output_width"])

    def generate(
        self,
        autoregressive_index: int,
        cache: _FakeStream,
        frames: torch.Tensor,
    ) -> torch.Tensor | None:
        return cache.step(frames)

    def finalize(self, autoregressive_index: int, cache: _FakeStream) -> None:
        pass

    def flush(self, cache: _FakeStream) -> torch.Tensor | None:
        return cache.flush()


def _install_fake_pipeline(monkeypatch: pytest.MonkeyPatch) -> list[_FakePipeline]:
    created: list[_FakePipeline] = []

    def load(_: SwiftVRPostProcessorConfig) -> _FakePipeline:
        pipeline = _FakePipeline()
        created.append(pipeline)
        return pipeline

    monkeypatch.setattr(postprocess, "_load_swiftvr_pipeline", load)
    return created


def test_swiftvr_preserves_arbitrary_frame_counts_and_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created = _install_fake_pipeline(monkeypatch)
    config = SwiftVRPostProcessorConfig(
        device="cpu", chunk_size=8, scale=2, prewarm=False
    )
    session = config.setup().start(VideoSpec(height=4, width=6, fps=30))
    first = session.process(
        VideoChunk(
            tensor=torch.zeros((3, 3, 4, 6)),
            layout="tchw",
            metadata={"chunk": 0},
        )
    )
    second = session.process(
        VideoChunk(
            tensor=torch.zeros((10, 3, 4, 6)),
            layout="tchw",
            metadata={"chunk": 1},
        )
    )
    tail = session.flush()
    output = concatenate_video_chunks([*first, *second, *tail], layout="tchw")

    assert first == []
    assert output.shape == (13, 3, 8, 12)
    assert len(created) == 1
    assert created[0].starts == [{"output_height": 8, "output_width": 12, "overlap": 0}]
    assert second[0].metadata == {
        "source": "swiftvr",
        "input_chunks": ({"chunk": 0}, {"chunk": 1}),
    }
    assert tail[0].metadata == {
        "source": "swiftvr_tail",
        "input_chunks": ({"chunk": 1},),
    }


@pytest.mark.parametrize("frame_count", [1, 2, 3, 4, 5, 8, 9, 17])
def test_swiftvr_flush_emits_one_frame_per_input(
    monkeypatch: pytest.MonkeyPatch, frame_count: int
) -> None:
    _install_fake_pipeline(monkeypatch)
    config = SwiftVRPostProcessorConfig(
        device="cpu", chunk_size=8, scale=2, prewarm=False
    )
    session = config.setup().start(VideoSpec(height=4, width=4))
    outputs = session.process(
        VideoChunk(tensor=torch.zeros((frame_count, 3, 4, 4)), layout="tchw")
    )
    outputs.extend(session.flush())

    result = concatenate_video_chunks(outputs, layout="tchw")
    assert result.shape[0] == frame_count


def test_swiftvr_reuses_resident_pipeline_across_sessions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created = _install_fake_pipeline(monkeypatch)
    config = SwiftVRPostProcessorConfig(device="cpu", chunk_size=8, prewarm=True)
    processor = config.setup()

    first = processor.start(VideoSpec(height=4, width=4))
    first.prepare()
    second = config.setup().start(VideoSpec(height=4, width=4))
    second.prepare()

    assert config.setup() is processor
    assert len(created) == 1
    # One prewarm stream plus one fresh state object for each session.
    assert len(created[0].starts) == 3


def test_swiftvr_reports_exact_scaled_output_spec() -> None:
    output = SwiftVRPostProcessorConfig(scale=4).output_spec(
        VideoSpec(height=360, width=640, fps=30)
    )

    assert output == VideoSpec(height=1440, width=2560, fps=30)


def test_swiftvr_rejects_empty_input_chunk(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_pipeline(monkeypatch)
    session = (
        SwiftVRPostProcessorConfig(device="cpu", prewarm=False)
        .setup()
        .start(VideoSpec(height=4, width=4))
    )

    with pytest.raises(ValueError, match="at least one frame"):
        session.process(VideoChunk(tensor=torch.zeros((0, 3, 4, 4)), layout="tchw"))


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"scale": 0}, "scale"),
        ({"chunk_size": 6}, "chunk_size"),
        ({"dit_overlap": -1}, "dit_overlap"),
    ],
)
def test_swiftvr_rejects_invalid_config(kwargs: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        SwiftVRPostProcessorConfig(**kwargs)
