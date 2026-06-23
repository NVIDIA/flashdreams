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

"""CPU checks for Omnidreams runner output post-processing."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest
import torch
from omnidreams.runner import OmnidreamsRunner

from flashdreams.infra.postprocess import (
    VideoChunk,
    VideoPostprocessChainConfig,
    VideoPostProcessor,
    VideoPostProcessorConfig,
    VideoPostProcessorSession,
    VideoSpec,
    resize_bvtchw,
    to_bvtchw,
    to_minus_one_one,
)

pytestmark = pytest.mark.ci_cpu


@dataclass(kw_only=True)
class _RepeatSpatialConfig(VideoPostProcessorConfig):
    _target: type["_RepeatSpatial"] = field(default_factory=lambda: _RepeatSpatial)
    scale: int = 2


class _RepeatSpatial(VideoPostProcessor[_RepeatSpatialConfig]):
    def start(self, spec: VideoSpec) -> VideoPostProcessorSession:
        return _RepeatSpatialSession(scale=self.config.scale)


class _RepeatSpatialSession(VideoPostProcessorSession):
    def __init__(self, *, scale: int) -> None:
        self._scale = scale

    def process(self, chunk: VideoChunk) -> list[VideoChunk]:
        canonical = to_bvtchw(
            to_minus_one_one(chunk.tensor, value_range=chunk.value_range),
            layout=chunk.layout,
        )
        scaled = resize_bvtchw(
            canonical,
            height=canonical.shape[-2] * self._scale,
            width=canonical.shape[-1] * self._scale,
            mode="nearest",
        )
        return [VideoChunk(tensor=scaled, layout="bvtchw")]

    def flush(self) -> list[VideoChunk]:
        return []


def _runner_with_postprocess(
    postprocess: VideoPostprocessChainConfig,
) -> OmnidreamsRunner:
    runner = object.__new__(OmnidreamsRunner)
    runner.config = SimpleNamespace(postprocess=postprocess)
    return runner


def test_default_output_keeps_hdmap_and_generated_canvas() -> None:
    runner = _runner_with_postprocess(VideoPostprocessChainConfig())
    condition = torch.full((1, 1, 2, 3, 2, 3), -1.0)
    video = torch.full((1, 1, 2, 3, 2, 3), 1.0)

    canvas, description = runner._prepare_canvas_for_write(
        condition=condition,
        video=video,
        fps=30,
    )

    assert description == "HDMap/RGB canvas"
    assert canvas.shape == (2, 4, 3, 3)
    assert torch.equal(canvas[:, :2], torch.full((2, 2, 3, 3), -1.0))
    assert torch.equal(canvas[:, 2:], torch.full((2, 2, 3, 3), 1.0))


def test_postprocess_output_writes_scaled_generated_rgb_only() -> None:
    runner = _runner_with_postprocess(
        VideoPostprocessChainConfig(processors=(_RepeatSpatialConfig(scale=2),))
    )
    condition = torch.full((1, 1, 2, 3, 2, 3), -1.0)
    video = torch.full((1, 1, 2, 3, 2, 3), 1.0)

    canvas, description = runner._prepare_canvas_for_write(
        condition=condition,
        video=video,
        fps=30,
    )

    assert description == "postprocessed RGB video"
    assert canvas.shape == (2, 4, 6, 3)
    assert torch.equal(canvas, torch.full((2, 4, 6, 3), 1.0))


def test_postprocess_handles_each_generated_view_independently() -> None:
    runner = _runner_with_postprocess(
        VideoPostprocessChainConfig(processors=(_RepeatSpatialConfig(scale=2),))
    )
    condition = torch.full((1, 2, 2, 3, 2, 3), -1.0)
    video = torch.empty((1, 2, 2, 3, 2, 3))
    video[:, 0] = -0.5
    video[:, 1] = 0.5

    canvas, _description = runner._prepare_canvas_for_write(
        condition=condition,
        video=video,
        fps=30,
    )

    assert canvas.shape == (2, 4, 12, 3)
    assert torch.equal(canvas[:, :, :6], torch.full((2, 4, 6, 3), -0.5))
    assert torch.equal(canvas[:, :, 6:], torch.full((2, 4, 6, 3), 0.5))
