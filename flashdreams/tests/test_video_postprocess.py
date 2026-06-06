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

"""CPU tests for generic video post-processing utilities."""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest
import torch

from flashdreams.infra.postprocess import (
    VideoChunk,
    VideoPostprocessChainConfig,
    VideoPostProcessor,
    VideoPostProcessorConfig,
    VideoPostProcessorSession,
    VideoSpec,
    concatenate_video_chunks,
    from_bvtchw,
    postprocess_video_tensor,
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
        return _RepeatSpatialSession(self.config.scale)


class _RepeatSpatialSession(VideoPostProcessorSession):
    def __init__(self, scale: int) -> None:
        self._scale = scale

    def process(self, chunk: VideoChunk) -> list[VideoChunk]:
        canonical = to_bvtchw(
            to_minus_one_one(chunk.tensor, value_range=chunk.value_range),
            layout=chunk.layout,
        )
        resized = resize_bvtchw(
            canonical,
            height=canonical.shape[-2] * self._scale,
            width=canonical.shape[-1] * self._scale,
            mode="nearest",
        )
        return [VideoChunk(tensor=resized, layout="bvtchw")]

    def flush(self) -> list[VideoChunk]:
        return []


@dataclass(kw_only=True)
class _DelayUntilConfig(VideoPostProcessorConfig):
    _target: type["_DelayUntil"] = field(default_factory=lambda: _DelayUntil)
    min_frames: int = 2


class _DelayUntil(VideoPostProcessor[_DelayUntilConfig]):
    def start(self, spec: VideoSpec) -> VideoPostProcessorSession:
        return _DelayUntilSession(self.config.min_frames)


class _DelayUntilSession(VideoPostProcessorSession):
    def __init__(self, min_frames: int) -> None:
        self._min_frames = min_frames
        self._buffer: torch.Tensor | None = None

    def process(self, chunk: VideoChunk) -> list[VideoChunk]:
        canonical = to_bvtchw(
            to_minus_one_one(chunk.tensor, value_range=chunk.value_range),
            layout=chunk.layout,
        )
        self._buffer = (
            canonical
            if self._buffer is None
            else torch.cat([self._buffer, canonical], dim=2)
        )
        if self._buffer.shape[2] < self._min_frames:
            return []
        ready = self._buffer
        self._buffer = None
        return [VideoChunk(tensor=ready, layout="bvtchw")]

    def flush(self) -> list[VideoChunk]:
        if self._buffer is None:
            return []
        tail = self._buffer
        self._buffer = None
        return [VideoChunk(tensor=tail, layout="bvtchw")]


def test_noop_chain_returns_same_tchw_tensor() -> None:
    video = torch.linspace(-1.0, 1.0, steps=3 * 3 * 4 * 5).reshape(3, 3, 4, 5)

    result = postprocess_video_tensor(
        video,
        layout="tchw",
        value_range="minus_one_one",
        postprocess=VideoPostprocessChainConfig(),
        fps=24,
    )

    assert torch.equal(result, video)


def test_layout_and_uint8_round_trip() -> None:
    video = torch.randint(0, 256, (2, 4, 5, 3), dtype=torch.uint8)
    canonical = to_bvtchw(video, layout="thwc")
    restored = from_bvtchw(canonical, layout="thwc")

    assert torch.equal(restored, video)
    normalized = to_minus_one_one(video, value_range="uint8")
    assert normalized.dtype == torch.float32
    assert normalized.min() >= -1.0
    assert normalized.max() <= 1.0


def test_chain_can_resize_and_preserve_requested_layout_and_range() -> None:
    video = torch.zeros((1, 2, 2, 2, 3), dtype=torch.uint8)
    video[..., 0] = 255
    chain = VideoPostprocessChainConfig(processors=(_RepeatSpatialConfig(scale=2),))

    result = postprocess_video_tensor(
        video,
        layout="bthwc",
        value_range="uint8",
        postprocess=chain,
        fps=16,
    )

    assert result.shape == (1, 2, 4, 4, 3)
    assert result.dtype == torch.uint8
    assert torch.equal(result[..., 0], torch.full((1, 2, 4, 4), 255, dtype=torch.uint8))
    assert torch.equal(result[..., 1:], torch.zeros((1, 2, 4, 4, 2), dtype=torch.uint8))


def test_chain_flushes_buffered_chunks_through_downstream_processors() -> None:
    chunk_a = VideoChunk(tensor=torch.zeros((1, 3, 1, 2, 2)), layout="bcthw")
    chunk_b = VideoChunk(tensor=torch.ones((1, 3, 1, 2, 2)), layout="bcthw")
    chain = VideoPostprocessChainConfig(
        processors=(
            _DelayUntilConfig(min_frames=3),
            _RepeatSpatialConfig(scale=2),
        )
    ).setup(VideoSpec(height=2, width=2, fps=8))

    assert chain.process(chunk_a) == []
    assert chain.process(chunk_b) == []
    flushed = chain.flush()
    result = concatenate_video_chunks(
        flushed,
        layout="bcthw",
        value_range="minus_one_one",
    )

    assert result.shape == (1, 3, 2, 4, 4)
    assert torch.equal(result[:, :, 0], torch.zeros((1, 3, 4, 4)))
    assert torch.equal(result[:, :, 1], torch.ones((1, 3, 4, 4)))
