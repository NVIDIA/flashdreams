# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for stateful output-stream post-processing."""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest
import torch

import flashdreams.infra.postprocess.stream as postprocess_stream_module
from flashdreams.infra.postprocess import (
    VideoChunk,
    VideoPostprocessChainConfig,
    VideoPostProcessor,
    VideoPostProcessorConfig,
    VideoPostProcessorSession,
    VideoPostprocessStream,
    VideoSpec,
)
from flashdreams.infra.postprocess.base import _VideoPostprocessChainSession

pytestmark = pytest.mark.ci_cpu


@dataclass(kw_only=True)
class _BufferConfig(VideoPostProcessorConfig):
    _target: type["_Buffer"] = field(default_factory=lambda: _Buffer)


class _Buffer(VideoPostProcessor[_BufferConfig]):
    def start(self, spec: VideoSpec) -> VideoPostProcessorSession:
        return _BufferSession(spec)


class _BufferSession(VideoPostProcessorSession):
    def __init__(self, spec: VideoSpec) -> None:
        self.spec = spec
        self.chunk: VideoChunk | None = None
        self.prepare_calls = 0

    def prepare(self) -> None:
        self.prepare_calls += 1

    def process(self, chunk: VideoChunk) -> list[VideoChunk]:
        self.chunk = chunk
        return []

    def flush(self) -> list[VideoChunk]:
        if self.chunk is None:
            return []
        chunk, self.chunk = self.chunk, None
        return [chunk]


@dataclass(kw_only=True)
class _ScaleSpecConfig(_BufferConfig):
    scale: int = 2

    def output_spec(self, input_spec: VideoSpec) -> VideoSpec:
        return VideoSpec(
            height=input_spec.height * self.scale,
            width=input_spec.width * self.scale,
            fps=input_spec.fps,
            channels=input_spec.channels,
        )


def _stream(*processors: VideoPostProcessorConfig, per_view: bool = False):
    return VideoPostprocessStream(
        postprocess=VideoPostprocessChainConfig(processors=processors),
        output_layout="bvtchw" if per_view else "tchw",
        per_view=per_view,
    )


def test_stream_buffers_then_flushes_once_and_closes() -> None:
    stream = _stream(_BufferConfig())
    video = torch.ones(3, 3, 4, 5)

    output = stream.process(video, autoregressive_index=0)
    tail = stream.finish()

    assert output.shape[0] == 0
    assert tail is not None and torch.equal(tail, video)
    assert stream.finish() is None
    with pytest.raises(RuntimeError, match="after finish"):
        stream.process(video, autoregressive_index=1)


def test_stream_prepares_session_once_before_processing() -> None:
    stream = _stream(_BufferConfig())
    first = torch.ones(3, 3, 4, 5)
    second = torch.ones(2, 3, 4, 5)

    stream.process(first, autoregressive_index=0)
    session = stream.state.sessions[-1]
    assert isinstance(session, _VideoPostprocessChainSession)
    processor_session = session._sessions[0]
    assert isinstance(processor_session, _BufferSession)
    assert processor_session.prepare_calls == 1

    stream.process(second, autoregressive_index=1)
    assert processor_session.prepare_calls == 1


def test_stream_profiles_buffering_as_a_separate_ar_stat(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeEventProfiler:
        def __init__(self, *, synchronize_distributed: bool) -> None:
            assert not synchronize_distributed

        def record(self, stage: str) -> None:
            assert stage == "postprocess"

        def sync_and_summarize(self) -> dict[str, float]:
            return {"postprocess": 0.125}

    monkeypatch.setattr(postprocess_stream_module, "EventProfiler", _FakeEventProfiler)
    stream = VideoPostprocessStream(
        postprocess=VideoPostprocessChainConfig(processors=(_BufferConfig(),)),
        output_layout="tchw",
        profile=True,
    )
    video = torch.ones(3, 3, 4, 5)

    output = stream.process(video, autoregressive_index=0)
    stats = stream.add_process_stats({"total_ms": 10.0})

    assert output.shape[0] == 0
    assert stats == {
        "total_ms": 10.0,
        "postprocess": {
            "elapsed_ms": 0.125,
            "input_frames": 3,
            "output_frames": 0,
            "buffering": True,
        },
    }


def test_stream_collects_generated_chunks_without_postprocess() -> None:
    stream = VideoPostprocessStream(
        postprocess=VideoPostprocessChainConfig(),
        output_layout="bcthw",
    )
    first = torch.ones((1, 3, 2, 4, 5))
    empty = torch.empty((1, 3, 0, 4, 5))
    second = torch.full((1, 3, 1, 4, 5), 2.0)

    assert stream.process(first, autoregressive_index=0) is first
    stream.process(empty, autoregressive_index=1)
    stream.process(second, autoregressive_index=2)
    output = stream.finish()

    assert output is not None
    assert output.shape == (1, 3, 3, 4, 5)
    assert torch.equal(output[:, :, :2], first)
    assert torch.equal(output[:, :, 2:], second)


def test_chain_propagates_output_spec_to_downstream_session() -> None:
    chain = VideoPostprocessChainConfig(
        processors=(_ScaleSpecConfig(scale=2), _BufferConfig())
    )
    session = chain.setup(VideoSpec(height=4, width=6, fps=12))

    assert isinstance(session, _VideoPostprocessChainSession)
    assert isinstance(session._sessions[1], _BufferSession)
    assert session._sessions[1].spec == VideoSpec(height=8, width=12, fps=12)


def test_stream_rejects_input_spec_changes() -> None:
    stream = _stream(_BufferConfig())
    stream.process(torch.ones(3, 3, 4, 5), autoregressive_index=0)

    with pytest.raises(ValueError, match="specification changed"):
        stream.process(torch.ones(3, 3, 8, 5), autoregressive_index=1)


class _ContentDependentFlushSession(_BufferSession):
    def flush(self) -> list[VideoChunk]:
        if self.chunk is None or not bool(self.chunk.tensor.any()):
            return []
        return super().flush()


class _ContentDependentFlush(_Buffer):
    def start(self, spec: VideoSpec) -> VideoPostProcessorSession:
        return _ContentDependentFlushSession(spec)


@dataclass(kw_only=True)
class _ContentDependentFlushConfig(VideoPostProcessorConfig):
    _target: type["_ContentDependentFlush"] = field(
        default_factory=lambda: _ContentDependentFlush
    )


def test_per_view_flush_rejects_asymmetric_output() -> None:
    stream = _stream(_ContentDependentFlushConfig(), per_view=True)
    video = torch.zeros(1, 2, 3, 3, 4, 5)
    video[:, 1] = 1
    stream.process(video, autoregressive_index=0)

    with pytest.raises(ValueError, match="all views or none"):
        stream.finish()
