# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for the Self-Forcing application, against a stand-in model.

What the real model generates cannot be asserted on a CPU runner, or cheaply
anywhere. What can be is everything around it: that the model is loaded once,
that a rollout is described and initialized before it generates, that each step
continues the last, and that the frames reach a file. The stand-in generates
frames of the shape and range the real pipeline does, so the seam this covers
is the one the real model plugs into.
"""

import shutil
from pathlib import Path
from typing import Any

import pytest
import torch
from t2v_self_forcing import SelfForcingT2VApplication, default_session_desc

from flashdreams.runtime_v2.session_desc import SessionDesc
from flashdreams.runtime_v2.user_input_events import UserInputEvents
from flashdreams.runtime_v2.video_tensor import VideoTensorLayout
from flashdreams.testing_v2.t2v_conformance import (
    ExpectedFrameStats,
    check_t2v_model_impl,
)

pytestmark = pytest.mark.ci_cpu

needs_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None,
    reason="writing an MP4 needs ffmpeg on PATH",
)

_WIDTH = 128
"""Frame width the stand-in generates. Not square, so a transposed frame cannot
pass unnoticed, and a multiple of the compression ratio."""

_HEIGHT = 64
"""Frame height it generates."""

_COMPRESSION_RATIO = 8
"""Pixels per latent in each direction, as the real Wan decoder has."""

_FIRST_BLOCK_FRAMES = 9
"""Frames the first block decodes, as the real causal decoder emits."""

_BLOCK_FRAMES = 12
"""Frames every block after it decodes."""

_PROMPT = "A cat surfing"
"""Prompt the tests generate from."""

_NO_EVENTS = UserInputEvents([])
"""What a text-to-video session is given every step."""


## Stand-in model


class _FakeDecoder:
    spatial_compression_ratio = _COMPRESSION_RATIO


class _FakePipeline:
    """Generate frames of the shape and range the real pipeline generates."""

    def __init__(self) -> None:
        self.decoder = _FakeDecoder()
        self.device: str | None = None
        self.eval_count = 0
        self.caches: list[dict[str, Any]] = []
        self.generated: list[int] = []
        self.finalized: list[int] = []
        self.closed = False
        self._frames_generated = 0

    def to(self, device: str) -> "_FakePipeline":
        self.device = device
        return self

    def eval(self) -> "_FakePipeline":
        self.eval_count += 1
        return self

    def initialize_cache(self, **kwargs: Any) -> object:
        self.caches.append(kwargs)
        self._frames_generated = 0
        return object()

    def generate(self, *, autoregressive_index: int, cache: object) -> torch.Tensor:
        del cache
        self.generated.append(autoregressive_index)
        count = _FIRST_BLOCK_FRAMES if autoregressive_index == 0 else _BLOCK_FRAMES
        frames = torch.stack(
            [self._frame(self._frames_generated + index) for index in range(count)]
        )
        self._frames_generated += count
        return frames

    def finalize(self, *, autoregressive_index: int, cache: object) -> dict[str, float]:
        del cache
        self.finalized.append(autoregressive_index)
        return {"total_ms": 1.5}

    def close(self) -> None:
        self.closed = True

    def _frame(self, frame_index: int) -> torch.Tensor:
        """Return a grey frame whose shade moves with time.

        Mid grey rather than black or white, and moving rather than still, so
        the checks a caller makes of a real video are meaningful here too.
        """
        shade = -0.5 + (frame_index % 8) / 8.0
        return torch.full((3, _HEIGHT, _WIDTH), shade, dtype=torch.float32)


class _FakePipelineConfig:
    def __init__(self, pipeline: _FakePipeline) -> None:
        self.pipeline = pipeline
        self.setup_count = 0

    def setup(self) -> _FakePipeline:
        self.setup_count += 1
        return self.pipeline


## Helpers


def _session_desc(
    layout: VideoTensorLayout = VideoTensorLayout.tchw,
    *,
    width: int = _WIDTH,
    height: int = _HEIGHT,
) -> SessionDesc:
    return SessionDesc(
        output_layout=layout,
        frames_per_second_for_ui=60,
        frames_per_second_for_step=16,
        video_width=width,
        video_height=height,
    )


def _application(
    pipeline: _FakePipeline | None = None,
) -> tuple[SelfForcingT2VApplication, _FakePipeline]:
    """Return an initialized application and the model it will load."""
    pipeline = pipeline or _FakePipeline()
    app = SelfForcingT2VApplication(pipeline_config=_FakePipelineConfig(pipeline))
    app.init(["--prompt", _PROMPT, "--device", "cpu"])
    return app, pipeline


## Application


def test_a_run_needs_something_to_generate_from() -> None:
    app = SelfForcingT2VApplication(pipeline_config=_FakePipelineConfig(_FakePipeline()))
    with pytest.raises(ValueError, match="--prompt is required"):
        app.init([])
    with pytest.raises(ValueError, match="--prompt is required"):
        app.init(["--prompt", "   "])


def test_a_session_cannot_be_created_before_the_application_is_told_what_to_do() -> None:
    app = SelfForcingT2VApplication(pipeline_config=_FakePipelineConfig(_FakePipeline()))
    with pytest.raises(RuntimeError, match="init.. must run before create_session"):
        app.create_session(default_session_desc())


def test_the_model_loads_once_and_every_session_shares_it() -> None:
    app, pipeline = _application()
    config = app.pipeline_config
    assert isinstance(config, _FakePipelineConfig)

    app.create_session(_session_desc())
    app.create_session(_session_desc())

    assert config.setup_count == 1
    assert pipeline.device == "cpu"
    assert pipeline.eval_count == 1


def test_the_model_is_not_loaded_until_a_session_wants_it() -> None:
    app, _ = _application()
    config = app.pipeline_config
    assert isinstance(config, _FakePipelineConfig)
    assert config.setup_count == 0


def test_closing_the_application_releases_the_model() -> None:
    app, pipeline = _application()
    app.create_session(_session_desc())

    app.close()

    assert pipeline.closed


def test_compilation_can_be_turned_off_for_a_run() -> None:
    """The real config compiles by default, which costs minutes on first use."""
    app = SelfForcingT2VApplication()

    app.init(["--prompt", _PROMPT, "--no-compile"])

    transformer = app.pipeline_config.diffusion_model.transformer
    assert transformer.compile_network is False


## Session


def test_a_session_rejects_a_layout_the_model_does_not_emit() -> None:
    app, _ = _application()
    with pytest.raises(ValueError, match="only produces tchw output"):
        app.create_session(_session_desc(VideoTensorLayout.bcthw))


@pytest.mark.parametrize("width,height", [(130, 64), (128, 60)])
def test_a_session_rejects_frames_that_are_not_a_whole_number_of_latents(
    width: int, height: int
) -> None:
    app, _ = _application()
    with pytest.raises(ValueError, match=f"multiples of {_COMPRESSION_RATIO}"):
        app.create_session(_session_desc(width=width, height=height))


def test_a_rollout_starts_from_the_prompt_at_the_requested_size() -> None:
    app, pipeline = _application()
    session = app.create_session(_session_desc())

    session.init()

    assert pipeline.caches == [
        {
            "text": [_PROMPT],
            "image": None,
            "height": _HEIGHT // _COMPRESSION_RATIO,
            "width": _WIDTH // _COMPRESSION_RATIO,
        }
    ]


def test_a_step_before_the_rollout_starts_is_refused() -> None:
    app, _ = _application()
    session = app.create_session(_session_desc())
    with pytest.raises(RuntimeError, match="init.. must run before step"):
        session.step(0, _NO_EVENTS)


def test_a_step_generates_a_block_and_advances_the_rollout() -> None:
    app, pipeline = _application()
    session = app.create_session(_session_desc())
    session.init()

    first = session.step(0, _NO_EVENTS)
    second = session.step(1, _NO_EVENTS)

    # Generating a block and advancing past it are separate calls, and a step
    # is only over once both have run.
    assert pipeline.generated == [0, 1]
    assert pipeline.finalized == [0, 1]
    assert (first.step_index, second.step_index) == (0, 1)
    assert first.output_layout is VideoTensorLayout.tchw
    assert first.metrics == {"total_ms": 1.5}


def test_a_result_reports_the_frames_it_carries() -> None:
    """The first block of a causal decode is shorter than the rest."""
    app, _ = _application()
    session = app.create_session(_session_desc())
    session.init()

    first = session.step(0, _NO_EVENTS)
    second = session.step(1, _NO_EVENTS)

    assert first.frame_count == _FIRST_BLOCK_FRAMES
    assert tuple(first.output.shape) == (_FIRST_BLOCK_FRAMES, 3, _HEIGHT, _WIDTH)
    assert second.frame_count == _BLOCK_FRAMES


def test_resetting_starts_the_rollout_again_from_the_same_prompt() -> None:
    app, pipeline = _application()
    session = app.create_session(_session_desc())
    session.init()
    session.step(0, _NO_EVENTS)

    session.reset()

    assert len(pipeline.caches) == 2
    assert pipeline.caches[0] == pipeline.caches[1]


def test_closing_a_session_leaves_the_model_for_the_next_one() -> None:
    app, pipeline = _application()
    session = app.create_session(_session_desc())
    session.init()

    session.close()

    assert not pipeline.closed
    with pytest.raises(RuntimeError, match="init.. must run before step"):
        session.step(0, _NO_EVENTS)


## The shared check, over the whole batch path


@needs_ffmpeg
def test_a_run_writes_every_generated_frame_to_an_mp4(tmp_path: Path) -> None:
    steps = 3
    path = tmp_path / "clip.mp4"

    result = check_t2v_model_impl(
        SelfForcingT2VApplication(pipeline_config=_FakePipelineConfig(_FakePipeline())),
        _session_desc(),
        steps=steps,
        commandline_args=["--prompt", _PROMPT, "--device", "cpu"],
        expected=ExpectedFrameStats(
            frame_count=_FIRST_BLOCK_FRAMES + (steps - 1) * _BLOCK_FRAMES,
            mean_luminance=(64.0, 192.0),
            min_frame_difference=1.0,
        ),
        mp4_path=path,
    )

    assert result.passed, result.failures
    assert result.frames_per_step == (_FIRST_BLOCK_FRAMES, _BLOCK_FRAMES, _BLOCK_FRAMES)
    assert result.metrics == ({"total_ms": 1.5},) * steps
    assert path.stat().st_size > 0


def test_the_check_reports_what_a_run_failed_to_generate() -> None:
    result = check_t2v_model_impl(
        SelfForcingT2VApplication(pipeline_config=_FakePipelineConfig(_FakePipeline())),
        _session_desc(),
        steps=1,
        commandline_args=["--prompt", _PROMPT, "--device", "cpu"],
        expected=ExpectedFrameStats(
            frame_count=_FIRST_BLOCK_FRAMES + 1,
            mean_luminance=(250.0, 255.0),
            min_frame_difference=1000.0,
        ),
    )

    assert not result.passed
    assert len(result.failures) == 3
    assert "generated 9" in result.failures[0]
