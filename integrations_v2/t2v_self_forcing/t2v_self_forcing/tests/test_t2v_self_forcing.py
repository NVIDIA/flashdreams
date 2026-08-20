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

import pytest
from self_forcing.config import RUNNER_WAN21_T2V_1PT3B
from t2v_self_forcing import SelfForcingT2VApplication

from flashdreams.runtime_v2.session_desc import SessionDesc
from flashdreams.runtime_v2.user_input_events import UserInputEvents
from flashdreams.runtime_v2.video_tensor import VideoTensorLayout
from flashdreams.t2v_v2.testing import (
    ExpectedFrameStats,
    FakeT2VPipeline,
    FakeT2VPipelineConfig,
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


## Helpers


def _stand_in() -> FakeT2VPipeline:
    """Return a stand-in generating what the constants above describe.

    Passed rather than left to the stand-in's own defaults, because these tests
    assert the exact shapes it emits.
    """
    return FakeT2VPipeline(
        width=_WIDTH,
        height=_HEIGHT,
        compression_ratio=_COMPRESSION_RATIO,
        first_block_frames=_FIRST_BLOCK_FRAMES,
        block_frames=_BLOCK_FRAMES,
    )


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
    pipeline: FakeT2VPipeline | None = None,
) -> tuple[SelfForcingT2VApplication, FakeT2VPipeline]:
    """Return an initialized application and the model it will load."""
    pipeline = pipeline or _stand_in()
    app = SelfForcingT2VApplication(pipeline_config=FakeT2VPipelineConfig(pipeline))
    app.init(["--prompt", _PROMPT, "--device", "cpu"])
    return app, pipeline


## Application


def test_a_run_needs_something_to_generate_from() -> None:
    app = SelfForcingT2VApplication(pipeline_config=FakeT2VPipelineConfig(_stand_in()))
    with pytest.raises(ValueError, match="--prompt is required"):
        app.init([])
    with pytest.raises(ValueError, match="--prompt is required"):
        app.init(["--prompt", "   "])


def test_a_session_cannot_be_created_before_the_application_is_told_what_to_do() -> (
    None
):
    app = SelfForcingT2VApplication(pipeline_config=FakeT2VPipelineConfig(_stand_in()))
    with pytest.raises(RuntimeError, match="init.. must run before create_session"):
        app.create_session(_session_desc())


def test_the_model_loads_once_and_every_session_shares_it() -> None:
    app, pipeline = _application()
    config = app.pipeline_config
    assert isinstance(config, FakeT2VPipelineConfig)

    app.create_session(_session_desc())
    app.create_session(_session_desc())

    assert config.setup_count == 1
    assert pipeline.device == "cpu"
    assert pipeline.eval_count == 1


def test_the_model_is_not_loaded_until_a_session_wants_it() -> None:
    app, _ = _application()
    config = app.pipeline_config
    assert isinstance(config, FakeT2VPipelineConfig)
    assert config.setup_count == 0


def test_the_model_says_what_it_generates_without_being_told() -> None:
    """These numbers are the checkpoint's, and are only written down once.

    They come from the runner config this integration ships, which is the point
    of deriving the defaults from it: a caller wanting the clip the model was
    trained to generate passes no size flags at all.
    """
    app, _ = _application()

    desc = app.session_desc()

    assert (desc.video_width, desc.video_height) == (
        RUNNER_WAN21_T2V_1PT3B.pixel_width,
        RUNNER_WAN21_T2V_1PT3B.pixel_height,
    )
    assert desc.frames_per_second_for_step == RUNNER_WAN21_T2V_1PT3B.fps
    assert desc.output_layout is VideoTensorLayout.tchw
    assert app.total_blocks == RUNNER_WAN21_T2V_1PT3B.total_blocks


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
        SelfForcingT2VApplication(pipeline_config=FakeT2VPipelineConfig(_stand_in())),
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
        SelfForcingT2VApplication(pipeline_config=FakeT2VPipelineConfig(_stand_in())),
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
