# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for the rollout every text-to-video model generates through.

``T2VSession`` encodes the prompt into a cache once, then generates one
autoregressive block per step. Driving it in order is the same job for every
model, so it is covered here once against the shared stand-in pipeline.
"""

import pytest

from flashdreams.runtime_v2.session_desc import SessionDesc
from flashdreams.runtime_v2.user_input_events import UserInputEvents
from flashdreams.runtime_v2.video_tensor import VideoTensorLayout
from flashdreams.t2v_v2.application import T2VApplication
from flashdreams.t2v_v2.defaults import T2VApplicationDefaults
from flashdreams.t2v_v2.testing import FakeT2VPipeline, FakeT2VPipelineConfig

pytestmark = pytest.mark.ci_cpu

_WIDTH = 128
"""Frame width the stand-in generates. Not square, so a transposed frame cannot
pass unnoticed, and a whole number of latents across."""

_HEIGHT = 64
"""Frame height it generates."""

_COMPRESSION_RATIO = 8
"""Pixels per latent in each direction, as a real video decoder reports."""

_FIRST_BLOCK_FRAMES = 9
"""Frames the first block decodes, as a real causal decoder emits."""

_BLOCK_FRAMES = 12
"""Frames every block after it decodes."""

_FPS = 16
"""Rate the generated frames are meant to play at."""

_TOTAL_BLOCKS = 7
"""Rollout length under test."""

_PROMPT = "A cat surfing"
"""Prompt the tests generate from."""

_NO_EVENTS = UserInputEvents([])
"""What a text-to-video session is given every step, reading no input."""


## Helpers


def _stand_in() -> FakeT2VPipeline:
    """Return a stand-in generating what the constants above describe.

    Said rather than defaulted, since these tests assert the exact shapes.
    """
    return FakeT2VPipeline(
        width=_WIDTH,
        height=_HEIGHT,
        compression_ratio=_COMPRESSION_RATIO,
        first_block_frames=_FIRST_BLOCK_FRAMES,
        block_frames=_BLOCK_FRAMES,
    )


def _session_desc() -> SessionDesc:
    return SessionDesc(
        output_layout=VideoTensorLayout.tchw,
        frames_per_second_for_ui=60,
        frames_per_second_for_step=_FPS,
        video_width=_WIDTH,
        video_height=_HEIGHT,
    )


def _application() -> tuple[T2VApplication, FakeT2VPipeline]:
    """Return an initialized application and the model it will load."""
    pipeline = _stand_in()
    app = T2VApplication(
        defaults=T2VApplicationDefaults(
            pipeline_config=FakeT2VPipelineConfig(pipeline),
            total_blocks=_TOTAL_BLOCKS,
            pixel_width=_WIDTH,
            pixel_height=_HEIGHT,
            device="cpu",
            fps=_FPS,
            output_layout=VideoTensorLayout.tchw,
        )
    )
    app.init(["--prompt", _PROMPT])
    return app, pipeline


## One rollout


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

    # Generating a block and advancing past it are separate calls.
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


def test_a_rollout_finishes_once_it_has_generated_its_blocks() -> None:
    """How long a run lasts comes from here, so nobody counts steps for it."""
    app, _ = _application()
    session = app.create_session(_session_desc())
    session.init()

    for step_index in range(_TOTAL_BLOCKS):
        assert not session.is_finished()
        session.step(step_index, _NO_EVENTS)

    assert session.is_finished()


def test_a_reset_rollout_has_its_whole_length_again() -> None:
    app, _ = _application()
    session = app.create_session(_session_desc())
    session.init()
    for step_index in range(_TOTAL_BLOCKS):
        session.step(step_index, _NO_EVENTS)

    session.reset()

    assert not session.is_finished()


def test_closing_a_session_leaves_the_model_for_the_next_one() -> None:
    app, pipeline = _application()
    session = app.create_session(_session_desc())
    session.init()

    session.close()

    assert not pipeline.closed
    with pytest.raises(RuntimeError, match="init.. must run before step"):
        session.step(0, _NO_EVENTS)
