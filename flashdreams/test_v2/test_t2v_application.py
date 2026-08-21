# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for the text-to-video application every t2v integration configures.

The part that is the same for every model: the flags, what they resolve to,
what an application refuses to generate, and when the model is loaded.
"""

import argparse
from types import SimpleNamespace
from typing import Any

import pytest

from flashdreams.runtime_v2.session_desc import SessionDesc
from flashdreams.runtime_v2.video_tensor import VideoTensorLayout
from flashdreams.t2v_v2.application import T2VApplication
from flashdreams.t2v_v2.defaults import T2VApplicationDefaults
from flashdreams.t2v_v2.session import T2VSession

pytestmark = pytest.mark.ci_cpu

_COMPRESSION_RATIO = 8
"""Pixels per latent in each direction, as a real video decoder has."""

_WIDTH = 128
"""Default frame width under test. Not square, so a transposed description
cannot pass unnoticed, and a whole number of latents across."""

_HEIGHT = 64
"""Default frame height under test."""

_FPS = 16
"""Default rate under test."""

_TOTAL_BLOCKS = 7
"""Default rollout length under test."""

_PROMPT = "A cat surfing"
"""Prompt the tests generate from."""


## Stand-in model


class FakeDecoder:
    spatial_compression_ratio = _COMPRESSION_RATIO


class FakePipeline:
    """A loaded model, as far as the application can tell."""

    def __init__(self) -> None:
        self.decoder = FakeDecoder()
        self.device: str | None = None
        self.eval_count = 0
        self.closed = False

    def to(self, device: str) -> "FakePipeline":
        self.device = device
        return self

    def eval(self) -> "FakePipeline":
        self.eval_count += 1
        return self

    def close(self) -> None:
        self.closed = True


class FakePipelineConfig:
    """Record how often the model was loaded, which is the expensive part."""

    def __init__(self) -> None:
        self.pipeline = FakePipeline()
        self.setup_count = 0

    def setup(self) -> FakePipeline:
        self.setup_count += 1
        return self.pipeline


class RecordingRollout(T2VSession):
    """Session remembering how long a rollout it was given."""

    def __init__(
        self, pipeline: Any, prompt: str, session_desc: SessionDesc, total_blocks: int
    ) -> None:
        super().__init__(pipeline, prompt, session_desc, total_blocks)
        self.blocks_to_generate = total_blocks


class ApplicationUnderTest(T2VApplication):
    """The shared application, with a session that says what it was told."""

    session_type = RecordingRollout


## Helpers


def _defaults(**overrides: Any) -> T2VApplicationDefaults:
    settings: dict[str, Any] = {
        "pipeline_config": FakePipelineConfig(),
        "total_blocks": _TOTAL_BLOCKS,
        "pixel_width": _WIDTH,
        "pixel_height": _HEIGHT,
        "device": "cpu",
        "fps": _FPS,
        "output_layout": VideoTensorLayout.tchw,
    }
    return T2VApplicationDefaults(**(settings | overrides))


def _application(
    defaults: T2VApplicationDefaults | None = None,
    *,
    args: list[str] | None = None,
) -> T2VApplication:
    """Return an application told to generate ``_PROMPT``."""
    app = ApplicationUnderTest(defaults=defaults or _defaults())
    app.init(["--prompt", _PROMPT, *(args or [])])
    return app


def _pipeline_config(app: T2VApplication) -> FakePipelineConfig:
    config = app.pipeline_config
    assert isinstance(config, FakePipelineConfig)
    return config


def _rollout_length(app: T2VApplication) -> int:
    """Return the rollout length the sessions this creates were given.

    An application answers no such question: the length reaches a run through
    the session, which reports itself finished at the end of it.
    """
    session = app.create_session(app.session_desc())
    assert isinstance(session, RecordingRollout)
    return session.blocks_to_generate


def _session_desc(
    layout: VideoTensorLayout = VideoTensorLayout.tchw,
    *,
    width: int = _WIDTH,
    height: int = _HEIGHT,
) -> SessionDesc:
    return SessionDesc(
        output_layout=layout,
        frames_per_second_for_ui=60,
        frames_per_second_for_step=_FPS,
        video_width=width,
        video_height=height,
    )


def _runner_config(**overrides: Any) -> SimpleNamespace:
    """Return a stand-in for the v1 runner config an integration ships."""
    settings: dict[str, Any] = {
        "pipeline": FakePipelineConfig(),
        "total_blocks": 60,
        "pixel_width": 832,
        "pixel_height": 480,
        "fps": 16,
        "postprocess_output_layout": "tchw",
    }
    return SimpleNamespace(**(settings | overrides))


## What an integration contributes


def test_the_defaults_a_runner_config_states_are_not_written_down_again() -> None:
    runner_config = _runner_config()

    defaults = T2VApplicationDefaults.from_runner_config(runner_config)

    assert defaults.pipeline_config is runner_config.pipeline
    assert (defaults.pixel_width, defaults.pixel_height) == (832, 480)
    assert (defaults.fps, defaults.total_blocks) == (16, 60)
    assert defaults.output_layout is VideoTensorLayout.tchw


def test_a_runner_config_that_cannot_drive_this_says_what_it_is_missing() -> None:
    """It is duck-typed, so this is the only place a mismatch can be caught."""
    with pytest.raises(TypeError, match="missing.*pixel_width"):
        T2VApplicationDefaults.from_runner_config(
            SimpleNamespace(pipeline=FakePipelineConfig(), total_blocks=1)
        )


def test_a_rollout_length_can_be_given_instead_of_the_runners() -> None:
    # And is then the one field a runner config need not carry at all.
    runner_config = _runner_config()
    del runner_config.total_blocks

    defaults = T2VApplicationDefaults.from_runner_config(runner_config, total_blocks=4)

    assert defaults.total_blocks == 4


@pytest.mark.parametrize(
    "declared,expected",
    [
        ("bcthw", VideoTensorLayout.bcthw),
        (VideoTensorLayout.btchw, VideoTensorLayout.btchw),
        (None, VideoTensorLayout.tchw),
    ],
)
def test_a_v1_layout_becomes_a_v2_layout(
    declared: Any, expected: VideoTensorLayout
) -> None:
    """A v1 runner config spells its layout as a string; this API uses an enum."""
    defaults = T2VApplicationDefaults.from_runner_config(
        _runner_config(postprocess_output_layout=declared)
    )

    assert defaults.output_layout is expected


## The command line


def test_a_model_describes_the_clip_it_was_trained_for_without_loading() -> None:
    """Asked before the application is told anything, since a caller has to
    describe a session before it can ask for one."""
    app = T2VApplication(defaults=_defaults())

    desc = app.session_desc()

    assert desc == _session_desc()
    assert _pipeline_config(app).setup_count == 0


def test_the_rollout_length_can_be_overridden() -> None:
    app = _application(args=["--total-blocks", "3"])

    assert _rollout_length(app) == 3


def test_a_run_needs_something_to_generate_from() -> None:
    app = T2VApplication(defaults=_defaults())

    with pytest.raises(ValueError, match="--prompt is required"):
        app.init([])
    with pytest.raises(ValueError, match="--prompt is required"):
        app.init(["--prompt", "   "])


def test_a_run_that_would_generate_no_video_is_refused() -> None:
    with pytest.raises(ValueError, match="--total-blocks must be"):
        _application(args=["--total-blocks", "0"])


def test_no_session_is_created_before_the_application_is_told_what_to_do() -> None:
    app = T2VApplication(defaults=_defaults())

    with pytest.raises(RuntimeError, match="init.. must run before create_session"):
        app.create_session(_session_desc())


## Loading the model


def test_the_model_loads_once_and_every_session_shares_it() -> None:
    app = _application()
    config = _pipeline_config(app)

    first = app.create_session(_session_desc())
    second = app.create_session(_session_desc())

    assert config.setup_count == 1
    assert config.pipeline.device == "cpu"
    assert config.pipeline.eval_count == 1
    assert first is not second


def test_closing_the_application_releases_the_model() -> None:
    app = _application()
    config = _pipeline_config(app)
    app.create_session(_session_desc())

    app.close()

    assert config.pipeline.closed


## What a model will not generate


def test_a_layout_the_model_does_not_emit_is_refused_before_it_loads() -> None:
    """A checkpoint of several gigabytes is a long wait for a certain refusal."""
    app = _application()
    config = _pipeline_config(app)

    with pytest.raises(ValueError, match="only produces tchw output"):
        app.create_session(_session_desc(VideoTensorLayout.bcthw))

    assert config.setup_count == 0


@pytest.mark.parametrize("width,height", [(130, 64), (128, 60)])
def test_frames_that_are_not_a_whole_number_of_latents_are_refused(
    width: int, height: int
) -> None:
    app = _application()

    with pytest.raises(ValueError, match=f"multiples of {_COMPRESSION_RATIO}"):
        app.create_session(_session_desc(width=width, height=height))


## What an integration can change


def test_an_integration_can_add_a_flag_of_its_own() -> None:
    class WithGuidance(T2VApplication):
        def __init__(self) -> None:
            super().__init__(defaults=_defaults())
            self.guidance_scale: float | None = None

        def _configure_argument_parser(self, parser: argparse.ArgumentParser) -> None:
            parser.add_argument("--guidance-scale", type=float, default=1.0)

        def _apply_parsed_arguments(self, args: argparse.Namespace) -> None:
            self.guidance_scale = args.guidance_scale

    app = WithGuidance()
    app.init(["--prompt", _PROMPT, "--guidance-scale", "7.5"])

    assert app.guidance_scale == 7.5


def test_an_integration_can_refuse_a_rollout_length_its_model_cannot_generate() -> None:
    class SingleShot(T2VApplication):
        def _validate_total_blocks(self, total_blocks: int) -> None:
            if total_blocks != 1:
                raise ValueError("this model generates its clip in one block")

    app = SingleShot(defaults=_defaults())

    with pytest.raises(ValueError, match="one block"):
        app.init(["--prompt", _PROMPT, "--total-blocks", "2"])


def test_an_integration_can_replace_the_session() -> None:
    class Recording(T2VSession):
        pass

    class WithOwnSession(T2VApplication):
        session_type = Recording

    app = WithOwnSession(defaults=_defaults())
    app.init(["--prompt", _PROMPT])

    assert isinstance(app.create_session(_session_desc()), Recording)


@pytest.mark.parametrize(
    "args,expected",
    [([], None), (["--compile"], True), (["--no-compile"], False)],
)
def test_compilation_is_only_reconfigured_when_a_run_asks(
    args: list[str], expected: bool | None
) -> None:
    """Compiling costs minutes and buys back milliseconds a step, so which a run
    wants depends on how long it is. Unasked, the model's own config decides."""

    class RecordingCompile(T2VApplication):
        def __init__(self) -> None:
            super().__init__(defaults=_defaults())
            self.asked_for: bool | None = None

        def _apply_compile_override(self, pipeline_config: Any, enabled: bool) -> Any:
            self.asked_for = enabled
            return pipeline_config

    app = RecordingCompile()
    app.init(["--prompt", _PROMPT, *args])

    assert app.asked_for is expected


@pytest.mark.parametrize("args,expected", [([], None), (["--seed", "11"], 11)])
def test_the_noise_is_only_seeded_when_a_run_asks(
    args: list[str], expected: int | None
) -> None:
    """A benchmark comparing two clips seeds them; a model asked for something
    to watch keeps whatever seed its own config chose."""

    class RecordingSeed(T2VApplication):
        def __init__(self) -> None:
            super().__init__(defaults=_defaults())
            self.asked_for: int | None = None

        def _apply_seed_override(self, pipeline_config: Any, seed: int) -> Any:
            self.asked_for = seed
            return pipeline_config

    app = RecordingSeed()
    app.init(["--prompt", _PROMPT, *args])

    assert app.asked_for == expected


def test_a_seed_reaches_the_model_where_a_model_keeps_one() -> None:
    """Straight onto the config the pipeline is built from, so a model that
    draws its own noise draws the same noise twice."""
    diffusion_model = SimpleNamespace(seed=42)
    defaults = _defaults(
        pipeline_config=SimpleNamespace(diffusion_model=diffusion_model)
    )

    app = T2VApplication(defaults=defaults)
    app.init(["--prompt", _PROMPT, "--seed", "7"])

    assert app.pipeline_config.diffusion_model.seed == 7
    # Derived rather than edited: an integration's config is shared.
    assert diffusion_model.seed == 42
