# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for the command line that runs a text-to-video application.

The end-to-end runs here use a stand-in for a model, so what they cover is the
wiring: that the application is found, given its own arguments, asked what it
would generate, run against the window the arguments chose, and closed. Whether
a real checkpoint generates anything worth watching is a GPU question, asked in
the integration that owns the checkpoint.
"""

import json
import shutil
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
import torch

from flashdreams.api_v2.application import IApplication
from flashdreams.api_v2.client_window import IClientWindow
from flashdreams.runtime_v2.applications import (
    APPLICATION_ENTRY_POINT_GROUP,
    create_application,
    registered_application_slugs,
)
from flashdreams.runtime_v2.mp4_client_window import Mp4ClientWindow
from flashdreams.runtime_v2.session_desc import SessionDesc
from flashdreams.runtime_v2.step_result import StepResult
from flashdreams.runtime_v2.user_input_events import UserInputEvents
from flashdreams.runtime_v2.video_tensor import VideoTensorLayout
from flashdreams.t2v_v2 import cli
from flashdreams.t2v_v2.application import T2VApplication
from flashdreams.t2v_v2.defaults import T2VApplicationDefaults

pytestmark = pytest.mark.ci_cpu

needs_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None,
    reason="writing an MP4 needs ffmpeg on PATH",
)

_WIDTH = 128
"""Frame width the stand-in generates. A whole number of latents across, and not
square, so a transposed frame cannot pass unnoticed."""

_HEIGHT = 64
"""Frame height it generates."""

_COMPRESSION_RATIO = 8
"""Pixels per latent in each direction."""

_BLOCK_FRAMES = 4
"""Frames a step generates. Small, because these tests encode real files."""

_TOTAL_BLOCKS = 3
"""Rollout length the stand-in application says it normally generates."""

_PROMPT = "A cat surfing"
"""Prompt the tests generate from."""


## Stand-in model


class FakeDecoder:
    spatial_compression_ratio = _COMPRESSION_RATIO


class FakePipeline:
    """Generate frames of the shape and range a real pipeline generates."""

    def __init__(self, *, fail_at: int | None = None) -> None:
        """
        Args:
            fail_at: Step to fail at, for exercising what a failed run releases.
        """
        self.decoder = FakeDecoder()
        self.closed = False
        self.prompts: list[str] = []
        self.generated: list[int] = []
        self._fail_at = fail_at

    def to(self, device: str) -> "FakePipeline":
        del device
        return self

    def eval(self) -> "FakePipeline":
        return self

    def initialize_cache(self, **kwargs: Any) -> object:
        self.prompts.extend(kwargs["text"])
        return object()

    def generate(self, *, autoregressive_index: int, cache: object) -> torch.Tensor:
        del cache
        self.generated.append(autoregressive_index)
        if autoregressive_index == self._fail_at:
            raise RuntimeError("generate failed")
        # A different shade per step, so a file that dropped or repeated one
        # does not read back as a correct video.
        shade = -0.5 + (autoregressive_index % 4) / 4.0
        return torch.full(
            (_BLOCK_FRAMES, 3, _HEIGHT, _WIDTH), shade, dtype=torch.float32
        )

    def finalize(self, *, autoregressive_index: int, cache: object) -> dict[str, float]:
        del cache
        # A measurement, as a real pipeline reports one, so a run asked what it
        # cost has something to answer with.
        return {"total_ms": 20.0 + autoregressive_index}

    def close(self) -> None:
        self.closed = True


class FakePipelineConfig:
    def __init__(self, pipeline: FakePipeline) -> None:
        self.pipeline = pipeline

    def setup(self) -> FakePipeline:
        return self.pipeline


class StubT2VApplication(T2VApplication):
    """A text-to-video application whose model costs nothing to load."""

    def __init__(self, pipeline: FakePipeline) -> None:
        super().__init__(
            defaults=T2VApplicationDefaults(
                pipeline_config=FakePipelineConfig(pipeline),
                total_blocks=_TOTAL_BLOCKS,
                pixel_width=_WIDTH,
                pixel_height=_HEIGHT,
                device="cpu",
                fps=8,
                output_layout=VideoTensorLayout.tchw,
            )
        )


class NotATextToVideoApplication(IApplication):
    """An application with no way to say what session it would generate."""

    def init(self, commandline_args: Sequence[str]) -> None:
        del commandline_args

    def create_session(self, session_desc: SessionDesc) -> Any:
        raise AssertionError("should never be reached")


class RecordingWindow(IClientWindow):
    """Stand in for a window with a client, recording what it was given."""

    def __init__(self) -> None:
        self.results: list[StepResult] = []

    def get_user_input_events(self) -> UserInputEvents:
        return UserInputEvents([])

    def open(self, session_desc: SessionDesc) -> None:
        self.session_desc = session_desc

    def write(self, result: StepResult) -> None:
        self.results.append(result)

    def close(self) -> None:
        return


## Splitting the command line


def test_arguments_after_the_separator_belong_to_the_application() -> None:
    own, application = cli.split_arguments(
        ["slug", "--mode", "mp4", "--", "--prompt", "a cat", "--mode", "fancy"]
    )

    # --mode appears on both sides, which is the point: an application is free
    # to declare a flag this command also has.
    assert own == ["slug", "--mode", "mp4"]
    assert application == ["--prompt", "a cat", "--mode", "fancy"]


def test_an_application_taking_no_arguments_needs_no_separator() -> None:
    assert cli.split_arguments(["slug", "--mode", "mp4"]) == (
        ["slug", "--mode", "mp4"],
        [],
    )


def test_a_separator_with_nothing_after_it_is_an_empty_application_line() -> None:
    assert cli.split_arguments(["slug", "--"]) == (["slug"], [])


## Finding the application


def test_an_unknown_slug_says_what_is_installed() -> None:
    with pytest.raises(LookupError, match="No FlashDreams v2 application matches"):
        create_application("no-such-application")


def test_a_slug_with_no_entry_point_is_read_as_a_module(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An integration is reachable by the name of the package it ships, so it
    runs straight from a checkout that has registered nothing."""
    _write_application_module(
        tmp_path,
        "stub_integration",
        "class Stub(IApplication):\n"
        "    def init(self, commandline_args): pass\n"
        "    def create_session(self, session_desc): raise NotImplementedError\n"
        "def create_app():\n"
        "    return Stub()\n",
        monkeypatch,
    )

    application = create_application("stub-integration")

    assert type(application).__name__ == "Stub"


def test_a_module_without_a_factory_says_so(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_application_module(tmp_path, "no_factory", "", monkeypatch)

    with pytest.raises(TypeError, match="does not expose create_app"):
        create_application("no-factory")


def test_an_application_on_the_older_contract_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A v1 application is not a v2 one, and saying what arrived is the only
    way a reader can tell which of the two they installed."""
    _write_application_module(
        tmp_path,
        "wrong_contract",
        "def create_app():\n    return object()\n",
        monkeypatch,
    )

    with pytest.raises(TypeError, match="returned object; expected an IApplication"):
        create_application("wrong-contract")


def test_an_empty_slug_is_refused() -> None:
    with pytest.raises(ValueError, match="slug is required"):
        create_application("  ")


def test_what_is_installed_is_reported_in_a_stable_order() -> None:
    slugs = registered_application_slugs()

    assert slugs == tuple(sorted(set(slugs)))
    assert APPLICATION_ENTRY_POINT_GROUP == "flashdreams.applications_v2"


def _write_application_module(
    root: Path, name: str, body: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Put an importable application package on the path, as an install would."""
    package = root / name
    package.mkdir()
    (package / "__init__.py").write_text(
        "from flashdreams.api_v2.application import IApplication\n" + body
    )
    monkeypatch.syspath_prepend(root)


## Running


@needs_ffmpeg
def test_a_run_writes_what_the_application_generated(tmp_path: Path) -> None:
    pipeline = FakePipeline()
    path = tmp_path / "clip.mp4"

    cli.run_application(
        StubT2VApplication(pipeline),
        Mp4ClientWindow(path),
        application_args=["--prompt", _PROMPT, "--total-blocks", "2"],
    )

    assert path.stat().st_size > 0
    assert pipeline.prompts == [_PROMPT]
    assert pipeline.generated == [0, 1]


def test_a_run_lasts_as_long_as_the_rollout_the_application_asked_for() -> None:
    """Nothing here counts steps: the session reports itself finished."""
    pipeline = FakePipeline()

    cli.run_application(
        StubT2VApplication(pipeline),
        RecordingWindow(),
        application_args=["--prompt", _PROMPT],
    )

    assert pipeline.generated == list(range(_TOTAL_BLOCKS))


def test_the_model_is_released_when_a_run_fails() -> None:
    # A model holds most of a GPU, so a run that gave up part way through has
    # to put it back whether or not it produced anything.
    pipeline = FakePipeline(fail_at=1)

    with pytest.raises(RuntimeError, match="generate failed"):
        cli.run_application(
            StubT2VApplication(pipeline),
            RecordingWindow(),
            application_args=["--prompt", _PROMPT],
        )

    assert pipeline.closed


@needs_ffmpeg
def test_a_run_can_record_what_generating_the_clip_cost(tmp_path: Path) -> None:
    """A clip says nothing about what it took to generate, so a benchmark run
    asks for both and the file it writes is the one the harness reads."""
    stats_path = tmp_path / "stats_run.json"
    clip_path = tmp_path / "clip.mp4"

    cli.run_application(
        StubT2VApplication(FakePipeline()),
        Mp4ClientWindow(clip_path, stats_path=stats_path),
        application_args=["--prompt", _PROMPT, "--total-blocks", "2"],
    )

    # Both files, and every step in the one recording them. What a stats file
    # says is the sink's own business, covered in test_metrics_output_sink.py.
    assert clip_path.stat().st_size > 0
    payload = json.loads(stats_path.read_text(encoding="utf-8"))
    assert [step["step_index"] for step in payload["steps"]] == [0, 1]
    assert [step["frame_count"] for step in payload["steps"]] == [
        _BLOCK_FRAMES,
        _BLOCK_FRAMES,
    ]


@needs_ffmpeg
def test_nothing_is_measured_unless_a_run_asks(tmp_path: Path) -> None:
    cli.run_application(
        StubT2VApplication(FakePipeline()),
        Mp4ClientWindow(tmp_path / "clip.mp4"),
        application_args=["--prompt", _PROMPT, "--total-blocks", "1"],
    )

    assert list(tmp_path.glob("*.json")) == []


def test_an_application_that_will_not_start_reports_why() -> None:
    with pytest.raises(ValueError, match="--prompt is required"):
        cli.run_application(StubT2VApplication(FakePipeline()), RecordingWindow())


## The command itself


@needs_ffmpeg
def test_the_command_reports_the_file_it_wrote(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    pipeline = FakePipeline()
    monkeypatch.setattr(
        cli, "create_application", lambda slug: StubT2VApplication(pipeline)
    )
    path = tmp_path / "clip.mp4"

    cli.entrypoint(
        [
            "stub",
            "--output-path",
            str(path),
            "--",
            "--prompt",
            _PROMPT,
            "--total-blocks",
            "1",
        ]
    )

    assert capsys.readouterr().out.strip() == str(path)
    assert path.stat().st_size > 0


@needs_ffmpeg
def test_the_command_can_be_asked_to_measure_the_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        cli, "create_application", lambda slug: StubT2VApplication(FakePipeline())
    )
    stats_path = tmp_path / "stats_run.json"

    cli.entrypoint(
        [
            "stub",
            "--output-path",
            str(tmp_path / "clip.mp4"),
            "--stats-path",
            str(stats_path),
            "--",
            "--prompt",
            _PROMPT,
            "--total-blocks",
            "1",
        ]
    )

    assert json.loads(stats_path.read_text(encoding="utf-8"))["steps"] != []


def test_a_mode_other_than_a_file_takes_its_window_from_the_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The window comes from the arguments, and only the file one is built here."""
    window = RecordingWindow()
    asked_for: list[str] = []
    monkeypatch.setattr(
        cli, "create_application", lambda slug: StubT2VApplication(FakePipeline())
    )
    monkeypatch.setattr(
        cli,
        "create_client_window",
        lambda parsed: (asked_for.append(parsed.mode), window)[1],
    )

    cli.entrypoint(["stub", "--mode", "webrtc", "--", "--prompt", _PROMPT])

    assert asked_for == ["webrtc"]
    assert len(window.results) == _TOTAL_BLOCKS


def test_the_command_needs_somewhere_to_write() -> None:
    with pytest.raises(SystemExit):
        cli.entrypoint(["stub"])


def test_an_application_that_cannot_describe_a_session_cannot_be_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only text-to-video says what it would generate, so only it runs here."""
    monkeypatch.setattr(
        cli, "create_application", lambda slug: NotATextToVideoApplication()
    )

    with pytest.raises(TypeError, match="only runs text-to-video"):
        cli.entrypoint(["stub", "--output-path", str(tmp_path / "clip.mp4")])
