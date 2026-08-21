# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for the command line that runs a v2 application.

The runs here use a stand-in for a model, so what they cover is the wiring: the
application found, given its arguments, asked what session it would generate,
run against the window the arguments chose, and closed.
"""

import argparse
import json
import shutil
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
import torch

from flashdreams.api_v2.application import IApplication
from flashdreams.api_v2.client_window import IClientWindow
from flashdreams.api_v2.session import ISession
from flashdreams.runtime_v2 import cli
from flashdreams.runtime_v2.application_registry import (
    APPLICATION_ENTRY_POINT_GROUP,
    create_application,
    registered_application_slugs,
)
from flashdreams.runtime_v2.client_window_factory import ClientWindowMode
from flashdreams.runtime_v2.session_desc import SessionDesc
from flashdreams.runtime_v2.step_result import StepResult
from flashdreams.runtime_v2.user_input_events import UserInputEvents
from flashdreams.runtime_v2.video_tensor import VideoTensorLayout
from flashdreams.t2v_v2.application import T2VApplication
from flashdreams.t2v_v2.defaults import T2VApplicationDefaults
from flashdreams.t2v_v2.testing import FakeT2VPipeline, FakeT2VPipelineConfig

pytestmark = pytest.mark.ci_cpu

needs_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None,
    reason="writing an MP4 needs ffmpeg on PATH",
)

_BLOCK_FRAMES = 4
"""Frames a step generates. Small, because these tests encode real files."""

_TOTAL_BLOCKS = 3
"""Rollout length the stand-in application says it normally generates."""

_PROMPT = "A cat surfing"
"""Prompt the tests generate from."""


## Stand-in model


def _stand_in(*, fail_at: int | None = None) -> FakeT2VPipeline:
    """Return the shared stand-in, generating small blocks of one size."""
    return FakeT2VPipeline(
        first_block_frames=_BLOCK_FRAMES, block_frames=_BLOCK_FRAMES, fail_at=fail_at
    )


class StubT2VApplication(T2VApplication):
    """A text-to-video application whose model costs nothing to load."""

    def __init__(self, pipeline: FakeT2VPipeline) -> None:
        super().__init__(
            defaults=T2VApplicationDefaults(
                pipeline_config=FakeT2VPipelineConfig(pipeline),
                total_blocks=_TOTAL_BLOCKS,
                pixel_width=pipeline.width,
                pixel_height=pipeline.height,
                device="cpu",
                fps=8,
                output_layout=VideoTensorLayout.tchw,
            )
        )


class UndescribedApplication(IApplication):
    """An application that generates whatever it is asked for.

    Having no clip of its own in mind, the description it is handed is the one
    the command line built.
    """

    def __init__(self) -> None:
        self.asked_for: SessionDesc | None = None

    def init(self, commandline_args: Sequence[str]) -> None:
        del commandline_args

    def create_session(self, session_desc: SessionDesc) -> ISession:
        self.asked_for = session_desc
        return OneStepSession(session_desc)


class OneStepSession(ISession):
    """A session generating one frame and reporting itself finished."""

    def __init__(self, session_desc: SessionDesc) -> None:
        self._session_desc = session_desc
        self._generated = False

    def init(self) -> None:
        return

    @property
    def session_desc(self) -> SessionDesc:
        return self._session_desc

    def step(self, step_index: int, events: UserInputEvents) -> StepResult:
        del events
        self._generated = True
        return StepResult(
            step_index=step_index,
            output=torch.zeros(
                1, 3, self._session_desc.video_height, self._session_desc.video_width
            ),
            frame_count=1,
            output_layout=self._session_desc.output_layout,
        )

    def is_finished(self) -> bool:
        return self._generated


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

    # --mode on both sides is the point: an application may declare it too.
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


class StubMode(ClientWindowMode):
    """A mode handing the command a window the test can look inside."""

    def __init__(self, name: str, window: IClientWindow) -> None:
        self.name = name
        self._window = window

    def create(self, parsed_args: argparse.Namespace) -> IClientWindow:
        del parsed_args
        return self._window


def _install(
    monkeypatch: pytest.MonkeyPatch,
    application: IApplication,
    window: IClientWindow | None = None,
) -> None:
    """Point the command at this application, and at this window when given one.

    Only a run that is not writing a file needs one; a file run builds its
    window from the arguments like any other.
    """
    monkeypatch.setattr(cli, "create_application", lambda slug: application)
    if window is not None:
        monkeypatch.setattr(
            cli, "client_window_mode", lambda name: StubMode(name, window)
        )


@needs_ffmpeg
def test_a_run_writes_what_the_application_generated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    pipeline = _stand_in()
    _install(monkeypatch, StubT2VApplication(pipeline))
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
            "2",
        ]
    )

    assert capsys.readouterr().out.strip() == str(path)
    assert path.stat().st_size > 0
    assert pipeline.caches[0]["text"] == [_PROMPT]
    assert pipeline.generated == [0, 1]


def test_the_model_is_released_when_a_run_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A model holds most of a GPU, so a failed run still has to put it back.
    pipeline = _stand_in(fail_at=1)
    _install(monkeypatch, StubT2VApplication(pipeline), RecordingWindow())

    with pytest.raises(RuntimeError, match="generate failed"):
        cli.entrypoint(["stub", "--mode", "webrtc", "--", "--prompt", _PROMPT])

    assert pipeline.closed


@needs_ffmpeg
def test_a_run_can_record_what_generating_the_clip_cost(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A clip says nothing about what it took to generate, so a benchmark run
    asks for both and the file it writes is the one the harness reads."""
    _install(monkeypatch, StubT2VApplication(_stand_in()))
    stats_path = tmp_path / "stats_run.json"
    clip_path = tmp_path / "clip.mp4"

    cli.entrypoint(
        [
            "stub",
            "--output-path",
            str(clip_path),
            "--stats-path",
            str(stats_path),
            "--",
            "--prompt",
            _PROMPT,
            "--total-blocks",
            "2",
        ]
    )

    # What a stats file says is the sink's business, covered in its own tests.
    assert clip_path.stat().st_size > 0
    payload = json.loads(stats_path.read_text(encoding="utf-8"))
    assert [step["step_index"] for step in payload["steps"]] == [0, 1]
    assert [step["frame_count"] for step in payload["steps"]] == [
        _BLOCK_FRAMES,
        _BLOCK_FRAMES,
    ]


@needs_ffmpeg
def test_nothing_is_measured_unless_a_run_asks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install(monkeypatch, StubT2VApplication(_stand_in()))

    cli.entrypoint(
        [
            "stub",
            "--output-path",
            str(tmp_path / "clip.mp4"),
            "--",
            "--prompt",
            _PROMPT,
            "--total-blocks",
            "1",
        ]
    )

    assert list(tmp_path.glob("*.json")) == []


def test_an_application_that_will_not_start_reports_why(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install(monkeypatch, StubT2VApplication(_stand_in()), RecordingWindow())

    with pytest.raises(ValueError, match="--prompt is required"):
        cli.entrypoint(["stub", "--mode", "webrtc"])


## Describing the session to run


def test_an_application_with_no_session_of_its_own_is_described_by_the_arguments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Which is what lets this command run something that is not a model."""
    application = UndescribedApplication()
    _install(monkeypatch, application, RecordingWindow())

    cli.entrypoint(
        [
            "stub",
            "--mode",
            "webrtc",
            "--pixel-width",
            "64",
            "--pixel-height",
            "32",
            "--fps",
            "12",
            "--layout",
            "bcthw",
        ]
    )

    assert application.asked_for == SessionDesc(
        output_layout=VideoTensorLayout.bcthw,
        frames_per_second_for_step=12,
        video_width=64,
        video_height=32,
    )


def test_a_model_generates_what_it_was_trained_for_unless_asked_otherwise(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipeline = _stand_in()
    window = RecordingWindow()
    _install(monkeypatch, StubT2VApplication(pipeline), window)

    cli.entrypoint(
        ["stub", "--mode", "webrtc", "--pixel-width", "64", "--", "--prompt", _PROMPT]
    )

    assert window.session_desc.video_width == 64
    # What nobody asked about is still the model's own.
    assert window.session_desc.video_height == pipeline.height


## The command itself


def test_the_run_goes_to_the_window_the_mode_asked_for(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Which mode that is, and what it takes, is not this command's business."""
    window = RecordingWindow()
    asked_for: list[str] = []
    _install(monkeypatch, StubT2VApplication(_stand_in()))
    monkeypatch.setattr(
        cli,
        "client_window_mode",
        lambda name: (asked_for.append(name), StubMode(name, window))[1],
    )

    cli.entrypoint(["stub", "--mode", "webrtc", "--", "--prompt", _PROMPT])

    assert asked_for == ["webrtc"]
    # And nothing counted the steps: the session reported itself finished.
    assert len(window.results) == _TOTAL_BLOCKS


def test_the_command_needs_somewhere_to_write() -> None:
    with pytest.raises(SystemExit):
        cli.entrypoint(["stub"])
