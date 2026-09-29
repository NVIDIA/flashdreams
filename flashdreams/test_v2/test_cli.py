# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for flashdreams-run-v2 ``--timeout`` and ``--total-model-steps``."""

import argparse
from collections.abc import Sequence

import pytest

from flashdreams.api_v2.application import IApplication
from flashdreams.api_v2.client_window import IClientWindow
from flashdreams.api_v2.session import ISession
from flashdreams.runtime_v2 import cli
from flashdreams.runtime_v2.client_window_factory import ClientWindowMode
from flashdreams.runtime_v2.session_desc import SessionDesc
from flashdreams.runtime_v2.step_result import StepResult
from flashdreams.runtime_v2.user_input_events import UserInputEvents

pytestmark = pytest.mark.ci_cpu


class _StubApplication(IApplication):
    def init(self, commandline_args: Sequence[str]) -> None:
        del commandline_args

    def create_session(self, session_desc: SessionDesc) -> ISession:
        del session_desc
        raise AssertionError("These tests stop before a session is created.")


class _StubWindow(IClientWindow):
    def get_user_input_events(self) -> UserInputEvents:
        return UserInputEvents([])

    def open(self, session_desc: SessionDesc) -> None:
        del session_desc

    def write(self, result: StepResult) -> None:
        del result

    def close(self) -> None:
        return


class _StubMode(ClientWindowMode):
    def __init__(self, name: str) -> None:
        self.name = name

    def check_arguments(self, parsed_args: argparse.Namespace) -> None:
        if self.name == "mp4" and getattr(parsed_args, "output_path", None) is None:
            raise ValueError("--output-path is required when writing an MP4.")

    def create(self, parsed_args: argparse.Namespace) -> IClientWindow:
        del parsed_args
        return _StubWindow()


def _record_run(
    monkeypatch: pytest.MonkeyPatch,
) -> list[tuple[float | None, int | None]]:
    received: list[tuple[float | None, int | None]] = []

    def record(
        self: object,
        session_desc: SessionDesc,
        commandline_args: Sequence[str],
        *,
        timeout_seconds: float | None = None,
        steps: int | None = None,
    ) -> None:
        del self, session_desc, commandline_args
        received.append((timeout_seconds, steps))

    monkeypatch.setattr(cli, "create_application", lambda slug: _StubApplication())
    monkeypatch.setattr(cli, "client_window_mode", _StubMode)
    monkeypatch.setattr(cli.ApplicationRunner, "run", record)
    return received


def test_mp4_requires_timeout_or_total_model_steps(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Omitting both is the case ``--mode mp4`` refuses."""
    with pytest.raises(SystemExit):
        cli.entrypoint(["stub", "--mode", "mp4", "--output-path", "out.mp4"])

    assert "requires --timeout and/or --total-model-steps" in capsys.readouterr().err


def test_mp4_accepts_timeout_unbound(monkeypatch: pytest.MonkeyPatch) -> None:
    """Naming ``unbound`` is enough; omit and ``unbound`` are not the same token."""
    received = _record_run(monkeypatch)

    cli.entrypoint(
        ["stub", "--mode", "mp4", "--output-path", "out.mp4", "--timeout", "unbound"]
    )

    assert received == [(None, None)]


def test_mp4_accepts_total_model_steps_unbound(monkeypatch: pytest.MonkeyPatch) -> None:
    """The other flag's ``unbound`` also satisfies ``--mode mp4``."""
    received = _record_run(monkeypatch)

    cli.entrypoint(
        [
            "stub",
            "--mode",
            "mp4",
            "--output-path",
            "out.mp4",
            "--total-model-steps",
            "unbound",
        ]
    )

    assert received == [(None, None)]


def test_mp4_accepts_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    """A seconds value alone is enough, and it is what the runner receives."""
    received = _record_run(monkeypatch)

    cli.entrypoint(
        ["stub", "--mode", "mp4", "--output-path", "out.mp4", "--timeout", "60"]
    )

    assert received == [(60.0, None)]


@pytest.mark.parametrize("steps, expected", [("50", 50), ("0", 0)])
def test_mp4_accepts_total_model_steps(
    monkeypatch: pytest.MonkeyPatch, steps: str, expected: int
) -> None:
    """A steps value alone is enough. ``0`` is a count, not ``unbound``."""
    received = _record_run(monkeypatch)

    cli.entrypoint(
        [
            "stub",
            "--mode",
            "mp4",
            "--output-path",
            "out.mp4",
            "--total-model-steps",
            steps,
        ]
    )

    assert received == [(None, expected)]


def test_mp4_passes_timeout_and_total_model_steps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both may be named; the command does not treat them as exclusive."""
    received = _record_run(monkeypatch)

    cli.entrypoint(
        [
            "stub",
            "--mode",
            "mp4",
            "--output-path",
            "out.mp4",
            "--total-model-steps",
            "50",
            "--timeout",
            "60",
        ]
    )

    assert received == [(60.0, 50)]


def test_mp4_accepts_timeout_and_total_model_steps_unbound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Naming both as ``unbound`` still satisfies ``--mode mp4``."""
    received = _record_run(monkeypatch)

    cli.entrypoint(
        [
            "stub",
            "--mode",
            "mp4",
            "--output-path",
            "out.mp4",
            "--timeout",
            "unbound",
            "--total-model-steps",
            "unbound",
        ]
    )

    assert received == [(None, None)]


@pytest.mark.parametrize("mode", ["webrtc", "native-window"])
def test_webrtc_and_native_window_may_omit_timeout_and_total_model_steps(
    monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    """The requirement is ``--mode mp4``'s, not every mode's."""
    received = _record_run(monkeypatch)

    cli.entrypoint(["stub", "--mode", mode])

    assert received == [(None, None)]


@pytest.mark.parametrize(
    "mode, extra, expected",
    [
        ("webrtc", ["--timeout", "2.5"], (2.5, None)),
        ("webrtc", ["--total-model-steps", "20"], (None, 20)),
        ("native-window", ["--timeout", "2.5"], (2.5, None)),
        ("native-window", ["--total-model-steps", "20"], (None, 20)),
    ],
)
def test_webrtc_and_native_window_pass_timeout_and_total_model_steps(
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
    extra: list[str],
    expected: tuple[float | None, int | None],
) -> None:
    """The flags belong to the command, so every mode forwards them."""
    received = _record_run(monkeypatch)

    cli.entrypoint(["stub", "--mode", mode, *extra])

    assert received == [expected]


def test_timeout_unbound_is_the_same_as_omitting_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    received = _record_run(monkeypatch)

    cli.entrypoint(["stub", "--mode", "webrtc", "--timeout", "unbound"])

    assert received == [(None, None)]


def test_total_model_steps_unbound_is_the_same_as_omitting_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    received = _record_run(monkeypatch)

    cli.entrypoint(["stub", "--mode", "webrtc", "--total-model-steps", "unbound"])

    assert received == [(None, None)]


@pytest.mark.parametrize("timeout", ["0", "-1", "nan", "inf"])
def test_the_command_rejects_an_invalid_timeout(timeout: str) -> None:
    with pytest.raises(SystemExit):
        cli.entrypoint(["stub", "--mode", "webrtc", "--timeout", timeout])


@pytest.mark.parametrize("steps", ["-1", "1.5", "nan"])
def test_the_command_rejects_an_invalid_step_limit(steps: str) -> None:
    with pytest.raises(SystemExit):
        cli.entrypoint(["stub", "--mode", "webrtc", "--total-model-steps", steps])
