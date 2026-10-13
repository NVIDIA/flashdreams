# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for flashdreams-run-v2 parse and forwarding.

Covers ``--timeout`` / ``--total-model-steps`` into ``ApplicationRunner.run``,
and the application's preferred client-window mode when ``--mode`` is omitted.
Leftover steps across replacements are covered in ``test_application_runner.py``.
"""

import argparse
from collections.abc import Callable, Sequence
from pathlib import Path
from types import SimpleNamespace

import pytest

from flashdreams.api_v2.application import IApplication
from flashdreams.api_v2.client_window import IClientWindow
from flashdreams.api_v2.session import ISession
from flashdreams.runtime_v2 import cli, client_window_factory
from flashdreams.runtime_v2.application_runner import ApplicationRunner, Unbound
from flashdreams.runtime_v2.client_window_factory import ClientWindowMode
from flashdreams.runtime_v2.mp4_client_window import Mp4ClientWindow
from flashdreams.runtime_v2.session_desc import PresentationMode, SessionDesc
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
    def __init__(self, name: str, *, modes: object = None) -> None:
        del modes
        self.name = name

    def check_arguments(self, parsed_args: argparse.Namespace) -> None:
        if self.name == "mp4" and getattr(parsed_args, "output_path", None) is None:
            raise ValueError("--output-path is required when writing an MP4.")

    def create(self, parsed_args: argparse.Namespace) -> IClientWindow:
        del parsed_args
        return _StubWindow()


def _record_run(
    monkeypatch: pytest.MonkeyPatch,
) -> list[tuple[float | Unbound, int | Unbound]]:
    received: list[tuple[float | Unbound, int | Unbound]] = []

    def record(
        self: object,
        session_desc: SessionDesc,
        commandline_args: Sequence[str],
        *,
        timeout_seconds: float | Unbound = Unbound.unbound,
        steps: int | Unbound = Unbound.unbound,
    ) -> None:
        del self, session_desc, commandline_args
        received.append((timeout_seconds, steps))

    monkeypatch.setattr(cli, "create_application", lambda slug: _StubApplication())
    monkeypatch.setattr(cli, "client_window_mode", _StubMode)
    monkeypatch.setattr(cli.ApplicationRunner, "run", record)
    return received


def test_mp4_requires_timeout_or_total_model_steps(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Omitting both is the case ``--mode mp4`` refuses."""
    _record_run(monkeypatch)
    with pytest.raises(SystemExit):
        cli.entrypoint(["stub", "--mode", "mp4", "--output-path", "out.mp4"])

    assert "requires --timeout and/or --total-model-steps" in capsys.readouterr().err


def test_default_mp4_requires_timeout_or_total_model_steps(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """An omitted ``--mode`` still lands on mp4 and needs a declared stop."""
    _record_run(monkeypatch)
    with pytest.raises(SystemExit):
        cli.entrypoint(["stub", "--output-path", "out.mp4"])

    assert "requires --timeout and/or --total-model-steps" in capsys.readouterr().err


def test_mp4_accepts_timeout_unbound(monkeypatch: pytest.MonkeyPatch) -> None:
    """Naming ``unbound`` is enough; omit and ``unbound`` are not the same token."""
    received = _record_run(monkeypatch)

    cli.entrypoint(
        ["stub", "--mode", "mp4", "--output-path", "out.mp4", "--timeout", "unbound"]
    )

    assert received == [(Unbound.unbound, Unbound.unbound)]


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

    assert received == [(Unbound.unbound, Unbound.unbound)]


def test_mp4_accepts_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    """A seconds value alone is enough, and it is what the runner receives."""
    received = _record_run(monkeypatch)

    cli.entrypoint(
        ["stub", "--mode", "mp4", "--output-path", "out.mp4", "--timeout", "60"]
    )

    assert received == [(60.0, Unbound.unbound)]


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

    assert received == [(Unbound.unbound, expected)]


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

    assert received == [(Unbound.unbound, Unbound.unbound)]


@pytest.mark.parametrize("mode", ["webrtc", "native-window"])
def test_webrtc_and_native_window_may_omit_timeout_and_total_model_steps(
    monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    """The requirement is ``--mode mp4``'s, not every mode's."""
    received = _record_run(monkeypatch)

    cli.entrypoint(["stub", "--mode", mode])

    assert received == [(Unbound.unbound, Unbound.unbound)]


@pytest.mark.parametrize(
    "mode, extra, expected",
    [
        ("webrtc", ["--timeout", "2.5"], (2.5, Unbound.unbound)),
        ("webrtc", ["--total-model-steps", "20"], (Unbound.unbound, 20)),
        ("native-window", ["--timeout", "2.5"], (2.5, Unbound.unbound)),
        ("native-window", ["--total-model-steps", "20"], (Unbound.unbound, 20)),
    ],
)
def test_webrtc_and_native_window_pass_timeout_and_total_model_steps(
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
    extra: list[str],
    expected: tuple[float | Unbound, int | Unbound],
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

    assert received == [(Unbound.unbound, Unbound.unbound)]


def test_total_model_steps_unbound_is_the_same_as_omitting_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    received = _record_run(monkeypatch)

    cli.entrypoint(["stub", "--mode", "webrtc", "--total-model-steps", "unbound"])

    assert received == [(Unbound.unbound, Unbound.unbound)]


@pytest.mark.parametrize("timeout", ["0", "-1", "nan", "inf"])
def test_the_command_rejects_an_invalid_timeout(timeout: str) -> None:
    with pytest.raises(SystemExit):
        cli.entrypoint(["stub", "--mode", "webrtc", "--timeout", timeout])


@pytest.mark.parametrize("steps", ["-1", "1.5", "nan"])
def test_the_command_rejects_an_invalid_step_limit(steps: str) -> None:
    with pytest.raises(SystemExit):
        cli.entrypoint(["stub", "--mode", "webrtc", "--total-model-steps", steps])


class _Application(IApplication):
    def __init__(self) -> None:
        self.initialized = False

    def init(self, commandline_args: Sequence[str]) -> None:
        parser = argparse.ArgumentParser(prog="test-application")
        parser.parse_args(commandline_args)
        self.initialized = True

    def create_session(self, session_desc: SessionDesc) -> ISession:
        raise AssertionError("The CLI test must not run a model session.")


class _PreferredApplication(_Application):
    def __init__(self, mode: str) -> None:
        super().__init__()
        self.mode = mode
        self.default_queries = 0

    def default_client_window_mode(self) -> str:
        assert not self.initialized
        self.default_queries += 1
        return self.mode


class _TestMode(ClientWindowMode):
    name = "test-window"

    def __init__(self) -> None:
        self.checked = False
        self.created = False

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        self.output_action = parser.add_argument("--test-output", type=Path)

    def check_arguments(self, parsed_args: argparse.Namespace) -> None:
        if getattr(parsed_args, self.output_action.dest) is None:
            raise ValueError("--test-output is required")
        self.checked = True

    def create(self, parsed_args: argparse.Namespace) -> Mp4ClientWindow:
        self.created = True
        return Mp4ClientWindow(getattr(parsed_args, self.output_action.dest))


def _install(
    monkeypatch: pytest.MonkeyPatch,
    application: IApplication,
    *,
    load_mode: Callable[[], object] | None = None,
) -> tuple[_TestMode, list[tuple[SessionDesc, Sequence[str]]]]:
    mode = _TestMode()
    runs: list[tuple[SessionDesc, Sequence[str]]] = []
    monkeypatch.setattr(cli, "create_application", lambda _slug: application)
    monkeypatch.setattr(cli, "registered_application_slugs", lambda: ("test-app",))
    entry_point = SimpleNamespace(
        name=mode.name,
        value="test_plugin:mode",
        load=(lambda: mode) if load_mode is None else load_mode,
    )
    monkeypatch.setattr(
        client_window_factory,
        "entry_points",
        lambda *, group: (entry_point,),
    )

    def run(
        _runner: ApplicationRunner,
        session_desc: SessionDesc,
        commandline_args: Sequence[str],
        *,
        timeout_seconds: float | Unbound = Unbound.unbound,
        steps: int | Unbound = Unbound.unbound,
    ) -> None:
        del timeout_seconds, steps
        runs.append((session_desc, commandline_args))
        application.init(commandline_args)

    monkeypatch.setattr(ApplicationRunner, "run", run)
    return mode, runs


def test_inherited_default_remains_mp4(monkeypatch: pytest.MonkeyPatch) -> None:
    application = _Application()
    mode, runs = _install(monkeypatch, application)

    cli.entrypoint(["test-app", "--output-path", "output.mp4", "--timeout", "unbound"])

    assert application.initialized
    assert not mode.created
    assert runs[0][0].presentation_mode == PresentationMode.ON_DEMAND


@pytest.mark.parametrize("explicit", [False, True])
def test_application_default_and_explicit_override(
    monkeypatch: pytest.MonkeyPatch, explicit: bool
) -> None:
    application = _PreferredApplication("test-window")
    mode, runs = _install(monkeypatch, application)
    arguments = ["test-app", "--test-output", "plugin.mp4"]
    if explicit:
        arguments.extend(
            [
                "--mode",
                "mp4",
                "--output-path",
                "explicit.mp4",
                "--timeout",
                "unbound",
            ]
        )

    cli.entrypoint(arguments)

    assert application.initialized
    assert mode.created is not explicit
    assert application.default_queries == (0 if explicit else 1)
    assert runs[0][0].presentation_mode == (
        PresentationMode.ON_DEMAND if explicit else PresentationMode.CONTINUOUS
    )


@pytest.mark.parametrize("preferred", ["test-window", "not-installed"])
def test_invalid_default_or_arguments_fail_before_initialization(
    monkeypatch: pytest.MonkeyPatch, preferred: str
) -> None:
    application = _PreferredApplication(preferred)
    mode, runs = _install(monkeypatch, application)

    with pytest.raises(SystemExit) as error:
        cli.entrypoint(["test-app"])

    assert error.value.code == 2
    assert not application.initialized
    assert not mode.created
    assert not runs


def test_application_help_does_not_resolve_or_validate_a_window(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    application = _PreferredApplication("not-installed")
    mode, runs = _install(monkeypatch, application)

    with pytest.raises(SystemExit) as error:
        cli.entrypoint(["test-app", "--", "--help"])

    assert error.value.code == 0
    assert "test-application" in capsys.readouterr().out
    assert application.default_queries == 0
    assert not mode.created
    assert not runs


def test_runtime_help_does_not_construct_an_application(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _install(monkeypatch, _Application())
    monkeypatch.setattr(
        cli, "create_application", lambda _slug: pytest.fail("Application constructed")
    )

    with pytest.raises(SystemExit) as error:
        cli.entrypoint(["test-app", "--help"])

    assert error.value.code == 0
    help_text = " ".join(capsys.readouterr().out.split())
    assert "application's preference, otherwise mp4" in help_text


def test_mode_factory_preserves_argument_state_for_each_cli_invocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    instances: list[_TestMode] = []

    def create_mode() -> _TestMode:
        mode = _TestMode()
        instances.append(mode)
        return mode

    _mode, runs = _install(monkeypatch, _Application(), load_mode=lambda: create_mode)

    for invocation in (1, 2):
        cli.entrypoint(
            ["test-app", "--mode", "test-window", "--test-output", "plugin.mp4"]
        )
        assert len(instances) == invocation
        assert instances[-1].checked
        assert instances[-1].created
        assert len(runs) == invocation


def _broken_mode_loader(stage: str) -> Callable[[], object]:
    def fail() -> object:
        raise ModuleNotFoundError("missing optional backend")

    return fail if stage == "load" else lambda: fail


@pytest.mark.parametrize("stage", ["load", "factory"])
@pytest.mark.parametrize("cli_request", ["mp4", "runtime-help", "application-help"])
def test_unavailable_unselected_mode_keeps_mp4_and_help_usable(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    stage: str,
    cli_request: str,
) -> None:
    application = _Application()
    mode, runs = _install(
        monkeypatch, application, load_mode=_broken_mode_loader(stage)
    )

    if cli_request == "mp4":
        cli.entrypoint(
            [
                "test-app",
                "--mode",
                "mp4",
                "--output-path",
                "output.mp4",
                "--timeout",
                "unbound",
            ]
        )
        assert application.initialized
        assert len(runs) == 1
    else:
        arguments = ["test-app"]
        if cli_request == "application-help":
            arguments.append("--")
        arguments.append("--help")
        with pytest.raises(SystemExit) as error:
            cli.entrypoint(arguments)
        assert error.value.code == 0
        assert not application.initialized
        assert not runs
    assert not mode.created
    assert "test-window" in caplog.text
    assert "missing optional backend" in caplog.text


@pytest.mark.parametrize("stage", ["load", "factory"])
@pytest.mark.parametrize("explicit", [False, True])
def test_unavailable_requested_mode_does_not_fall_back_or_initialize(
    monkeypatch: pytest.MonkeyPatch, stage: str, explicit: bool
) -> None:
    application = _PreferredApplication("test-window")
    mode, runs = _install(
        monkeypatch, application, load_mode=_broken_mode_loader(stage)
    )
    arguments = ["test-app", "--output-path", "fallback.mp4"]
    if explicit:
        arguments.extend(["--mode", "test-window"])

    with pytest.raises(SystemExit) as error:
        cli.entrypoint(arguments)

    assert error.value.code == 2
    assert not application.initialized
    assert not mode.created
    assert not runs
