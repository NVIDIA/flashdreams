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

"""CPU tests for application-selected V2 client-window defaults."""

import argparse
from collections.abc import Callable, Sequence
from pathlib import Path
from types import SimpleNamespace

import pytest

from flashdreams.api_v2.application import IApplication
from flashdreams.api_v2.session import ISession
from flashdreams.runtime_v2 import cli, client_window_factory
from flashdreams.runtime_v2.application_runner import ApplicationRunner
from flashdreams.runtime_v2.client_window_factory import ClientWindowMode
from flashdreams.runtime_v2.mp4_client_window import Mp4ClientWindow
from flashdreams.runtime_v2.session_desc import PresentationMode, SessionDesc

pytestmark = pytest.mark.ci_cpu


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
        timeout_seconds: float | None,
    ) -> None:
        runs.append((session_desc, commandline_args))
        application.init(commandline_args)

    monkeypatch.setattr(ApplicationRunner, "run", run)
    return mode, runs


def test_inherited_default_remains_mp4(monkeypatch: pytest.MonkeyPatch) -> None:
    application = _Application()
    mode, runs = _install(monkeypatch, application)

    cli.entrypoint(["test-app", "--output-path", "output.mp4"])

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
        arguments.extend(["--mode", "mp4", "--output-path", "explicit.mp4"])

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
        cli.entrypoint(["test-app", "--mode", "mp4", "--output-path", "output.mp4"])
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
