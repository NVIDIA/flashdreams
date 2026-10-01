# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for the runtime preparation boundary."""

import sys
import threading
import warnings
from pathlib import Path

import pytest
import torch._dynamo as dynamo
from torch._dynamo.callback import CallbackArgs, CallbackTrigger

from flashdreams.api_v2.application import IApplication
from flashdreams.api_v2.session import ISession
from flashdreams.runtime_v2 import preparation_guard
from flashdreams.runtime_v2.preparation_guard import (
    PreparationGuard,
    PreparationPhaseError,
    PreparationPhaseWarning,
    resolve_preparation_policy,
)
from flashdreams.runtime_v2.session_desc import SessionDesc

pytestmark = pytest.mark.ci_cpu


class _Application(IApplication):
    def init(self, commandline_args: list[str]) -> None:
        del commandline_args

    def create_session(self, session_desc: SessionDesc) -> ISession:
        del session_desc
        raise NotImplementedError

    def audit(self, event: str) -> None:
        sys.audit(event, "test")

    def incident(self) -> None:
        sys.audit("subprocess.Popen", "test")

    def farther(self) -> None:
        self.incident()

    def farthest(self) -> None:
        self.farther()

    def compile(self, arguments: CallbackArgs) -> None:
        dynamo.callback_handler.run_start_callbacks(arguments)


def test_preparation_guard_reports_only_while_active_and_incident_first(
    tmp_path: Path,
) -> None:
    application = _Application()
    issues_path = tmp_path / "issues.txt"
    guard = PreparationGuard(
        policy="error",
        issues_path=issues_path,
    )

    application.incident()
    guard.check("compile a model")
    with guard:
        with pytest.raises(PreparationPhaseError) as caught:
            application.farthest()
    application.incident()
    guard.check("compile a model")

    message = str(caught.value)
    assert "outside IApplication.init" in message
    assert message.index("in incident") < message.index("in farther")
    assert message.index("in farther") < message.index("in farthest")
    assert message.index("in farthest") < message.index(
        "in test_preparation_guard_reports_only_while_active_and_incident_first"
    )
    report = issues_path.read_text(encoding="utf-8")
    assert "=== FlashDreams preparation issue ===\n" in report
    assert message in report
    assert report.endswith("=== End FlashDreams preparation issue ===\n\n")


def test_preparation_guard_rejects_network_and_compilation_after_init(
    tmp_path: Path,
) -> None:
    application = _Application()
    guard = PreparationGuard(
        policy="error",
        issues_path=tmp_path / "issues.txt",
    )
    compile_args = CallbackArgs(CallbackTrigger.DYNAMO, "test")

    with guard:
        with pytest.raises(PreparationPhaseError, match="open a network connection"):
            application.audit("socket.connect")
        with pytest.raises(PreparationPhaseError, match="compile a model"):
            application.compile(compile_args)

    sys.audit("subprocess.Popen", "command")


def test_preparation_guard_warns_in_flashdreams_cache_when_requested(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FLASHDREAMS_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("FLASHDREAMS_PREPARATION_POLICY", "warn")
    monkeypatch.delenv("FLASHDREAMS_PREPARATION_ISSUES_PATH", raising=False)
    application = _Application()
    guard = PreparationGuard()

    with guard:
        with pytest.warns(PreparationPhaseWarning, match="outside IApplication.init"):
            application.incident()
    application.incident()

    report = (tmp_path / "preparation_issues.txt").read_text(encoding="utf-8")
    assert "Attempted to launch an external process" in report


def test_preparation_guard_warns_for_unconnected_worker_thread(
    tmp_path: Path,
) -> None:
    issues_path = tmp_path / "preparation_issues.txt"
    guard = PreparationGuard(policy="warn", issues_path=issues_path)
    caught: list[warnings.WarningMessage] = []

    def incident() -> None:
        with warnings.catch_warnings(record=True) as worker_warnings:
            warnings.simplefilter("always")
            sys.audit("subprocess.Popen", "test")
        caught.extend(worker_warnings)

    with guard:
        worker = threading.Thread(target=incident)
        worker.start()
        worker.join()

    assert [warning.category for warning in caught] == [PreparationPhaseWarning]
    report = issues_path.read_text(encoding="utf-8")
    assert "Attempted to launch an external process" in report
    assert "in incident" in report


def test_preparation_guard_defaults_to_none_and_honors_none_override(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    issues_path = tmp_path / "preparation_issues.txt"
    monkeypatch.delenv("FLASHDREAMS_PREPARATION_POLICY", raising=False)
    assert resolve_preparation_policy("none") == "none"
    assert resolve_preparation_policy("warn") == "warn"
    monkeypatch.setenv("FLASHDREAMS_PREPARATION_POLICY", "none")
    assert resolve_preparation_policy("warn") == "none"
    monkeypatch.setenv("FLASHDREAMS_PREPARATION_ISSUES_PATH", str(issues_path))
    application = _Application()
    guard = PreparationGuard()

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with guard:
            application.incident()
        application.incident()
        guard.check("compile a model")

    assert not issues_path.exists()


def test_preparation_hooks_activate_on_enter_only_for_an_enabled_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    installed: list[bool] = []
    monkeypatch.setattr(
        preparation_guard, "_install_hooks", lambda: installed.append(True)
    )
    enabled = PreparationGuard(policy="warn")
    disabled = PreparationGuard(policy="none")

    assert installed == []
    with disabled:
        assert installed == []
    with enabled:
        assert installed == [True]
