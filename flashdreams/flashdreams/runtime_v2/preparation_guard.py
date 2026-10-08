# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reject preparation work after application initialization."""

from __future__ import annotations

import inspect
import os
import sys
import threading
import warnings
from pathlib import Path
from types import TracebackType
from typing import Any, Literal

from flashdreams.core.io.disk import default_flashdreams_cache_dir

_AUDIT_OPERATIONS = {
    "os.exec": "launch an external process",
    "os.posix_spawn": "launch an external process",
    "os.spawn": "launch an external process",
    "os.startfile": "launch an external process",
    "os.startfile/2": "launch an external process",
    "os.system": "launch an external process",
    "socket.connect": "open a network connection",
    "subprocess.Popen": "launch an external process",
}
_POLICY_ENV = "FLASHDREAMS_PREPARATION_POLICY"
_ISSUES_PATH_ENV = "FLASHDREAMS_PREPARATION_ISSUES_PATH"
_ISSUES_FILENAME = "preparation_issues.txt"
_ISSUES_LOCK = threading.Lock()
_ACTIVE_GUARD: PreparationGuard | None = None
_HOOKS_INSTALLED = False

PreparationPolicy = Literal["none", "warn", "error"]


class PreparationPhaseError(RuntimeError):
    """Preparation-only work was attempted after ``IApplication.init``."""


class PreparationPhaseWarning(RuntimeWarning):
    """Preparation-only work was attempted after ``IApplication.init``."""


def resolve_preparation_policy(
    default: PreparationPolicy = "none",
) -> PreparationPolicy:
    """Return the environment policy, or ``default`` when it is unset."""
    configured = os.getenv(_POLICY_ENV, default)
    if configured == "none":
        return "none"
    if configured == "warn":
        return "warn"
    if configured == "error":
        return "error"
    raise ValueError(f"{_POLICY_ENV} must be 'none', 'warn', or 'error'.")


def write_preparation_issue(
    message: str,
    issues_path: str | Path | None = None,
) -> None:
    """Append one self-contained issue to the shared preparation report."""
    configured_path = issues_path or os.getenv(_ISSUES_PATH_ENV)
    path = (
        Path(configured_path).expanduser()
        if configured_path is not None
        else default_flashdreams_cache_dir() / _ISSUES_FILENAME
    )
    with _ISSUES_LOCK:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as report:
            report.write(
                "=== FlashDreams preparation issue ===\n"
                f"{message}\n"
                "=== End FlashDreams preparation issue ===\n\n"
            )


class PreparationGuard:
    """Reject application preparation work while the context is active."""

    def __init__(
        self,
        *,
        policy: PreparationPolicy | None = None,
        issues_path: str | Path | None = None,
    ) -> None:
        self._policy = policy or resolve_preparation_policy()
        configured_path = issues_path or os.getenv(_ISSUES_PATH_ENV)
        self._issues_path = (
            Path(configured_path).expanduser()
            if configured_path is not None
            else default_flashdreams_cache_dir() / _ISSUES_FILENAME
        )

    def __enter__(self) -> PreparationGuard:
        global _ACTIVE_GUARD

        if self._policy == "none":
            return self
        _install_hooks()
        if _ACTIVE_GUARD is not None:
            raise RuntimeError("An application preparation guard is already active.")
        _ACTIVE_GUARD = self
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        exception_traceback: TracebackType | None,
    ) -> None:
        global _ACTIVE_GUARD

        del exception_type, exception, exception_traceback
        if self._policy == "none":
            return
        if _ACTIVE_GUARD is self:
            _ACTIVE_GUARD = None

    def check(self, operation: str) -> None:
        """Report an operation while this guard is active."""
        if self._policy == "none" or _ACTIVE_GUARD is not self:
            return
        message = _violation_message(operation)
        try:
            write_preparation_issue(message, self._issues_path)
        except OSError as error:
            warnings.warn(
                f"Could not write preparation issue to {self._issues_path}: {error}",
                RuntimeWarning,
                stacklevel=2,
            )
        if self._policy == "error":
            raise PreparationPhaseError(message)
        warnings.warn(message, PreparationPhaseWarning, stacklevel=2)


def _install_hooks() -> None:
    """Install the process-wide hooks once; inactive hooks are no-ops."""
    global _HOOKS_INSTALLED

    if _HOOKS_INSTALLED:
        return

    sys.addaudithook(_audit_hook)

    import torch._dynamo as dynamo  # noqa: PLC0415

    dynamo.on_compile_start(_compile_started)
    _HOOKS_INSTALLED = True


def _audit_hook(event: str, arguments: tuple[Any, ...]) -> None:
    del arguments
    operation = _AUDIT_OPERATIONS.get(event)
    if operation is not None and _ACTIVE_GUARD is not None:
        _ACTIVE_GUARD.check(operation)


def _compile_started(arguments: Any) -> None:
    del arguments
    if _ACTIVE_GUARD is not None:
        _ACTIVE_GUARD.check("compile a model")


def _violation_message(operation: str) -> str:
    stack: list[tuple[str, int, str]] = []
    frame = inspect.currentframe()
    try:
        while frame is not None:
            filename = frame.f_code.co_filename
            if Path(filename) != Path(__file__):
                stack.append((filename, frame.f_lineno, frame.f_code.co_name))
            frame = frame.f_back
    finally:
        del frame

    rendered_stack = "\n".join(
        f"  {index}. {filename}:{lineno} in {name}"
        for index, (filename, lineno, name) in enumerate(stack, start=1)
    )
    return (
        f"Attempted to {operation} outside IApplication.init.\n"
        "Call stack (incident first):\n"
        f"{rendered_stack}"
    )
