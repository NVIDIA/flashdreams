# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Output target selection for ``flashdreams-run``."""

from __future__ import annotations

import importlib
import runpy
import shlex
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Literal, Protocol, TypeAlias, runtime_checkable

from flashdreams.infra.runner import RunnerConfig

OutputMode: TypeAlias = Literal["cli", "webrtc", "local-window"]


class OutputTargetUnavailableError(ValueError):
    """Raised when a runner cannot be launched through a requested output."""


@dataclass(frozen=True, slots=True)
class OutputLaunchOptions:
    """Common launch options shared by non-CLI output targets."""

    host: str | None = None
    port: int | None = None
    prefer_sw_encoder: bool = False
    local_window_manifest: Path | None = None


@dataclass(frozen=True, slots=True)
class OutputTargetSpec:
    """A concrete output target module plus argv translated from a runner config."""

    mode: OutputMode
    label: str
    module: str
    argv: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    @property
    def command(self) -> str:
        """Return a copy-pasteable module command for diagnostics."""
        return shlex.join(("python", "-m", self.module, *self.argv))


@runtime_checkable
class OutputTargetAdapter(Protocol):
    """Integration-owned non-CLI output capabilities for a runner config."""

    def supported_modes(
        self,
        config: RunnerConfig,
        options: OutputLaunchOptions,
    ) -> tuple[OutputMode, ...]: ...

    def resolve(
        self,
        config: RunnerConfig,
        *,
        mode: OutputMode,
        options: OutputLaunchOptions,
    ) -> OutputTargetSpec | None: ...


def available_output_modes(
    config: RunnerConfig,
    options: OutputLaunchOptions | None = None,
) -> tuple[OutputMode, ...]:
    """Return output modes known to support ``config``."""
    options = options or OutputLaunchOptions()
    adapter = _resolve_adapter(config)
    if adapter is None:
        return ("cli",)
    modes = adapter.supported_modes(config, options)
    invalid = [mode for mode in modes if mode == "cli"]
    if invalid:
        raise ValueError("Output adapters must not declare the built-in CLI mode.")
    return ("cli", *dict.fromkeys(modes))


def resolve_output_target(
    config: RunnerConfig,
    *,
    mode: OutputMode,
    options: OutputLaunchOptions | None = None,
) -> OutputTargetSpec:
    """Resolve a non-CLI output target for a runner config."""
    if mode == "cli":
        raise ValueError("CLI mode is run directly by the selected Runner.")
    options = options or OutputLaunchOptions()
    adapter = _resolve_adapter(config)
    spec = (
        None if adapter is None else adapter.resolve(config, mode=mode, options=options)
    )
    if spec is None:
        supported = ", ".join(available_output_modes(config, options))
        raise OutputTargetUnavailableError(
            f"Output mode {mode!r} is not available for runner "
            f"{config.runner_name!r}. Supported modes: {supported}."
        )
    if spec.mode != mode:
        raise ValueError(
            f"Output adapter returned mode {spec.mode!r} while resolving {mode!r}."
        )
    return spec


def launch_output_target(spec: OutputTargetSpec) -> None:
    """Execute an output target module as if launched with ``python -m``."""
    original_argv = sys.argv
    sys.argv = [spec.module, *spec.argv]
    try:
        runpy.run_module(spec.module, run_name="__main__")
    finally:
        sys.argv = original_argv


def _resolve_adapter(config: RunnerConfig) -> OutputTargetAdapter | None:
    path = config.output_adapter
    if not path:
        return None
    return _load_output_adapter(path)


@lru_cache(maxsize=None)
def _load_output_adapter(path: str) -> OutputTargetAdapter:
    try:
        module_name, attribute = path.split(":", 1)
    except ValueError as exc:
        raise ValueError(
            "RunnerConfig.output_adapter must use 'module:attribute' syntax; "
            f"got {path!r}."
        ) from exc
    value = getattr(importlib.import_module(module_name), attribute)
    if callable(value) and not isinstance(value, OutputTargetAdapter):
        value = value()
    if not isinstance(value, OutputTargetAdapter):
        raise TypeError(
            f"Output adapter {path!r} does not implement OutputTargetAdapter."
        )
    return value


__all__ = [
    "OutputLaunchOptions",
    "OutputMode",
    "OutputTargetSpec",
    "OutputTargetAdapter",
    "OutputTargetUnavailableError",
    "available_output_modes",
    "launch_output_target",
    "resolve_output_target",
]
