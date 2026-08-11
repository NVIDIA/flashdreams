# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Model-neutral launch capabilities for ``flashdreams-run``."""

from __future__ import annotations

import importlib
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path
from typing import Literal, Protocol, TypeAlias, runtime_checkable

from flashdreams.infra.runner import RunnerConfig

LaunchMode: TypeAlias = Literal["run", "mp4", "null", "webrtc", "local-window"]


class LaunchModeUnavailableError(ValueError):
    """Raised when a runner does not implement a selected launch mode."""


@dataclass(frozen=True, slots=True)
class LaunchOptions:
    """Model-neutral settings passed from the central CLI to an integration."""

    host: str | None = None
    port: int | None = None
    prefer_sw_encoder: bool = False
    legacy_world_manifest: Path | None = None
    launch_manifest: Path | None = None
    scenario: Mapping[str, object] = field(default_factory=dict)
    output: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ResolvedLaunch:
    """Validated launch ready to execute without invoking another CLI."""

    mode: LaunchMode
    label: str
    launch: Callable[[], object] = field(repr=False)
    summary: Mapping[str, object] = field(default_factory=dict)
    notes: tuple[str, ...] = ()


@runtime_checkable
class LaunchCapability(Protocol):
    """Integration-owned modes and launch construction for one runner config."""

    def supported_modes(
        self,
        config: RunnerConfig,
        options: LaunchOptions,
    ) -> tuple[LaunchMode, ...]: ...

    def resolve(
        self,
        config: RunnerConfig,
        *,
        mode: LaunchMode,
        options: LaunchOptions,
    ) -> ResolvedLaunch | None: ...


def available_launch_modes(
    config: RunnerConfig,
    options: LaunchOptions | None = None,
) -> tuple[LaunchMode, ...]:
    """Return the modes implemented for ``config``."""
    options = options or LaunchOptions()
    capability = _resolve_capability(config)
    if capability is None:
        return ("run",)
    modes = capability.supported_modes(config, options)
    if "run" in modes:
        raise ValueError("Launch capabilities must not declare built-in mode 'run'.")
    return ("run", *dict.fromkeys(modes))


def resolve_launch(
    config: RunnerConfig,
    *,
    mode: LaunchMode,
    options: LaunchOptions | None = None,
) -> ResolvedLaunch:
    """Validate and construct a non-``run`` launch."""
    if mode == "run":
        raise ValueError("Mode 'run' is executed directly by the selected Runner.")
    options = options or LaunchOptions()
    capability = _resolve_capability(config)
    resolved = (
        None
        if capability is None
        else capability.resolve(config, mode=mode, options=options)
    )
    if resolved is None:
        supported = ", ".join(available_launch_modes(config, options))
        raise LaunchModeUnavailableError(
            f"Launch mode {mode!r} is not available for runner "
            f"{config.runner_name!r}. Supported modes: {supported}."
        )
    if resolved.mode != mode:
        raise ValueError(
            f"Launch capability returned mode {resolved.mode!r} while resolving "
            f"{mode!r}."
        )
    return resolved


def _resolve_capability(config: RunnerConfig) -> LaunchCapability | None:
    path = config.launch_capability
    if not path:
        return None
    return _load_launch_capability(path)


@cache
def _load_launch_capability(path: str) -> LaunchCapability:
    try:
        module_name, attribute = path.split(":", 1)
    except ValueError as exc:
        raise ValueError(
            "RunnerConfig.launch_capability must use 'module:attribute' syntax; "
            f"got {path!r}."
        ) from exc
    value = getattr(importlib.import_module(module_name), attribute)
    if callable(value) and not isinstance(value, LaunchCapability):
        value = value()
    if not isinstance(value, LaunchCapability):
        raise TypeError(
            f"Launch capability {path!r} does not implement LaunchCapability."
        )
    return value


__all__ = [
    "LaunchCapability",
    "LaunchMode",
    "LaunchModeUnavailableError",
    "LaunchOptions",
    "ResolvedLaunch",
    "available_launch_modes",
    "resolve_launch",
]
