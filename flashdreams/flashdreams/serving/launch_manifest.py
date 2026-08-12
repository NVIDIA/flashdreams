# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Versioned manifests for ``flashdreams-run`` launch modes."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from flashdreams.infra.config import derive_config
from flashdreams.infra.runner import RunnerConfig

SCHEMA_VERSION = 1
_TOP_LEVEL_FIELDS = frozenset(
    {"schema_version", "runner", "mode", "runner_overrides", "scenario", "output"}
)
_MODES = frozenset({"run", "mp4", "null", "webrtc", "local-window"})


@dataclass(frozen=True, kw_only=True, slots=True)
class FlashDreamsLaunchManifest:
    """Resolved, model-neutral launch manifest."""

    path: Path
    schema_version: int
    runner: str
    mode: str
    runner_overrides: Mapping[str, Any] = field(default_factory=dict)
    scenario: Mapping[str, Any] = field(default_factory=dict)
    output: Mapping[str, Any] = field(default_factory=dict)

    def apply_runner_overrides(self, config: RunnerConfig) -> RunnerConfig:
        """Return ``config`` with manifest overrides applied recursively."""
        if config.runner_name != self.runner:
            raise ValueError(
                f"Manifest runner {self.runner!r} does not match "
                f"selected runner {config.runner_name!r}."
            )
        return derive_config(config, **dict(self.runner_overrides))


def load_launch_manifest(path: str | Path) -> FlashDreamsLaunchManifest:
    """Load and strictly validate one YAML launch manifest."""
    manifest_path = Path(path).expanduser().resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Launch manifest path does not exist or is not a file: {manifest_path}. "
            "Manifest paths are resolved relative to the current working directory "
            f"({Path.cwd()})."
        )
    raw = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise TypeError(f"Launch manifest {manifest_path} must contain a YAML mapping.")
    unknown = sorted(set(raw) - _TOP_LEVEL_FIELDS)
    if unknown:
        raise ValueError(
            f"Unknown launch manifest fields in {manifest_path}: {', '.join(unknown)}."
        )
    schema_version = raw.get("schema_version")
    if schema_version != SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported launch manifest schema_version={schema_version!r}; "
            f"expected {SCHEMA_VERSION}."
        )
    runner = _required_string(raw, "runner", manifest_path)
    mode = _required_string(raw, "mode", manifest_path)
    if mode not in _MODES:
        raise ValueError(
            f"Launch manifest {manifest_path} has unsupported mode {mode!r}; "
            f"expected one of: {', '.join(sorted(_MODES))}."
        )
    manifest_dir = manifest_path.parent
    return FlashDreamsLaunchManifest(
        path=manifest_path,
        schema_version=SCHEMA_VERSION,
        runner=runner,
        mode=mode,
        runner_overrides=_mapping(raw, "runner_overrides", manifest_path),
        scenario=_resolve_paths(
            _mapping(raw, "scenario", manifest_path), manifest_dir=manifest_dir
        ),
        output=_resolve_paths(
            _mapping(raw, "output", manifest_path), manifest_dir=manifest_dir
        ),
    )


def _required_string(raw: Mapping[str, Any], key: str, path: Path) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Launch manifest {path} requires non-empty {key!r}.")
    return value.strip()


def _mapping(raw: Mapping[str, Any], key: str, path: Path) -> Mapping[str, Any]:
    value = raw.get(key, {})
    if not isinstance(value, dict):
        raise TypeError(f"Launch manifest {path} field {key!r} must be a mapping.")
    return value


def _resolve_paths(value: Any, *, manifest_dir: Path, key: str = "") -> Any:
    if isinstance(value, dict):
        return {
            str(child_key): _resolve_paths(
                child_value,
                manifest_dir=manifest_dir,
                key=str(child_key),
            )
            for child_key, child_value in value.items()
        }
    if isinstance(value, list):
        return [
            _resolve_paths(item, manifest_dir=manifest_dir, key=key) for item in value
        ]
    if isinstance(value, str) and _is_path_key(key):
        path = Path(value).expanduser()
        return path if path.is_absolute() else (manifest_dir / path).resolve()
    return value


def _is_path_key(key: str) -> bool:
    normalized = key.replace("-", "_")
    return normalized in {"path", "output"} or normalized.endswith(
        ("_path", "_paths", "_dir")
    )


__all__ = [
    "SCHEMA_VERSION",
    "FlashDreamsLaunchManifest",
    "load_launch_manifest",
]
