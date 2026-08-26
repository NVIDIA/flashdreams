# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Strict YAML configuration validation helpers."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import yaml


class StrictConfigError(ValueError):
    """Invalid strict YAML configuration."""


def load_yaml_mapping(path: Path, *, suffix: str | None = None) -> dict[str, Any]:
    """Load one YAML document as a mapping.

    Args:
        path: YAML file to load.
        suffix: Required filename suffix; ``None`` accepts any filename.

    Returns:
        Parsed root mapping.

    Raises:
        StrictConfigError: The path or YAML document is invalid.
    """
    path = path.expanduser().resolve()
    if not path.is_file():
        raise StrictConfigError(f"Configuration path does not exist: {path}")
    if suffix is not None and not path.name.endswith(suffix):
        raise StrictConfigError(f"Configuration must use the {suffix} suffix: {path}")
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise StrictConfigError(f"Could not parse {path}: {exc}") from exc
    return require_mapping(value, str(path))


def require_mapping(value: Any, context: str) -> dict[str, Any]:
    """Return ``value`` after validating that it is a string-keyed mapping."""
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise StrictConfigError(f"{context} must be a mapping with string keys")
    return value


def require_exact_keys(value: dict[str, Any], expected: set[str], context: str) -> None:
    """Require a mapping to contain exactly ``expected`` keys."""
    missing = sorted(expected - value.keys())
    unknown = sorted(value.keys() - expected)
    if missing:
        raise StrictConfigError(
            f"{context} is missing required keys: {', '.join(missing)}"
        )
    if unknown:
        raise StrictConfigError(f"{context} has unknown keys: {', '.join(unknown)}")


def require_version(value: dict[str, Any], context: str) -> None:
    """Require schema version one."""
    version = value.get("schema_version")
    if type(version) is not int or version != 1:
        raise StrictConfigError(f"{context}.schema_version must be 1")


def require_bool(value: Any, context: str) -> bool:
    """Return a strictly typed Boolean value."""
    if type(value) is not bool:
        raise StrictConfigError(f"{context} must be a boolean")
    return value


def require_int(value: Any, context: str, *, minimum: int = 1) -> int:
    """Return an integer at or above ``minimum``."""
    if type(value) is not int or value < minimum:
        raise StrictConfigError(f"{context} must be an integer >= {minimum}")
    return value


def require_float(
    value: Any,
    context: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    """Return a finite numeric value within the requested range."""
    if type(value) not in (int, float):
        raise StrictConfigError(f"{context} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise StrictConfigError(f"{context} must be finite")
    if minimum is not None and result < minimum:
        raise StrictConfigError(f"{context} must be >= {minimum}")
    if maximum is not None and result > maximum:
        raise StrictConfigError(f"{context} must be <= {maximum}")
    return result
