# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Strict YAML configuration validation helpers."""

from __future__ import annotations

import math
import types
from dataclasses import MISSING, fields, is_dataclass, replace
from pathlib import Path
from typing import Any, Literal, Union, get_args, get_origin, get_type_hints

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


def overlay_dataclass(
    base: Any,
    values: dict[str, Any],
    context: str,
    *,
    base_dir: Path,
) -> Any:
    """Strictly overlay a YAML mapping onto a frozen configuration dataclass.

    Unknown fields are rejected, omitted fields retain their typed defaults,
    and relative :class:`~pathlib.Path` values resolve beside the YAML file.
    Nested dataclasses and tuples are handled recursively.

    Args:
        base: Lower-precedence dataclass instance.
        values: Partial YAML mapping to apply.
        context: Field path used in validation errors.
        base_dir: Directory used to resolve relative paths.

    Returns:
        A replaced dataclass instance containing the validated overlay.

    Raises:
        StrictConfigError: A field is unknown, incorrectly typed, or invalid.
    """
    if not is_dataclass(base) or isinstance(base, type):
        raise TypeError(f"{context} base must be a dataclass instance")
    known = {item.name: item for item in fields(base)}
    unknown = sorted(values.keys() - known.keys())
    if unknown:
        raise StrictConfigError(f"{context} has unknown keys: {', '.join(unknown)}")
    hints = get_type_hints(type(base))
    updates = {
        name: _convert_typed_value(
            raw,
            hints[name],
            f"{context}.{name}",
            base_dir=base_dir,
            current=getattr(base, name),
        )
        for name, raw in values.items()
    }
    try:
        return replace(base, **updates)
    except (TypeError, ValueError) as exc:
        raise StrictConfigError(f"{context} is invalid: {exc}") from exc


def _convert_typed_value(
    value: Any,
    expected: Any,
    context: str,
    *,
    base_dir: Path,
    current: Any = None,
) -> Any:
    origin = get_origin(expected)
    args = get_args(expected)
    if origin in (Union, types.UnionType):
        if value is None and type(None) in args:
            return None
        errors: list[str] = []
        for candidate in (arg for arg in args if arg is not type(None)):
            try:
                return _convert_typed_value(
                    value,
                    candidate,
                    context,
                    base_dir=base_dir,
                    current=current,
                )
            except StrictConfigError as exc:
                errors.append(str(exc))
        raise StrictConfigError(errors[-1] if errors else f"{context} is invalid")
    if origin is Literal:
        if value not in args or type(value) not in {type(item) for item in args}:
            choices = ", ".join(repr(item) for item in args)
            raise StrictConfigError(f"{context} must be one of {choices}")
        return value
    if origin is tuple:
        if not isinstance(value, list):
            raise StrictConfigError(f"{context} must be a sequence")
        if len(args) == 2 and args[1] is Ellipsis:
            item_type = args[0]
            return tuple(
                _convert_typed_value(
                    item,
                    item_type,
                    f"{context}[{index}]",
                    base_dir=base_dir,
                )
                for index, item in enumerate(value)
            )
        if len(value) != len(args):
            raise StrictConfigError(f"{context} must contain {len(args)} values")
        return tuple(
            _convert_typed_value(
                item,
                item_type,
                f"{context}[{index}]",
                base_dir=base_dir,
            )
            for index, (item, item_type) in enumerate(zip(value, args, strict=True))
        )
    if isinstance(expected, type) and is_dataclass(expected):
        mapping = require_mapping(value, context)
        if is_dataclass(current):
            return overlay_dataclass(current, mapping, context, base_dir=base_dir)
        known = {item.name: item for item in fields(expected)}
        unknown = sorted(mapping.keys() - known.keys())
        if unknown:
            raise StrictConfigError(f"{context} has unknown keys: {', '.join(unknown)}")
        missing = sorted(
            name
            for name, item in known.items()
            if name not in mapping
            and item.default is MISSING
            and item.default_factory is MISSING
        )
        if missing:
            raise StrictConfigError(
                f"{context} is missing required keys: {', '.join(missing)}"
            )
        hints = get_type_hints(expected)
        converted = {
            name: _convert_typed_value(
                raw,
                hints[name],
                f"{context}.{name}",
                base_dir=base_dir,
            )
            for name, raw in mapping.items()
        }
        try:
            return expected(**converted)
        except (TypeError, ValueError) as exc:
            raise StrictConfigError(f"{context} is invalid: {exc}") from exc
    if expected is Path:
        if not isinstance(value, str):
            raise StrictConfigError(f"{context} must be a path string")
        path = Path(value).expanduser()
        return path if path.is_absolute() else (base_dir / path).resolve()
    if expected is bool:
        return require_bool(value, context)
    if expected is int:
        if type(value) is not int:
            raise StrictConfigError(f"{context} must be an integer")
        return value
    if expected is float:
        return require_float(value, context)
    if expected is str:
        if not isinstance(value, str):
            raise StrictConfigError(f"{context} must be a string")
        return value
    raise StrictConfigError(f"{context} has unsupported type {expected!r}")
