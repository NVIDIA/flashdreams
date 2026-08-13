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

"""YAML pipeline-preset catalogs and declarative object-graph materialization."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Generic, Protocol, TypeVar, cast, runtime_checkable

import yaml

SCHEMA_VERSION = 1
"""Supported YAML pipeline-preset schema version."""

_ROOT_FIELDS = {"schema_version", "default_preset_id", "presets"}
_PRESET_FIELDS = {"provider", "runtime", "pipeline"}

RuntimeOptionsT = TypeVar("RuntimeOptionsT")
RuntimeOptionsT_co = TypeVar("RuntimeOptionsT_co", covariant=True)


class RuntimeOptionsParser(Protocol[RuntimeOptionsT_co]):
    """Parse application-specific runtime options from one preset."""

    def __call__(self, value: object, *, path: str) -> RuntimeOptionsT_co:
        """Parse runtime options with a source path for validation errors."""
        ...


@dataclass(frozen=True, slots=True)
class PipelinePreset(Generic[RuntimeOptionsT]):
    """Pipeline-provider selection and options for one preset."""

    provider: str
    """``module:attribute`` reference to a pipeline provider."""

    runtime: RuntimeOptionsT
    """Application-specific rollout and presentation defaults."""

    pipeline: Mapping[str, object]
    """Provider-owned pipeline construction options."""


@dataclass(frozen=True, slots=True)
class PresetCatalog(Generic[RuntimeOptionsT]):
    """Validated collection of named pipeline presets."""

    default_preset_id: str
    """Preset selected when a caller omits the preset identity."""

    presets: Mapping[str, PipelinePreset[RuntimeOptionsT]]
    """Preset definitions keyed by stable pipeline identity."""

    def resolve(
        self, preset_id: str | None
    ) -> tuple[str, PipelinePreset[RuntimeOptionsT]]:
        """Resolve an optional preset identity against this catalog."""
        selected = self.default_preset_id if preset_id is None else preset_id
        try:
            return selected, self.presets[selected]
        except KeyError as exc:
            available = ", ".join(sorted(self.presets))
            raise ValueError(
                f"Unknown pipeline preset {selected!r}. YAML presets: {available}."
            ) from exc


@runtime_checkable
class PipelineProvider(Protocol):
    """Construct a pipeline config from preset-owned options."""

    def create_pipeline_config(
        self,
        *,
        preset_id: str,
        options: Mapping[str, object],
    ) -> object:
        """Create a pipeline config for one resolved preset.

        Args:
            preset_id: Selected preset identity.
            options: Provider-owned options loaded from the preset YAML.

        Returns:
            Pipeline config for application-specific validation.
        """
        ...


class ObjectGraphPipelineProvider:
    """Materialize nested Python configs from a YAML object graph.

    A mapping containing ``_target`` imports and calls the named object with
    the remaining mapping entries as keyword arguments. ``_ref`` imports an
    object without calling it, and ``_tuple`` preserves tuple-valued config
    fields that YAML otherwise represents as lists.
    """

    def create_pipeline_config(
        self,
        *,
        preset_id: str,
        options: Mapping[str, object],
    ) -> object:
        """Materialize one pipeline config object graph."""
        return materialize_object_graph(
            options,
            path=f"presets.{preset_id}.pipeline",
        )


def load_pipeline_preset_catalog(
    path: str | Path,
    *,
    runtime_options_parser: RuntimeOptionsParser[RuntimeOptionsT],
) -> PresetCatalog[RuntimeOptionsT]:
    """Load and validate a pipeline-preset catalog from a YAML file.

    Args:
        path: YAML catalog path.
        runtime_options_parser: Application-specific runtime-options parser.

    Returns:
        Validated preset catalog.
    """
    source = Path(path)
    return parse_pipeline_preset_catalog(
        source.read_text(encoding="utf-8"),
        source_name=str(source),
        runtime_options_parser=runtime_options_parser,
    )


def parse_pipeline_preset_catalog(
    raw_text: str,
    *,
    source_name: str,
    runtime_options_parser: RuntimeOptionsParser[RuntimeOptionsT],
) -> PresetCatalog[RuntimeOptionsT]:
    """Parse and validate a pipeline-preset catalog from YAML text.

    Args:
        raw_text: YAML catalog contents.
        source_name: Source label included in validation errors.
        runtime_options_parser: Application-specific runtime-options parser.

    Returns:
        Validated preset catalog.
    """
    raw = yaml.safe_load(raw_text)
    root = _mapping(raw, path=source_name)
    _require_exact_fields(root, expected=_ROOT_FIELDS, path=source_name)
    if root["schema_version"] != SCHEMA_VERSION:
        raise ValueError(
            f"{source_name}.schema_version must be {SCHEMA_VERSION}, "
            f"got {root['schema_version']!r}."
        )

    default_preset_id = _nonempty_string(
        root["default_preset_id"], path=f"{source_name}.default_preset_id"
    )
    raw_presets = _mapping(root["presets"], path=f"{source_name}.presets")
    if not raw_presets:
        raise ValueError(f"{source_name}.presets must not be empty.")

    presets: dict[str, PipelinePreset[RuntimeOptionsT]] = {}
    for raw_name, raw_preset in raw_presets.items():
        name = _nonempty_string(raw_name, path=f"{source_name}.presets key")
        preset_path = f"{source_name}.presets.{name}"
        preset = _mapping(raw_preset, path=preset_path)
        _require_exact_fields(preset, expected=_PRESET_FIELDS, path=preset_path)
        provider = _nonempty_string(preset["provider"], path=f"{preset_path}.provider")
        runtime = runtime_options_parser(
            preset["runtime"],
            path=f"{preset_path}.runtime",
        )
        pipeline = _mapping(preset["pipeline"], path=f"{preset_path}.pipeline")
        presets[name] = PipelinePreset(
            provider=provider,
            runtime=runtime,
            pipeline=MappingProxyType(dict(pipeline)),
        )

    if default_preset_id not in presets:
        raise ValueError(
            f"{source_name}.default_preset_id {default_preset_id!r} is not "
            "defined under presets."
        )
    return PresetCatalog(
        default_preset_id=default_preset_id,
        presets=MappingProxyType(presets),
    )


def load_pipeline_provider(reference: str) -> PipelineProvider:
    """Load a pipeline-provider instance from ``module:attribute``."""
    candidate = resolve_reference(reference)
    provider = candidate() if isinstance(candidate, type) else candidate
    if not isinstance(provider, PipelineProvider):
        raise TypeError(
            f"Pipeline provider {reference!r} resolved to "
            f"{type(provider).__name__}, which does not implement "
            "create_pipeline_config()."
        )
    return provider


def resolve_reference(reference: str) -> object:
    """Resolve one ``module:attribute`` reference."""
    module_name, separator, attribute_path = reference.partition(":")
    if not separator or not module_name or not attribute_path:
        raise ValueError(
            f"Invalid Python reference {reference!r}; expected module:attribute."
        )
    value: object = import_module(module_name)
    for attribute in attribute_path.split("."):
        try:
            value = getattr(value, attribute)
        except AttributeError as exc:
            raise ValueError(
                f"Python reference {reference!r} has no attribute {attribute!r}."
            ) from exc
    return value


def materialize_object_graph(value: object, *, path: str = "root") -> object:
    """Materialize declarative ``_target``, ``_ref``, and ``_tuple`` nodes.

    Args:
        value: YAML-decoded object graph.
        path: Source path included in validation errors.

    Returns:
        Recursively materialized Python object.
    """
    if isinstance(value, list):
        return [
            materialize_object_graph(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    if not isinstance(value, Mapping):
        return value

    mapping = cast(dict[str, object], dict(value))
    if any(not isinstance(key, str) for key in mapping):
        raise TypeError(f"{path} keys must be strings.")
    reserved = {key for key in mapping if key.startswith("_")}
    if "_ref" in mapping:
        if reserved != {"_ref"} or len(mapping) != 1:
            raise ValueError(f"{path}._ref cannot be combined with other keys.")
        reference = mapping["_ref"]
        if not isinstance(reference, str):
            raise TypeError(f"{path}._ref must be a string.")
        return resolve_reference(reference)

    if "_tuple" in mapping:
        if reserved != {"_tuple"} or len(mapping) != 1:
            raise ValueError(f"{path}._tuple cannot be combined with other keys.")
        items = mapping["_tuple"]
        if not isinstance(items, list):
            raise TypeError(f"{path}._tuple must contain a YAML list.")
        return tuple(
            materialize_object_graph(item, path=f"{path}._tuple[{index}]")
            for index, item in enumerate(items)
        )

    if "_target" not in mapping:
        if reserved:
            names = ", ".join(sorted(reserved))
            raise ValueError(f"Unsupported reserved keys at {path}: {names}.")
        return {
            str(key): materialize_object_graph(item, path=f"{path}.{key}")
            for key, item in mapping.items()
        }

    if reserved != {"_target"}:
        names = ", ".join(sorted(reserved - {"_target"}))
        raise ValueError(f"Unsupported reserved keys at {path}: {names}.")
    reference = mapping["_target"]
    if not isinstance(reference, str):
        raise TypeError(f"{path}._target must be a string.")
    target = resolve_reference(reference)
    if not callable(target):
        raise TypeError(f"{path}._target {reference!r} is not callable.")
    target_callable = cast(Callable[..., object], target)
    kwargs = {
        str(key): materialize_object_graph(item, path=f"{path}.{key}")
        for key, item in mapping.items()
        if key != "_target"
    }
    try:
        return target_callable(**kwargs)
    except Exception as exc:
        raise ValueError(
            f"Failed to construct {path} with {reference!r}: {exc}"
        ) from exc


def _mapping(value: object, *, path: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{path} must be a mapping, got {type(value).__name__}.")
    mapping = cast(dict[object, object], dict(cast(Any, value)))
    if any(not isinstance(key, str) for key in mapping):
        raise TypeError(f"{path} keys must be strings.")
    return cast(dict[str, object], mapping)


def _require_exact_fields(
    value: Mapping[str, object], *, expected: set[str], path: str
) -> None:
    fields = {str(key) for key in value}
    missing = expected - fields
    unknown = fields - expected
    if missing or unknown:
        details: list[str] = []
        if missing:
            details.append(f"missing {sorted(missing)}")
        if unknown:
            details.append(f"unknown {sorted(unknown)}")
        raise ValueError(f"Invalid fields at {path}: {'; '.join(details)}.")


def _nonempty_string(value: object, *, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TypeError(f"{path} must be a non-empty string.")
    return value.strip()


__all__ = [
    "ObjectGraphPipelineProvider",
    "PipelinePreset",
    "PipelineProvider",
    "PresetCatalog",
    "RuntimeOptionsParser",
    "load_pipeline_preset_catalog",
    "load_pipeline_provider",
    "materialize_object_graph",
    "parse_pipeline_preset_catalog",
    "resolve_reference",
]
