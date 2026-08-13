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

"""Text-to-video runtime options for shared pipeline-preset catalogs."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Any, cast

from flashdreams.core.pipeline_presets import (
    PipelinePreset,
    PresetCatalog,
    load_pipeline_preset_catalog,
    parse_pipeline_preset_catalog,
)
from flashdreams.infra.postprocess import VideoTensorLayout

_REQUIRED_RUNTIME_FIELDS = {
    "prompt",
    "pixel_height",
    "pixel_width",
    "fps",
    "output_layout",
}
_OPTIONAL_RUNTIME_FIELDS = {"total_blocks"}
_VIDEO_LAYOUTS = {"tchw", "btchw", "bcthw", "bvtchw"}


@dataclass(frozen=True, slots=True)
class RuntimePresetOptions:
    """Host-facing rollout and presentation options for one T2V preset."""

    prompt: str
    """Default text prompt."""

    pixel_height: int
    """Output video height in pixels."""

    pixel_width: int
    """Output video width in pixels."""

    fps: int
    """Presentation frame rate."""

    output_layout: VideoTensorLayout
    """Decoded tensor layout exposed to the host."""

    total_blocks: int | None = None
    """Default finite-session length; ``None`` requires an MP4 CLI override."""


def load_preset_catalog(
    path: str | Path | None = None,
) -> PresetCatalog[RuntimePresetOptions]:
    """Load the packaged or caller-supplied T2V pipeline-preset catalog.

    Args:
        path: YAML path; ``None`` loads the catalog packaged with ``t2v_app``.

    Returns:
        Validated preset catalog.
    """
    if path is not None:
        return load_pipeline_preset_catalog(
            path,
            runtime_options_parser=_load_runtime_options,
        )

    source = files("t2v_app").joinpath("pipeline_presets.yaml")
    return parse_pipeline_preset_catalog(
        source.read_text(encoding="utf-8"),
        source_name=str(source),
        runtime_options_parser=_load_runtime_options,
    )


def _load_runtime_options(value: object, *, path: str) -> RuntimePresetOptions:
    runtime = _mapping(value, path=path)
    _require_fields(
        runtime,
        required=_REQUIRED_RUNTIME_FIELDS,
        optional=_OPTIONAL_RUNTIME_FIELDS,
        path=path,
    )
    layout = _nonempty_string(runtime["output_layout"], path=f"{path}.output_layout")
    if layout not in _VIDEO_LAYOUTS:
        allowed = ", ".join(sorted(_VIDEO_LAYOUTS))
        raise ValueError(
            f"{path}.output_layout must be one of {allowed}, got {layout!r}."
        )
    return RuntimePresetOptions(
        prompt=_nonempty_string(runtime["prompt"], path=f"{path}.prompt"),
        pixel_height=_positive_int(
            runtime["pixel_height"], path=f"{path}.pixel_height"
        ),
        pixel_width=_positive_int(runtime["pixel_width"], path=f"{path}.pixel_width"),
        fps=_positive_int(runtime["fps"], path=f"{path}.fps"),
        output_layout=cast(VideoTensorLayout, layout),
        total_blocks=(
            None
            if runtime.get("total_blocks") is None
            else _positive_int(runtime["total_blocks"], path=f"{path}.total_blocks")
        ),
    )


def _mapping(value: object, *, path: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{path} must be a mapping, got {type(value).__name__}.")
    mapping = cast(dict[object, object], dict(cast(Any, value)))
    if any(not isinstance(key, str) for key in mapping):
        raise TypeError(f"{path} keys must be strings.")
    return cast(dict[str, object], mapping)


def _require_fields(
    value: Mapping[str, object],
    *,
    required: set[str],
    optional: set[str],
    path: str,
) -> None:
    fields = {str(key) for key in value}
    missing = required - fields
    unknown = fields - required - optional
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


def _positive_int(value: object, *, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{path} must be an integer.")
    if value <= 0:
        raise ValueError(f"{path} must be > 0.")
    return value


__all__ = [
    "PipelinePreset",
    "PresetCatalog",
    "RuntimePresetOptions",
    "load_preset_catalog",
]
