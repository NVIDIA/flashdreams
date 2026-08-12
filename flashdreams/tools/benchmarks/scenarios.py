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

"""Benchmark scenario definitions and JSON loading helpers."""

from __future__ import annotations

import json
import os
import re
import shlex
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

_SCENARIO_ID_RE = re.compile(r"^[a-zA-Z0-9_.-]+$")
_ENV_PLACEHOLDER_RE = re.compile(r"\{env:([A-Za-z_][A-Za-z0-9_]*)\}")
_CONTEXT_PLACEHOLDER_RE = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")
CompareRegion = Literal["full", "bottom-half"]
TimeoutStatus = Literal["fail", "pass"]


@dataclass(frozen=True, kw_only=True)
class QualityCommandConfig:
    """External quality metric command run after a scenario command completes."""

    id: str
    command: tuple[str, ...]
    metrics_path: str | None = None
    cwd: str | None = None
    env: Mapping[str, str] = field(default_factory=dict)
    timeout_s: float | None = None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "QualityCommandConfig":
        command_value = data.get("command", data.get("argv"))
        if command_value is None:
            raise ValueError("quality command needs a 'command' or 'argv' field")
        quality_id = _required_str(data, "id")
        _validate_id(quality_id, "quality command id")
        return cls(
            id=quality_id,
            command=_string_tuple(command_value, "quality command"),
            metrics_path=_optional_str(data.get("metrics_path"), "metrics_path"),
            cwd=_optional_str(data.get("cwd"), "cwd"),
            env=_str_mapping(data.get("env", {}), "env"),
            timeout_s=_optional_float(data.get("timeout_s"), "timeout_s"),
        )


@dataclass(frozen=True, kw_only=True)
class ReportGroupConfig:
    """HTML report grouping metadata for one scenario."""

    id: str
    name: str | None = None

    @classmethod
    def from_value(cls, value: object) -> "ReportGroupConfig | None":
        if value is None:
            return None
        if isinstance(value, str):
            _validate_id(value, "report_group")
            return cls(id=value)
        if isinstance(value, Mapping):
            data = {str(key): item for key, item in value.items()}
            group_id = _required_str(data, "id")
            _validate_id(group_id, "report_group id")
            return cls(
                id=group_id,
                name=_optional_str(data.get("name"), "report_group.name"),
            )
        raise ValueError("report_group must be a string or object when provided")

    def to_manifest(self) -> dict[str, str | None]:
        return {
            "id": self.id,
            "name": self.name,
        }


@dataclass(frozen=True, kw_only=True)
class BenchmarkScenario:
    """One command-backed local benchmark scenario.

    The harness treats existing runners and demos as opaque commands. It injects
    the scenario output directory when the command supports ``--output-dir`` and
    then normalizes logs, MP4s, and ``stats_*.json`` outputs.
    """

    id: str
    name: str
    command: tuple[str, ...]
    description: str = ""
    tags: tuple[str, ...] = ()
    report_group: ReportGroupConfig | None = None
    cwd: str | None = None
    env: Mapping[str, str] = field(default_factory=dict)
    timeout_s: float | None = None
    warmup_steps: int = 0
    output_dir_arg: str | None = "--output-dir"
    artifact_globs: tuple[str, ...] = ("*.mp4", "*.json", "*.pt", "*.png", "*.jpg")
    video_globs: tuple[str, ...] = ("*.mp4", "**/*.mp4")
    stats_globs: tuple[str, ...] = ("stats_*.json", "**/stats_*.json")
    quality_commands: tuple[QualityCommandConfig, ...] = ()
    quality_compare_region: CompareRegion | None = None
    quality_baseline_compare: bool = True
    timeout_status: TimeoutStatus = "fail"

    def __post_init__(self) -> None:
        _validate_id(self.id, "scenario id")
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps must be non-negative")
        if self.timeout_s is not None and self.timeout_s <= 0:
            raise ValueError("timeout_s must be positive when provided")
        if self.timeout_status not in ("fail", "pass"):
            raise ValueError("timeout_status must be 'fail' or 'pass'")
        if self.quality_compare_region not in (None, "full", "bottom-half"):
            raise ValueError("quality_compare_region must be 'full' or 'bottom-half'")
        if not isinstance(self.quality_baseline_compare, bool):
            raise ValueError("quality_baseline_compare must be a boolean")
        if not self.command:
            raise ValueError(f"scenario {self.id!r} needs a non-empty command")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "BenchmarkScenario":
        command_value = data.get("command", data.get("argv"))
        if command_value is None:
            raise ValueError("scenario needs a 'command' or 'argv' field")
        scenario_id = _required_str(data, "id")
        name = str(data.get("name") or scenario_id)
        quality_commands = tuple(
            QualityCommandConfig.from_dict(item)
            for item in _mapping_sequence(
                data.get("quality_commands", ()), "quality_commands"
            )
        )
        return cls(
            id=scenario_id,
            name=name,
            description=str(data.get("description") or ""),
            tags=_string_tuple(data.get("tags", ()), "tags"),
            report_group=ReportGroupConfig.from_value(data.get("report_group")),
            command=_string_tuple(command_value, "command"),
            cwd=_optional_str(data.get("cwd"), "cwd"),
            env=_str_mapping(data.get("env", {}), "env"),
            timeout_s=_optional_float(data.get("timeout_s"), "timeout_s"),
            warmup_steps=int(data.get("warmup_steps", 0)),
            output_dir_arg=_optional_str(
                data.get("output_dir_arg", "--output-dir"), "output_dir_arg"
            ),
            artifact_globs=_string_tuple(
                data.get(
                    "artifact_globs", ("*.mp4", "*.json", "*.pt", "*.png", "*.jpg")
                ),
                "artifact_globs",
            ),
            video_globs=_string_tuple(
                data.get("video_globs", ("*.mp4", "**/*.mp4")), "video_globs"
            ),
            stats_globs=_string_tuple(
                data.get("stats_globs", ("stats_*.json", "**/stats_*.json")),
                "stats_globs",
            ),
            quality_commands=quality_commands,
            quality_compare_region=_optional_compare_region(
                data.get("quality_compare_region")
            ),
            quality_baseline_compare=_bool_field(
                data.get("quality_baseline_compare", True),
                "quality_baseline_compare",
            ),
            timeout_status=_timeout_status(data.get("timeout_status", "fail")),
        )

    def rendered_command(
        self,
        *,
        context: Mapping[str, str],
        env: Mapping[str, str] | None = None,
    ) -> tuple[str, ...]:
        command = tuple(
            render_template(value, context=context, env=env) for value in self.command
        )
        if self.output_dir_arg is None:
            return command
        if _has_output_dir_arg(command, self.output_dir_arg):
            return command
        return (*command, self.output_dir_arg, context["output_dir"])

    def rendered_cwd(
        self,
        *,
        context: Mapping[str, str],
        env: Mapping[str, str] | None = None,
    ) -> str:
        if self.cwd is None:
            return context["repo_root"]
        return render_template(self.cwd, context=context, env=env)

    def rendered_env(
        self,
        *,
        context: Mapping[str, str],
        base_env: Mapping[str, str] | None = None,
    ) -> dict[str, str]:
        env = dict(base_env or os.environ)
        for key, value in self.env.items():
            env[str(key)] = render_template(value, context=context, env=env)
        return env

    def to_manifest(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "tags": list(self.tags),
            "report_group": (
                None if self.report_group is None else self.report_group.to_manifest()
            ),
            "command": list(self.command),
            "cwd": self.cwd,
            "env": dict(self.env),
            "timeout_s": self.timeout_s,
            "warmup_steps": self.warmup_steps,
            "output_dir_arg": self.output_dir_arg,
            "artifact_globs": list(self.artifact_globs),
            "video_globs": list(self.video_globs),
            "stats_globs": list(self.stats_globs),
            "quality_compare_region": self.quality_compare_region,
            "quality_baseline_compare": self.quality_baseline_compare,
            "quality_commands": [
                {
                    "id": command.id,
                    "command": list(command.command),
                    "metrics_path": command.metrics_path,
                    "cwd": command.cwd,
                    "env": dict(command.env),
                    "timeout_s": command.timeout_s,
                }
                for command in self.quality_commands
            ],
            "timeout_status": self.timeout_status,
        }


def built_in_scenarios() -> dict[str, BenchmarkScenario]:
    """Return public local-developer scenarios backed by ``flashdreams-run``."""
    scenarios = (
        BenchmarkScenario(
            id="self-forcing-taehv-smoke",
            name="Self-Forcing Wan2.1 TAEHV smoke",
            description=(
                "Short streaming T2V rollout through the existing "
                "self-forcing runner. Chunk 0 is treated as warmup."
            ),
            tags=("public", "runner", "streaming", "t2v"),
            command=(
                "flashdreams-run",
                "self-forcing-wan2.1-t2v-1.3b-taehv",
                "--total-blocks",
                "8",
            ),
            warmup_steps=1,
            timeout_s=7200,
        ),
        BenchmarkScenario(
            id="wan21-t2v-single",
            name="Wan2.1 T2V single-step",
            description="Single-step prompt-only Wan2.1 runner.",
            tags=("public", "runner", "t2v"),
            command=("flashdreams-run", "wan21-t2v-1.3b-480p"),
            warmup_steps=0,
            timeout_s=7200,
        ),
        BenchmarkScenario(
            id="cosmos2-t2v-single",
            name="Cosmos-Predict2 T2V single-step",
            description="Single-step prompt-only Cosmos-Predict2 runner.",
            tags=("public", "runner", "t2v"),
            command=("flashdreams-run", "cosmos2-t2v-2b-720p"),
            warmup_steps=0,
            timeout_s=7200,
        ),
        BenchmarkScenario(
            id="omnidreams-sv-example-smoke",
            name="Omnidreams single-view example smoke",
            description=(
                "Short single-view Omnidreams rollout using the runner's "
                "bundled example-data path."
            ),
            tags=("public", "runner", "world-model", "i2v"),
            command=(
                "flashdreams-run",
                "omnidreams",
                "--example-data",
                "--total-blocks",
                "8",
            ),
            warmup_steps=1,
            timeout_s=7200,
        ),
    )
    return {scenario.id: scenario for scenario in scenarios}


def load_scenario_file(path: Path) -> dict[str, BenchmarkScenario]:
    """Load custom scenarios from a JSON suite file."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        raw_scenarios = payload
    elif isinstance(payload, dict):
        raw_scenarios = payload.get("scenarios")
        if raw_scenarios is None:
            raise ValueError(f"{path} needs a top-level 'scenarios' list")
    else:
        raise ValueError(f"{path} must contain a JSON object or list")
    scenarios = [
        BenchmarkScenario.from_dict(item)
        for item in _mapping_sequence(raw_scenarios, "scenarios")
    ]
    duplicates = _duplicates(scenario.id for scenario in scenarios)
    if duplicates:
        raise ValueError(f"duplicate scenario ids in {path}: {sorted(duplicates)}")
    return {scenario.id: scenario for scenario in scenarios}


def render_template(
    value: str,
    *,
    context: Mapping[str, str],
    env: Mapping[str, str] | None = None,
) -> str:
    """Render harness placeholders in a command, cwd, or env value."""
    source_env = env or os.environ

    def replace_env(match: re.Match[str]) -> str:
        name = match.group(1)
        if name not in source_env:
            raise KeyError(f"environment variable {name!r} is required")
        return source_env[name]

    rendered = _ENV_PLACEHOLDER_RE.sub(replace_env, value)

    def replace_context(match: re.Match[str]) -> str:
        name = match.group(1)
        if name not in context:
            raise KeyError(f"unknown benchmark placeholder {name!r}")
        return context[name]

    return _CONTEXT_PLACEHOLDER_RE.sub(replace_context, rendered)


def _has_output_dir_arg(command: Sequence[str], output_dir_arg: str) -> bool:
    return any(
        arg == output_dir_arg or arg.startswith(f"{output_dir_arg}=") for arg in command
    )


def _string_tuple(value: object, field_name: str) -> tuple[str, ...]:
    if isinstance(value, str):
        return tuple(shlex.split(value))
    if not isinstance(value, Sequence):
        raise ValueError(f"{field_name} must be a string or sequence of strings")
    out = tuple(str(item) for item in value)
    if any(item == "" for item in out):
        raise ValueError(f"{field_name} must not contain empty strings")
    return out


def _required_str(data: Mapping[str, Any], field_name: str) -> str:
    value = data.get(field_name)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


def _optional_str(value: object, field_name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string when provided")
    return value


def _optional_float(value: object, field_name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be numeric when provided")
    return float(value)


def _optional_compare_region(value: object) -> CompareRegion | None:
    if value is None:
        return None
    if value == "full":
        return "full"
    if value == "bottom-half":
        return "bottom-half"
    raise ValueError(
        f"quality_compare_region must be 'full' or 'bottom-half', got {value!r}"
    )


def _timeout_status(value: object) -> TimeoutStatus:
    if value == "fail":
        return "fail"
    if value == "pass":
        return "pass"
    raise ValueError(f"timeout_status must be 'fail' or 'pass', got {value!r}")


def _bool_field(value: object, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field_name} must be a boolean")
    return value


def _str_mapping(value: object, field_name: str) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be an object")
    return {str(key): str(item) for key, item in value.items()}


def _mapping_sequence(value: object, field_name: str) -> tuple[Mapping[str, Any], ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{field_name} must be a list of objects")
    out: list[Mapping[str, Any]] = []
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            raise ValueError(f"{field_name}[{index}] must be an object")
        out.append({str(key): item_value for key, item_value in item.items()})
    return tuple(out)


def _validate_id(value: str, field_name: str) -> None:
    if not _SCENARIO_ID_RE.match(value):
        raise ValueError(
            f"{field_name} {value!r} may only contain letters, numbers, '.', '_', and '-'"
        )


def _duplicates(values: Sequence[str] | Any) -> set[str]:
    seen: set[str] = set()
    dupes: set[str] = set()
    for value in values:
        if value in seen:
            dupes.add(value)
        seen.add(value)
    return dupes
