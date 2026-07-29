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

"""Normalize benchmark stats files and log summaries into metric records."""

from __future__ import annotations

import csv
import json
import math
import re
import statistics
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeGuard

_PERF_SUMMARY_RE = re.compile(
    r"^\[perf\]\[(?P<label>[^\]]+) summary\]\s+"
    r"samples=(?P<samples>\d+)\s+warmup=(?P<warmup>\d+)\s+"
    r"window=(?P<window>\d+);(?P<body>.*)$"
)
_SUMMARY_METRIC_RE = re.compile(
    r"(?P<name>[A-Za-z0-9_]+)\s+"
    r"(?P<median>-?\d+(?:\.\d+)?)(?P<median_unit>ms|fps)?/"
    r"(?P<p90>-?\d+(?:\.\d+)?)(?P<p90_unit>ms|fps)?\s+med/p90"
)
_INTERACTIVE_DRIVE_SESSION_RE = re.compile(
    r"\[flashdreams-session\]\s+"
    r"(?P<phase>start|continue)"
    r"(?:\s+block_index=(?P<block_index>\d+))?"
    r"\s+total_ms=(?P<total_ms>-?\d+(?:\.\d+)?)"
)
_PROFILE_E2E_RE = re.compile(r"\[profile\]\s+e2e\s+(?P<body>.*)$")
_KEY_VALUE_NUMBER_RE = re.compile(r"(?P<key>[A-Za-z0-9_]+)=(?P<value>-?\d+(?:\.\d+)?)")

_INDEX_FIELDS = ("autoregressive_index", "step", "step_index", "chunk_index", "index")
_NON_METRIC_FIELDS = frozenset(
    {
        "autoregressive_index",
        "step",
        "step_index",
        "chunk_index",
        "index",
        "keys",
        "flush",
        "status",
        "id",
        "name",
        "mp4_path",
        "pt_path",
        "prompt",
        "start_image",
        "schedule",
    }
)
_KEY_OVERRIDES = {
    "total_ms_wo_finalize": "total_wo_finalize_s",
    "gen_ms": "generate_s",
    "model_ms": "model_step_s",
    "cache_ms": "cache_seed_prune_s",
    "copy_ms": "gpu_to_cpu_copy_s",
}


@dataclass(frozen=True, kw_only=True)
class MetricRecord:
    """One normalized metric row."""

    scenario_id: str
    record_type: str
    record_index: int | None = None
    source: str | None = None
    metrics: Mapping[str, float | int] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "record_type": self.record_type,
            "record_index": self.record_index,
            "source": self.source,
            "metrics": dict(self.metrics),
            "metadata": dict(self.metadata),
        }


def records_from_stats_file(
    path: Path,
    *,
    scenario_id: str,
    source_root: Path,
) -> list[MetricRecord]:
    """Read a runner stats JSON file and return normalized metric records."""
    payload = _read_json(path)
    if payload is None:
        return []
    source = _relpath(path, source_root)
    if isinstance(payload, list):
        return _records_from_rows(payload, scenario_id=scenario_id, source=source)
    if isinstance(payload, dict):
        records: list[MetricRecord] = []
        chunks = payload.get("chunks")
        if isinstance(chunks, list):
            records.extend(
                _records_from_rows(chunks, scenario_id=scenario_id, source=source)
            )
        summary_metrics = _normalize_metrics(payload)
        if summary_metrics:
            records.append(
                MetricRecord(
                    scenario_id=scenario_id,
                    record_type="summary",
                    source=source,
                    metrics=summary_metrics,
                    metadata=_metadata_fields(payload),
                )
            )
        return records
    return []


def records_from_log(
    path: Path,
    *,
    scenario_id: str,
    source_root: Path,
) -> list[MetricRecord]:
    """Parse structured perf-summary lines from a command log."""
    source = _relpath(path, source_root)
    records: list[MetricRecord] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        for parser in (
            _record_from_perf_summary_line,
            _record_from_interactive_drive_session_line,
            _record_from_profile_e2e_line,
        ):
            record = parser(line, scenario_id=scenario_id, source=source)
            if record is not None:
                records.append(record)
    return records


def record_from_quality_metrics(
    metrics_path: Path,
    *,
    scenario_id: str,
    quality_id: str,
    source_root: Path,
) -> MetricRecord | None:
    """Convert a quality command's metrics JSON into one record."""
    if not metrics_path.exists():
        return None
    payload = _read_json(metrics_path)
    if payload is None:
        return None
    metrics_payload = payload
    if isinstance(payload, Mapping):
        nested_metrics = payload.get("metrics")
        if isinstance(nested_metrics, Mapping):
            metrics_payload = nested_metrics
    metrics = flatten_numeric_metrics(metrics_payload)
    if not metrics:
        return None
    return MetricRecord(
        scenario_id=scenario_id,
        record_type="quality",
        source=_relpath(metrics_path, source_root),
        metrics=metrics,
        metadata={"quality_id": quality_id},
    )


def lifecycle_record(
    *,
    scenario_id: str,
    wall_time_s: float,
    returncode: int | None,
    status: str,
    command: str,
    timed_out: bool,
) -> MetricRecord:
    """Create a lifecycle metric row for the launched process."""
    metrics: dict[str, float | int] = {"command_wall_s": wall_time_s}
    if returncode is not None:
        metrics["returncode"] = returncode
    return MetricRecord(
        scenario_id=scenario_id,
        record_type="lifecycle",
        metrics=metrics,
        metadata={"status": status, "command": command, "timed_out": timed_out},
    )


def summarize_records(
    records: Iterable[MetricRecord],
    *,
    warmup_steps: int,
) -> dict[str, dict[str, float | int]]:
    """Compute compact stats by metric name, skipping warmup step records."""
    retained: list[MetricRecord] = []
    step_count = 0
    for record in records:
        if record.record_type == "step":
            if step_count < warmup_steps:
                step_count += 1
                continue
            step_count += 1
        retained.append(record)

    values_by_key: dict[str, list[float]] = {}
    for record in retained:
        for key, value in record.metrics.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            if not math.isfinite(float(value)):
                continue
            values_by_key.setdefault(key, []).append(float(value))

    return {
        key: _summary_stats(values)
        for key, values in sorted(values_by_key.items())
        if values
    }


def write_metrics_ndjson(records: Iterable[MetricRecord], path: Path) -> Path:
    """Write records as newline-delimited JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.to_dict(), sort_keys=True))
            handle.write("\n")
    return path


def write_metrics_csv(records: Iterable[MetricRecord], path: Path) -> Path:
    """Write a flattened CSV view of the metric records."""
    rows = [_flatten_record(record) for record in records]
    metric_keys = sorted(
        {key for row in rows for key in row if key.startswith("metric.")}
    )
    fieldnames = [
        "scenario_id",
        "record_type",
        "record_index",
        "source",
        *metric_keys,
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def flatten_numeric_metrics(
    payload: object,
    *,
    prefix: str = "",
) -> dict[str, float | int]:
    """Return a flattened numeric subset of a JSON-like object."""
    metrics: dict[str, float | int] = {}
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            nested_prefix = f"{prefix}.{key}" if prefix else str(key)
            metrics.update(flatten_numeric_metrics(value, prefix=nested_prefix))
    elif isinstance(payload, list):
        for index, value in enumerate(payload):
            nested_prefix = f"{prefix}.{index}" if prefix else str(index)
            metrics.update(flatten_numeric_metrics(value, prefix=nested_prefix))
    elif _is_number(payload):
        metrics[prefix] = payload
    return metrics


def _records_from_rows(
    rows: Sequence[object],
    *,
    scenario_id: str,
    source: str,
) -> list[MetricRecord]:
    records: list[MetricRecord] = []
    fallback_index = 0
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        metrics = _normalize_metrics(row)
        if not metrics:
            continue
        records.append(
            MetricRecord(
                scenario_id=scenario_id,
                record_type="step",
                record_index=_record_index(row, fallback_index),
                source=source,
                metrics=metrics,
                metadata=_metadata_fields(row),
            )
        )
        fallback_index += 1
    return records


def _record_from_perf_summary_line(
    line: str,
    *,
    scenario_id: str,
    source: str,
) -> MetricRecord | None:
    match = _PERF_SUMMARY_RE.match(line)
    if match is None:
        return None
    body = match.group("body")
    metrics: dict[str, float | int] = {}
    for metric_match in _SUMMARY_METRIC_RE.finditer(body):
        name = metric_match.group("name")
        median = float(metric_match.group("median"))
        p90 = float(metric_match.group("p90"))
        median_unit = metric_match.group("median_unit")
        p90_unit = metric_match.group("p90_unit")
        median_key, median_value = _summary_value_key(
            name, median, median_unit, "median"
        )
        p90_key, p90_value = _summary_value_key(name, p90, p90_unit, "p90")
        metrics[median_key] = median_value
        metrics[p90_key] = p90_value
    if not metrics:
        return None
    return MetricRecord(
        scenario_id=scenario_id,
        record_type="log_summary",
        source=source,
        metrics=metrics,
        metadata={
            "label": match.group("label"),
            "samples": int(match.group("samples")),
            "warmup": int(match.group("warmup")),
            "window": int(match.group("window")),
        },
    )


def _record_from_interactive_drive_session_line(
    line: str,
    *,
    scenario_id: str,
    source: str,
) -> MetricRecord | None:
    match = _INTERACTIVE_DRIVE_SESSION_RE.search(line)
    if match is None:
        return None
    block_index = match.group("block_index")
    record_index = 0 if block_index is None else int(block_index)
    return MetricRecord(
        scenario_id=scenario_id,
        record_type="step",
        record_index=record_index,
        source=source,
        metrics={"total_s": float(match.group("total_ms")) / 1000.0},
        metadata={"phase": match.group("phase")},
    )


def _record_from_profile_e2e_line(
    line: str,
    *,
    scenario_id: str,
    source: str,
) -> MetricRecord | None:
    match = _PROFILE_E2E_RE.search(line)
    if match is None:
        return None
    metrics: dict[str, float | int] = {}
    metadata: dict[str, object] = {"label": "e2e"}
    for item in _KEY_VALUE_NUMBER_RE.finditer(match.group("body")):
        key = item.group("key")
        value = float(item.group("value"))
        if key == "samples":
            metadata[key] = int(value)
            continue
        metric_key, metric_value = _normalize_metric(key, value)
        metrics[metric_key] = metric_value
    if not metrics:
        return None
    return MetricRecord(
        scenario_id=scenario_id,
        record_type="log_summary",
        source=source,
        metrics=metrics,
        metadata=metadata,
    )


def _summary_value_key(
    name: str, value: float, unit: str | None, statistic: str
) -> tuple[str, float]:
    if unit == "ms" or (unit is None and name.endswith("_s")):
        return f"{name}_{statistic}_s", value / 1000.0 if unit == "ms" else value
    if unit == "fps" or name.endswith("_fps"):
        return f"{name}_{statistic}_fps", value
    return f"{name}_{statistic}", value


def _normalize_metrics(row: Mapping[Any, object]) -> dict[str, float | int]:
    metrics: dict[str, float | int] = {}
    for key, value in row.items():
        key_str = str(key)
        if key_str in _NON_METRIC_FIELDS or not _is_number(value):
            continue
        metric_key, metric_value = _normalize_metric(key_str, value)
        metrics[metric_key] = metric_value
    return metrics


def _normalize_metric(key: str, value: float | int) -> tuple[str, float | int]:
    if key in _KEY_OVERRIDES:
        normalized_key = _KEY_OVERRIDES[key]
        if key.endswith("_ms") or "_ms" in key:
            return normalized_key, float(value) / 1000.0
        return normalized_key, value
    if key.endswith("_ms"):
        return f"{key[:-3]}_s", float(value) / 1000.0
    return key, value


def _metadata_fields(row: Mapping[Any, object]) -> dict[str, object]:
    return {
        str(key): value
        for key, value in row.items()
        if str(key) in _NON_METRIC_FIELDS and _jsonable_scalar(value)
    }


def _record_index(row: Mapping[Any, object], fallback: int) -> int:
    for field_name in _INDEX_FIELDS:
        value = row.get(field_name)
        if isinstance(value, bool):
            continue
        if isinstance(value, int):
            return value
        if isinstance(value, float) and value.is_integer():
            return int(value)
    return fallback


def _summary_stats(values: list[float]) -> dict[str, float | int]:
    sorted_values = sorted(values)
    return {
        "count": len(sorted_values),
        "min": min(sorted_values),
        "max": max(sorted_values),
        "mean": statistics.fmean(sorted_values),
        "median": statistics.median(sorted_values),
        "p90": _percentile_nearest_rank(sorted_values, 0.9),
    }


def _percentile_nearest_rank(sorted_values: list[float], percentile: float) -> float:
    if not sorted_values:
        raise ValueError("percentile needs at least one value")
    index = max(0, math.ceil(percentile * len(sorted_values)) - 1)
    return sorted_values[min(index, len(sorted_values) - 1)]


def _flatten_record(record: MetricRecord) -> dict[str, object]:
    row: dict[str, object] = {
        "scenario_id": record.scenario_id,
        "record_type": record.record_type,
        "record_index": "" if record.record_index is None else record.record_index,
        "source": record.source or "",
    }
    for key, value in record.metrics.items():
        row[f"metric.{key}"] = value
    return row


def _is_number(value: object) -> TypeGuard[float | int]:
    return not isinstance(value, bool) and isinstance(value, (int, float))


def _read_json(path: Path) -> object | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _jsonable_scalar(value: object) -> bool:
    return value is None or isinstance(value, (str, int, float, bool))


def _relpath(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()
