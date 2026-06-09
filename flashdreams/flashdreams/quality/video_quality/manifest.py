# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Schema loader for video-quality regression case manifests."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

KNOWN_SUITES = frozenset(
    {"calibration", "per_commit", "nightly", "quarantine", "vlm_experimental"}
)
KNOWN_THRESHOLD_OPS = frozenset({">", ">=", "<", "<=", "==", "!="})
KNOWN_SEVERITIES = frozenset({"critical", "warning", "info"})


@dataclass(frozen=True)
class Threshold:
    """One metric threshold from a case manifest."""

    id: str
    metric: str
    op: str
    value: float | bool | int | str
    severity: str


@dataclass(frozen=True)
class Window:
    """A named time window in seconds."""

    start_s: float
    end_s: float


@dataclass(frozen=True)
class KnownBadClip:
    """A known-bad calibration clip and the threshold failures it should trigger."""

    id: str
    path: str
    expected_failures: tuple[str, ...]


@dataclass(frozen=True)
class CaseAssets:
    """Asset references for one video-quality case."""

    known_good: str | None
    generated: str | None
    ground_truth: str | None
    first_frame: str | None
    conditioning: str | None
    prompt: str | None
    known_bad: tuple[KnownBadClip, ...]
    sha256: dict[str, str]


@dataclass(frozen=True)
class VideoQualityCase:
    """One video-quality regression case."""

    id: str
    description: str
    suites: tuple[str, ...]
    assets: CaseAssets
    metrics: tuple[str, ...]
    thresholds: tuple[Threshold, ...]
    windows: dict[str, Window]
    source: dict[str, Any]
    generation: dict[str, Any]
    hf_dataset: str | None
    hf_revision: str | None

    def belongs_to_suite(self, suite: str) -> bool:
        """Return whether this case should run for ``suite``."""
        return suite in self.suites


@dataclass(frozen=True)
class VideoQualityManifest:
    """A parsed video-quality manifest."""

    path: Path
    schema_version: int
    suites: tuple[str, ...]
    default_hf_dataset: str | None
    cases: tuple[VideoQualityCase, ...]

    def select_cases(
        self, *, suite: str | None = None, case_id: str | None = None
    ) -> tuple[VideoQualityCase, ...]:
        """Return cases matching the requested suite and optional case id."""
        cases = self.cases
        if suite is not None:
            if suite not in self.suites:
                raise ValueError(
                    f"Unknown suite {suite!r}; manifest suites are {self.suites}"
                )
            cases = tuple(case for case in cases if case.belongs_to_suite(suite))
        if case_id is not None:
            cases = tuple(case for case in cases if case.id == case_id)
        if not cases:
            filters = []
            if suite is not None:
                filters.append(f"suite={suite!r}")
            if case_id is not None:
                filters.append(f"case_id={case_id!r}")
            raise ValueError(f"No video-quality cases matched {', '.join(filters)}")
        return cases


def load_manifest(path: Path | str) -> VideoQualityManifest:
    """Read and validate a video-quality case manifest."""
    manifest_path = Path(path)
    with manifest_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError(f"{manifest_path} must contain a YAML mapping")

    schema_version = _required_int(data, "schema_version", context="manifest")
    if schema_version != 1:
        raise ValueError(f"Unsupported schema_version {schema_version}; expected 1")

    suites = _required_str_tuple(data, "suites", context="manifest")
    unknown_suites = sorted(set(suites) - KNOWN_SUITES)
    if unknown_suites:
        raise ValueError(f"Unknown suite labels in manifest: {unknown_suites}")

    default_hf_dataset = _optional_str(data, "default_hf_dataset")

    raw_cases = data.get("cases")
    if not isinstance(raw_cases, list):
        raise ValueError("manifest.cases must be a list")

    cases = tuple(
        _parse_case(item, suites=suites, default_hf_dataset=default_hf_dataset, index=i)
        for i, item in enumerate(raw_cases)
    )
    case_ids = [case.id for case in cases]
    duplicate_ids = sorted(
        {case_id for case_id in case_ids if case_ids.count(case_id) > 1}
    )
    if duplicate_ids:
        raise ValueError(f"Duplicate case ids in manifest: {duplicate_ids}")

    return VideoQualityManifest(
        path=manifest_path,
        schema_version=schema_version,
        suites=suites,
        default_hf_dataset=default_hf_dataset,
        cases=cases,
    )


def _parse_case(
    value: Any,
    *,
    suites: tuple[str, ...],
    default_hf_dataset: str | None,
    index: int,
) -> VideoQualityCase:
    context = f"cases[{index}]"
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be a mapping")

    case_id = _required_str(value, "id", context=context)
    case_context = f"case {case_id!r}"
    case_suites = _required_str_tuple(value, "suites", context=case_context)
    unknown_suites = sorted(set(case_suites) - set(suites))
    if unknown_suites:
        raise ValueError(
            f"{case_context} uses suites not declared by manifest: {unknown_suites}"
        )

    raw_assets = value.get("assets")
    if not isinstance(raw_assets, dict):
        raise ValueError(f"{case_context}.assets must be a mapping")

    raw_windows = value.get("windows", {})
    if not isinstance(raw_windows, dict):
        raise ValueError(f"{case_context}.windows must be a mapping")
    windows = {
        name: _parse_window(name, item, context=case_context)
        for name, item in raw_windows.items()
    }

    raw_thresholds = value.get("thresholds")
    if not isinstance(raw_thresholds, list) or not raw_thresholds:
        raise ValueError(f"{case_context}.thresholds must be a non-empty list")
    thresholds = tuple(
        _parse_threshold(item, context=case_context) for item in raw_thresholds
    )
    threshold_ids = [threshold.id for threshold in thresholds]
    duplicate_thresholds = sorted(
        {
            threshold_id
            for threshold_id in threshold_ids
            if threshold_ids.count(threshold_id) > 1
        }
    )
    if duplicate_thresholds:
        raise ValueError(
            f"{case_context} has duplicate threshold ids: {duplicate_thresholds}"
        )

    source = _optional_mapping(value, "source")
    hf_dataset = _optional_str(value, "hf_dataset")
    if hf_dataset is None and source.get("type") != "synthetic":
        hf_dataset = default_hf_dataset

    return VideoQualityCase(
        id=case_id,
        description=_required_str(value, "description", context=case_context),
        suites=case_suites,
        assets=_parse_assets(raw_assets, context=case_context),
        metrics=_optional_str_tuple(value, "metrics", context=case_context),
        thresholds=thresholds,
        windows=windows,
        source=source,
        generation=_optional_mapping(value, "generation"),
        hf_dataset=hf_dataset,
        hf_revision=_optional_str(value, "hf_revision"),
    )


def _parse_assets(value: dict[str, Any], *, context: str) -> CaseAssets:
    known_bad_raw = value.get("known_bad", [])
    if not isinstance(known_bad_raw, list):
        raise ValueError(f"{context}.assets.known_bad must be a list")

    sha256_raw = value.get("sha256", {})
    if not isinstance(sha256_raw, dict) or not all(
        isinstance(k, str) and isinstance(v, str) for k, v in sha256_raw.items()
    ):
        raise ValueError(f"{context}.assets.sha256 must be a string mapping")

    return CaseAssets(
        known_good=_optional_str(value, "known_good"),
        generated=_optional_str(value, "generated"),
        ground_truth=_optional_str(value, "ground_truth"),
        first_frame=_optional_str(value, "first_frame"),
        conditioning=_optional_str(value, "conditioning"),
        prompt=_optional_str(value, "prompt"),
        known_bad=tuple(
            _parse_known_bad(item, context=f"{context}.assets.known_bad[{i}]")
            for i, item in enumerate(known_bad_raw)
        ),
        sha256=dict(sha256_raw),
    )


def _parse_known_bad(value: Any, *, context: str) -> KnownBadClip:
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be a mapping")
    return KnownBadClip(
        id=_required_str(value, "id", context=context),
        path=_required_str(value, "path", context=context),
        expected_failures=_optional_str_tuple(
            value, "expected_failures", context=context
        ),
    )


def _parse_threshold(value: Any, *, context: str) -> Threshold:
    if not isinstance(value, dict):
        raise ValueError(f"{context}.thresholds entries must be mappings")
    threshold_id = _required_str(value, "id", context=f"{context}.threshold")
    op = _required_str(value, "op", context=f"{context}.threshold {threshold_id!r}")
    if op not in KNOWN_THRESHOLD_OPS:
        raise ValueError(
            f"{context}.threshold {threshold_id!r} has unsupported op {op!r}"
        )
    severity = _required_str(
        value, "severity", context=f"{context}.threshold {threshold_id!r}"
    )
    if severity not in KNOWN_SEVERITIES:
        raise ValueError(
            f"{context}.threshold {threshold_id!r} has unsupported severity {severity!r}"
        )
    threshold_value = value.get("value")
    if not isinstance(threshold_value, (bool, int, float, str)):
        raise ValueError(f"{context}.threshold {threshold_id!r}.value must be scalar")
    return Threshold(
        id=threshold_id,
        metric=_required_str(
            value, "metric", context=f"{context}.threshold {threshold_id!r}"
        ),
        op=op,
        value=threshold_value,
        severity=severity,
    )


def _parse_window(name: str, value: Any, *, context: str) -> Window:
    if not isinstance(name, str) or not name:
        raise ValueError(f"{context}.windows contains an invalid window name {name!r}")
    if not isinstance(value, dict):
        raise ValueError(f"{context}.windows.{name} must be a mapping")
    start_s = _required_float(value, "start_s", context=f"{context}.windows.{name}")
    end_s = _required_float(value, "end_s", context=f"{context}.windows.{name}")
    if start_s < 0:
        raise ValueError(f"{context}.windows.{name}.start_s must be non-negative")
    if end_s <= start_s:
        raise ValueError(f"{context}.windows.{name}.end_s must be greater than start_s")
    return Window(start_s=start_s, end_s=end_s)


def _required_str(value: dict[str, Any], key: str, *, context: str) -> str:
    out = value.get(key)
    if not isinstance(out, str) or not out:
        raise ValueError(f"{context}.{key} must be a non-empty string")
    return out


def _optional_str(value: dict[str, Any], key: str) -> str | None:
    out = value.get(key)
    if out is None:
        return None
    if not isinstance(out, str) or not out:
        raise ValueError(f"{key} must be a non-empty string when set")
    return out


def _required_int(value: dict[str, Any], key: str, *, context: str) -> int:
    out = value.get(key)
    if not isinstance(out, int) or isinstance(out, bool):
        raise ValueError(f"{context}.{key} must be an integer")
    return out


def _required_float(value: dict[str, Any], key: str, *, context: str) -> float:
    out = value.get(key)
    if not isinstance(out, (int, float)):
        raise ValueError(f"{context}.{key} must be a number")
    return float(out)


def _required_str_tuple(
    value: dict[str, Any], key: str, *, context: str
) -> tuple[str, ...]:
    out = _optional_str_tuple(value, key, context=context)
    if not out:
        raise ValueError(f"{context}.{key} must be a non-empty string list")
    return out


def _optional_str_tuple(
    value: dict[str, Any], key: str, *, context: str
) -> tuple[str, ...]:
    out = value.get(key, [])
    if not isinstance(out, list) or not all(
        isinstance(item, str) and item for item in out
    ):
        raise ValueError(f"{context}.{key} must be a string list")
    return tuple(out)


def _optional_mapping(value: dict[str, Any], key: str) -> dict[str, Any]:
    out = value.get(key, {})
    if not isinstance(out, dict):
        raise ValueError(f"{key} must be a mapping when set")
    return dict(out)
