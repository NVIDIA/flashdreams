# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Baseline comparison helpers for OmniDreams evaluation summaries."""

from __future__ import annotations

import json
import operator
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

BASELINE_SCHEMA_VERSION = 1
CHECK_REPORT_SCHEMA_VERSION = 1

_OPERATORS = {
    "==": operator.eq,
    "!=": operator.ne,
    ">": operator.gt,
    ">=": operator.ge,
    "<": operator.lt,
    "<=": operator.le,
}
_NUMERIC_OPERATORS = {">", ">=", "<", "<="}


@dataclass(frozen=True)
class BaselineCheckResult:
    """Result for one expected-metric check."""

    name: str
    source: str
    severity: str
    passed: bool
    actual: Any = None
    expected: Any = None
    op: str | None = None
    min_allowed: Any = None
    max_allowed: Any = None
    tolerance: Any = None
    relative_tolerance: Any = None
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_json(path: Path) -> Any:
    """Load a JSON document from ``path``."""

    return json.loads(path.read_text(encoding="utf-8"))


def write_baseline_check_json(report: Mapping[str, Any], output: Path) -> None:
    """Write a baseline check report JSON artifact."""

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def check_summary_against_baseline(
    summary: Mapping[str, Any],
    baseline: Mapping[str, Any],
    *,
    summary_path: Path | None = None,
    baseline_path: Path | None = None,
) -> dict[str, Any]:
    """Compare an evaluation summary against a checked-in metric baseline."""

    checks = baseline.get("checks")
    if not isinstance(checks, list):
        raise ValueError("baseline must contain a list field named 'checks'")

    results = [_evaluate_check(summary, check) for check in checks]
    critical_failures = sum(
        1
        for result in results
        if not result.passed and result.severity.lower() == "critical"
    )
    warning_failures = sum(
        1
        for result in results
        if not result.passed and result.severity.lower() != "critical"
    )
    return {
        "kind": "omnidreams_eval_baseline_check",
        "schema_version": CHECK_REPORT_SCHEMA_VERSION,
        "baseline_id": baseline.get("baseline_id"),
        "baseline_path": str(baseline_path) if baseline_path is not None else None,
        "summary_path": str(summary_path) if summary_path is not None else None,
        "passed": critical_failures == 0,
        "critical_failures": critical_failures,
        "warning_failures": warning_failures,
        "check_count": len(results),
        "checks": [result.to_dict() for result in results],
    }


def format_baseline_check_report(report: Mapping[str, Any]) -> str:
    """Render a compact Markdown report for terminal output and logs."""

    status = "PASS" if report.get("passed") else "FAIL"
    lines: list[str] = [
        f"Baseline check: {status}",
    ]
    baseline_id = report.get("baseline_id")
    if baseline_id:
        lines.append(f"Baseline: `{baseline_id}`")
    lines.extend(
        [
            "",
            "| Status | Severity | Check | Actual | Expected |",
            "| --- | --- | --- | ---: | --- |",
        ]
    )
    for result in report.get("checks", []):
        check_passed = bool(result.get("passed"))
        severity = str(result.get("severity", "critical"))
        row_status = (
            "PASS"
            if check_passed
            else ("FAIL" if severity.lower() == "critical" else "WARN")
        )
        lines.append(
            "| "
            f"{row_status} | "
            f"{severity} | "
            f"`{result.get('name', '')}` | "
            f"{_format_value(result.get('actual'))} | "
            f"{_format_expectation(result)} |"
        )
    lines.append("")
    return "\n".join(lines)


def resolve_summary_path(summary: Mapping[str, Any], path: str) -> Any:
    """Resolve a baseline source path against a summary object.

    The path language is intentionally small:

    - ``a.b.c`` looks up nested mapping keys.
    - ``items[0]`` selects a list index.
    - ``items[key=value]`` selects the first mapping in a list with a matching key.
    - ``items[value]`` selects a mapping by common identity keys such as
      ``artifact``, ``split``, ``name``, ``id``, or ``metric``.
    """

    if not path:
        raise ValueError("summary path must not be empty")
    current: Any = summary
    for segment in _split_path(path):
        key, selectors = _parse_segment(segment)
        if key:
            current = _mapping_get(current, key)
        for selector in selectors:
            current = _apply_selector(current, selector)
    return current


def _evaluate_check(
    summary: Mapping[str, Any], raw_check: Mapping[str, Any]
) -> BaselineCheckResult:
    if not isinstance(raw_check, Mapping):
        raise ValueError("each baseline check must be an object")
    source = raw_check.get("source") or raw_check.get("path")
    if not isinstance(source, str) or not source:
        raise ValueError("each baseline check must define a non-empty 'source'")

    name = str(raw_check.get("name") or source)
    severity = str(raw_check.get("severity") or "critical")

    try:
        actual = resolve_summary_path(summary, source)
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        return BaselineCheckResult(
            name=name,
            source=source,
            severity=severity,
            passed=False,
            message=str(exc),
        )

    op = raw_check.get("op")
    expected = raw_check.get("expected", raw_check.get("value"))
    min_allowed = raw_check.get("min_allowed")
    max_allowed = raw_check.get("max_allowed")
    tolerance = raw_check.get("tolerance")
    relative_tolerance = raw_check.get("relative_tolerance")

    if op is not None:
        if not isinstance(op, str) or op not in _OPERATORS:
            raise ValueError(f"unsupported baseline operator for {name!r}: {op!r}")
        if "expected" not in raw_check and "value" not in raw_check:
            raise ValueError(f"baseline check {name!r} with op {op!r} needs expected")
        passed, message = _compare_operator(actual, expected, op)
    elif min_allowed is not None or max_allowed is not None:
        passed, message = _compare_range(actual, min_allowed, max_allowed)
    elif tolerance is not None or relative_tolerance is not None:
        if "expected" not in raw_check and "value" not in raw_check:
            raise ValueError(f"baseline check {name!r} with tolerance needs expected")
        passed, message = _compare_tolerance(
            actual,
            expected,
            tolerance=tolerance,
            relative_tolerance=relative_tolerance,
        )
    elif "expected" in raw_check or "value" in raw_check:
        passed = actual == expected
        message = "" if passed else f"expected {expected!r}, got {actual!r}"
    else:
        raise ValueError(
            f"baseline check {name!r} must define op, range, tolerance, or expected"
        )

    return BaselineCheckResult(
        name=name,
        source=source,
        severity=severity,
        passed=passed,
        actual=actual,
        expected=expected,
        op=op,
        min_allowed=min_allowed,
        max_allowed=max_allowed,
        tolerance=tolerance,
        relative_tolerance=relative_tolerance,
        message=message,
    )


def _compare_operator(actual: Any, expected: Any, op: str) -> tuple[bool, str]:
    if op in _NUMERIC_OPERATORS and (
        not _is_number(actual) or not _is_number(expected)
    ):
        return False, f"operator {op!r} requires numeric actual and expected"
    passed = bool(_OPERATORS[op](actual, expected))
    message = "" if passed else f"expected actual {op} {expected!r}, got {actual!r}"
    return passed, message


def _compare_range(actual: Any, min_allowed: Any, max_allowed: Any) -> tuple[bool, str]:
    if not _is_number(actual):
        return False, "range check requires numeric actual value"
    if min_allowed is not None:
        if not _is_number(min_allowed):
            return False, "min_allowed must be numeric"
        if actual < min_allowed:
            return False, f"actual {actual!r} is below min_allowed {min_allowed!r}"
    if max_allowed is not None:
        if not _is_number(max_allowed):
            return False, "max_allowed must be numeric"
        if actual > max_allowed:
            return False, f"actual {actual!r} is above max_allowed {max_allowed!r}"
    return True, ""


def _compare_tolerance(
    actual: Any,
    expected: Any,
    *,
    tolerance: Any,
    relative_tolerance: Any,
) -> tuple[bool, str]:
    if not _is_number(actual) or not _is_number(expected):
        return False, "tolerance check requires numeric actual and expected"
    allowed = 0.0
    if tolerance is not None:
        if not _is_number(tolerance):
            return False, "tolerance must be numeric"
        allowed = max(allowed, float(tolerance))
    if relative_tolerance is not None:
        if not _is_number(relative_tolerance):
            return False, "relative_tolerance must be numeric"
        allowed = max(allowed, abs(float(expected)) * float(relative_tolerance))
    delta = abs(float(actual) - float(expected))
    passed = delta <= allowed
    message = (
        ""
        if passed
        else (
            f"actual {actual!r} differs from expected {expected!r} by {delta:g}, "
            f"allowed {allowed:g}"
        )
    )
    return passed, message


def _split_path(path: str) -> list[str]:
    out: list[str] = []
    start = 0
    bracket_depth = 0
    for index, char in enumerate(path):
        if char == "[":
            bracket_depth += 1
        elif char == "]":
            bracket_depth -= 1
            if bracket_depth < 0:
                raise ValueError(f"unbalanced selector in path: {path}")
        elif char == "." and bracket_depth == 0:
            out.append(path[start:index])
            start = index + 1
    if bracket_depth:
        raise ValueError(f"unbalanced selector in path: {path}")
    out.append(path[start:])
    if any(not segment for segment in out):
        raise ValueError(f"empty path segment in: {path}")
    return out


def _parse_segment(segment: str) -> tuple[str, list[str]]:
    if "[" not in segment:
        return segment, []
    key = segment[: segment.index("[")]
    selectors: list[str] = []
    rest = segment[len(key) :]
    while rest:
        if not rest.startswith("["):
            raise ValueError(f"invalid selector segment: {segment}")
        close = rest.find("]")
        if close < 0:
            raise ValueError(f"unclosed selector segment: {segment}")
        selectors.append(rest[1:close])
        rest = rest[close + 1 :]
    return key, selectors


def _mapping_get(value: Any, key: str) -> Any:
    if not isinstance(value, Mapping):
        raise TypeError(
            f"cannot look up key {key!r} on non-object {type(value).__name__}"
        )
    if key not in value:
        raise KeyError(f"summary path key not found: {key}")
    return value[key]


def _apply_selector(value: Any, selector: str) -> Any:
    selector = selector.strip()
    if not selector:
        raise ValueError("empty selector")
    if isinstance(value, Mapping):
        key = _strip_quotes(selector)
        if key not in value:
            raise KeyError(f"summary path key not found: {key}")
        return value[key]
    if not _is_sequence(value):
        raise TypeError(f"cannot apply selector [{selector}] to {type(value).__name__}")
    if selector.lstrip("-").isdigit():
        return value[int(selector)]
    if "=" in selector:
        key, expected = selector.split("=", 1)
        return _find_mapping_by_key(value, key.strip(), _strip_quotes(expected.strip()))
    return _find_mapping_by_identity(value, _strip_quotes(selector))


def _find_mapping_by_key(items: Sequence[Any], key: str, expected: str) -> Any:
    for item in items:
        if isinstance(item, Mapping) and str(item.get(key)) == expected:
            return item
    raise KeyError(f"no list item has {key}={expected!r}")


def _find_mapping_by_identity(items: Sequence[Any], expected: str) -> Any:
    for key in ("name", "id", "split", "artifact", "metric"):
        try:
            return _find_mapping_by_key(items, key, expected)
        except KeyError:
            continue
    raise KeyError(f"no list item matched selector {expected!r}")


def _strip_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def _is_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    )


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _format_expectation(result: Mapping[str, Any]) -> str:
    min_allowed = result.get("min_allowed")
    max_allowed = result.get("max_allowed")
    if min_allowed is not None or max_allowed is not None:
        if min_allowed is not None and max_allowed is not None:
            return f"{_format_value(min_allowed)}..{_format_value(max_allowed)}"
        if min_allowed is not None:
            return f">= {_format_value(min_allowed)}"
        return f"<= {_format_value(max_allowed)}"
    op = result.get("op")
    if op:
        return f"{op} {_format_value(result.get('expected'))}"
    tolerance = result.get("tolerance")
    relative_tolerance = result.get("relative_tolerance")
    if tolerance is not None or relative_tolerance is not None:
        parts = [f"target {_format_value(result.get('expected'))}"]
        if tolerance is not None:
            parts.append(f"abs {_format_value(tolerance)}")
        if relative_tolerance is not None:
            parts.append(f"rel {_format_value(relative_tolerance)}")
        return ", ".join(parts)
    return _format_value(result.get("expected"))


def _format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    if value is None:
        return ""
    return str(value)
