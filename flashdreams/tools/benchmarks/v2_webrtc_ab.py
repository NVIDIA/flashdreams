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

"""Fresh-process WebRTC A/B benchmark for v2 applications.

The benchmark is deliberately a developer tool rather than part of
``runtime_v2``. Applications are resolved by their entry-point slug and every
model-specific setting is supplied by JSON. A small loopback aiortc peer
exercises server-side frame preparation, WebRTC encoding/decoding, and receiver
materialization without requiring a browser window.

Run a configured benchmark with::

    python -m tools.benchmarks.v2_webrtc_ab \
        --config configs/v2_webrtc_benchmarks.json \
        --benchmark cam2v-lingbot-hud-ab \
        --output-dir artifacts/benchmarks/cam2v-lingbot-hud-ab

Each case runs in a fresh worker process. The worker opens WebRTC and waits for
the receiver data channel and video track before allowing model generation to
start. At shutdown it gives the receiver time to consume every frame that the
sender committed to the encoder. Frames evicted from the bounded unsent queue
before that handoff remain visible in the sender-drop metrics.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import subprocess
import sys
import threading
import time
import traceback
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from itertools import pairwise
from pathlib import Path
from typing import Any

from numpy import uint64

from flashdreams.api_v2.client_window import IClientWindow
from flashdreams.runtime_v2.session_desc import SessionDesc
from flashdreams.runtime_v2.step_result import StepResult
from flashdreams.runtime_v2.user_input_event import CloseUserInputEvent
from flashdreams.runtime_v2.user_input_events import UserInputEvents

_SCHEMA_VERSION = 1
_RUN_ARTIFACT_TYPE = "flashdreams.tools.benchmarks.v2_webrtc_run"
_SUMMARY_ARTIFACT_TYPE = "flashdreams.tools.benchmarks.v2_webrtc_ab"


@dataclass(frozen=True, slots=True)
class BenchmarkDefinition:
    """Validated configuration for one A/B benchmark."""

    id: str
    application: str
    common_application_args: tuple[str, ...]
    variants: Mapping[str, tuple[str, ...]]
    run_order: tuple[str, ...]
    baseline: str
    candidate: str
    warmup_steps: int
    measured_steps: int
    receiver_ready_timeout_s: float
    receiver_drain_timeout_s: float
    worker_timeout_s: float
    session_overrides: Mapping[str, Any]
    metadata: Mapping[str, Any]
    acceptance: Mapping[str, float]


@dataclass(frozen=True, slots=True)
class RunRequest:
    """One fresh-process run derived from a benchmark definition."""

    run_id: str
    label: str
    application: str
    application_args: tuple[str, ...]
    warmup_steps: int
    measured_steps: int
    receiver_ready_timeout_s: float
    receiver_drain_timeout_s: float
    session_overrides: Mapping[str, Any]
    metadata: Mapping[str, Any]
    repo_root: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable worker request."""
        return {
            "run_id": self.run_id,
            "label": self.label,
            "application": self.application,
            "application_args": list(self.application_args),
            "warmup_steps": self.warmup_steps,
            "measured_steps": self.measured_steps,
            "receiver_ready_timeout_s": self.receiver_ready_timeout_s,
            "receiver_drain_timeout_s": self.receiver_drain_timeout_s,
            "session_overrides": dict(self.session_overrides),
            "metadata": dict(self.metadata),
            "repo_root": self.repo_root,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> RunRequest:
        """Validate and construct a worker request."""
        return cls(
            run_id=_required_string(payload, "run_id"),
            label=_required_string(payload, "label"),
            application=_required_string(payload, "application"),
            application_args=_string_tuple(
                payload.get("application_args", ()), "application_args"
            ),
            warmup_steps=_nonnegative_int(payload, "warmup_steps"),
            measured_steps=_positive_int(payload, "measured_steps"),
            receiver_ready_timeout_s=_positive_float(
                payload, "receiver_ready_timeout_s"
            ),
            receiver_drain_timeout_s=_positive_float(
                payload, "receiver_drain_timeout_s"
            ),
            session_overrides=_mapping(
                payload.get("session_overrides", {}), "session_overrides"
            ),
            metadata=_mapping(payload.get("metadata", {}), "metadata"),
            repo_root=_required_string(payload, "repo_root"),
        )


def load_benchmark_definition(path: Path, benchmark_id: str) -> BenchmarkDefinition:
    """Load one benchmark definition from a JSON config file."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise TypeError(f"{path} must contain a JSON object")
    raw_benchmarks = payload.get("benchmarks")
    if not isinstance(raw_benchmarks, Sequence) or isinstance(
        raw_benchmarks, str | bytes
    ):
        raise TypeError(f"{path} needs a top-level 'benchmarks' list")
    matches = [
        item
        for item in raw_benchmarks
        if isinstance(item, Mapping) and item.get("id") == benchmark_id
    ]
    if len(matches) != 1:
        raise ValueError(
            f"benchmark {benchmark_id!r} must occur exactly once in {path}; "
            f"found {len(matches)}"
        )
    return _definition_from_dict(matches[0])


def build_run_matrix(
    definition: BenchmarkDefinition, *, repo_root: Path
) -> tuple[RunRequest, ...]:
    """Expand the stable run order into uniquely named worker requests."""
    counts: Counter[str] = Counter()
    requests: list[RunRequest] = []
    for label in definition.run_order:
        counts[label] += 1
        requests.append(
            RunRequest(
                run_id=f"{label}_{counts[label]}",
                label=label,
                application=definition.application,
                application_args=(
                    *definition.common_application_args,
                    *definition.variants[label],
                ),
                warmup_steps=definition.warmup_steps,
                measured_steps=definition.measured_steps,
                receiver_ready_timeout_s=definition.receiver_ready_timeout_s,
                receiver_drain_timeout_s=definition.receiver_drain_timeout_s,
                session_overrides=definition.session_overrides,
                metadata=definition.metadata,
                repo_root=str(repo_root.resolve()),
            )
        )
    return tuple(requests)


def percentile(values: Sequence[float], fraction: float) -> float | None:
    """Return a linearly interpolated percentile, or ``None`` when empty."""
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("fraction must be between 0 and 1")
    finite = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not finite:
        return None
    position = (len(finite) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return finite[lower]
    weight = position - lower
    return finite[lower] * (1.0 - weight) + finite[upper] * weight


def distribution(values: Sequence[float]) -> dict[str, float | int | None]:
    """Return count, median, p90, minimum, and maximum for finite values."""
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return {
        "count": len(finite),
        "median": statistics.median(finite) if finite else None,
        "p90": percentile(finite, 0.9),
        "min": min(finite) if finite else None,
        "max": max(finite) if finite else None,
    }


def summarize_run(raw: Mapping[str, Any], *, warmup_steps: int) -> dict[str, Any]:
    """Summarize one raw worker artifact with warmup excluded."""
    model_results = _mapping_list(raw.get("model_results"))
    window_writes = _mapping_list(raw.get("window_writes"))
    receiver_frames = _mapping_list(raw.get("receiver_frames"))
    model_step_indices = [
        index
        for item in model_results
        if (index := _optional_int(item.get("step_index"))) is not None
    ]
    model_step_indices_contiguous = model_step_indices == list(
        range(len(model_results))
    )
    generated_frames = sum(int(item.get("frame_count", 0)) for item in model_results)
    steady_model = [
        item
        for item in model_results
        if _optional_int(item.get("step_index")) is not None
        and int(item["step_index"]) >= warmup_steps
    ]
    warmup_frames = sum(
        int(item.get("frame_count", 0))
        for item in model_results
        if _optional_int(item.get("step_index")) is not None
        and int(item["step_index"]) < warmup_steps
    )
    steady_writes = _skip_frame_prefix(window_writes, warmup_frames)
    submitted_frames = sum(int(item.get("frame_count", 0)) for item in window_writes)
    sender_metrics = _mapping(
        raw.get("webrtc_sender_metrics", {}), "webrtc_sender_metrics"
    )
    sender_enqueued_frames = _optional_int(
        sender_metrics.get("webrtc_sender_enqueued_count")
    )
    sender_handed_off_frames = _optional_int(
        sender_metrics.get("webrtc_sender_handed_off_count")
    )
    sender_dropped_frames = _optional_int(
        sender_metrics.get("webrtc_sender_dropped_for_lag_count")
    )
    expected_sender_enqueued_frames = submitted_frames
    expected_receiver_frames = (
        submitted_frames
        if sender_handed_off_frames is None
        else sender_handed_off_frames
    )
    # WebRTC receive timestamps necessarily lag write timestamps. Excluding
    # warmup by a write-time cutoff therefore admits trailing warmup frames.
    # The benchmark's unconditional zero-drop integrity gate makes ordinal
    # exclusion exact for accepted runs.
    steady_receiver = receiver_frames[warmup_frames:]

    model_metrics = _metric_distributions(steady_model)
    write_metrics = _metric_distributions(steady_writes)
    model_fps = _model_step_fps(steady_model)
    decoded_fps = _completion_fps(steady_receiver, "received_at_s", frame_count=1)
    receiver_times = _numeric_values(steady_receiver, "received_at_s")
    receiver_intervals = [right - left for left, right in pairwise(receiver_times)]
    trailing_receiver_fps = _trailing_rates(receiver_times, window_seconds=2.0)
    received_pts = [
        int(item["pts"])
        for item in receiver_frames
        if _optional_int(item.get("pts")) is not None
    ]
    received_frames = len(receiver_frames)

    return {
        "run_id": raw.get("run_id"),
        "label": raw.get("label"),
        "status": raw.get("status", "fail"),
        "exception": raw.get("exception"),
        "wall_time_s": raw.get("wall_time_s"),
        "model_step_count": len(model_results),
        "model_step_indices_contiguous": model_step_indices_contiguous,
        "steady_model_step_count": len(steady_model),
        "generated_frame_count": generated_frames,
        "warmup_frame_count": warmup_frames,
        "submitted_frame_count": submitted_frames,
        "expected_sender_enqueued_frame_count": expected_sender_enqueued_frames,
        "sender_enqueued_frame_count": sender_enqueued_frames,
        "sender_handed_off_frame_count": sender_handed_off_frames,
        "sender_dropped_for_lag_count": sender_dropped_frames,
        "sender_discarded_on_close_count": _optional_int(
            sender_metrics.get("webrtc_sender_discarded_on_close_count")
        ),
        "sender_materialized_frame_count": _optional_int(
            sender_metrics.get("webrtc_sender_materialized_count")
        ),
        "expected_receiver_frame_count": expected_receiver_frames,
        "received_frame_count": received_frames,
        "missing_receiver_frame_count": max(
            0, expected_receiver_frames - received_frames
        ),
        "extra_receiver_frame_count": max(
            0, received_frames - expected_receiver_frames
        ),
        "receiver_drain_complete": bool(raw.get("receiver_drain_complete", False)),
        "receiver_pts_strictly_increasing": len(received_pts) == received_frames
        and all(right > left for left, right in pairwise(received_pts)),
        "steady_model_fps": model_fps,
        "decoded_steady_fps": decoded_fps,
        "decoded_interarrival_s": distribution(receiver_intervals),
        "decoded_trailing_2s_fps": distribution(trailing_receiver_fps),
        "receiver_decode_s": distribution(_numeric_values(steady_receiver, "decode_s")),
        "window_write_s": distribution(_numeric_values(steady_writes, "write_wall_s")),
        "model_metrics": model_metrics,
        "window_metrics": write_metrics,
        "raw_artifact": raw.get("raw_artifact"),
        "log_path": raw.get("log_path"),
    }


def evaluate_acceptance(
    runs: Sequence[Mapping[str, Any]], definition: BenchmarkDefinition
) -> dict[str, Any]:
    """Validate every run, then compare nth baseline/candidate pairs."""
    expected_model_steps = definition.warmup_steps + definition.measured_steps
    run_integrity: list[dict[str, Any]] = []
    for run in runs:
        criteria = _run_integrity_criteria(
            run,
            expected_model_steps=expected_model_steps,
            expected_measured_steps=definition.measured_steps,
        )
        run_integrity.append(
            {
                "run_id": run.get("run_id"),
                "label": run.get("label"),
                "status": run.get("status"),
                "criteria": criteria,
                "passed": run.get("status") == "pass"
                and all(item["passed"] for item in criteria.values()),
            }
        )

    baselines = [item for item in runs if item.get("label") == definition.baseline]
    candidates = [item for item in runs if item.get("label") == definition.candidate]
    pair_count = min(len(baselines), len(candidates))
    pairs: list[dict[str, Any]] = []
    for index in range(pair_count):
        baseline = baselines[index]
        candidate = candidates[index]
        criteria = _pair_criteria(
            baseline,
            candidate,
            thresholds=definition.acceptance,
        )
        pairs.append(
            {
                "pair_index": index,
                "baseline_run_id": baseline.get("run_id"),
                "candidate_run_id": candidate.get("run_id"),
                "criteria": criteria,
                "passed": bool(criteria)
                and all(item["passed"] for item in criteria.values()),
            }
        )
    expected_pairs = definition.run_order.count(definition.baseline)
    all_runs_passed = bool(run_integrity) and all(
        item["passed"] for item in run_integrity
    )
    return {
        "baseline": definition.baseline,
        "candidate": definition.candidate,
        "run_integrity": run_integrity,
        "pair_count": pair_count,
        "expected_pair_count": expected_pairs,
        "pairs": pairs,
        "passed": (
            all_runs_passed
            and pair_count == expected_pairs
            and bool(pairs)
            and all(pair["passed"] for pair in pairs)
        ),
    }


def render_markdown(summary: Mapping[str, Any]) -> str:
    """Render a compact human-readable A/B report."""
    protocol = _mapping(summary.get("protocol", {}), "protocol")
    runs = _mapping_list(summary.get("runs"))
    acceptance = _mapping(summary.get("acceptance", {}), "acceptance")
    lines = [
        f"# WebRTC A/B benchmark: {summary.get('benchmark_id', 'unknown')}",
        "",
        (
            "This is a loopback server-to-aiortc-decoder measurement. It does "
            "not measure browser DOM or display compositing."
        ),
        "",
        f"- application: `{protocol.get('application', 'unknown')}`",
        f"- warmup model steps: `{protocol.get('warmup_steps', 'unknown')}`",
        f"- measured model steps: `{protocol.get('measured_steps', 'unknown')}`",
        f"- run order: `{', '.join(str(item) for item in protocol.get('run_order', []))}`",
        "",
        (
            "| run | status | model FPS | decoded FPS | gap p90 | gap max | "
            "publish p90 | model-step p90 | sender drops | missing decoder frames |"
        ),
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for run in runs:
        model_metrics = _mapping(run.get("model_metrics", {}), "model_metrics")
        publish = _mapping(
            model_metrics.get("runtime_presentation_publish_wait_s", {}),
            "runtime_presentation_publish_wait_s",
        )
        model_step = _mapping(
            model_metrics.get("model_step_wall_s", {}), "model_step_wall_s"
        )
        decoded_interarrival = _mapping(
            run.get("decoded_interarrival_s", {}), "decoded_interarrival_s"
        )
        lines.append(
            "| {run_id} | {status} | {model_fps} | {decoded_fps} | {gap_p90} | "
            "{gap_max} | {publish} | {model_step} | {sender_drops} | {missing} |".format(
                run_id=run.get("run_id", "unknown"),
                status=run.get("status", "unknown"),
                model_fps=_format_number(run.get("steady_model_fps")),
                decoded_fps=_format_number(run.get("decoded_steady_fps")),
                gap_p90=_format_seconds(decoded_interarrival.get("p90")),
                gap_max=_format_seconds(decoded_interarrival.get("max")),
                publish=_format_seconds(publish.get("p90")),
                model_step=_format_seconds(model_step.get("p90")),
                sender_drops=run.get("sender_dropped_for_lag_count", "?"),
                missing=run.get("missing_receiver_frame_count", "?"),
            )
        )

    lines.extend(["", "## Acceptance", ""])
    integrity_records = _mapping_list(acceptance.get("run_integrity"))
    if not integrity_records:
        lines.append("No run-integrity records were produced.")
        lines.append("")
    for integrity in integrity_records:
        lines.append(
            f"### Run {integrity.get('run_id', 'unknown')}: "
            f"{'PASS' if integrity.get('passed') else 'FAIL'}"
        )
        lines.append("")
        criteria = _mapping(integrity.get("criteria", {}), "criteria")
        for name, raw_criterion in criteria.items():
            criterion = _mapping(raw_criterion, name)
            lines.append(
                f"- {'PASS' if criterion.get('passed') else 'FAIL'} `{name}`: "
                f"observed={_format_number(criterion.get('observed'))}, "
                f"expected={_format_number(criterion.get('threshold'))}"
            )
        lines.append("")

    lines.append("## A/B thresholds")
    lines.append("")
    pairs = _mapping_list(acceptance.get("pairs"))
    if not pairs:
        lines.append("No complete baseline/candidate pairs were produced.")
    for pair in pairs:
        lines.append(
            f"### Pair {int(pair.get('pair_index', 0)) + 1}: "
            f"{'PASS' if pair.get('passed') else 'FAIL'}"
        )
        lines.append("")
        criteria = _mapping(pair.get("criteria", {}), "criteria")
        for name, raw_criterion in criteria.items():
            criterion = _mapping(raw_criterion, name)
            lines.append(
                f"- {'PASS' if criterion.get('passed') else 'FAIL'} `{name}`: "
                f"observed={_format_number(criterion.get('observed'))}, "
                f"threshold={_format_number(criterion.get('threshold'))}"
            )
        lines.append("")
    lines.append(
        f"Overall acceptance: **{'PASS' if acceptance.get('passed') else 'FAIL'}**"
    )
    lines.append("")
    return "\n".join(lines)


def run_benchmark(
    definition: BenchmarkDefinition,
    *,
    output_dir: Path,
    repo_root: Path,
    dry_run: bool,
    enforce_acceptance: bool,
) -> tuple[dict[str, Any], int]:
    """Run every fresh-process case and write raw and summarized artifacts."""
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    requests = build_run_matrix(definition, repo_root=repo_root)
    _write_json(
        output_dir / "definition.json",
        {
            "schema_version": _SCHEMA_VERSION,
            "benchmark_id": definition.id,
            "requests": [request.to_dict() for request in requests],
        },
    )

    if dry_run:
        raw_runs = [_dry_run_artifact(request) for request in requests]
        environment: Mapping[str, Any] = {}
    else:
        from tools.benchmarks.environment import collect_environment

        environment = collect_environment(repo_root)
        raw_runs = [
            _launch_worker(request, output_dir, timeout_s=definition.worker_timeout_s)
            for request in requests
        ]
    summarized = [
        summarize_run(raw, warmup_steps=definition.warmup_steps) for raw in raw_runs
    ]
    acceptance = evaluate_acceptance(summarized, definition)
    summary = {
        "schema_version": _SCHEMA_VERSION,
        "artifact_type": _SUMMARY_ARTIFACT_TYPE,
        "created_at": _utc_now(),
        "benchmark_id": definition.id,
        "dry_run": dry_run,
        "environment": environment,
        "metadata": dict(definition.metadata),
        "protocol": {
            "application": definition.application,
            "common_application_args": list(definition.common_application_args),
            "variants": {
                key: list(value) for key, value in definition.variants.items()
            },
            "run_order": list(definition.run_order),
            "warmup_steps": definition.warmup_steps,
            "measured_steps": definition.measured_steps,
            "receiver": "loopback-aiortc-rgb24",
            "receiver_ready_gated": True,
            "fresh_worker_process_per_run": True,
        },
        "runs": summarized,
        "acceptance": acceptance,
    }
    _write_json(output_dir / "webrtc_ab.json", summary)
    (output_dir / "webrtc_ab.md").write_text(render_markdown(summary), encoding="utf-8")
    if dry_run:
        return summary, 0
    failed_run = any(item.get("status") != "pass" for item in summarized)
    failed_acceptance = enforce_acceptance and not acceptance["passed"]
    return summary, 1 if failed_run or failed_acceptance else 0


def _launch_worker(
    request: RunRequest, output_dir: Path, *, timeout_s: float
) -> dict[str, Any]:
    run_dir = output_dir / "runs" / request.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    request_path = run_dir / "request.json"
    raw_path = run_dir / "raw.json"
    log_path = run_dir / "command.log"
    _write_json(request_path, request.to_dict())
    # Do not mistake a previous run's artifact for output from a worker that
    # timed out before writing its own result.
    raw_path.unlink(missing_ok=True)
    command = (
        sys.executable,
        "-m",
        "tools.benchmarks.v2_webrtc_ab",
        "--worker-request",
        str(request_path),
        "--worker-output",
        str(raw_path),
    )
    started = time.perf_counter()
    timed_out = False
    returncode: int | None = None
    worker_exception: str | None = None
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    try:
        with log_path.open("w", encoding="utf-8") as log:
            log.write(f"command: {' '.join(command)}\n")
            log.flush()
            completed = subprocess.run(
                command,
                cwd=request.repo_root,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=timeout_s,
                check=False,
            )
        returncode = completed.returncode
    except subprocess.TimeoutExpired as error:
        timed_out = True
        worker_exception = f"worker timed out after {timeout_s:.1f}s: {error}"
    wall_time_s = time.perf_counter() - started
    artifact_exception: str | None = None
    payload: dict[str, Any] = {}
    if raw_path.exists():
        try:
            raw_payload = json.loads(raw_path.read_text(encoding="utf-8"))
            if not isinstance(raw_payload, dict):
                raise TypeError("worker artifact must contain a JSON object")
            payload = raw_payload
        except (OSError, UnicodeError, json.JSONDecodeError, TypeError) as error:
            artifact_exception = (
                f"worker produced an invalid artifact: {type(error).__name__}: {error}"
            )
    else:
        artifact_exception = "worker did not produce an artifact"
    payload.update(
        {
            "run_id": request.run_id,
            "label": request.label,
            "worker_command": list(command),
            "worker_returncode": returncode,
            "worker_timed_out": timed_out,
            "worker_wall_time_s": wall_time_s,
            "raw_artifact": str(raw_path.relative_to(output_dir)),
            "log_path": str(log_path.relative_to(output_dir)),
        }
    )
    if worker_exception is not None:
        payload["status"] = "fail"
        payload["exception"] = worker_exception
    elif artifact_exception is not None:
        payload["status"] = "fail"
        payload["exception"] = artifact_exception
    elif returncode != 0:
        payload["status"] = "fail"
        payload.setdefault("exception", f"worker exited with status {returncode}")
    payload.setdefault("model_results", [])
    payload.setdefault("window_writes", [])
    payload.setdefault("receiver_frames", [])
    _write_json(raw_path, payload)
    return payload


def _dry_run_artifact(request: RunRequest) -> dict[str, Any]:
    return {
        "schema_version": _SCHEMA_VERSION,
        "artifact_type": _RUN_ARTIFACT_TYPE,
        "run_id": request.run_id,
        "label": request.label,
        "status": "dry_run",
        "model_results": [],
        "window_writes": [],
        "receiver_frames": [],
    }


def _run_worker(request: RunRequest, output_path: Path) -> int:
    """Run one application and loopback receiver in the current process."""
    import asyncio

    artifact: dict[str, Any] = {
        "schema_version": _SCHEMA_VERSION,
        "artifact_type": _RUN_ARTIFACT_TYPE,
        "run_id": request.run_id,
        "label": request.label,
        "status": "fail",
        "request": request.to_dict(),
        "started_at": _utc_now(),
        "model_results": [],
        "window_writes": [],
        "receiver_frames": [],
        "receiver_drain_complete": False,
        "webrtc_sender_metrics": {},
        "exception": None,
    }
    started = time.perf_counter()
    try:
        asyncio.run(_run_worker_async(request, artifact, started_at=started))
    except Exception as error:  # noqa: BLE001 - persist any worker failure
        artifact["exception"] = "".join(traceback.format_exception(error))
    artifact["wall_time_s"] = time.perf_counter() - started
    artifact["finished_at"] = _utc_now()
    if artifact["exception"] is None:
        artifact["status"] = "pass"
    _write_json(output_path, artifact)
    return 0 if artifact["status"] == "pass" else 1


async def _run_worker_async(
    request: RunRequest,
    artifact: dict[str, Any],
    *,
    started_at: float,
) -> None:
    import asyncio

    from flashdreams.runtime_v2.application_registry import (
        create_application,
    )
    from flashdreams.runtime_v2.session_runner import run_session

    application = create_application(request.application)
    session: Any | None = None
    window: Any | None = None
    runner: threading.Thread | None = None
    runner_errors: list[Exception] = []
    try:
        described = application.session_desc()
        session_desc = _apply_session_overrides(described, request.session_overrides)
        application.init(request.application_args)
        session = application.create_session(session_desc)
        sink = _RawMetricsSink(artifact["model_results"], started_at=started_at)
        window = _GatedWindow(
            artifact["window_writes"],
            started_at=started_at,
            ready_timeout_s=request.receiver_ready_timeout_s,
            drain_timeout_s=request.receiver_drain_timeout_s,
        )
        artifact["session"] = _session_record(session_desc)

        def run_target() -> None:
            try:
                run_session(
                    session,
                    window,
                    metrics_output_sink=sink,
                    steps=request.warmup_steps + request.measured_steps,
                )
            except Exception as error:  # noqa: BLE001 - propagate across threads
                runner_errors.append(error)

        runner = threading.Thread(
            target=run_target,
            name=f"webrtc-benchmark-{request.run_id}",
        )
        runner.start()
        await _receive_frames(
            window,
            artifact["receiver_frames"],
            runner=runner,
            ready_timeout_s=request.receiver_ready_timeout_s,
            drain_timeout_s=request.receiver_drain_timeout_s,
            started_at=started_at,
        )
        await asyncio.to_thread(runner.join, request.receiver_drain_timeout_s + 10.0)
        if runner.is_alive():
            raise TimeoutError("session runner did not stop after receiver drain")
        if runner_errors:
            raise runner_errors[0]
        artifact["receiver_drain_complete"] = window.receiver_drain_complete
        artifact["webrtc_sender_metrics"] = window.metrics_snapshot()
    finally:
        if runner is not None and runner.is_alive():
            if window is not None:
                window.request_session_close()
                window.force_receiver_drain()
            # A CUDA model step is not host-preemptible. Wait for it to observe
            # shutdown before releasing the application-owned pipeline; the
            # outer worker timeout remains the hard failure bound.
            await asyncio.to_thread(runner.join)
        application.close()


class _RawMetricsSink:
    """Raw model-result collector written by the worker's model thread."""

    def __init__(self, records: list[dict[str, Any]], *, started_at: float) -> None:
        self._records = records
        self._started_at = started_at

    def open(self, session_desc: Any) -> None:
        del session_desc

    def write(self, result: StepResult) -> None:
        self._records.append(
            {
                "step_index": int(result.step_index),
                "frame_count": int(result.frame_count),
                "recorded_at_s": time.perf_counter() - self._started_at,
                "metrics": dict(result.metrics),
            }
        )

    def close(self) -> None:
        return


class _GatedWindow(IClientWindow):
    """Lazy WebRTC window wrapper that gates start and drains on close."""

    def __init__(
        self,
        records: list[dict[str, Any]],
        *,
        started_at: float,
        ready_timeout_s: float,
        drain_timeout_s: float,
    ) -> None:
        from flashdreams.runtime_v2.webrtc_client_window import (
            WebRTCClientWindow,
        )

        self._window = WebRTCClientWindow(host="127.0.0.1", port=0)
        self._records = records
        self._started_at = started_at
        self._ready_timeout_s = ready_timeout_s
        self._drain_timeout_s = drain_timeout_s
        self._opened = threading.Event()
        self._receiver_ready = threading.Event()
        self._closing = threading.Event()
        self._receiver_drained = threading.Event()
        self._close_requested = threading.Event()
        self._lock = threading.Lock()
        self._submitted_frames = 0
        self._receiver_drain_complete = False
        self._close_delivered = False

    @property
    def url(self) -> str:
        return self._window.server.url

    @property
    def submitted_frames(self) -> int:
        with self._lock:
            return self._submitted_frames

    @property
    def closing(self) -> bool:
        return self._closing.is_set()

    @property
    def receiver_drain_complete(self) -> bool:
        return self._receiver_drain_complete

    @property
    def sender_settled(self) -> bool:
        """Return whether every enqueued frame was handed off or dropped."""
        metrics = self.metrics_snapshot()
        enqueued = int(metrics["webrtc_sender_enqueued_count"])
        handed_off = int(metrics["webrtc_sender_handed_off_count"])
        dropped = int(metrics["webrtc_sender_dropped_for_lag_count"])
        depth = int(metrics["webrtc_sender_queue_depth_count"])
        return depth == 0 and handed_off + dropped >= enqueued

    @property
    def expected_receiver_frames(self) -> int:
        """Return real frames handed to the encoder."""
        handed_off = int(self.metrics_snapshot()["webrtc_sender_handed_off_count"])
        return handed_off

    def wait_until_open(self, timeout_s: float) -> bool:
        return self._opened.wait(timeout_s)

    def mark_receiver_ready(self) -> None:
        self._receiver_ready.set()

    def mark_receiver_drained(self) -> None:
        self._receiver_drain_complete = True
        self._receiver_drained.set()

    def force_receiver_drain(self) -> None:
        self._receiver_drained.set()

    def request_session_close(self) -> None:
        """Ask the runner to stop through the public input-event contract."""
        self._close_requested.set()

    def open(self, session_desc: SessionDesc) -> None:
        self._window.open(session_desc)
        self._opened.set()
        if not self._receiver_ready.wait(self._ready_timeout_s):
            raise TimeoutError("WebRTC receiver was not ready before model start")

    def get_user_input_events(self) -> UserInputEvents:
        events = list(self._window.get_user_input_events().get_events())
        with self._lock:
            if self._close_requested.is_set() and not self._close_delivered:
                events.append(CloseUserInputEvent(timestamp=uint64(0)))
                self._close_delivered = True
        return UserInputEvents(events)

    def metrics_snapshot(self) -> dict[str, float | int]:
        """Delegate the runtime's non-blocking WebRTC sender telemetry."""
        return self._window.metrics_snapshot()

    def write(self, result: StepResult) -> None:
        started = time.perf_counter()
        self._window.write(result)
        completed = time.perf_counter()
        record = {
            "step_index": int(result.step_index),
            "frame_count": int(result.frame_count),
            "submitted_at_s": completed - self._started_at,
            "write_wall_s": completed - started,
            "metrics": dict(result.metrics),
        }
        with self._lock:
            self._submitted_frames += int(result.frame_count)
            record["submitted_frame_count"] = self._submitted_frames
            self._records.append(record)

    def close(self) -> None:
        deadline = time.monotonic() + self._drain_timeout_s
        self._closing.set()
        self._window.close()
        self._receiver_drained.wait(max(0.0, deadline - time.monotonic()))


async def _receive_frames(
    window: _GatedWindow,
    records: list[dict[str, Any]],
    *,
    runner: threading.Thread,
    ready_timeout_s: float,
    drain_timeout_s: float,
    started_at: float,
) -> None:
    import asyncio

    from aiohttp import ClientSession
    from aiortc import (
        MediaStreamTrack,
        RTCDataChannel,
        RTCPeerConnection,
        RTCSessionDescription,
    )
    from aiortc.mediastreams import MediaStreamError
    from av import VideoFrame

    if not await asyncio.to_thread(window.wait_until_open, ready_timeout_s):
        raise TimeoutError("WebRTC server did not open before receiver negotiation")
    peer = RTCPeerConnection()
    channel: RTCDataChannel = peer.createDataChannel("controls")
    peer.addTransceiver("video", direction="recvonly")
    channel_opened = asyncio.Event()
    track_future: asyncio.Future[MediaStreamTrack] = (
        asyncio.get_running_loop().create_future()
    )

    @channel.on("open")
    def on_open() -> None:
        channel_opened.set()

    @peer.on("track")
    def on_track(track: MediaStreamTrack) -> None:
        if not track_future.done():
            track_future.set_result(track)

    try:
        await peer.setLocalDescription(await peer.createOffer())
        async with (
            ClientSession() as client,
            client.post(
                f"{window.url}api/webrtc/offer",
                json={
                    "sdp": peer.localDescription.sdp,
                    "type": peer.localDescription.type,
                },
            ) as response,
        ):
            if response.status != 200:
                body = await response.text()
                raise RuntimeError(f"WebRTC offer failed: {response.status} {body}")
            answer = await response.json()
        await peer.setRemoteDescription(
            RTCSessionDescription(sdp=answer["sdp"], type=answer["type"])
        )
        await asyncio.wait_for(channel_opened.wait(), timeout=ready_timeout_s)
        track = await asyncio.wait_for(track_future, timeout=ready_timeout_s)
        window.mark_receiver_ready()

        closing_started_at: float | None = None
        while True:
            if window.closing:
                closing_started_at = closing_started_at or time.monotonic()
                if time.monotonic() - closing_started_at >= drain_timeout_s:
                    break
            try:
                received = await asyncio.wait_for(track.recv(), timeout=1.0)
            except asyncio.TimeoutError:
                if not runner.is_alive() and not window.closing:
                    break
                continue
            except MediaStreamError:
                break
            if not isinstance(received, VideoFrame):
                raise TypeError("WebRTC video track returned a non-video frame")
            frame = received
            decode_started = time.perf_counter()
            pixels = frame.to_ndarray(format="rgb24")
            decoded_at = time.perf_counter()
            time_base = frame.time_base
            records.append(
                {
                    "frame_index": len(records),
                    "pts": frame.pts,
                    "time_base": None if time_base is None else float(time_base),
                    "received_at_s": decoded_at - started_at,
                    "decode_s": decoded_at - decode_started,
                    "shape": list(pixels.shape),
                }
            )
        if (
            window.closing
            and window.sender_settled
            and len(records) >= window.expected_receiver_frames
        ):
            window.mark_receiver_drained()
    finally:
        window.force_receiver_drain()
        await peer.close()


def _apply_session_overrides(described: Any, overrides: Mapping[str, Any]) -> Any:
    from flashdreams.runtime_v2.session_desc import (
        BackpressureMode,
        PresentationMode,
        SessionDesc,
    )
    from flashdreams.runtime_v2.video_tensor import VideoTensorLayout

    fields = dict(overrides)
    if "output_layout" in fields:
        fields["output_layout"] = VideoTensorLayout(fields["output_layout"])
    if "backpressure_mode" in fields:
        fields["backpressure_mode"] = BackpressureMode(fields["backpressure_mode"])
    if "presentation_mode" in fields:
        fields["presentation_mode"] = PresentationMode(fields["presentation_mode"])
    if described is None:
        return SessionDesc(**fields)
    return replace(described, **fields)


def _session_record(session_desc: Any) -> dict[str, Any]:
    return {
        "output_layout": session_desc.output_layout.value,
        "backpressure_mode": session_desc.backpressure_mode.value,
        "presentation_mode": session_desc.presentation_mode.value,
        "frames_per_second_for_step": session_desc.frames_per_second_for_step,
        "frames_per_second_for_ui": session_desc.frames_per_second_for_ui,
        "video_width": session_desc.video_width,
        "video_height": session_desc.video_height,
    }


def _definition_from_dict(payload: Mapping[str, Any]) -> BenchmarkDefinition:
    variants_payload = _mapping(payload.get("variants"), "variants")
    variants: dict[str, tuple[str, ...]] = {}
    for label, value in variants_payload.items():
        if isinstance(value, Mapping):
            args = value.get("application_args", ())
        else:
            args = value
        variants[str(label)] = _string_tuple(args, f"variants.{label}")
    if len(variants) < 2:
        raise ValueError("variants must define at least two cases")
    baseline = _required_string(payload, "baseline")
    candidate = _required_string(payload, "candidate")
    if baseline == candidate:
        raise ValueError("baseline and candidate must differ")
    if baseline not in variants or candidate not in variants:
        raise ValueError("baseline and candidate must name configured variants")
    run_order = _string_tuple(payload.get("run_order", ()), "run_order")
    if not run_order:
        run_order = (baseline, candidate, candidate, baseline)
    unknown = sorted(set(run_order).difference(variants))
    if unknown:
        raise ValueError(f"run_order contains unknown variants: {unknown}")
    if run_order.count(baseline) != run_order.count(candidate):
        raise ValueError(
            "run_order must contain equally many baseline and candidate runs"
        )
    acceptance_payload = _mapping(payload.get("acceptance", {}), "acceptance")
    supported_acceptance = {
        "decoded_fps_ratio_min",
        "model_fps_ratio_min",
        "model_step_wall_median_ratio_max",
        "model_step_wall_p90_ratio_max",
        "publish_wait_p90_s_max",
        "decoded_interarrival_p90_s_max",
        "decoded_interarrival_max_s_max",
    }
    unknown_acceptance = sorted(
        set(acceptance_payload).difference(supported_acceptance)
    )
    if unknown_acceptance:
        raise ValueError(
            f"acceptance contains unknown thresholds: {unknown_acceptance}"
        )
    acceptance: dict[str, float] = {}
    for key, value in acceptance_payload.items():
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise TypeError(f"acceptance.{key} must be numeric")
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError(f"acceptance.{key} must be finite")
        if key == "publish_wait_p90_s_max":
            if numeric < 0.0:
                raise ValueError(f"acceptance.{key} must be non-negative")
        elif numeric <= 0.0:
            raise ValueError(f"acceptance.{key} must be positive")
        acceptance[str(key)] = numeric
    return BenchmarkDefinition(
        id=_required_string(payload, "id"),
        application=_required_string(payload, "application"),
        common_application_args=_string_tuple(
            payload.get("common_application_args", ()), "common_application_args"
        ),
        variants=variants,
        run_order=run_order,
        baseline=baseline,
        candidate=candidate,
        warmup_steps=_nonnegative_int(payload, "warmup_steps"),
        measured_steps=_positive_int(payload, "measured_steps"),
        receiver_ready_timeout_s=_positive_float(
            payload, "receiver_ready_timeout_s", default=30.0
        ),
        receiver_drain_timeout_s=_positive_float(
            payload, "receiver_drain_timeout_s", default=30.0
        ),
        worker_timeout_s=_positive_float(payload, "worker_timeout_s", default=3600.0),
        session_overrides=_mapping(
            payload.get("session_overrides", {}), "session_overrides"
        ),
        metadata=_mapping(payload.get("metadata", {}), "metadata"),
        acceptance=acceptance,
    )


def _run_integrity_criteria(
    run: Mapping[str, Any],
    *,
    expected_model_steps: int,
    expected_measured_steps: int,
) -> dict[str, dict[str, Any]]:
    """Return non-negotiable completeness checks for one worker run."""
    expected_sender_enqueued = run.get("expected_sender_enqueued_frame_count")
    sender_enqueued = run.get("sender_enqueued_frame_count")
    model_metrics = _mapping(run.get("model_metrics", {}), "model_metrics")
    model_step_wall = _mapping(
        model_metrics.get("model_step_wall_s", {}), "model_step_wall_s"
    )
    publish_wait = _mapping(
        model_metrics.get("runtime_presentation_publish_wait_s", {}),
        "runtime_presentation_publish_wait_s",
    )
    return {
        "model_step_count": _criterion(
            run.get("model_step_count"), expected_model_steps, comparison="equal"
        ),
        "model_step_indices_contiguous": _criterion(
            1 if run.get("model_step_indices_contiguous") else 0,
            1,
            comparison="equal",
        ),
        "steady_model_step_count": _criterion(
            run.get("steady_model_step_count"),
            expected_measured_steps,
            comparison="equal",
        ),
        "model_step_wall_sample_count": _criterion(
            model_step_wall.get("count"),
            expected_measured_steps,
            comparison="equal",
        ),
        "presentation_publish_wait_sample_count": _criterion(
            publish_wait.get("count"),
            expected_measured_steps,
            comparison="equal",
        ),
        "submitted_frame_count": _criterion(
            run.get("submitted_frame_count"),
            run.get("generated_frame_count"),
            comparison="equal",
        ),
        "receiver_drain_complete": _criterion(
            1 if run.get("receiver_drain_complete") else 0,
            1,
            comparison="equal",
        ),
        "receiver_pts_strictly_increasing": _criterion(
            1 if run.get("receiver_pts_strictly_increasing") else 0,
            1,
            comparison="equal",
        ),
        "sender_dropped_for_lag_count": _criterion(
            run.get("sender_dropped_for_lag_count"), 0, comparison="equal"
        ),
        "sender_enqueued_frame_count": _criterion(
            sender_enqueued,
            expected_sender_enqueued,
            comparison="equal",
        ),
        "sender_handed_off_frame_count": _criterion(
            run.get("sender_handed_off_frame_count"),
            sender_enqueued,
            comparison="equal",
        ),
        "sender_materialized_frame_count": _criterion(
            run.get("sender_materialized_frame_count"),
            expected_sender_enqueued,
            comparison="equal",
        ),
        "sender_discarded_on_close_count": _criterion(
            run.get("sender_discarded_on_close_count"), 0, comparison="equal"
        ),
        "missing_receiver_frame_count": _criterion(
            run.get("missing_receiver_frame_count"), 0, comparison="equal"
        ),
        "extra_receiver_frame_count": _criterion(
            run.get("extra_receiver_frame_count"), 0, comparison="equal"
        ),
    }


def _pair_criteria(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    thresholds: Mapping[str, float],
) -> dict[str, dict[str, Any]]:
    criteria: dict[str, dict[str, Any]] = {}

    def minimum_ratio(
        name: str, candidate_value: Any, baseline_value: Any, threshold_name: str
    ) -> None:
        if threshold_name not in thresholds:
            return
        ratio = _safe_ratio(candidate_value, baseline_value)
        threshold = float(thresholds[threshold_name])
        criteria[name] = _criterion(ratio, threshold, comparison="min")

    def maximum_ratio(
        name: str, candidate_value: Any, baseline_value: Any, threshold_name: str
    ) -> None:
        if threshold_name not in thresholds:
            return
        ratio = _safe_ratio(candidate_value, baseline_value)
        threshold = float(thresholds[threshold_name])
        criteria[name] = _criterion(ratio, threshold, comparison="max")

    minimum_ratio(
        "decoded_fps_ratio",
        candidate.get("decoded_steady_fps"),
        baseline.get("decoded_steady_fps"),
        "decoded_fps_ratio_min",
    )
    minimum_ratio(
        "model_fps_ratio",
        candidate.get("steady_model_fps"),
        baseline.get("steady_model_fps"),
        "model_fps_ratio_min",
    )
    baseline_metrics = _mapping(baseline.get("model_metrics", {}), "model_metrics")
    candidate_metrics = _mapping(candidate.get("model_metrics", {}), "model_metrics")
    baseline_step = _mapping(
        baseline_metrics.get("model_step_wall_s", {}), "model_step_wall_s"
    )
    candidate_step = _mapping(
        candidate_metrics.get("model_step_wall_s", {}), "model_step_wall_s"
    )
    maximum_ratio(
        "model_step_wall_median_ratio",
        candidate_step.get("median"),
        baseline_step.get("median"),
        "model_step_wall_median_ratio_max",
    )
    maximum_ratio(
        "model_step_wall_p90_ratio",
        candidate_step.get("p90"),
        baseline_step.get("p90"),
        "model_step_wall_p90_ratio_max",
    )
    if "publish_wait_p90_s_max" in thresholds:
        publish = _mapping(
            candidate_metrics.get("runtime_presentation_publish_wait_s", {}),
            "runtime_presentation_publish_wait_s",
        )
        criteria["publish_wait_p90_s"] = _criterion(
            publish.get("p90"),
            float(thresholds["publish_wait_p90_s_max"]),
            comparison="max",
        )
    decoded_interarrival = _mapping(
        candidate.get("decoded_interarrival_s", {}), "decoded_interarrival_s"
    )
    for field, threshold_name in (
        ("p90", "decoded_interarrival_p90_s_max"),
        ("max", "decoded_interarrival_max_s_max"),
    ):
        if threshold_name in thresholds:
            criteria[f"decoded_interarrival_{field}_s"] = _criterion(
                decoded_interarrival.get(field),
                float(thresholds[threshold_name]),
                comparison="max",
            )
    return criteria


def _criterion(observed: Any, threshold: Any, *, comparison: str) -> dict[str, Any]:
    numeric = _optional_float_value(observed)
    expected = _optional_float_value(threshold)
    if comparison == "min":
        passed = numeric is not None and expected is not None and numeric >= expected
    elif comparison == "max":
        passed = numeric is not None and expected is not None and numeric <= expected
    elif comparison == "equal":
        passed = numeric is not None and expected is not None and numeric == expected
    else:
        raise ValueError(f"Unsupported criterion comparison: {comparison}")
    return {
        "observed": numeric,
        "threshold": expected,
        "comparison": comparison,
        "passed": passed,
        "missing": numeric is None or expected is None,
    }


def _metric_distributions(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_name: dict[str, list[float]] = {}
    for record in records:
        metrics = record.get("metrics")
        if not isinstance(metrics, Mapping):
            continue
        for name, value in metrics.items():
            numeric = _optional_float_value(value)
            if numeric is not None:
                by_name.setdefault(str(name), []).append(numeric)
    return {name: distribution(values) for name, values in sorted(by_name.items())}


def _skip_frame_prefix(
    records: Sequence[Mapping[str, Any]], frame_count: int
) -> list[Mapping[str, Any]]:
    skipped = 0
    retained: list[Mapping[str, Any]] = []
    for record in records:
        current = int(record.get("frame_count", 0))
        if skipped + current <= frame_count:
            skipped += current
            continue
        retained.append(record)
    return retained


def _model_step_fps(records: Sequence[Mapping[str, Any]]) -> float | None:
    """Return aggregate FPS from every record's model-step wall time."""
    if not records:
        return None
    frame_count = 0
    wall_time_s = 0.0
    for record in records:
        current_frames = _optional_int(record.get("frame_count"))
        metrics = record.get("metrics")
        if (
            current_frames is None
            or current_frames <= 0
            or not isinstance(metrics, Mapping)
        ):
            return None
        current_wall_time_s = _optional_float_value(metrics.get("model_step_wall_s"))
        if current_wall_time_s is None or current_wall_time_s <= 0.0:
            return None
        frame_count += current_frames
        wall_time_s += current_wall_time_s
    return frame_count / wall_time_s


def _completion_fps(
    records: Sequence[Mapping[str, Any]],
    timestamp_key: str,
    *,
    frame_count: int | None = None,
) -> float | None:
    timestamped = [
        item
        for item in records
        if _optional_float_value(item.get(timestamp_key)) is not None
    ]
    if len(timestamped) < 2:
        return None
    started = float(timestamped[0][timestamp_key])
    completed = float(timestamped[-1][timestamp_key])
    elapsed = completed - started
    if elapsed <= 0.0:
        return None
    frames = sum(
        frame_count if frame_count is not None else int(item.get("frame_count", 0))
        for item in timestamped[1:]
    )
    return frames / elapsed


def _trailing_rates(
    timestamps: Sequence[float], *, window_seconds: float
) -> list[float]:
    rates: list[float] = []
    left = 0
    for right, current in enumerate(timestamps):
        while left < right and timestamps[left] < current - window_seconds:
            left += 1
        elapsed = current - timestamps[left]
        if elapsed >= window_seconds * 0.75 and right > left:
            rates.append((right - left) / elapsed)
    return rates


def _numeric_values(records: Sequence[Mapping[str, Any]], key: str) -> list[float]:
    values: list[float] = []
    for record in records:
        value = _optional_float_value(record.get(key))
        if value is not None:
            values.append(value)
    return values


def _safe_ratio(numerator: Any, denominator: Any) -> float | None:
    left = _optional_float_value(numerator)
    right = _optional_float_value(denominator)
    if left is None or right is None or right <= 0.0:
        return None
    return left / right


def _mapping(value: Any, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be an object")
    return {str(key): item for key, item in value.items()}


def _mapping_list(value: Any) -> list[Mapping[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        return []
    return [item for item in value if isinstance(item, Mapping)]


def _required_string(payload: Mapping[str, Any], field_name: str) -> str:
    value = payload.get(field_name)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


def _string_tuple(value: Any, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        raise TypeError(f"{field_name} must be a list of strings")
    if any(not isinstance(item, str) or not item for item in value):
        raise ValueError(f"{field_name} must contain only non-empty strings")
    return tuple(value)


def _nonnegative_int(payload: Mapping[str, Any], field_name: str) -> int:
    value = payload.get(field_name)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")
    return value


def _positive_int(payload: Mapping[str, Any], field_name: str) -> int:
    value = payload.get(field_name)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer")
    return value


def _positive_float(
    payload: Mapping[str, Any], field_name: str, *, default: float | None = None
) -> float:
    value = payload.get(field_name, default)
    if isinstance(value, bool) or not isinstance(value, int | float) or value <= 0:
        raise ValueError(f"{field_name} must be a positive number")
    return float(value)


def _optional_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _optional_float_value(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _format_number(value: Any) -> str:
    numeric = _optional_float_value(value)
    return "--" if numeric is None else f"{numeric:.3f}"


def _format_seconds(value: Any) -> str:
    numeric = _optional_float_value(value)
    return "--" if numeric is None else f"{numeric * 1_000.0:.3f} ms"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(
        f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
    )
    try:
        temporary_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--benchmark")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--enforce-acceptance",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--worker-request", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--worker-output", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    worker_mode = args.worker_request is not None or args.worker_output is not None
    if worker_mode:
        if args.worker_request is None or args.worker_output is None:
            parser.error("worker mode requires --worker-request and --worker-output")
        return args
    for field, flag in (
        (args.config, "--config"),
        (args.benchmark, "--benchmark"),
        (args.output_dir, "--output-dir"),
    ):
        if field is None:
            parser.error(f"{flag} is required")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    """Run an orchestrator or its private fresh-process worker."""
    args = _parse_args(argv)
    if args.worker_request is not None:
        payload = json.loads(args.worker_request.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError("worker request must contain a JSON object")
        return _run_worker(RunRequest.from_dict(payload), args.worker_output)
    definition = load_benchmark_definition(args.config, args.benchmark)
    summary, returncode = run_benchmark(
        definition,
        output_dir=args.output_dir,
        repo_root=args.repo_root,
        dry_run=args.dry_run,
        enforce_acceptance=args.enforce_acceptance,
    )
    output_dir = args.output_dir.resolve()
    print(f"WebRTC A/B JSON: {output_dir / 'webrtc_ab.json'}", flush=True)
    print(f"WebRTC A/B report: {output_dir / 'webrtc_ab.md'}", flush=True)
    print(
        f"WebRTC A/B acceptance: "
        f"{'PASS' if summary['acceptance']['passed'] else 'FAIL'}",
        flush=True,
    )
    return returncode


if __name__ == "__main__":
    raise SystemExit(main())
