# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Timing evidence from headless input, physics, and presentation observations."""

from __future__ import annotations

import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


def _records(path: Path):
    if path.exists():
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                yield json.loads(line)


def _milliseconds(values_us: list[float]) -> dict[str, float | int | None]:
    values = sorted(value / 1000 for value in values_us)
    return {
        "count": len(values),
        "median": statistics.median(values) if values else None,
        "p95": values[math.ceil(len(values) * 0.95) - 1] if values else None,
        "max": max(values) if values else None,
    }


def write_timing_report(directory: Path) -> dict[str, Any]:
    """Join recorded milestones without treating requested motion as visual response.

    Input-to-physics timing uses the start of the recorded physics advance.
    Presentation identifies a selected frame, not movement inferred from its
    pixels. Missing observations and ambiguous equal-timestamp inputs remain
    unmeasured rather than being assigned zero latency.
    """
    manifest = json.loads((directory / "manifest.json").read_text())
    ready_ns = manifest.get("ready_ns")
    inputs: dict[str, dict[str, Any]] = {}
    timestamp_ids: dict[int, set[str]] = defaultdict(set)
    generated = {}
    delays = []
    rejected = []
    for event in _records(directory / "input_events.jsonl"):
        status = event["status"]
        if status == "rejected":
            rejected.append(event)
        elif status == "generation_returned":
            generated[(event["rollout_epoch"], event["model_step"])] = event[
                "elapsed_us"
            ]
        elif status == "generation_delay":
            delays.append(event)
        elif event.get("id") is not None and status != "rejected":
            identifier = event["id"]
            row = inputs.setdefault(identifier, {"id": identifier})
            for key in (
                "timestamp_us",
                "received_us",
                "command",
                "dispatched_us",
                "consumed_us",
            ):
                if key in event:
                    row[key] = event[key]
            row["status"] = status
            if "timestamp_us" in event:
                timestamp_ids[event["timestamp_us"]].add(identifier)

    presented = {}
    presentation_intervals = []
    previous_presentation = None
    feedback_changes = []
    previous_feedback = None
    for event in _records(directory / "presentation.jsonl"):
        elapsed = event.get("elapsed_us")
        if elapsed is None:
            continue
        if previous_presentation is not None:
            presentation_intervals.append(elapsed - previous_presentation)
        previous_presentation = elapsed
        source = event.get("source")
        feedback = event.get("hud_feedback")
        if feedback != previous_feedback:
            feedback_changes.append(
                {"elapsed_us": elapsed, "feedback": feedback, "source": source}
            )
            previous_feedback = feedback
        if source is not None:
            presented.setdefault((source["rollout_epoch"], source["frame"]), elapsed)

    epochs: dict[int, list[tuple[int, int]]] = defaultdict(list)
    matched: dict[int, list[dict[str, Any]]] = defaultdict(list)
    race_events = []
    previous_race_event = None
    final_game = None
    presentation_age = []
    for record in _records(directory / "telemetry.jsonl"):
        epoch = record["rollout_epoch"]
        completed_ns = record.get("physics_completed_ns")
        started_ns = record.get("physics_started_ns")
        if completed_ns is not None:
            epochs[epoch].append((record["simulation_timestamp_us"], completed_ns))
        first_presentation = presented.get((epoch, record["frame"]))
        if (
            completed_ns is not None
            and ready_ns is not None
            and first_presentation is not None
        ):
            presentation_age.append(
                first_presentation - (completed_ns - ready_ns) / 1000
            )
        transition = record.get("transition_timestamp_us")
        if transition is not None and started_ns is not None and ready_ns is not None:
            matched[transition].append(
                {
                    "rollout_epoch": epoch,
                    "frame": record["frame"],
                    "physics_started_us": (started_ns - ready_ns) / 1000,
                    "physics_completed_us": (completed_ns - ready_ns) / 1000
                    if completed_ns is not None
                    else None,
                    "generation_returned_us": generated.get(
                        (epoch, record["model_step"])
                    ),
                    "first_presentation_us": first_presentation,
                }
            )
        final_game = record.get("game")
        if final_game is not None and final_game.get("event") is not None:
            event_key = (
                epoch,
                final_game["event"],
                final_game.get("elapsed_time_us"),
                final_game.get("checkpoint_index"),
                final_game.get("completed_laps"),
            )
            if event_key != previous_race_event:
                race_events.append(
                    {
                        "rollout_epoch": epoch,
                        "frame": record["frame"],
                        "simulation_timestamp_us": record["simulation_timestamp_us"],
                        "event": final_game["event"],
                        "game": final_game,
                    }
                )
                previous_race_event = event_key

    input_results = []
    for row in inputs.values():
        timestamp = row.get("timestamp_us")
        if timestamp is None or row.get("command", {}).get("type") == "close":
            continue
        matches = matched.get(timestamp, [])
        ambiguous = len(timestamp_ids[timestamp]) != 1 or len(matches) > 1
        observation = matches[0] if matches and not ambiguous else {}
        result = {
            **row,
            **observation,
            "physics_association": "ambiguous"
            if ambiguous
            else "observed"
            if observation
            else "unobserved",
        }
        for key in (
            "dispatched_us",
            "consumed_us",
            "physics_started_us",
            "generation_returned_us",
            "first_presentation_us",
        ):
            value = result.get(key)
            result["scheduled_to_" + key.removesuffix("_us") + "_ms"] = (
                None if value is None else (value - timestamp) / 1000
            )
        input_results.append(result)

    progress = []
    for epoch, samples in epochs.items():
        sim_us = samples[-1][0] - samples[0][0]
        wall_us = (samples[-1][1] - samples[0][1]) / 1000
        progress.append(
            {
                "rollout_epoch": epoch,
                "physics_samples": len(samples),
                "simulation_elapsed_s": sim_us / 1e6,
                "wall_elapsed_s": wall_us / 1e6,
                "simulation_to_wall_ratio": sim_us / wall_us if wall_us > 0 else None,
                "physics_gap_ms": _milliseconds(
                    [(b[1] - a[1]) / 1000 for a, b in zip(samples, samples[1:])]
                ),
            }
        )
    report = {
        "format_version": 1,
        "capture_complete": manifest.get("capture_complete", False),
        "run_complete": manifest.get("complete", False),
        "termination_reason": manifest.get("termination_reason"),
        "generation_adherence": "not_evaluated",
        "presentation_measurement": "selected frame metadata; not a measurement of visual response",
        "physics_progress": progress,
        "inputs": input_results,
        "rejected_inputs": rejected,
        "physics_to_first_presentation_ms": _milliseconds(presentation_age),
        "presentation_gap_ms": _milliseconds(presentation_intervals),
        "hud_feedback_changes": feedback_changes,
        "generation_delays": delays,
        "race_events": race_events,
        "final_game": final_game,
    }
    (directory / "timing_summary.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    return report
