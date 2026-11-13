# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Robotaxi game outcomes alongside shared runtime timing diagnostics."""

import json
from pathlib import Path

from omnidreams_game_engine.debug_report import (
    _records,
)
from omnidreams_game_engine.debug_report import (
    write_timing_report as engine_timing_report,
)


def write_timing_report(directory: Path):
    """Add race outcomes and route charts to the engine's timing evidence."""
    report = engine_timing_report(directory)
    race_events = []
    previous = None
    final_game = None
    for record in _records(directory / "telemetry.jsonl"):
        final_game = record.get("game")
        if final_game is None or final_game.get("event") is None:
            continue
        key = (
            record["rollout_epoch"],
            final_game["event"],
            final_game.get("elapsed_time_us"),
            final_game.get("checkpoint_index"),
            final_game.get("completed_laps"),
        )
        if key != previous:
            race_events.append(
                {
                    "rollout_epoch": record["rollout_epoch"],
                    "frame": record["frame"],
                    "simulation_timestamp_us": record["simulation_timestamp_us"],
                    "event": final_game["event"],
                    "game": final_game,
                }
            )
            previous = key
    report.update(race_events=race_events, final_game=final_game)
    (directory / "timing_summary.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    from crazy_robotaxi.debug_overview import write_capture_overview

    write_capture_overview(directory)
    return report
