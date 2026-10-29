# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Route and race overview charts from recorded headless gameplay."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from itertools import pairwise
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape


def _number(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value) if math.isfinite(value) else None
    return None


def write_capture_overview(directory: Path) -> dict[str, Any]:
    """Write SVG charts and JSON evidence without inferring unrecorded outcomes.

    Times are relative to each rollout's first recorded simulation timestamp.
    Paths use physical vehicle observations when available. Missing off-road
    flags remain unknown; an interrupted capture never implies race completion.
    """
    from crazy_robotaxi.debug_report import _records

    manifest = json.loads((directory / "manifest.json").read_text())
    epochs: dict[int, list[dict[str, Any]]] = defaultdict(list)
    origins: dict[int, float] = {}
    for record in _records(directory / "telemetry.jsonl"):
        timestamp = _number(record.get("simulation_timestamp_us"))
        if timestamp is None:
            continue
        epoch = int(record.get("rollout_epoch", 0))
        origin = origins.setdefault(epoch, timestamp)
        vehicle = record.get("physics_vehicle") or record.get("vehicle") or {}
        game = record.get("game") or {}
        off_road = record.get("off_road", game.get("off_road"))
        epochs[epoch].append(
            {
                "frame": record.get("frame"),
                "time_s": (timestamp - origin) / 1e6,
                "x_m": _number(vehicle.get("x_m")),
                "y_m": _number(vehicle.get("y_m")),
                "z_m": _number(vehicle.get("z_m")),
                "speed_mps": _number(vehicle.get("speed_mps")),
                "collision": bool(record.get("actor_collision"))
                or bool(record.get("static_barrier_collision")),
                "off_road": off_road if isinstance(off_road, bool) else None,
                "checkpoint_index": _number(game.get("checkpoint_index")),
                "completed_laps": _number(game.get("completed_laps")),
                "game": game,
            }
        )
    summaries = []
    for epoch, points in epochs.items():
        distance = sum(
            math.hypot(b["x_m"] - a["x_m"], b["y_m"] - a["y_m"])
            for a, b in pairwise(points)
            if b["time_s"] >= a["time_s"]
            and all(p[k] is not None for p in (a, b) for k in ("x_m", "y_m"))
        )
        completion = next(
            (
                p
                for p in points
                if p["game"].get("event") == "race_complete"
                or (
                    p["checkpoint_index"] is not None
                    and _number(p["game"].get("final_time_us")) is not None
                )
            ),
            None,
        )
        events = []
        previous = None
        for p in points:
            game = p["game"]
            key = (
                game.get("event"),
                game.get("checkpoint_index"),
                game.get("completed_laps"),
            )
            if key != previous and game.get("event") is not None:
                events.append({"time_s": p["time_s"], "event": game["event"]})
            previous = key
        summaries.append(
            {
                "rollout_epoch": epoch,
                "samples": len(points),
                "simulation_elapsed_s": points[-1]["time_s"],
                "sampled_distance_m": distance,
                "speed_range_mps": _range(points, "speed_mps"),
                "elevation_range_m": _range(points, "z_m"),
                "collision_observations": sum(p["collision"] for p in points),
                "off_road_observations": sum(p["off_road"] is True for p in points)
                if any(p["off_road"] is not None for p in points)
                else None,
                "race_completed": completion is not None,
                "completion_time_s": None
                if completion is None
                else completion["time_s"],
                "final_time_us": None
                if completion is None
                else completion["game"].get("final_time_us"),
                "final_game": points[-1]["game"],
                "race_events": events,
            }
        )
    report = {
        "format_version": 1,
        "capture_complete": manifest.get("capture_complete", False),
        "termination_reason": manifest.get("termination_reason"),
        "time_basis": "simulation seconds relative to each rollout's first observation",
        "path_basis": "recorded physics_vehicle, falling back to vehicle",
        "distance_basis": "sum of observed XY segments; gaps can underestimate distance",
        "generation_adherence": "not_evaluated",
        "epochs": summaries,
    }
    (directory / "overview.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    (directory / "overview.svg").write_text(_svg(epochs, report), encoding="utf-8")
    return report


def _range(points: list[dict[str, Any]], key: str) -> list[float] | None:
    values = [p[key] for p in points if p[key] is not None]
    return [min(values), max(values)] if values else None


def _svg(epochs: dict[int, list[dict[str, Any]]], report: dict[str, Any]) -> str:
    colors = ("#137cbd", "#8053b8", "#00876c", "#c65319")
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1440" height="1000" viewBox="0 0 1440 1000">',
        '<rect width="1440" height="1000" fill="#f5f7fa"/>',
        '<g font-family="sans-serif" fill="#172b4d">',
    ]

    def text(x, y, value, size=15, **attrs):
        attributes = " ".join(f'{k.replace("_", "-")}="{v}"' for k, v in attrs.items())
        parts.append(
            f'<text x="{x}" y="{y}" font-size="{size}" {attributes}>{escape(str(value))}</text>'
        )

    text(36, 42, "Gameplay capture overview", 27, font_weight="bold")
    status = "Complete capture" if report["capture_complete"] else "Incomplete capture"
    text(
        36,
        70,
        f"{status} · Stop: {report['termination_reason'] or 'unreported'} · Recorded gameplay; visual adherence not evaluated",
    )
    for i, summary in enumerate(report["epochs"][:6]):
        game = summary["final_game"]
        state = (
            "RACE COMPLETE"
            if summary["race_completed"]
            else game.get("session_state", "no race outcome")
        )
        progress = (
            f" · Checkpoint {game['checkpoint_index']}/{game.get('checkpoint_count', '?')}"
            f" · Laps {game.get('completed_laps', '?')}/{game.get('lap_count', '?')}"
            if game.get("checkpoint_index") is not None
            else ""
        )
        text(
            36,
            101 + 21 * i,
            f"Rollout {summary['rollout_epoch']}: {summary['simulation_elapsed_s']:.1f} s · {summary['sampled_distance_m']:.0f} m sampled · {state}{progress}",
            fill=colors[i % len(colors)],
        )
    top = 246

    def plot(title, xkey, ykey, box, *, equal=False, stepped=False):
        left, upper, width, height = box
        parts.append(
            f'<rect x="{left - 18}" y="{upper - 45}" width="{width + 45}" height="{height + 100}" rx="12" fill="white"/>'
        )
        text(left, upper - 18, title, 18, font_weight="bold")
        valid = [
            p
            for points in epochs.values()
            for p in points
            if p[xkey] is not None and p[ykey] is not None
        ]
        if not valid:
            text(
                left + 20,
                upper + height / 2,
                "No observations recorded",
                fill="#617188",
            )
            return
        xmin, xmax = min(p[xkey] for p in valid), max(p[xkey] for p in valid)
        ymin, ymax = min(p[ykey] for p in valid), max(p[ykey] for p in valid)
        dx, dy = max(xmax - xmin, 1.0), max(ymax - ymin, 1.0)
        if equal:
            unit = max(dx / width, dy / height) * 1.08
            dx, dy = width * unit, height * unit
        else:
            dy *= 1.12
        xmin, ymin = (xmin + xmax - dx) / 2, (ymin + ymax - dy) / 2

        def project(p):
            return left + (p[xkey] - xmin) / dx * width, upper + height - (
                p[ykey] - ymin
            ) / dy * height

        for i in range(5):
            fraction = i / 4
            xx, yy = left + width * fraction, upper + height * (1 - fraction)
            parts.append(
                f'<path d="M {xx} {upper} V {upper + height} M {left} {yy} H {left + width}" fill="none" stroke="#e7ecf2"/>'
            )
            text(
                xx,
                upper + height + 23,
                f"{xmin + fraction * dx:.1f}",
                12,
                text_anchor="middle",
            )
            text(left - 7, yy + 4, f"{ymin + fraction * dy:.1f}", 12, text_anchor="end")
        for index, points in enumerate(epochs.values()):
            color = colors[index % len(colors)]
            commands = []
            previous = None
            for p in points:
                if p[xkey] is None or p[ykey] is None:
                    previous = None
                    continue
                xx, yy = project(p)
                if previous is None or p["time_s"] < previous["time_s"]:
                    commands.append(f"M {xx:.2f} {yy:.2f}")
                elif stepped:
                    commands.append(f"H {xx:.2f} V {yy:.2f}")
                else:
                    commands.append(f"L {xx:.2f} {yy:.2f}")
                previous = p
            parts.append(
                f'<path d="{" ".join(commands)}" fill="none" stroke="{color}" stroke-width="2.2"/>'
            )
            for p in points:
                if p[xkey] is None or p[ykey] is None:
                    continue
                xx, yy = project(p)
                if p["off_road"] is True:
                    parts.append(
                        f'<circle cx="{xx:.2f}" cy="{yy:.2f}" r="4" fill="#e5a000"/>'
                    )
                if p["collision"]:
                    parts.append(
                        f'<path d="M {xx - 4:.2f} {yy - 4:.2f} l 8 8 m -8 0 l 8 -8" stroke="#d32943" stroke-width="2"/>'
                    )
            if equal:
                located = [
                    p for p in points if p[xkey] is not None and p[ykey] is not None
                ]
                for label, point in zip(("Start", "End"), located[:1] + located[-1:]):
                    xx, yy = project(point)
                    parts.append(
                        f'<circle cx="{xx:.2f}" cy="{yy:.2f}" r="5" fill="{color}" stroke="white"/>'
                    )
                    text(xx + 8, yy - 8, label, 12, fill=color)
        text(
            left + width / 2,
            upper + height + 44,
            "World X (m)" if equal else "Simulation time (s)",
            13,
            text_anchor="middle",
        )

    plot("Driven path · world Y (m)", "x_m", "y_m", (96, top, 545, 570), equal=True)
    plot("Speed (m/s)", "time_s", "speed_mps", (788, top, 560, 140))
    plot("Elevation (m)", "time_s", "z_m", (788, top + 245, 560, 140))
    plot(
        "Native checkpoint index",
        "time_s",
        "checkpoint_index",
        (788, top + 490, 560, 140),
        stepped=True,
    )
    collisions = sum(e["collision_observations"] for e in report["epochs"])
    offroad = [
        e["off_road_observations"]
        for e in report["epochs"]
        if e["off_road_observations"] is not None
    ]
    text(
        40,
        933,
        f"Red ×: collision observations ({collisions}) · Amber dots: off-road observations ({sum(offroad) if offroad else 'unavailable'})",
    )
    text(
        40,
        960,
        "Each color is a separate rollout. Missing coordinates break paths. Checkpoint indices may reset between laps; final native state is recorded in overview.json.",
        13,
    )
    parts.append("</g></svg>")
    return "\n".join(parts)
