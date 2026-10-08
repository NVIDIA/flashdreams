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

"""Real-model headless driving scenarios with saved observations and capture checks."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path

from flashdreams.runtime_v2.cli import entrypoint


def check_capture(directory: Path) -> dict[str, float | int]:
    """Check real driving observations and return a compact physics summary."""
    manifest = json.loads((directory / "manifest.json").read_text())
    assert manifest["complete"], f"Incomplete run: {manifest}"
    records = [
        json.loads(line)
        for line in (directory / "telemetry.jsonl").read_text().splitlines()
    ]
    assert len(records) == manifest["frame_count"] == manifest["frames_written"]
    assert [record["frame"] for record in records] == list(range(len(records)))
    timestamps = [record["simulation_timestamp_us"] for record in records]
    assert all(b > a for a, b in zip(timestamps, timestamps[1:]))
    for name in ("generated", "hdmap", "physics"):
        probed = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-count_frames",
                "-show_entries",
                "stream=nb_read_frames,width,height",
                "-of",
                "json",
                str(directory / f"{name}.mp4"),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        (stream,) = json.loads(probed.stdout)["streams"]
        assert int(stream["nb_read_frames"]) == len(records)
        assert (stream["width"], stream["height"]) == (
            manifest["width"],
            manifest["height"],
        )
    errors = [record["ground_error_m"] for record in records]
    assert all(error is not None and math.isfinite(error) for error in errors)
    speeds = [record["vehicle"]["speed_mps"] for record in records]
    contacts = sum(record["static_barrier_collision"] for record in records)
    ragdolls = sum(record["physics_vehicle"]["ragdoll_active"] for record in records)
    return {
        "frames": len(records),
        "final_speed_mps": speeds[-1],
        "max_ground_error_m": max(map(abs, errors)),
        "curb_contact_frames": contacts,
        "ragdoll_frames": ragdolls,
    }


def main() -> None:
    """Run a named script through the ordinary demo CLI and check its capture."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "scenario",
        choices=("raceway_start", "flat_acceleration", "curb_impact"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--slug", default="crazy-robotaxi-omnidreams-perf")
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--snapshot-every", type=int, default=30)
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    tests = Path(__file__).resolve().parent
    if args.scenario == "raceway_start":
        map_path = (
            tests.parent / "crazy_robotaxi/maps/flashdreams_raceway.robotaxi.yaml"
        )
        course = "grand-prix"
    else:
        map_path = tests / "maps/debug_flat.robotaxi.yaml"
        course = "debug-lap"
    command = [
        args.slug,
        "--mode",
        "robotaxi-debug",
        "--debug-output-dir",
        str(args.output_dir),
        "--debug-snapshot-every",
        str(args.snapshot_every),
        "--timeout",
        str(args.timeout),
        "--",
        "--map",
        str(map_path),
        "--game-mode",
        "race",
        "--race-course",
        course,
        "--drive-script",
        str(tests / "scenarios" / f"{args.scenario}.yaml"),
        "--seed",
        str(args.seed),
    ]
    if args.config is not None:
        command += ["--config", str(args.config)]
    entrypoint(command)
    summary = check_capture(args.output_dir.expanduser().resolve())
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
