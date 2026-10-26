# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Paired first-frame captures with identical straight-line driving inputs.

Run from the repository root with the model environment::

    python apps/crazy_robotaxi/tests/run_first_frame_comparison.py \
        --output-dir /tmp/robotaxi-first-frames --maps flashdreams_raceway

Omit ``--maps`` to compare all five bundled maps. ``--seeds 42 7`` repeats
each pair in fresh processes. Capture and control assertions do not establish
visual quality: inspect ``comparison.mp4`` for image-related drift.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
from ruamel.yaml import YAML

if __package__:
    from apps.crazy_robotaxi.tests.run_headless_scenario import check_capture
else:
    from run_headless_scenario import check_capture

MAP_NAMES = (
    "flashdreams_raceway",
    "boulevard_district",
    "pacific_breakwater",
    "harbor_market",
    "mesa_gardens",
)
_TESTS = Path(__file__).resolve().parent
_ROOT = _TESTS.parents[2]
_SCRIPT = _TESTS / "scenarios/first_frame_acceleration.yaml"
_REFERENCE = _ROOT / "apps/omnidreams_game_engine/omnidreams_game_engine/screenshot.jpg"


def prepare_pair(map_path: Path, reference: Path, directory: Path) -> dict[str, Path]:
    """Copy inputs and create map variants differing only in the first image.

    Disable traffic equally in both copies to exclude actor motion and impacts.
    Retain the authored geometry, spawn, and prompts. Never edit bundled assets.
    """
    directory.mkdir(parents=True, exist_ok=False)
    yaml = YAML(typ="safe")
    document = yaml.load(map_path.read_text())
    document["traffic_count"] = 0
    document.pop("traffic", None)
    for spawn in document["spawns"]:
        spawn["image"] = str((map_path.parent / spawn["image"]).resolve())
    candidate = Path(document["spawns"][0]["image"])
    paths = {}
    for label, image in (("reference", reference), ("candidate", candidate)):
        saved_image = directory / (label + image.suffix)
        shutil.copyfile(image, saved_image)
        variant = copy.deepcopy(document)
        variant["spawns"][0]["image"] = str(saved_image.resolve())
        paths[label] = directory / f"{label}.robotaxi.yaml"
        with paths[label].open("w") as stream:
            yaml.dump(variant, stream)
    # The app discovers every *.robotaxi.yaml sibling as a selectable map.
    shutil.copyfile(map_path, directory / "source.yaml")
    shutil.copyfile(_SCRIPT, directory / "drive.yaml")
    (directory / "settings.yaml").write_text("schema_version: 1\n")
    (directory / "sha256.json").write_text(
        json.dumps(
            {
                path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                for path in sorted(directory.iterdir())
                if path.is_file()
            },
            indent=2,
        )
    )
    return paths


def capture_command(
    slug: str, map_path: Path, output: Path, seed: int, timeout: float
) -> list[str]:
    """Build a fresh-process invocation of the ordinary headless application."""
    inputs = map_path.parent
    return [
        sys.executable,
        "-c",
        "from flashdreams.runtime_v2.cli import entrypoint; entrypoint()",
        slug,
        "--mode",
        "robotaxi-debug",
        "--debug-output-dir",
        str(output),
        "--debug-snapshot-every",
        "15",
        "--timeout",
        str(timeout),
        "--",
        "--map",
        str(map_path),
        "--game-mode",
        "free-roam",
        "--drive-script",
        str(inputs / "drive.yaml"),
        "--config",
        str(inputs / "settings.yaml"),
        "--width",
        "1280",
        "--height",
        "704",
        "--seed",
        str(seed),
        "--no-visual-flare",
        "--no-live-edit-dynamic-prompts",
        "--no-live-edit-style",
        "--no-live-edit-weather",
        "--no-live-edit-items",
        "--no-live-edit-coins",
        "--no-live-edit-obstacle",
    ]


def check_straight_motion(records: list[dict]) -> dict[str, float]:
    """Reject steering, impacts, or a non-straight/non-accelerating trajectory."""
    assert len(records) == 105, "Expected the complete stationary/acceleration script"
    for index, record in enumerate(records):
        command = record["command"]
        assert command["steer"] == 0, "Steering confounds the image comparison"
        assert command["throttle"] == (0 if index < 15 else 1)
        assert not any(
            command[key] for key in ("brake", "handbrake", "reverse", "stop")
        )
        assert not record["actor_collision"], "Actor collision confounds the comparison"
        assert not record["static_barrier_collision"], (
            "Curb response confounds the comparison"
        )
        assert not record["physics_vehicle"]["ragdoll_active"]
    vehicles = [record["vehicle"] for record in records]
    yaw = np.asarray([vehicle["yaw_rad"] for vehicle in vehicles])
    delta_yaw = np.arctan2(np.sin(yaw - yaw[0]), np.cos(yaw - yaw[0]))
    xy = np.asarray([[vehicle["x_m"], vehicle["y_m"]] for vehicle in vehicles])
    displacement = xy - xy[0]
    forward = displacement @ [math.cos(yaw[0]), math.sin(yaw[0])]
    lateral = displacement @ [-math.sin(yaw[0]), math.cos(yaw[0])]
    speeds = np.asarray([vehicle["speed_mps"] for vehicle in vehicles])
    assert np.isfinite(np.concatenate((yaw, xy.ravel(), speeds))).all()
    # Small numerical/physics motion is permitted; visible turns are not.
    assert np.max(np.abs(delta_yaw)) < 0.01, "Vehicle turned despite zero steering"
    assert np.max(np.abs(lateral)) < 0.1, "Vehicle departed the straight trajectory"
    assert np.max(np.abs(speeds[:15])) < 0.05, "Stationary prefix moved"
    assert forward[-1] > 1 and speeds[-1] > speeds[0] + 1, "Vehicle did not accelerate"
    return {
        "distance_forward_m": float(forward[-1]),
        "max_lateral_displacement_m": float(np.max(np.abs(lateral))),
        "max_yaw_change_rad": float(np.max(np.abs(delta_yaw))),
        "final_speed_mps": float(speeds[-1]),
    }


def check_matching_records(reference: list[dict], candidate: list[dict]) -> None:
    """Require the same controls, timestamps, camera poses, and speeds in a pair."""
    assert len(reference) == len(candidate), "Capture lengths differ"
    for first, second in zip(reference, candidate, strict=True):
        for key in ("frame", "simulation_timestamp_us", "command"):
            assert first[key] == second[key], (
                f"Mismatched {key} at frame {first['frame']}"
            )
        np.testing.assert_allclose(
            first["rig_pose_world"],
            second["rig_pose_world"],
            rtol=0,
            atol=1e-5,
            equal_nan=False,
            err_msg=f"Camera trajectories differ at frame {first['frame']}",
        )
        assert (
            abs(first["vehicle"]["speed_mps"] - second["vehicle"]["speed_mps"]) < 1e-5
        )
        for key, value in first["physics_vehicle"].items():
            other = second["physics_vehicle"][key]
            if value is None or isinstance(value, bool):
                assert value == other, f"Physical state differs: {key}"
            else:
                assert other is not None and abs(value - other) < 1e-5, key


def decoded_video_hash(path: Path) -> str:
    """Hash decoded pixels, excluding MP4 metadata and encoding differences."""
    return subprocess.check_output(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(path),
            "-map",
            "0:v:0",
            "-f",
            "hash",
            "-hash",
            "sha256",
            "-",
        ],
        text=True,
    ).strip()


def write_comparison(directory: Path) -> None:
    """Show reference/candidate output above their shared HD map and physics."""
    sources = (
        ("reference/generated.mp4", "Original screenshot"),
        ("candidate/generated.mp4", "Map initial frame"),
        ("reference/hdmap.mp4", "Shared HDmap conditioning"),
        ("reference/physics.mp4", "Shared physics"),
    )
    command = ["ffmpeg", "-y", "-v", "error"]
    filters = []
    for index, (path, label) in enumerate(sources):
        command += ["-i", str(directory / path)]
        filters.append(
            f"[{index}:v]scale=640:352,pad=640:384:0:32:black,"
            f"drawtext=text='{label}':x=12:y=8:fontsize=18:fontcolor=white[v{index}]"
        )
    filters += [
        "[v0][v1]hstack[top]",
        "[v2][v3]hstack[bottom]",
        "[top][bottom]vstack[out]",
    ]
    subprocess.run(
        command
        + [
            "-filter_complex",
            ";".join(filters),
            "-map",
            "[out]",
            "-c:v",
            "libx264",
            "-crf",
            "20",
            "-threads",
            "2",
            "-movflags",
            "+faststart",
            str(directory / "comparison.mp4"),
        ],
        check=True,
    )


def main() -> None:
    """Run paired image comparisons, preserving evidence even when a pair fails."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--maps", nargs="+", choices=MAP_NAMES, default=list(MAP_NAMES))
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--slug", default="crazy-robotaxi-omnidreams")
    parser.add_argument("--timeout", type=float, default=600)
    args = parser.parse_args()
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    report: dict[str, Any] = {
        "commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=_ROOT, text=True
        ).strip(),
        "slug": args.slug,
        "visual_quality": "unscored; inspect comparison videos",
        "traffic": "disabled equally in both test copies",
        "pairs": [],
    }
    for name in dict.fromkeys(args.maps):
        for seed in dict.fromkeys(args.seeds):
            directory = output / name / f"seed-{seed}"
            row: dict[str, Any] = {
                "map": name,
                "seed": seed,
                "capture_checks": "failed",
            }
            report["pairs"].append(row)
            try:
                maps = prepare_pair(
                    _ROOT
                    / f"apps/crazy_robotaxi/crazy_robotaxi/maps/{name}.robotaxi.yaml",
                    _REFERENCE,
                    directory / "inputs",
                )
                records = {}
                manifests = {}
                for label, map_path in maps.items():
                    capture = directory / label
                    command = capture_command(
                        args.slug, map_path, capture, seed, args.timeout
                    )
                    (directory / f"{label}-command.json").write_text(
                        json.dumps(command, indent=2)
                    )
                    print(f"Running {name} seed={seed} {label}", flush=True)
                    with (directory / f"{label}.log").open("w") as log:
                        subprocess.run(
                            command,
                            cwd=_ROOT,
                            stdout=log,
                            stderr=subprocess.STDOUT,
                            check=True,
                            timeout=args.timeout + 120,
                        )
                    check_capture(capture)
                    records[label] = [
                        json.loads(line)
                        for line in (capture / "telemetry.jsonl")
                        .read_text()
                        .splitlines()
                    ]
                    row[label] = check_straight_motion(records[label])
                    manifests[label] = json.loads(
                        (capture / "manifest.json").read_text()
                    )
                for key in (
                    "model_seed",
                    "game_seed",
                    "model_preset",
                    "fps",
                    "width",
                    "height",
                    "frame_count",
                ):
                    assert manifests["reference"][key] == manifests["candidate"][key], (
                        key
                    )
                check_matching_records(records["reference"], records["candidate"])
                reference_hash = decoded_video_hash(directory / "reference/hdmap.mp4")
                assert reference_hash == decoded_video_hash(
                    directory / "candidate/hdmap.mp4"
                ), "HD-map conditioning differs"
                row["hdmap_sha256"] = reference_hash
                write_comparison(directory)
                row["capture_checks"] = "passed"
                row["comparison"] = str(
                    (directory / "comparison.mp4").relative_to(output)
                )
            except (
                AssertionError,
                OSError,
                ValueError,
                subprocess.SubprocessError,
            ) as error:
                row["error"] = str(error)
                print(f"FAILED {name} seed={seed}: {error}", flush=True)
            finally:
                (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    if any(row["capture_checks"] != "passed" for row in report["pairs"]):
        raise SystemExit(f"Comparison checks failed; inspect {output / 'report.json'}")
    print(
        f"Capture/control checks passed. Review visual drift in {output / 'report.json'}"
    )


if __name__ == "__main__":
    main()
