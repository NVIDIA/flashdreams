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

"""Isolated-process runner for the optional torchao output-projection experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any


def main() -> None:
    """Run a bounded comparison and preserve successful and failed measurements."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--torchao-eager-numerics",
        action="store_true",
        help="Use local cast/division rounding options for compiled torchao cases.",
    )
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        parser.error(
            "output must be empty to preserve evidence and cold-cache isolation"
        )
    output.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[4]
    target = Path(__file__).with_name("test_quantized_linear_benchmark.py")
    manifest_path = output / "manifest.json"
    # Preserve the working implementation as well as the base commit.
    (output / "source.patch").write_bytes(
        subprocess.check_output(["git", "diff", "HEAD"], cwd=root)
    )
    (output / "runner.py").write_bytes(Path(__file__).read_bytes())
    manifest = {
        "command": sys.argv,
        "commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "diff_sha256": hashlib.sha256(
            subprocess.check_output(["git", "diff", "HEAD"], cwd=root)
        ).hexdigest(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "runs": [],
    }
    (output / "environment.txt").write_text(
        subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
        if subprocess.run(
            [sys.executable, "-m", "pip", "--version"], capture_output=True
        ).returncode
        == 0
        else subprocess.check_output(
            ["uv", "pip", "freeze", "--python", sys.executable], text=True
        )
    )
    (output / "nvidia-smi.txt").write_text(
        subprocess.check_output(["nvidia-smi"], text=True)
    )
    summaries = []
    for repeat in range(args.repeats):
        for backend in ("bf16", "flashdreams", "torchao"):
            for mode in ("eager", "compile", "graph", "compile_graph"):
                states = ("cold", "warm") if "compile" in mode else ("fresh",)
                cache = output / "cache" / f"{repeat}-{backend}-{mode}"
                for state in states:
                    label = f"{repeat}-{backend}-{mode}-{state}"
                    temporary = output / "tmp" / label
                    temporary.mkdir(parents=True)
                    env = dict(os.environ)
                    env.update(
                        {
                            "PYTHONPATH": str(root / "flashdreams"),
                            "TORCHINDUCTOR_CACHE_DIR": str(cache / "inductor"),
                            "TRITON_CACHE_DIR": str(cache / "triton"),
                            "CUDA_CACHE_PATH": str(cache / "cuda"),
                            "FLASHDREAMS_BENCH_CACHE_STATE": state,
                            "FLASHDREAMS_TORCHAO_EAGER_NUMERICS": str(
                                int(args.torchao_eager_numerics)
                            ),
                            "TMPDIR": str(temporary),
                            "TORCH_LOGS": "graph_breaks,recompiles",
                            "OMP_NUM_THREADS": "8",
                        }
                    )
                    record: dict[str, Any] = {
                        "label": label,
                        "repeat": repeat,
                        "backend": backend,
                        "mode": mode,
                        "cache_state": state,
                    }
                    command = [
                        sys.executable,
                        "-m",
                        "pytest",
                        f"{target}::test_torchao_comparison[{mode}-{backend}]",
                        "-p",
                        "no:manual_marker",
                        "-m",
                        "manual",
                        "--benchmark-only",
                        "--benchmark-save-data",
                        "--benchmark-json",
                        str(output / f"{label}.json"),
                        "-q",
                    ]
                    record["command"] = command
                    print(f"Running {label}", flush=True)
                    with (output / f"{label}.log").open("w") as log:
                        try:
                            result = subprocess.run(
                                command,
                                cwd=root,
                                env=env,
                                stdout=log,
                                stderr=subprocess.STDOUT,
                                timeout=1200,
                            )
                            record["returncode"] = result.returncode
                        except subprocess.TimeoutExpired:
                            record["returncode"] = "timeout"
                    manifest["runs"].append(record)
                    manifest_path.write_text(json.dumps(manifest, indent=2))
                    result_file = output / f"{label}.json"
                    try:
                        rows = json.loads(result_file.read_text())["benchmarks"]
                        if not isinstance(rows, list) or len(rows) != 1:
                            raise ValueError("expected one benchmark row")
                        info, stats = rows[0]["extra_info"], rows[0]["stats"]
                        if info["backend"] != backend or info["execution"] != mode:
                            raise ValueError("benchmark backend or execution mismatch")
                        samples = stats["data"]
                        if not isinstance(samples, list) or len(samples) != 50:
                            raise ValueError("expected 50 timing samples")
                        for key, value in (
                            ("stats.median", stats["median"]),
                            *(("timing sample", value) for value in samples),
                        ):
                            if (
                                type(value) not in (int, float)
                                or not math.isfinite(value)
                                or value <= 0
                            ):
                                raise ValueError(f"{key} must be finite and positive")
                        for key in (
                            "cuda_event_median_ms",
                            "preparation_ms",
                            "first_call_ms",
                            "capture_first_call_ms",
                            "steady_peak_allocated_bytes",
                            "relative_l2",
                        ):
                            value = info[key]
                            if (
                                type(value) not in (int, float)
                                or not math.isfinite(value)
                                or value < 0
                            ):
                                raise ValueError(
                                    f"{key} must be finite and nonnegative"
                                )
                        raw_ms = sorted(value * 1000 for value in samples)
                    except (
                        FileNotFoundError,
                        KeyError,
                        TypeError,
                        ValueError,
                        OverflowError,
                    ) as exc:
                        record["artifact_error"] = f"invalid benchmark export: {exc}"
                    else:
                        info.update(
                            {
                                "label": label,
                                "returncode": record["returncode"],
                                "wall_median_ms": statistics.median(raw_ms),
                                "wall_p90_ms": raw_ms[int(0.9 * (len(raw_ms) - 1))],
                                "rows_per_second": 4800 / stats["median"],
                            }
                        )
                        summaries.append(info)
                        (output / "measurements.json").write_text(
                            json.dumps(summaries, indent=2)
                        )
                    lines = [
                        "# Component measurements",
                        "",
                        "Times are milliseconds; memory is MiB. See manifest and per-run logs for failures.",
                        "",
                        "| Run | status | wall median | wall p90 | CUDA median | prepare | first call | capture+replay | peak allocated | relative L2 |",
                        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
                    ]
                    for item in summaries:
                        values = [
                            item[key]
                            for key in (
                                "wall_median_ms",
                                "wall_p90_ms",
                                "cuda_event_median_ms",
                                "preparation_ms",
                                "first_call_ms",
                                "capture_first_call_ms",
                            )
                        ]
                        lines.append(
                            "| "
                            + item["label"]
                            + (
                                " | pass | "
                                if item["returncode"] == 0
                                else " | FAIL | "
                            )
                            + " | ".join(f"{value:.4f}" for value in values)
                            + f" | {item['steady_peak_allocated_bytes'] / 2**20:.2f} | {item['relative_l2']:.6f} |"
                        )
                    manifest_path.write_text(json.dumps(manifest, indent=2))
                    (output / "summary.md").write_text("\n".join(lines) + "\n")
    failures = [
        run
        for run in manifest["runs"]
        if run["returncode"] != 0 or "artifact_error" in run
    ]
    print(
        f"Completed {len(manifest['runs'])} processes; {len(failures)} failed. Results: {output}",
        flush=True,
    )
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
