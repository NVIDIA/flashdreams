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

"""Concurrent independent-GPU LingBot benchmark coordinator."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from lingbot.config import PIPELINE_CONFIGS
from lingbot.disagg.benchmark import _metric_summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replicas", type=int, default=8)
    parser.add_argument(
        "--model",
        choices=sorted(PIPELINE_CONFIGS),
        default="lingbot-world-fast-taehv-window15-sink3",
    )
    parser.add_argument("--example-idx", type=int, default=0)
    parser.add_argument("--warmup-blocks", type=int, default=6)
    parser.add_argument("--measured-blocks", type=int, default=5)
    parser.add_argument("--pixel-height", type=int, default=464)
    parser.add_argument("--pixel-width", type=int, default=832)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--compile-threads-per-replica", type=int, default=4)
    parser.add_argument("--timeout-s", type=float, default=3600.0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/lingbot_aggregated_8xcp1"),
    )
    return parser.parse_args()


def _child_command(
    args: argparse.Namespace,
    *,
    replica_id: int,
    barrier_dir: Path,
    output_dir: Path,
) -> list[str]:
    """Build one isolated CP1 worker command.

    Args:
        args: Coordinator arguments.
        replica_id: GPU and session index.
        barrier_dir: Shared post-warmup synchronization directory.
        output_dir: Worker-specific result directory.

    Returns:
        Argument vector for one ``torchrun`` child process.
    """
    return [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node=1",
        "-m",
        "lingbot.disagg.benchmark_aggregated",
        "--model",
        args.model,
        "--example-idx",
        str(args.example_idx),
        "--warmup-blocks",
        str(args.warmup_blocks),
        "--measured-blocks",
        str(args.measured_blocks),
        "--pixel-height",
        str(args.pixel_height),
        "--pixel-width",
        str(args.pixel_width),
        "--fps",
        str(args.fps),
        "--cp-method",
        "ulysses",
        "--bandwidth-probe-mib",
        "1",
        "--bandwidth-probe-iters",
        "1",
        "--comparison-json",
        str(args.output_dir / "no-comparison.json"),
        "--replica-id",
        str(replica_id),
        "--measurement-barrier-dir",
        str(barrier_dir),
        "--output-dir",
        str(output_dir),
    ]


def _summarize(worker_documents: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize independent workers over their shared measurement window.

    Args:
        worker_documents: Raw ``benchmark.json`` documents from every worker.

    Returns:
        Aggregate throughput, latency, memory, and synchronization metrics.
    """
    starts = [
        document["environment"]["measurement_window"]["started_at"]
        for document in worker_documents
    ]
    finishes = [
        document["environment"]["measurement_window"]["finished_at"]
        for document in worker_documents
    ]
    measured_records = [
        record
        for document in worker_documents
        for record in document["records"]
        if not record["warmup"]
    ]
    total_frames = sum(record["output_frames"] for record in measured_records)
    measurement_wall_s = max(finishes) - min(starts)
    worker_fps = [document["summary"]["fps"] for document in worker_documents]
    worker_latency_medians = [
        document["summary"]["latency_ms"]["median"] for document in worker_documents
    ]
    rollout_peak = [
        document["summary"]["memory"]["peak_gib_by_rank"][0]
        for document in worker_documents
    ]
    initialization_peak = [
        document["summary"]["memory"]["initialization_peak_gib_by_rank"][0]
        for document in worker_documents
    ]
    steady = [
        document["summary"]["memory"]["steady_allocated_gib_by_rank"][0]
        for document in worker_documents
    ]
    return {
        "aggregate_fps": total_frames / measurement_wall_s,
        "sum_of_worker_fps": sum(worker_fps),
        "total_output_frames": total_frames,
        "measurement_wall_s": measurement_wall_s,
        "measurement_start_skew_ms": (max(starts) - min(starts)) * 1000.0,
        "measurement_finish_skew_ms": (max(finishes) - min(finishes)) * 1000.0,
        "per_session_fps": _metric_summary(worker_fps),
        "per_session_median_latency_ms": _metric_summary(worker_latency_medians),
        "all_chunk_latency_ms": _metric_summary(
            [record["end_to_end_ms"] for record in measured_records]
        ),
        "memory": {
            "rollout_peak_gib_by_gpu": rollout_peak,
            "rollout_peak_gib_per_gpu": _metric_summary(rollout_peak),
            "rollout_peak_gib_node_total": sum(rollout_peak),
            "initialization_peak_gib_by_gpu": initialization_peak,
            "initialization_peak_gib_per_gpu": _metric_summary(initialization_peak),
            "initialization_peak_gib_node_total": sum(initialization_peak),
            "steady_allocated_gib_by_gpu": steady,
            "steady_allocated_gib_node_total": sum(steady),
        },
    }


def _write_report(
    args: argparse.Namespace,
    *,
    worker_documents: list[dict[str, Any]],
    worker_commands: list[list[str]],
    summary: dict[str, Any],
) -> None:
    """Write machine-readable and Markdown coordinator reports.

    Args:
        args: Coordinator arguments.
        worker_documents: Raw result from each independent worker.
        worker_commands: Exact subprocess argument vectors.
        summary: Aggregate benchmark metrics.
    """
    first_environment = worker_documents[0]["environment"]
    command = shlex.join(
        [
            "uv",
            "run",
            "--package",
            "flashdreams-lingbot",
            "python",
            "-m",
            "lingbot.disagg.benchmark_independent",
            *sys.argv[1:],
        ]
    )
    environment = {
        "command": command,
        "commit": first_environment["commit"],
        "worktree_dirty": first_environment["worktree_dirty"],
        "hostname": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "model": args.model,
        "resolution": [args.pixel_height, args.pixel_width],
        "warmup_blocks": args.warmup_blocks,
        "measured_blocks": args.measured_blocks,
        "replicas": args.replicas,
        "gpus": [document["environment"]["gpus"][0] for document in worker_documents],
        "torch": first_environment["torch"],
        "cuda": first_environment["cuda"],
        "cudnn": first_environment["cudnn"],
        "driver": first_environment["driver"],
        "precision": first_environment["precision"],
        "checkpoint": first_environment["checkpoint"],
        "worker_commands": [shlex.join(item) for item in worker_commands],
    }
    document = {
        "environment": environment,
        "summary": summary,
        "workers": worker_documents,
    }
    (args.output_dir / "benchmark.json").write_text(
        json.dumps(document, indent=2) + "\n"
    )

    memory = summary["memory"]
    markdown = f"""# LingBot eight independent aggregated workers

Eight H100s each own one complete CP1 encoder + DiT + decoder pipeline and one
session. The workers synchronize after warmup and run their five measured
chunks concurrently.

## Result

| Metric | Value |
| --- | ---: |
| Aggregate generated FPS | **{summary["aggregate_fps"]:.2f}** |
| Per-session FPS, median / p90 | {summary["per_session_fps"]["median"]:.2f} / {summary["per_session_fps"]["p90"]:.2f} |
| Chunk latency, median / p90 | {summary["all_chunk_latency_ms"]["median"]:.2f} / {summary["all_chunk_latency_ms"]["p90"]:.2f} ms |
| Shared measurement wall time | {summary["measurement_wall_s"]:.3f} s |
| Measurement start skew | {summary["measurement_start_skew_ms"]:.2f} ms |
| Rollout peak HBM per GPU, min–max | {memory["rollout_peak_gib_per_gpu"]["min"]:.2f}–{memory["rollout_peak_gib_per_gpu"]["max"]:.2f} GiB |
| Initialization peak HBM per GPU, min–max | {memory["initialization_peak_gib_per_gpu"]["min"]:.2f}–{memory["initialization_peak_gib_per_gpu"]["max"]:.2f} GiB |
| Rollout peak HBM, node total | {memory["rollout_peak_gib_node_total"]:.2f} GiB |

## Reproduction

```bash
{command}
```

- Repository revision: `{environment["commit"]}`{" (modified worktree)" if environment["worktree_dirty"] else ""}
- Slurm: job `{environment["slurm_job_id"]}` on `{environment["hostname"]}`
- GPU: `{environment["gpus"][0]}` × {len(environment["gpus"])}
- Resolution: `{args.pixel_width}x{args.pixel_height}`
- Warmup / measured blocks per worker: {args.warmup_blocks} / {args.measured_blocks}
"""
    (args.output_dir / "README.md").write_text(markdown)


def main() -> None:
    """Launch independent CP1 workers and summarize concurrent serving."""
    args = _parse_args()
    if args.replicas <= 0:
        raise ValueError("replicas must be positive.")
    if args.replicas > 8:
        raise ValueError("replicas cannot exceed the eight GPUs in one node.")

    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    barrier_dir = args.output_dir / "barrier"
    barrier_dir.mkdir(exist_ok=True)
    for stale_file in barrier_dir.iterdir():
        stale_file.unlink()

    processes: list[subprocess.Popen[bytes]] = []
    log_files: list[Any] = []
    worker_commands: list[list[str]] = []
    try:
        for replica_id in range(args.replicas):
            worker_dir = args.output_dir / f"worker-{replica_id}"
            worker_dir.mkdir(exist_ok=True)
            command = _child_command(
                args,
                replica_id=replica_id,
                barrier_dir=barrier_dir,
                output_dir=worker_dir,
            )
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = str(replica_id)
            environment["TORCHINDUCTOR_COMPILE_THREADS"] = str(
                args.compile_threads_per_replica
            )
            log_file = (worker_dir / "run.log").open("wb")
            log_files.append(log_file)
            worker_commands.append(command)
            processes.append(
                subprocess.Popen(
                    command,
                    env=environment,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                )
            )

        deadline = time.monotonic() + args.timeout_s
        ready_files = [barrier_dir / f"ready-{index}" for index in range(args.replicas)]
        while not all(path.is_file() for path in ready_files):
            failed = [
                (index, process.returncode)
                for index, process in enumerate(processes)
                if process.poll() is not None and process.returncode != 0
            ]
            if failed:
                raise RuntimeError(f"Workers failed before measurement: {failed}.")
            if time.monotonic() >= deadline:
                raise TimeoutError("Independent workers did not finish warmup in time.")
            time.sleep(0.1)

        (barrier_dir / "release").write_text("release\n")
        return_codes = [process.wait(timeout=args.timeout_s) for process in processes]
        failed = [
            (index, return_code)
            for index, return_code in enumerate(return_codes)
            if return_code != 0
        ]
        if failed:
            raise RuntimeError(f"Workers failed during measurement: {failed}.")
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for log_file in log_files:
            log_file.close()

    worker_documents = [
        json.loads((args.output_dir / f"worker-{index}" / "benchmark.json").read_text())
        for index in range(args.replicas)
    ]
    summary = _summarize(worker_documents)
    _write_report(
        args,
        worker_documents=worker_documents,
        worker_commands=worker_commands,
        summary=summary,
    )


if __name__ == "__main__":
    main()
