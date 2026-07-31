# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for independent aggregated LingBot benchmark reporting."""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import pytest
from lingbot.disagg.benchmark_independent import _child_command, _summarize

pytestmark = pytest.mark.ci_cpu


def _worker_document(
    *,
    started_at: float,
    finished_at: float,
    latency_ms: float,
    fps: float,
    memory_gib: float,
) -> dict:
    return {
        "environment": {
            "measurement_window": {
                "started_at": started_at,
                "finished_at": finished_at,
            }
        },
        "summary": {
            "fps": fps,
            "latency_ms": {"median": latency_ms},
            "memory": {
                "peak_gib_by_rank": [memory_gib],
                "initialization_peak_gib_by_rank": [memory_gib + 5.0],
                "steady_allocated_gib_by_rank": [memory_gib - 2.0],
            },
        },
        "records": [
            {
                "warmup": False,
                "output_frames": 12,
                "end_to_end_ms": latency_ms,
            }
        ],
    }


def test_summarize_uses_shared_measurement_window() -> None:
    summary = _summarize(
        [
            _worker_document(
                started_at=10.0,
                finished_at=12.0,
                latency_ms=2000.0,
                fps=6.0,
                memory_gib=60.0,
            ),
            _worker_document(
                started_at=10.01,
                finished_at=12.01,
                latency_ms=2010.0,
                fps=5.97,
                memory_gib=61.0,
            ),
        ]
    )

    assert summary["aggregate_fps"] == pytest.approx(24 / 2.01)
    assert summary["sum_of_worker_fps"] == pytest.approx(11.97)
    assert summary["measurement_start_skew_ms"] == pytest.approx(10.0)
    assert summary["all_chunk_latency_ms"]["median"] == pytest.approx(2005.0)
    assert summary["memory"]["rollout_peak_gib_node_total"] == 121.0
    assert summary["memory"]["initialization_peak_gib_node_total"] == 131.0


def test_child_command_launches_one_isolated_rank(tmp_path: Path) -> None:
    args = Namespace(
        model="lingbot-world-fast-taehv-window15-sink3",
        example_idx=0,
        warmup_blocks=6,
        measured_blocks=5,
        pixel_height=464,
        pixel_width=832,
        fps=16,
        output_dir=tmp_path,
    )

    command = _child_command(
        args,
        replica_id=3,
        barrier_dir=tmp_path / "barrier",
        output_dir=tmp_path / "worker-3",
    )

    assert "--nproc_per_node=1" in command
    assert command[command.index("--replica-id") + 1] == "3"
    assert command[command.index("--pixel-height") + 1] == "464"
