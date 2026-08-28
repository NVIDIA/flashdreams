# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for LingBot benchmark plotting."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.benchmark.lingbot._plot import plot_modules, plot_pipeline

pytestmark = pytest.mark.ci_cpu


def _record(group: str, implementation: str, median: float) -> dict[str, object]:
    return {
        "name": f"{group}[{implementation}]",
        "group": group,
        "param": implementation,
        "stats": {"median": median},
    }


def _write_results(path: Path, records: list[dict[str, object]]) -> None:
    path.write_text(
        json.dumps(
            {
                "benchmarks": records,
                "commit_info": {"id": "0123456789abcdef"},
                "datetime": "2026-08-26T12:00:00",
            }
        ),
        encoding="utf-8",
    )


def test_plot_representative_modules_and_pipeline(tmp_path: Path) -> None:
    module_input = tmp_path / "module.json"
    pipeline_input = tmp_path / "pipeline.json"
    output_dir = tmp_path / "plots"
    attention_implementation = "optimized-fa2-full-tma"
    _write_results(
        module_input,
        [
            _record(group, implementation, median)
            for group in (
                "lingbot-dit-self-attention",
                "lingbot-dit-cross-attention",
            )
            for implementation, median in (
                ("torch", 0.010),
                (attention_implementation, 0.006),
            )
        ]
        + [
            _record("lingbot-dit-block", "torch", 0.100),
            _record("lingbot-dit-block", "optimized-self-cross", 0.060),
        ],
    )
    _write_results(
        pipeline_input,
        [
            _record(group, implementation, median)
            for group in (
                "lingbot-dit-network",
                "lingbot-full-pipeline-generate",
            )
            for implementation, median in (
                ("torch", 1.0),
                ("optimized-self-cross", 0.7),
            )
        ],
    )

    plot_modules(module_input, output_dir)
    plot_pipeline(pipeline_input, output_dir)

    assert (output_dir / "modules.png").is_file()
    assert (output_dir / "dit_block.png").is_file()
    assert (output_dir / "network_pipeline.png").is_file()


def test_plot_full_block_search_as_heatmap(tmp_path: Path) -> None:
    input_path = tmp_path / "module.json"
    output_dir = tmp_path / "plots"
    optimized = "optimized-cudnn-fuse-kv-no-tma"
    _write_results(
        input_path,
        [
            _record(group, "torch", 0.010)
            for group in (
                "lingbot-dit-self-attention",
                "lingbot-dit-cross-attention",
            )
        ]
        + [
            _record("lingbot-dit-block", "self-torch-cross-torch", 0.100),
            _record(
                "lingbot-dit-block",
                f"self-{optimized}-cross-{optimized}",
                0.060,
            ),
        ],
    )

    plot_modules(input_path, output_dir)

    assert (output_dir / "dit_block.png").is_file()
