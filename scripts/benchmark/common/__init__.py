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

"""Shared parsing and rendering helpers for benchmark plots."""

from __future__ import annotations

import argparse
import json
import math
import os
from collections.abc import Hashable, Mapping, Sequence
from pathlib import Path
from typing import TypeVar

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import CenteredNorm
from matplotlib.figure import Figure
from matplotlib.image import AxesImage
from matplotlib.text import Text
from matplotlib.ticker import FuncFormatter

RowKey = TypeVar("RowKey", bound=Hashable)


def benchmark_artifact_dir(default: Path) -> Path:
    """Select the default or full-benchmark artifact directory."""
    return (
        default / "full"
        if os.environ.get("FLASHDREAMS_RUN_FULL_BENCHMARK") == "1"
        else default
    )


def add_plot_io_arguments(
    parser: argparse.ArgumentParser,
    default_input: Path,
    default_output_dir: Path,
) -> None:
    """Add the standard benchmark input and plot output arguments."""
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        default=default_input,
        help=f"pytest-benchmark JSON path (default: {default_input})",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=default_output_dir,
        help=f"output directory (default: {default_output_dir})",
    )


def load_benchmark_json(input_path: Path) -> tuple[dict[str, object], list[object]]:
    """Load a pytest-benchmark JSON object and its benchmark records."""
    try:
        payload = json.loads(input_path.read_text(encoding="utf-8"))
    except OSError as error:
        raise SystemExit(f"Cannot read benchmark JSON {input_path}: {error}") from error
    except json.JSONDecodeError as error:
        raise SystemExit(
            f"Cannot parse benchmark JSON {input_path}: {error}"
        ) from error

    if not isinstance(payload, dict) or not isinstance(payload.get("benchmarks"), list):
        raise SystemExit(f"Benchmark JSON {input_path} has no benchmarks list")
    return payload, payload["benchmarks"]


def benchmark_median_ms(record: object) -> float:
    """Return one benchmark's positive finite median in milliseconds."""
    if not isinstance(record, dict):
        raise SystemExit("Benchmark <unnamed> has no positive median")

    stats = record.get("stats")
    median = stats.get("median") if isinstance(stats, dict) else None
    if (
        isinstance(median, bool)
        or not isinstance(median, (int, float))
        or not math.isfinite(median)
        or median <= 0
    ):
        raise SystemExit(
            f"Benchmark {record.get('name', '<unnamed>')} has no positive median"
        )
    return median * 1000.0


def benchmark_subtitle(payload: Mapping[str, object]) -> str:
    """Build the standard benchmark-environment subtitle."""
    parts = ["Median latency in ms (lower is faster)"]
    commit_info = payload.get("commit_info")
    commit_id = commit_info.get("id") if isinstance(commit_info, dict) else None
    if isinstance(commit_id, str) and commit_id:
        parts.append(f"commit {commit_id[:10]}")
    timestamp = payload.get("datetime")
    if isinstance(timestamp, str) and timestamp:
        parts.append(timestamp)
    return " · ".join(parts)


def latency_comparison_label(
    value: float | None,
    reference: float | None,
    *,
    is_reference: bool,
    precision: int = 2,
) -> str:
    """Format latency and multiplicative performance against a row reference."""
    if value is None:
        return "N/A"
    latency = f"{value:.{precision}f} ms"
    if is_reference:
        return f"{latency}\n1.00× reference"
    if reference is None:
        return f"{latency}\nreference unavailable"
    if value < reference:
        return f"{latency}\n{reference / value:.2f}× faster"
    if value > reference:
        return f"{latency}\n{value / reference:.2f}× slower"
    return f"{latency}\nsame as reference"


def relative_matrix_with_fastest(
    rows: Sequence[RowKey],
    columns: Sequence[str],
    values_ms: Mapping[tuple[RowKey, str], float],
    reference_column: str,
) -> tuple[list[list[float]], dict[RowKey, str | None]]:
    """Build relative values plus each row's fastest non-reference result."""
    matrix: list[list[float]] = []
    fastest_by_row: dict[RowKey, str | None] = {}
    for row in rows:
        reference = values_ms.get((row, reference_column))
        fastest = min(
            (
                column
                for column in columns
                if column != reference_column and (row, column) in values_ms
            ),
            key=lambda column: values_ms[(row, column)],
            default=None,
        )
        fastest_by_row[row] = fastest
        relative_values = [
            values_ms.get((row, column), math.nan) / reference
            if reference is not None
            else math.nan
            for column in columns
        ]
        relative_values.append(
            values_ms[(row, fastest)] / reference
            if reference is not None and fastest is not None
            else math.nan
        )
        matrix.append(relative_values)
    return matrix, fastest_by_row


def draw_relative_heatmap(axes: Axes, matrix: Sequence[Sequence[float]]) -> AxesImage:
    """Draw runtime ratios in log2 space centered on the reference value."""
    ratios = np.asarray(matrix, dtype=float)
    log2_ratios = np.full_like(ratios, np.nan)
    np.log2(
        ratios,
        out=log2_ratios,
        where=np.isfinite(ratios) & (ratios > 0),
    )
    return axes.imshow(
        log2_ratios,
        aspect="auto",
        cmap="RdYlGn_r",
        norm=CenteredNorm(vcenter=0.0),
    )


def relative_text_color(image: AxesImage, relative_value: float) -> str:
    """Choose readable annotation text for a heatmap cell."""
    if not math.isfinite(relative_value) or relative_value <= 0:
        return "black"
    colors = image.cmap(image.norm(np.asarray([math.log2(relative_value)])))
    red, green, blue, _ = np.asarray(colors)[0]
    linear = tuple(
        channel / 12.92 if channel <= 0.04045 else ((channel + 0.055) / 1.055) ** 2.4
        for channel in (red, green, blue)
    )
    luminance = 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]
    return "white" if luminance < 0.179 else "black"


def annotate_relative_cell(
    axes: Axes,
    image: AxesImage,
    row_index: int,
    column_index: int,
    label: str,
    relative_value: float,
) -> Text:
    """Add a centered heatmap annotation with contrast-aware text."""
    return axes.text(
        column_index,
        row_index,
        label,
        ha="center",
        va="center",
        color=relative_text_color(image, relative_value),
    )


def add_relative_colorbar(
    figure: Figure,
    axes: Axes,
    image: AxesImage,
    label: str,
    *,
    multiline_directions: bool = False,
) -> None:
    """Add the standard relative-runtime colorbar and direction labels."""
    colorbar = figure.colorbar(
        image,
        ax=axes,
        label=label,
        format=FuncFormatter(lambda value, _position: f"{2**value:.3g}×"),
    )
    slower = "Slower\n↑" if multiline_directions else "Slower ↑"
    faster = "↓\nFaster" if multiline_directions else "↓ Faster"
    colorbar.ax.text(0.5, 1.02, slower, ha="center", transform=colorbar.ax.transAxes)
    colorbar.ax.text(
        0.5,
        -0.04,
        faster,
        ha="center",
        va="top",
        transform=colorbar.ax.transAxes,
    )


def save_figure(figure: Figure, output_path: Path) -> None:
    """Create the output directory, save a figure, and release it."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path)
    plt.close(figure)
