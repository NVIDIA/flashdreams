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

"""Performance matrices for Torch and Triton quantizer benchmarks."""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt

from scripts.benchmark.common import (
    add_plot_io_arguments,
    add_relative_colorbar,
    annotate_relative_cell,
    benchmark_median_ms,
    benchmark_subtitle,
    draw_relative_heatmap,
    latency_comparison_label,
    load_benchmark_json,
    save_figure,
)

_DEFAULT_OUTPUT_DIR = Path("artifacts/benchmark/flashdreams/accelerated/quantization")
_DEFAULT_INPUT = _DEFAULT_OUTPUT_DIR / "benchmark.json"

_OPERATIONS = ("quantize", "dequantize")
_FORMATS = ("float8_e4m3fn", "float8_e5m2", "int8")
_GRANULARITIES = ("slice", "tensor")
_IMPLEMENTATIONS = ("torch", "triton")
_GROUP_PATTERN = re.compile(
    r"(?P<operation>quantize|dequantize)-"
    r"(?P<format>float8_e4m3fn|float8_e5m2|int8)-"
    r"(?P<granularity>slice|tensor)"
)
_FORMAT_LABELS = {
    "float8_e4m3fn": "FP8 E4M3",
    "float8_e5m2": "FP8 E5M2",
    "int8": "INT8",
}

CellKey = tuple[str, str, str, int, str]


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse benchmark input and plot output paths.

    Args:
        argv: Command-line arguments; ``None`` reads ``sys.argv``.

    Returns:
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Plot Torch and Triton quantizer median latency as PNG matrices."
    )
    add_plot_io_arguments(parser, _DEFAULT_INPUT, _DEFAULT_OUTPUT_DIR)
    return parser.parse_args(argv)


def _load_results(input_path: Path) -> tuple[dict[CellKey, float], str]:
    """Load quantizer median timings from benchmark JSON.

    Args:
        input_path: Pytest-benchmark JSON file to parse.

    Returns:
        Median milliseconds by operation, format, granularity, scale count,
        and implementation, plus the plot subtitle.

    Raises:
        SystemExit: The input cannot be read or lacks complete quantizer data.
    """
    payload, benchmark_records = load_benchmark_json(input_path)

    values_ms: dict[CellKey, float] = {}
    for record in benchmark_records:
        if not isinstance(record, dict):
            continue
        group = record.get("group")
        match = _GROUP_PATTERN.fullmatch(group) if isinstance(group, str) else None
        if match is None:
            continue

        extra_info = record.get("extra_info")
        implementation = (
            extra_info.get("implementation") if isinstance(extra_info, dict) else None
        )
        if (
            not isinstance(implementation, str)
            or implementation not in _IMPLEMENTATIONS
        ):
            raise SystemExit(
                f"Benchmark {record.get('name', '<unnamed>')} has no supported "
                "implementation metadata"
            )

        scale_count = (
            extra_info.get("scale_count", 1) if isinstance(extra_info, dict) else 1
        )
        if (
            isinstance(scale_count, bool)
            or not isinstance(scale_count, int)
            or scale_count < 1
        ):
            raise SystemExit(
                f"Benchmark {record.get('name', '<unnamed>')} has an invalid "
                "scale count"
            )
        cell = (
            match.group("operation"),
            match.group("format"),
            match.group("granularity"),
            scale_count,
            implementation,
        )
        if cell in values_ms:
            raise SystemExit(f"Duplicate quantizer benchmark cell for {cell}")
        values_ms[cell] = benchmark_median_ms(record)

    missing_operations = [
        operation
        for operation in _OPERATIONS
        if not any(cell[0] == operation for cell in values_ms)
    ]
    if missing_operations:
        raise SystemExit(
            f"Benchmark JSON {input_path} lacks quantizer operations: "
            f"{', '.join(missing_operations)}"
        )
    return values_ms, benchmark_subtitle(payload)


def _cell_label(
    value: float | None, reference: float | None, *, is_reference: bool
) -> str:
    """Format a latency and its relationship to the Torch reference.

    Args:
        value: Cell latency in milliseconds; ``None`` marks a missing result.
        reference: Torch latency for the same format and granularity.
        is_reference: Whether the cell is the Torch reference.

    Returns:
        Two-line latency and relative-performance annotation.
    """
    return latency_comparison_label(
        value,
        reference,
        is_reference=is_reference,
        precision=3,
    )


def _write_png(
    output_path: Path,
    operation: str,
    values_ms: dict[CellKey, float],
    subtitle: str,
) -> None:
    """Write one operation's Torch-versus-Triton latency heatmap.

    Args:
        output_path: Destination PNG file.
        operation: Quantizer operation represented by the matrix.
        values_ms: Median milliseconds keyed by benchmark configuration.
        subtitle: Benchmark environment summary.

    Raises:
        SystemExit: Any format and granularity pair lacks its Torch reference.
    """
    scale_counts = sorted({cell[3] for cell in values_ms if cell[0] == operation})
    columns = tuple(
        (scale_count, granularity, implementation)
        for scale_count in scale_counts
        for granularity in _GRANULARITIES
        for implementation in _IMPLEMENTATIONS
    )
    missing_references = [
        (format, granularity, scale_count)
        for format in _FORMATS
        for granularity in _GRANULARITIES
        for scale_count in scale_counts
        if (operation, format, granularity, scale_count, "torch") not in values_ms
    ]
    if missing_references:
        raise SystemExit(
            f"{operation.capitalize()} results lack Torch references: "
            + ", ".join(
                f"{format}/{granularity}/{scale_count}-scale"
                for format, granularity, scale_count in missing_references
            )
        )

    matrix: list[list[float]] = []
    for format in _FORMATS:
        row = []
        for scale_count, granularity, implementation in columns:
            reference = values_ms[
                (operation, format, granularity, scale_count, "torch")
            ]
            value = values_ms.get(
                (operation, format, granularity, scale_count, implementation)
            )
            row.append(value / reference if value is not None else math.nan)
        matrix.append(row)

    figure, axes = plt.subplots()
    default_width, default_height = figure.get_size_inches()
    figure.set_size_inches(
        max(default_width, len(columns) * 1.8),
        max(default_height, len(_FORMATS) * 1.2),
    )
    image = draw_relative_heatmap(axes, matrix)
    axes.set_xticks(
        range(len(columns)),
        labels=[
            (
                f"{scale_count} {'scale' if scale_count == 1 else 'scales'}\n"
                f"{granularity.capitalize()}\n{implementation.capitalize()}"
                if len(scale_counts) > 1
                else f"{granularity.capitalize()}\n{implementation.capitalize()}"
            )
            for scale_count, granularity, implementation in columns
        ],
    )
    axes.set_yticks(
        range(len(_FORMATS)),
        labels=[_FORMAT_LABELS[format] for format in _FORMATS],
    )
    axes.set_xlabel(
        "Scale count, granularity, and implementation"
        if len(scale_counts) > 1
        else "Scale granularity and implementation"
    )
    axes.set_ylabel("Quantized dtype")
    axes.set_title(f"{operation.capitalize()} · Torch vs Triton\n{subtitle}")

    for row_index, format in enumerate(_FORMATS):
        for column_index, (scale_count, granularity, implementation) in enumerate(
            columns
        ):
            reference = values_ms[
                (operation, format, granularity, scale_count, "torch")
            ]
            value = values_ms.get(
                (operation, format, granularity, scale_count, implementation)
            )
            label = _cell_label(
                value,
                reference,
                is_reference=implementation == "torch",
            )

            relative_value = matrix[row_index][column_index]
            annotate_relative_cell(
                axes,
                image,
                row_index,
                column_index,
                label,
                relative_value,
            )

    add_relative_colorbar(
        figure,
        axes,
        image,
        "Runtime relative to Torch (×)",
    )
    figure.tight_layout()
    save_figure(figure, output_path)


def main(argv: list[str] | None = None) -> None:
    """Generate separate quantize and dequantize performance matrices."""
    args = _parse_args(argv)
    values_ms, subtitle = _load_results(args.input)
    for operation in _OPERATIONS:
        output = args.output_dir / f"{operation}.png"
        _write_png(output, operation, values_ms, subtitle)
        print(f"Wrote {output}")


if __name__ == "__main__":
    main()
