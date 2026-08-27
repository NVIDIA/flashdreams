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

"""Performance matrices for quantized GEMM benchmark results."""

from __future__ import annotations

import argparse
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
    relative_matrix_with_fastest,
    save_figure,
)

_DEFAULT_OUTPUT_DIR = Path("artifacts/benchmark/flashdreams/accelerated/quantization")
_DEFAULT_INPUT = _DEFAULT_OUTPUT_DIR / "benchmark.json"
_FASTEST_CONFIG_COLUMN = "fastest-quantized-config"
_REFERENCE_CONFIG = "full-precision"
"""Benchmark parameter ID used as each row's performance reference."""

_CONFIG_LABELS = {
    "full-precision": "Full precision",
    "float8_e4m3fn-slice": "FP8 E4M3\nslice",
    "float8_e4m3fn-tensor": "FP8 E4M3\ntensor",
    "float8_e5m2-x-float8_e4m3fn-slice": "FP8 E5M2 × E4M3\nslice",
    "float8_e5m2-x-float8_e4m3fn-tensor": "FP8 E5M2 × E4M3\ntensor",
    "int8-slice": "INT8\nslice",
    "int8-tensor": "INT8\ntensor",
}
_FORMAT_LABELS = {"fp16": "FP16", "bf16": "BF16", "fp32": "FP32"}
_GEOMETRY_LABELS = {
    "square-4096": "Square 4096×4096×4096",
    "mha-query-output": "MHA query/output 4800×2048×2048",
    "mha-fused-qkv": "MHA fused QKV 4800×2048×6144",
    "mha-cross-fused-kv": "MHA cross fused K/V 28800×2048×4096",
}
_SCOPE_LABELS = {
    "end-to-end": "End-to-end",
    "gemm-only": "GEMM only",
}

RowKey = tuple[str, str, str, str]
CellKey = tuple[RowKey, str]


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse benchmark input and plot output paths.

    Args:
        argv: Command-line arguments; ``None`` reads ``sys.argv``.

    Returns:
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Plot quantized GEMM median latency as PNG matrices."
    )
    add_plot_io_arguments(parser, _DEFAULT_INPUT, _DEFAULT_OUTPUT_DIR)
    return parser.parse_args(argv)


def _load_matrix(
    input_path: Path,
) -> tuple[list[RowKey], list[str], dict[CellKey, float], str]:
    """Load quantized GEMM rows and median timings from benchmark JSON.

    Args:
        input_path: Pytest-benchmark JSON file to parse.

    Returns:
        Ordered row keys, configuration columns, median milliseconds by cell,
        and plot subtitle.

    Raises:
        SystemExit: The input cannot be read or does not contain compatible
            quantized GEMM records.
    """
    payload, benchmark_records = load_benchmark_json(input_path)

    rows: list[RowKey] = []
    columns: list[str] = []
    values_ms: dict[CellKey, float] = {}

    for record in benchmark_records:
        if not isinstance(record, dict):
            continue
        group = record.get("group")
        param = record.get("param")
        if not isinstance(group, str) or not isinstance(param, str):
            continue
        match = re.fullmatch(
            r"quantized-gemm-(?:(?P<geometry>.+)-)?"
            r"(?P<effective>fp16|bf16|fp32)-"
            r"(?P<scope>end-to-end|gemm-only)",
            group,
        )
        if match is None:
            continue
        geometry_in_group = match.group("geometry")
        geometry = geometry_in_group or "square-4096"
        effective_format = match.group("effective")
        timing_scope = match.group("scope")
        parameter_match = re.fullmatch(r"(fp16|bf16|fp32)-(.+)", param)
        if parameter_match is None:
            raise SystemExit(f"Unsupported quantized GEMM parameter {param!r}")
        source_format, configuration = parameter_match.groups()
        if geometry_in_group is not None:
            suffix = f"-{geometry}"
            if not configuration.endswith(suffix):
                raise SystemExit(f"Unsupported quantized GEMM parameter {param!r}")
            configuration = configuration.removesuffix(suffix)

        row = (timing_scope, geometry, effective_format, source_format)
        cell = (row, configuration)
        if cell in values_ms:
            raise SystemExit(f"Duplicate benchmark cell for {row} and {configuration}")
        if row not in rows:
            rows.append(row)
        if configuration not in columns:
            columns.append(configuration)
        values_ms[cell] = benchmark_median_ms(record)

    if not values_ms:
        raise SystemExit(
            f"Benchmark JSON {input_path} contains no quantized GEMM results"
        )

    for row in rows:
        timing_scope, geometry, effective_format, _ = row
        reference_row = (
            timing_scope,
            geometry,
            effective_format,
            effective_format,
        )
        reference = values_ms.get((reference_row, _REFERENCE_CONFIG))
        if reference is not None:
            values_ms.setdefault((row, _REFERENCE_CONFIG), reference)

    return rows, columns, values_ms, benchmark_subtitle(payload)


def _config_label(configuration: str) -> str:
    """Return a compact display label for a benchmark configuration."""
    if configuration == _FASTEST_CONFIG_COLUMN:
        return "Fastest quantized\nconfig"
    return _CONFIG_LABELS.get(configuration, configuration.replace("-", " "))


def _row_label(row: RowKey) -> str:
    """Return a geometry and effective-dtype label."""
    _, geometry, effective_format, source_format = row
    geometry_label = _GEOMETRY_LABELS.get(geometry, geometry.replace("-", " "))
    effective = _FORMAT_LABELS.get(effective_format, effective_format)
    if source_format == effective_format:
        return f"{geometry_label}\n{effective}"
    source = _FORMAT_LABELS.get(source_format, source_format)
    return f"{geometry_label}\n{effective} effective · {source} source"


def _write_png(
    output_path: Path,
    timing_scope: str,
    rows: list[RowKey],
    columns: list[str],
    values_ms: dict[CellKey, float],
    subtitle: str,
) -> None:
    """Write one timing scope's median-latency heatmap as a PNG.

    Args:
        output_path: Destination PNG file.
        timing_scope: Timed region represented by every row.
        rows: Ordered effective- and source-format rows.
        columns: Ordered GEMM configuration columns.
        values_ms: Median milliseconds keyed by row and configuration.
        subtitle: Benchmark environment summary.

    Raises:
        SystemExit: Results have no full-precision reference configuration.
    """
    if _REFERENCE_CONFIG not in columns:
        raise SystemExit("Benchmark results have no full-precision configuration")

    display_columns = [*columns, _FASTEST_CONFIG_COLUMN]
    matrix, fastest_configurations = relative_matrix_with_fastest(
        rows, columns, values_ms, _REFERENCE_CONFIG
    )

    figure, axes = plt.subplots()
    default_width, default_height = figure.get_size_inches()
    figure.set_size_inches(
        max(default_width, len(display_columns) * 1.8),
        max(default_height, len(rows) * 1.2),
    )
    image = draw_relative_heatmap(axes, matrix)
    axes.set_xticks(
        range(len(display_columns)),
        labels=[_config_label(column) for column in display_columns],
        rotation=45,
        ha="right",
        rotation_mode="anchor",
    )
    axes.axvline(len(columns) - 0.5, color="black", linewidth=1.5)
    axes.set_yticks(range(len(rows)), labels=[_row_label(row) for row in rows])
    axes.set_xlabel("GEMM configuration")
    axes.set_ylabel("Effective / source dtype")
    axes.set_title(f"Quantized GEMM {_SCOPE_LABELS[timing_scope]}\n{subtitle}")

    for row_index, row in enumerate(rows):
        reference = values_ms.get((row, _REFERENCE_CONFIG))
        for column_index, column in enumerate(display_columns):
            if column == _FASTEST_CONFIG_COLUMN:
                fastest = fastest_configurations[row]
                if fastest is None:
                    value = None
                    label = "N/A"
                else:
                    value = values_ms[(row, fastest)]
                    label = (
                        f"{_config_label(fastest)}\n"
                        f"{latency_comparison_label(value, reference, is_reference=False)}"
                    )
            else:
                value = values_ms.get((row, column))
                label = latency_comparison_label(
                    value,
                    reference,
                    is_reference=column == _REFERENCE_CONFIG,
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
        "Runtime relative to full precision (×)",
    )
    figure.tight_layout()
    save_figure(figure, output_path)


def main(argv: list[str] | None = None) -> None:
    """Generate separate end-to-end and GEMM-only performance matrices."""
    args = _parse_args(argv)
    rows, columns, values_ms, subtitle = _load_matrix(args.input)
    timing_scopes = list(dict.fromkeys(row[0] for row in rows))
    for timing_scope in timing_scopes:
        panel_rows = [row for row in rows if row[0] == timing_scope]
        panel_values = {
            cell: value
            for cell, value in values_ms.items()
            if cell[0][0] == timing_scope
        }
        output = args.output_dir / f"{timing_scope.replace('-', '_')}.png"
        _write_png(
            output,
            timing_scope,
            panel_rows,
            columns,
            panel_values,
            subtitle,
        )
        print(f"Wrote {output}")


if __name__ == "__main__":
    main()
