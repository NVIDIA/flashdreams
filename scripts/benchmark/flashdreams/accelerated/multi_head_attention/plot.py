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

"""Performance matrices for accelerated attention benchmark results."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from scripts.benchmark.common import (
    add_plot_io_arguments,
    add_relative_colorbar,
    annotate_relative_cell,
    benchmark_artifact_dir,
    benchmark_median_ms,
    benchmark_subtitle,
    draw_relative_heatmap,
    latency_comparison_label,
    load_benchmark_json,
    relative_matrix_with_fastest,
    save_figure,
)

_DEFAULT_OUTPUT_DIR = benchmark_artifact_dir(
    Path("artifacts/benchmark/flashdreams/accelerated/multi_head_attention")
)
_DEFAULT_INPUT = _DEFAULT_OUTPUT_DIR / "benchmark.json"
_FASTEST_IMPLEMENTATION_COLUMN = "fastest-implementation-config"
_HIGHLIGHTED_ROWS = {
    ("head", "before_kv_cache", False, False): "Cosmos",
    ("inner", "before_kv_cache", True, True): "Wan",
}
_REFERENCE_IMPLEMENTATION = "reference-torch"

RowKey = tuple[str, str, str, bool, bool]
CellKey = tuple[RowKey, str]


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse benchmark input and plot output paths.

    Args:
        argv: Command-line arguments; ``None`` reads ``sys.argv``.

    Returns:
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Plot self- and cross-attention latency as three PNG matrices."
    )
    add_plot_io_arguments(parser, _DEFAULT_INPUT, _DEFAULT_OUTPUT_DIR)
    return parser.parse_args(argv)


def _load_matrix(
    input_path: Path,
) -> tuple[list[RowKey], list[str], dict[CellKey, float], str]:
    """Load accelerated attention rows and median timings from benchmark JSON.

    Args:
        input_path: Pytest-benchmark JSON file to parse.

    Returns:
        Ordered row keys, implementation columns, median milliseconds by cell,
        and plot subtitle.

    Raises:
        SystemExit: The input cannot be read or does not contain compatible
            accelerated attention records.
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
            r"multi-head-attention-(self|cross(?:-query-only)?)-norm-(.+)-rope-"
            r"(interleaved|split)(?:-scope-(before-kv-cache|after-kv-cache))?"
            r"-bias-(on|off)",
            group,
        )
        if match is None:
            continue
        attention, norm, rope, scope_state, bias_state = match.groups()
        attention_type = (
            "cross_attention_query_only"
            if attention == "cross-query-only"
            else f"{attention}_attention"
        )
        rope_scope = (scope_state or "before-kv-cache").replace("-", "_")
        rope_interleaved = rope == "interleaved"
        bias = bias_state == "on"
        shared_id = group.removeprefix(f"multi-head-attention-{attention}-")
        implementation_prefix = f"{shared_id}-"
        if not param.startswith(implementation_prefix):
            raise SystemExit(f"Unsupported attention parameter {param!r}")
        implementation = param.removeprefix(implementation_prefix)

        row = (attention_type, norm, rope_scope, rope_interleaved, bias)
        cell = (row, implementation)
        if cell in values_ms:
            raise SystemExit(f"Duplicate benchmark cell for {row} and {implementation}")
        if row not in rows:
            rows.append(row)
        if implementation not in columns:
            columns.append(implementation)
        values_ms[cell] = benchmark_median_ms(record)

    if not values_ms:
        raise SystemExit(
            f"Benchmark JSON {input_path} contains no accelerated attention results"
        )

    return rows, columns, values_ms, benchmark_subtitle(payload)


def _row_label(row: RowKey) -> str:
    _, norm, rope_scope, rope_interleaved, bias = row
    rope = "interleaved" if rope_interleaved else "split"
    scope = rope_scope.removesuffix("_kv_cache").replace("_", " ")
    label = f"norm {norm} | rope {rope} {scope} cache | bias {'on' if bias else 'off'}"
    highlight = _HIGHLIGHTED_ROWS.get((norm, rope_scope, rope_interleaved, bias))
    return f"{label} | {highlight}" if highlight is not None else label


def _column_label(column: str) -> str:
    if column == _FASTEST_IMPLEMENTATION_COLUMN:
        return "fastest implementation config"
    return (
        " ".join(
            "fuse qkv" if token == "full" else token for token in column.split("-")
        )
        .replace(" projection ", "\nprojection ")
        .replace(" quantized sdpa", "\nquantized sdpa")
    )


def _write_png(
    output_path: Path,
    attention_type: str,
    rows: list[RowKey],
    columns: list[str],
    values_ms: dict[CellKey, float],
    subtitle: str,
) -> None:
    """Write one attention type's median-latency heatmap as a PNG.

    Args:
        output_path: Destination PNG file.
        attention_type: Attention family represented by every row.
        rows: Ordered attention policy rows.
        columns: Ordered implementation configuration columns.
        values_ms: Median milliseconds keyed by row and implementation.
        subtitle: Benchmark environment summary.
    """
    attention_label = attention_type.replace("_", " ").title()
    if _REFERENCE_IMPLEMENTATION not in columns:
        raise SystemExit("Benchmark results have no reference-torch implementation")
    display_columns = [*columns, _FASTEST_IMPLEMENTATION_COLUMN]
    matrix, fastest_implementations = relative_matrix_with_fastest(
        rows, columns, values_ms, _REFERENCE_IMPLEMENTATION
    )
    figure, axes = plt.subplots()
    default_width, default_height = figure.get_size_inches()
    # ponytail: Linear sizing assumes current short labels; measure rendered
    # text extents if benchmark labels become substantially longer.
    figure.set_size_inches(
        max(default_width, len(display_columns) * 2.0),
        max(default_height, len(rows) * 0.9),
    )
    image = draw_relative_heatmap(axes, matrix)
    axes.set_xticks(
        range(len(display_columns)),
        labels=[_column_label(column) for column in display_columns],
        rotation=45,
        ha="right",
        rotation_mode="anchor",
    )
    axes.axvline(len(columns) - 0.5, color="black", linewidth=1.5)
    axes.set_yticks(range(len(rows)), labels=[_row_label(row) for row in rows])
    for row_index, (_, norm, rope_scope, rope_interleaved, bias) in enumerate(rows):
        if (norm, rope_scope, rope_interleaved, bias) in _HIGHLIGHTED_ROWS:
            axes.add_patch(
                Rectangle(
                    (-0.5, row_index - 0.5),
                    len(display_columns),
                    1,
                    fill=False,
                    edgecolor="black",
                    linewidth=2.5,
                    clip_on=False,
                    zorder=3,
                )
            )
    axes.set_xlabel("Implementation configuration")
    axes.set_ylabel("Attention configuration")
    axes.set_title(f"{attention_label} performance\n{subtitle}")
    for row_index, row in enumerate(rows):
        reference = values_ms.get((row, _REFERENCE_IMPLEMENTATION))
        for column_index, column in enumerate(display_columns):
            if column == _FASTEST_IMPLEMENTATION_COLUMN:
                fastest = fastest_implementations[row]
                if fastest is None:
                    value = None
                    label = "N/A"
                else:
                    value = values_ms[(row, fastest)]
                    relative_label = latency_comparison_label(
                        value, reference, is_reference=False
                    ).splitlines()[1]
                    implementation, backend, config = _column_label(fastest).split(
                        maxsplit=2
                    )
                    label = f"{implementation} {backend}\n{config}\n{relative_label}"
            else:
                value = values_ms.get((row, column))
                label = latency_comparison_label(
                    value,
                    reference,
                    is_reference=column == _REFERENCE_IMPLEMENTATION,
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
        "Runtime relative to torch reference (×)",
    )
    figure.tight_layout()
    save_figure(figure, output_path)


def main(argv: list[str] | None = None) -> None:
    """Generate self-attention and both cross-attention performance matrices."""
    args = _parse_args(argv)
    rows, columns, values_ms, subtitle = _load_matrix(args.input)
    attention_types = list(dict.fromkeys(row[0] for row in rows))
    for attention_type in attention_types:
        panel_rows = [row for row in rows if row[0] == attention_type]
        panel_values = {
            cell: value
            for cell, value in values_ms.items()
            if cell[0][0] == attention_type
        }
        output = args.output_dir / f"{attention_type}.png"
        _write_png(
            output,
            attention_type,
            panel_rows,
            columns,
            panel_values,
            subtitle,
        )
        print(f"Wrote {output}")


if __name__ == "__main__":
    main()
