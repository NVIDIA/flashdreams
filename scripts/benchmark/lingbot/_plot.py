# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared rendering for LingBot benchmark plots."""

from __future__ import annotations

import math
import re
from collections.abc import Callable
from pathlib import Path

import matplotlib.pyplot as plt

from scripts.benchmark.common import (
    add_relative_colorbar,
    annotate_relative_cell,
    benchmark_median_ms,
    benchmark_subtitle,
    draw_relative_heatmap,
    latency_comparison_label,
    load_benchmark_json,
    save_figure,
)

_ATTENTION_PANELS = (
    ("lingbot-dit-self-attention", "Self-attention"),
    ("lingbot-dit-cross-attention", "Cross-attention"),
)
_DIT_BLOCK_PANEL = ("lingbot-dit-block", "DiT block")
_PIPELINE_GENERATE_PANEL = (
    "lingbot-full-pipeline-generate",
    "Pipeline generate",
)
_END_TO_END_PANELS = (
    ("lingbot-dit-network", "Network eval"),
    _PIPELINE_GENERATE_PANEL,
)

_FUSION_LABELS = {
    "none": "None",
    "fuse-kv": "Fuse KV",
    "full": "Fuse QKV",
}
_END_TO_END_LABELS = {
    "torch": "Torch\nself + cross",
    "optimized-self": "Optimized self\nTorch cross",
    "optimized-cross": "Torch self\nOptimized cross",
    "optimized-self-cross": "Optimized\nself + cross",
}

BenchmarkValues = dict[str, dict[str, float]]
Panel = tuple[str, str]


def _load_results(
    input_path: Path,
    panels: tuple[Panel, ...],
) -> tuple[BenchmarkValues, list[str], str]:
    """Load median timings for the requested LingBot benchmark groups."""
    payload, benchmark_records = load_benchmark_json(input_path)
    values: BenchmarkValues = {group: {} for group, _ in panels}
    configurations: list[str] = []

    for record in benchmark_records:
        if not isinstance(record, dict):
            continue
        group = record.get("group")
        if not isinstance(group, str) or group not in values:
            continue
        implementation = record.get("param")
        if not isinstance(implementation, str) or not implementation:
            raise SystemExit(
                f"Benchmark {record.get('name', '<unnamed>')} has no parameter ID"
            )
        if implementation in values[group]:
            raise SystemExit(f"Duplicate benchmark for {group} and {implementation}")

        values[group][implementation] = benchmark_median_ms(record)
        if implementation not in configurations:
            configurations.append(implementation)

    missing = [group for group, results in values.items() if not results]
    if missing:
        raise SystemExit(
            f"Benchmark JSON {input_path} lacks groups: {', '.join(missing)}"
        )
    return values, configurations, benchmark_subtitle(payload)


def _attention_configuration_label(implementation: str) -> str:
    """Format one isolated attention implementation as an axis label."""
    if implementation == "torch":
        return "Torch\nreference"
    match = re.fullmatch(
        r"optimized-(cudnn|fa2)-(none|full|fuse-kv)-(no-tma|tma)"
        r"(?:-projection-(float8-e4m3fn))?(-quantized-sdpa)?",
        implementation,
    )
    if match is None:
        return "\n".join(implementation.split("-"))
    backend, fusion, tma, projection, quantized_sdpa = match.groups()
    labels = [
        "Optimized",
        "cuDNN" if backend == "cudnn" else "FA2",
        f"Fusion: {_FUSION_LABELS[fusion]}",
        "No TMA" if tma == "no-tma" else "TMA",
    ]
    if projection is not None:
        labels.extend(("Projection:", "FP8 E4M3"))
    if quantized_sdpa is not None:
        labels.extend(("SDPA:", "FP8 E4M3"))
    return "\n".join(labels)


def _end_to_end_configuration_label(implementation: str) -> str:
    """Format one network, pipeline, or representative block case."""
    return _END_TO_END_LABELS.get(
        implementation,
        "\n".join(implementation.split("-")),
    )


def _bar_label(
    latency_ms: float,
    reference_ms: float,
    *,
    is_reference: bool,
) -> str:
    precision = 3 if latency_ms < 1 else 2 if latency_ms < 10 else 1
    return latency_comparison_label(
        latency_ms,
        reference_ms,
        is_reference=is_reference,
        precision=precision,
    )


def _write_bar_figure(
    output_path: Path,
    title: str,
    panels: tuple[Panel, ...],
    values: BenchmarkValues,
    configurations: list[str],
    subtitle: str,
    configuration_label: Callable[[str], str],
    xlabel: str,
) -> None:
    panel_configurations = [
        configuration
        for configuration in configurations
        if any(configuration in values[group] for group, _ in panels)
    ]
    figure, axes = plt.subplots(
        len(panels),
        1,
        figsize=(max(12.0, len(panel_configurations) * 1.3), len(panels) * 5.0),
        layout="constrained",
    )
    axes_list = [axes] if len(panels) == 1 else list(axes)
    for axes_item, (group, panel_title) in zip(axes_list, panels, strict=True):
        reference = values[group].get("torch")
        if reference is None:
            raise SystemExit(f"Benchmark results for {group} have no torch reference")
        ordered_configurations = sorted(
            panel_configurations,
            key=lambda configuration: values[group].get(configuration, math.inf),
        )
        latencies = [
            values[group].get(configuration, math.nan)
            for configuration in ordered_configurations
        ]
        bars = axes_item.bar(
            range(len(ordered_configurations)),
            latencies,
            color=[
                "C2" if configuration == "torch" else "C0"
                for configuration in ordered_configurations
            ],
        )
        finite_latencies = [value for value in latencies if math.isfinite(value)]
        axes_item.set_ylim(0, max(finite_latencies) * 1.2)
        axes_item.set_title(panel_title)
        axes_item.set_ylabel("Median\nlatency\n(ms)")

        for index, (bar, latency, configuration) in enumerate(
            zip(bars, latencies, ordered_configurations, strict=True)
        ):
            if not math.isfinite(latency):
                bar.set_visible(False)
                axes_item.text(
                    index,
                    0.02,
                    "N/A\nnot measured",
                    transform=axes_item.get_xaxis_transform(),
                    ha="center",
                    va="bottom",
                )
                continue
            axes_item.annotate(
                _bar_label(
                    latency,
                    reference,
                    is_reference=configuration == "torch",
                ),
                xy=(bar.get_x() + bar.get_width() / 2, latency),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
            )
        axes_item.set_xticks(
            range(len(ordered_configurations)),
            labels=[
                configuration_label(configuration)
                for configuration in ordered_configurations
            ],
        )

    figure.supxlabel(xlabel)
    figure.suptitle(f"{title}\n{subtitle}")
    save_figure(figure, output_path)


def _split_block_implementation(implementation: str) -> tuple[str, str]:
    """Split a full-search block ID into self- and cross-attention IDs."""
    if not implementation.startswith("self-"):
        raise SystemExit(f"Unsupported DiT block parameter {implementation!r}")
    self_implementation, separator, cross_implementation = implementation[
        len("self-") :
    ].partition("-cross-")
    if not separator or not self_implementation or not cross_implementation:
        raise SystemExit(f"Unsupported DiT block parameter {implementation!r}")
    return self_implementation, cross_implementation


def _write_block_heatmap(
    output_path: Path,
    values: BenchmarkValues,
    subtitle: str,
) -> None:
    block_values = values[_DIT_BLOCK_PANEL[0]]
    self_configurations: list[str] = []
    cross_configurations: list[str] = []
    latencies: dict[tuple[str, str], float] = {}
    for implementation, latency in block_values.items():
        self_implementation, cross_implementation = _split_block_implementation(
            implementation
        )
        cell = (self_implementation, cross_implementation)
        if cell in latencies:
            raise SystemExit(
                "Duplicate DiT block benchmark for "
                f"{self_implementation} and {cross_implementation}"
            )
        latencies[cell] = latency
        if self_implementation not in self_configurations:
            self_configurations.append(self_implementation)
        if cross_implementation not in cross_configurations:
            cross_configurations.append(cross_implementation)

    reference_key = ("torch", "torch")
    reference = latencies.get(reference_key)
    if reference is None:
        raise SystemExit("DiT block results have no all-Torch reference")
    matrix = [
        [
            latencies.get((self_implementation, cross_implementation), math.nan)
            / reference
            for cross_implementation in cross_configurations
        ]
        for self_implementation in self_configurations
    ]
    figure, axes = plt.subplots(
        figsize=(
            max(12.0, len(cross_configurations) * 2.0),
            max(8.0, len(self_configurations) * 0.9),
        ),
        layout="constrained",
    )
    image = draw_relative_heatmap(axes, matrix)
    axes.set_xticks(
        range(len(cross_configurations)),
        labels=[
            _attention_configuration_label(configuration).removeprefix("Optimized\n")
            for configuration in cross_configurations
        ],
    )
    axes.set_yticks(
        range(len(self_configurations)),
        labels=[
            _attention_configuration_label(configuration).removeprefix("Optimized\n")
            for configuration in self_configurations
        ],
    )
    axes.set_xlabel("Cross-attention\nimplementation")
    axes.set_ylabel("Self-attention\nimplementation")
    axes.set_title(f"LingBot DiT block performance\n{subtitle}")

    for self_index, self_implementation in enumerate(self_configurations):
        for cross_index, cross_implementation in enumerate(cross_configurations):
            latency = latencies.get((self_implementation, cross_implementation))
            label = (
                "N/A\nnot measured"
                if latency is None
                else _bar_label(
                    latency,
                    reference,
                    is_reference=(
                        self_implementation,
                        cross_implementation,
                    )
                    == reference_key,
                )
            )
            annotate_relative_cell(
                axes,
                image,
                self_index,
                cross_index,
                label,
                matrix[self_index][cross_index],
            )

    add_relative_colorbar(
        figure,
        axes,
        image,
        "Runtime relative to\nall-Torch reference (×)",
        multiline_directions=True,
    )
    save_figure(figure, output_path)


def plot_modules(input_path: Path, output_dir: Path) -> None:
    """Generate attention-module and DiT-block benchmark figures."""
    panels = (*_ATTENTION_PANELS, _DIT_BLOCK_PANEL)
    values, configurations, subtitle = _load_results(input_path, panels)

    attention_output = output_dir / "modules.png"
    _write_bar_figure(
        attention_output,
        "LingBot attention module benchmarks",
        _ATTENTION_PANELS,
        values,
        configurations,
        subtitle,
        _attention_configuration_label,
        "Configuration\n(SDPA backend / QKV fusion / TMA / projection dtype)",
    )
    print(f"Wrote {attention_output}")

    block_output = output_dir / "dit_block.png"
    if all(
        implementation.startswith("self-")
        for implementation in values[_DIT_BLOCK_PANEL[0]]
    ):
        _write_block_heatmap(block_output, values, subtitle)
    else:
        _write_bar_figure(
            block_output,
            "LingBot DiT block benchmark",
            (_DIT_BLOCK_PANEL,),
            values,
            configurations,
            subtitle,
            _end_to_end_configuration_label,
            "Configuration\n(self-attention + cross-attention)",
        )
    print(f"Wrote {block_output}")


def plot_pipeline(
    input_path: Path,
    output_dir: Path,
    *,
    pipeline_generate_only: bool = False,
) -> None:
    """Generate network and pipeline benchmark figures."""
    panels = (
        (_PIPELINE_GENERATE_PANEL,) if pipeline_generate_only else _END_TO_END_PANELS
    )
    values, configurations, subtitle = _load_results(input_path, panels)
    if pipeline_generate_only:
        output_path = output_dir / "pipeline_generate.png"
        title = "LingBot pipeline generate benchmark"
    else:
        output_path = output_dir / "network_pipeline.png"
        title = "LingBot network and pipeline benchmarks"
    _write_bar_figure(
        output_path,
        title,
        panels,
        values,
        configurations,
        subtitle,
        _end_to_end_configuration_label,
        "Configuration\n(self-attention + cross-attention)",
    )
    print(f"Wrote {output_path}")
