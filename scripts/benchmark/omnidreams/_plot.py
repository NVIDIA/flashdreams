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

"""Shared rendering for Omnidreams benchmark plots."""

from __future__ import annotations

import math
import re
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
    ("omnidreams-dit-self-attention", "Self-attention"),
    ("omnidreams-dit-cross-attention", "Cross-attention"),
)
_DIT_BLOCK_PANEL = ("omnidreams-dit-block", "DiT block")
_PIPELINE_GENERATE_PANEL = (
    "omnidreams-full-pipeline-generate",
    "Pipeline generate",
)
_END_TO_END_PANELS = (
    ("omnidreams-dit-network", "Network eval"),
    _PIPELINE_GENERATE_PANEL,
)
_PANELS = (*_ATTENTION_PANELS, _DIT_BLOCK_PANEL, *_END_TO_END_PANELS)

_NATIVE_LABELS = {
    "omnidreams-torch": "PyTorch\nOmnidreams\nBF16",
    "cuda": "CUDA\ncuDNN\nFP8",
    "cuda-sparge": "CUDA\nSparge\nFP8",
    "cuda-sage3": "CUDA\nSage3\nBF16",
    "cuda-sage3-fp8": "CUDA\nSage3\nFP8",
}
"""Implementation labels for configurations without parseable optimized IDs."""

_FUSION_LABELS = {
    "none": "None",
    "fuse-kv": "Fuse KV",
    "full": "Fuse QKV",
}
"""Display labels by stable QKV fusion ID."""

BenchmarkValues = dict[str, dict[str, float]]
Panel = tuple[str, str]


def _load_results(
    input_path: Path,
    panels: tuple[Panel, ...] = _PANELS,
) -> tuple[BenchmarkValues, list[str], str]:
    """Load Omnidreams median timings and environment metadata.

    Args:
        input_path: Pytest-benchmark JSON file to parse.
        panels: Benchmark groups required by the requested figures.

    Returns:
        Median milliseconds by group and implementation, configuration order,
        and plot subtitle.

    Raises:
        SystemExit: The input cannot be read or lacks complete Omnidreams data.
    """
    payload, benchmark_records = load_benchmark_json(input_path)

    values: BenchmarkValues = {group: {} for group, _ in panels}
    configurations: list[str] = []

    for record in benchmark_records:
        if not isinstance(record, dict):
            continue
        group = record.get("group")
        if not isinstance(group, str) or group not in values:
            continue
        param = record.get("param")
        if not isinstance(param, str) or not param:
            raise SystemExit(
                f"Benchmark {record.get('name', '<unnamed>')} has no parameter ID"
            )
        implementation = param
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


def _backend_label(backend: str) -> str:
    return "cuDNN" if backend == "cudnn" else backend.upper()


def _attention_configuration_label(implementation: str) -> str:
    """Format one attention implementation as a multiline axis label.

    Args:
        implementation: Stable benchmark implementation identifier.

    Returns:
        Backend, fusion, TMA, projection, and SDPA labels.
    """
    if implementation == "omnidreams":
        return "Omnidreams\nreference"
    match = re.fullmatch(
        r"(?:optimized|triton)-(cudnn|fa2)-(none|full|fuse-kv)-(no-tma|tma)"
        r"(?:-projection-(float8-e4m3fn))?(-quantized-sdpa)?",
        implementation,
    )
    if match is None:
        return "\n".join(implementation.split("-"))
    backend, fusion, tma, projection, quantized_sdpa = match.groups()
    labels = [
        "Optimized",
        _backend_label(backend),
        f"Fusion: {_FUSION_LABELS[fusion]}",
        "No TMA" if tma == "no-tma" else "TMA",
    ]
    if projection is not None:
        labels.extend(("Projection:", "FP8 E4M3"))
    if quantized_sdpa is not None:
        labels.extend(("SDPA:", "FP8 E4M3"))
    return "\n".join(labels)


def _end_to_end_configuration_label(implementation: str) -> str:
    """Format one network or pipeline implementation as a multiline axis label.

    Args:
        implementation: Stable benchmark implementation identifier.

    Returns:
        Implementation, precision, self-attention, and cross-attention labels.
    """
    native_label = _NATIVE_LABELS.get(implementation)
    if native_label is not None:
        return native_label

    selected = re.fullmatch(
        r"(?:optimized|triton)-(cudnn|fa2)-(bf16|fp8|quantized-sdpa)-"
        r"self-(none|full|fuse-kv)-(no-tma|tma)-"
        r"cross-(none|fuse-kv)-(no-tma|tma)",
        implementation,
    )
    if selected is not None:
        backend, precision, self_fusion, self_tma, cross_fusion, cross_tma = (
            selected.groups()
        )
        precision_label = {
            "bf16": "BF16",
            "fp8": "FP8 projections",
            "quantized-sdpa": "FP8 projections + SDPA",
        }[precision]
        return "\n".join(
            (
                "Optimized",
                _backend_label(backend),
                precision_label,
                "Self:",
                _FUSION_LABELS[self_fusion],
                "No TMA" if self_tma == "no-tma" else "TMA",
                "Cross:",
                _FUSION_LABELS[cross_fusion],
                "No TMA" if cross_tma == "no-tma" else "TMA",
            )
        )

    legacy = re.fullmatch(
        r"(?:optimized|triton)-(cudnn|fa2)-(bf16)-(none|full|fuse-kv)"
        r"(-omnidreams-cross)?",
        implementation,
    )
    if legacy is None:
        return "\n".join(implementation.split("-"))
    backend, precision, self_fusion, omnidreams_cross = legacy.groups()
    cross_fusion = "fuse-kv" if self_fusion == "full" else self_fusion
    cross_labels = (
        ("Omnidreams", "No fusion")
        if omnidreams_cross is not None
        else (_FUSION_LABELS[cross_fusion], "TMA")
    )
    return "\n".join(
        (
            "Optimized",
            _backend_label(backend),
            precision.upper(),
            "Self:",
            _FUSION_LABELS[self_fusion],
            "TMA",
            "Cross:",
            *cross_labels,
        )
    )


def _bar_label(
    latency_ms: float,
    reference_ms: float,
    *,
    is_reference: bool,
) -> str:
    """Format latency and relative performance against a reference.

    Args:
        latency_ms: Bar latency in milliseconds.
        reference_ms: Reference latency in milliseconds.
        is_reference: Whether the bar or cell is the reference.

    Returns:
        Two-line latency and relative-performance annotation.
    """
    precision = 3 if latency_ms < 1 else 2 if latency_ms < 10 else 1
    return latency_comparison_label(
        latency_ms,
        reference_ms,
        is_reference=is_reference,
        precision=precision,
    )


def _panel_configurations(
    panels: tuple[Panel, ...],
    values: BenchmarkValues,
    configurations: list[str],
) -> list[str]:
    return [
        configuration
        for configuration in configurations
        if any(configuration in values[group] for group, _ in panels)
    ]


def _write_bar_figure(
    output_path: Path,
    title: str,
    panels: tuple[Panel, ...],
    values: BenchmarkValues,
    configurations: list[str],
    subtitle: str,
) -> None:
    """Write aligned median-latency bar charts as one PNG.

    Args:
        output_path: Destination PNG file.
        title: Figure title.
        panels: Benchmark group and display-title pairs.
        values: Median milliseconds by group and implementation.
        configurations: Stable implementation order.
        subtitle: Benchmark environment summary.
    """
    end_to_end = all(
        panel in (*_END_TO_END_PANELS, _DIT_BLOCK_PANEL) for panel in panels
    )
    reference_configuration = "omnidreams-torch" if end_to_end else "omnidreams"
    configuration_label = (
        _end_to_end_configuration_label
        if end_to_end
        else _attention_configuration_label
    )
    panel_configurations = _panel_configurations(panels, values, configurations)
    figure, axes = plt.subplots(
        len(panels),
        1,
        figsize=(
            max(16.0, len(panel_configurations) * 1.3),
            len(panels) * 5.0,
        ),
        layout="constrained",
    )
    axes_list = [axes] if len(panels) == 1 else list(axes)
    for axes_item, (group, panel_title) in zip(axes_list, panels, strict=True):
        reference = values[group].get(reference_configuration)
        if reference is None:
            raise SystemExit(
                f"Benchmark results for {group} have no {reference_configuration} reference"
            )
        ordered_configurations = sorted(
            panel_configurations,
            key=lambda configuration: values[group].get(configuration, math.inf),
        )
        latencies = [
            values[group].get(configuration, math.nan)
            for configuration in ordered_configurations
        ]
        colors = [
            "C2"
            if configuration == reference_configuration
            else "C1"
            if configuration.startswith("cuda")
            else "C0"
            for configuration in ordered_configurations
        ]
        bars = axes_item.bar(
            range(len(ordered_configurations)),
            latencies,
            color=colors,
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
                    is_reference=configuration == reference_configuration,
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
            rotation=0,
            ha="center",
        )

    figure.supxlabel(
        "Configuration\n(self-attention + cross-attention)"
        if end_to_end
        else "Configuration\n(SDPA backend / QKV fusion /\nTMA / projection dtype)"
    )
    figure.suptitle(f"{title}\n{subtitle}")
    save_figure(figure, output_path)


def _split_block_implementation(
    implementation: str,
) -> tuple[str, str]:
    """Split one DiT block ID into self- and cross-attention implementations.

    Args:
        implementation: Stable DiT block benchmark identifier.

    Returns:
        Self- and cross-attention implementation identifiers.

    Raises:
        SystemExit: The identifier does not encode both implementations.
    """
    prefix = "self-"
    if not implementation.startswith(prefix):
        raise SystemExit(f"Unsupported DiT block parameter {implementation!r}")
    self_implementation, separator, cross_implementation = implementation[
        len(prefix) :
    ].partition("-cross-")
    if not separator or not self_implementation or not cross_implementation:
        raise SystemExit(f"Unsupported DiT block parameter {implementation!r}")
    return self_implementation, cross_implementation


def _write_block_figure(
    output_path: Path,
    values: BenchmarkValues,
    subtitle: str,
) -> None:
    """Write the DiT block self-by-cross implementation heatmap.

    Args:
        output_path: Destination PNG file.
        values: Median milliseconds by group and implementation.
        subtitle: Benchmark environment summary.
    """
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

    reference_key = ("omnidreams", "omnidreams")
    reference = latencies.get(reference_key)
    if reference is None:
        raise SystemExit("DiT block results have no all-Omnidreams reference")

    matrix = [
        [
            latencies.get((self_implementation, cross_implementation), math.nan)
            / reference
            for cross_implementation in cross_configurations
            # ponytail: Linear sizing assumes current short labels; measure rendered
            # text extents if benchmark labels become substantially longer.
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
        rotation=0,
        ha="center",
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
    axes.set_title(f"Omnidreams DiT block performance\n{subtitle}")

    for self_index, self_implementation in enumerate(self_configurations):
        for cross_index, cross_implementation in enumerate(cross_configurations):
            latency = latencies.get((self_implementation, cross_implementation))
            if latency is None:
                label = "N/A\nnot measured"
            else:
                label = _bar_label(
                    latency,
                    reference,
                    is_reference=(
                        self_implementation,
                        cross_implementation,
                    )
                    == reference_key,
                )
            relative_value = matrix[self_index][cross_index]
            annotate_relative_cell(
                axes,
                image,
                self_index,
                cross_index,
                label,
                relative_value,
            )

    add_relative_colorbar(
        figure,
        axes,
        image,
        "Runtime relative to\nall-Omnidreams reference (×)",
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
        "Omnidreams attention module benchmarks",
        _ATTENTION_PANELS,
        values,
        configurations,
        subtitle,
    )
    print(f"Wrote {attention_output}")

    block_output = output_dir / "dit_block.png"
    if all(
        implementation.startswith("self-")
        for implementation in values[_DIT_BLOCK_PANEL[0]]
    ):
        _write_block_figure(block_output, values, subtitle)
    else:
        _write_bar_figure(
            block_output,
            "Omnidreams DiT block benchmark",
            (_DIT_BLOCK_PANEL,),
            values,
            configurations,
            subtitle,
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
        title = "Omnidreams pipeline generate benchmark"
    else:
        output_path = output_dir / "network_pipeline.png"
        title = "Omnidreams network and pipeline benchmarks"
    _write_bar_figure(
        output_path,
        title,
        panels,
        values,
        configurations,
        subtitle,
    )
    print(f"Wrote {output_path}")
