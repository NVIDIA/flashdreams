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

"""Bar plots for Omnidreams module and end-to-end benchmark results."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt

_DEFAULT_INPUT = Path("artifacts/benchmark/omnidreams/benchmark.json")
_DEFAULT_OUTPUT_DIR = Path("artifacts/benchmark/omnidreams")

_MODULE_PANELS = (
    ("omnidreams-dit-self-attention", "Self-attention"),
    ("omnidreams-dit-cross-attention", "Cross-attention"),
    ("omnidreams-dit-block", "DiT block"),
)
_END_TO_END_PANELS = (
    ("omnidreams-dit-network", "Network eval"),
    ("omnidreams-full-pipeline-generate", "Pipeline generate"),
    ("omnidreams-full-pipeline-finalize", "Pipeline finalize"),
)
_PANELS = (*_MODULE_PANELS, *_END_TO_END_PANELS)

_END_TO_END_LABELS = {
    "omnidreams_torch": "Omnidreams PyTorch\nself + cross",
    "triton_fa2_fp8_full": "Triton FA2 FP8\nQKV self + KV cross",
    "triton_cudnn_bf16_full": "Triton cuDNN BF16\nQKV self + KV cross",
    "triton_cudnn_bf16_full_omnidreams_cross": (
        "Triton cuDNN BF16\nQKV self + Omnidreams cross"
    ),
    "cuda": "Native CUDA FP8\ncuDNN",
    "cuda_sparge": "Native CUDA FP8\nSparge",
    "cuda_sage3": "Native CUDA BF16\nSage3",
    "cuda_sage3_fp8": "Native CUDA FP8\nSage3",
}
"""Compact labels for the selected mixed self/cross end-to-end configurations."""

BenchmarkValues = dict[str, dict[str, float]]
Panel = tuple[str, str]


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse benchmark input and plot output paths.

    Args:
        argv: Command-line arguments; ``None`` reads ``sys.argv``.

    Returns:
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Plot Omnidreams median benchmark latency as bar charts."
    )
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        default=_DEFAULT_INPUT,
        help=f"pytest-benchmark JSON path (default: {_DEFAULT_INPUT})",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=_DEFAULT_OUTPUT_DIR,
        help=f"output directory (default: {_DEFAULT_OUTPUT_DIR})",
    )
    return parser.parse_args(argv)


def _load_results(input_path: Path) -> tuple[BenchmarkValues, list[str], str]:
    """Load Omnidreams median timings and environment metadata.

    Args:
        input_path: Pytest-benchmark JSON file to parse.

    Returns:
        Median milliseconds by group and implementation, configuration order,
        and plot subtitle.

    Raises:
        SystemExit: The input cannot be read or lacks complete Omnidreams data.
    """
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

    values: BenchmarkValues = {group: {} for group, _ in _PANELS}
    configurations: list[str] = []
    first_extra: dict[str, object] | None = None

    for record in payload["benchmarks"]:
        if not isinstance(record, dict):
            continue
        group = record.get("group")
        if not isinstance(group, str) or group not in values:
            continue
        extra = record.get("extra_info")
        implementation = (
            extra.get("implementation") if isinstance(extra, dict) else None
        )
        if not isinstance(implementation, str) or not implementation:
            raise SystemExit(
                f"Benchmark {record.get('name', '<unnamed>')} has no implementation"
            )

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
        if implementation in values[group]:
            raise SystemExit(f"Duplicate benchmark for {group} and {implementation}")

        values[group][implementation] = median * 1000.0
        if implementation not in configurations:
            configurations.append(implementation)
        if first_extra is None:
            first_extra = extra

    missing = [group for group, results in values.items() if not results]
    if missing or first_extra is None:
        raise SystemExit(
            f"Benchmark JSON {input_path} lacks groups: {', '.join(missing)}"
        )
    return values, configurations, _subtitle(payload, first_extra)


def _subtitle(payload: dict[str, object], extra: dict[str, object]) -> str:
    """Build a compact benchmark-environment subtitle."""
    parts = ["Median latency in ms (lower is faster)"]
    gpu = extra.get("gpu")
    if isinstance(gpu, str) and gpu:
        parts.append(gpu)
    warmups = extra.get("warmup_rounds")
    rounds = extra.get("benchmark_rounds")
    if isinstance(warmups, int) and isinstance(rounds, int):
        parts.append(f"{warmups} warmups / {rounds} rounds")
    commit_info = payload.get("commit_info")
    commit_id = commit_info.get("id") if isinstance(commit_info, dict) else None
    if isinstance(commit_id, str) and commit_id:
        parts.append(f"commit {commit_id[:10]}")
    timestamp = payload.get("datetime")
    if isinstance(timestamp, str) and timestamp:
        parts.append(timestamp)
    return " · ".join(parts)


def _configuration_label(
    implementation: str,
    *,
    end_to_end: bool = False,
) -> str:
    """Format a stable implementation name as a compact axis label.

    Args:
        implementation: Stable benchmark implementation identifier.
        end_to_end: Whether to describe both self- and cross-attention branches.

    Returns:
        Compact multi-line label for the plot axis.
    """
    if end_to_end:
        label = _END_TO_END_LABELS.get(implementation)
        if label is not None:
            return label
    if implementation == "triton_cudnn_bf16_full_omnidreams_cross":
        return _END_TO_END_LABELS[implementation]
    if implementation == "omnidreams_torch":
        return "Omnidreams\nPyTorch"
    if implementation == "cuda":
        return "Native\nCUDA"
    if implementation.startswith("triton_"):
        parts = implementation.removeprefix("triton_").split("_", maxsplit=2)
        if len(parts) == 3:
            backend, precision, fusion = parts
            backend_label = "cuDNN" if backend == "cudnn" else backend.upper()
            fusion_label = (
                "FUSE QKV" if fusion == "full" else fusion.replace("_", " ").upper()
            )
            return f"{backend_label}\n{precision.upper()}\n{fusion_label}"
    return implementation.replace("_", "\n")


def _bar_label(
    latency_ms: float,
    reference_ms: float,
    *,
    is_reference: bool,
) -> str:
    """Format latency and relative performance against PyTorch.

    Args:
        latency_ms: Bar latency in milliseconds.
        reference_ms: PyTorch reference latency in milliseconds.
        is_reference: Whether the bar is the PyTorch reference.

    Returns:
        Two-line latency and relative-performance annotation.
    """
    if latency_ms < 1:
        latency_label = f"{latency_ms:.3f} ms"
    elif latency_ms < 10:
        latency_label = f"{latency_ms:.2f} ms"
    else:
        latency_label = f"{latency_ms:.1f} ms"
    if is_reference:
        return f"{latency_label}\nreference"
    if latency_ms < reference_ms:
        return f"{latency_label}\n{(reference_ms / latency_ms - 1) * 100:.0f}% faster"
    if latency_ms > reference_ms:
        return f"{latency_label}\n{(latency_ms / reference_ms - 1) * 100:.0f}% slower"
    return f"{latency_label}\nsame as reference"


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
    """Write three aligned median-latency bar charts as one PNG.

    Args:
        output_path: Destination PNG file.
        title: Figure title.
        panels: Benchmark group and display-title pairs.
        values: Median milliseconds by group and implementation.
        configurations: Stable implementation order.
        subtitle: Benchmark environment summary.
    """
    end_to_end = panels == _END_TO_END_PANELS
    panel_configurations = _panel_configurations(panels, values, configurations)
    figure, axes = plt.subplots(
        len(panels),
        1,
        figsize=(max(16.0, len(panel_configurations) * 1.3), 15.0),
        layout="constrained",
    )
    axes_list = list(axes)
    for axes_item, (group, panel_title) in zip(axes_list, panels, strict=True):
        reference = values[group]["omnidreams_torch"]
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
            if configuration == "omnidreams_torch"
            else "C1"
            if configuration == "cuda"
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
        axes_item.set_ylabel("Median latency (ms)")

        for index, (bar, latency, configuration) in enumerate(
            zip(bars, latencies, ordered_configurations, strict=True)
        ):
            if not math.isfinite(latency):
                bar.set_visible(False)
                axes_item.text(
                    index,
                    0.02,
                    "N/A",
                    transform=axes_item.get_xaxis_transform(),
                    ha="center",
                    va="bottom",
                )
                continue
            axes_item.annotate(
                _bar_label(
                    latency,
                    reference,
                    is_reference=configuration == "omnidreams_torch",
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
                _configuration_label(
                    configuration,
                    end_to_end=end_to_end,
                )
                for configuration in ordered_configurations
            ],
        )

    figure.supxlabel(
        "Configuration (self-attention + cross-attention)"
        if end_to_end
        else "Configuration (SDPA backend / precision / QKV fusion)"
    )
    figure.suptitle(f"{title}\n{subtitle}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path)
    plt.close(figure)


def main(argv: list[str] | None = None) -> None:
    """Generate module and end-to-end Omnidreams benchmark figures."""
    args = _parse_args(argv)
    values, configurations, subtitle = _load_results(args.input)
    outputs = (
        (
            args.output_dir / "modules.png",
            "Omnidreams module benchmarks",
            _MODULE_PANELS,
        ),
        (
            args.output_dir / "network_pipeline.png",
            "Omnidreams network and pipeline benchmarks",
            _END_TO_END_PANELS,
        ),
    )
    for output_path, title, panels in outputs:
        _write_bar_figure(
            output_path,
            title,
            panels,
            values,
            configurations,
            subtitle,
        )
        print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
