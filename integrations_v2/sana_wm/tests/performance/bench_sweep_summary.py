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

"""Aggregate SANA-WM precision benchmark summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_item(raw: str) -> dict[str, Any]:
    if ":" not in raw:
        raise argparse.ArgumentTypeError(
            f"expected LABEL:/path/to/bench.json, got {raw!r}"
        )
    label, path_raw = raw.split(":", 1)
    label = label.strip()
    path = Path(path_raw)
    if not label:
        raise argparse.ArgumentTypeError(f"empty label in {raw!r}")
    if not path.exists():
        raise argparse.ArgumentTypeError(f"bench summary does not exist: {path}")
    summary = json.loads(path.read_text(encoding="utf-8"))
    benchmark = summary.get("benchmark")
    if not isinstance(benchmark, dict):
        raise argparse.ArgumentTypeError(f"{path} has no benchmark object")
    official = benchmark.get("official")
    flashdreams = benchmark.get("flashdreams")
    if official is not None and not isinstance(official, (int, float)):
        raise argparse.ArgumentTypeError(
            f"{path} benchmark official value must be numeric or null"
        )
    if flashdreams is not None and not isinstance(flashdreams, (int, float)):
        raise argparse.ArgumentTypeError(
            f"{path} benchmark flashdreams value must be numeric or null"
        )
    if official is None and flashdreams is None:
        raise argparse.ArgumentTypeError(
            f"{path} benchmark must contain an official or flashdreams value"
        )
    metric = benchmark.get("metric")
    if not isinstance(metric, str):
        raise argparse.ArgumentTypeError(f"{path} benchmark must contain a metric")
    return {
        "label": label,
        "path": str(path),
        "official": float(official) if isinstance(official, (int, float)) else None,
        "flashdreams": (
            float(flashdreams) if isinstance(flashdreams, (int, float)) else None
        ),
        "metric": metric,
        "unit": benchmark.get("unit", "ms"),
        "variant": benchmark.get("variant")
        or summary.get("variant")
        or summary.get("inputs", {}).get("variant", "bidirectional"),
        "inputs": summary.get("inputs", {}),
    }


def _render_chart(rows: list[dict[str, Any]]) -> str:
    has_official = any(row["official"] is not None for row in rows)
    has_flashdreams = any(row["flashdreams"] is not None for row in rows)
    unit_label = (
        "ms/chunk"
        if rows and rows[0]["metric"] == "steady_state_generation_ms_per_chunk"
        else "ms"
    )
    lines = [
        (
            f"# SANA-WM Precision Sweep Summary ({unit_label})"
            if has_flashdreams
            else f"# SANA-WM Precision Sweep Upstream Summary ({unit_label})"
        ),
        "",
    ]
    columns = ["precision"]
    aligns = ["---"]
    if has_official:
        columns.append("official")
        aligns.append("---:")
    if has_flashdreams:
        columns.append("flashdreams")
        aligns.append("---:")
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("| " + " | ".join(aligns) + " |")
    for row in rows:
        values = [row["label"]]
        if has_official:
            values.append(
                f"{row['official']:.2f}" if row["official"] is not None else "n/a"
            )
        if has_flashdreams:
            values.append(
                f"{row['flashdreams']:.2f}" if row["flashdreams"] is not None else "n/a"
            )
        lines.append("| " + " | ".join(values) + " |")
    lines.append("")
    return "\n".join(lines)


def _render_report(rows: list[dict[str, Any]]) -> str:
    has_official = any(row["official"] is not None for row in rows)
    has_flashdreams = any(row["flashdreams"] is not None for row in rows)
    metric = rows[0]["metric"] if rows else "steady_state_generation_ms_per_clip"
    metric_text = (
        "steady-state generation latency per produced chunk"
        if metric == "steady_state_generation_ms_per_chunk"
        else "steady-state in-process generation latency per generated clip"
    )
    lines = [
        "# SANA-WM precision benchmark sweep",
        "",
        f"The chart metric is {metric_text}.",
        "Model-card chart data is grouped by GPU/device in each precision "
        "subdirectory's `perf.md`.",
        "",
    ]
    columns = ["precision"]
    aligns = ["---"]
    if has_official:
        columns.append("official")
        aligns.append("---:")
    if has_flashdreams:
        columns.append("FlashDreams")
        aligns.append("---:")
    columns.append("source")
    aligns.append("---")
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("| " + " | ".join(aligns) + " |")
    for row in rows:
        values = [row["label"]]
        if has_official:
            values.append(
                f"{row['official']:.2f} ms" if row["official"] is not None else "n/a"
            )
        if has_flashdreams:
            values.append(
                f"{row['flashdreams']:.2f} ms"
                if row["flashdreams"] is not None
                else "n/a"
            )
        values.append(f"`{row['path']}`")
        lines.append("| " + " | ".join(values) + " |")
    lines.append("")
    return "\n".join(lines)


def _validate_rows(rows: list[dict[str, Any]]) -> None:
    for row in rows:
        if row["variant"] == "streaming":
            continue
        if row["metric"] != "steady_state_generation_ms_per_clip":
            raise ValueError(
                "SANA-WM_bidirectional benchmark summaries must use "
                "steady_state_generation_ms_per_clip; discard stale "
                "generation_ms_per_clip data and rerun bench.sh."
            )
        inputs = row.get("inputs", {})
        stage1_precision = inputs.get("stage1_precision")
        refiner_precision = inputs.get("refiner_precision")
        label = row["label"].strip().lower()
        if (
            label in {"fp8", "fp4"}
            or stage1_precision in {"fp8", "fp4"}
            or refiner_precision in {"fp8", "fp4"}
        ):
            if row["official"] is not None:
                raise ValueError(
                    "upstream SANA-WM_bidirectional benchmarks are BF16-only; "
                    "FP8 and FP4 bidirectional rows must be FlashDreams-only."
                )
            if row["flashdreams"] is None:
                raise ValueError(
                    "FP8 and FP4 bidirectional rows must contain a FlashDreams value."
                )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--item",
        action="append",
        type=_load_item,
        required=True,
        help="Precision row as LABEL:/path/to/bench.json. Repeat once per row.",
    )
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--output-chart-md", type=Path, required=True)
    args = parser.parse_args(argv)

    metrics = {row["metric"] for row in args.item}
    if len(metrics) != 1:
        raise ValueError(f"cannot aggregate mixed benchmark metrics: {sorted(metrics)}")
    units = {row["unit"] for row in args.item}
    if len(units) != 1:
        raise ValueError(f"cannot aggregate mixed benchmark units: {sorted(units)}")
    _validate_rows(args.item)
    has_official = any(row["official"] is not None for row in args.item)
    has_flashdreams = any(row["flashdreams"] is not None for row in args.item)
    payload = {
        "benchmark": {
            "metric": args.item[0]["metric"],
            "unit": args.item[0]["unit"],
            "has_official": has_official,
            "has_flashdreams": has_flashdreams,
        },
        "rows": args.item,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    report = _render_report(args.item)
    args.output_md.write_text(report, encoding="utf-8")
    args.output_chart_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_chart_md.write_text(_render_chart(args.item), encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
