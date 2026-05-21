# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate static SVG benchmark charts for docs."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "docs" / "benchmarks" / "benchmark_results.json"
OUT_DIR = ROOT / "docs" / "source" / "_static" / "perf"

METHOD_ORDER = ["flashdreams", "official", "fastvideo", "lightx2v"]
METHOD_COLORS = {
    "flashdreams": "#76B900",
    "official": "#4C78A8",
    "fastvideo": "#F58518",
    "lightx2v": "#E45756",
}
HARDWARE_ORDER = ["H100", "GB200", "GB300"]


def _escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _load_records() -> list[dict]:
    payload = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    return payload["records"]


def _select_values(
    records: list[dict], workload: str, parallelism: str
) -> dict[str, dict[str, float | None]]:
    values = {hw: {method: None for method in METHOD_ORDER} for hw in HARDWARE_ORDER}
    for rec in records:
        if rec.get("workload") != workload:
            continue
        if rec.get("parallelism") != parallelism:
            continue
        hw = rec.get("hardware")
        method = rec.get("method")
        if hw not in values or method not in values[hw]:
            continue
        if rec.get("status") != "pass":
            continue
        values[hw][method] = rec.get("metrics", {}).get("total_ms")
    return values


def _render_chart(
    title: str,
    values: dict[str, dict[str, float | None]],
    output_path: Path,
) -> None:
    width = 980
    height = 460
    margin_left = 80
    margin_right = 40
    margin_top = 70
    margin_bottom = 75
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom

    all_vals = [
        v
        for hw in HARDWARE_ORDER
        for v in values[hw].values()
        if isinstance(v, (int, float))
    ]
    y_max = max(all_vals) * 1.15 if all_vals else 1.0

    group_count = len(HARDWARE_ORDER)
    group_w = plot_w / group_count
    bar_w = 32
    bar_gap = 10
    group_inner = len(METHOD_ORDER) * bar_w + (len(METHOD_ORDER) - 1) * bar_gap

    lines: list[str] = []
    lines.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-label="{_escape(title)}">'
    )
    lines.append('<rect width="100%" height="100%" fill="white"/>')
    lines.append(
        f'<text x="{width / 2}" y="34" text-anchor="middle" font-size="20" font-family="Arial, sans-serif" fill="#111">{_escape(title)}</text>'
    )

    # Axes
    x0, y0 = margin_left, margin_top + plot_h
    lines.append(
        f'<line x1="{x0}" y1="{margin_top}" x2="{x0}" y2="{y0}" stroke="#333" stroke-width="1"/>'
    )
    lines.append(
        f'<line x1="{x0}" y1="{y0}" x2="{x0 + plot_w}" y2="{y0}" stroke="#333" stroke-width="1"/>'
    )

    # Y ticks
    ticks = 5
    for i in range(ticks + 1):
        v = y_max * i / ticks
        y = y0 - (v / y_max) * plot_h
        lines.append(
            f'<line x1="{x0 - 5}" y1="{y:.2f}" x2="{x0 + plot_w}" y2="{y:.2f}" stroke="#E0E0E0" stroke-width="1"/>'
        )
        lines.append(
            f'<text x="{x0 - 10}" y="{y + 4:.2f}" text-anchor="end" font-size="11" font-family="Arial, sans-serif" fill="#444">{int(v)}</text>'
        )

    lines.append(
        f'<text x="{18}" y="{margin_top + plot_h / 2}" transform="rotate(-90, 18, {margin_top + plot_h / 2})" text-anchor="middle" font-size="12" font-family="Arial, sans-serif" fill="#333">Latency (ms)</text>'
    )

    # Bars
    for g_idx, hw in enumerate(HARDWARE_ORDER):
        group_x0 = margin_left + g_idx * group_w + (group_w - group_inner) / 2
        for m_idx, method in enumerate(METHOD_ORDER):
            val = values[hw][method]
            bx = group_x0 + m_idx * (bar_w + bar_gap)
            if val is None:
                by = y0 - 22
                lines.append(
                    f'<rect x="{bx:.2f}" y="{by:.2f}" width="{bar_w}" height="22" fill="none" stroke="#999" stroke-dasharray="4,3"/>'
                )
                lines.append(
                    f'<text x="{bx + bar_w / 2:.2f}" y="{by - 4:.2f}" text-anchor="middle" font-size="10" font-family="Arial, sans-serif" fill="#666">N/A</text>'
                )
                continue

            bh = (val / y_max) * plot_h
            by = y0 - bh
            color = METHOD_COLORS[method]
            lines.append(
                f'<rect x="{bx:.2f}" y="{by:.2f}" width="{bar_w}" height="{bh:.2f}" fill="{color}"/>'
            )
            lines.append(
                f'<text x="{bx + bar_w / 2:.2f}" y="{by - 6:.2f}" text-anchor="middle" font-size="10" font-family="Arial, sans-serif" fill="#222">{val:.0f}</text>'
            )

        lines.append(
            f'<text x="{group_x0 + group_inner / 2:.2f}" y="{y0 + 22}" text-anchor="middle" font-size="12" font-family="Arial, sans-serif" fill="#222">{_escape(hw)}</text>'
        )

    # Legend
    legend_x = margin_left
    legend_y = 50
    step = 180
    for idx, method in enumerate(METHOD_ORDER):
        x = legend_x + idx * step
        lines.append(
            f'<rect x="{x}" y="{legend_y - 10}" width="12" height="12" fill="{METHOD_COLORS[method]}" stroke="#666" stroke-width="0.6"/>'
        )
        lines.append(
            f'<text x="{x + 18}" y="{legend_y}" font-size="12" font-family="Arial, sans-serif" fill="#222">{_escape(method)}</text>'
        )

    lines.append("</svg>")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    records = _load_records()

    self_forcing = _select_values(
        records, workload="self_forcing_block6", parallelism="1xGPU"
    )
    _render_chart(
        title="Self-Forcing (6th block) Total Latency",
        values=self_forcing,
        output_path=OUT_DIR / "self_forcing_total_ms.svg",
    )

    lingbot = _select_values(
        records, workload="lingbot_world_block6", parallelism="4xGPU"
    )
    _render_chart(
        title="Lingbot-World (6th block, 4xGPU) Total Latency",
        values=lingbot,
        output_path=OUT_DIR / "lingbot_total_ms.svg",
    )

    print(f"Generated charts in {OUT_DIR}")


if __name__ == "__main__":
    main()
