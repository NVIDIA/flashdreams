# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Render disaggregated CP6 versus aggregated CP8 wall time and HBM as SVG."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any, TypedDict

_COLORS = {
    "Encoder": "#59a14f",
    "Input handoff": "#8cd17d",
    "DiT denoise": "#4e79a7",
    "KV finalize": "#b07aa1",
    "Output handoff": "#76b7b2",
    "Decoder": "#f28e2b",
    "Coordination": "#bab0ac",
}


class _WallRow(TypedDict):
    """One stacked wall-time row."""

    label: str
    total: float
    parts: dict[str, float]


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("disaggregated", type=Path)
    parser.add_argument("aggregated", type=Path)
    parser.add_argument("output", type=Path)
    return parser.parse_args()


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _wall_rows(
    disaggregated: dict[str, Any],
    aggregated: dict[str, Any],
) -> list[_WallRow]:
    disagg = disaggregated["summary"]
    agg = aggregated["summary"]
    rows: list[_WallRow] = [
        {
            "label": "Disaggregated CP6 ring",
            "total": disagg["latency_ms"]["median"],
            "parts": {
                "Encoder": disagg["encoder_ms"]["median"],
                "Input handoff": (
                    disagg["encoder_to_cp_leader"]["handoff_ms"]["median"]
                    + disagg["cp_input_fanout_ms"]["median"]
                ),
                "DiT denoise": disagg["dit_ms"]["median"],
                "KV finalize": disagg["finalize_ms"]["median"],
                "Output handoff": disagg["cp_leader_to_decoder"]["handoff_ms"][
                    "median"
                ],
                "Decoder": disagg["decoder_ms"]["median"],
            },
        },
        {
            "label": "Aggregated CP8 Ulysses",
            "total": agg["latency_ms"]["median"],
            "parts": {
                "Encoder": agg["encoder_ms"]["median"],
                "DiT denoise": agg["dit_ms"]["median"],
                "KV finalize": agg["finalize_ms"]["median"],
                "Decoder": agg["decoder_ms"]["median"],
            },
        },
    ]
    for row in rows:
        row["parts"]["Coordination"] = max(
            0.0,
            row["total"] - sum(row["parts"].values()),
        )
    return rows


def _svg(
    disaggregated: dict[str, Any],
    aggregated: dict[str, Any],
) -> str:
    width, height = 1200, 720
    rows = _wall_rows(disaggregated, aggregated)
    max_wall = 800.0
    chart_x, chart_w = 245.0, 860.0
    wall_y = [145.0, 225.0]
    bar_h = 48.0
    pieces = [
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
            f'height="{height}" viewBox="0 0 {width} {height}" role="img" '
            'aria-labelledby="title description">'
        ),
        (
            '<title id="title">LingBot disaggregated CP6 and aggregated CP8 '
            "performance comparison</title>"
        ),
        (
            '<desc id="description">Median per-chunk component wall time and '
            "per-rank peak allocated HBM on eight H100 GPUs.</desc>"
        ),
        "<style>",
        "text{font-family:Inter,Arial,sans-serif;fill:#202124}",
        ".title{font-size:26px;font-weight:600}.subtitle{font-size:15px;fill:#5f6368}",
        ".axis{font-size:12px;fill:#5f6368}.label{font-size:14px;font-weight:600}",
        ".value{font-size:12px;font-weight:600}.grid{stroke:#dadce0;stroke-width:1}",
        "</style>",
        '<rect width="1200" height="720" fill="#ffffff"/>',
        '<text x="40" y="42" class="title">LingBot: stage-local CP6 vs full-pipeline CP8</text>',
        (
            '<text x="40" y="68" class="subtitle">8× H100 80 GB · BF16 · '
            "six warmup + five measured blocks · CP6 832×464 · CP8 832×448</text>"
        ),
        '<text x="40" y="108" class="label">Median steady-state wall time</text>',
    ]
    for tick in range(0, 801, 100):
        x = chart_x + chart_w * tick / max_wall
        pieces.extend(
            [
                f'<line x1="{x:.1f}" y1="124" x2="{x:.1f}" y2="295" class="grid"/>',
                (
                    f'<text x="{x:.1f}" y="315" class="axis" '
                    f'text-anchor="middle">{tick}</text>'
                ),
            ]
        )
    pieces.append(
        '<text x="1155" y="315" class="axis" text-anchor="end">milliseconds</text>'
    )
    for row, y in zip(rows, wall_y):
        pieces.append(
            f'<text x="225" y="{y + 29:.1f}" class="label" text-anchor="end">'
            f"{html.escape(row['label'])}</text>"
        )
        x = chart_x
        for label, value in row["parts"].items():
            segment_w = chart_w * value / max_wall
            pieces.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{segment_w:.1f}" '
                f'height="{bar_h:.1f}" fill="{_COLORS[label]}"/>'
            )
            if segment_w >= 45:
                pieces.append(
                    f'<text x="{x + segment_w / 2:.1f}" y="{y + 29:.1f}" '
                    f'class="value" text-anchor="middle">{value:.0f}</text>'
                )
            x += segment_w
        pieces.append(
            f'<text x="{x + 8:.1f}" y="{y + 29:.1f}" class="value">'
            f"{row['total']:.0f} ms</text>"
        )

    for index, label in enumerate(_COLORS):
        x = 75 + index * 154
        pieces.extend(
            [
                f'<rect x="{x}" y="340" width="12" height="12" fill="{_COLORS[label]}"/>',
                f'<text x="{x + 18}" y="350" class="axis">{html.escape(label)}</text>',
            ]
        )

    pieces.extend(
        [
            '<text x="40" y="400" class="label">Peak allocated HBM by rank</text>',
            (
                '<text x="40" y="423" class="subtitle">Stage-local CP6 totals '
                "251.07 GiB; eight full-pipeline replicas total 327.03 GiB (+30.3%)</text>"
            ),
        ]
    )
    mem_base_y, mem_top_y = 626.0, 448.0
    mem_h = mem_base_y - mem_top_y
    for tick in (0, 10, 20, 30, 40, 50):
        y = mem_base_y - mem_h * tick / 50.0
        pieces.extend(
            [
                f'<line x1="75" y1="{y:.1f}" x2="1160" y2="{y:.1f}" class="grid"/>',
                f'<text x="65" y="{y + 4:.1f}" class="axis" text-anchor="end">{tick}</text>',
            ]
        )
    pieces.append(
        '<text x="30" y="540" class="axis" text-anchor="middle" '
        'transform="rotate(-90 30 540)">GiB</text>'
    )

    disagg_memory = disaggregated["environment"]["peak_memory_gib_by_rank"]
    agg_memory = aggregated["summary"]["memory"]["peak_gib_by_rank"]
    memory_groups = (
        (
            "Disaggregated CP6",
            disagg_memory,
            ["Encoder", *(["DiT denoise"] * 6), "Decoder"],
            105.0,
        ),
        (
            "Aggregated CP8",
            agg_memory,
            ["DiT denoise"] * 8,
            650.0,
        ),
    )
    for group, values, stages, start_x in memory_groups:
        for rank, (value, stage) in enumerate(zip(values, stages)):
            x = start_x + rank * 53.0
            bar_height = mem_h * value / 50.0
            y = mem_base_y - bar_height
            pieces.extend(
                [
                    f'<rect x="{x:.1f}" y="{y:.1f}" width="34" '
                    f'height="{bar_height:.1f}" fill="{_COLORS[stage]}"/>',
                    (
                        f'<text x="{x + 17:.1f}" y="{y - 6:.1f}" class="axis" '
                        f'text-anchor="middle">{value:.1f}</text>'
                    ),
                    (
                        f'<text x="{x + 17:.1f}" y="645" class="axis" '
                        f'text-anchor="middle">G{rank}</text>'
                    ),
                ]
            )
        center = start_x + (len(values) - 1) * 53.0 / 2 + 17.0
        pieces.append(
            f'<text x="{center:.1f}" y="681" class="label" text-anchor="middle">'
            f"{html.escape(group)}</text>"
        )
    pieces.append(
        '<text x="650" y="704" class="axis">Aggregated bars contain encoder + DiT + decoder on every GPU.</text>'
    )
    pieces.append("</svg>")
    return "\n".join(pieces) + "\n"


def main() -> None:
    """Write the SVG comparison."""
    args = _args()
    args.output.write_text(_svg(_read(args.disaggregated), _read(args.aggregated)))


if __name__ == "__main__":
    main()
