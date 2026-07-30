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

"""Render the tracked three-stage wall-time and GPU-memory comparison as SVG."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any

_COLORS = {
    "Encoder": "#59a14f",
    "Encoder → DiT": "#8cd17d",
    "DiT": "#4e79a7",
    "DiT → decoder": "#76b7b2",
    "Decoder": "#f28e2b",
    "Coordination": "#bab0ac",
}


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("scaled", type=Path)
    parser.add_argument("output", type=Path)
    return parser.parse_args()


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _walltime(baseline: dict[str, Any], scaled: dict[str, Any]) -> list[dict[str, Any]]:
    base = baseline["summary"]
    base_parts = {
        "Encoder": base["encoder_ms"]["median"],
        "Encoder → DiT": base["encoder_to_dit"]["handoff_ms"]["median"],
        "DiT": base["dit_ms"]["median"] + base["finalize_ms"]["median"],
        "DiT → decoder": base["dit_to_decoder"]["handoff_ms"]["median"],
        "Decoder": base["decoder_ms"]["median"],
    }
    scaled_summary = scaled["summary"]
    scaled_parts = {
        "Encoder": scaled_summary["encoder_wave_ms"]["median"],
        "Encoder → DiT": scaled_summary["encoder_to_dit"][
            "aggregate_handoff_ms_per_wave"
        ]["median"],
        "DiT": scaled_summary["dit_critical_path_ms"]["median"],
        "DiT → decoder": scaled_summary["dit_to_decoder"][
            "aggregate_handoff_ms_per_wave"
        ]["median"],
        "Decoder": scaled_summary["decoder_wave_ms"]["median"],
    }
    rows: list[dict[str, Any]] = [
        {
            "label": "1E : 1D : 1V",
            "total": base["latency_ms"]["median"],
            "parts": base_parts,
        },
        {
            "label": "1E : 6D : 1V",
            "total": scaled_summary["wave_latency_ms"]["median"],
            "parts": scaled_parts,
        },
    ]
    for row in rows:
        row["parts"]["Coordination"] = max(
            0.0, row["total"] - sum(row["parts"].values())
        )
    return rows


def _memory(
    baseline: dict[str, Any], scaled: dict[str, Any]
) -> list[tuple[str, float, str]]:
    base = baseline["environment"]["peak_memory_gib_by_stage"]
    scaled_memory = scaled["environment"]["peak_memory_gib_by_rank"]
    return [
        ("E0", base["encoder"], "Encoder"),
        ("D1", base["dit"], "DiT"),
        ("V2", base["decoder"], "Decoder"),
        ("E0", scaled_memory[0], "Encoder"),
        *[
            (f"D{rank}", scaled_memory[rank], "DiT")
            for rank in range(1, len(scaled_memory) - 1)
        ],
        ("V7", scaled_memory[-1], "Decoder"),
    ]


def _svg(baseline: dict[str, Any], scaled: dict[str, Any]) -> str:
    width, height = 1200, 720
    wall_rows = _walltime(baseline, scaled)
    max_wall = 3000.0
    chart_x, chart_w = 165.0, 965.0
    wall_y = [142.0, 232.0]
    bar_h = 52.0
    pieces = [
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}" role="img" '
            'aria-labelledby="title description">'
        ),
        '<title id="title">LingBot disaggregated inference wall time and GPU memory</title>',
        (
            '<desc id="description">Stacked wall-time comparison for one versus six '
            "DiT workers and per-rank peak allocated GPU memory.</desc>"
        ),
        "<style>",
        "text{font-family:Inter,Arial,sans-serif;fill:#202124}",
        ".title{font-size:26px;font-weight:600}.subtitle{font-size:15px;fill:#5f6368}",
        ".axis{font-size:13px;fill:#5f6368}.label{font-size:15px;font-weight:600}",
        ".value{font-size:13px;font-weight:600}.grid{stroke:#dadce0;stroke-width:1}",
        "</style>",
        '<rect width="1200" height="720" fill="#ffffff"/>',
        '<text x="40" y="43" class="title">LingBot stage allocation: wall time and memory</text>',
        (
            '<text x="40" y="70" class="subtitle">H100 80 GB · BF16 · 832×464 · '
            "six warmup and five measured waves</text>"
        ),
        '<text x="40" y="110" class="label">Median steady-state wall time</text>',
    ]
    for tick in range(0, 3001, 500):
        x = chart_x + chart_w * tick / max_wall
        pieces.extend(
            [
                f'<line x1="{x:.1f}" y1="126" x2="{x:.1f}" y2="303" class="grid"/>',
                f'<text x="{x:.1f}" y="322" class="axis" text-anchor="middle">{tick}</text>',
            ]
        )
    pieces.append(
        '<text x="1130" y="322" class="axis" text-anchor="end">milliseconds</text>'
    )
    for row, y in zip(wall_rows, wall_y):
        pieces.append(
            f'<text x="145" y="{y + 32:.1f}" class="label" text-anchor="end">'
            f"{html.escape(row['label'])}</text>"
        )
        x = chart_x
        for label, value in row["parts"].items():
            segment_w = chart_w * value / max_wall
            pieces.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{segment_w:.1f}" '
                f'height="{bar_h:.1f}" fill="{_COLORS[label]}"/>'
            )
            if segment_w >= 48:
                pieces.append(
                    f'<text x="{x + segment_w / 2:.1f}" y="{y + 31:.1f}" '
                    f'class="value" text-anchor="middle">{value:.0f}</text>'
                )
            x += segment_w
        pieces.append(
            f'<text x="{x + 9:.1f}" y="{y + 32:.1f}" class="value">'
            f"{row['total']:.0f} ms</text>"
        )

    legend_x = 165
    for index, label in enumerate(_COLORS):
        x = legend_x + index * 157
        pieces.extend(
            [
                f'<rect x="{x}" y="342" width="13" height="13" fill="{_COLORS[label]}"/>',
                f'<text x="{x + 19}" y="353" class="axis">{html.escape(label)}</text>',
            ]
        )

    pieces.extend(
        [
            '<text x="40" y="402" class="label">Peak allocated GPU memory by rank</text>',
            (
                '<text x="40" y="426" class="subtitle">E = encoder, D = DiT, V = decoder; '
                "the gap to 80 GiB is headroom, not free schedulable memory</text>"
            ),
        ]
    )
    memory = _memory(baseline, scaled)
    groups = [(memory[:3], 155.0, "1E : 1D : 1V"), (memory[3:], 600.0, "1E : 6D : 1V")]
    mem_base_y = 650.0
    mem_top_y = 455.0
    mem_h = mem_base_y - mem_top_y
    for tick in (0, 20, 40, 60, 80):
        y = mem_base_y - mem_h * tick / 80.0
        pieces.extend(
            [
                f'<line x1="90" y1="{y:.1f}" x2="1145" y2="{y:.1f}" class="grid"/>',
                f'<text x="78" y="{y + 5:.1f}" class="axis" text-anchor="end">{tick}</text>',
            ]
        )
    pieces.append(
        '<text x="43" y="555" class="axis" text-anchor="middle" '
        'transform="rotate(-90 43 555)">GiB</text>'
    )
    for rows, start_x, group_label in groups:
        gap = 68.0 if len(rows) > 3 else 96.0
        bar_width = 43.0
        center = start_x + ((len(rows) - 1) * gap + bar_width) / 2
        pieces.append(
            f'<text x="{center:.1f}" y="692" class="label" '
            f'text-anchor="middle">{group_label}</text>'
        )
        for index, (rank, value, stage) in enumerate(rows):
            x = start_x + index * gap
            bar_height = mem_h * value / 80.0
            y = mem_base_y - bar_height
            pieces.extend(
                [
                    (
                        f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width:.1f}" '
                        f'height="{bar_height:.1f}" fill="{_COLORS[stage]}"/>'
                    ),
                    (
                        f'<text x="{x + bar_width / 2:.1f}" y="{y - 7:.1f}" '
                        f'class="value" text-anchor="middle">{value:.1f}</text>'
                    ),
                    (
                        f'<text x="{x + bar_width / 2:.1f}" y="670" '
                        f'class="axis" text-anchor="middle">{rank}</text>'
                    ),
                ]
            )
    pieces.append("</svg>")
    return "\n".join(pieces) + "\n"


def main() -> None:
    """Read benchmark JSON documents and write an SVG comparison."""
    args = _args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(_svg(_read(args.baseline), _read(args.scaled)))


if __name__ == "__main__":
    main()
