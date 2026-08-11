# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Render LingBot CP1, CP4, and CP6 single-session results as SVG."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any

_COLORS = {
    "Encoder": "#59a14f",
    "Input handoff": "#8cd17d",
    "DiT": "#4e79a7",
    "Output handoff": "#76b7b2",
    "Decoder": "#f28e2b",
    "Coordination": "#bab0ac",
}


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("cp4", type=Path)
    parser.add_argument("cp6", type=Path)
    parser.add_argument("output", type=Path)
    return parser.parse_args()


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _walltime(
    baseline: dict[str, Any],
    cp4: dict[str, Any],
    cp6: dict[str, Any],
) -> list[dict[str, Any]]:
    base = baseline["summary"]
    rows: list[dict[str, Any]] = [
        {
            "label": "CP1",
            "total": base["latency_ms"]["median"],
            "parts": {
                "Encoder": base["encoder_ms"]["median"],
                "Input handoff": base["encoder_to_dit"]["handoff_ms"]["median"],
                "DiT": (base["dit_ms"]["median"] + base["finalize_ms"]["median"]),
                "Output handoff": base["dit_to_decoder"]["handoff_ms"]["median"],
                "Decoder": base["decoder_ms"]["median"],
            },
        }
    ]
    for label, document in (("CP4 Ulysses", cp4), ("CP6 ring", cp6)):
        summary = document["summary"]
        rows.append(
            {
                "label": label,
                "total": summary["latency_ms"]["median"],
                "parts": {
                    "Encoder": summary["encoder_ms"]["median"],
                    "Input handoff": (
                        summary["encoder_to_cp_leader"]["handoff_ms"]["median"]
                        + summary["cp_input_fanout_ms"]["median"]
                    ),
                    "DiT": summary["dit_critical_path_ms"]["median"],
                    "Output handoff": summary["cp_leader_to_decoder"]["handoff_ms"][
                        "median"
                    ],
                    "Decoder": summary["decoder_ms"]["median"],
                },
            }
        )
    for row in rows:
        row["parts"]["Coordination"] = max(
            0.0,
            row["total"] - sum(row["parts"].values()),
        )
    return rows


def _memory_rows(
    baseline: dict[str, Any],
    cp4: dict[str, Any],
    cp6: dict[str, Any],
) -> list[tuple[str, list[tuple[str, float, str]]]]:
    base = baseline["environment"]["peak_memory_gib_by_stage"]

    def cp_rows(document: dict[str, Any]) -> list[tuple[str, float, str]]:
        memory = document["environment"]["peak_memory_gib_by_rank"]
        return [
            ("E0", memory[0], "Encoder"),
            *[(f"D{rank}", memory[rank], "DiT") for rank in range(1, len(memory) - 1)],
            (f"V{len(memory) - 1}", memory[-1], "Decoder"),
        ]

    return [
        (
            "CP1",
            [
                ("E0", base["encoder"], "Encoder"),
                ("D1", base["dit"], "DiT"),
                ("V2", base["decoder"], "Decoder"),
            ],
        ),
        ("CP4 Ulysses", cp_rows(cp4)),
        ("CP6 ring", cp_rows(cp6)),
    ]


def _svg(
    baseline: dict[str, Any],
    cp4: dict[str, Any],
    cp6: dict[str, Any],
) -> str:
    width, height = 1200, 780
    rows = _walltime(baseline, cp4, cp6)
    max_wall = 2500.0
    chart_x, chart_w = 170.0, 940.0
    wall_y = [135.0, 207.0, 279.0]
    bar_h = 43.0
    pieces = [
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
            f'height="{height}" viewBox="0 0 {width} {height}" role="img" '
            'aria-labelledby="title description">'
        ),
        (
            '<title id="title">LingBot single-session context-parallel wall time '
            "and GPU memory</title>"
        ),
        (
            '<desc id="description">Median wall-time components and per-rank peak '
            "allocated memory for CP1, CP4 Ulysses, and CP6 ring.</desc>"
        ),
        "<style>",
        "text{font-family:Inter,Arial,sans-serif;fill:#202124}",
        ".title{font-size:26px;font-weight:600}.subtitle{font-size:15px;fill:#5f6368}",
        ".axis{font-size:12px;fill:#5f6368}.label{font-size:15px;font-weight:600}",
        ".value{font-size:12px;font-weight:600}.grid{stroke:#dadce0;stroke-width:1}",
        "</style>",
        '<rect width="1200" height="780" fill="#ffffff"/>',
        (
            '<text x="40" y="42" class="title">LingBot minimum single-session '
            "latency</text>"
        ),
        (
            '<text x="40" y="68" class="subtitle">H100 80 GB · BF16 · 832×464 · '
            "six warmup and five measured blocks</text>"
        ),
        '<text x="40" y="105" class="label">Median steady-state wall time</text>',
    ]
    for tick in range(0, 2501, 500):
        x = chart_x + chart_w * tick / max_wall
        pieces.extend(
            [
                f'<line x1="{x:.1f}" y1="121" x2="{x:.1f}" y2="331" class="grid"/>',
                (
                    f'<text x="{x:.1f}" y="350" class="axis" '
                    f'text-anchor="middle">{tick}</text>'
                ),
            ]
        )
    pieces.append(
        '<text x="1170" y="350" class="axis" text-anchor="end">milliseconds</text>'
    )
    for row, y in zip(rows, wall_y):
        pieces.append(
            f'<text x="150" y="{y + 27:.1f}" class="label" text-anchor="end">'
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
                    f'<text x="{x + segment_w / 2:.1f}" y="{y + 26:.1f}" '
                    f'class="value" text-anchor="middle">{value:.0f}</text>'
                )
            x += segment_w
        pieces.append(
            f'<text x="{x + 8:.1f}" y="{y + 27:.1f}" class="value">'
            f"{row['total']:.0f} ms</text>"
        )

    for index, label in enumerate(_COLORS):
        x = 170 + index * 156
        pieces.extend(
            [
                f'<rect x="{x}" y="370" width="12" height="12" fill="{_COLORS[label]}"/>',
                f'<text x="{x + 18}" y="380" class="axis">{html.escape(label)}</text>',
            ]
        )

    pieces.extend(
        [
            '<text x="40" y="423" class="label">Peak allocated GPU memory by rank</text>',
            (
                '<text x="40" y="446" class="subtitle">E = encoder, D = DiT, '
                "V = decoder; CP4 leaves two GPUs available for other work</text>"
            ),
        ]
    )
    mem_base_y, mem_top_y = 701.0, 478.0
    mem_h = mem_base_y - mem_top_y
    for tick in (0, 20, 40, 60, 80):
        y = mem_base_y - mem_h * tick / 80.0
        pieces.extend(
            [
                f'<line x1="90" y1="{y:.1f}" x2="1160" y2="{y:.1f}" class="grid"/>',
                f'<text x="78" y="{y + 5:.1f}" class="axis" text-anchor="end">{tick}</text>',
            ]
        )
    pieces.append(
        '<text x="43" y="590" class="axis" text-anchor="middle" '
        'transform="rotate(-90 43 590)">GiB</text>'
    )
    layouts = [(125.0, 58.0), (395.0, 47.0), (765.0, 43.0)]
    for (group_label, memory), (start_x, gap) in zip(
        _memory_rows(baseline, cp4, cp6),
        layouts,
    ):
        bar_width = 31.0
        center = start_x + ((len(memory) - 1) * gap + bar_width) / 2
        pieces.append(
            f'<text x="{center:.1f}" y="748" class="label" '
            f'text-anchor="middle">{html.escape(group_label)}</text>'
        )
        for index, (rank, value, stage) in enumerate(memory):
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
                        f'<text x="{x + bar_width / 2:.1f}" y="{y - 6:.1f}" '
                        f'class="value" text-anchor="middle">{value:.1f}</text>'
                    ),
                    (
                        f'<text x="{x + bar_width / 2:.1f}" y="721" '
                        f'class="axis" text-anchor="middle">{rank}</text>'
                    ),
                ]
            )
    pieces.append("</svg>")
    return "\n".join(pieces) + "\n"


def main() -> None:
    """Read benchmark documents and write the single-session SVG."""
    args = _args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        _svg(
            _read(args.baseline),
            _read(args.cp4),
            _read(args.cp6),
        )
    )


if __name__ == "__main__":
    main()
