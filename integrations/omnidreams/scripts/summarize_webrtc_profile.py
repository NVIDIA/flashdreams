#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import ast
import re
import statistics as st
from pathlib import Path


CHUNK_RE = re.compile(
    r"Chunk done chunk=(?P<chunk>\d+).*?"
    r"gen_ms=(?P<gen_ms>[0-9.]+).*?"
    r"enqueue_ms=(?P<enqueue_ms>[0-9.]+).*?"
    r"play_ms=(?P<play_ms>[0-9.]+).*?"
    r"queue_depth=(?P<queue_depth>\d+).*?"
    r"lag_ms=(?P<lag_ms>[0-9.]+).*?"
    r"profile=(?P<profile>\{.*\})"
)

DEFAULT_KEYS = [
    "gen_ms",
    "wrapper_render_condition_ms",
    "renderer_ctx_render_ms",
    "ctx_render_plugin_cuda_ms_sum",
    "ctx_render_plugin_cuda_ms_avg",
    "pipeline_total_ms",
    "pipeline_total_ms_wo_finalize",
    "enqueue_ms",
    "total_ms",
    "play_ms",
    "lag_ms",
]


def percentile(values: list[float], q: float) -> float:
    if not values:
        raise ValueError("percentile requires at least one value")
    ordered = sorted(values)
    index = int(q * (len(ordered) - 1))
    return ordered[index]


def load_rows(path: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    session = -1
    prev_chunk: int | None = None
    for line in path.read_text(errors="replace").splitlines():
        match = CHUNK_RE.search(line)
        if match is None:
            continue
        chunk = int(match.group("chunk"))
        if prev_chunk is None or chunk <= prev_chunk:
            session += 1
        prev_chunk = chunk
        profile = ast.literal_eval(match.group("profile"))
        row = {
            "session": float(session),
            "chunk": float(chunk),
            "gen_ms": float(match.group("gen_ms")),
            "enqueue_ms": float(match.group("enqueue_ms")),
            "play_ms": float(match.group("play_ms")),
            "queue_depth": float(match.group("queue_depth")),
            "lag_ms": float(match.group("lag_ms")),
        }
        row.update({key: float(value) for key, value in profile.items()})
        rows.append(row)
    return rows


def print_stats(rows: list[dict[str, float]], keys: list[str]) -> None:
    for key in keys:
        values = [row[key] for row in rows if key in row]
        if not values:
            continue
        print(
            f"{key:34s} avg={st.mean(values):7.1f} "
            f"p50={st.median(values):7.1f} p90={percentile(values, 0.9):7.1f} "
            f"min={min(values):7.1f} max={max(values):7.1f}"
        )


def print_top_profile_keys(rows: list[dict[str, float]], limit: int) -> None:
    skip = {"session", "chunk", "queue_depth", "renderer_height", "renderer_width"}
    means = []
    keys = sorted(set().union(*(row.keys() for row in rows)))
    for key in keys:
        if key in skip or key.endswith("_count"):
            continue
        if not key.endswith("_ms") and not key.endswith("_ms_sum"):
            continue
        values = [row[key] for row in rows if key in row]
        if values:
            means.append((st.mean(values), key))
    for mean, key in sorted(means, reverse=True)[:limit]:
        print(f"{key:34s} avg={mean:7.1f}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize OmniDreams WebRTC chunk profile logs."
    )
    parser.add_argument("log", type=Path)
    parser.add_argument("--min-chunk", type=int, default=4)
    parser.add_argument(
        "--session",
        default="latest",
        help="'latest', 'all', or a numeric session index inferred from chunk resets.",
    )
    parser.add_argument("--top", type=int, default=12)
    args = parser.parse_args()

    rows = load_rows(args.log)
    print(f"parsed_chunks={len(rows)}")
    if not rows:
        return

    sessions = sorted({int(row["session"]) for row in rows})
    print(f"sessions={sessions}")
    if args.session == "latest":
        wanted_sessions = {sessions[-1]}
    elif args.session == "all":
        wanted_sessions = set(sessions)
    else:
        wanted_sessions = {int(args.session)}

    selected = [
        row
        for row in rows
        if int(row["session"]) in wanted_sessions and row["chunk"] >= args.min_chunk
    ]
    print(
        f"selected_chunks={len(selected)} "
        f"session={args.session} min_chunk={args.min_chunk}"
    )
    if not selected:
        return

    print("\nprimary stats")
    print_stats(selected, DEFAULT_KEYS)
    print("\ntop profile timings")
    print_top_profile_keys(selected, args.top)


if __name__ == "__main__":
    main()
