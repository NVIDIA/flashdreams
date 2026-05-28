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

"""Aggregate per-image native-bench stats into a single PR-ready markdown table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

_RUNNER_NAME = "hy-worldplay-wan-i2v-5b"
"""Filename stem the native runner uses for its mp4 / stats artifacts."""


def _format_per_chunk_ms(stats: dict[str, Any]) -> str:
    """Render per-chunk CUDA-event timing as ``c0=12.3ms, c1=18.4ms``."""
    per_chunk = stats.get("per_chunk_ms") or {}
    if not per_chunk:
        return "n/a"
    parts: list[str] = []
    for k, v in per_chunk.items():
        # ``chunk_0`` -> ``c0`` so cells stay narrow.
        label = k.replace("chunk_", "c") if k.startswith("chunk_") else k
        parts.append(f"{label}={float(v):.1f}")
    return ", ".join(parts)


def _read_stats(image_dir: Path) -> dict[str, Any] | None:
    """Return the runner's stats JSON for one image, or ``None`` if missing."""
    path = image_dir / f"stats_{_RUNNER_NAME}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _build_table(rows: list[tuple[str, dict[str, Any] | None]]) -> str:
    """Render the perf rows as a markdown table."""
    header = [
        "| image | mp4 | elapsed (s) | peak GPU mem (GiB) | per-chunk (ms) |",
        "| --- | --- | --- | --- | --- |",
    ]

    def cell(stats: dict[str, Any] | None, key: str, fmt: str) -> str:
        if stats is None:
            return "n/a"
        value = stats.get(key)
        if value is None:
            return "n/a"
        return fmt.format(value)

    out = list(header)
    for stem, stats in rows:
        mp4_path = f"`{stem}/{_RUNNER_NAME}.mp4`"
        elapsed = cell(stats, "elapsed_s", "{:.2f}")
        peak = cell(stats, "peak_gpu_mem_gib", "{:.2f}")
        per_chunk = _format_per_chunk_ms(stats) if stats is not None else "n/a"
        out.append(f"| `{stem}` | {mp4_path} | {elapsed} | {peak} | {per_chunk} |")
    return "\n".join(out)


def _render_report(
    *,
    rows: list[tuple[str, dict[str, Any] | None]],
    num_chunk: int,
    pose: str,
    seed: int,
) -> str:
    """Wrap the perf table with a header summarising the rollout knobs."""
    missing = [stem for stem, stats in rows if stats is None]
    lines = [
        "# HY-WorldPlay WAN-5B I2V: native batch bench",
        "",
        "## Settings",
        "",
        f"- `num_chunk={num_chunk}` (pose `{pose}`), `seed={seed}`",
        f"- runner: `{_RUNNER_NAME}` (native plugin path)",
        "",
        "## Per-image perf",
        "",
        _build_table(rows),
        "",
    ]
    if missing:
        lines.extend(
            [
                "## Missing",
                "",
                "These images had no `stats_*.json` written -- the run "
                "probably failed before reaching the persistence step. "
                "Re-run them individually with verbose logging:",
                "",
            ]
        )
        for stem in missing:
            lines.append(f"- `{stem}`")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    """Parse CLI args, walk the output dir, and write the markdown report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--num-chunk", type=int, required=True)
    parser.add_argument("--pose", type=str, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()

    rows: list[tuple[str, dict[str, Any] | None]] = []
    for child in sorted(args.output_dir.iterdir()):
        if not child.is_dir():
            continue
        rows.append((child.name, _read_stats(child)))

    if not rows:
        raise SystemExit(
            f"no per-image subdirectories under {args.output_dir}; "
            "did bench_batch.sh actually run?"
        )

    report = _render_report(
        rows=rows,
        num_chunk=args.num_chunk,
        pose=args.pose,
        seed=args.seed,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report)
    print(report)


if __name__ == "__main__":
    main()
