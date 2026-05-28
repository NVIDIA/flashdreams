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

"""Combine the native + vendor MP4s and stats JSONs into a PR-ready markdown table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import imageio.v3 as iio
import numpy as np

_RUNNER_NAME = "hy-worldplay-wan-i2v-5b"
"""Filename stem both runners use when writing their mp4 / stats artifacts."""

_VISIBLE_THRESHOLD = 5.0
"""Per-frame mean ``|Delta|`` (uint8) above which a viewer can spot the
difference. Matches the threshold the README cites for the parity caveat."""


def _load_stats(side_dir: Path) -> dict[str, Any]:
    """Read ``stats_<runner>.json`` written by the runner on rank zero."""
    path = side_dir / f"stats_{_RUNNER_NAME}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"missing stats file {path}; did the run finish on rank 0?"
        )
    return json.loads(path.read_text())


def _load_video(side_dir: Path) -> np.ndarray:
    """Decode the runner's mp4 to a ``[T, H, W, 3]`` uint8 array."""
    path = side_dir / f"{_RUNNER_NAME}.mp4"
    if not path.exists():
        raise FileNotFoundError(f"missing mp4 {path}")
    return iio.imread(path)


def _format_per_chunk_ms(stats: dict[str, Any]) -> str:
    """Render the per-chunk CUDA-event timing as ``c0=12.3ms, c1=18.4ms``."""
    per_chunk = stats.get("per_chunk_ms") or {}
    if not per_chunk:
        return "n/a"
    parts: list[str] = []
    for k, v in per_chunk.items():
        # ``chunk_0`` -> ``c0`` so the cell stays narrow in the markdown
        # table.
        label = k.replace("chunk_", "c") if k.startswith("chunk_") else k
        parts.append(f"{label}={float(v):.1f}ms")
    return ", ".join(parts)


def _perf_table(native: dict[str, Any], vendor: dict[str, Any]) -> str:
    """Build the perf markdown table comparing the two backends."""
    rows = [
        "| metric | native | vendor |",
        "| --- | --- | --- |",
    ]

    def cell(stats: dict[str, Any], key: str, fmt: str = "{:.3f}") -> str:
        value = stats.get(key)
        if value is None:
            return "n/a"
        return fmt.format(value)

    rows.append(
        f"| elapsed (s) | {cell(native, 'elapsed_s', '{:.2f}')}"
        f" | {cell(vendor, 'elapsed_s', '{:.2f}')} |"
    )
    rows.append(
        f"| peak GPU mem (GiB) | {cell(native, 'peak_gpu_mem_gib', '{:.2f}')}"
        f" | {cell(vendor, 'peak_gpu_mem_gib', '{:.2f}')} |"
    )
    rows.append(
        f"| per-chunk timing | {_format_per_chunk_ms(native)}"
        f" | {_format_per_chunk_ms(vendor)} |"
    )
    return "\n".join(rows)


def _parity_block(native_mp4: np.ndarray, vendor_mp4: np.ndarray) -> str:
    """Compute the mean / max ``|Delta|`` and frame-count crossing the visible bar."""
    if native_mp4.shape != vendor_mp4.shape:
        return (
            f"`shape mismatch: native={native_mp4.shape}, "
            f"vendor={vendor_mp4.shape}` -- skipping numeric parity diff."
        )
    diff = np.abs(native_mp4.astype(np.int16) - vendor_mp4.astype(np.int16))
    per_frame = diff.reshape(diff.shape[0], -1).mean(axis=1)
    visible = int((per_frame > _VISIBLE_THRESHOLD).sum())
    return "\n".join(
        [
            f"- mean `|Delta|`: **{diff.mean():.3f}** / 255",
            f"- max  `|Delta|`: **{int(diff.max())}** / 255",
            (
                f"- frames with mean `|Delta|` > {_VISIBLE_THRESHOLD}: "
                f"**{visible}** / {native_mp4.shape[0]}"
            ),
        ]
    )


def _render_report(
    *,
    native_stats: dict[str, Any],
    vendor_stats: dict[str, Any],
    native_mp4: np.ndarray,
    vendor_mp4: np.ndarray,
    image_path: Path,
    pose: str,
    num_chunk: int,
    seed: int,
) -> str:
    """Stitch the input summary, perf table, and parity block into one markdown blob."""
    lines = [
        "# HY-WorldPlay WAN-5B I2V: native vs vendor bench",
        "",
        "## Inputs",
        "",
        f"- image: `{image_path}`",
        f"- pose: `{pose}` (`num_chunk={num_chunk}`)",
        f"- seed: `{seed}`",
        (
            f"- native frames: `{native_mp4.shape}`, "
            f"vendor frames: `{vendor_mp4.shape}`"
        ),
        "",
        "## Perf",
        "",
        _perf_table(native_stats, vendor_stats),
        "",
        "## Parity (native mp4 vs vendor mp4)",
        "",
        _parity_block(native_mp4, vendor_mp4),
        "",
        (
            "Reference: `<= 20 / 255` mean is the phase 2b.6 acceptance bar; "
            "`<= 5 / 255` per-frame is the visible-difference threshold."
        ),
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    """Parse CLI args, run the comparison, and write the markdown report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-dir", type=Path, required=True)
    parser.add_argument("--vendor-dir", type=Path, required=True)
    parser.add_argument("--image-path", type=Path, required=True)
    parser.add_argument("--pose", type=str, required=True)
    parser.add_argument("--num-chunk", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    report = _render_report(
        native_stats=_load_stats(args.native_dir),
        vendor_stats=_load_stats(args.vendor_dir),
        native_mp4=_load_video(args.native_dir),
        vendor_mp4=_load_video(args.vendor_dir),
        image_path=args.image_path,
        pose=args.pose,
        num_chunk=args.num_chunk,
        seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report)
    print(report)


if __name__ == "__main__":
    main()
