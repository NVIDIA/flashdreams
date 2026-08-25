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


def _load_stats(side_dir: Path) -> dict[str, Any] | list[dict[str, Any]] | None:
    """Read ``stats_<runner>.json`` (omnidreams-style per-AR-step list or wall-clock dict)."""
    path = side_dir / f"stats_{_RUNNER_NAME}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _load_video(side_dir: Path) -> np.ndarray:
    """Decode the runner's mp4 to a ``[T, H, W, 3]`` uint8 array."""
    path = side_dir / f"{_RUNNER_NAME}.mp4"
    if not path.exists():
        raise FileNotFoundError(f"missing mp4 {path}")
    return iio.imread(path)


def _median(values: list[float]) -> float | None:
    """Return the median of ``values`` (``None`` when empty)."""
    if not values:
        return None
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    return (
        sorted_vals[n // 2]
        if n % 2
        else 0.5 * (sorted_vals[n // 2 - 1] + sorted_vals[n // 2])
    )


def _stage_median_post_warmup(
    stats: list[dict[str, Any]] | None,
    stage_key: str,
    warmup_chunks: int,
) -> tuple[float | None, int]:
    """Median ``{stage_key}_ms`` over the AR steps past ``warmup_chunks``.

    Returns:
        ``(median_ms, n_kept)``. ``median_ms`` is ``None`` when the
        side reported no per-AR-step stats (e.g. vendor's wall-clock
        only dict) or the post-warmup slice is empty.
    """
    if not isinstance(stats, list):
        return None, 0
    kept = [entry.get(stage_key) for entry in stats[warmup_chunks:]]
    kept_floats = [float(v) for v in kept if isinstance(v, (int, float))]
    return _median(kept_floats), len(kept_floats)


def _format_per_chunk_stage(stats: list[dict[str, Any]] | None, stage_key: str) -> str:
    """Render per-AR-step ``{stage_key}_ms`` as ``c0=12.3ms, c1=18.4ms``."""
    if not isinstance(stats, list):
        return "n/a"
    parts: list[str] = []
    for entry in stats:
        ar_idx = entry.get("autoregressive_index")
        v = entry.get(stage_key)
        if not isinstance(v, (int, float)) or ar_idx is None:
            continue
        parts.append(f"c{ar_idx}={float(v):.1f}ms")
    return ", ".join(parts) if parts else "n/a"


def _wall_clock(stats: Any) -> str:
    """Render the side's overall wall-clock (``elapsed_s`` for vendor, ``total_ms_wo_finalize`` sum for native)."""
    if isinstance(stats, list):
        total = sum(
            float(entry.get("total_ms_wo_finalize", 0.0))
            for entry in stats
            if isinstance(entry.get("total_ms_wo_finalize"), (int, float))
        )
        if total:
            return f"{total / 1000:.2f} s (sum of per-AR total_ms_wo_finalize)"
        return "n/a"
    if isinstance(stats, dict):
        value = stats.get("elapsed_s")
        if isinstance(value, (int, float)):
            return f"{float(value):.2f} s (wall clock)"
    return "n/a"


def _peak_gpu_mem(stats: Any) -> str:
    """Render peak GPU memory (last AR step's ``mem_peak_gib`` for native, ``peak_gpu_mem_gib`` for vendor)."""
    if isinstance(stats, list) and stats:
        for entry in reversed(stats):
            v = entry.get("mem_peak_gib")
            if isinstance(v, (int, float)):
                return f"{float(v):.2f}"
    if isinstance(stats, dict):
        v = stats.get("peak_gpu_mem_gib")
        if isinstance(v, (int, float)):
            return f"{float(v):.2f}"
    return "n/a"


def _perf_table(
    native: list[dict[str, Any]] | dict[str, Any] | None,
    vendor: list[dict[str, Any]] | dict[str, Any] | None,
    warmup_chunks: int,
) -> str:
    """Build the perf markdown table comparing the two backends."""
    rows = [
        "| metric | native | vendor |",
        "| --- | --- | --- |",
        f"| wall clock | {_wall_clock(native)} | {_wall_clock(vendor)} |",
        f"| peak GPU mem (GiB) | {_peak_gpu_mem(native)} | {_peak_gpu_mem(vendor)} |",
    ]

    for stage in ("encode", "diffuse", "decode"):
        nval, n_n = _stage_median_post_warmup(native, f"{stage}_ms", warmup_chunks)
        vval, n_v = _stage_median_post_warmup(vendor, f"{stage}_ms", warmup_chunks)
        nc = f"{nval:.1f} ms (n={n_n})" if nval is not None else "n/a"
        vc = f"{vval:.1f} ms (n={n_v})" if vval is not None else "n/a"
        rows.append(f"| {stage} median (post-warmup) | {nc} | {vc} |")

    rows.append(
        f"| per-AR diffuse | {_format_per_chunk_stage(native, 'diffuse_ms')}"
        f" | {_format_per_chunk_stage(vendor, 'diffuse_ms')} |"
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
    native_stats: Any,
    vendor_stats: Any,
    native_mp4: np.ndarray,
    vendor_mp4: np.ndarray,
    image_path: Path,
    pose: str,
    num_chunk: int,
    seed: int,
    warmup_chunks: int,
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
        f"- warmup chunks discarded: `{warmup_chunks}`",
        (f"- native frames: `{native_mp4.shape}`, vendor frames: `{vendor_mp4.shape}`"),
        "",
        "## Perf",
        "",
        _perf_table(native_stats, vendor_stats, warmup_chunks),
        "",
        (
            "Both sides report per-AR-step encode / diffuse / decode / "
            "finalize / total_ms / mem_*_gib via flashdreams's "
            "``EventProfiler``. Native goes through the built-in "
            "``StreamInferencePipeline`` profiler "
            "(``enable_sync_and_profile=True``); vendor uses the "
            "monkey-patched ``WanPipeline`` wrapper in "
            "``vendor_profile_patch.py`` (no ``encode`` stage on vendor "
            "since upstream's chunk loop excludes the one-time first-frame "
            "encode that runs at ``WanRunner.__init__``)."
        ),
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
    parser.add_argument(
        "--warmup-chunks",
        type=int,
        default=0,
        help="Drop the first N AR steps from the post-warmup stage medians.",
    )
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
        warmup_chunks=args.warmup_chunks,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report)
    print(report)


if __name__ == "__main__":
    main()
