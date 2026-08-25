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

"""Summarize SANA-WM streaming chunk-boundary continuity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from diff_parity import _load_frames


def summarize_frames(frames: np.ndarray, *, chunk_size: int) -> dict[str, Any]:
    """Return consecutive-frame deltas with chunk-boundary highlights."""
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}.")
    if frames.shape[0] < 2:
        raise ValueError(f"At least two frames are required, got {frames.shape[0]}.")

    diffs = np.abs(frames[1:].astype(np.int16) - frames[:-1].astype(np.int16))
    per_transition = diffs.reshape(diffs.shape[0], -1).mean(axis=1)
    boundary_indices = list(range(chunk_size - 1, len(per_transition), chunk_size))
    boundary_set = set(boundary_indices)
    nonboundary = np.asarray(
        [value for idx, value in enumerate(per_transition) if idx not in boundary_set],
        dtype=np.float64,
    )
    boundary = np.asarray(
        [per_transition[idx] for idx in boundary_indices], dtype=np.float64
    )

    nonboundary_p95 = (
        float(np.percentile(nonboundary, 95)) if nonboundary.size else None
    )
    boundary_mean = float(boundary.mean()) if boundary.size else None
    ratio = None
    if (
        nonboundary_p95 is not None
        and nonboundary_p95 > 0.0
        and boundary_mean is not None
    ):
        ratio = boundary_mean / nonboundary_p95

    return {
        "shape": list(frames.shape),
        "chunk_size": int(chunk_size),
        "mean_abs_delta": float(per_transition.mean()),
        "p95_abs_delta": float(np.percentile(per_transition, 95)),
        "nonboundary_mean_abs_delta": float(nonboundary.mean())
        if nonboundary.size
        else None,
        "nonboundary_p95_abs_delta": nonboundary_p95,
        "boundary_mean_abs_delta": boundary_mean,
        "boundary_max_abs_delta": float(boundary.max()) if boundary.size else None,
        "boundary_to_nonboundary_p95_ratio": ratio,
        "boundary_transitions": [
            {
                "from_frame": int(idx),
                "to_frame": int(idx + 1),
                "mean_abs_delta": float(per_transition[idx]),
            }
            for idx in boundary_indices
        ],
    }


def _comparison(
    upstream: dict[str, Any] | None,
    flashdreams: dict[str, Any] | None,
) -> dict[str, Any]:
    if upstream is None or flashdreams is None:
        return {}
    upstream_boundary = upstream.get("boundary_mean_abs_delta")
    flashdreams_boundary = flashdreams.get("boundary_mean_abs_delta")
    upstream_ratio = upstream.get("boundary_to_nonboundary_p95_ratio")
    flashdreams_ratio = flashdreams.get("boundary_to_nonboundary_p95_ratio")
    return {
        "shape_match": upstream.get("shape") == flashdreams.get("shape"),
        "flashdreams_boundary_mean_minus_upstream": (
            float(flashdreams_boundary - upstream_boundary)
            if isinstance(upstream_boundary, (int, float))
            and isinstance(flashdreams_boundary, (int, float))
            else None
        ),
        "flashdreams_boundary_ratio_minus_upstream": (
            float(flashdreams_ratio - upstream_ratio)
            if isinstance(upstream_ratio, (int, float))
            and isinstance(flashdreams_ratio, (int, float))
            else None
        ),
    }


def _format_optional_float(value: object) -> str:
    return f"{value:.4f}" if isinstance(value, (int, float)) else "n/a"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, default=None)
    parser.add_argument("--flashdreams", type=Path, default=None)
    parser.add_argument("--chunk-size", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    if args.upstream is None and args.flashdreams is None:
        raise ValueError("At least one of --upstream or --flashdreams is required.")

    upstream = (
        summarize_frames(_load_frames(args.upstream), chunk_size=args.chunk_size)
        if args.upstream is not None
        else None
    )
    flashdreams = (
        summarize_frames(_load_frames(args.flashdreams), chunk_size=args.chunk_size)
        if args.flashdreams is not None
        else None
    )
    payload = {
        "chunk_size": int(args.chunk_size),
        "upstream": upstream,
        "flashdreams": flashdreams,
        "comparison": _comparison(upstream, flashdreams),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    for label, summary in (("upstream", upstream), ("flashdreams", flashdreams)):
        if summary is None:
            continue
        print(
            f"{label}: boundary mean "
            f"{_format_optional_float(summary['boundary_mean_abs_delta'])}, "
            f"nonboundary p95 "
            f"{_format_optional_float(summary['nonboundary_p95_abs_delta'])}, "
            f"ratio "
            f"{_format_optional_float(summary['boundary_to_nonboundary_p95_ratio'])}"
        )


if __name__ == "__main__":
    main()
