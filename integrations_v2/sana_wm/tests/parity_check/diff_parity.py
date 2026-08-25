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

"""Compute SANA-WM frame parity as mean absolute uint8 delta."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def _load_frames(path: Path) -> np.ndarray:
    loaded = np.load(path)
    if isinstance(loaded, np.lib.npyio.NpzFile):
        try:
            if "frames" not in loaded:
                raise ValueError(f"{path} must contain a 'frames' array.")
            frames = loaded["frames"]
        finally:
            loaded.close()
    else:
        frames = loaded
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"{path} must contain [T,H,W,3] frames; got {frames.shape}.")
    if frames.dtype != np.uint8:
        frames = np.clip(frames, 0, 255).astype(np.uint8)
    return frames


def _summary(upstream: np.ndarray, flashdreams: np.ndarray) -> dict[str, Any]:
    if upstream.shape != flashdreams.shape:
        raise ValueError(
            "shape mismatch: "
            f"upstream={tuple(upstream.shape)} flashdreams={tuple(flashdreams.shape)}"
        )
    diff = np.abs(upstream.astype(np.int16) - flashdreams.astype(np.int16))
    per_frame = diff.reshape(diff.shape[0], -1).mean(axis=1)
    return {
        "shape": list(upstream.shape),
        "mean_abs_delta": float(diff.mean()),
        "mean_abs_delta_over_255": float(diff.mean() / 255.0),
        "max_abs_delta": int(diff.max()),
        "per_frame_mean_abs_delta": [float(v) for v in per_frame],
        "per_frame_max_abs_delta": [
            int(v) for v in diff.reshape(diff.shape[0], -1).max(axis=1)
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--flashdreams", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    summary = _summary(_load_frames(args.upstream), _load_frames(args.flashdreams))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(
        "mean |Delta|: "
        f"{summary['mean_abs_delta']:.4f} / 255 "
        f"({summary['mean_abs_delta_over_255']:.6f})"
    )
    print(f"max  |Delta|: {summary['max_abs_delta']} / 255")
    for idx, value in enumerate(summary["per_frame_mean_abs_delta"]):
        print(f"frame {idx:04d}: mean |Delta| {value:.4f} / 255")


if __name__ == "__main__":
    main()
