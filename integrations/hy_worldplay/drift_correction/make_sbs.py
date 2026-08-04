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

"""Labeled side-by-side MP4s for owner eyeball verdicts (ffmpeg hstack).

Pairs every non-``base`` config's videos against the matching ``base``
video in an eval/demo directory and writes
``sbs_base_LEFT_vs_{config}_RIGHT_{name}.mp4`` next to them — which side
is which is in the filename (owner convention; the box ffmpeg has no
``drawtext``, so no burned-in labels).

Run from the repo root::

    EVAL_OUT=.../outputs/demo_static \
        python integrations/hy_worldplay/drift_correction/make_sbs.py
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

EVAL_DIR = Path(
    os.environ.get(
        "EVAL_OUT", "integrations/hy_worldplay/drift_correction/outputs/eval"
    )
)


def hstack(left: Path, right: Path, out: Path) -> None:
    """Write a horizontal side-by-side of two same-size videos."""
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(left),
            "-i",
            str(right),
            "-filter_complex",
            "[0:v][1:v]hstack",
            "-c:v",
            "libx264",
            "-crf",
            "18",
            str(out),
        ],
        check=True,
    )


def main() -> None:
    base_dir = EVAL_DIR / "base"
    assert base_dir.is_dir(), f"no base config under {EVAL_DIR}"
    configs = [
        d.name for d in sorted(EVAL_DIR.iterdir()) if d.is_dir() and d.name != "base"
    ]
    for config in configs:
        for mp4 in sorted((EVAL_DIR / config).glob("*.mp4")):
            base_mp4 = base_dir / mp4.name
            if not base_mp4.exists():
                print(f"{config}/{mp4.name}: no base counterpart, skipping")
                continue
            out = EVAL_DIR / f"sbs_base_LEFT_vs_{config}_RIGHT_{mp4.stem}.mp4"
            if out.exists():
                continue
            hstack(base_mp4, mp4, out)
            print(f"wrote {out.name}", flush=True)


if __name__ == "__main__":
    main()
