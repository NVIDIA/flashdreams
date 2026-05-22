# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Video frame decode: prefer ``decord`` when installed, else OpenCV.

PyPI ``decord`` only ships wheels for Linux x86_64 and Windows amd64. Linux
aarch64 / macOS / other platforms resolve ``flashdreams-eval`` without decord
and use ``opencv-python-headless`` instead (see ``internal/flashdreams_eval``).
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np


def get_video_frame_count(path: str | Path) -> int:
    """Return the reported number of frames (best-effort for OpenCV)."""
    p = str(path)
    try:
        import decord

        return len(decord.VideoReader(p))
    except ImportError:
        pass

    import cv2

    cap = cv2.VideoCapture(p)
    if not cap.isOpened():
        raise RuntimeError(f"could not open video: {p}")
    try:
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()
    return max(n, 0)


def get_video_frame_batch(path: str | Path, indices: List[int]) -> np.ndarray:
    """Return RGB uint8 frames ``(N, H, W, 3)`` for the given zero-based frame indices."""
    p = str(path)
    if not indices:
        return np.zeros((0, 1, 1, 3), dtype=np.uint8)

    try:
        import decord
    except ImportError:
        decord = None  # type: ignore[assignment]

    if decord is not None:
        vr = decord.VideoReader(p, num_threads=4)
        batch = vr.get_batch(indices)
        if hasattr(batch, "asnumpy"):
            frames = batch.asnumpy()
        elif hasattr(batch, "numpy"):
            frames = batch.numpy()
        else:
            frames = np.array(batch)
        vr.seek(0)
        del vr
        return frames

    import cv2

    cap = cv2.VideoCapture(p)
    if not cap.isOpened():
        raise RuntimeError(
            f"could not open video (install decord on a supported platform, or ensure "
            f"opencv can read this file): {p}"
        )
    out: list[np.ndarray] = []
    try:
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, float(idx))
            ok, frame = cap.read()
            if not ok or frame is None:
                raise RuntimeError(
                    f"failed reading frame {idx} from {p}; the container/codec may not "
                    f"support frame-accurate seeks — try re-encoding to H.264 in an MP4 container"
                )
            out.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        cap.release()
    return np.stack(out, axis=0)
