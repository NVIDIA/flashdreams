# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Lossless frame dump for the dual-decode latent-codec comparison.

Writes PNGs straight from the decoder output, BEFORE the H.264 encoder on the WebRTC
path. That ordering is the whole point: the question under test is what a latent codec
does to the pixels, and routing the comparison through a lossy video codec would layer
its artifacts on top of the effect being measured -- at the low error rates involved
(int8-8s is ~0.05% in the latent domain) H.264 would plausibly dominate.

PNG is lossless, so the dumped frames ARE the decoder output. ``ffmpeg`` can assemble
them into an FFV1 or CRF-0 file afterwards for convenient viewing, still lossless.

Enable with ``FD_DUMP_FRAMES=/path/to/dir``. Optionally cap the volume with
``FD_DUMP_MAX_FRAMES`` (default 240 per branch) -- at 704x1280 each PNG is roughly
1-2 MB, so an unbounded run fills a disk quickly.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from loguru import logger
from torch import Tensor

_dump_dir: Path | None = None
_dump_resolved = False
_counts: dict[str, int] = {}
_max_frames = 240


def dump_dir() -> Path | None:
    """Resolve ``FD_DUMP_FRAMES`` once; ``None`` disables dumping."""
    global _dump_dir, _dump_resolved, _max_frames
    if not _dump_resolved:
        raw = os.environ.get("FD_DUMP_FRAMES")
        if raw:
            _dump_dir = Path(raw)
            _dump_dir.mkdir(parents=True, exist_ok=True)
            _max_frames = int(os.environ.get("FD_DUMP_MAX_FRAMES", "240"))
            logger.info(
                "Frame dump ACTIVE: {} (max {} frames/branch, lossless PNG, pre-H.264)",
                _dump_dir,
                _max_frames,
            )
        _dump_resolved = True
    return _dump_dir


def _to_uint8(video: Tensor) -> np.ndarray:
    """Collapse leading dims and map [-1, 1] -> uint8 [0, 255] as [N, H, W, 3].

    Rounds rather than truncates, and clamps: the decoder's output is nominally in
    [-1, 1] but nothing guarantees it, and a wrapped value would read as a bright
    speckle that could be mistaken for a codec artifact.
    """
    v = video.detach().float()
    v = v.reshape(-1, *v.shape[-3:])          # [N, 3, H, W]
    v = ((v.clamp(-1.0, 1.0) + 1.0) * 127.5).round().to(torch.uint8)
    return v.permute(0, 2, 3, 1).cpu().numpy()  # [N, H, W, 3]


def dump_branch(video: Any, branch: str, autoregressive_index: int) -> None:
    """Write one chunk's frames for ``branch`` as PNGs, if dumping is enabled."""
    d = dump_dir()
    if d is None or not isinstance(video, Tensor):
        return
    start = _counts.get(branch, 0)
    if start >= _max_frames:
        return

    from PIL import Image

    frames = _to_uint8(video)
    out = d / branch
    out.mkdir(parents=True, exist_ok=True)
    for i, frame in enumerate(frames):
        idx = start + i
        if idx >= _max_frames:
            break
        # Zero-padded so lexical order matches temporal order, which is what
        # ffmpeg's image sequence reader assumes.
        Image.fromarray(frame).save(out / f"{idx:06d}.png", compress_level=1)
    _counts[branch] = min(start + len(frames), _max_frames)

    if autoregressive_index == 0:
        logger.info(
            "[DUMP] {} chunk ar={} -> {} frames {}x{} at {}",
            branch,
            autoregressive_index,
            len(frames),
            frames.shape[2],
            frames.shape[1],
            out,
        )
