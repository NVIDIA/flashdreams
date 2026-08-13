# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Ordered local-media references for MiniMax H3 ref2va requests."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import torch

ReferenceKind = Literal["image", "video", "audio"]


@dataclass(frozen=True)
class MiniMaxH3ReferenceSpec:
    """One validated ``kind:path`` reference, preserving request order."""

    kind: ReferenceKind
    path: Path

    def manifest(self) -> dict[str, str | int]:
        """Return the source identity used by restart-safe checkpoints."""
        stat = self.path.stat()
        return {
            "kind": self.kind,
            "path": str(self.path),
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        }


def parse_reference_specs(
    entries: Sequence[str],
) -> tuple[MiniMaxH3ReferenceSpec, ...]:
    """Parse and enforce H3's documented ordered-reference limits."""
    specs: list[MiniMaxH3ReferenceSpec] = []
    for entry in entries:
        kind, separator, path_value = entry.partition(":")
        if not separator or kind not in {"image", "video", "audio"}:
            raise ValueError(
                f"invalid reference {entry!r}; expected image:path, video:path, or audio:path"
            )
        path = Path(path_value).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"reference file not found: {path}")
        specs.append(MiniMaxH3ReferenceSpec(kind=cast(ReferenceKind, kind), path=path))

    if not specs:
        raise ValueError("ref2va requires at least one --reference")
    limits = {"image": 9, "video": 3, "audio": 3}
    for kind, limit in limits.items():
        count = sum(spec.kind == kind for spec in specs)
        if count > limit:
            raise ValueError(f"MiniMax H3 accepts at most {limit} {kind} references")
    if len(specs) > 12:
        raise ValueError("MiniMax H3 accepts at most 12 references in total")
    if all(spec.kind == "audio" for spec in specs):
        raise ValueError(
            "an audio reference must be paired with an image or video reference"
        )
    return tuple(specs)


def load_references(specs: tuple[MiniMaxH3ReferenceSpec, ...]) -> list[Any]:
    """Decode references through Diffusers' official H3 media containers."""
    from diffusers.modular_pipelines.minimax_h3 import (
        MiniMaxH3AudioReference,
        MiniMaxH3ImageReference,
        MiniMaxH3VideoReference,
    )

    classes = {
        "image": MiniMaxH3ImageReference,
        "video": MiniMaxH3VideoReference,
        "audio": MiniMaxH3AudioReference,
    }
    references: list[Any] = [classes[spec.kind].from_file(spec.path) for spec in specs]
    for reference in references:
        if not reference.has_audio or reference.sample_rate in {None, 32000}:
            continue
        import av
        import numpy as np

        waveform = reference.audio.detach().cpu().to(torch.float32).numpy()
        layout = "mono" if waveform.shape[0] == 1 else "stereo"
        frame = av.AudioFrame.from_ndarray(waveform, format="fltp", layout=layout)
        frame.sample_rate = reference.sample_rate
        resampler = av.AudioResampler(format="fltp", layout=layout, rate=32000)
        resampled = [*resampler.resample(frame), *resampler.resample(None)]
        reference.audio = torch.from_numpy(
            np.concatenate([chunk.to_ndarray() for chunk in resampled], axis=-1)
        ).to(torch.float32)
        reference.sample_rate = 32000
    return references


__all__ = [
    "MiniMaxH3ReferenceSpec",
    "ReferenceKind",
    "load_references",
    "parse_reference_specs",
]
