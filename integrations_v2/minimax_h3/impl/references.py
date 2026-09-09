# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Ordered local media for MiniMax H3 conditioning."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

import numpy as np
import torch
from PIL import Image, ImageOps

ReferenceKind = Literal["image", "video", "audio"]


@dataclass(frozen=True)
class MiniMaxH3ReferenceSpec:
    """One validated local input in request order."""

    kind: ReferenceKind
    """Media modality."""
    path: Path
    """Existing local media file."""


@dataclass
class MiniMaxH3Reference:
    """Decoded conditioning media before or after normalization."""

    kind: ReferenceKind
    """Media modality."""
    image: Image.Image | None = None
    """RGB reference image."""
    frames: np.ndarray | None = None
    """RGB video frames in THWC layout."""
    fps: float | None = None
    """Frame rate of the decoded video."""
    audio: torch.Tensor | None = None
    """Mono or stereo floating-point waveform, channels first."""
    sample_rate: int | None = None
    """Waveform samples per second."""

    @property
    def has_audio(self) -> bool:
        """Return whether this reference contains a soundtrack."""
        return self.audio is not None


def parse_reference_specs(entries: Sequence[str]) -> tuple[MiniMaxH3ReferenceSpec, ...]:
    """Validate ordered local references and the released model's limits."""
    result = []
    for entry in entries:
        kind, separator, value = entry.partition(":")
        if not separator or kind not in {"image", "audio", "video"}:
            raise ValueError(
                f"Invalid reference {entry!r}; expected image:path, video:path, or audio:path"
            )
        path = Path(value).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        result.append(MiniMaxH3ReferenceSpec(kind=cast(ReferenceKind, kind), path=path))
    validate_references(result)
    return tuple(result)


def validate_references(references: Sequence) -> None:
    """Reject missing, excessive, or audio-only reference sets."""
    if not references or len(references) > 12:
        raise ValueError("ref2va requires between 1 and 12 references")
    if any(ref.kind not in {"image", "video", "audio"} for ref in references):
        raise ValueError("Reference modality must be image, video, or audio")
    for kind, limit in (("image", 9), ("video", 3), ("audio", 3)):
        if sum(ref.kind == kind for ref in references) > limit:
            raise ValueError(f"MiniMax H3 accepts at most {limit} {kind} references")
    if all(ref.kind == "audio" for ref in references):
        raise ValueError("Audio references require an image or video reference")


def _read_audio(path: Path) -> tuple[torch.Tensor | None, int | None]:
    import av

    with av.open(str(path)) as container:
        if not container.streams.audio:
            return None, None
        stream = container.streams.audio[0]
        if len(stream.layout.channels) not in {1, 2}:
            raise ValueError("Reference soundtrack must be mono or stereo")
        # Match the previous local-media loader's single PyAV resampling pass.
        layout = "mono" if len(stream.layout.channels) == 1 else "stereo"
        resampler = av.AudioResampler(format="fltp", layout=layout, rate=32000)
        chunks = []
        for frame in container.decode(stream):
            chunks.extend(chunk.to_ndarray() for chunk in resampler.resample(frame))
        chunks.extend(chunk.to_ndarray() for chunk in resampler.resample(None))
        if not chunks:
            raise ValueError(f"Audio stream contains no samples: {path}")
        return torch.from_numpy(np.concatenate(chunks, axis=-1)).float(), 32000


def load_references(
    specs: tuple[MiniMaxH3ReferenceSpec, ...],
) -> list[MiniMaxH3Reference]:
    """Decode local RGB images and video/audio streams without model dependencies."""
    validate_references(specs)
    result = []
    for spec in specs:
        if spec.kind == "image":
            with Image.open(spec.path) as image:
                result.append(
                    MiniMaxH3Reference(
                        kind="image",
                        image=ImageOps.exif_transpose(image).convert("RGB"),
                    )
                )
            continue
        audio, sample_rate = _read_audio(spec.path)
        if spec.kind == "audio":
            if audio is None:
                raise ValueError(f"No audio stream: {spec.path}")
            result.append(
                MiniMaxH3Reference(kind="audio", audio=audio, sample_rate=sample_rate)
            )
            continue
        import av

        with av.open(str(spec.path)) as container:
            if not container.streams.video:
                raise ValueError(f"No video stream: {spec.path}")
            stream = container.streams.video[0]
            fps = float(stream.average_rate or stream.guessed_rate or 0)
            frames, rotation = [], 0.0
            for frame in container.decode(stream):
                rotation = frame.rotation
                frames.append(frame.to_ndarray(format="rgb24"))
        if not frames or not np.isfinite(fps) or fps <= 0:
            raise ValueError(
                f"Video must contain frames with a positive frame rate: {spec.path}"
            )
        frames = np.stack(frames)
        turns = round(rotation / 90.0) % 4
        if turns:
            frames = np.ascontiguousarray(np.rot90(frames, k=-turns, axes=(1, 2)))
        result.append(
            MiniMaxH3Reference(
                kind="video",
                frames=frames,
                fps=fps,
                audio=audio,
                sample_rate=sample_rate,
            )
        )
    return result
