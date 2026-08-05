# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""WebRTC video encoder facade for runtime-selected output paths."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

import torch
from aiortc import MediaStreamTrack

from flashdreams.serving.webrtc.media import BufferedVideoTrack

EncoderBackend = Literal["auto", "nvenc", "default"]


class EncoderInitError(RuntimeError):
    """Raised when a forced encoder backend cannot be initialized."""


@dataclass(slots=True, frozen=True)
class ChunkDeliveryResult:
    """Uniform result shape for encoder delivery."""

    backend: str
    num_frames: int
    num_keyframes: int
    encode_ms: float


@runtime_checkable
class VideoEncoder(Protocol):
    """Encoder backend paired with a compatible WebRTC media track."""

    fps: int
    backend: str
    prefers_codec: str | None

    def create_track(self, *, maxsize: int) -> BufferedVideoTrack: ...

    async def deliver_chunk(
        self,
        chunk: torch.Tensor,
        track: MediaStreamTrack,
        *,
        force_keyframe: bool = False,
    ) -> ChunkDeliveryResult: ...

    def close(self) -> None: ...


class DefaultRTCEncoder:
    """Software encoder that uses aiortc's normal media track path."""

    prefers_codec: str | None = None
    backend = "aiortc"

    def __init__(self, *, fps: int) -> None:
        if fps <= 0:
            raise ValueError(f"fps must be > 0, got {fps}")
        self.fps = fps

    def create_track(self, *, maxsize: int) -> BufferedVideoTrack:
        return BufferedVideoTrack(fps=self.fps, maxsize=maxsize)

    async def deliver_chunk(
        self,
        chunk: torch.Tensor,
        track: MediaStreamTrack,
        *,
        force_keyframe: bool = False,
    ) -> ChunkDeliveryResult:
        del force_keyframe
        if not isinstance(track, BufferedVideoTrack):
            raise TypeError(
                "DefaultRTCEncoder requires a BufferedVideoTrack; got "
                f"{type(track).__name__}."
            )
        enqueued = await track.enqueue_chunk(chunk)
        return ChunkDeliveryResult(
            backend=self.backend,
            num_frames=enqueued,
            num_keyframes=0,
            encode_ms=0.0,
        )

    def close(self) -> None:
        return


def select_encoder(
    *,
    backend: EncoderBackend,
    width: int,
    height: int,
    fps: int,
    bitrate: int,
    gpu_id: int,
    gop: int = 30,
) -> VideoEncoder:
    """Select an encoder implementation available on this base branch."""
    del width, height, bitrate, gpu_id, gop
    if backend == "nvenc":
        raise EncoderInitError(
            "encoder_backend='nvenc' requires the NVENC WebRTC encoder "
            "implementation, which is not present on this base branch."
        )
    return DefaultRTCEncoder(fps=fps)


__all__ = [
    "ChunkDeliveryResult",
    "DefaultRTCEncoder",
    "EncoderBackend",
    "EncoderInitError",
    "VideoEncoder",
    "select_encoder",
]
