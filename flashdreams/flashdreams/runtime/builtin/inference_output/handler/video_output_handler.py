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

"""Video artifact handler for generated frame chunks."""

from __future__ import annotations

import math
from pathlib import Path

import torch
from flashdreams.infra.runner_io import (
    DEFAULT_RUNNER_INSTALL_HINT,
    write_video_tensor,
)
from flashdreams.runtime.builtin.inference_output.frame_chunk import (
    FrameChunkOutput,
)
from flashdreams.runtime.output_system import InferenceOutputHandler
from pydantic import validate_call
from torch import Tensor

_TIMESTAMP_ABS_TOLERANCE_SECONDS = 1e-6
"""Tolerance for accumulated floating-point presentation timestamps."""


class VideoOutputHandler(InferenceOutputHandler):
    """Collect frame chunks and write a horizontally tiled video artifact."""

    artifact_path: Path
    """Path written when :meth:`finish` is called."""

    _chunks: list[Tensor]
    """CPU frame chunks awaiting artifact creation."""

    _chunk_shape: tuple[int, int, int, int, int] | None
    """Stable non-temporal ``[B, V, C, H, W]`` shape, if established."""

    _fps: float | None
    """Presentation frame rate established by the first chunk."""

    _next_timestamp: float | None
    """Expected presentation timestamp for the next chunk."""

    _finished: bool
    """Whether the artifact has been written successfully."""

    def __init__(self, artifact_path: str | Path) -> None:
        """Initialize an empty video artifact handler.

        Args:
            artifact_path: Destination passed to the shared FFmpeg video writer.
        """
        self.artifact_path = Path(artifact_path)
        self._chunks = []
        self._chunk_shape = None
        self._fps = None
        self._next_timestamp = None
        self._finished = False

    @validate_call
    def __call__(self, inference_output: FrameChunkOutput) -> None:
        """Collect one frame chunk for the output artifact.

        Args:
            inference_output: RGB frames in ``[B, V, T, C, H, W]`` layout.

        Raises:
            RuntimeError: :meth:`finish` has already written the artifact.
            ValueError: The chunk shape, frame rate, or timestamp is inconsistent
                with the output stream.
        """
        if self._finished:
            raise RuntimeError("cannot receive frame chunks after finish()")

        chunk = inference_output.value
        if chunk.ndim != 6:
            raise ValueError(
                "expected a rank-6 frame chunk in [B, V, T, C, H, W] layout; "
                f"got rank {chunk.ndim} with shape {tuple(chunk.shape)}"
            )
        batch, views, frames, channels, height, width = map(int, chunk.shape)
        if batch != 1 or channels != 3:
            raise ValueError(
                "expected frame chunk shape [B=1, V, T, C=3, H, W]; "
                f"got {tuple(chunk.shape)}"
            )
        if any(size <= 0 for size in (views, frames, height, width)):
            raise ValueError(
                "expected every frame chunk axis to be positive; "
                f"got {tuple(chunk.shape)}"
            )

        chunk_shape = (batch, views, channels, height, width)
        if self._chunk_shape is not None and chunk_shape != self._chunk_shape:
            raise ValueError(
                "expected frame chunks to share [B, V, C, H, W] dimensions; "
                f"expected {self._chunk_shape}, got {chunk_shape}"
            )

        fps = inference_output.fps
        if self._fps is not None and not math.isclose(
            fps,
            self._fps,
            rel_tol=1e-9,
            abs_tol=0.0,
        ):
            raise ValueError(f"expected frame chunk fps {self._fps}; got {fps}")
        if self._next_timestamp is not None and not math.isclose(
            inference_output.start_timestamp,
            self._next_timestamp,
            rel_tol=0.0,
            abs_tol=_TIMESTAMP_ABS_TOLERANCE_SECONDS,
        ):
            raise ValueError(
                "expected contiguous frame chunk timestamp "
                f"{self._next_timestamp}; got {inference_output.start_timestamp}"
            )

        self._chunks.append(chunk.detach().cpu())
        self._chunk_shape = chunk_shape
        self._fps = fps
        self._next_timestamp = inference_output.start_timestamp + frames / fps

    def finish(self) -> Path:
        """Finish receiving chunks and write the video artifact.

        Returns:
            Path written by the shared video writer. Repeated calls return the
            same path without writing the artifact again.

        Raises:
            ValueError: No frame chunks have been received.
            RuntimeError: Internal stream metadata is incomplete.
        """
        if self._finished:
            return self.artifact_path
        if not self._chunks:
            raise ValueError("cannot finish a video output without frame chunks")
        if self._fps is None:
            raise RuntimeError("video output frame rate was not initialized")

        video = torch.cat(self._chunks, dim=2)
        _, views, frames, channels, height, width = map(int, video.shape)
        canvas = (
            video[0]
            .permute(1, 3, 0, 4, 2)
            .reshape(frames, height, views * width, channels)
        )
        artifact_path = write_video_tensor(
            canvas,
            self.artifact_path,
            fps=self._fps,
            layout="thwc",
            install_hint=DEFAULT_RUNNER_INSTALL_HINT,
        )
        self.artifact_path = artifact_path
        self._chunks.clear()
        self._finished = True
        return artifact_path


__all__ = ["VideoOutputHandler"]
