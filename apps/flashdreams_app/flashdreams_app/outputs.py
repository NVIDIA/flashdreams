# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Host-owned presentation targets."""

from __future__ import annotations

from pathlib import Path

from flashdreams.infra.postprocess import VideoTensorLayout
from flashdreams.runtime.output import OutputArtifact
from flashdreams.runtime.types import StepResult
from flashdreams.runtime.video_output import Mp4VideoOutputTarget


class FileOutput:
    """Collect generated chunks and write one MP4 file when the run completes."""

    def __init__(
        self,
        *,
        path: Path,
        fps: int | float,
        output_layout: VideoTensorLayout,
        enabled: bool = True,
    ) -> None:
        self._target = Mp4VideoOutputTarget(
            output_path=path,
            fps=fps,
            output_layout=output_layout,
            enabled=enabled,
        )

    def open(self) -> None:
        """Open the underlying video writer."""
        self._target.open()

    def write(self, result: StepResult) -> None:
        """Append a generated chunk."""
        self._target.write(result)

    def close(self) -> tuple[OutputArtifact, ...]:
        """Finalize the MP4 and return its artifact metadata."""
        return tuple(self._target.close())
