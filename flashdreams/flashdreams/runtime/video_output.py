# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Video output targets for the runtime API."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from flashdreams.infra.postprocess import VideoTensorLayout
from flashdreams.infra.runner_io import (
    DEFAULT_RUNNER_INSTALL_HINT,
    write_video_tensor,
)
from flashdreams.infra.video_output import VideoResultCollector, prepare_video_for_mp4
from flashdreams.runtime.output import OutputArtifact
from flashdreams.runtime.types import StepResult

VideoWriter = Callable[..., Path]


@dataclass(slots=True)
class Mp4VideoOutputTarget:
    """Write layout-aware runtime step results to one MP4 artifact."""

    output_path: Path
    fps: int | float
    output_layout: VideoTensorLayout = "bvtchw"
    writer: VideoWriter = field(default=write_video_tensor, repr=False)
    install_hint: str = DEFAULT_RUNNER_INSTALL_HINT
    move_to_cpu: bool = True
    enabled: bool = True
    _opened: bool = field(default=False, init=False, repr=False)
    _collector: VideoResultCollector | None = field(
        default=None,
        init=False,
        repr=False,
    )

    @property
    def closed(self) -> bool:
        return not self._opened

    def open(self) -> None:
        self._collector = VideoResultCollector(
            output_layout=self.output_layout,
            enabled=self.enabled,
            move_to_cpu=self.move_to_cpu,
        )
        self._opened = True

    def write(self, result: StepResult) -> None:
        if not self._opened or self._collector is None:
            raise RuntimeError("Cannot write to a closed output target.")
        if result.layout is None:
            raise TypeError(
                "Mp4VideoOutputTarget requires a video StepResult with layout."
            )
        if result.layout != self.output_layout:
            raise ValueError(
                "Mp4VideoOutputTarget received layout "
                f"{result.layout!r}; expected {self.output_layout!r}."
            )
        self._collector.add(result)

    def close(self) -> Sequence[OutputArtifact]:
        if self._collector is None:
            self._opened = False
            return ()

        collector = self._collector
        self._collector = None
        self._opened = False
        video = collector.finish()
        if video is None:
            return ()
        writable_video, writable_layout = prepare_video_for_mp4(
            video, layout=self.output_layout
        )
        path = self.writer(
            writable_video,
            self.output_path,
            fps=self.fps,
            layout=writable_layout,
            install_hint=self.install_hint,
        )
        return (
            OutputArtifact(
                kind="video/mp4",
                uri=str(path),
                metadata={
                    "fps": self.fps,
                    "source_layout": self.output_layout,
                    "shape": tuple(int(dim) for dim in video.shape),
                    "stats_history": tuple(collector.stats_history),
                },
            ),
        )


__all__ = ["Mp4VideoOutputTarget"]
