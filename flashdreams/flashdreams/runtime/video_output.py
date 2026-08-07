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
from flashdreams.infra.video_output import VideoOutputStream
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
    _opened: bool = field(default=False, init=False, repr=False)
    _stream: VideoOutputStream | None = field(
        default=None,
        init=False,
        repr=False,
    )

    @property
    def closed(self) -> bool:
        return not self._opened

    def open(self) -> None:
        self._stream = VideoOutputStream(
            postprocess_stream=None,
            output_layout=self.output_layout,
            collect_output=True,
            move_to_cpu=self.move_to_cpu,
        )
        self._opened = True

    def write(self, result: StepResult) -> None:
        if not self._opened or self._stream is None:
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
        stats = dict(result.metrics)
        stats_extra: dict[str, object] = {
            "step_index": result.step_index,
            "frames": result.frame_count,
        }
        if result.output_window is not None:
            stats_extra["output_start_s"] = result.output_window.start_s
            stats_extra["output_end_s"] = result.output_window.end_s
        self._stream.process(
            result.video_chunk,
            autoregressive_index=result.step_index,
            stats=stats if stats else None,
            stats_extra=stats_extra,
        )

    def close(self) -> Sequence[OutputArtifact]:
        if self._stream is None:
            self._opened = False
            return ()

        stream = self._stream
        self._stream = None
        self._opened = False
        video = stream.finish()
        if video is None:
            return ()
        path = stream.write_mp4(
            video,
            self.output_path,
            fps=self.fps,
            layout=self.output_layout,
            writer=self.writer,
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
                    "stats_history": tuple(stream.stats_history),
                },
            ),
        )


__all__ = ["Mp4VideoOutputTarget"]
