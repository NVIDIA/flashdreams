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

"""Shared video output contracts for runners and serving transports."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor

from flashdreams.infra.postprocess import VideoPostprocessStream, VideoTensorLayout


def video_layout_time_dim(layout: VideoTensorLayout) -> int:
    """Return the time-axis index for a supported RGB video tensor layout."""
    if layout == "tchw":
        return 0
    if layout == "btchw":
        return 1
    if layout in ("bcthw", "bvtchw"):
        return 2
    raise ValueError(f"unsupported video layout: {layout}")


def infer_video_num_frames(tensor: Tensor, *, layout: VideoTensorLayout) -> int:
    """Infer a video chunk's frame count from its declared layout."""
    return int(tensor.shape[video_layout_time_dim(layout)])


@dataclass(slots=True)
class VideoStepResult:
    """One generated video chunk plus per-step metadata.

    The field names intentionally match the pre-existing WebRTC result shape
    so serving runtimes and output helpers share layout-aware chunk metadata.
    """

    chunk_index: int
    num_frames: int
    video_chunk: Tensor
    stats: dict[str, float] | None = None
    layout: VideoTensorLayout | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_video_chunk(
        cls,
        *,
        chunk_index: int,
        video_chunk: Tensor,
        layout: VideoTensorLayout,
        stats: dict[str, float] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> "VideoStepResult":
        """Build a result and infer ``num_frames`` from ``layout``."""
        return cls(
            chunk_index=chunk_index,
            num_frames=infer_video_num_frames(video_chunk, layout=layout),
            video_chunk=video_chunk,
            stats=stats,
            layout=layout,
            metadata=dict(metadata or {}),
        )


class RunnerVideoOutputStream:
    """Post-process, collect, and summarize runner video chunks."""

    def __init__(
        self,
        *,
        postprocess_stream: VideoPostprocessStream | None,
        output_layout: VideoTensorLayout,
        collect_output: bool = True,
        move_to_cpu: bool = True,
        empty_message: str = "runner emitted no video frames",
    ) -> None:
        self.postprocess_stream = postprocess_stream
        self.output_layout = output_layout
        self._time_dim = video_layout_time_dim(output_layout)
        self._collect_output = collect_output
        self.move_to_cpu = move_to_cpu
        self.empty_message = empty_message
        self._chunks: list[Tensor] = []
        self._closed = False
        self.stats_history: list[dict[str, object]] = []

    @property
    def collect_output(self) -> bool:
        """Return whether this stream collects chunks for rank-zero writing."""
        return self._collect_output

    def process(
        self,
        video_chunk: Tensor,
        *,
        autoregressive_index: int,
        stats: dict[str, float] | None = None,
        stats_extra: Mapping[str, object] | None = None,
    ) -> None:
        """Process one generated chunk and collect it when this rank writes output."""
        if self._closed:
            raise RuntimeError("cannot process video after finish()")
        processed = video_chunk
        if self.postprocess_stream is not None:
            processed = self.postprocess_stream.process(
                video_chunk,
                autoregressive_index=autoregressive_index,
            )
        self._append_if_nonempty(processed)
        if self.collect_output and stats is not None:
            if self.postprocess_stream is None:
                combined_stats: dict[str, object] = dict(stats)
            else:
                combined_stats = self.postprocess_stream.add_process_stats(stats)
            entry: dict[str, object] = {
                "autoregressive_index": autoregressive_index,
                **combined_stats,
            }
            if stats_extra is not None:
                entry.update(stats_extra)
            self.stats_history.append(entry)

    def finish(self) -> Tensor | None:
        """Flush post-processing and return the collected rank-zero video."""
        if self._closed:
            return None
        self._closed = True
        if self.postprocess_stream is not None:
            flushed = self.postprocess_stream.finish()
            if flushed is not None:
                self._append_if_nonempty(flushed)
        return self._collected_output()

    def _append_if_nonempty(self, output: Tensor) -> None:
        if not self.collect_output or output.shape[self._time_dim] == 0:
            return
        self._chunks.append(output.cpu() if self.move_to_cpu else output)

    def _collected_output(self) -> Tensor | None:
        if not self.collect_output:
            return None
        if not self._chunks:
            raise ValueError(self.empty_message)
        if len(self._chunks) == 1:
            return self._chunks.pop()
        output = torch.cat(self._chunks, dim=self._time_dim)
        self._chunks.clear()
        return output


__all__ = [
    "RunnerVideoOutputStream",
    "VideoStepResult",
    "infer_video_num_frames",
    "video_layout_time_dim",
]
