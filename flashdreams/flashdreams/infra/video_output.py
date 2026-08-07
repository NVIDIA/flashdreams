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

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from flashdreams.infra.acceleration.frame_prefetch import LazyCudaFrame
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


class LazyRGBFrame(LazyCudaFrame):
    """Defer RGB frame host materialization until a host-only consumer needs it."""

    def __init__(
        self,
        frames_hwc_uint8: Any,
        frame_index: int,
        *,
        source_event: object | None = None,
    ) -> None:
        super().__init__(
            frames_hwc_uint8,
            frame_index,
            source_event=source_event,
            lost_source_message=(
                "Lazy RGB frame lost its source tensor before materialization."
            ),
            already_materialized_message=(
                "Lazy RGB frame was already materialized on the host."
            ),
        )


def video_tensor_to_hwc_uint8(
    video: Tensor,
    *,
    layout: VideoTensorLayout,
    batch_index: int = 0,
    view_index: int = 0,
) -> Tensor:
    """Convert a GPU/CPU RGB video chunk to uint8 ``[T, H, W, C]`` tensor.

    The conversion preserves the source device. Callers that need host frames
    can materialize the returned tensor later; GPU-aware consumers can keep it
    resident for interop or hardware encoding.
    """
    if layout == "tchw":
        frames = video
    elif layout == "btchw":
        frames = video[batch_index]
    elif layout == "bcthw":
        frames = video[batch_index].permute(1, 0, 2, 3)
    elif layout == "bvtchw":
        frames = video[batch_index, view_index]
    else:
        raise ValueError(f"unsupported video layout: {layout!r}")

    if frames.ndim != 4:
        raise ValueError(
            f"expected a 4D [T,C,H,W] frame tensor, got {tuple(frames.shape)}"
        )
    if frames.shape[1] != 3:
        raise ValueError(
            "expected RGB video frames with C=3 in [T,C,H,W], "
            f"got {tuple(frames.shape)}"
        )

    if frames.dtype != torch.uint8:
        frames = frames.clamp(-1.0, 1.0)
        frames = ((frames + 1.0) * 127.5).round().to(torch.uint8)
    return frames.detach().permute(0, 2, 3, 1).contiguous()


def lazy_rgb_frames_from_video_tensor(
    video: Tensor,
    *,
    layout: VideoTensorLayout,
    batch_index: int = 0,
    view_index: int = 0,
    record_cuda_event: bool = True,
) -> list[LazyRGBFrame]:
    """Return lazy per-frame RGB handles backed by one video tensor chunk."""
    frames = video_tensor_to_hwc_uint8(
        video,
        layout=layout,
        batch_index=batch_index,
        view_index=view_index,
    )
    source_event = None
    if record_cuda_event and frames.is_cuda:
        source_event = torch.cuda.Event()
        source_event.record(torch.cuda.current_stream(frames.device))
    return [
        LazyRGBFrame(frames, frame_index, source_event=source_event)
        for frame_index in range(frames.shape[0])
    ]


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
    ) -> VideoStepResult:
        """Build a result and infer ``num_frames`` from ``layout``."""
        return cls(
            chunk_index=chunk_index,
            num_frames=infer_video_num_frames(video_chunk, layout=layout),
            video_chunk=video_chunk,
            stats=stats,
            layout=layout,
            metadata=dict(metadata or {}),
        )

    def lazy_rgb_frames(
        self,
        *,
        batch_index: int = 0,
        view_index: int = 0,
        record_cuda_event: bool = True,
    ) -> list[LazyRGBFrame]:
        """Expose this chunk as lazy per-frame RGB handles."""
        if self.layout is None:
            raise ValueError("VideoStepResult.layout is required for frame extraction")
        return lazy_rgb_frames_from_video_tensor(
            self.video_chunk,
            layout=self.layout,
            batch_index=batch_index,
            view_index=view_index,
            record_cuda_event=record_cuda_event,
        )

    def video_hwc_uint8(
        self,
        *,
        batch_index: int = 0,
        view_index: int = 0,
    ) -> Tensor:
        """Return this chunk as a uint8 ``[T,H,W,C]`` tensor on its source device."""
        if self.layout is None:
            raise ValueError("VideoStepResult.layout is required for frame extraction")
        return video_tensor_to_hwc_uint8(
            self.video_chunk,
            layout=self.layout,
            batch_index=batch_index,
            view_index=view_index,
        )


class VideoOutputStream:
    """Post-process and optionally collect generated video tensors.

    Runner CLI, realtime serving, and local presentation all use this same
    tensor-in/tensor-out boundary. Transport-specific result envelopes and
    frame conversions happen after :meth:`process`.
    """

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
    ) -> Tensor:
        """Process one chunk, optionally collect it, and return emitted frames."""
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
        return processed

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

    def make_step_result(
        self,
        video_chunk: Tensor,
        *,
        autoregressive_index: int,
        stats: dict[str, float] | None = None,
        metadata: Mapping[str, Any] | None = None,
        sync_device: torch.device | str | None = None,
    ) -> VideoStepResult:
        """Process a chunk and package the emitted frames for a live consumer.

        ``sync_device`` is useful for consumers such as WebRTC that hand a
        GPU-resident result to another subsystem immediately after generation.
        It synchronizes only when it names a CUDA device and never moves the
        emitted tensor to the host.
        """
        processed = self.process(
            video_chunk,
            autoregressive_index=autoregressive_index,
            stats=stats,
        )
        if sync_device is not None:
            device = torch.device(sync_device)
            if device.type == "cuda":
                torch.cuda.current_stream(device).synchronize()
        return VideoStepResult.from_video_chunk(
            chunk_index=autoregressive_index,
            video_chunk=processed.detach(),
            layout=self.output_layout,
            stats=stats,
            metadata=metadata,
        )

    def finish_to_mp4(
        self,
        output_path: str | Path,
        *,
        fps: int | float,
        writer: Callable[..., Path] | None = None,
        install_hint: str | None = None,
    ) -> Path | None:
        """Finish this collecting stream and write its frames as one MP4.

        The stream converts its declared output layout to the runner I/O
        layout, including tiling ``bvtchw`` views horizontally.
        """
        video = self.finish()
        if video is None:
            return None
        return self.write_mp4(
            video,
            output_path,
            fps=fps,
            layout=self.output_layout,
            writer=writer,
            install_hint=install_hint,
        )

    def write_mp4(
        self,
        video: Tensor,
        output_path: str | Path,
        *,
        fps: int | float,
        layout: VideoTensorLayout | str | None = None,
        writer: Callable[..., Path] | None = None,
        install_hint: str | None = None,
    ) -> Path:
        """Write video frames as MP4 using this stream's runner output path.

        ``layout`` defaults to :attr:`output_layout`; callers that compose a
        presentation canvas can pass the runner-I/O ``thwc`` layout directly.
        """
        from flashdreams.infra.runner_io import (
            DEFAULT_RUNNER_INSTALL_HINT,
            write_video_tensor,
        )

        writable_video, writable_layout = prepare_video_for_mp4(
            video, layout=layout or self.output_layout
        )
        output_writer = writer or write_video_tensor
        path = output_writer(
            writable_video,
            output_path,
            fps=fps,
            layout=writable_layout,
            install_hint=install_hint or DEFAULT_RUNNER_INSTALL_HINT,
        )
        return path

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


def prepare_video_for_mp4(
    video: Tensor,
    *,
    layout: VideoTensorLayout | str,
) -> tuple[Tensor, str]:
    """Convert a stream output into a layout accepted by runner MP4 I/O."""
    if layout in {"thwc", "tchw", "btchw", "bcthw"}:
        return video, layout
    if layout == "bvtchw":
        if video.ndim != 6:
            raise ValueError(
                "layout='bvtchw' expects a 6D [B,V,T,C,H,W] tensor, "
                f"got {tuple(video.shape)}."
            )
        if video.shape[0] != 1:
            raise ValueError(
                "layout='bvtchw' MP4 writing expects a single batch element, "
                f"got {tuple(video.shape)}."
            )
        _, views, frames, channels, height, width = video.shape
        canvas = (
            video[0]
            .permute(1, 3, 0, 4, 2)
            .contiguous()
            .reshape(frames, height, views * width, channels)
        )
        return canvas, "thwc"
    raise ValueError(f"unsupported video layout for MP4: {layout!r}")


__all__ = [
    "LazyRGBFrame",
    "VideoOutputStream",
    "VideoStepResult",
    "infer_video_num_frames",
    "lazy_rgb_frames_from_video_tensor",
    "prepare_video_for_mp4",
    "video_layout_time_dim",
    "video_tensor_to_hwc_uint8",
]
