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
from typing import Any, Literal, TypeAlias, cast

import torch
from torch import Tensor

from flashdreams.infra.acceleration.frame_prefetch import LazyCudaFrame
from flashdreams.infra.postprocess import VideoPostprocessStream, VideoTensorLayout
from flashdreams.infra.results import StepResult
from flashdreams.infra.time import TimeWindow

WritableVideoTensorLayout: TypeAlias = Literal["thwc", "tchw", "btchw", "bcthw"]


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
    expected_ndim = {
        "tchw": 4,
        "btchw": 5,
        "bcthw": 5,
        "bvtchw": 6,
    }.get(layout)
    if expected_ndim is None:
        raise ValueError(f"unsupported video layout: {layout!r}")
    if tensor.ndim != expected_ndim:
        raise ValueError(
            f"layout={layout!r} expects a {expected_ndim}D tensor, "
            f"got shape {tuple(tensor.shape)}."
        )
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


class VideoOutputStream:
    """Turn generated tensors into post-processed step results."""

    def __init__(
        self,
        *,
        postprocess_stream: VideoPostprocessStream | None,
        output_layout: VideoTensorLayout,
    ) -> None:
        self.postprocess_stream = postprocess_stream
        self.output_layout = output_layout
        self._closed = False
        self._last_step_index: int | None = None

    def process(
        self,
        video_chunk: Tensor,
        *,
        autoregressive_index: int,
        metrics: Mapping[str, float | int] | None = None,
        metadata: Mapping[str, Any] | None = None,
        output_window: TimeWindow | None = None,
    ) -> StepResult:
        """Post-process one generated chunk into the shared result boundary."""
        if self._closed:
            raise RuntimeError("cannot process video after finish()")
        processed = video_chunk
        result_metadata = dict(metadata or {})
        if self.postprocess_stream is not None:
            processed = self.postprocess_stream.process(
                video_chunk,
                autoregressive_index=autoregressive_index,
            )
            postprocess_stats = self.postprocess_stream.last_process_stats
            if postprocess_stats is not None:
                result_metadata["postprocess"] = postprocess_stats.as_dict()
        self._last_step_index = autoregressive_index
        return StepResult.from_video_chunk(
            step_index=autoregressive_index,
            video_chunk=processed.detach(),
            layout=self.output_layout,
            output_window=output_window,
            metrics=metrics,
            metadata=result_metadata,
        )

    def finish(self) -> StepResult | None:
        """Close the stream and return a post-processing tail, when present."""
        if self._closed:
            return None
        self._closed = True
        if self.postprocess_stream is None:
            return None
        flushed = self.postprocess_stream.finish()
        if flushed is None:
            return None
        if self._last_step_index is None:
            raise RuntimeError("post-processing emitted a tail before any video step")
        return StepResult.from_video_chunk(
            step_index=self._last_step_index,
            video_chunk=flushed.detach(),
            layout=self.output_layout,
            metadata={"postprocess_tail": True},
        )


class VideoResultCollector:
    """Collect video results for persistence or composed presentation."""

    def __init__(
        self,
        *,
        output_layout: VideoTensorLayout,
        enabled: bool = True,
        move_to_cpu: bool = True,
        empty_message: str = "runner emitted no video frames",
    ) -> None:
        self.output_layout = output_layout
        self.enabled = enabled
        self.move_to_cpu = move_to_cpu
        self.empty_message = empty_message
        self._time_dim = video_layout_time_dim(output_layout)
        self._chunks: list[Tensor] = []
        self.stats_history: list[dict[str, object]] = []

    @property
    def has_video(self) -> bool:
        """Whether at least one non-empty video chunk was collected."""
        return bool(self._chunks)

    def add(self, result: StepResult) -> None:
        """Collect one video result and its serializable statistics."""
        if result.layout != self.output_layout:
            raise ValueError(
                f"collector expected layout {self.output_layout!r}, "
                f"got {result.layout!r}."
            )
        if not self.enabled:
            return
        if result.frame_count > 0:
            chunk = result.video_chunk
            self._chunks.append(chunk.cpu() if self.move_to_cpu else chunk)
        entry: dict[str, object] = {
            "step_index": result.step_index,
            "frames": result.frame_count,
            **result.metrics,
        }
        if result.output_window is not None:
            entry["output_start_s"] = result.output_window.start_s
            entry["output_end_s"] = result.output_window.end_s
        if "postprocess" in result.metadata:
            entry["postprocess"] = result.metadata["postprocess"]
        if result.metadata.get("postprocess_tail"):
            entry["postprocess_tail"] = True
        self.stats_history.append(entry)

    def finish(self) -> Tensor | None:
        """Concatenate and return all collected video chunks."""
        if not self.enabled:
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
) -> tuple[Tensor, WritableVideoTensorLayout]:
    """Convert a stream output into a layout accepted by runner MP4 I/O."""
    if layout in {"thwc", "tchw", "btchw", "bcthw"}:
        return video, cast(WritableVideoTensorLayout, layout)
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
    "VideoResultCollector",
    "infer_video_num_frames",
    "lazy_rgb_frames_from_video_tensor",
    "prepare_video_for_mp4",
    "video_layout_time_dim",
    "video_tensor_to_hwc_uint8",
]
