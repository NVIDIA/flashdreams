# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generated inference result contracts shared by runtimes and consumers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from torch import Tensor

from flashdreams.infra.postprocess import VideoTensorLayout
from flashdreams.infra.time import TimeWindow

if TYPE_CHECKING:
    from flashdreams.infra.video_output import LazyRGBFrame


@dataclass(frozen=True, kw_only=True, slots=True)
class StepResult:
    """Generated output and metadata returned by one inference step.

    Video results use :meth:`from_video_chunk`, which records a required tensor
    layout and derives the frame count once. Non-video results may use the
    regular constructor without a layout.
    """

    __hash__ = None

    step_index: int
    output: Any = None
    frame_count: int = 0
    layout: VideoTensorLayout | None = None
    output_window: TimeWindow | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    metrics: Mapping[str, float | int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.step_index < 0:
            raise ValueError("StepResult.step_index must be >= 0.")
        if self.frame_count < 0:
            raise ValueError("StepResult.frame_count must be >= 0.")
        if self.layout is not None:
            from flashdreams.infra.video_output import infer_video_num_frames

            video_chunk = self.video_chunk
            derived_frame_count = infer_video_num_frames(
                video_chunk,
                layout=self.layout,
            )
            if self.frame_count not in (0, derived_frame_count):
                raise ValueError(
                    "StepResult.frame_count does not match the declared video "
                    f"layout: expected {derived_frame_count}, got {self.frame_count}."
                )
            object.__setattr__(self, "frame_count", derived_frame_count)
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))
        object.__setattr__(self, "metrics", MappingProxyType(dict(self.metrics)))

    @classmethod
    def from_video_chunk(
        cls,
        *,
        step_index: int,
        video_chunk: Tensor,
        layout: VideoTensorLayout,
        output_window: TimeWindow | None = None,
        metadata: Mapping[str, Any] | None = None,
        metrics: Mapping[str, float | int] | None = None,
    ) -> StepResult:
        """Build one layout-aware generated-video result."""
        return cls(
            step_index=step_index,
            output=video_chunk,
            layout=layout,
            output_window=output_window,
            metadata=dict(metadata or {}),
            metrics=dict(metrics or {}),
        )

    @property
    def video_chunk(self) -> Tensor:
        """Return the video tensor or fail if this is not a video result."""
        if self.layout is None:
            raise ValueError("StepResult.layout is required for video output.")
        if not isinstance(self.output, Tensor):
            raise TypeError(
                "A video StepResult requires a torch.Tensor output, "
                f"got {type(self.output).__name__}."
            )
        return self.output

    def lazy_rgb_frames(
        self,
        *,
        batch_index: int = 0,
        view_index: int = 0,
        record_cuda_event: bool = True,
    ) -> list[LazyRGBFrame]:
        """Expose this video result as lazy per-frame RGB handles."""
        from flashdreams.infra.video_output import lazy_rgb_frames_from_video_tensor

        return lazy_rgb_frames_from_video_tensor(
            self.video_chunk,
            layout=self._video_layout(),
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
        """Return this video result as uint8 ``[T,H,W,C]`` on its device."""
        from flashdreams.infra.video_output import video_tensor_to_hwc_uint8

        return video_tensor_to_hwc_uint8(
            self.video_chunk,
            layout=self._video_layout(),
            batch_index=batch_index,
            view_index=view_index,
        )

    def _video_layout(self) -> VideoTensorLayout:
        if self.layout is None:
            raise ValueError("StepResult.layout is required for video output.")
        return self.layout


__all__ = ["StepResult"]
