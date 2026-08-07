# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Local-window output target for decoded RGB video results."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from flashdreams.infra.video_output import VideoStepResult
from flashdreams.runtime.output import OutputArtifact
from flashdreams.runtime.output_schema import RGB_VIDEO, OutputTargetRequirement
from flashdreams.runtime.types import StepResult
from flashdreams.serving.presentation.base import (
    HudOverlay,
    PresenterBackend,
    SupportsPrepareFrame,
)
from flashdreams.serving.presentation.frame import DisplayFrame
from flashdreams.serving.presentation.local_window import (
    LocalWindowPresenter,
    WindowConfig,
)

PresenterFactory = Callable[..., PresenterBackend]
DisplayFrameProjector = Callable[
    [StepResult, VideoStepResult, int, Any, int],
    DisplayFrame,
]


def _default_frame_projector(
    result: StepResult,
    video: VideoStepResult,
    frame_index: int,
    image: Any,
    timestamp_us: int,
) -> DisplayFrame:
    del result, video, frame_index
    return DisplayFrame(image=image, timestamp_us=timestamp_us)


@dataclass(slots=True)
class LocalWindowVideoOutputTarget:
    """Present decoded RGB video chunks in a native local window."""

    overlay: HudOverlay
    """Demo-owned chrome and local UI behavior."""

    config: WindowConfig = field(default_factory=WindowConfig)
    """Native window configuration."""

    batch_index: int = 0
    """Batch element displayed from each generated result."""

    view_index: int = 0
    """View displayed from each generated result."""

    presenter_factory: PresenterFactory = field(
        default=LocalWindowPresenter,
        repr=False,
    )
    """Factory used to construct the native presenter."""

    close_presenter_on_close: bool = True
    """Whether target close also closes the constructed presenter."""

    frame_projector: DisplayFrameProjector | None = field(default=None, repr=False)
    """App-owned projection from generated result metadata to display frames."""

    _presenter: PresenterBackend | None = field(default=None, init=False, repr=False)
    """Presenter owned while the target is open."""

    _opened: bool = field(default=False, init=False, repr=False)
    """Whether this target currently accepts writes."""

    @property
    def output_requirement(self) -> OutputTargetRequirement:
        """Require decoded RGB video chunks."""
        return OutputTargetRequirement(
            modalities=frozenset({RGB_VIDEO}),
            python_type=VideoStepResult,
        )

    @property
    def should_stop(self) -> bool:
        """Return whether the user closed the native window."""
        return self._presenter is not None and self._presenter.should_close

    def open(self) -> None:
        """Construct the native presenter for a new run."""
        if self._opened:
            raise RuntimeError("LocalWindowVideoOutputTarget is already open.")
        if self._presenter is None:
            self._presenter = self.presenter_factory(
                overlay=self.overlay,
                config=self.config,
            )
        self._opened = True

    def poll(self) -> None:
        """Process pending native-window events."""
        if not self._opened or self._presenter is None:
            raise RuntimeError("Cannot poll a closed output target.")
        self._presenter.process_events()

    def write(self, result: StepResult) -> None:
        """Present every RGB frame carried by one inference step."""
        presenter = self._presenter
        if not self._opened or presenter is None:
            raise RuntimeError("Cannot write to a closed output target.")
        video_result = result.output
        if not isinstance(video_result, VideoStepResult):
            raise TypeError(
                "LocalWindowVideoOutputTarget requires StepResult.output to be "
                f"VideoStepResult, got {type(video_result).__name__}."
            )

        frames = video_result.lazy_rgb_frames(
            batch_index=self.batch_index,
            view_index=self.view_index,
        )
        for frame_index, image in enumerate(frames):
            self.poll()
            if self.should_stop:
                return
            timestamp_us = _frame_timestamp_us(
                result=result,
                frame_index=frame_index,
                frame_count=len(frames),
            )
            projector = self.frame_projector or _default_frame_projector
            display_frame = projector(
                result,
                video_result,
                frame_index,
                image,
                timestamp_us,
            )
            if isinstance(presenter, SupportsPrepareFrame):
                presenter.prepare_frame(display_frame)
            presenter.present_frame(display_frame)

    def close(self) -> Sequence[OutputArtifact]:
        """Finish one session without producing a persistent artifact."""
        self._opened = False
        if self.close_presenter_on_close:
            self.shutdown()
        return ()

    def reset_camera(self) -> None:
        """Drop camera state before presenting a new producer/session."""
        presenter = self._presenter
        reset = None if presenter is None else getattr(presenter, "reset_camera", None)
        if callable(reset):
            reset()

    def shutdown(self) -> None:
        """Release the retained presenter at application shutdown."""
        presenter, self._presenter = self._presenter, None
        self._opened = False
        if presenter is not None:
            presenter.close()


def _frame_timestamp_us(
    *,
    result: StepResult,
    frame_index: int,
    frame_count: int,
) -> int:
    output_window = result.output_window
    if output_window is None or frame_count <= 0:
        return 0
    frame_interval_s = (output_window.end_s - output_window.start_s) / frame_count
    return round((output_window.start_s + frame_index * frame_interval_s) * 1_000_000)


__all__ = [
    "DisplayFrameProjector",
    "LocalWindowVideoOutputTarget",
    "PresenterFactory",
]
