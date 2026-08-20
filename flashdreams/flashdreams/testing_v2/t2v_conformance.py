# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared check a text-to-video integration runs to cover the batch path."""

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import numpy.typing as npt

from flashdreams.api_v2.application import IApplication
from flashdreams.api_v2.output_sink import OutputSink
from flashdreams.runtime_v2.batch_runner import run_batch
from flashdreams.runtime_v2.mp4_output_sink import Mp4OutputSink
from flashdreams.runtime_v2.session_desc import SessionDesc
from flashdreams.runtime_v2.step_result import StepResult
from flashdreams.runtime_v2.video_encoder import result_to_rgb24_frames


@dataclass(frozen=True, kw_only=True, slots=True)
class ExpectedFrameStats:
    """What a caller expects a run to have generated.

    Every field is optional, and one left out is not checked. They describe a
    video loosely on purpose: a model that samples cannot be expected to
    produce a particular picture, but it can be expected to produce a picture
    at all.
    """

    frame_count: int | None = None
    """Frames the whole run should generate."""

    mean_luminance: tuple[float, float] | None = None
    """Range the mean pixel value should land in, from ``0`` to ``255``.

    A run that fell over usually leaves black or white frames, so a middle
    range catches it without describing what was generated.
    """

    min_frame_difference: float | None = None
    """Smallest mean change from one frame to the next, from ``0`` to ``255``.

    A model asked for video should produce frames that differ. A still picture
    repeated is a failure this catches.
    """


@dataclass(frozen=True, kw_only=True, slots=True)
class T2VCheckResult:
    """What a run generated, and how it compared to what was expected."""

    failures: tuple[str, ...]
    """Expectations the run did not meet, in the order they were checked."""

    frames_per_step: tuple[int, ...]
    """Frames each step generated. A model whose first chunk is a different
    size to the rest shows up here."""

    mean_luminance: float
    """Mean pixel value over every frame, from ``0`` to ``255``."""

    frame_difference: float
    """Mean change from one frame to the next, over the whole run."""

    metrics: tuple[dict[str, float | int], ...] = field(default_factory=tuple)
    """Whatever each step reported, such as generation timings."""

    mp4_path: Path | None = None
    """File written, when one was asked for."""

    @property
    def passed(self) -> bool:
        """Whether the run met every expectation it was given."""
        return not self.failures

    @property
    def frame_count(self) -> int:
        """Frames the whole run generated."""
        return sum(self.frames_per_step)


def check_t2v_model_impl(
    application: IApplication,
    session_desc: SessionDesc,
    *,
    steps: int,
    expected: ExpectedFrameStats,
    commandline_args: Sequence[str] = (),
    mp4_path: str | Path | None = None,
) -> T2VCheckResult:
    """Run a text-to-video application for ``steps`` steps and inspect the video.

    The coverage a text-to-video integration gets from one call: the
    application loads, resolves a session, generates, and what it generated is
    a video rather than a run that merely finished. An integration supplies its
    own application and the loosest expectations that would still catch a
    broken model.

    The application is initialized and closed here, so a caller gets one run
    per call. Frames are read the way a sink reads them, so a result this
    accepts is one an MP4 or a window could also present.

    Args:
        application: Uninitialized application to run.
        session_desc: Session to ask the application for. What the session
            resolves it to is what the frames are checked against.
        steps: Steps to generate. Enough to reach steady state, since a model
            whose first chunk differs is only interesting from the second.
        expected: What the generated video should look like.
        commandline_args: Application arguments, such as a prompt.
        mp4_path: File to write as well, for a person to watch. Frames are
            checked either way.

    Returns:
        What the run generated, and which expectations it missed.

    Raises:
        Whatever the run raises. A model that fails to load or generate is a
        failure of the integration rather than of an expectation, so it is not
        collected into the result.
    """
    if steps <= 0:
        raise ValueError(f"steps must be > 0, got {steps}.")

    inspector = _FrameInspector(Mp4OutputSink(mp4_path) if mp4_path else None)
    application.init(commandline_args)
    try:
        run_batch(application.create_session(session_desc), inspector, steps=steps)
    finally:
        application.close()

    return _compare(inspector, expected, Path(mp4_path) if mp4_path else None)


class _FrameInspector(OutputSink):
    """Measure what a run generates, and pass it on to a file when asked to."""

    def __init__(self, mp4: Mp4OutputSink | None) -> None:
        """
        Args:
            mp4: File sink to write as well, or ``None`` to only measure.
        """
        self._mp4 = mp4
        self._session_desc: SessionDesc | None = None
        self._last_frame: npt.NDArray[np.uint8] | None = None
        self.frames_per_step: list[int] = []
        self.metrics: list[dict[str, float | int]] = []
        self.luminance_sum = 0.0
        self.difference_sum = 0.0
        self.difference_count = 0

    def open(self, session_desc: SessionDesc) -> None:
        self._session_desc = session_desc
        if self._mp4 is not None:
            self._mp4.open(session_desc)

    def write(self, result: StepResult) -> None:
        if self._session_desc is None:
            raise RuntimeError("open() must run before write().")
        frames = result_to_rgb24_frames(result, self._session_desc)
        self.frames_per_step.append(len(frames))
        self.metrics.append(dict(result.metrics))
        self.luminance_sum += float(frames.mean()) * len(frames)
        # The previous step's last frame leads this one, so the change across a
        # step boundary counts like any other.
        sequence = frames
        if self._last_frame is not None:
            sequence = np.concatenate([self._last_frame[np.newaxis], frames])
        if len(sequence) > 1:
            change = np.abs(np.diff(sequence.astype(np.int16), axis=0))
            self.difference_sum += float(change.mean()) * (len(sequence) - 1)
            self.difference_count += len(sequence) - 1
        self._last_frame = frames[-1]
        if self._mp4 is not None:
            self._mp4.write(result)

    def close(self) -> None:
        if self._mp4 is not None:
            self._mp4.close()


def _compare(
    inspector: _FrameInspector,
    expected: ExpectedFrameStats,
    mp4_path: Path | None,
) -> T2VCheckResult:
    """Measure what was generated and collect the expectations it missed."""
    frame_count = sum(inspector.frames_per_step)
    luminance = inspector.luminance_sum / frame_count if frame_count else 0.0
    difference = (
        inspector.difference_sum / inspector.difference_count
        if inspector.difference_count
        else 0.0
    )

    failures: list[str] = []
    if expected.frame_count is not None and frame_count != expected.frame_count:
        failures.append(
            f"Expected {expected.frame_count} frames, generated {frame_count} "
            f"as {inspector.frames_per_step}."
        )
    if expected.mean_luminance is not None:
        low, high = expected.mean_luminance
        if not low <= luminance <= high:
            failures.append(
                f"Mean luminance {luminance:.1f} is outside [{low}, {high}]."
            )
    if (
        expected.min_frame_difference is not None
        and difference < expected.min_frame_difference
    ):
        failures.append(
            f"Frames change by {difference:.2f} on average, less than the "
            f"{expected.min_frame_difference} expected of a video."
        )

    return T2VCheckResult(
        failures=tuple(failures),
        frames_per_step=tuple(inspector.frames_per_step),
        mean_luminance=luminance,
        frame_difference=difference,
        metrics=tuple(inspector.metrics),
        mp4_path=mp4_path,
    )
