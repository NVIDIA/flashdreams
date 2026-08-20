# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Test support for text-to-video integrations, shipped for them to import.

Nothing here runs in production: the application, the session, and the command
line do not use it. It is the shared check an integration's own tests call to
cover the batch path, named as ``numpy.testing`` and ``torch.testing`` are so
that an importer can see what it is getting.

It also holds the stand-in model those tests run on a CPU. The pipeline contract
lives in :class:`~flashdreams.t2v_v2.session.T2VSession`, which every
integration shares, so the stand-in for it belongs here rather than being
written again by each of them.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import torch

from flashdreams.api_v2.application import IApplication
from flashdreams.api_v2.client_window import IClientWindow
from flashdreams.api_v2.output_sink import OutputSink
from flashdreams.runtime_v2.mp4_output_sink import Mp4OutputSink
from flashdreams.runtime_v2.session_desc import SessionDesc
from flashdreams.runtime_v2.session_runner import run_session
from flashdreams.runtime_v2.step_result import StepResult
from flashdreams.runtime_v2.user_input_events import UserInputEvents
from flashdreams.runtime_v2.video_encoder import result_to_rgb24_frames
from flashdreams.t2v_v2.application import T2VApplication


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
    session_desc: SessionDesc | None = None,
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
            resolves it to is what the frames are checked against. ``None`` asks
            the application what it would generate, which is what a check of the
            real model wants; a stand-in generating some other size says so here
            instead.
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
        if session_desc is None:
            session_desc = _described_by(application)
        run_session(
            application.create_session(session_desc),
            _InspectingClientWindow(inspector),
            steps=steps,
        )
    finally:
        application.close()

    return _compare(inspector, expected, Path(mp4_path) if mp4_path else None)


class FakeT2VPipeline:
    """A model's worth of behaviour, without a model.

    Generates frames of the shape and range a real text-to-video pipeline does,
    on whatever device it is handed, so an integration's tests can cover the
    seam a real checkpoint plugs into on a CPU runner. What it decodes is a
    moving grey rather than a picture, which is enough for the frame checks
    :func:`check_t2v_model_impl` makes.

    Every call is recorded, so a test can assert the rollout was driven in
    order: ``caches`` holds the arguments each rollout was initialized with, and
    ``generated`` and ``finalized`` hold the autoregressive index of each step.
    """

    def __init__(
        self,
        *,
        width: int = 128,
        height: int = 64,
        compression_ratio: int = 8,
        first_block_frames: int = 9,
        block_frames: int = 12,
    ) -> None:
        """
        Args:
            width: Frame width to generate. Not square by default, so a
                transposed frame cannot pass unnoticed.
            height: Frame height to generate.
            compression_ratio: Pixels one latent covers in each direction, as
                the decoder this stands in for reports.
            first_block_frames: Frames the first block decodes. A causal decoder
                usually decodes fewer for its first block than the rest.
            block_frames: Frames every block after the first decodes.
        """
        self.decoder = _FakeDecoder(compression_ratio)
        self.width = width
        self.height = height
        self.first_block_frames = first_block_frames
        self.block_frames = block_frames
        self.device: str | None = None
        self.eval_count = 0
        self.caches: list[dict[str, Any]] = []
        self.generated: list[int] = []
        self.finalized: list[int] = []
        self.closed = False
        self._frames_generated = 0

    def to(self, device: str) -> "FakeT2VPipeline":
        self.device = device
        return self

    def eval(self) -> "FakeT2VPipeline":
        self.eval_count += 1
        return self

    def initialize_cache(self, **kwargs: Any) -> object:
        self.caches.append(kwargs)
        self._frames_generated = 0
        return object()

    def generate(self, *, autoregressive_index: int, cache: object) -> torch.Tensor:
        del cache
        self.generated.append(autoregressive_index)
        count = (
            self.first_block_frames if autoregressive_index == 0 else self.block_frames
        )
        frames = torch.stack(
            [self._frame(self._frames_generated + index) for index in range(count)]
        )
        self._frames_generated += count
        return frames

    def finalize(self, *, autoregressive_index: int, cache: object) -> dict[str, float]:
        del cache
        self.finalized.append(autoregressive_index)
        return {"total_ms": 1.5}

    def close(self) -> None:
        self.closed = True

    def _frame(self, frame_index: int) -> torch.Tensor:
        """Return a grey frame whose shade moves with time.

        Mid grey rather than black or white, and moving rather than still, so
        the checks a caller makes of a real video are meaningful here too.
        """
        shade = -0.5 + (frame_index % 8) / 8.0
        return torch.full((3, self.height, self.width), shade, dtype=torch.float32)


class FakeT2VPipelineConfig:
    """A pipeline config that builds a stand-in rather than loading a model.

    What an application takes in place of the config its runner names, so a
    test can watch when the model is built as well as what it generates.
    """

    def __init__(self, pipeline: FakeT2VPipeline | None = None) -> None:
        """
        Args:
            pipeline: Stand-in to build. A default one is made when none is
                given, for a test that only cares that something was built.
        """
        self.pipeline = pipeline if pipeline is not None else FakeT2VPipeline()
        self.setup_count = 0

    def setup(self) -> FakeT2VPipeline:
        self.setup_count += 1
        return self.pipeline


class _FakeDecoder:
    """The one thing a session asks a decoder for."""

    def __init__(self, spatial_compression_ratio: int) -> None:
        self.spatial_compression_ratio = spatial_compression_ratio


def _described_by(application: IApplication) -> SessionDesc:
    """Ask an initialized application what session it would generate.

    Raises:
        TypeError: It has no way to say, so the caller has to describe one.
    """
    if not isinstance(application, T2VApplication):
        raise TypeError(
            f"{type(application).__name__} cannot describe the session it would "
            "generate, so check_t2v_model_impl needs one passed in."
        )
    return application.session_desc()


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


class _InspectingClientWindow(IClientWindow):
    """Drive a run against the inspector, reporting no input.

    A run goes through ``run_session``, which drives a window rather than a
    sink, and what a check wants driven is the inspector. This is what
    :class:`~flashdreams.runtime_v2.mp4_client_window.Mp4ClientWindow` is for a
    run writing a file: the input half of a window nobody is on the other end
    of.
    """

    def __init__(self, sink: OutputSink) -> None:
        """
        Args:
            sink: Where every result goes.
        """
        self._sink = sink

    def get_user_input_events(self) -> UserInputEvents:
        """Report nothing, since a check presses no keys."""
        return UserInputEvents([])

    def open(self, session_desc: SessionDesc) -> None:
        """Open the inspected sink for this session."""
        self._sink.open(session_desc)

    def write(self, result: StepResult) -> None:
        """Hand one step's result to the inspected sink."""
        self._sink.write(result)

    def close(self) -> None:
        """Close the inspected sink."""
        self._sink.close()


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
