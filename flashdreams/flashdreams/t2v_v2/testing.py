# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Test support for text-to-video integrations, shipped for them to import.

Nothing here runs in production. It is the shared check an integration's tests
call, and the stand-in model they run it against, named as ``numpy.testing`` and
``torch.testing`` are.
"""

import os
import shutil
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import torch

from flashdreams.api_v2.client_window import IClientWindow
from flashdreams.api_v2.output_sink import OutputSink
from flashdreams.runtime_v2.mp4_output_sink import Mp4OutputSink
from flashdreams.runtime_v2.session_desc import SessionDesc
from flashdreams.runtime_v2.session_runner import run_session
from flashdreams.runtime_v2.step_result import StepResult
from flashdreams.runtime_v2.user_input_events import UserInputEvents
from flashdreams.runtime_v2.video_encoder import result_to_rgb24_frames
from flashdreams.t2v_v2.application import T2VApplication

_REAL_CLIP_LUMINANCE = (16.0, 240.0)
"""Mean pixel value a real clip has to land inside, from ``0`` to ``255``.

Loose on purpose: what it catches is a run that came back blank.
"""

_REAL_CLIP_FRAME_DIFFERENCE = 0.5
"""How much consecutive frames of a real clip have to differ, on that scale."""


@dataclass(frozen=True, kw_only=True, slots=True)
class ExpectedFrameStats:
    """What a caller expects a run to have generated.

    Every field is optional, and one left out is not checked. A model that
    samples cannot be expected to produce a particular picture, but it can be
    expected to produce a picture at all.
    """

    frame_count: int | None = None
    """Frames the whole run should generate."""

    mean_luminance: tuple[float, float] | None = None
    """Range the mean pixel value should land in, from ``0`` to ``255``."""

    min_frame_difference: float | None = None
    """Smallest mean change from one frame to the next, on that scale."""


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
    application: T2VApplication,
    session_desc: SessionDesc | None = None,
    *,
    steps: int,
    expected: ExpectedFrameStats,
    commandline_args: Sequence[str] = (),
    mp4_path: str | Path | None = None,
) -> T2VCheckResult:
    """Run a text-to-video application for ``steps`` steps and inspect the video.

    The coverage an integration gets from one call: the application loads,
    resolves a session, generates, and what it generated is a video rather than
    a run that merely finished. It is initialized and closed here, and frames
    are read the way a sink reads them.

    Args:
        application: Uninitialized application to run.
        session_desc: Session to ask for, or ``None`` to take the application's
            own. A stand-in generating some other size says so here.
        steps: Steps to generate. Enough to reach steady state, since a model
            whose first chunk differs is only interesting from the second.
        expected: What the generated video should look like.
        commandline_args: Application arguments, such as a prompt.
        mp4_path: File to write as well, for a person to watch.

    Returns:
        What the run generated, and which expectations it missed.

    Raises:
        Whatever the run raises. A model that fails to load or generate is a
        failure of the integration rather than of an expectation.
    """
    if steps <= 0:
        raise ValueError(f"steps must be > 0, got {steps}.")

    inspector = _FrameInspector(Mp4OutputSink(mp4_path) if mp4_path else None)
    application.init(commandline_args)
    try:
        if session_desc is None:
            session_desc = application.session_desc()
        run_session(
            application.create_session(session_desc),
            _InspectingClientWindow(inspector),
            steps=steps,
        )
    finally:
        application.close()

    return _compare(inspector, expected, Path(mp4_path) if mp4_path else None)


def real_model_run_skip_reason(run_env: str) -> str | None:
    """Return why the real model cannot be run here, or ``None`` if it can.

    Such a run needs a GPU and downloads tens of gigabytes of checkpoint, so it
    is asked for rather than automatic. It carries ``ci_gpu`` and skips unless
    ``run_env`` is set; the ``manual`` marker describes it better and cannot be
    used, since ``pytest-manual-marker`` xfails those at setup.
    """
    if not os.environ.get(run_env):
        return f"set {run_env}=1 to download the checkpoints and generate a clip"
    if not torch.cuda.is_available():
        return "the model needs a GPU"
    if shutil.which("ffmpeg") is None:
        return "writing an MP4 needs ffmpeg on PATH"
    return None


def check_real_model_generates_a_clip(
    application: T2VApplication,
    *,
    prompt: str,
    steps: int,
    frame_count: int,
    mp4_path: str | Path,
) -> T2VCheckResult:
    """Generate a clip with a real checkpoint and check that it is a video.

    Every integration's real-model run is this one, and only the numbers differ.
    No session is described, since the clip worth watching is the one the model
    was trained for. Where it landed is printed, so a run made with ``-s`` says
    where to look.

    Args:
        application: Uninitialized application over the real model.
        prompt: Text to generate from, usually the integration's own default.
        steps: Blocks to generate.
        frame_count: Frames those blocks should decode to.
        mp4_path: File to write.
    """
    result = check_t2v_model_impl(
        application,
        steps=steps,
        # Compilation costs minutes and buys back milliseconds a block, which is
        # the wrong trade for a handful of blocks.
        commandline_args=["--prompt", prompt, "--no-compile"],
        expected=ExpectedFrameStats(
            frame_count=frame_count,
            mean_luminance=_REAL_CLIP_LUMINANCE,
            min_frame_difference=_REAL_CLIP_FRAME_DIFFERENCE,
        ),
        mp4_path=mp4_path,
    )
    print(f"\nwrote {mp4_path}\n{result}")
    return result


class FakeT2VPipeline:
    """A model's worth of behaviour, without a model.

    Generates frames of the shape and range a real text-to-video pipeline does,
    so an integration's tests can cover the seam a checkpoint plugs into on a
    CPU. Every call is recorded, so a test can assert the rollout was driven in
    order.
    """

    def __init__(
        self,
        *,
        width: int = 128,
        height: int = 64,
        compression_ratio: int = 8,
        first_block_frames: int = 9,
        block_frames: int = 12,
        fail_at: int | None = None,
    ) -> None:
        """
        Args:
            width: Frame width to generate. Not square by default, so a
                transposed frame cannot pass unnoticed.
            height: Frame height to generate.
            compression_ratio: Pixels one latent covers in each direction.
            first_block_frames: Frames the first block decodes, which a causal
                decoder usually has fewer of than the rest.
            block_frames: Frames every block after the first decodes.
            fail_at: Step to fail generating at, for covering a run that gave
                up part way through.
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
        self._fail_at = fail_at
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
        if autoregressive_index == self._fail_at:
            raise RuntimeError("generate failed")
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
        the checks made of a real video are meaningful here too.
        """
        shade = -0.5 + (frame_index % 8) / 8.0
        return torch.full((3, self.height, self.width), shade, dtype=torch.float32)


class FakeT2VPipelineConfig:
    """A pipeline config that builds a stand-in rather than loading a model."""

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

    What ``Mp4ClientWindow`` is for a run writing a file: the input half of a
    window nobody is on the other end of.
    """

    def __init__(self, sink: OutputSink) -> None:
        self._sink = sink

    def get_user_input_events(self) -> UserInputEvents:
        return UserInputEvents([])

    def open(self, session_desc: SessionDesc) -> None:
        self._sink.open(session_desc)

    def write(self, result: StepResult) -> None:
        self._sink.write(result)

    def close(self) -> None:
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
