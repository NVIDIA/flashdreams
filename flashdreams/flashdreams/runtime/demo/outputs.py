# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared demo output contracts and output construction."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Protocol, runtime_checkable

from flashdreams.infra.postprocess import VideoTensorLayout
from flashdreams.infra.runner_io import (
    DEFAULT_RUNNER_INSTALL_HINT,
    write_video_tensor,
)
from flashdreams.infra.video_output import VideoResultCollector, prepare_video_for_mp4
from flashdreams.runtime._utils import freeze_mapping
from flashdreams.runtime.output import NullOutputTarget, OutputArtifact, OutputTarget
from flashdreams.runtime.types import StepResult
from flashdreams.runtime.video_output import Mp4VideoOutputTarget, VideoWriter

from .spec import Mp4OutputSpec, NullOutputSpec, OutputSpec, WebRTCOutputSpec


@dataclass(frozen=True, kw_only=True, slots=True)
class SessionInfo:
    """Output-facing metadata known after session setup."""

    output_layout: str | None = None
    steady_output_frame_count: int | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.output_layout is not None and not self.output_layout.strip():
            raise ValueError("SessionInfo.output_layout must be non-empty when set.")
        if (
            self.steady_output_frame_count is not None
            and self.steady_output_frame_count < 0
        ):
            raise ValueError(
                "SessionInfo.steady_output_frame_count must be >= 0 when set."
            )
        object.__setattr__(self, "metadata", freeze_mapping(self.metadata))


@dataclass(frozen=True, kw_only=True, slots=True)
class OutputDecision:
    """Flow-control decision returned by an output sink after one step."""

    should_stop: bool = False
    dropped: bool = False
    drop_policy: Literal["none", "drop_newest", "drop_oldest"] = "none"
    backpressure_s: float = 0.0
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.drop_policy not in {"none", "drop_newest", "drop_oldest"}:
            raise ValueError(f"Unsupported drop_policy={self.drop_policy!r}.")
        if self.backpressure_s < 0:
            raise ValueError("OutputDecision.backpressure_s must be >= 0.")
        object.__setattr__(self, "metadata", freeze_mapping(self.metadata))


@runtime_checkable
class OutputSink(Protocol):
    """Consumes generated session outputs for a demo run mode."""

    produces_artifacts: bool

    def open(self, session_info: SessionInfo) -> None:
        """Prepare output resources for a session."""
        ...

    def begin_generation(self, generation: int) -> None:
        """Start an output generation, discarding stale live output if needed."""
        ...

    def write(self, result: StepResult) -> OutputDecision:
        """Consume one generated result and return output flow-control state."""
        ...

    def close(self) -> Sequence[OutputArtifact]:
        """Finalize output resources and return produced artifacts."""
        ...


@dataclass(slots=True)
class NullOutputSink:
    """Output sink for headless runs and fake-model vertical-slice tests."""

    store_results: bool = False
    produces_artifacts: bool = False
    output_count: int = field(default=0, init=False)
    results: list[Mapping[str, object]] = field(default_factory=list, init=False)
    opened: bool = field(default=False, init=False)
    closed: bool = field(default=False, init=False)
    session_info: SessionInfo | None = field(default=None, init=False)
    generation: int | None = field(default=None, init=False)

    def open(self, session_info: SessionInfo) -> None:
        self.session_info = session_info
        self.output_count = 0
        self.results.clear()
        self.opened = True
        self.closed = False

    def begin_generation(self, generation: int) -> None:
        if generation < 0:
            raise ValueError("generation must be >= 0.")
        self.generation = generation

    def write(self, result: StepResult) -> OutputDecision:
        if not self.opened or self.closed:
            raise RuntimeError("Cannot write to a closed output sink.")
        self.output_count += 1
        if self.store_results:
            self.results.append(_result_record(result))
        return OutputDecision()

    def close(self) -> Sequence[OutputArtifact]:
        self.closed = True
        return ()


@dataclass(slots=True)
class Mp4OutputSink:
    """MP4 artifact sink for shared demo drivers."""

    output_path: Path
    fps: int | float
    output_layout: VideoTensorLayout = "bvtchw"
    writer: VideoWriter = field(default=write_video_tensor, repr=False)
    install_hint: str = DEFAULT_RUNNER_INSTALL_HINT
    move_to_cpu: bool = True
    enabled: bool = True
    produces_artifacts: bool = True
    _opened: bool = field(default=False, init=False, repr=False)
    _closed: bool = field(default=True, init=False, repr=False)
    _collector: VideoResultCollector | None = field(
        default=None,
        init=False,
        repr=False,
    )
    _artifacts: tuple[OutputArtifact, ...] | None = field(
        default=None,
        init=False,
        repr=False,
    )
    session_info: SessionInfo | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        if float(self.fps) <= 0:
            raise ValueError("Mp4OutputSink.fps must be > 0.")
        self.output_path = Path(self.output_path)

    def open(self, session_info: SessionInfo) -> None:
        self.session_info = session_info
        self._collector = VideoResultCollector(
            output_layout=self.output_layout,
            enabled=self.enabled,
            move_to_cpu=self.move_to_cpu,
        )
        self._artifacts = None
        self._opened = True
        self._closed = False

    def begin_generation(self, generation: int) -> None:
        if generation < 0:
            raise ValueError("generation must be >= 0.")
        # MP4 recording is continuous across realtime resets; WebRTC is the sink
        # that drops stale generations.

    def write(self, result: StepResult) -> OutputDecision:
        if not self._opened or self._closed or self._collector is None:
            raise RuntimeError("Cannot write to a closed output sink.")
        if result.layout is None:
            raise TypeError("Mp4OutputSink requires a video StepResult with layout.")
        if result.layout != self.output_layout:
            raise ValueError(
                "Mp4OutputSink received layout "
                f"{result.layout!r}; expected {self.output_layout!r}."
            )
        self._collector.add(result)
        return OutputDecision()

    def close(self) -> Sequence[OutputArtifact]:
        if self._artifacts is not None:
            return self._artifacts
        if self._collector is None:
            self._opened = False
            self._closed = True
            self._artifacts = ()
            return self._artifacts

        collector = self._collector
        self._collector = None
        self._opened = False
        self._closed = True
        video = collector.finish()
        if video is None:
            self._artifacts = ()
            return self._artifacts
        writable_video, writable_layout = prepare_video_for_mp4(
            video,
            layout=self.output_layout,
        )
        path = self.writer(
            writable_video,
            self.output_path,
            fps=self.fps,
            layout=writable_layout,
            install_hint=self.install_hint,
        )
        self._artifacts = (
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
        return self._artifacts


def build_output_sink(
    output: OutputSpec,
    *,
    mp4_writer: VideoWriter | None = None,
) -> OutputSink:
    """Build a shared demo output sink from a demo output spec."""
    if isinstance(output, NullOutputSpec):
        return NullOutputSink(store_results=output.store_results)
    if isinstance(output, Mp4OutputSpec):
        writer = mp4_writer or write_video_tensor
        return Mp4OutputSink(
            output_path=Path(output.path),
            fps=output.fps,
            output_layout=output.output_layout,
            writer=writer,
            move_to_cpu=output.move_to_cpu,
        )
    if isinstance(output, WebRTCOutputSpec):
        raise ValueError("WebRTC output requires a realtime transport sink.")
    raise TypeError(f"Unsupported demo output spec: {type(output).__name__}.")


def _result_record(result: StepResult) -> Mapping[str, object]:
    record: dict[str, object] = {
        "step_index": result.step_index,
        "frame_count": result.frame_count,
        "metrics": dict(result.metrics),
        "metadata": dict(result.metadata),
    }
    if result.layout is not None:
        record["layout"] = result.layout
    if result.output_window is not None:
        record["output_window"] = (
            result.output_window.start_s,
            result.output_window.end_s,
        )
    return freeze_mapping(record)


def build_output_target(
    output: OutputSpec,
    *,
    mp4_writer: VideoWriter | None = None,
) -> OutputTarget:
    """Build a replay output target from a demo output spec."""
    if isinstance(output, NullOutputSpec):
        return NullOutputTarget(store_results=output.store_results)
    if isinstance(output, Mp4OutputSpec):
        output_path = Path(output.path)
        if mp4_writer is not None:
            return Mp4VideoOutputTarget(
                output_path=output_path,
                fps=output.fps,
                output_layout=output.output_layout,
                writer=mp4_writer,
                move_to_cpu=output.move_to_cpu,
            )
        return Mp4VideoOutputTarget(
            output_path=output_path,
            fps=output.fps,
            output_layout=output.output_layout,
            move_to_cpu=output.move_to_cpu,
        )
    if isinstance(output, WebRTCOutputSpec):
        raise ValueError("WebRTC output does not create a replay OutputTarget.")
    raise TypeError(f"Unsupported demo output spec: {type(output).__name__}.")


__all__ = [
    "Mp4OutputSink",
    "NullOutputSink",
    "OutputDecision",
    "OutputSink",
    "SessionInfo",
    "build_output_sink",
    "build_output_target",
]
