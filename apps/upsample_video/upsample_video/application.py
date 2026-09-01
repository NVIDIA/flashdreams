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

"""Big Buck Bunny upsampling through the FlashDreams v2 application API."""

from __future__ import annotations

import argparse
import os
import shutil
import tempfile
import zipfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from flashdreams.api_v2.application import IApplication
from flashdreams.api_v2.loop import IModelLoop
from flashdreams.api_v2.session import ISession
from flashdreams.core.io.disk import default_flashdreams_cache_dir
from flashdreams.infra.postprocess import (
    VideoChunk,
    VideoPostProcessorConfig,
    VideoPostProcessorSession,
    VideoSpec,
)
from flashdreams.infra.postprocess.base import concatenate_video_chunks
from flashdreams.infra.runner_io import resolve_input_path
from flashdreams.runtime_v2.session_desc import (
    BackpressureMode,
    PresentationMode,
    SessionDesc,
)
from flashdreams.runtime_v2.step_result import StepResult
from flashdreams.runtime_v2.user_input_events import UserInputEvents
from flashdreams.runtime_v2.video_tensor import VideoTensorLayout

_BIG_BUCK_BUNNY_FILENAME = "big_buck_bunny_480p_h264.mov"
"""Video contained by the Blender-hosted demo archive."""

_BIG_BUCK_BUNNY_URL = (
    "https://download.blender.org/peach/bigbuckbunny_movies/"
    f"{_BIG_BUCK_BUNNY_FILENAME}.zip"
)
"""Public Blender archive used by the uninteractive upsampling demo."""

_BIG_BUCK_BUNNY_SPEC = VideoSpec(height=480, width=853, fps=24.0)
"""Source dimensions and frame rate of the bundled Big Buck Bunny encode."""

_DEFAULT_MAX_CHUNKS = 4
"""Number of chunks processed by default, keeping the demo short enough to inspect."""

_INPUT_CACHE_DIR = default_flashdreams_cache_dir() / "upsample-video"
"""User-writable cache for the compressed demo input and extracted MP4."""

InputLoader = Callable[[int], "LoadedVideo"]


@dataclass(frozen=True, slots=True)
class LoadedVideo:
    """Bounded input video decoded for one application run."""

    frames: Tensor
    """Normalized source frames in ``[T, C, H, W]`` layout."""

    spec: VideoSpec
    """Spatial and timing metadata reported by the source file."""


@dataclass(frozen=True, slots=True)
class UpsampleVideoApplicationDefaults:
    """Integration-provided defaults for the reusable upsampling demo."""

    processor: VideoPostProcessorConfig
    """Video post-processor selected by the model integration."""

    first_chunk_size: int
    """Input frames consumed by the cold-start model step."""

    steady_chunk_size: int
    """Input frames consumed by every steady-state model step."""

    model_name: str
    """Model configuration name reported in session metadata."""

    max_chunks: int = _DEFAULT_MAX_CHUNKS
    """Default number of source-video chunks to process."""


@dataclass(frozen=True, slots=True)
class _ApplicationConfig:
    """Resolved upsampling settings and input shared with one session."""

    processor: VideoPostProcessorConfig
    """Video post-processor settings selected by the application factory."""

    video: LoadedVideo
    """Bounded Big Buck Bunny input held on CPU."""

    chunks: tuple[tuple[int, int], ...]
    """Input frame ranges consumed by consecutive model steps."""


@dataclass(slots=True)
class UpsampleVideoModelState:
    """Mutable stream state owned by the upsampling model loop."""

    config: _ApplicationConfig
    """Resolved application settings and source frames."""

    session_desc: SessionDesc
    """Output contract presented by the runtime."""

    processor_session: VideoPostProcessorSession
    """Stateful video processor for the current rollout."""

    chunks_generated: int = 0
    """Number of input ranges already consumed."""


class UpsampleVideoModelLoop(IModelLoop[UpsampleVideoModelState]):
    """Upsample one bounded input range per model step."""

    def step(self, step_index: int, events: UserInputEvents) -> list[StepResult]:
        """Upsample the next source-video range.

        Args:
            step_index: Zero-based chunk index since the latest reset.
            events: User input ignored by this uninteractive application.

        Returns:
            One upsampled video channel in ``bcthw`` layout.

        Raises:
            RuntimeError: Steps arrive out of sequence or the processor buffers
                an expected complete chunk without emitting output.
        """
        del events
        state = self.state
        if step_index != state.chunks_generated:
            raise RuntimeError(
                "Upsample-video step is out of sequence: expected "
                f"{state.chunks_generated}, got {step_index}."
            )

        start, size = state.config.chunks[step_index]
        source = state.config.video.frames[start : start + size]
        emitted = state.processor_session.process(
            VideoChunk(
                tensor=source,
                layout="tchw",
                metadata={"input_start": start, "input_frames": size},
            )
        )
        if step_index == len(state.config.chunks) - 1:
            emitted.extend(state.processor_session.flush())
        if not emitted:
            raise RuntimeError(
                f"The video processor emitted no output for complete input chunk {step_index}."
            )

        output = concatenate_video_chunks(emitted, layout="bcthw").detach()
        frame_count = int(output.shape[2])
        state.chunks_generated += 1
        return [
            StepResult(
                step_index=step_index,
                output=output,
                frame_count=frame_count,
                output_layout=state.session_desc.output_layout,
                metrics={
                    "input_frames": size,
                    "output_frames": frame_count,
                },
            )
        ]

    def is_finished(self) -> bool:
        """Return whether every bounded input range has been upsampled."""
        return self.state.chunks_generated >= len(self.state.config.chunks)

    def reset(self) -> None:
        """Start a fresh processor stream over the same source frames."""
        state = self.state
        state.processor_session = _start_processor(state.config)
        state.chunks_generated = 0

    def close(self) -> None:
        """Release state retained by the processor session."""
        self.state.processor_session = _ClosedPostProcessorSession()


class UpsampleVideoSession(ISession):
    """One finite Big Buck Bunny upsampling run."""

    def __init__(self, config: _ApplicationConfig, session_desc: SessionDesc) -> None:
        self._config = config
        self._session_desc = session_desc
        self._state: UpsampleVideoModelState | None = None

    def init(self) -> None:
        """Create the stream processor and register the model loop."""
        state = UpsampleVideoModelState(
            config=self._config,
            session_desc=self._session_desc,
            processor_session=_start_processor(self._config),
        )
        self._state = state
        self.register_model_loop(UpsampleVideoModelLoop, state=state)

    @property
    def session_desc(self) -> SessionDesc:
        """Return the fixed output shape and timing contract."""
        return self._session_desc

    def close(self) -> None:
        """Release the session-owned source and processor references."""
        self._state = None


class UpsampleVideoApplication(IApplication):
    """Upsample a short Big Buck Bunny excerpt without interactive controls."""

    def __init__(
        self,
        *,
        defaults: UpsampleVideoApplicationDefaults,
        input_loader: InputLoader | None = None,
        input_spec: VideoSpec = _BIG_BUCK_BUNNY_SPEC,
    ) -> None:
        """Create a lazy upsampling application.

        Args:
            defaults: Model integration defaults for this application.
            input_loader: Test seam replacing download and bounded video decode.
            input_spec: Expected source-video contract.
        """
        self.defaults = defaults
        self._input_loader = input_loader or _load_big_buck_bunny
        self._input_spec = input_spec
        self._config: _ApplicationConfig | None = None

    def init(self, commandline_args: Sequence[str]) -> None:
        """Parse execution settings and decode the bounded source excerpt.

        Args:
            commandline_args: Application-specific command-line arguments.

        Raises:
            ValueError: A setting is invalid or the decoded source does not
                match the advertised input contract.
        """
        parser = argparse.ArgumentParser(
            prog="flashdreams-run-v2 <upsample-video-slug> --",
            description="Upsample a short Big Buck Bunny excerpt.",
        )
        parser.add_argument(
            "--max-chunks",
            type=int,
            default=self.defaults.max_chunks,
            help="number of source-video chunks to process",
        )
        args = parser.parse_args(list(commandline_args))

        if args.max_chunks <= 0:
            raise ValueError(f"--max-chunks must be > 0, got {args.max_chunks}.")

        first_size = self.defaults.first_chunk_size
        steady_size = self.defaults.steady_chunk_size
        requested_frames = first_size + steady_size * (args.max_chunks - 1)
        video = self._input_loader(requested_frames)
        _validate_loaded_video(video, self._input_spec)
        chunks = _build_chunks(
            total_frames=int(video.frames.shape[0]),
            first_size=first_size,
            steady_size=steady_size,
            max_chunks=args.max_chunks,
        )
        self._config = _ApplicationConfig(
            processor=self.defaults.processor,
            video=video,
            chunks=chunks,
        )

    def session_desc(self) -> SessionDesc:
        """Return the output contract for the fixed demo input."""
        output = self.defaults.processor.output_spec(self._input_spec)
        assert output.fps is not None
        return SessionDesc(
            output_layout=VideoTensorLayout.bcthw,
            backpressure_mode=BackpressureMode.BLOCK,
            presentation_mode=PresentationMode.ON_DEMAND,
            frames_per_second_for_ui=round(output.fps),
            frames_per_second_for_step=round(output.fps),
            video_width=output.width,
            video_height=output.height,
            metadata={
                "application": "upsample-video",
                "model": self.defaults.model_name,
                "input": _BIG_BUCK_BUNNY_FILENAME,
            },
        )

    def create_session(self, session_desc: SessionDesc) -> ISession:
        """Create one finite upsampling session.

        Args:
            session_desc: Runtime-requested output contract.

        Returns:
            Uninitialized upsampling session.

        Raises:
            RuntimeError: :meth:`init` has not run.
            ValueError: The requested layout or frame size differs from the
                model's output contract.
        """
        if self._config is None:
            raise RuntimeError("UpsampleVideoApplication.init() must run first.")
        expected = self.session_desc()
        if session_desc.output_layout is not expected.output_layout:
            raise ValueError(
                "Upsample-video only produces bcthw output, got "
                f"{session_desc.output_layout.value}."
            )
        actual_size = (session_desc.video_width, session_desc.video_height)
        expected_size = (expected.video_width, expected.video_height)
        if actual_size != expected_size:
            raise ValueError(
                "Upsample-video output size must be "
                f"{expected_size[0]}x{expected_size[1]}, got "
                f"{actual_size[0]}x{actual_size[1]}."
            )
        return UpsampleVideoSession(self._config, session_desc)

    def close(self) -> None:
        """Release the decoded source excerpt."""
        self._config = None


class _ClosedPostProcessorSession(VideoPostProcessorSession):
    """Terminal placeholder that retains no processor state."""

    def process(self, chunk: VideoChunk) -> list[VideoChunk]:
        """Reject processing after loop shutdown."""
        del chunk
        raise RuntimeError("Upsample-video model loop is closed.")

    def flush(self) -> list[VideoChunk]:
        """Return no tail after loop shutdown."""
        return []


def _start_processor(config: _ApplicationConfig) -> VideoPostProcessorSession:
    session = config.processor.setup().start(config.video.spec)
    session.prepare()
    return session


def _build_chunks(
    *, total_frames: int, first_size: int, steady_size: int, max_chunks: int
) -> tuple[tuple[int, int], ...]:
    chunks: list[tuple[int, int]] = []
    start = 0
    while start < total_frames and len(chunks) < max_chunks:
        target = first_size if not chunks else steady_size
        size = min(target, total_frames - start)
        chunks.append((start, size))
        start += size
    if not chunks:
        raise ValueError("Big Buck Bunny input contains no decodable frames.")
    return tuple(chunks)


def _validate_loaded_video(video: LoadedVideo, expected: VideoSpec) -> None:
    if video.frames.ndim != 4 or video.frames.shape[1] != 3:
        raise ValueError(
            "Big Buck Bunny frames must have [T, C=3, H, W] shape, got "
            f"{tuple(video.frames.shape)}."
        )
    shape = (int(video.frames.shape[-2]), int(video.frames.shape[-1]))
    expected_shape = (expected.height, expected.width)
    if (
        shape != expected_shape
        or (video.spec.height, video.spec.width) != expected_shape
    ):
        raise ValueError(
            f"Big Buck Bunny input must be {expected.width}x{expected.height}, "
            f"got {shape[1]}x{shape[0]}."
        )
    if video.spec.channels != 3:
        raise ValueError(
            f"Big Buck Bunny input must be RGB, got {video.spec.channels} channels."
        )
    if video.spec.fps is None or expected.fps is None:
        raise ValueError("Big Buck Bunny input must report a frame rate.")
    if round(video.spec.fps) != round(expected.fps):
        raise ValueError(
            f"Big Buck Bunny input must be {expected.fps:g} fps, "
            f"got {video.spec.fps:g}."
        )


def _load_big_buck_bunny(max_frames: int) -> LoadedVideo:
    path = _resolve_big_buck_bunny()
    try:
        import imageio_ffmpeg  # noqa: PLC0415
    except ImportError as error:  # pragma: no cover - dependency gate
        raise ImportError(
            "Decoding the upsample-video input needs imageio-ffmpeg. "
            "Install the flashdreams-upsample-video-v2 package."
        ) from error

    reader = imageio_ffmpeg.read_frames(str(path), pix_fmt="rgb24")
    try:
        metadata = next(reader)
        width, height = metadata["size"]
        frames = []
        for frame_bytes in reader:
            frames.append(
                np.frombuffer(frame_bytes, dtype=np.uint8)
                .reshape(height, width, 3)
                .copy()
            )
            if len(frames) == max_frames:
                break
    finally:
        reader.close()
    if not frames:
        raise ValueError(f"Big Buck Bunny input contains no frames: {path}")

    array = np.stack(frames)
    tensor = torch.from_numpy(array).float().div(127.5).sub(1.0)
    tensor = tensor.permute(0, 3, 1, 2).contiguous()
    return LoadedVideo(
        frames=tensor,
        spec=VideoSpec(
            height=height,
            width=width,
            fps=float(metadata["fps"]),
            channels=3,
        ),
    )


def _resolve_big_buck_bunny() -> Path:
    archive = resolve_input_path(
        _BIG_BUCK_BUNNY_URL,
        cache_dir=_INPUT_CACHE_DIR,
    )
    output = _INPUT_CACHE_DIR / _BIG_BUCK_BUNNY_FILENAME
    if output.is_file() and output.stat().st_size > 0:
        return output

    _INPUT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as bundle:
        matches = [
            member
            for member in bundle.infolist()
            if not member.is_dir()
            and Path(member.filename).name == _BIG_BUCK_BUNNY_FILENAME
        ]
        if len(matches) != 1:
            raise ValueError(
                f"Expected one {_BIG_BUCK_BUNNY_FILENAME!r} in {archive}, "
                f"found {len(matches)}."
            )
        handle, temporary_name = tempfile.mkstemp(
            prefix=f".{_BIG_BUCK_BUNNY_FILENAME}.",
            suffix=".tmp",
            dir=_INPUT_CACHE_DIR,
        )
        temporary = Path(temporary_name)
        try:
            with (
                os.fdopen(handle, "wb") as destination,
                bundle.open(matches[0]) as source,
            ):
                shutil.copyfileobj(source, destination)
            if temporary.stat().st_size == 0:
                raise ValueError(f"Extracted an empty video from {archive}.")
            os.replace(temporary, output)
        finally:
            temporary.unlink(missing_ok=True)
    return output


__all__ = [
    "LoadedVideo",
    "UpsampleVideoApplication",
    "UpsampleVideoApplicationDefaults",
    "UpsampleVideoModelLoop",
    "UpsampleVideoModelState",
    "UpsampleVideoSession",
]
