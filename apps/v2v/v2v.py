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

"""Video-to-video transformation through the FlashDreams v2 application API."""

from __future__ import annotations

import argparse
import math
import os
import shutil
import tempfile
import zipfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
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
"""Public Blender archive used when no input video is selected."""

_BIG_BUCK_BUNNY_SPEC = VideoSpec(height=480, width=853, fps=24.0)
"""Source dimensions and frame rate of the bundled Big Buck Bunny encode."""

_DEFAULT_MAX_CHUNKS = 100
"""Number of Big Buck Bunny chunks processed by default."""

_INPUT_CACHE_DIR = default_flashdreams_cache_dir() / "v2v"
"""User-writable cache for remote inputs and the extracted demo MP4."""

InputLoader = Callable[[str | Path | None, int | None], "LoadedVideo"]


@dataclass(frozen=True, slots=True)
class LoadedVideo:
    """Bounded input video decoded for one application run."""

    frames: Tensor
    """Normalized source frames in ``[T, C, H, W]`` layout."""

    spec: VideoSpec
    """Spatial and timing metadata reported by the source file."""


@dataclass(frozen=True, slots=True)
class V2VApplicationDefaults:
    """Integration-provided defaults for the reusable V2V application."""

    processor: VideoPostProcessorConfig
    """Video post-processor selected by the model integration."""

    first_chunk_size: int
    """Input frames consumed by the cold-start model step."""

    steady_chunk_size: int
    """Input frames consumed by every steady-state model step."""

    model_name: str
    """Model configuration name reported in session metadata."""

    max_chunks: int = _DEFAULT_MAX_CHUNKS
    """Default chunk limit when no input video is selected."""


@dataclass(frozen=True, slots=True)
class _ApplicationConfig:
    """Resolved V2V settings and input shared with one session."""

    processor: VideoPostProcessorConfig
    """Video post-processor settings selected by the application factory."""

    video: LoadedVideo
    """Bounded input video held on CPU."""

    input_name: str
    """Human-readable source name reported in session metadata."""

    chunks: tuple[tuple[int, int], ...]
    """Input frame ranges consumed by consecutive model steps."""


@dataclass(slots=True)
class V2VModelState:
    """Mutable stream state owned by the V2V model loop."""

    config: _ApplicationConfig
    """Resolved application settings and source frames."""

    session_desc: SessionDesc
    """Output contract presented by the runtime."""

    processor_session: VideoPostProcessorSession
    """Stateful video processor for the current rollout."""

    chunks_generated: int = 0
    """Number of input ranges already consumed."""


class V2VModelLoop(IModelLoop[V2VModelState]):
    """Transform one bounded input range per model step."""

    def step(self, step_index: int, events: UserInputEvents) -> list[StepResult]:
        """Transform the next source-video range.

        Args:
            step_index: Zero-based chunk index since the latest reset.
            events: User input ignored by this uninteractive application.

        Returns:
            One transformed video channel in ``bcthw`` layout.

        Raises:
            RuntimeError: Steps arrive out of sequence or the processor buffers
                an expected complete chunk without emitting output.
        """
        del events
        state = self.state
        if step_index != state.chunks_generated:
            raise RuntimeError(
                "V2V step is out of sequence: expected "
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
        """Return whether every bounded input range has been transformed."""
        return self.state.chunks_generated >= len(self.state.config.chunks)

    def reset(self) -> None:
        """Start a fresh processor stream over the same source frames."""
        state = self.state
        state.processor_session = _start_processor(state.config)
        state.chunks_generated = 0

    def close(self) -> None:
        """Release state retained by the processor session."""
        self.state.processor_session = _ClosedPostProcessorSession()


class V2VApplicationSession(ISession):
    """One finite video-to-video run."""

    def __init__(self, config: _ApplicationConfig, session_desc: SessionDesc) -> None:
        self._config = config
        self._session_desc = session_desc
        self._state: V2VModelState | None = None

    def init(self) -> None:
        """Create the stream processor and register the model loop."""
        state = V2VModelState(
            config=self._config,
            session_desc=self._session_desc,
            processor_session=_start_processor(self._config),
        )
        self._state = state
        self.register_model_loop(V2VModelLoop, state=state)

    @property
    def session_desc(self) -> SessionDesc:
        """Return the fixed output shape and timing contract."""
        return self._session_desc

    def close(self) -> None:
        """Release the session-owned source and processor references."""
        self._state = None


class V2VApplication(IApplication):
    """Transform a selected video or a default Big Buck Bunny excerpt."""

    def __init__(
        self,
        *,
        defaults: V2VApplicationDefaults,
        input_loader: InputLoader | None = None,
        input_spec: VideoSpec = _BIG_BUCK_BUNNY_SPEC,
    ) -> None:
        """Create a lazy video-to-video application.

        Args:
            defaults: Model integration defaults for this application.
            input_loader: Test seam replacing input resolution and bounded decode.
            input_spec: Default source-video contract advertised before initialization.
        """
        self.defaults = defaults
        self._input_loader = input_loader or _load_input_video
        self._input_spec = input_spec
        self._config: _ApplicationConfig | None = None

    def init(self, commandline_args: Sequence[str]) -> None:
        """Parse execution settings and decode the bounded source excerpt.

        Args:
            commandline_args: Application-specific command-line arguments.

        Raises:
            ValueError: A setting or the decoded input video is invalid.
        """
        parser = argparse.ArgumentParser(
            prog="flashdreams-run-v2 <v2v-slug> --",
            description="Transform an input video with Big Buck Bunny as the default.",
        )
        parser.add_argument(
            "--video-path",
            default=None,
            help="local video path or HTTP(S) URL; defaults to Big Buck Bunny",
        )
        parser.add_argument(
            "--max-chunks",
            type=int,
            default=None,
            help=(
                "maximum source-video chunks to process; selected videos default "
                f"to all chunks, Big Buck Bunny defaults to {self.defaults.max_chunks}"
            ),
        )
        args = parser.parse_args(list(commandline_args))

        if args.max_chunks is not None and args.max_chunks <= 0:
            raise ValueError(f"--max-chunks must be > 0, got {args.max_chunks}.")

        max_chunks = (
            self.defaults.max_chunks
            if args.video_path is None and args.max_chunks is None
            else args.max_chunks
        )
        first_size = self.defaults.first_chunk_size
        steady_size = self.defaults.steady_chunk_size
        requested_frames = (
            None if max_chunks is None else first_size + steady_size * (max_chunks - 1)
        )
        video = self._input_loader(args.video_path, requested_frames)
        _validate_loaded_video(video)
        chunks = _build_chunks(
            total_frames=int(video.frames.shape[0]),
            first_size=first_size,
            steady_size=steady_size,
            max_chunks=max_chunks,
        )
        self._config = _ApplicationConfig(
            processor=self.defaults.processor,
            video=video,
            input_name=(
                _BIG_BUCK_BUNNY_FILENAME
                if args.video_path is None
                else Path(args.video_path).name
            ),
            chunks=chunks,
        )

    def session_desc(self) -> SessionDesc:
        """Return the output contract for the selected or default input."""
        input_spec = (
            self._input_spec if self._config is None else self._config.video.spec
        )
        output = self.defaults.processor.output_spec(input_spec)
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
                "application": "v2v",
                "model": self.defaults.model_name,
                "input": (
                    _BIG_BUCK_BUNNY_FILENAME
                    if self._config is None
                    else self._config.input_name
                ),
            },
        )

    def create_session(self, session_desc: SessionDesc) -> ISession:
        """Create one finite video-to-video session.

        Args:
            session_desc: Runtime-requested output contract.

        Returns:
            Uninitialized video-to-video session.

        Raises:
            RuntimeError: :meth:`init` has not run.
            ValueError: The requested layout or frame size differs from the
                model's output contract.
        """
        if self._config is None:
            raise RuntimeError("V2VApplication.init() must run first.")
        expected = self.session_desc()
        if session_desc.output_layout is not expected.output_layout:
            raise ValueError(
                "V2V only produces bcthw output, got "
                f"{session_desc.output_layout.value}."
            )
        actual_size = (session_desc.video_width, session_desc.video_height)
        expected_size = (expected.video_width, expected.video_height)
        default_output = self.defaults.processor.output_spec(self._input_spec)
        default_size = (default_output.width, default_output.height)
        if actual_size not in (expected_size, default_size):
            raise ValueError(
                "V2V output size must be "
                f"{expected_size[0]}x{expected_size[1]}, got "
                f"{actual_size[0]}x{actual_size[1]}."
            )
        resolved = replace(
            session_desc,
            output_layout=expected.output_layout,
            frames_per_second_for_ui=expected.frames_per_second_for_ui,
            frames_per_second_for_step=expected.frames_per_second_for_step,
            video_width=expected.video_width,
            video_height=expected.video_height,
            metadata=expected.metadata,
        )
        return V2VApplicationSession(self._config, resolved)

    def close(self) -> None:
        """Release the decoded input video."""
        self._config = None


class _ClosedPostProcessorSession(VideoPostProcessorSession):
    """Terminal placeholder that retains no processor state."""

    def process(self, chunk: VideoChunk) -> list[VideoChunk]:
        """Reject processing after loop shutdown."""
        del chunk
        raise RuntimeError("V2V model loop is closed.")

    def flush(self) -> list[VideoChunk]:
        """Return no tail after loop shutdown."""
        return []


def _start_processor(config: _ApplicationConfig) -> VideoPostProcessorSession:
    session = config.processor.setup().start(config.video.spec)
    session.prepare()
    return session


def _build_chunks(
    *, total_frames: int, first_size: int, steady_size: int, max_chunks: int | None
) -> tuple[tuple[int, int], ...]:
    chunks: list[tuple[int, int]] = []
    start = 0
    while start < total_frames and (max_chunks is None or len(chunks) < max_chunks):
        target = first_size if not chunks else steady_size
        size = min(target, total_frames - start)
        chunks.append((start, size))
        start += size
    if not chunks:
        raise ValueError("Input video contains no decodable frames.")
    return tuple(chunks)


def _validate_loaded_video(video: LoadedVideo) -> None:
    if video.frames.ndim != 4 or video.frames.shape[1] != 3:
        raise ValueError(
            "Input video frames must have [T, C=3, H, W] shape, got "
            f"{tuple(video.frames.shape)}."
        )
    shape = (int(video.frames.shape[-2]), int(video.frames.shape[-1]))
    spec_shape = (video.spec.height, video.spec.width)
    if shape != spec_shape:
        raise ValueError(
            "Input video frame dimensions do not match its metadata: "
            f"frames are {shape[1]}x{shape[0]}, metadata is "
            f"{video.spec.width}x{video.spec.height}."
        )
    if video.spec.channels != 3:
        raise ValueError(
            f"Input video must be RGB, got {video.spec.channels} channels."
        )
    if (
        video.spec.fps is None
        or not math.isfinite(video.spec.fps)
        or video.spec.fps <= 0
    ):
        raise ValueError("Input video must report a positive finite frame rate.")


def _load_input_video(value: str | Path | None, max_frames: int | None) -> LoadedVideo:
    path = (
        _resolve_big_buck_bunny()
        if value is None
        else resolve_input_path(value, cache_dir=_INPUT_CACHE_DIR)
    )
    try:
        import imageio_ffmpeg  # noqa: PLC0415
    except ImportError as error:  # pragma: no cover - dependency gate
        raise ImportError(
            "Decoding the v2v input needs imageio-ffmpeg. "
            "Install the flashdreams-v2v package."
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
            if max_frames is not None and len(frames) == max_frames:
                break
    finally:
        reader.close()
    if not frames:
        raise ValueError(f"Input video contains no frames: {path}")

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
    "V2VApplication",
    "V2VApplicationDefaults",
    "V2VModelLoop",
    "V2VModelState",
    "V2VApplicationSession",
]
