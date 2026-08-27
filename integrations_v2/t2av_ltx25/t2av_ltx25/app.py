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

"""LTX 2.5 one-shot synchronized audio-video application for runtime V2."""

import argparse
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch

from flashdreams.api_v2.application import IApplication
from flashdreams.api_v2.loop import IModelLoop
from flashdreams.api_v2.session import ISession
from flashdreams.runtime_v2.audio_output import AudioOutput
from flashdreams.runtime_v2.session_desc import (
    BackpressureMode,
    PresentationMode,
    SessionDesc,
)
from flashdreams.runtime_v2.step_result import StepResult
from flashdreams.runtime_v2.user_input_events import UserInputEvents
from flashdreams.runtime_v2.video_tensor import VideoTensorLayout

from .backend import (
    DEFAULT_AUDIO_CHANNELS,
    DEFAULT_AUDIO_SAMPLE_RATE,
    BackendLoadConfig,
    GenerationRequest,
    LTX25Backend,
    load_diffusers_backend,
)

_DEFAULT_WIDTH = 768
_DEFAULT_HEIGHT = 512
_DEFAULT_FRAMES_PER_SECOND = 24
_DEFAULT_UI_FRAMES_PER_SECOND = 60
_DEFAULT_NUM_FRAMES = 121
_DEFAULT_SEED = 42
_MAX_NUM_FRAMES = 241
_SPATIAL_ALIGNMENT = 32

BackendLoader = Callable[[BackendLoadConfig], LTX25Backend]


@dataclass(frozen=True, kw_only=True, slots=True)
class LTX25Config:
    """Validated application arguments shared by every session."""

    prompt: str
    """Text prompt conditioning the joint output."""

    seed: int
    """Non-negative random seed."""

    num_frames: int
    """One-shot frame count on LTX's temporal grid."""

    backend: BackendLoadConfig
    """Model construction and residency settings."""


@dataclass(slots=True)
class LTX25ModelState:
    """Mutable state owned by one model loop."""

    backend: LTX25Backend
    """Application-scoped model backend."""

    config: LTX25Config
    """Validated generation arguments."""

    session_desc: SessionDesc
    """Runtime media contract for this session."""

    model_load_s: float
    """One-time backend load duration, reported with the first result."""

    generated: bool = False
    """Whether this loop has emitted its one joint result."""


class LTX25ModelLoop(IModelLoop[LTX25ModelState]):
    """Generate the complete clip once and let runtime V2 present and encode it."""

    def step(self, step_index: int, events: UserInputEvents) -> list[StepResult]:
        """Generate one synchronized tensor pair.

        Args:
            step_index: Zero-based generation index supplied by the runtime.
            events: Input events not seen by this loop before; LTX is noninteractive.

        Returns:
            A single video channel carrying timeline-aligned PCM.

        Raises:
            RuntimeError: The backend is called twice without a reset or emits media
                that does not match the runtime contract.
        """
        del events
        if self.state.generated:
            raise RuntimeError("LTX 2.5 generation has already completed.")

        desc = self.state.session_desc
        started_at = time.perf_counter()
        media = self.state.backend.generate(
            GenerationRequest(
                prompt=self.state.config.prompt,
                seed=self.state.config.seed,
                num_frames=self.state.config.num_frames,
                width=desc.video_width,
                height=desc.video_height,
                frame_rate=desc.frames_per_second_for_step,
            )
        )
        generation_s = time.perf_counter() - started_at
        _validate_generated_media(media.video, media.audio, self.state)
        self.state.generated = True

        metrics = dict(media.metrics)
        metrics.update(
            {
                "model_load_s": self.state.model_load_s,
                "generation_s": generation_s,
                "generation_fps": self.state.config.num_frames / generation_s,
            }
        )
        return [
            StepResult(
                step_index=step_index,
                output=media.video,
                frame_count=self.state.config.num_frames,
                output_layout=desc.output_layout,
                metrics=metrics,
                audio=AudioOutput(
                    samples=media.audio,
                    sample_rate=DEFAULT_AUDIO_SAMPLE_RATE,
                    sample_offset=0,
                ),
            )
        ]

    def is_finished(self) -> bool:
        """Return whether this one-shot loop has emitted its result."""
        return self.state.generated

    def reset(self) -> None:
        """Make the same seeded request runnable again."""
        self.state.generated = False


class LTX25Session(ISession):
    """One isolated LTX generation sharing the application-scoped backend."""

    def __init__(
        self,
        *,
        backend: LTX25Backend,
        config: LTX25Config,
        session_desc: SessionDesc,
        model_load_s: float,
    ) -> None:
        """
        Args:
            backend: Loaded LTX model owned by the application.
            config: Validated generation settings.
            session_desc: Runtime media contract already validated by the app.
            model_load_s: One-time backend load duration for metrics.
        """
        self._backend = backend
        self._config = config
        self._session_desc = session_desc
        self._model_load_s = model_load_s

    def init(self) -> None:
        """Register a fresh one-shot model loop; the default UI blits its frames."""
        self.register_model_loop(
            LTX25ModelLoop,
            state=LTX25ModelState(
                backend=self._backend,
                config=self._config,
                session_desc=self._session_desc,
                model_load_s=self._model_load_s,
            ),
        )

    @property
    def session_desc(self) -> SessionDesc:
        """Return the exact runtime contract this session honours."""
        return self._session_desc


class LTX25Application(IApplication):
    """Load LTX 2.5 once and create one-shot synchronized V2 sessions."""

    def __init__(self, backend_loader: BackendLoader = load_diffusers_backend) -> None:
        """
        Args:
            backend_loader: Model loader seam; tests supply a deterministic stand-in.
        """
        self._backend_loader = backend_loader
        self._config: LTX25Config | None = None
        self._backend: LTX25Backend | None = None
        self._model_load_s = 0.0

    def session_desc(self) -> SessionDesc:
        """Describe the natural LTX 2.5 media contract without loading the model."""
        return SessionDesc(
            output_layout=VideoTensorLayout.tchw,
            backpressure_mode=BackpressureMode.BLOCK,
            presentation_mode=PresentationMode.ONLY_PRESENT_NEW,
            frames_per_second_for_ui=_DEFAULT_UI_FRAMES_PER_SECOND,
            frames_per_second_for_step=_DEFAULT_FRAMES_PER_SECOND,
            video_width=_DEFAULT_WIDTH,
            video_height=_DEFAULT_HEIGHT,
            audio_sample_rate=DEFAULT_AUDIO_SAMPLE_RATE,
            audio_channels=DEFAULT_AUDIO_CHANNELS,
        )

    def init(self, commandline_args: Sequence[str]) -> None:
        """Parse and validate generation arguments without loading weights.

        Args:
            commandline_args: Application-specific arguments after the CLI separator.

        Raises:
            ValueError: A seed or frame count is outside the supported domain.
        """
        parser = argparse.ArgumentParser(
            prog="t2av-ltx25",
            description="Generate synchronized audio and video with LTX 2.5.",
        )
        parser.add_argument("--prompt", required=True)
        parser.add_argument("--num-frames", type=int, default=_DEFAULT_NUM_FRAMES)
        parser.add_argument("--seed", type=int, default=_DEFAULT_SEED)
        parser.add_argument("--device", default="cuda")
        parser.add_argument(
            "--offload",
            choices=("model", "sequential", "none"),
            default="model",
        )
        parser.add_argument("--local-files-only", action="store_true")
        args = parser.parse_args(list(commandline_args))

        prompt = args.prompt.strip()
        if not prompt:
            raise ValueError("--prompt must contain non-whitespace text.")
        if args.seed < 0:
            raise ValueError(f"--seed must be >= 0, got {args.seed}.")
        if not 1 <= args.num_frames <= _MAX_NUM_FRAMES:
            raise ValueError(
                f"--num-frames must be from 1 through {_MAX_NUM_FRAMES}, "
                f"got {args.num_frames}."
            )
        if (args.num_frames - 1) % 8:
            raise ValueError(
                f"--num-frames must have the form 8k + 1, got {args.num_frames}."
            )
        device = args.device.strip()
        if not device:
            raise ValueError("--device must contain non-whitespace text.")

        self._config = LTX25Config(
            prompt=prompt,
            seed=args.seed,
            num_frames=args.num_frames,
            backend=BackendLoadConfig(
                device=device,
                offload=args.offload,
                local_files_only=args.local_files_only,
            ),
        )

    def create_session(self, session_desc: SessionDesc) -> ISession:
        """Validate the media contract, then lazily load and share the model.

        Args:
            session_desc: Runtime media contract to honour.

        Returns:
            A new session with isolated loop state.

        Raises:
            RuntimeError: init has not run or the loaded backend disagrees with
                the declared audio format.
            ValueError: The requested runtime contract cannot preserve LTX output.
        """
        config = self._config
        if config is None:
            raise RuntimeError(
                "LTX25Application.init() must run before create_session()."
            )
        _validate_session_desc(session_desc)
        if self._backend is None:
            started_at = time.perf_counter()
            backend = self._backend_loader(config.backend)
            self._model_load_s = time.perf_counter() - started_at
            if (
                backend.sample_rate != DEFAULT_AUDIO_SAMPLE_RATE
                or backend.audio_channels != DEFAULT_AUDIO_CHANNELS
            ):
                sample_rate = backend.sample_rate
                audio_channels = backend.audio_channels
                backend.close()
                raise RuntimeError(
                    "LTX backend audio contract must be "
                    f"{DEFAULT_AUDIO_SAMPLE_RATE} Hz stereo, got "
                    f"{sample_rate} Hz/{audio_channels} channels."
                )
            self._backend = backend
        return LTX25Session(
            backend=self._backend,
            config=config,
            session_desc=session_desc,
            model_load_s=self._model_load_s,
        )

    def close(self) -> None:
        """Release the application-scoped backend exactly once."""
        backend = self._backend
        self._backend = None
        if backend is not None:
            backend.close()


def _validate_session_desc(desc: SessionDesc) -> None:
    """Reject runtime contracts that cannot preserve synchronized LTX output."""
    if desc.output_layout is not VideoTensorLayout.tchw:
        raise ValueError(
            f"LTX 2.5 produces tchw video, got {desc.output_layout.value}."
        )
    if desc.video_width % _SPATIAL_ALIGNMENT or desc.video_height % _SPATIAL_ALIGNMENT:
        raise ValueError(
            "LTX 2.5 width and height must be divisible by "
            f"{_SPATIAL_ALIGNMENT}, got {desc.video_width}x{desc.video_height}."
        )
    if (
        desc.audio_sample_rate != DEFAULT_AUDIO_SAMPLE_RATE
        or desc.audio_channels != DEFAULT_AUDIO_CHANNELS
    ):
        raise ValueError(
            "LTX 2.5 requires a 48000 Hz stereo SessionDesc, got "
            f"{desc.audio_sample_rate} Hz/{desc.audio_channels} channels."
        )
    if desc.backpressure_mode is not BackpressureMode.BLOCK:
        raise ValueError("Synchronized LTX output requires blocking backpressure.")
    if desc.presentation_mode is not PresentationMode.ONLY_PRESENT_NEW:
        raise ValueError("Synchronized LTX output requires presenting only new frames.")


def _validate_generated_media(
    video: torch.Tensor,
    audio: torch.Tensor,
    state: LTX25ModelState,
) -> None:
    """Reject backend tensors that disagree with the validated session contract."""
    desc = state.session_desc
    expected_video = (
        state.config.num_frames,
        3,
        desc.video_height,
        desc.video_width,
    )
    if video.dtype is not torch.uint8 or tuple(video.shape) != expected_video:
        raise RuntimeError(
            "LTX backend video must be contiguous uint8 tchw with shape "
            f"{expected_video}, got {tuple(video.shape)} and {video.dtype}."
        )
    if not video.is_contiguous():
        raise RuntimeError("LTX backend video must be contiguous.")
    if audio.ndim != 2 or not audio.is_floating_point() or audio.shape[0] != 2:
        raise RuntimeError(
            "LTX backend audio must be floating-point stereo [2, samples], got "
            f"{tuple(audio.shape)} and {audio.dtype}."
        )
    if audio.shape[1] <= 0:
        raise RuntimeError(
            "LTX backend audio must have shape [2, positive samples], got "
            f"{tuple(audio.shape)}."
        )


def create_app() -> IApplication:
    """Return a new LTX 2.5 application for entry-point discovery."""
    return LTX25Application()


__all__ = ["LTX25Application", "LTX25Session", "create_app"]
