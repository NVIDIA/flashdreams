# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One-shot Qwen Image Edit application for FlashDreams V2."""

from __future__ import annotations

import argparse
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import torch
from PIL import Image

from flashdreams.api_v2.application import IApplication
from flashdreams.api_v2.loop import IModelLoop
from flashdreams.api_v2.session import ISession
from flashdreams.runtime_v2.application_runner import ApplicationRunner
from flashdreams.runtime_v2.client_window_factory import (
    add_client_window_arguments,
    client_window_mode,
    create_client_window,
)
from flashdreams.runtime_v2.session_desc import (
    BackpressureMode,
    PresentationMode,
    SessionDesc,
)
from flashdreams.runtime_v2.step_result import StepResult
from flashdreams.runtime_v2.user_input_events import UserInputEvents
from flashdreams.runtime_v2.video_tensor import VideoTensorLayout

from .editor import QwenImageEditor


class ImageEditor(Protocol):
    """Generate an edited image from one source image and instruction."""

    def generate(
        self,
        image: Image.Image | str | Path,
        prompt: str,
        *,
        output_size: tuple[int, int],
        seed: int = 0,
        num_inference_steps: int | None = None,
        negative_prompt: str | None = None,
        true_cfg_scale: float | None = None,
    ) -> Image.Image: ...


EditorFactory = Callable[[torch.device], ImageEditor]


@dataclass(frozen=True, slots=True)
class _Config:
    input_path: Path
    prompt: str
    seed: int
    steps: int | None
    device: torch.device
    negative_prompt: str
    true_cfg_scale: float


@dataclass(slots=True)
class _ModelState:
    config: _Config
    editor: ImageEditor
    session_desc: SessionDesc
    generated: bool = False


class QwenImageEditModelLoop(IModelLoop[_ModelState]):
    """Generate exactly one edited image."""

    def step(self, step_index: int, events: UserInputEvents) -> list[StepResult]:
        """Run Qwen Image Edit and return its image as one TCHW frame."""
        del events
        state = self.state
        image = state.editor.generate(
            state.config.input_path,
            state.config.prompt,
            output_size=(
                state.session_desc.video_width,
                state.session_desc.video_height,
            ),
            seed=state.config.seed,
            num_inference_steps=state.config.steps,
            negative_prompt=state.config.negative_prompt,
            true_cfg_scale=state.config.true_cfg_scale,
        )
        array = np.asarray(image, dtype=np.uint8).copy()
        output = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
        state.generated = True
        return [
            StepResult(
                step_index=step_index,
                output=output,
                frame_count=1,
                output_layout=VideoTensorLayout.tchw,
            )
        ]

    def is_finished(self) -> bool:
        """Finish after the one requested image."""
        return self.state.generated

    def reset(self) -> None:
        """Allow one new image after a session reset."""
        self.state.generated = False


class QwenImageEditSession(ISession):
    """One image-edit request using an application-owned editor."""

    def __init__(
        self, config: _Config, editor: ImageEditor, session_desc: SessionDesc
    ) -> None:
        self._config = config
        self._editor = editor
        self._session_desc = session_desc

    def init(self) -> None:
        """Register the one-shot model loop."""
        self.register_model_loop(
            QwenImageEditModelLoop,
            state=_ModelState(self._config, self._editor, self._session_desc),
        )

    @property
    def session_desc(self) -> SessionDesc:
        """Return the requested image contract."""
        return self._session_desc


class QwenImageEditApplication(IApplication):
    """Parse image-edit requests and share one lazily loaded editor."""

    def __init__(self, editor_factory: EditorFactory | None = None) -> None:
        self._editor_factory = editor_factory or (
            lambda device: QwenImageEditor(device=device)
        )
        self._config: _Config | None = None
        self._editor: ImageEditor | None = None

    def init(self, commandline_args: Sequence[str]) -> None:
        """Parse the input image, prompt, seed, steps, and device."""
        parser = argparse.ArgumentParser(prog="qwen-image-edit-v2")
        parser.add_argument("--input", type=Path, required=True)
        parser.add_argument("--prompt", required=True)
        parser.add_argument("--seed", type=int, default=0)
        parser.add_argument("--steps", type=int)
        parser.add_argument("--device", default="cuda")
        parser.add_argument("--negative-prompt", default=" ")
        parser.add_argument("--true-cfg-scale", type=float, default=4.0)
        args = parser.parse_args(list(commandline_args))
        if not args.input.is_file():
            raise FileNotFoundError(args.input)
        if args.seed < 0:
            raise ValueError("--seed must be non-negative")
        if args.steps is not None and args.steps <= 0:
            raise ValueError("--steps must be positive")
        if not math.isfinite(args.true_cfg_scale) or args.true_cfg_scale <= 1.0:
            raise ValueError("--true-cfg-scale must be finite and greater than 1")
        self._config = _Config(
            args.input,
            args.prompt,
            args.seed,
            args.steps,
            torch.device(args.device),
            args.negative_prompt,
            args.true_cfg_scale,
        )

    def create_session(self, session_desc: SessionDesc) -> ISession:
        """Create one image-edit session."""
        if self._config is None:
            raise RuntimeError("QwenImageEditApplication.init() must run first")
        if session_desc.output_layout is not VideoTensorLayout.tchw:
            raise ValueError("Qwen Image Edit requires tchw output")
        if session_desc.video_width % 16 or session_desc.video_height % 16:
            raise ValueError("Qwen Image Edit dimensions must be divisible by 16")
        if self._editor is None:
            self._editor = self._editor_factory(self._config.device)
        return QwenImageEditSession(self._config, self._editor, session_desc)

    def close(self) -> None:
        """Release application-owned model components."""
        self._editor = None


def create_app() -> IApplication:
    """Return a new Qwen Image Edit application."""
    return QwenImageEditApplication()


def main(commandline_args: Sequence[str] | None = None) -> int:
    """Run one image edit through the V2 runtime."""
    parser = argparse.ArgumentParser(prog="qwen-image-edit-v2")
    add_client_window_arguments(parser)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=704)
    parser.add_argument("application_args", nargs=argparse.REMAINDER)
    args = parser.parse_args(commandline_args)
    try:
        client_window_mode(args.mode).check_arguments(args)
    except ValueError as error:
        parser.error(str(error))
    application_args = list(args.application_args)
    if application_args[:1] == ["--"]:
        application_args.pop(0)
    window = create_client_window(args)
    message = client_window_mode(args.mode).starting(window)
    if message:
        print(message, flush=True)
    ApplicationRunner(create_app(), window).run(
        SessionDesc(
            output_layout=VideoTensorLayout.tchw,
            backpressure_mode=BackpressureMode.BLOCK,
            presentation_mode=PresentationMode.ON_DEMAND,
            frames_per_second_for_ui=1,
            frames_per_second_for_step=1,
            video_width=args.width,
            video_height=args.height,
        ),
        application_args,
    )
    message = client_window_mode(args.mode).finished(window)
    if message:
        print(message, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
