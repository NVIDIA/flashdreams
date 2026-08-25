# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reusable action-conditioned video application for the v2 runtime."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from flashdreams.api_v2.application import IApplication
from flashdreams.api_v2.loop import IModelLoop, IUILoop, invoke_async
from flashdreams.api_v2.session import ISession
from flashdreams.infra.config import derive_config
from flashdreams.runtime_v2.session_desc import SessionDesc
from flashdreams.runtime_v2.slangpy_ui_loop import SlangPyUILoop
from flashdreams.runtime_v2.step_result import StepResult
from flashdreams.runtime_v2.user_input_event import (
    GamepadUserInputEventData,
    KeyboardInputState,
    KeyboardUserInputEventData,
)
from flashdreams.runtime_v2.user_input_events import UserInputEvents
from flashdreams.runtime_v2.video_tensor import VideoTensorLayout

_DEFAULT_BLOCKS = 20
_INTRINSICS_REFERENCE_HEIGHT = 480
_INTRINSICS_REFERENCE_WIDTH = 832


@dataclass(frozen=True, slots=True)
class Action2VApplicationDefaults:
    """Default values supplied by a camera-control model integration."""

    title: str
    """Window title presented by the application."""

    slug: str
    """Registered application slug used in CLI help."""

    preset_id: str
    """Pipeline preset selected when ``--preset-id`` is omitted."""

    prompt: str
    """Fallback prompt used when no prompt asset or override is available."""

    frames_per_second: int
    """Generation frame rate."""

    pixel_width: int
    """Default output width in pixels."""

    pixel_height: int
    """Default output height in pixels."""

    total_blocks: int = _DEFAULT_BLOCKS
    """Default number of autoregressive blocks."""


@dataclass(frozen=True, slots=True)
class Action2VApplicationHooks:
    """Model-owned callables consumed by the camera-control application."""

    pipeline_configs: Mapping[str, Any]
    """Available model pipeline configs keyed by preset ID."""

    image_loader: Callable[..., Tensor]
    """Load and preprocess the first frame."""

    action_loader: Callable[..., Any]
    """Load the integration-specific action trace for a rollout."""

    example_loader: Callable[..., Action2VInputPaths]
    """Resolve or download a complete set of example inputs."""

    keyboard_factory: Callable[[int], Any]
    """Create the input resampler for a generation frame rate."""

    camera_factory: Callable[[], Any]
    """Create the camera-pose integrator."""

    control_factory: Callable[..., Any]
    """Build one model-specific camera-control input."""

    rollout_initializer: Callable[..., None] | None = None
    """Bind optional model-owned rollout state after the pipeline is loaded."""

    preset_prompts: Mapping[str, str] = field(default_factory=dict)
    """Optional prompt defaults keyed by preset ID."""


@dataclass(frozen=True, slots=True)
class Action2VInputPaths:
    """Input assets required to initialize an action-conditioned rollout."""

    image_path: Path
    """First-frame image."""

    action_path: Path
    """Action or camera-trajectory input."""

    calibration_path: Path | None = None
    """Optional camera calibration input."""

    prompt_path: Path | None = None
    """Optional text file containing the rollout prompt."""


@dataclass(frozen=True, slots=True)
class Action2VConfig:
    prompt: str
    image_path: Path
    action_path: Path
    calibration_path: Path | None
    total_blocks: int
    device: str


@dataclass(slots=True)
class Action2VModelState:
    pipeline_factory: Callable[[], Any]
    config: Action2VConfig
    desc: SessionDesc
    hooks: Action2VApplicationHooks
    cache: Any = None
    trace: Any = None
    blocks_generated: int = 0
    prompt: str = ""
    keyboard: Any = None
    camera: Any = None
    ui_loop: IUILoop[Any] | None = None
    controller_keys: frozenset[str] = frozenset()
    pipeline: Any = None

    def request_prompt(self, prompt: str) -> None:
        prompt = prompt.strip()
        if not prompt:
            raise ValueError("prompt must be non-empty.")
        self.prompt = prompt
        self.cache = None
        self.blocks_generated = 0
        self._notify("Prompt accepted; restarting the rollout.")

    def _notify(self, status: str) -> None:
        if self.ui_loop is not None:
            invoke_async(
                self.ui_loop,
                lambda state, status=status: state.set_status(status),
            )


class Action2VModelLoop(IModelLoop[Action2VModelState]):
    """Own action-to-video cache, input integration, and generation."""

    def step(self, step_index: int, events: UserInputEvents) -> list[StepResult]:
        state = self.state
        if state.pipeline is None:
            state._notify("Loading the action-to-video pipeline...")
            state.pipeline = state.pipeline_factory()
        if state.cache is None:
            self._initialize_rollout()
        self._apply_controls(events)
        assert state.trace is not None
        assert state.keyboard is not None
        assert state.camera is not None
        ar_idx = state.blocks_generated
        num_frames = int(state.pipeline.get_num_output_frames(ar_idx))
        segments, frame_times = state.keyboard.sample_chunk(num_frames)
        poses = torch.from_numpy(
            state.camera.integrate_chunk(segments=segments, frame_times=frame_times)
        )
        intrinsics = state.trace.intrinsics[:1].expand(num_frames, -1).clone()
        control = state.hooks.control_factory(
            intrinsics=intrinsics.to(device=state.pipeline.device, dtype=torch.float32),
            poses=poses.to(device=state.pipeline.device, dtype=torch.float32),
            world_scale=state.trace.world_scale,
        )
        state._notify(f"Generating camera block {ar_idx + 1}/{state.config.total_blocks}...")
        frames = state.pipeline.generate(
            autoregressive_index=ar_idx,
            cache=state.cache,
            input=control,
        )
        metrics = state.pipeline.finalize(
            autoregressive_index=ar_idx,
            cache=state.cache,
        )
        state.blocks_generated += 1
        state._notify(
            "Generation complete."
            if self.is_finished()
            else f"Generated block {state.blocks_generated}/{state.config.total_blocks}."
        )
        return [
            StepResult(
                step_index=step_index,
                output=frames.detach(),
                frame_count=int(frames.shape[0]),
                output_layout=state.desc.output_layout,
                metrics=dict(metrics or {}),
            )
        ]

    def is_finished(self) -> bool:
        return self.state.blocks_generated >= self.state.config.total_blocks

    def reset(self) -> None:
        self.state.cache = None
        self.state.trace = None
        self.state.blocks_generated = 0
        self.state.keyboard = None
        self.state.camera = None
        self.state._notify("Rollout reset.")

    def close(self) -> None:
        self.state.cache = None
        self.state.trace = None

    def _initialize_rollout(self) -> None:
        state = self.state
        state._notify("Loading the first frame and camera calibration...")
        trace = state.hooks.action_loader(
            action_path=state.config.action_path,
            calibration_path=state.config.calibration_path,
            total_blocks=state.config.total_blocks,
            pixel_height=state.desc.video_height,
            pixel_width=state.desc.video_width,
            intrinsics_reference_height=_INTRINSICS_REFERENCE_HEIGHT,
            intrinsics_reference_width=_INTRINSICS_REFERENCE_WIDTH,
        )
        first_frame = state.hooks.image_loader(
            state.config.image_path,
            pixel_height=state.desc.video_height,
            pixel_width=state.desc.video_width,
            device=torch.device(state.pipeline.device),
            dtype=torch.bfloat16,
            interpolation="cubic",
        )
        state.trace = trace
        if state.hooks.rollout_initializer is not None:
            state.hooks.rollout_initializer(
                pipeline=state.pipeline,
                action_path=state.config.action_path,
                calibration_path=state.config.calibration_path,
                total_blocks=state.config.total_blocks,
                trace=trace,
            )
        state.cache = state.pipeline.initialize_cache(
            text=[state.prompt],
            image=first_frame,
        )
        state.keyboard = state.hooks.keyboard_factory(
            state.desc.frames_per_second_for_step
        )
        state.camera = state.hooks.camera_factory()
        state.camera.reset(trace.poses[0].cpu().numpy())

    def _apply_controls(self, events: UserInputEvents) -> None:
        keyboard = self.state.keyboard
        if keyboard is None:
            return
        for event in events.get_events():
            data = event.get_event_data()
            arrival = float(event.get_timestamp()) / 1_000_000.0
            if isinstance(data, KeyboardUserInputEventData):
                key = _drive_key(data.key)
                if key is not None:
                    keyboard.on_edge(
                        arrival_t=arrival,
                        event=(
                            "key_down"
                            if data.state is KeyboardInputState.PRESSED
                            else "key_up"
                        ),
                        key=key,
                    )
            elif isinstance(data, GamepadUserInputEventData) and data.action == "state":
                keys = _gamepad_keys(data)
                for key in self.state.controller_keys - keys:
                    keyboard.on_edge(arrival_t=arrival, event="key_up", key=key)
                for key in keys - self.state.controller_keys:
                    keyboard.on_edge(arrival_t=arrival, event="key_down", key=key)
                self.state.controller_keys = keys


@dataclass(slots=True)
class Action2VUIState:
    model_loop: IModelLoop[Action2VModelState]
    title: str
    prompt: str
    status: str = "W/S move, A/D yaw, Q/E strafe, I/K pitch."
    prompt_widget: Any | None = field(default=None, init=False, repr=False)
    status_widget: Any | None = field(default=None, init=False, repr=False)

    def set_status(self, status: str) -> None:
        self.status = status
        if self.status_widget is not None:
            self.status_widget.text = status


class Action2VUILoop(SlangPyUILoop[Action2VUIState]):
    """Own the action-to-video widgets and composite model output."""

    def step_ui(
        self, ui: Any, step_index: int, events: UserInputEvents
    ) -> Tensor | None:
        del step_index, events
        if self.state.prompt_widget is None:
            window = ui.Window(
                ui.screen, self.state.title, position=(16, 16), size=(560, 230)
            )
            self.state.prompt_widget = ui.InputText(
                window, "Prompt", self.state.prompt, self._set_prompt, multi_line=True
            )
            ui.Button(window, "Restart with prompt", self._submit_prompt)
            self.state.status_widget = ui.Text(window, self.state.status)
        return self.presented_model_frame()

    def _set_prompt(self, prompt: str) -> None:
        self.state.prompt = prompt

    def _submit_prompt(self) -> None:
        prompt = self.state.prompt.strip()
        if not prompt:
            self.state.set_status("Enter a prompt before restarting.")
            return
        self.state.set_status("Prompt update queued.")
        invoke_async(
            self.state.model_loop,
            lambda state, prompt=prompt: state.request_prompt(prompt),
        )


class Action2VSession(ISession):
    def __init__(
        self,
        *,
        pipeline_factory: Callable[[], Any],
        config: Action2VConfig,
        desc: SessionDesc,
        hooks: Action2VApplicationHooks,
        title: str,
        ui_renderer: Any | None,
    ) -> None:
        self._pipeline_factory = pipeline_factory
        self._config = config
        self._desc = desc
        self._hooks = hooks
        self._title = title
        self._ui_renderer = ui_renderer

    @property
    def session_desc(self) -> SessionDesc:
        return self._desc

    def init(self) -> None:
        state = Action2VModelState(
            pipeline_factory=self._pipeline_factory,
            config=self._config,
            desc=self._desc,
            hooks=self._hooks,
            prompt=self._config.prompt,
        )
        model_loop = self.register_model_loop(Action2VModelLoop, state=state)
        kwargs = (
            {"renderer": self._ui_renderer}
            if self._ui_renderer is not None
            else {"width": self._desc.video_width, "height": self._desc.video_height}
        )
        ui_loop = self.register_ui_loop(
            Action2VUILoop,
            state=Action2VUIState(
                model_loop=model_loop,
                title=self._title,
                prompt=self._config.prompt,
            ),
            **kwargs,
        )
        state.ui_loop = ui_loop


class Action2VApplication(IApplication):
    """Load one configured model and create action-to-video sessions."""

    def __init__(
        self,
        *,
        defaults: Action2VApplicationDefaults,
        hooks: Action2VApplicationHooks,
        pipeline_config: Any | None = None,
        ui_renderer_factory: Callable[[int, int], Any] | None = None,
    ) -> None:
        self._defaults = defaults
        self._hooks = hooks
        self._pipeline_config = pipeline_config
        self._ui_renderer_factory = ui_renderer_factory
        self._config: Action2VConfig | None = None
        self._pipeline: Any = None
        self._desc = SessionDesc(
            output_layout=VideoTensorLayout.tchw,
            frames_per_second_for_ui=60,
            frames_per_second_for_step=defaults.frames_per_second,
            video_width=defaults.pixel_width,
            video_height=defaults.pixel_height,
        )

    def init(self, commandline_args: Sequence[str]) -> None:
        parser = argparse.ArgumentParser(
            prog=f"flashdreams-run-v2 {self._defaults.slug} --"
        )
        parser.add_argument("--preset-id", default=self._defaults.preset_id)
        parser.add_argument("--prompt", default=None)
        parser.add_argument("--image-path", type=Path)
        parser.add_argument("--action-path", type=Path)
        parser.add_argument("--calibration-path", type=Path)
        parser.add_argument("--example-index", type=int, default=0)
        parser.add_argument(
            "--total-blocks", type=int, default=self._defaults.total_blocks
        )
        parser.add_argument("--device", default="cuda")
        parser.add_argument(
            "--compile", action=argparse.BooleanOptionalAction, default=None
        )
        args = parser.parse_args(list(commandline_args))
        if args.total_blocks <= 0:
            raise ValueError("--total-blocks must be > 0.")
        if args.image_path is None or args.action_path is None:
            if args.image_path is not None or args.action_path is not None:
                raise ValueError("Pass --image-path and --action-path together.")
            example = self._hooks.example_loader(
                is_rank_zero=True, example_idx=args.example_index
            )
            args.image_path = example.image_path
            args.action_path = example.action_path
            args.calibration_path = example.calibration_path
            prompt_path = example.prompt_path
        else:
            prompt_path = None
        for path in (args.image_path, args.action_path, args.calibration_path):
            if path is not None and not path.is_file():
                raise FileNotFoundError(path)
        pipeline_config = (
            self._pipeline_config or self._hooks.pipeline_configs[args.preset_id]
        )
        if args.compile is not None:
            pipeline_config = derive_config(
                pipeline_config,
                diffusion_model={"transformer": {"compile_network": args.compile}},
            )
        self._pipeline_config = pipeline_config
        prompt = (args.prompt or "").strip()
        if not prompt and prompt_path is not None and prompt_path.is_file():
            prompt = next(
                (line.strip() for line in prompt_path.read_text().splitlines() if line.strip()),
                "",
            )
        if not prompt:
            prompt = self._hooks.preset_prompts.get(args.preset_id, "").strip()
        if not prompt:
            prompt = self._defaults.prompt
        self._config = Action2VConfig(
            prompt=prompt,
            image_path=args.image_path,
            action_path=args.action_path,
            calibration_path=args.calibration_path,
            total_blocks=args.total_blocks,
            device=args.device,
        )

    def session_desc(self) -> SessionDesc:
        return self._desc

    def create_session(self, session_desc: SessionDesc) -> ISession:
        config = self._config
        if config is None or self._pipeline_config is None:
            raise RuntimeError("init() must run before create_session().")
        if session_desc.output_layout is not VideoTensorLayout.tchw:
            raise ValueError("Action2V requires tchw output.")
        return Action2VSession(
            pipeline_factory=lambda: self._load_pipeline(config.device),
            config=config,
            desc=session_desc,
            hooks=self._hooks,
            title=self._defaults.title,
            ui_renderer=(
                None
                if self._ui_renderer_factory is None
                else self._ui_renderer_factory(
                    session_desc.video_width, session_desc.video_height
                )
            ),
        )

    def close(self) -> None:
        pipeline = self._pipeline
        self._pipeline = None
        close = getattr(pipeline, "close", None)
        if callable(close):
            close()

    def _load_pipeline(self, device: str) -> Any:
        """Build the shared pipeline lazily from the model thread."""
        if self._pipeline_config is None:
            raise RuntimeError("init() must run before pipeline loading.")
        if self._pipeline is None:
            self._pipeline = self._pipeline_config.setup().to(device).eval()
        return self._pipeline


def _drive_key(key: str) -> str | None:
    aliases = {
        "arrowup": "w",
        "arrowdown": "s",
        "arrowleft": "a",
        "arrowright": "d",
    }
    key = key.strip().lower()
    key = aliases.get(key, key)
    return key if key in {"w", "a", "s", "d", "q", "e", "i", "k"} else None


def _gamepad_keys(event: GamepadUserInputEventData) -> frozenset[str]:
    x = event.axes[0] if len(event.axes) > 0 else 0.0
    y = event.axes[1] if len(event.axes) > 1 else 0.0
    keys: set[str] = set()
    if x < -0.2:
        keys.add("a")
    elif x > 0.2:
        keys.add("d")
    if y < -0.2:
        keys.add("w")
    elif y > 0.2:
        keys.add("s")
    return frozenset(keys)


__all__ = [
    "Action2VInputPaths",
    "Action2VApplicationDefaults",
    "Action2VApplicationHooks",
    "Action2VApplication",
    "Action2VModelLoop",
    "Action2VUILoop",
]
