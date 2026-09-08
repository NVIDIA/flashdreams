# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""HD-map-conditioned video driving with native v2 and SlangPy UI loops."""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from functools import partial
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from torch import Tensor

from flashdreams.api_v2.application import IApplication
from flashdreams.api_v2.loop import IModelLoop, IUILoop, invoke_async
from flashdreams.infra.config import derive_config
from flashdreams.infra.pipeline import StreamInferencePipelineConfig
from flashdreams.infra.postprocess import VideoPostprocessChainConfig
from flashdreams.plugins.registry import discover_postprocess_presets
from flashdreams.runtime.keyboard import normalize_key
from flashdreams.runtime_v2.session_desc import BackpressureMode, SessionDesc
from flashdreams.runtime_v2.step_result import StepResult
from flashdreams.runtime_v2.user_input_event import (
    GamepadUserInputEvent,
    GameWheelUserInputEvent,
    KeyboardInputState,
    KeyboardUserInputEvent,
)
from flashdreams.runtime_v2.user_input_events import UserInputEvents
from flashdreams.runtime_v2.video_tensor import VideoTensorLayout
from interactive_drive.backends.base import RenderBackend
from interactive_drive.backends.world_model import (
    WorldModelRenderBackend,
)
from interactive_drive.config import (
    AppConfig,
    BevConfig,
    ChunkConfig,
    RasterConfig,
    VehicleConfig,
    WorldModelProfileConfig,
)
from interactive_drive.input.keyboard import command_from_snapshot
from interactive_drive.scene_download import download_default_scene
from interactive_drive.scene_loader import load_scene_bundle
from interactive_drive.simulation.ego_vehicle_kinematics import (
    build_ground_snapper,
    sample_chunk_trajectory,
    state_from_initial_pose,
)
from interactive_drive.simulation.game_physics import GamePhysicsWorld
from interactive_drive.types import (
    ControlSnapshot,
    DriverCommand,
    FrameChunk,
    SceneBundle,
    VehicleState,
)

BackendFactory = Callable[[AppConfig], RenderBackend]
SceneLoader = Callable[..., SceneBundle]
ViewMode = Literal["rgb", "hdmap", "physx"]

_VIEW_MODES: tuple[ViewMode, ...] = ("rgb", "hdmap", "physx")
_VIEW_MODE_KEYS: dict[str, ViewMode] = dict(zip(("1", "2", "3"), _VIEW_MODES))
_GAMEPAD_REVERSE_BUTTON = 5
_GAMEPAD_RESET_BUTTON = 9


@dataclass(frozen=True, slots=True)
class InteractiveDriveApplicationDefaults:
    """Defaults supplied by a scene-driving model integration."""

    title: str = "Scene Drive"
    """Window title."""

    slug: str = "scene-drive"
    """Application slug shown in parser diagnostics."""

    total_blocks: int = 60
    """Default number of generated blocks."""

    fps: int = 30
    width: int = 1280
    height: int = 704

    pipeline_config: StreamInferencePipelineConfig | None = None
    """Model-owned streaming inference pipeline configuration."""


@dataclass(frozen=True, slots=True)
class InteractiveDriveConfig:
    app: AppConfig
    total_blocks: int
    view_mode: ViewMode
    no_ui: bool = False


@dataclass(frozen=True, slots=True)
class DriveTelemetry:
    """Thread-safe model telemetry published to an application UI loop."""

    speed_mps: float
    reverse: bool
    blocks_generated: int
    frames_in_chunk: int
    scene_path: Path
    variant: str
    postprocess_enabled: bool
    model_loop_ms: float


@dataclass(slots=True)
class DriveInputState:
    """Driving input state owned and updated by one runtime loop."""

    pressed_keys: set[str] = field(default_factory=set)
    """Normalized driving keys currently held down."""

    controller_command: DriverCommand | None = None
    """Latest wheel or gamepad command; ``None`` when none is connected."""

    def apply(self, events: UserInputEvents) -> None:
        """Apply one loop's user-input events to this loop's state."""
        for event in events.get_events():
            if isinstance(event, KeyboardUserInputEvent):
                key = _normalize_drive_key(event.key)
                if key is None:
                    continue
                if event.state is KeyboardInputState.PRESSED:
                    self.pressed_keys.add(key)
                else:
                    self.pressed_keys.discard(key)
            elif isinstance(event, GameWheelUserInputEvent):
                self.controller_command = (
                    None
                    if event.action == "disconnected"
                    else DriverCommand(
                        throttle=event.throttle,
                        brake=event.brake,
                        steer=-event.steering,
                        steer_is_direct=True,
                        manual_control=True,
                    )
                )
            elif isinstance(event, GamepadUserInputEvent):
                self.controller_command = _gamepad_command(event)

    def command(self) -> DriverCommand:
        """Return the command represented by this loop's current input state."""
        if self.controller_command is not None:
            return self.controller_command
        return command_from_snapshot(ControlSnapshot(pressed=self.pressed_keys))

    def source(self) -> str:
        """Return a short label for the active input source."""
        if self.controller_command is not None:
            return "wheel/gamepad"
        return "keyboard" if self.pressed_keys else "idle"


@dataclass(slots=True)
class InteractiveDriveModelState:
    backend_factory: BackendFactory
    config: InteractiveDriveConfig
    desc: SessionDesc
    scene_loader: SceneLoader
    scene: SceneBundle | None = None
    vehicle: VehicleState | None = None
    ground_snapper: Any | None = None
    next_timestamp_us: int = 0
    blocks_generated: int = 0
    first_chunk: bool = True
    drive_input: DriveInputState = field(default_factory=DriveInputState)
    """Driving input updated on the model thread."""

    last_command: DriverCommand = field(default_factory=DriverCommand)
    view_mode: ViewMode = "rgb"
    postprocess_enabled: bool = True
    pending_prompt: str | None = None
    reset_pending: bool = False
    controller_reset_pressed: bool = False
    ui_loop: IUILoop[Any] | None = None
    backend: RenderBackend | None = None
    physics_world: GamePhysicsWorld | None = None

    def restart(self, prompt: str) -> None:
        prompt = prompt.strip()
        if not prompt:
            raise ValueError("prompt must be non-empty.")
        self.pending_prompt = prompt
        self.reset_pending = True
        self._notify("Restart queued on the model loop.")

    def set_view_mode(self, view_mode: ViewMode) -> None:
        self.view_mode = view_mode
        if self.ui_loop is not None:
            invoke_async(
                self.ui_loop,
                lambda state, view_mode=view_mode: state.set_view_mode(view_mode),
            )

    def select_scene(self, scene_path: Path, variant: str) -> None:
        """Queue a fresh rollout for a scene and weather variant."""
        self.config = replace(
            self.config,
            app=replace(
                self.config.app,
                scene_path=Path(scene_path),
                variant=variant,
            ),
        )
        self.reset_pending = True
        self._notify(f"Scene change queued: {Path(scene_path).stem} ({variant}).")

    def select_variant(self, variant: str) -> None:
        """Queue a fresh rollout using another variant of the current scene."""
        self.select_scene(self.config.app.scene_path, variant)

    def set_postprocess_enabled(self, enabled: bool) -> None:
        """Toggle generated-video post-processing without rebuilding the model."""
        self.postprocess_enabled = bool(enabled)
        if self.backend is not None:
            self.backend.set_postprocess_enabled(self.postprocess_enabled)
        self._notify(
            "Post-processing enabled."
            if self.postprocess_enabled
            else "Post-processing disabled."
        )

    def _notify(self, status: str) -> None:
        if self.ui_loop is not None:
            invoke_async(
                self.ui_loop,
                lambda state, status=status: state.set_status(status),
            )

    def _publish_drive_telemetry(self, chunk: FrameChunk, model_loop_ms: float) -> None:
        """Send controls, vehicle state, and chunk metrics to the UI."""
        if self.ui_loop is None or self.vehicle is None:
            return
        telemetry = DriveTelemetry(
            speed_mps=self.vehicle.speed_mps,
            reverse=self.last_command.reverse or self.vehicle.speed_mps < -0.01,
            blocks_generated=self.blocks_generated,
            frames_in_chunk=len(chunk.frames),
            scene_path=self.config.app.scene_path,
            variant=self.config.app.variant,
            postprocess_enabled=self.postprocess_enabled,
            model_loop_ms=model_loop_ms,
        )
        invoke_async(
            self.ui_loop,
            lambda state, telemetry=telemetry: state.set_drive_telemetry(telemetry),
        )


class InteractiveDriveModelLoop(IModelLoop[InteractiveDriveModelState]):
    """Own scene state, simulation, model cache, and backend execution."""

    def step(self, step_index: int, events: UserInputEvents) -> list[StepResult]:
        step_started_at = time.perf_counter()
        state = self.state
        self._apply_events(events)
        if state.scene is None or state.reset_pending:
            self._initialize_rollout()
        assert state.scene is not None
        assert state.vehicle is not None
        assert state.backend is not None
        chunk_size = (
            state.backend.initial_chunk_frames
            if state.first_chunk
            else state.backend.chunk_frames
        )
        command = self._command()
        state.last_command = command
        trajectory = sample_chunk_trajectory(
            start_state=state.vehicle,
            start_timestamp_us=state.next_timestamp_us,
            command=command,
            chunk_size=chunk_size,
            chunk_config=state.config.app.chunk,
            vehicle_config=state.config.app.vehicle,
            ground_snapper=state.ground_snapper,
            physics_world=state.physics_world,
            capture_physics_debug=state.view_mode == "physx",
        )
        chunk = (
            state.backend.render_first_chunk(trajectory)
            if state.first_chunk
            else state.backend.render_next_chunk(trajectory)
        )
        state.vehicle = trajectory.boundary_state_after_chunk
        state.next_timestamp_us = int(
            trajectory.timestamps_us[-1] + state.config.app.chunk.frame_interval_us
        )
        state.first_chunk = False
        state.blocks_generated += 1
        output = _frame_chunk_tensor(chunk, state.view_mode)
        state._notify(
            "Rollout complete."
            if self.is_finished()
            else _telemetry_status(state.vehicle, state.blocks_generated)
        )
        results = [
            StepResult(
                step_index=step_index,
                output=output,
                frame_count=int(output.shape[0]),
                output_layout=state.desc.output_layout,
                metrics={},
            )
        ]
        bev_output = self._bev_chunk_tensor(chunk)
        if bev_output is not None:
            results.append(
                StepResult(
                    step_index=step_index,
                    output=bev_output,
                    frame_count=int(bev_output.shape[0]),
                    output_layout=state.desc.output_layout,
                    metrics={},
                )
            )
        model_loop_ms = (time.perf_counter() - step_started_at) * 1000.0
        state._publish_drive_telemetry(chunk, model_loop_ms)
        return results

    @staticmethod
    def _bev_chunk_tensor(chunk: FrameChunk) -> Tensor | None:
        """Build a BEV channel aligned one-to-one with the emitted video frames."""
        bev_values = [frame.bev_host_uint8 for frame in chunk.frames]
        if not any(value is not None for value in bev_values):
            return None
        if any(value is None for value in bev_values):
            raise ValueError(
                "The render backend must provide one BEV frame per emitted frame."
            )

        frames: list[Tensor] = []
        for value in bev_values:
            array = np.asarray(value)
            tensor = torch.from_numpy(np.ascontiguousarray(array))
            if tensor.ndim != 3:
                raise ValueError(
                    f"Expected HWC BEV frame, received shape {tuple(tensor.shape)}"
                )
            frames.append(tensor.permute(2, 0, 1))
        return torch.stack(frames)

    def is_finished(self) -> bool:
        total = self.state.config.total_blocks
        return total > 0 and self.state.blocks_generated >= total

    def reset(self) -> None:
        self.state.reset_pending = True

    def close(self) -> None:
        if self.state.backend is not None:
            self.state.backend.close()
            self.state.backend = None
        if self.state.physics_world is not None:
            self.state.physics_world.close()
            self.state.physics_world = None
        self.state.scene = None

    def _initialize_rollout(self) -> None:
        state = self.state
        app_config = state.config.app
        prompt = state.pending_prompt
        state._notify("Loading scene conditioning on the model loop...")
        if state.backend is None:
            state.backend = state.backend_factory(app_config)
        backend = state.backend
        scene = state.scene_loader(
            app_config.scene_path,
            app_config.camera_name,
            app_config.variant,
            prompt if prompt is not None else app_config.prompt_override,
            app_config.raster,
        )
        if state.scene is None:
            backend.warmup_model()
        else:
            backend.reset_scene_conditioning()
        backend.load_scene(scene)
        backend.set_postprocess_enabled(state.postprocess_enabled)
        state.scene = scene
        state.vehicle = state_from_initial_pose(
            scene.initial_rig_to_world,
            scene.initial_yaw_rad,
            scene.initial_speed_mps,
        )
        state.ground_snapper = build_ground_snapper(scene)
        if state.physics_world is not None:
            state.physics_world.close()
        state.physics_world = GamePhysicsWorld(scene, app_config.vehicle)
        state.next_timestamp_us = scene.initial_timestamp_us
        state.blocks_generated = 0
        state.first_chunk = True
        state.reset_pending = False
        state.pending_prompt = None

    def _apply_events(self, events: UserInputEvents) -> None:
        state = self.state
        for event in events.get_events():
            if isinstance(event, KeyboardUserInputEvent):
                view_mode = _VIEW_MODE_KEYS.get(event.key.strip().lower())
                if view_mode is not None:
                    if event.state is KeyboardInputState.PRESSED:
                        state.set_view_mode(view_mode)
            elif isinstance(event, GamepadUserInputEvent):
                reset_pressed = event.action == "state" and _gamepad_button_pressed(
                    event, _GAMEPAD_RESET_BUTTON
                )
                if reset_pressed and not state.controller_reset_pressed:
                    state.reset_pending = True
                    state._notify("Restart queued.")
                state.controller_reset_pressed = reset_pressed
        state.drive_input.apply(events)

    def _command(self) -> DriverCommand:
        return self.state.drive_input.command()


class _InteractiveDriveApplicationBase(IApplication):
    """Parse shared driving options and own model construction state."""

    def __init__(
        self,
        *,
        defaults: InteractiveDriveApplicationDefaults | None = None,
        scene_loader: SceneLoader = load_scene_bundle,
    ) -> None:
        defaults = defaults or InteractiveDriveApplicationDefaults()
        self._title = defaults.title
        self._slug = defaults.slug
        self._default_blocks = defaults.total_blocks
        self._default_fps = defaults.fps
        self._default_width = defaults.width
        self._default_height = defaults.height
        self._backend_factory = partial(
            _build_backend,
            pipeline_config=defaults.pipeline_config,
        )
        self._scene_loader = scene_loader
        self._config: InteractiveDriveConfig | None = None
        self._desc = SessionDesc(
            output_layout=VideoTensorLayout.tchw,
            backpressure_mode=BackpressureMode.BLOCK,
            # Run the UI faster than the generated-video frame rate so stale
            # chunks are done presenting before model-thread finishes its next step.
            frames_per_second_for_ui=60,
            frames_per_second_for_step=defaults.fps,
            video_width=defaults.width,
            video_height=defaults.height,
        )

    def init(self, commandline_args: Sequence[str]) -> None:
        parser = argparse.ArgumentParser(prog=f"flashdreams-run-v2 {self._slug} --")
        parser.add_argument(
            "--scene",
            type=Path,
            help=(
                "Local USDZ scene. If omitted, download the built-in default "
                "scene from Hugging Face."
            ),
        )
        parser.add_argument("--prompt")
        parser.add_argument("--camera", default="camera_front_wide_120fov")
        parser.add_argument("--variant", default="default")
        parser.add_argument("--total-blocks", type=int, default=self._default_blocks)
        parser.add_argument("--fps", type=int, default=self._default_fps)
        parser.add_argument("--width", type=int, default=self._default_width)
        parser.add_argument("--height", type=int, default=self._default_height)
        parser.add_argument("--view", choices=_VIEW_MODES, default="rgb")
        parser.add_argument(
            "--no-ui",
            action="store_true",
            help="Disable the application UI and present model output directly.",
        )
        parser.add_argument(
            "--game-mode",
            action="store_true",
            help=("Enable the vehicle speed limit and actor/static collisions."),
        )
        parser.add_argument(
            "--postprocess-preset",
            default="",
            choices=tuple(sorted(discover_postprocess_presets())),
            help=(
                "Video post-processing preset for generated world-model frames. "
                "A configured preset starts enabled and can be toggled in the HUD."
            ),
        )
        parser.add_argument(
            "--world-model-profile",
            action="store_true",
            help="Enable synchronized world-model profiling.",
        )
        parser.add_argument("--world-model-device", default="cuda:0")
        parser.add_argument("--world-model-seed", type=int)
        parser.add_argument(
            "--world-model-debug-condition-frame-dir",
            type=Path,
        )
        args = parser.parse_args(list(commandline_args))
        scene = args.scene
        if scene is None:
            scene = download_default_scene()
        if not scene.is_file():
            raise FileNotFoundError(scene)
        if args.total_blocks < 0:
            raise ValueError("--total-blocks must be >= 0 (0 means unbounded).")
        if args.fps <= 0 or args.width <= 0 or args.height <= 0:
            raise ValueError("--fps, --width, and --height must be > 0.")
        chunk = ChunkConfig(fps=args.fps)
        raster = RasterConfig(width=args.width, height=args.height)
        app_config = AppConfig(
            scene_path=scene,
            game_mode=args.game_mode,
            camera_name=args.camera,
            variant=args.variant,
            prompt_override=args.prompt,
            chunk=chunk,
            raster=raster,
            world_model_profile=WorldModelProfileConfig(
                enabled=args.world_model_profile
            ),
            world_model_device=args.world_model_device,
            world_model_seed=args.world_model_seed,
            world_model_debug_condition_frame_dir=(
                args.world_model_debug_condition_frame_dir
            ),
            postprocess=VideoPostprocessChainConfig(
                preset=args.postprocess_preset,
            ),
            bev=BevConfig(enabled=False),
            vehicle=VehicleConfig(),
        )
        self._config = InteractiveDriveConfig(
            app=app_config,
            total_blocks=args.total_blocks,
            view_mode=args.view,
            no_ui=args.no_ui,
        )
        self._desc = replace(
            self._desc,
            frames_per_second_for_step=app_config.chunk.fps,
            video_width=app_config.raster.width,
            video_height=app_config.raster.height,
        )

    def session_desc(self) -> SessionDesc:
        return self._desc


def _build_backend(
    config: AppConfig,
    *,
    pipeline_config: StreamInferencePipelineConfig | None,
) -> RenderBackend:
    if pipeline_config is None:
        raise ValueError(
            "The application defaults must provide pipeline_config for model rendering."
        )
    resolved_pipeline_config = derive_config(
        pipeline_config,
        enable_sync_and_profile=config.world_model_profile.enabled,
        diffusion_model=dict(
            seed=(42 if config.world_model_seed is None else config.world_model_seed)
        ),
    )
    pipeline = (
        resolved_pipeline_config.setup()
        .to(torch.device(config.world_model_device))
        .eval()
    )
    if config.world_model_seed is None:
        pipeline.diffusion_model.config.seed = None
    return WorldModelRenderBackend(
        pipeline=pipeline,
        chunk=config.chunk,
        raster=config.raster,
        bev=config.bev,
        vehicle=config.vehicle,
        postprocess=config.postprocess,
        debug_condition_frame_dir=config.world_model_debug_condition_frame_dir,
    )


def _normalize_drive_key(key: str) -> str | None:
    key = normalize_key(key)
    return key if key in {"w", "a", "s", "d", "space"} else None


def _gamepad_command(event: GamepadUserInputEvent) -> DriverCommand | None:
    if event.action == "disconnected":
        return None
    if event.action != "state":
        return None
    steer = -(event.axes[0] if event.axes else 0.0)
    throttle = event.buttons[7] if len(event.buttons) > 7 else 0.0
    brake = event.buttons[6] if len(event.buttons) > 6 else 0.0
    return DriverCommand(
        throttle=throttle,
        brake=brake,
        steer=steer,
        reverse=_gamepad_button_pressed(event, _GAMEPAD_REVERSE_BUTTON),
        steer_is_direct=True,
        manual_control=True,
    )


def _gamepad_button_pressed(event: GamepadUserInputEvent, index: int) -> bool:
    if len(event.pressed) > index:
        return event.pressed[index]
    return len(event.buttons) > index and event.buttons[index] > 0.5


def _frame_chunk_tensor(chunk: FrameChunk, view_mode: ViewMode) -> Tensor:
    frames: list[Tensor] = []
    for frame in chunk.frames:
        if view_mode == "physx":
            value = frame.physx_rgb_host_uint8
            if value is None:
                raise ValueError("PhysX view requires a PhysX debug frame.")
            physx = np.asarray(value)
            hdmap = np.asarray(frame.rgb_host_uint8)
            if physx.shape != hdmap.shape:
                raise ValueError(
                    "PhysX and HD-map frames must have matching shapes; "
                    f"received {physx.shape} and {hdmap.shape}."
                )
            if physx.ndim != 3 or physx.shape[-1] < 3:
                raise ValueError(
                    f"Expected HWC PhysX frame, received shape {physx.shape}"
                )
            collider_mask = np.any(physx[..., :3] != 0, axis=-1, keepdims=True)
            array = np.where(collider_mask, physx, hdmap)
        elif view_mode == "hdmap":
            array = np.asarray(frame.rgb_host_uint8)
        else:
            value = (
                frame.model_rgb_host_uint8
                if frame.model_rgb_host_uint8 is not None
                else frame.rgb_host_uint8
            )
            array = np.asarray(value)
        tensor = torch.from_numpy(np.ascontiguousarray(array))
        if tensor.ndim != 3:
            raise ValueError(
                f"Expected HWC frame, received shape {tuple(tensor.shape)}"
            )
        frames.append(tensor.permute(2, 0, 1))
    if not frames:
        raise ValueError("The world-model backend returned an empty frame chunk.")
    return torch.stack(frames)


def _telemetry_status(vehicle: VehicleState, blocks: int) -> str:
    speed_mph = abs(vehicle.speed_mps) * 2.236936
    return (
        f"Block {blocks}; speed {speed_mph:.1f} mph; steer {vehicle.steer_rad:.2f} rad."
    )


__all__ = [
    "BackendFactory",
    "DriveInputState",
    "DriveTelemetry",
    "InteractiveDriveApplicationDefaults",
    "InteractiveDriveConfig",
    "InteractiveDriveModelLoop",
    "InteractiveDriveModelState",
    "SceneLoader",
    "ViewMode",
]
