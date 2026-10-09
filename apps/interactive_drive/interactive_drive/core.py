# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""HD-map-conditioned video driving with native v2 and SlangPy UI loops."""

from __future__ import annotations

import argparse
import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from functools import partial
from pathlib import Path
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
import torch
import torch.distributed as dist
from torch import Tensor
from torch.nn import functional as F

from flashdreams.api_v2.application import IApplication
from flashdreams.api_v2.loop import IModelLoop, IUILoop, invoke_async
from flashdreams.core.distributed.parallel import ParallelContext
from flashdreams.infra.acceleration.frame_prefetch import LazyCudaFrame
from flashdreams.infra.config import derive_config
from flashdreams.infra.pipeline import StreamInferencePipelineConfig
from flashdreams.infra.postprocess import VideoPostprocessChainConfig, VideoSpec
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
    PresentationLayout,
    RasterConfig,
    VehicleConfig,
)
from interactive_drive.input.keyboard import command_from_snapshot
from interactive_drive.math3d import normalize_camera_name
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
    PresentedFrame,
    SceneBundle,
    VehicleState,
)

BackendFactory = Callable[[AppConfig], RenderBackend]
SceneLoader = Callable[..., SceneBundle]
ViewMode = Literal["rgb", "hdmap", "physx"]

HUD_DESIGN_SIZE = (1280.0, 704.0)
"""The viewport the HUD's panel sizes were chosen against."""

VIEW_MODES: tuple[ViewMode, ...] = ("rgb", "hdmap", "physx")
_VIEW_MODE_KEYS: dict[str, ViewMode] = dict(zip(("1", "2", "3"), VIEW_MODES))
_GAMEPAD_REVERSE_BUTTON = 5
_GAMEPAD_RESET_BUTTON = 9


def hud_canvas(width: int, height: int) -> tuple[int, int]:
    """A frame to composite a mosaic of this size into, so the HUD fits unshrunk.

    A whole multiple of the mosaic rather than :data:`HUD_DESIGN_SIZE` itself.
    The drive is resized into whatever this returns and the two aspect ratios
    do not match, so anything else would stretch the picture to buy sharper
    text.
    """
    design_width, design_height = HUD_DESIGN_SIZE
    factor = max(
        1,
        math.ceil(design_width / width),
        math.ceil(design_height / height),
    )
    return width * factor, height * factor


def view_modes_for(physics: bool) -> tuple[ViewMode, ...]:
    """The presentation streams this app can actually draw.

    The collider overlay comes from the physics world, so an integration
    running without one has nothing to put in that stream, and offering it
    anyway means a driver can ask for a frame that was never captured.
    """
    return (
        VIEW_MODES if physics else tuple(mode for mode in VIEW_MODES if mode != "physx")
    )


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

    scene: Path | None = None
    """What ``--scene`` falls back to, for an integration that reads its own.

    The built-in default is a USDZ downloaded from Hugging Face, which is no use
    to a model that cannot read one, so an integration bringing its own scene
    loader names the scene its loader expects instead.
    """

    sample: Path | None = None
    """What ``--sample`` falls back to, for a loader that reads a recording."""

    physics: bool = True
    """Simulate colliders.

    Off for an integration whose scene geometry stays on the GPU, where there
    is nothing for the host to build a collider from and native PhysX would be
    prepared for an empty world.
    """

    reverse: bool = True
    """Offer a reverse gear.

    Off for a model whose training footage only ever goes forwards, where asking
    to reverse produces a view it cannot draw. The request becomes braking
    rather than being dropped, since a driver pressing back wants to slow down
    and stopping is something the model has seen.
    """

    bev: bool = True
    """Render a top-down view beside the drive, and show the HUD's map panel.

    The overhead camera is one the app's own rasterizer appends to the rig, so a
    backend that draws its own control has no such view to present. Off leaves
    the panel out rather than standing an empty one next to the picture.
    """

    full_size_hud: bool = False
    """Composite into a canvas big enough to draw the HUD at its design size,
    where there is a HUD at all.

    The panels are fixed pixel sizes, so a model generating a frame smaller
    than :data:`HUD_DESIGN_SIZE` has its HUD drawn shrunk and then magnified
    again by whatever displays the stream, which is what makes the text look
    soft. On costs four times the pixels to overlay and to encode for a frame
    half the design size, and buys nothing for the drive itself, so it is for
    an integration meant to be looked at rather than one chosen for speed.
    """

    pipeline_config: StreamInferencePipelineConfig | None = None
    """Model-owned streaming inference pipeline configuration."""

    backend_factory: BackendFactory | None = None
    """Builds the model that renders the drive, for models that are not a
    ``StreamInferencePipelineConfig``.

    An integration supplies this or ``pipeline_config``, not both: each is a way
    of saying which model runs, and two answers to that has no sensible meaning.
    Whatever a factory needs beyond the scene is its own to close over, which is
    what the default does with the pipeline config.
    """

    parallel_context: ParallelContext | None = None
    """The mesh the model runs on, for an integration that shards one.

    The app does nothing with the axes and should not know what they are: how a
    model divides itself is the model's business, and the two here are a
    checkpoint's own. What the app does with this is the one fact a mesh implies
    for a user interface, that a drive has a single window and a single driver
    however many ranks are generating it, so ranks other than zero present
    nothing and build no HUD.

    ``None`` is a drive on one card, which is every integration that has not
    asked otherwise.
    """

    def __post_init__(self) -> None:
        if self.backend_factory is not None and self.pipeline_config is not None:
            raise ValueError(
                "Supply backend_factory or pipeline_config, not both: they are "
                "two ways of naming the model that renders the drive."
            )


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


@dataclass(frozen=True, slots=True)
class QueuedRestart:
    """A restart the HUD asked for, in a shape the other ranks can be told in.

    Plain and picklable, because this crosses the mesh. The scene and the
    variant travel rather than being applied where they were chosen, since a
    rank with no window would otherwise never hear that they had changed.
    """

    scene_path: Path | None = None
    variant: str | None = None
    prompt: str | None = None


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
    queued: QueuedRestart | None = None
    """What the HUD asked for, until the next step tells every rank about it."""

    control_group: Any = None
    """Where that telling happens. See :meth:`_control_group`."""

    controller_reset_pressed: bool = False
    ui_loop: IUILoop[Any] | None = None
    backend: RenderBackend | None = None
    owns_backend: bool = True
    parallel_context: ParallelContext | None = None
    """The mesh, so a rank can tell whether anybody is watching it."""

    physics_world: GamePhysicsWorld | None = None

    def restart(self, prompt: str | None = None) -> None:
        """Start the drive again, on a new prompt only if one is given.

        ``None`` rather than a default string, because restarting and
        rewording are different requests. A scene that conditions on something
        of its own, a prompt file or a sample's per-camera captions, loses it
        the moment a restart carries any prompt at all, and the driver who
        pressed a button labelled Restart did not ask for that.
        """
        if prompt is None:
            # Queued with nothing in it rather than with a prompt of ``None``,
            # which :meth:`_queue` merges and would use to wipe a wording asked
            # for earlier in the same step.
            self._queue()
        else:
            if not prompt.strip():
                raise ValueError("prompt must be non-empty when one is given.")
            self._queue(prompt=prompt.strip())
        self._notify("Restart queued on the model loop.")

    def _queue(self, **asked: Any) -> None:
        """Put a HUD action where :meth:`agree_on_queued` will find it.

        Merged rather than replaced, so two controls touched between one step
        and the next both take effect, which is what setting the fields
        directly used to give.
        """
        self.queued = replace(self.queued or QueuedRestart(), **asked)

    def _control_group(self) -> Any:
        """A gloo group for saying which button was pressed, made on first use.

        Not the mesh's own, which under NCCL would pickle through the rank's
        card and synchronize it once a step forever, to be told almost always
        that nothing happened. A step loop that overlaps drawing with
        generating is the wrong place to put a device sync, and ``StepAgreement``
        keeps a gloo group of its own for the same reason.

        On first use rather than at construction, which is collective all the
        same: the runtime admits every rank to a step before any of them takes
        it, so they reach this together or not at all.
        """
        if self.control_group is None:
            self.control_group = dist.new_group(backend="gloo")
        return self.control_group

    def agree_on_queued(self) -> None:
        """Give every rank the action the rank holding the window was asked for.

        The HUD draws where the window is, so a button reaches one rank's state
        and no other. Input events do not have this problem: they arrive at
        rank zero and the runtime broadcasts them before any rank steps, which
        is why the gamepad's reset button and the keyboard already agree. A
        restart only rank zero knew about would have it rebuilding the scene,
        and entering the conditioning's collectives, while every other rank was
        still generating, and the two sequences do not meet again.

        Asked every step, because whether a button was pressed is itself
        something only one rank knows, so there is no cheaper question to ask
        first. That is what makes the group it is asked on matter.
        """
        mesh = self.parallel_context
        queued = self.queued
        if mesh is not None and mesh.world_size > 1 and dist.is_initialized():
            payload: list[QueuedRestart | None] = [queued if mesh.is_main else None]
            dist.broadcast_object_list(payload, src=0, group=self._control_group())
            queued = payload[0]
        self.queued = None
        if queued is None:
            return
        if queued.scene_path is not None:
            self.config = replace(
                self.config,
                app=replace(
                    self.config.app,
                    scene_path=queued.scene_path,
                    variant=queued.variant or self.config.app.variant,
                ),
            )
        if queued.prompt is not None:
            self.pending_prompt = queued.prompt
        self.reset_pending = True

    def set_view_mode(self, view_mode: ViewMode) -> None:
        self.view_mode = view_mode
        if self.ui_loop is not None:
            invoke_async(
                self.ui_loop,
                lambda state, view_mode=view_mode: state.set_view_mode(view_mode),
            )

    def select_scene(self, scene_path: Path, variant: str) -> None:
        """Queue a fresh rollout for a scene and weather variant."""
        self._queue(scene_path=Path(scene_path), variant=variant)
        self._notify(f"Scene change queued: {Path(scene_path).stem} ({variant}).")

    def select_variant(self, variant: str) -> None:
        """Queue a fresh rollout using another variant of the current scene."""
        queued = self.queued
        scene = (
            self.config.app.scene_path
            if queued is None or queued.scene_path is None
            else queued.scene_path
        )
        self.select_scene(scene, variant)

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
        state.agree_on_queued()
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
        finalize_metrics = state.backend.finalize()
        state.vehicle = trajectory.boundary_state_after_chunk
        state.next_timestamp_us = int(
            trajectory.timestamps_us[-1] + state.config.app.chunk.frame_interval_us
        )
        state.first_chunk = False
        state.blocks_generated += 1
        finished = self.is_finished()
        if finished:
            chunk = replace(
                chunk,
                frames=(*chunk.frames, *state.backend.finish()),
            )
        mesh = state.parallel_context
        if mesh is not None and not mesh.is_main:
            # Generating happens on every rank, because the collectives inside
            # it need every rank to arrive. Presenting happens on the one with
            # a window: tiling a grid, resizing it and copying it out is the
            # part of a worker's step that buys nothing, so it stops here. No
            # result reads to the runtime as a step that presented nothing.
            return []
        output = _frame_chunk_tensor(
            chunk,
            state.view_mode,
            layout=state.config.app.presentation,
            output_size=(state.desc.video_height, state.desc.video_width),
            output_device=_output_cuda_device(state.config.app),
        )
        state._notify(
            "Rollout complete."
            if finished
            else _telemetry_status(state.vehicle, state.blocks_generated)
        )
        results = []
        if output.shape[0]:
            results.append(
                StepResult(
                    step_index=step_index,
                    output=output,
                    frame_count=int(output.shape[0]),
                    output_layout=state.desc.output_layout,
                    metrics=finalize_metrics,
                )
            )
        bev_output = self._bev_chunk_tensor(chunk)
        if bev_output is not None:
            results.append(
                StepResult(
                    step_index=step_index,
                    output=bev_output,
                    frame_count=int(bev_output.shape[0]),
                    output_layout=state.desc.output_layout,
                    metrics=finalize_metrics,
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
        if self.state.backend is not None and self.state.owns_backend:
            self.state.backend.close()
        self.state.backend = None
        if self.state.physics_world is not None:
            self.state.physics_world.close()
            self.state.physics_world = None
        self.state.scene = None
        if self.state.control_group is not None:
            # Locally and without a shutdown collective, the way the runtime
            # releases the group `StepAgreement` keeps: one rank can reach here
            # unwinding from a failure while the others are still generating.
            group, self.state.control_group = self.state.control_group, None
            dist.destroy_process_group(group)

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
            app_config.camera_names,
            app_config.variant,
            prompt if prompt is not None else app_config.prompt_override,
            app_config.raster,
            config=app_config,
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
        # A scene whose geometry never reaches the host has nothing to collide
        # with, and native PhysX is a download this app would then make for
        # an empty world. The kinematics already treat it as optional.
        state.physics_world = (
            GamePhysicsWorld(scene, app_config.vehicle) if app_config.physics else None
        )
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
                if view_mode is not None and view_mode in view_modes_for(
                    state.config.app.physics
                ):
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
        command = self.state.drive_input.command()
        if command.reverse and not self.state.config.app.reverse:
            # Braking rather than nothing: the keyboard reads S as reverse and
            # gives it full throttle, so dropping the gear alone would send the
            # car forwards on the key a driver pressed to go back.
            return replace(command, throttle=0.0, brake=1.0, reverse=False)
        return command


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
        self._default_scene = defaults.scene
        self._default_sample = defaults.sample
        self._physics = defaults.physics
        self._reverse = defaults.reverse
        self._bev = defaults.bev
        self._full_size_hud = defaults.full_size_hud
        self._parallel_context = defaults.parallel_context
        self._initial_session_video_size = (defaults.width, defaults.height)
        self._backend_factory: BackendFactory = defaults.backend_factory or partial(
            _build_backend,
            pipeline_config=defaults.pipeline_config,
        )
        # Building the backend during init is what lets a model load while the
        # user is still picking a scene, so it is worth doing whenever there is
        # a model to load. Only the raster-only default has nothing to prepare.
        self._backend_can_prepare = (
            defaults.backend_factory is not None or defaults.pipeline_config is not None
        )
        self._backend: RenderBackend | None = None
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

    def close(self) -> None:
        """Release the application-owned model backend."""
        backend = self._backend
        self._backend = None
        if backend is not None:
            backend.close()

    def init(self, commandline_args: Sequence[str]) -> None:
        parser = argparse.ArgumentParser(prog=f"flashdreams-run-v2 {self._slug} --")
        parser.add_argument(
            "--scene",
            type=Path,
            default=self._default_scene,
            help=(
                "The scene to drive: a local USDZ, or whatever an integration's "
                "own scene loader reads. If omitted and the integration names no "
                "default, download the built-in default scene from Hugging Face."
            ),
        )
        parser.add_argument(
            "--sample",
            type=Path,
            default=self._default_sample,
            help=(
                "A recording of the same drive, for a model that takes its "
                "prompt, framing or opening frames from real footage rather "
                "than from the scene. Ignored by scenes that carry their own."
            ),
        )
        parser.add_argument(
            "--start-frame",
            type=int,
            default=0,
            help="Frame of that recording to open on. Default: %(default)s.",
        )
        parser.add_argument("--prompt")
        parser.add_argument(
            "--camera",
            default="camera_front_wide_120fov",
            help=(
                "Camera to render, or a comma-separated rig of them. Several "
                "cameras are presented as a grid. Default: %(default)s."
            ),
        )
        parser.add_argument(
            "--present-camera",
            help=(
                "Present this camera alone, at its own size, instead of the grid. "
                "It is still conditioned on alongside the rest of the rig."
            ),
        )
        parser.add_argument("--variant", default="default")
        parser.add_argument("--total-blocks", type=int, default=self._default_blocks)
        parser.add_argument("--fps", type=int, default=self._default_fps)
        parser.add_argument("--width", type=int, default=self._default_width)
        parser.add_argument("--height", type=int, default=self._default_height)
        parser.add_argument(
            "--view", choices=view_modes_for(self._physics), default="rgb"
        )
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
            "--postprocess-device",
            default="cuda:0",
            help="CUDA device for the selected postprocessor. Defaults to cuda:0.",
        )
        parser.add_argument("--world-model-device", default="cuda:0")
        parser.add_argument(
            "--raster-device",
            default="cuda:0",
            help="CUDA device for Ludus rasterization. Defaults to cuda:0.",
        )
        parser.add_argument("--world-model-seed", type=int)
        parser.add_argument(
            "--world-model-debug-condition-frame-dir",
            type=Path,
        )
        args = parser.parse_args(list(commandline_args))
        scene = args.scene
        if scene is None:
            scene = download_default_scene()
        # A directory as readily as a file: a scene is whatever the loader in
        # use reads, and a model trained on recorded map data wants a clip of
        # it rather than an archive.
        if not scene.exists():
            raise FileNotFoundError(scene)
        if args.start_frame < 0:
            raise ValueError("--start-frame must be >= 0.")
        if args.total_blocks < 0:
            raise ValueError("--total-blocks must be >= 0 (0 means unbounded).")
        if args.fps <= 0 or args.width <= 0 or args.height <= 0:
            raise ValueError("--fps, --width, and --height must be > 0.")
        chunk = ChunkConfig(fps=args.fps)
        raster = RasterConfig(
            width=args.width,
            height=args.height,
            device=args.raster_device,
        )
        postprocess = VideoPostprocessChainConfig.from_preset(
            preset=args.postprocess_preset,
            device=args.postprocess_device,
        )
        app_config = AppConfig(
            scene_path=scene,
            sample_path=args.sample,
            start_frame=args.start_frame,
            physics=self._physics,
            reverse=self._reverse,
            game_mode=args.game_mode,
            camera_names=_camera_rig(args.camera),
            present_camera=args.present_camera,
            variant=args.variant,
            prompt_override=args.prompt,
            chunk=chunk,
            raster=raster,
            world_model_device=args.world_model_device,
            postprocess_preset=args.postprocess_preset,
            postprocess_device=args.postprocess_device,
            world_model_seed=args.world_model_seed,
            world_model_debug_condition_frame_dir=(
                args.world_model_debug_condition_frame_dir
            ),
            postprocess=postprocess,
            bev=BevConfig(enabled=False),
            vehicle=VehicleConfig(),
        )
        mesh = self._parallel_context
        self._config = InteractiveDriveConfig(
            app=app_config,
            total_blocks=args.total_blocks,
            view_mode=args.view,
            # A drive has one window and one driver however many ranks generate
            # it, so a rank nobody is watching builds no HUD rather than
            # building one and throwing it away. `--no-ui` on rank zero still
            # means what it says.
            no_ui=args.no_ui or (mesh is not None and not mesh.is_main),
        )
        output_spec = postprocess.output_spec(
            VideoSpec(
                width=app_config.raster.width,
                height=app_config.raster.height,
                fps=app_config.chunk.fps,
            )
        )
        # A presented frame is a camera's frame, or a grid of them, so the
        # session reports the grid's size rather than one camera's.
        layout = app_config.presentation
        width = output_spec.width * layout.columns
        height = output_spec.height * layout.rows
        # The driver's own flag rather than `self._config.no_ui`, which is also
        # set for a rank nobody is watching: those ranks present nothing at all,
        # so the size they declare costs nothing, and deriving it per rank would
        # leave the mesh disagreeing about the session for no gain.
        if self._full_size_hud and not args.no_ui:
            width, height = hud_canvas(width, height)
        self._desc = replace(
            self._desc,
            frames_per_second_for_step=app_config.chunk.fps,
            video_width=width,
            video_height=height,
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
        diffusion_model=dict(
            seed=(42 if config.world_model_seed is None else config.world_model_seed)
        ),
    )
    world_model_device = torch.device(config.world_model_device)
    with torch.cuda.device(world_model_device):
        pipeline = resolved_pipeline_config.setup().to(world_model_device).eval()
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


def _output_cuda_device(config: AppConfig) -> str:
    """Keep bypassed and processed frames on the UI presentation GPU."""
    return (
        config.postprocess_device
        if config.postprocess.is_enabled()
        else config.world_model_device
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


def _camera_rig(camera: str) -> tuple[str, ...]:
    """Split a comma-separated rig, keeping the order the caller asked for.

    Rig order is what the model conditions on, so it is the caller's to choose
    rather than ours to sort.
    """
    names = tuple(name.strip() for name in camera.split(",") if name.strip())
    if not names:
        raise ValueError("--camera needs at least one camera name.")
    normalized = [normalize_camera_name(name) for name in names]
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"--camera names the same camera twice: {camera!r}.")
    return names


def _frame_view_tensor(frame: PresentedFrame, view: int, view_mode: ViewMode) -> Tensor:
    """One camera's CHW image for the requested view mode."""
    if view_mode == "physx":
        hdmap = np.asarray(frame.view_rgb_host_uint8[view])
        if view != 0:
            # The collider overlay is drawn from the rig's first camera only,
            # so the rest of the grid shows the conditioning it was drawn over.
            return _chw_tensor(hdmap)
        value = frame.physx_rgb_host_uint8
        if value is None:
            raise ValueError("PhysX view requires a PhysX debug frame.")
        physx = np.asarray(value)
        if physx.shape != hdmap.shape:
            raise ValueError(
                "PhysX and HD-map frames must have matching shapes; "
                f"received {physx.shape} and {hdmap.shape}."
            )
        if physx.ndim != 3 or physx.shape[-1] < 3:
            raise ValueError(f"Expected HWC PhysX frame, received shape {physx.shape}")
        collider_mask = np.any(physx[..., :3] != 0, axis=-1, keepdims=True)
        return _chw_tensor(np.where(collider_mask, physx, hdmap))
    if view_mode == "hdmap":
        return _chw_tensor(np.asarray(frame.view_rgb_host_uint8[view]))
    model = frame.view_model_rgb_host_uint8
    value = model[view] if model else frame.view_rgb_host_uint8[view]
    if isinstance(value, LazyCudaFrame):
        try:
            resident = value.to_cuda_tensor()
        except RuntimeError:
            resident = None
        if torch.is_tensor(resident):
            if resident.ndim != 3 or resident.shape[-1] < 3:
                raise ValueError(
                    f"Expected HWC frame, received shape {tuple(resident.shape)}"
                )
            return resident[..., :3].permute(2, 0, 1)
    return _chw_tensor(np.asarray(value))


def _chw_tensor(array: npt.NDArray[Any]) -> Tensor:
    tensor = torch.from_numpy(np.ascontiguousarray(array))
    if tensor.ndim != 3:
        raise ValueError(f"Expected HWC frame, received shape {tuple(tensor.shape)}")
    return tensor.permute(2, 0, 1)


def _mosaic(cells: Sequence[Tensor], *, columns: int, rows: int) -> Tensor:
    """Tile per-camera CHW images into one CHW image, in rig order.

    Cameras fill the grid left to right and then top to bottom. Cells past the
    rig's end are black rather than an error, because three cameras do not
    divide into a rectangle and the fourth cell being empty is the honest
    picture of that.
    """
    first = cells[0]
    if len(cells) > columns * rows:
        raise ValueError(f"{len(cells)} cameras do not fit a {columns}x{rows} grid.")
    for cell in cells:
        if cell.shape != first.shape:
            raise ValueError(
                "Every camera in the grid must be the same size; received "
                f"{tuple(cell.shape)} alongside {tuple(first.shape)}."
            )
    if len(cells) == 1 and rows * columns == 1:
        # The grid a lone camera makes is the camera. Tiling it anyway would
        # clear and then fill a frame-sized buffer on every step of the drive
        # every integration but one is running.
        return first
    channels, height, width = first.shape
    grid = first.new_zeros((rows * columns, channels, height, width))
    for index, cell in enumerate(cells):
        grid[index] = cell
    return (
        grid.reshape(rows, columns, channels, height, width)
        # -> [C, rows, H, columns, W], so the two spatial merges are adjacent.
        .permute(2, 0, 3, 1, 4)
        .reshape(channels, rows * height, columns * width)
    )


def _frame_chunk_tensor(
    chunk: FrameChunk,
    view_mode: ViewMode,
    *,
    layout: PresentationLayout = PresentationLayout(1, 1, None),
    output_size: tuple[int, int] | None = None,
    output_device: str | torch.device | None = None,
) -> Tensor:
    """Convert an emitted frame chunk to TCHW at the session resolution."""
    frames: list[Tensor] = []
    for frame in chunk.frames:
        if layout.view_index is not None:
            frames.append(_frame_view_tensor(frame, layout.view_index, view_mode))
            continue
        cells = [
            _frame_view_tensor(frame, view, view_mode) for view in range(frame.views)
        ]
        frames.append(_mosaic(cells, columns=layout.columns, rows=layout.rows))
    if not frames:
        if output_size is None:
            raise ValueError("The world-model backend returned an empty frame chunk.")
        return torch.empty((0, 3, *output_size), dtype=torch.uint8)
    output = torch.stack(frames)
    # Moved before resizing rather than after: the HD-map and collider views
    # come off the host, and resizing there would put a chunk of bilinear over
    # the one OMP thread `torchrun` leaves us.
    if output_device:
        output = output.to(device=output_device, non_blocking=True)
    if output_size is not None and output.shape[-2:] != output_size:
        output = (
            F.interpolate(
                output.float(),
                size=output_size,
                mode="bilinear",
                align_corners=False,
            )
            .round_()
            .clamp_(0, 255)
            .to(torch.uint8)
        )
    return output


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
