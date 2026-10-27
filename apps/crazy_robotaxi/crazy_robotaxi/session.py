# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Crazy Robotaxi V2 session with model and Dear ImGui UI loops."""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, replace
from functools import cached_property, partial
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import torch
from omnidreams_game_engine.engine import EngineStep, GameEngine
from omnidreams_game_engine.input import DriverInput
from omnidreams_game_engine.model import WorldModelRollout
from omnidreams_game_engine.scene import SceneRequest
from omnidreams_game_engine.simulation.ego_vehicle_kinematics import (
    EgoVehicleKinematics,
)
from omnidreams_game_engine.simulation.ground_snap import GroundSnapper
from omnidreams_game_engine.types import DriverCommand, SceneDefinition

from crazy_robotaxi.controls import (
    BoundActionState,
    gamepad_driver_command,
    keyboard_drive_key,
    keyboard_driver_command,
    wheel_driver_command,
)
from crazy_robotaxi.debug import RobotaxiDebugUILoop, RobotaxiLiveDebugUILoop
from crazy_robotaxi.debug_input import DEBUG_CONTROL_KEY, HeadlessInput
from crazy_robotaxi.factory import build_taxi_engine
from crazy_robotaxi.free_roam import FreeRoamSnapshot
from crazy_robotaxi.game_selection import GameMapOption, GameSelection
from crazy_robotaxi.gameplay import RealtimeGameplay
from crazy_robotaxi.headless_ui import CrazyRobotaxiHeadlessUILoop
from crazy_robotaxi.live_edit.runtime_v2 import LiveEditAction, LiveEditHudStatus
from crazy_robotaxi.race import RaceGameSnapshot
from crazy_robotaxi.rules import TaxiGameSnapshot
from crazy_robotaxi.ui import (
    CrazyRobotaxiImGuiUILoop,
    TaxiHudState,
    build_hud_frames,
)
from flashdreams.api_v2.loop import IModelLoop, IUILoop, invoke_async
from flashdreams.api_v2.session import ISession
from flashdreams.runtime_v2.presentation_manager import PresentationManager
from flashdreams.runtime_v2.session_desc import SessionDesc
from flashdreams.runtime_v2.step_result import StepResult
from flashdreams.runtime_v2.user_input_events import UserInputEvents
from flashdreams.runtime_v2.video_tensor import VideoTensorLayout

if TYPE_CHECKING:
    from crazy_robotaxi.application import ApplicationConfig

_LOGGER = logging.getLogger(__name__)
_TRACE_LOGGER = logging.getLogger("flashdreams.runtime_v2.chunk_trace")
_TRACE_PREFIX = "[crazy-robotaxi-chunk-trace]"


@dataclass(slots=True)
class ModelState:
    """Generation state and the handoff to independently owned gameplay."""

    pipeline_factory: Callable[[], Any]
    scene_factory: Callable[[SceneRequest, Any], SceneDefinition]
    """Scene loader invoked after a complete UI selection."""

    config: ApplicationConfig
    session_desc: SessionDesc
    driver_input: DriverInput

    control_actions: BoundActionState = field(init=False)
    """Rising-edge gameplay actions for the process-start bindings."""

    ui_loop: IUILoop[TaxiHudState]
    """UI-loop endpoint used only through ``invoke_async``."""

    pipeline: Any | None = None
    """Lazily constructed after the client opens and a game is selected."""

    owns_pipeline: bool = True
    """Whether shutdown closes the pipeline instead of its application owner."""

    scene: SceneDefinition | None = None
    """Selected immutable scene; ``None`` while the startup menu is active."""

    rollout: WorldModelRollout | None = None
    game_selected: bool = False
    """Whether the UI has supplied a complete mode and map selection."""

    menu_video: torch.Tensor | None = None
    """Cached black model channel published while the menu is active."""
    last_video: torch.Tensor | None = None
    last_hdmap: torch.Tensor | None = None
    last_bev: torch.Tensor | None = None
    last_pose: np.ndarray | None = None
    last_speed_mps: float = 0.0
    """Speed aligned with the retained terminal presentation frame."""

    blocks_generated: int = 0
    last_debug_record: dict[str, Any] | None = None
    """Source identity retained when a terminal UI repeats the last model frame."""

    debug_frames: int = 0
    """Observed frames simulated in this rollout."""

    debug_ground: GroundSnapper | None = None
    """Ground sampler used to measure physical height before render alignment."""
    rollout_epoch: int = 0
    """Incremented whenever mutable game and model state is reset."""

    finished: bool = False
    realtime_miss_count: int = 0
    prewarm_complete: bool = False
    """Whether startup AR-shape warmup has completed for this session."""

    prewarm_wall_ms: float = 0.0
    """Wall time spent in hidden startup generation, excluding rollout creation."""

    gameplay: RealtimeGameplay | None = None
    generation_snapshot: object | None = None
    shutdown_event: threading.Event = field(default_factory=threading.Event)

    def __post_init__(self) -> None:
        self.control_actions = BoundActionState(self.config.controls)

    def build_engine(self, *, render_conditioning: bool = True) -> GameEngine:
        """Build an independently owned world using the shared immutable scene."""
        assert self.scene is not None
        engine = build_taxi_engine(
            scene=self.scene,
            game_config=self.config.game,
            raster=self.config.renderer.raster,
            bev=self.config.renderer.bev,
            frame_interval_s=1.0 / self.session_desc.frames_per_second_for_step,
            device=self.config.device,
            game_mode=self.config.game_mode,
            race_course_id=self.config.race_course_id,
            race_times_path=self.config.race_times_path,
            live_edit=self.config.live_edit,
            render_conditioning=render_conditioning,
        )
        if self.config.debug_capture:
            cast(EgoVehicleKinematics, engine.simulation).set_physx_debug_enabled(True)
        return engine

    def ensure_rollout(self) -> WorldModelRollout:
        """Build and prewarm renderer, PhysX, game, and cache on the model thread."""
        control = self.session_desc.metadata.get(DEBUG_CONTROL_KEY)
        if isinstance(control, HeadlessInput):
            control.model_ident = threading.get_ident()
            control.native_thread_id = threading.get_native_id()
        capture = getattr(self.config, "debug_capture", False)
        if self.rollout is None:
            scene = self.scene
            if scene is None:
                raise RuntimeError(
                    "Select a game mode and map before starting a rollout"
                )
            if self.pipeline is None:
                self._set_loading_status("LOADING WORLD MODEL")
                self.pipeline = self.pipeline_factory()
            self.rollout = WorldModelRollout(
                pipeline=self.pipeline,
                scene=scene,
                engine_factory=self.build_engine,
                trace_chunk_lifecycle=self.config.profile_input_latency,
            )
            if capture:
                self.debug_ground = None
                if (
                    scene.ground_mesh_vertices is not None
                    and scene.ground_mesh_faces is not None
                ):
                    self.debug_ground = GroundSnapper(
                        scene.ground_mesh_vertices, scene.ground_mesh_faces
                    )
        if not self.prewarm_complete:
            self._prewarm_rollout()
        self.start_gameplay()
        return self.rollout

    def start_gameplay(self) -> None:
        """Hand off mutable simulation after model warmup is complete."""
        assert self.rollout is not None
        control = self.session_desc.metadata.get(DEBUG_CONTROL_KEY)
        if self.gameplay is None:
            self.generation_snapshot = self.rollout.engine.current_game_frame
            # Generation only renders recorded states after warmup. Release its
            # unused native simulation; the worker creates/owns the live one.
            cast(GameEngine, self.rollout.engine).simulation.close()
            self.gameplay = RealtimeGameplay(
                engine_factory=partial(self.build_engine, render_conditioning=False),
                driver=self.driver_input,
                controls=self.config.controls,
                shutdown=self.shutdown_event,
                debug=control if isinstance(control, HeadlessInput) else None,
                rollout_epoch=self.rollout_epoch,
            )

    def select_game(self, selection: GameSelection) -> None:
        """Load the selected map and configure its rules on the model thread."""
        option = selection.map_option
        course = option.race_course(selection.race_course_id)
        if selection.mode == "race":
            if course is None:
                raise ValueError(
                    f"Unknown race course {selection.race_course_id!r} "
                    f"for map {option.map_id!r}"
                )
        elif selection.race_course_id is not None:
            raise ValueError("Only race mode can select a race course")

        self._set_loading_status(f"LOADING {option.name.upper()}")
        request = replace(
            self.config.scene_request,
            map_path=option.path,
            spawn_id=None if course is None else course.spawn_id,
        )
        scene = self.scene_factory(request, self.config.renderer.raster)
        self.close()
        self.config = replace(
            self.config,
            scene_request=request,
            game_mode=selection.mode,
            race_course_id=selection.race_course_id,
        )
        self.scene = scene
        self.game_selected = True
        self.prewarm_complete = False
        self.prewarm_wall_ms = 0.0
        self.reset()
        invoke_async(
            self.ui_loop,
            lambda ui_state, calibration=scene.selected_camera: (
                ui_state.activate_scene(calibration)
            ),
        )
        # Selection messages run outside model-step timing. Finish one-time
        # setup here so it cannot skew the first gameplay throughput sample.
        self.ensure_rollout()

    def menu_result(self, step_index: int) -> list[StepResult]:
        """Return a cached black frame while the UI waits for a menu choice."""
        if self.menu_video is None:
            self.menu_video = torch.full(
                (
                    1,
                    3,
                    self.session_desc.video_height,
                    self.session_desc.video_width,
                ),
                -1.0,
                dtype=torch.float32,
                device=self.config.device,
            )
        return [
            StepResult(
                step_index=step_index,
                output=self.menu_video,
                frame_count=1,
                output_layout=VideoTensorLayout.tchw,
            )
        ]

    def return_to_map_menu(self) -> None:
        """Tear down the active game and resume menu-frame publication."""
        self.close()
        self.scene = None
        self.game_selected = False
        self.prewarm_complete = False
        self.prewarm_wall_ms = 0.0
        self.reset()

    def request_exit(self) -> None:
        """Finish the model loop after the root menu requests exit."""
        self.finished = True

    def _prewarm_rollout(self) -> None:
        rollout = self.rollout
        assert rollout is not None
        block_count = self.config.prewarm_blocks
        if block_count == 0:
            self.prewarm_complete = True
            return

        started = time.perf_counter()
        _LOGGER.info(
            "[crazy-robotaxi] prewarming %d hidden AR blocks before presentation",
            block_count,
        )
        for autoregressive_index in range(block_count):
            current_block = autoregressive_index + 1
            self._set_loading_status(
                f"WARMING WORLD MODEL  {current_block}/{block_count}"
            )
            frame_count = rollout.frame_count(autoregressive_index)
            generated = rollout.step(
                autoregressive_index=autoregressive_index,
                commands=tuple(DriverCommand() for _ in range(frame_count)),
            )
            del generated

        # Retain process-lifetime compiled kernels and autotune results, but
        # discard every gameplay, conditioning, and AR-cache mutation. Cache-
        # bound CUDA graphs re-arm safely against the new storage.
        rollout.reset()
        self.prewarm_wall_ms = (time.perf_counter() - started) * 1000.0
        self.prewarm_complete = True
        self._set_loading_status("STARTING GAME")
        _LOGGER.info(
            "[crazy-robotaxi] prewarm complete in %.1f s; rollout reset for gameplay",
            self.prewarm_wall_ms / 1000.0,
        )

    def _set_loading_status(self, status: str) -> None:
        invoke_async(
            self.ui_loop,
            lambda ui_state, value=status: ui_state.set_loading_status(value),
        )

    def reset(self) -> None:
        if self.gameplay is not None:
            self.gameplay.close()
            self.gameplay = None
        self.generation_snapshot = None
        self.rollout_epoch += 1
        if getattr(self.config, "profile_input_latency", False):
            _log_chunk_trace(
                "rollout_reset",
                time_ns=time.monotonic_ns(),
                epoch=self.rollout_epoch,
            )
        self.blocks_generated = 0
        self.debug_frames = 0
        self.finished = False
        self.realtime_miss_count = 0
        self.last_debug_record = None
        self.last_video = None
        self.last_hdmap = None
        self.last_bev = None
        self.last_pose = None
        self.driver_input.reset()
        self.last_speed_mps = 0.0
        if self.rollout is not None:
            self.rollout.reset()

    def restart_game(self) -> None:
        """Reset the active rollout and its UI state."""
        self.reset()
        invoke_async(self.ui_loop, lambda ui_state: ui_state.reset())

    def close(self) -> None:
        if self.gameplay is not None:
            self.gameplay.close()
            self.gameplay = None
        rollout = self.rollout
        self.rollout = None
        if rollout is not None:
            rollout.close()

    def shutdown(self) -> None:
        """Close the active rollout and process-lifetime model pipeline."""
        self.close()
        pipeline = self.pipeline
        self.pipeline = None
        if self.owns_pipeline:
            close = getattr(pipeline, "close", None)
            if callable(close):
                close()

    def submit_player_name(self, name: str) -> None:
        """Submit a UI-validated leaderboard name on the model thread."""
        self.ensure_rollout()
        assert self.gameplay is not None
        self.gameplay.request("text", name)

    def request_live_edit_action(self, action: LiveEditAction) -> None:
        """Forward one UI action to the active live-edit gameplay state."""
        if self.gameplay is not None:
            self.gameplay.request("action", action)


class CrazyRobotaxiModelLoop(IModelLoop[ModelState]):
    """Run simulation, rules, conditioning, and generation in one V2 step."""

    def step(self, step_index: int, events: UserInputEvents) -> list[StepResult]:
        try:
            return self._step(step_index, events)
        except BaseException:
            control = self.state.session_desc.metadata.get(DEBUG_CONTROL_KEY)
            if isinstance(control, HeadlessInput) and control.capture is not None:
                control.capture.termination_reason = "model_error"
            raise

    def _step(self, step_index: int, events: UserInputEvents) -> list[StepResult]:
        state = self.state
        capture = getattr(state.config, "debug_capture", False)
        control = state.session_desc.metadata.get(DEBUG_CONTROL_KEY)
        runtime_generation = getattr(self, "_generation", 0)
        trace_enabled = getattr(state.config, "profile_input_latency", False)
        control_actions = state.control_actions.apply(events)
        if not state.game_selected:
            return state.menu_result(step_index)
        rollout = state.ensure_rollout()
        step_wall_started = time.perf_counter()
        step_cpu_started = time.thread_time()
        snapshot = state.generation_snapshot
        if (
            snapshot is not None
            and getattr(snapshot, "session_state", None)
            not in {"playing", "awaiting_start", "racing"}
            and state.gameplay is not None
            and state.gameplay.timeline.latest is not None
        ):
            snapshot = state.gameplay.timeline.latest.step.game_frames[0]
        if not isinstance(
            snapshot, (TaxiGameSnapshot, RaceGameSnapshot, FreeRoamSnapshot)
        ):
            raise TypeError("Crazy Robotaxi engine returned an unknown game frame")
        if "restart" in control_actions:
            state.restart_game()
            rollout = state.ensure_rollout()
            snapshot = state.generation_snapshot
            if not isinstance(
                snapshot, (TaxiGameSnapshot, RaceGameSnapshot, FreeRoamSnapshot)
            ):
                raise TypeError("Crazy Robotaxi reset returned an unknown game frame")
        active_states = {"playing", "awaiting_start", "racing"}
        autoregressive_index = -1
        simulation_timestamps_us: tuple[int, ...] | None = None
        cache_finalize_returned_ns: int | None = None
        live_edit_statuses: tuple[LiveEditHudStatus, ...] | None = None
        current_prompt = ""
        debug_records: tuple[dict[str, Any], ...] = ()
        if snapshot.session_state in active_states:
            current_prompt = rollout.scene.prompt
            live_edit = getattr(rollout.engine, "live_edit", None)
            autoregressive_index = state.blocks_generated
            frame_count = rollout.frame_count(autoregressive_index)
            valid_count = frame_count
            gameplay = state.gameplay
            assert gameplay is not None
            window = gameplay.timeline.prepare(
                frame_count,
                self._shutdown_event,
                max_rate=state.config.generation_max_rate,
            )
            if window is None:
                state.finished = True
                return []
            recorded = window.recorded_step()
            commands = recorded.trajectory.applied_commands
            transition_timestamps_us = window.transition_timestamps_us()
            if live_edit is not None:
                live_edit.use_recorded_frames(
                    tuple(window.samples[i].presentation for i in window.indices),
                    recorded.trajectory,
                )

            def observe(engine_step: EngineStep) -> None:
                nonlocal debug_records
                trajectory = engine_step.trajectory
                if len(trajectory.physics_debug_frames) != frame_count:
                    raise RuntimeError(
                        "Debug capture requires per-frame physics observations"
                    )
                records: list[dict[str, Any]] = []
                for index in range(valid_count):
                    physics = trajectory.physics_debug_frames[index]
                    physical = physics.vehicle_state
                    if physical is None:
                        raise RuntimeError("Physics observation has no vehicle state")
                    ground_height = (
                        None
                        if state.debug_ground is None
                        else state.debug_ground.ground_height_at(
                            physical.x_m, physical.y_m, physical.z_m
                        )
                    )
                    records.append(
                        {
                            "frame": state.debug_frames + index,
                            "produced_ns": time.monotonic_ns(),
                            "elapsed_us": control.elapsed_us()
                            if isinstance(control, HeadlessInput)
                            else None,
                            "transition_timestamp_us": transition_timestamps_us[index],
                            "physics_started_ns": physics.physics_started_ns,
                            "physics_completed_ns": physics.physics_completed_ns,
                            "rollout_epoch": state.rollout_epoch,
                            "simulation_timestamp_us": int(
                                trajectory.timestamps_us[index]
                            ),
                            "command": asdict(commands[index]),
                            "vehicle": asdict(trajectory.vehicle_states[index]),
                            "physics_vehicle": asdict(physical),
                            "rig_pose_world": trajectory.rig_poses_world[index].copy(),
                            "ground_height_m": ground_height,
                            "ground_error_m": None
                            if ground_height is None
                            else physical.z_m - ground_height,
                            "actor_collision": physics.actor_collision,
                            "static_barrier_collision": physics.static_barrier_collision,
                            "physics_colliders": asdict(physics),
                            "game": asdict(
                                cast(
                                    TaxiGameSnapshot
                                    | RaceGameSnapshot
                                    | FreeRoamSnapshot,
                                    engine_step.game_frames[index],
                                )
                            ),
                            "model_step": step_index,
                            "autoregressive_index": autoregressive_index,
                            "chunk_frame": index,
                            "gameplay_tick": window.start_tick + window.indices[index],
                        }
                    )
                debug_records = tuple(records)
                state.last_debug_record = records[-1]
                if isinstance(control, HeadlessInput) and control.capture is not None:
                    control.capture.observe(debug_records)
                state.debug_frames += valid_count
                if isinstance(control, HeadlessInput):
                    control.delay_generation(autoregressive_index, self._shutdown_event)

            generated = rollout.step(
                autoregressive_index=autoregressive_index,
                commands=commands,
                observe_engine_step=observe if capture else None,
                recorded_step=recorded,
            )
            if capture and isinstance(control, HeadlessInput):
                control.messages.put(
                    (
                        None,
                        {
                            "status": "generation_returned",
                            "model_step": step_index,
                            "rollout_epoch": state.rollout_epoch,
                            "elapsed_us": control.elapsed_us(),
                        },
                    )
                )
            gameplay.timeline.commit(window)
            state.generation_snapshot = generated.engine.game_frames[-1]
            if trace_enabled and generated._trace is not None:
                trace = generated._trace
                cache_finalize_returned_ns = trace.cache_finalize_returned_ns
                for phase, timestamp_ns in (
                    ("engine_step_started", trace.engine_step_started_ns),
                    ("engine_step_returned", trace.engine_step_returned_ns),
                    ("generate_started", trace.generate_started_ns),
                    ("generate_returned", trace.generate_returned_ns),
                    ("cache_finalize_returned", trace.cache_finalize_returned_ns),
                    ("rollout_step_returned", trace.rollout_step_returned_ns),
                ):
                    _log_chunk_trace(
                        phase,
                        time_ns=timestamp_ns,
                        generation=runtime_generation,
                        step=step_index,
                        epoch=state.rollout_epoch,
                        ar=autoregressive_index,
                        frames=frame_count,
                    )
            if live_edit is not None and live_edit.style is not None:
                live_edit.style.after_v2_chunk()
            if live_edit is not None:
                live_edit_statuses = live_edit.hud_statuses()
                if live_edit.style is not None:
                    current_prompt = live_edit.style.active_prompt or current_prompt
            state.blocks_generated += 1
            video = generated.video_bvtchw[0, 0]
            expected_shape = (
                3,
                state.session_desc.video_height,
                state.session_desc.video_width,
            )
            if tuple(video.shape[1:]) != expected_shape:
                raise ValueError(
                    "Generated video channels and geometry do not match the session: "
                    f"expected {expected_shape}, got {tuple(video.shape[1:])}"
                )
            engine_step = generated.engine
            hdmap = engine_step.condition.hdmap_bvtchw
            expected_hdmap_shape = (1, 1, int(video.shape[0]), *expected_shape)
            if tuple(hdmap.shape) != expected_hdmap_shape:
                raise ValueError(
                    "HD-map conditioning does not match the generated video: "
                    f"expected {expected_hdmap_shape}, got {tuple(hdmap.shape)}"
                )
            hdmap = hdmap[0, 0]
            game_frames = engine_step.game_frames
            poses = engine_step.trajectory.rig_poses_world
            if trace_enabled or capture:
                simulation_timestamps_us = tuple(
                    int(value) for value in engine_step.trajectory.timestamps_us
                )
            speeds_mps = tuple(
                vehicle.speed_mps for vehicle in engine_step.trajectory.vehicle_states
            )
            bev = engine_step.condition.bev_tchw
            if (
                capture
                and isinstance(control, HeadlessInput)
                and control.capture is not None
            ):
                control.capture.submit(debug_records, video, hdmap)
            metrics = dict(generated.metrics)
            metrics.update(
                gameplay_tick=gameplay.timeline.gameplay_tick,
                generation_tick=gameplay.timeline.generation_tick,
                generation_lag_ms=(
                    gameplay.timeline.gameplay_tick - gameplay.timeline.generation_tick
                )
                * 1000
                / 120,
                generation_start_tick=window.start_tick,
                generation_end_tick=window.end_tick,
            )
            if state.blocks_generated == 1 and state.prewarm_wall_ms > 0.0:
                metrics["startup_prewarm_wall_ms"] = state.prewarm_wall_ms
                metrics["startup_prewarm_blocks"] = state.config.prewarm_blocks
            state.last_video = video[-1:].detach()
            state.last_hdmap = hdmap[-1:].detach()
            state.last_bev = None if bev is None else bev[-1:].detach()
            state.last_pose = poses[-1].copy()
            state.last_speed_mps = speeds_mps[-1]
        else:
            if capture and state.config.no_ui:
                if isinstance(control, HeadlessInput) and control.capture is not None:
                    control.capture.termination_reason = "game_ended"
                # Paired observations stop when the game stops generating frames.
                state.finished = True
                return []
            if (
                state.last_video is None
                or state.last_hdmap is None
                or state.last_pose is None
            ):
                raise RuntimeError("Terminal game state has no generated frame")
            if state.last_debug_record is not None:
                debug_records = (state.last_debug_record,)
            video = state.last_video
            hdmap = state.last_hdmap
            game_frames = (snapshot,)
            poses = state.last_pose[None, ...]
            speeds_mps = (state.last_speed_mps,)
            bev = state.last_bev
            metrics = {}
            transition_timestamps_us = (None,) * int(video.shape[0])

        hud_frames = build_hud_frames(
            video,
            game_frames,
            poses,
            speeds_mps=speeds_mps,
            transition_timestamps_us=transition_timestamps_us,
            runtime_generation=runtime_generation,
            model_step_index=step_index,
            rollout_epoch=state.rollout_epoch,
            autoregressive_index=autoregressive_index,
            simulation_timestamps_us=simulation_timestamps_us,
            cache_finalize_returned_ns=cache_finalize_returned_ns,
            live_edit_statuses=live_edit_statuses,
            current_prompt=current_prompt,
        )
        if debug_records:
            hud_frames = tuple(
                replace(frame, debug_record=record)
                for frame, record in zip(hud_frames, debug_records, strict=True)
            )
        invoke_async(
            state.ui_loop,
            lambda ui_state, frames=hud_frames: ui_state.publish(frames),
        )
        if (
            state.config.total_blocks is not None
            and state.blocks_generated >= state.config.total_blocks
        ):
            state.finished = True
            if isinstance(control, HeadlessInput) and control.capture is not None:
                control.capture.termination_reason = "block_limit"
        count = int(video.shape[0])
        if snapshot.session_state in active_states:
            model_step_wall_ms = (time.perf_counter() - step_wall_started) * 1000.0
            model_step_cpu_ms = (time.thread_time() - step_cpu_started) * 1000.0
            chunk_duration_ms = (
                count * 1000 / state.session_desc.frames_per_second_for_step
            )
            realtime_margin_ms = chunk_duration_ms - model_step_wall_ms
            metrics.update(
                {
                    "model_step_cpu_ms": model_step_cpu_ms,
                    "chunk_duration_ms": chunk_duration_ms,
                }
            )
            if state.config.pipeline_profiling or isinstance(control, HeadlessInput):
                metrics["model_step_wall_ms"] = model_step_wall_ms
                metrics["realtime_margin_ms"] = realtime_margin_ms
            physx = engine_step.trajectory.physx_timings
            if physx is not None:
                metrics.update(
                    {
                        "physx_total_ms": physx.total_ms,
                        "physx_synchronize_ms": physx.synchronize_ms,
                        "physx_actor_update_ms": physx.actor_update_ms,
                        "physx_solver_ms": physx.solver_ms,
                        "physx_readback_ms": physx.readback_ms,
                        "physx_bridge_ms": physx.bridge_ms,
                        "physx_traffic_prepare_ms": physx.traffic_prepare_ms,
                        "physx_barrier_rebound_ms": physx.barrier_rebound_ms,
                        "physx_traffic_update_ms": physx.traffic_update_ms,
                        "physx_state_materialize_ms": (physx.state_materialize_ms),
                        "physx_bridge_other_ms": physx.bridge_other_ms,
                    }
                )
            if state.config.pipeline_profiling and realtime_margin_ms < 0.0:
                state.realtime_miss_count += 1
                if (
                    state.realtime_miss_count <= 3
                    or state.realtime_miss_count % 20 == 0
                ):
                    _LOGGER.warning(
                        "[crazy-robotaxi] chunk missed realtime budget: "
                        "step=%d frames=%d overrun_ms=%.1f wall_ms=%.1f "
                        "cpu_ms=%.1f engine_cpu_ms=%.1f",
                        step_index,
                        count,
                        -realtime_margin_ms,
                        model_step_wall_ms,
                        model_step_cpu_ms,
                        float(metrics.get("engine_cpu_ms", 0.0)),
                    )
        if isinstance(control, HeadlessInput):
            control.messages.put(
                (
                    None,
                    {
                        "status": "model_step",
                        "model_step": step_index,
                        "rollout_epoch": state.rollout_epoch,
                        "autoregressive_index": autoregressive_index,
                        "elapsed_us": control.elapsed_us(),
                        "native_thread_id": threading.get_native_id(),
                        "metrics": dict(metrics),
                    },
                )
            )
        results = [
            StepResult(
                step_index=step_index,
                output=video,
                frame_count=count,
                output_layout=VideoTensorLayout.tchw,
                metrics=metrics,
            ),
            StepResult(
                step_index=step_index,
                output=hdmap,
                frame_count=count,
                output_layout=VideoTensorLayout.tchw,
            ),
        ]
        if bev is not None:
            results.append(
                StepResult(
                    step_index=step_index,
                    output=bev,
                    frame_count=count,
                    output_layout=VideoTensorLayout.tchw,
                )
            )
        return results

    def is_finished(self) -> bool:
        return self.state.finished

    def reset(self) -> None:
        self.state.reset()

    def close(self) -> None:
        self.state.shutdown()


class CrazyRobotaxiSession(ISession):
    """Register the model loop and Crazy Robotaxi Dear ImGui UI loop."""

    def __init__(
        self,
        *,
        pipeline_factory: Callable[[], Any],
        scene_factory: Callable[[SceneRequest, Any], SceneDefinition],
        map_options: tuple[GameMapOption, ...],
        config: ApplicationConfig,
        session_desc: SessionDesc,
    ) -> None:
        self._pipeline_factory = pipeline_factory
        self._scene_factory = scene_factory
        self._map_options = map_options
        self._config = config
        self._session_desc = session_desc

    @property
    def session_desc(self) -> SessionDesc:
        return self._session_desc

    @cached_property
    def _presentation_manager(self) -> PresentationManager:
        """Return a frame manager initialized on the game device."""
        return PresentationManager(device=torch.device(self._config.device))

    def receive_input(self, events: UserInputEvents) -> None:
        """Feed driving input directly to gameplay while generation is busy."""
        state = getattr(self, "_game_state", None)
        if state is not None and state.gameplay is not None:
            if not isinstance(
                state.session_desc.metadata.get(DEBUG_CONTROL_KEY), HeadlessInput
            ):
                state.gameplay.submit_input(events)

    def init(self) -> None:
        presentation_width, presentation_height = (
            self._config.presentation_resolution_wh
            or (
                self._session_desc.video_width,
                self._session_desc.video_height,
            )
        )
        hud_state = TaxiHudState(
            width=self._session_desc.video_width,
            height=self._session_desc.video_height,
            calibration=None,
            presentation_size=(presentation_width, presentation_height),
            allow_exit=self._config.allow_exit,
            bev=self._config.renderer.bev,
            debug_control=self._session_desc.metadata.get(DEBUG_CONTROL_KEY),
            profile_input_latency=self._config.profile_input_latency,
            show_fps=self._config.show_fps,
            show_current_prompt=self._config.show_current_prompt,
            hud_enabled=self._config.hud_enabled,
            live_edit=self._config.live_edit,
            native_dit_disabled_for_live_edit=(
                self._config.native_dit_disabled_for_live_edit
            ),
            show_control_tooltips=self._config.show_control_hints,
            show_live_edit_buttons=self._config.show_live_edit_buttons,
            live_edit_mapping_location=self._config.live_edit_mapping_location,
            settings_document=self._config.settings_document,
            controls=self._config.controls,
            gamepad_button_style=self._config.gamepad_button_style,
            control_documents=self._config.control_documents,
            map_options=self._map_options,
            initial_game_mode=self._config.initial_game_mode,
            initial_map_path=self._config.initial_map_path,
            initial_race_course_id=self._config.initial_race_course_id,
        )
        if self._config.debug_capture and self._config.no_ui:
            ui_loop = self.register_ui_loop(RobotaxiDebugUILoop, state=hud_state)
        elif self._config.debug_capture and not self._config.no_ui:
            ui_loop = self.register_ui_loop(
                RobotaxiLiveDebugUILoop,
                state=hud_state,
                width=self._session_desc.video_width,
                height=self._session_desc.video_height,
            )
        elif self._config.no_ui:
            if (
                self._config.initial_game_mode is None
                or self._config.initial_map_path is None
                or self._config.total_blocks is None
            ):
                raise ValueError(
                    "--no-ui requires explicit --game-mode, --map, and --total-blocks"
                )
            if (
                self._config.initial_game_mode == "race"
                and self._config.initial_race_course_id is None
            ):
                # Without one the HUD stops at the course menu, which a
                # headless run has no way to answer: it would publish menu
                # frames forever instead of generating --total-blocks.
                raise ValueError("--no-ui in race mode requires --race-course")
            ui_loop = self.register_ui_loop(
                CrazyRobotaxiHeadlessUILoop,
                state=hud_state,
            )
        else:
            ui_loop = self.register_ui_loop(
                CrazyRobotaxiImGuiUILoop,
                state=hud_state,
                width=self._session_desc.video_width,
                height=self._session_desc.video_height,
            )
        model_loop = self.register_model_loop(
            CrazyRobotaxiModelLoop,
            state=ModelState(
                pipeline_factory=self._pipeline_factory,
                scene_factory=self._scene_factory,
                owns_pipeline=False,
                config=self._config,
                session_desc=self._session_desc,
                driver_input=DriverInput(
                    keyboard_command=partial(
                        keyboard_driver_command, self._config.controls.keyboard
                    ),
                    key_normalizer=partial(
                        keyboard_drive_key, self._config.controls.keyboard
                    ),
                    gamepad_command=partial(
                        gamepad_driver_command, self._config.controls.gamepad
                    ),
                    wheel_command=partial(
                        wheel_driver_command, self._config.controls.wheel
                    ),
                ),
                ui_loop=ui_loop,
                shutdown_event=self._shutdown_event,
            ),
        )
        self._game_state = model_loop.state
        hud_state.model_loop = model_loop
        hud_state.initialize_selection()


def _log_chunk_trace(phase: str, *, time_ns: int, **fields: object) -> None:
    """Emit one grep-friendly chunk lifecycle event."""
    details = " ".join(f"{name}={value}" for name, value in fields.items())
    _TRACE_LOGGER.info(
        "%s phase=%s time_ns=%d %s",
        _TRACE_PREFIX,
        phase,
        time_ns,
        details,
    )
