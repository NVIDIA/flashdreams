# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Crazy Robotaxi V2 session with model and Dear ImGui UI loops."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from functools import cached_property
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from omnidreams_game_engine.input import DriverInput
from omnidreams_game_engine.model import WorldModelRollout
from omnidreams_game_engine.scene import SceneRequest
from omnidreams_game_engine.types import DriverCommand, SceneDefinition

from crazy_robotaxi.factory import build_taxi_engine
from crazy_robotaxi.game_selection import GameMapOption, GameSelection
from crazy_robotaxi.race import RaceGameSnapshot
from crazy_robotaxi.rules import TaxiGameSnapshot
from crazy_robotaxi.ui import (
    CrazyRobotaxiImGuiUILoop,
    TaxiHudState,
    build_hud_frames,
)
from flashdreams.api_v2.loop import IModelLoop, IUILoop, invoke_async
from flashdreams.api_v2.session import ISession
from flashdreams.runtime_v2.input_timeline import RealtimeInputTimeline
from flashdreams.runtime_v2.presentation_manager import PresentationManager
from flashdreams.runtime_v2.session_desc import SessionDesc
from flashdreams.runtime_v2.step_result import StepResult
from flashdreams.runtime_v2.user_input_event import (
    GamepadUserInputEvent,
    KeyboardInputState,
    KeyboardUserInputEvent,
)
from flashdreams.runtime_v2.user_input_events import UserInputEvents
from flashdreams.runtime_v2.video_tensor import VideoTensorLayout

if TYPE_CHECKING:
    from crazy_robotaxi.application import ApplicationConfig

_LOGGER = logging.getLogger(__name__)
_TRACE_LOGGER = logging.getLogger("flashdreams.runtime_v2.chunk_trace")
_TRACE_PREFIX = "[crazy-robotaxi-chunk-trace]"
_GAMEPAD_START_BUTTON_INDEX = 9
"""Browser-standard gamepad index shared by Start and Nintendo Plus."""


@dataclass(slots=True)
class ModelState:
    """All mutable state owned by the one V2 model thread."""

    pipeline_factory: Callable[[], Any]
    scene_factory: Callable[[SceneRequest, Any], SceneDefinition]
    """Scene loader invoked after a complete UI selection."""

    config: ApplicationConfig
    session_desc: SessionDesc
    driver_input: DriverInput
    input_timeline: RealtimeInputTimeline = field(init=False)
    """Frame-rate sampling clock for timestamped driving transitions."""

    ui_loop: IUILoop[TaxiHudState]
    """UI-loop endpoint used only through ``invoke_async``."""

    pipeline: Any | None = None
    """Lazily constructed after the client opens and a game is selected."""

    scene: SceneDefinition | None = None
    """Selected immutable scene; ``None`` while the startup menu is active."""

    rollout: WorldModelRollout | None = None
    game_selected: bool = False
    """Whether the UI has supplied a complete mode and map selection."""

    menu_video: torch.Tensor | None = None
    """Cached black model channel published while the menu is active."""
    last_video: torch.Tensor | None = None
    last_bev: torch.Tensor | None = None
    last_pose: np.ndarray | None = None
    last_speed_mps: float = 0.0
    """Speed aligned with the retained terminal presentation frame."""

    blocks_generated: int = 0
    rollout_epoch: int = 0
    """Incremented whenever mutable game and model state is reset."""

    finished: bool = False
    realtime_miss_count: int = 0
    prewarm_complete: bool = False
    """Whether startup AR-shape warmup has completed for this session."""

    prewarm_wall_ms: float = 0.0
    """Wall time spent in hidden startup generation, excluding rollout creation."""

    def __post_init__(self) -> None:
        self.input_timeline = RealtimeInputTimeline(
            samples_per_second=self.session_desc.frames_per_second_for_step,
        )

    def ensure_rollout(self) -> WorldModelRollout:
        """Build and prewarm renderer, PhysX, game, and cache on the model thread."""
        if self.rollout is None:
            scene = self.scene
            if scene is None:
                raise RuntimeError(
                    "Select a game mode and map before starting a rollout"
                )
            if self.pipeline is None:
                self._set_loading_status("LOADING WORLD MODEL")
                self.pipeline = self.pipeline_factory()
            frame_interval_s = 1.0 / self.session_desc.frames_per_second_for_step
            self.rollout = WorldModelRollout(
                pipeline=self.pipeline,
                scene=scene,
                engine_factory=lambda: build_taxi_engine(
                    scene=scene,
                    game_config=self.config.game,
                    raster=self.config.renderer.raster,
                    bev=self.config.renderer.bev,
                    frame_interval_s=frame_interval_s,
                    device=self.config.device,
                    game_mode=self.config.game_mode,
                    race_course_id=self.config.race_course_id,
                    race_times_path=self.config.race_times_path,
                    live_edit=self.config.live_edit,
                ),
                trace_chunk_lifecycle=self.config.profile_input_latency,
            )
        if not self.prewarm_complete:
            self._prewarm_rollout()
        return self.rollout

    def select_game(self, selection: GameSelection) -> None:
        """Load the selected map and configure its rules on the model thread."""
        option = selection.map_option
        if selection.mode == "race":
            if selection.race_course_id not in option.race_course_ids:
                raise ValueError(
                    f"Unknown race course {selection.race_course_id!r} "
                    f"for map {option.map_id!r}"
                )
        elif selection.race_course_id is not None:
            raise ValueError("Taxi mode cannot select a race course")

        self._set_loading_status(f"LOADING {option.name.upper()}")
        request = replace(
            self.config.scene_request,
            map_path=option.path,
            variant=option.variant,
            prompt=(
                self.config.scene_request.prompt
                if option.path
                == self.config.scene_request.map_path.expanduser().resolve()
                else None
            ),
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
        self.rollout_epoch += 1
        if getattr(self.config, "profile_input_latency", False):
            _log_chunk_trace(
                "rollout_reset",
                time_ns=time.monotonic_ns(),
                epoch=self.rollout_epoch,
            )
        self.blocks_generated = 0
        self.finished = False
        self.realtime_miss_count = 0
        self.last_video = None
        self.last_bev = None
        self.last_pose = None
        self.driver_input.reset()
        self.input_timeline.reset()
        self.last_speed_mps = 0.0
        if self.rollout is not None:
            self.rollout.reset()

    def restart_game(self) -> None:
        """Reset the active rollout and its UI state."""
        self.reset()
        invoke_async(self.ui_loop, lambda ui_state: ui_state.reset())

    def close(self) -> None:
        rollout = self.rollout
        self.rollout = None
        if rollout is not None:
            rollout.close()

    def shutdown(self) -> None:
        """Close the active rollout and process-lifetime model pipeline."""
        self.close()
        pipeline = self.pipeline
        self.pipeline = None
        close = getattr(pipeline, "close", None)
        if callable(close):
            close()

    def submit_player_name(self, name: str) -> None:
        """Submit a UI-validated leaderboard name on the model thread."""
        rollout = self.ensure_rollout()
        rollout.engine.submit_text(name)


class CrazyRobotaxiModelLoop(IModelLoop[ModelState]):
    """Run simulation, rules, conditioning, and generation in one V2 step."""

    def step(self, step_index: int, events: UserInputEvents) -> list[StepResult]:
        state = self.state
        runtime_generation = getattr(self, "_generation", 0)
        trace_enabled = getattr(state.config, "profile_input_latency", False)
        # Match Interactive Drive: apply every unread edge before rollout setup,
        # reset handling, or simulation reads the retained command.
        input_times_s = state.driver_input.apply(events)
        if not state.game_selected:
            return state.menu_result(step_index)
        rollout = state.ensure_rollout()
        step_wall_started = time.perf_counter()
        step_cpu_started = time.thread_time()
        snapshot = rollout.engine.current_game_frame
        if not isinstance(snapshot, (TaxiGameSnapshot, RaceGameSnapshot)):
            raise TypeError("Crazy Robotaxi engine returned an unknown game frame")
        if _restart_requested(events):
            state.restart_game()
            rollout = state.ensure_rollout()
            snapshot = rollout.engine.current_game_frame
            if not isinstance(snapshot, (TaxiGameSnapshot, RaceGameSnapshot)):
                raise TypeError("Crazy Robotaxi reset returned an unknown game frame")
        active_states = {"playing", "awaiting_start", "racing"}
        autoregressive_index = -1
        simulation_timestamps_us: tuple[int, ...] | None = None
        cache_finalize_returned_ns: int | None = None
        if snapshot.session_state in active_states:
            live_edit = getattr(rollout.engine, "live_edit", None)
            if live_edit is not None:
                live_edit.process_events(events)
                if live_edit.style is not None:
                    live_edit.style.before_v2_chunk()
            autoregressive_index = state.blocks_generated
            frame_count = rollout.frame_count(autoregressive_index)
            input_window = state.input_timeline.next_window(
                frame_count,
                input_times_s=input_times_s,
            )
            sampled_commands, transition_timestamps_us = state.driver_input.sample(
                input_window
            )
            commands = tuple(
                _taxi_driver_command(command) for command in sampled_commands
            )
            if trace_enabled:
                sampled_at_ns = time.monotonic_ns()
                for frame_index, (command, transition_timestamp_us) in enumerate(
                    zip(commands, transition_timestamps_us, strict=True)
                ):
                    _log_chunk_trace(
                        "input_sampled",
                        time_ns=sampled_at_ns,
                        generation=runtime_generation,
                        step=step_index,
                        epoch=state.rollout_epoch,
                        ar=autoregressive_index,
                        frame=frame_index,
                        event_us=(
                            "none"
                            if transition_timestamp_us is None
                            else transition_timestamp_us
                        ),
                        window_start_us=round(input_window.start_s * 1_000_000),
                        window_end_us=round(input_window.end_s * 1_000_000),
                        throttle=command.throttle,
                        brake=command.brake,
                        steer=command.steer,
                        reverse=command.reverse,
                    )
            generated = rollout.step(
                autoregressive_index=autoregressive_index,
                commands=commands,
            )
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
            game_frames = engine_step.game_frames
            poses = engine_step.trajectory.rig_poses_world
            if trace_enabled:
                simulation_timestamps_us = tuple(
                    int(value) for value in engine_step.trajectory.timestamps_us
                )
            speeds_mps = tuple(
                vehicle.speed_mps for vehicle in engine_step.trajectory.vehicle_states
            )
            bev = engine_step.condition.bev_tchw
            metrics = dict(generated.metrics)
            if state.blocks_generated == 1 and state.prewarm_wall_ms > 0.0:
                metrics["startup_prewarm_wall_ms"] = state.prewarm_wall_ms
                metrics["startup_prewarm_blocks"] = state.config.prewarm_blocks
            state.last_video = video[-1:].detach()
            state.last_bev = None if bev is None else bev[-1:].detach()
            state.last_pose = poses[-1].copy()
            state.last_speed_mps = speeds_mps[-1]
        else:
            if state.last_video is None or state.last_pose is None:
                raise RuntimeError("Terminal game state has no generated frame")
            video = state.last_video
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
        count = int(video.shape[0])
        if snapshot.session_state in active_states:
            model_step_wall_ms = (time.perf_counter() - step_wall_started) * 1000.0
            model_step_cpu_ms = (time.thread_time() - step_cpu_started) * 1000.0
            chunk_duration_ms = (
                count / state.session_desc.frames_per_second_for_step * 1000.0
            )
            metrics.update(
                {
                    "model_step_cpu_ms": model_step_cpu_ms,
                    "chunk_duration_ms": chunk_duration_ms,
                }
            )
            if state.config.pipeline_profiling:
                realtime_margin_ms = chunk_duration_ms - model_step_wall_ms
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
        results = [
            StepResult(
                step_index=step_index,
                output=video,
                frame_count=count,
                output_layout=VideoTensorLayout.tchw,
                metrics=metrics,
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

    def init(self) -> None:
        hud_state = TaxiHudState(
            width=self._session_desc.video_width,
            height=self._session_desc.video_height,
            calibration=None,
            bev=self._config.renderer.bev,
            profile_input_latency=self._config.profile_input_latency,
            show_fps=self._config.show_fps,
            map_options=self._map_options,
            initial_game_mode=self._config.cli_game_mode,
            initial_map_path=self._config.cli_map_path,
            initial_race_course_id=self._config.cli_race_course_id,
        )
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
                config=self._config,
                session_desc=self._session_desc,
                driver_input=DriverInput(),
                ui_loop=ui_loop,
            ),
        )
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


def _restart_requested(events: UserInputEvents) -> bool:
    """Return whether this model step received a keyboard or gamepad restart."""
    for event in events.get_events():
        if (
            isinstance(event, KeyboardUserInputEvent)
            and event.state is KeyboardInputState.PRESSED
            and str(event.key).strip().lower() == "r"
        ):
            return True
        if (
            isinstance(event, GamepadUserInputEvent)
            and event.action == "state"
            and len(event.pressed) > _GAMEPAD_START_BUTTON_INDEX
            and event.pressed[_GAMEPAD_START_BUTTON_INDEX]
        ):
            return True
    return False


def _taxi_driver_command(command: DriverCommand) -> DriverCommand:
    """Apply Taxi's arcade pedal policy to shared non-direct drive commands."""
    if command.steer_is_direct:
        return command
    if command.brake > 0.0:
        return replace(
            command,
            throttle=0.0,
            brake=0.0,
            stop=False,
            handbrake=True,
            manual_control=True,
        )
    if command.reverse:
        command = replace(
            command,
            throttle=0.0,
            brake=command.throttle,
            reverse=False,
        )
    return replace(command, manual_control=True)
