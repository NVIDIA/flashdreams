# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Crazy Robotaxi V2 session with model and Dear ImGui UI loops."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
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
from flashdreams.runtime_v2.session_desc import SessionDesc
from flashdreams.runtime_v2.step_result import StepResult
from flashdreams.runtime_v2.user_input_event import (
    KeyboardInputState,
    KeyboardUserInputEvent,
)
from flashdreams.runtime_v2.user_input_events import UserInputEvents
from flashdreams.runtime_v2.video_tensor import VideoTensorLayout

if TYPE_CHECKING:
    from crazy_robotaxi.application import ApplicationConfig

_LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class ModelState:
    """All mutable state owned by the one V2 model thread."""

    pipeline: Any
    scene_factory: Callable[[SceneRequest, Any], SceneDefinition]
    """Scene loader invoked after a complete UI selection."""

    config: ApplicationConfig
    session_desc: SessionDesc
    driver_input: DriverInput
    ui_loop: IUILoop[TaxiHudState]
    """UI-loop endpoint used only through ``invoke_async``."""

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
    blocks_generated: int = 0
    finished: bool = False
    realtime_miss_count: int = 0
    prewarm_complete: bool = False
    """Whether startup AR-shape warmup has completed for this session."""

    prewarm_wall_ms: float = 0.0
    """Wall time spent in hidden startup generation, excluding rollout creation."""

    input_transition_count: int = 0
    """Cumulative resolved drive transitions consumed by model steps."""

    input_ignored_event_count: int = 0
    """Cumulative redundant drive events consumed by model steps."""

    input_dropped_transition_count: int = 0
    """Cumulative transitions displaced by fixed-size model chunks."""

    def ensure_rollout(self) -> WorldModelRollout:
        """Build and prewarm renderer, PhysX, game, and cache on the model thread."""
        if self.rollout is None:
            scene = self.scene
            if scene is None:
                raise RuntimeError(
                    "Select a game mode and map before starting a rollout"
                )
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
        self.driver_input.reset()
        self.blocks_generated = 0
        self.finished = False
        self.realtime_miss_count = 0
        self.last_video = None
        self.last_bev = None
        self.last_pose = None
        self.input_transition_count = 0
        self.input_ignored_event_count = 0
        self.input_dropped_transition_count = 0
        if self.rollout is not None:
            self.rollout.reset()

    def close(self) -> None:
        rollout = self.rollout
        self.rollout = None
        if rollout is not None:
            rollout.close()

    def submit_player_name(self, name: str) -> None:
        """Submit a UI-validated leaderboard name on the model thread."""
        rollout = self.ensure_rollout()
        rollout.engine.submit_text(name)


class CrazyRobotaxiModelLoop(IModelLoop[ModelState]):
    """Run simulation, rules, conditioning, and generation in one V2 step."""

    def step(self, step_index: int, events: UserInputEvents) -> list[StepResult]:
        state = self.state
        if not state.game_selected:
            return state.menu_result(step_index)
        rollout = state.ensure_rollout()
        step_wall_started = time.perf_counter()
        step_cpu_started = time.thread_time()
        snapshot = rollout.engine.current_game_frame
        if not isinstance(snapshot, (TaxiGameSnapshot, RaceGameSnapshot)):
            raise TypeError("Crazy Robotaxi engine returned an unknown game frame")
        if snapshot.session_state != "awaiting_name" and _restart_requested(events):
            state.reset()
            invoke_async(state.ui_loop, lambda ui_state: ui_state.reset())
            rollout = state.ensure_rollout()
            snapshot = rollout.engine.current_game_frame
            if not isinstance(snapshot, (TaxiGameSnapshot, RaceGameSnapshot)):
                raise TypeError("Crazy Robotaxi reset returned an unknown game frame")
        active_states = {"playing", "awaiting_start", "racing"}
        if snapshot.session_state in active_states:
            live_edit = getattr(rollout.engine, "live_edit", None)
            if live_edit is not None:
                live_edit.process_events(events)
                if live_edit.style is not None:
                    live_edit.style.before_v2_chunk()
            frame_count = rollout.frame_count(state.blocks_generated)
            input_batch = state.driver_input.reduce(
                events,
                frame_count=frame_count,
                frame_interval_s=(1.0 / state.session_desc.frames_per_second_for_step),
            )
            state.input_transition_count += input_batch.transition_count
            state.input_ignored_event_count += input_batch.ignored_event_count
            state.input_dropped_transition_count += input_batch.dropped_transition_count
            generated = rollout.step(
                autoregressive_index=state.blocks_generated,
                commands=input_batch.commands,
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
            bev = engine_step.condition.bev_tchw
            metrics = dict(generated.metrics)
            metrics.update(
                {
                    "input_transition_count": input_batch.transition_count,
                    "input_ignored_event_count": input_batch.ignored_event_count,
                    "input_dropped_transition_count": (
                        input_batch.dropped_transition_count
                    ),
                }
            )
            transition_timestamps_us = input_batch.transition_timestamps_us
            if state.blocks_generated == 1 and state.prewarm_wall_ms > 0.0:
                metrics["startup_prewarm_wall_ms"] = state.prewarm_wall_ms
                metrics["startup_prewarm_blocks"] = state.config.prewarm_blocks
            state.last_video = video[-1:].detach()
            state.last_bev = None if bev is None else bev[-1:].detach()
            state.last_pose = poses[-1].copy()
        else:
            if state.last_video is None or state.last_pose is None:
                raise RuntimeError("Terminal game state has no generated frame")
            video = state.last_video
            game_frames = (snapshot,)
            poses = state.last_pose[None, ...]
            bev = state.last_bev
            metrics = {}
            transition_timestamps_us = (None,) * int(video.shape[0])

        hud_frames = build_hud_frames(
            video,
            game_frames,
            poses,
            transition_timestamps_us=transition_timestamps_us,
            input_transition_count=state.input_transition_count,
            input_ignored_event_count=state.input_ignored_event_count,
            input_dropped_transition_count=state.input_dropped_transition_count,
            ready_event=_record_ready_event(video),
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
        self.state.close()


class CrazyRobotaxiSession(ISession):
    """Register the model loop and Crazy Robotaxi Dear ImGui UI loop."""

    def __init__(
        self,
        *,
        pipeline: Any,
        scene_factory: Callable[[SceneRequest, Any], SceneDefinition],
        map_options: tuple[GameMapOption, ...],
        config: ApplicationConfig,
        session_desc: SessionDesc,
    ) -> None:
        self._pipeline = pipeline
        self._scene_factory = scene_factory
        self._map_options = map_options
        self._config = config
        self._session_desc = session_desc

    @property
    def session_desc(self) -> SessionDesc:
        return self._session_desc

    def init(self) -> None:
        hud_state = TaxiHudState(
            width=self._session_desc.video_width,
            height=self._session_desc.video_height,
            calibration=None,
            bev=self._config.renderer.bev,
            profile_input_latency=self._config.profile_input_latency,
            show_fps=self._config.show_fps,
            map_options=self._map_options,
        )
        ui_loop = self.register_ui_loop(
            CrazyRobotaxiImGuiUILoop,
            state=hud_state,
            width=self._session_desc.video_width,
            height=self._session_desc.video_height,
            presentation_device=self._config.device,
        )
        model_loop = self.register_model_loop(
            CrazyRobotaxiModelLoop,
            state=ModelState(
                pipeline=self._pipeline,
                scene_factory=self._scene_factory,
                config=self._config,
                session_desc=self._session_desc,
                driver_input=DriverInput(self._config.driver_input),
                ui_loop=ui_loop,
            ),
        )
        hud_state.model_loop = model_loop


def _record_ready_event(video: torch.Tensor) -> torch.cuda.Event | None:
    if not video.is_cuda:
        return None
    event = torch.cuda.Event()
    event.record(torch.cuda.current_stream(video.device))
    return event


def _restart_requested(events: UserInputEvents) -> bool:
    """Return whether this model step received a pressed R key."""
    return any(
        isinstance(event, KeyboardUserInputEvent)
        and event.state is KeyboardInputState.PRESSED
        and str(event.key).strip().lower() == "r"
        for event in events.get_events()
    )
