# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Crazy Robotaxi V2 session with model and Dear ImGui UI loops."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from omnidreams_game_engine.input import DriverInput
from omnidreams_game_engine.math3d import rig_pose_from_vehicle_state
from omnidreams_game_engine.model import WorldModelRollout
from omnidreams_game_engine.types import DriverCommand, SceneDefinition

from crazy_robotaxi.factory import build_taxi_engine
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
from flashdreams.runtime_v2.user_input_events import UserInputEvents
from flashdreams.runtime_v2.video_tensor import VideoTensorLayout

if TYPE_CHECKING:
    from crazy_robotaxi.application import ApplicationConfig

_LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class ModelState:
    """All mutable state owned by the one V2 model thread."""

    pipeline: Any
    scene: SceneDefinition
    config: ApplicationConfig
    session_desc: SessionDesc
    driver_input: DriverInput
    ui_loop: IUILoop[TaxiHudState]
    """UI-loop endpoint used only through ``invoke_async``."""

    rollout: WorldModelRollout | None = None
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
            frame_interval_s = 1.0 / self.session_desc.frames_per_second_for_step
            self.rollout = WorldModelRollout(
                pipeline=self.pipeline,
                scene=self.scene,
                engine_factory=lambda: build_taxi_engine(
                    scene=self.scene,
                    game_config=self.config.game,
                    raster=self.config.renderer.raster,
                    bev=self.config.renderer.bev,
                    frame_interval_s=frame_interval_s,
                    device=self.config.device,
                ),
            )
        if not self.prewarm_complete:
            self._prewarm_rollout()
        return self.rollout

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
        rollout = state.ensure_rollout()
        step_wall_started = time.perf_counter()
        step_cpu_started = time.thread_time()
        snapshot = rollout.engine.current_game_frame
        if not isinstance(snapshot, TaxiGameSnapshot):
            raise TypeError("Taxi engine returned a non-taxi game frame")
        if snapshot.session_state == "playing":
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
        )
        invoke_async(
            state.ui_loop,
            lambda ui_state, frames=hud_frames: ui_state.publish(frames),
        )
        latest = game_frames[-1]
        if (
            isinstance(latest, TaxiGameSnapshot)
            and latest.session_state == "leaderboard"
        ):
            state.finished = True
        if (
            state.config.total_blocks is not None
            and state.blocks_generated >= state.config.total_blocks
        ):
            state.finished = True
        count = int(video.shape[0])
        if snapshot.session_state == "playing":
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
        scene: SceneDefinition,
        config: ApplicationConfig,
        session_desc: SessionDesc,
    ) -> None:
        self._pipeline = pipeline
        self._scene = scene
        self._config = config
        self._session_desc = session_desc

    @property
    def session_desc(self) -> SessionDesc:
        return self._session_desc

    def init(self) -> None:
        hud_state = TaxiHudState(
            width=self._session_desc.video_width,
            height=self._session_desc.video_height,
            calibration=self._scene.selected_camera,
            profile_input_latency=self._config.profile_input_latency,
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
                pipeline=self._pipeline,
                scene=self._scene,
                config=self._config,
                session_desc=self._session_desc,
                driver_input=DriverInput(self._config.driver_input),
                ui_loop=ui_loop,
            ),
        )
        hud_state.model_loop = model_loop
