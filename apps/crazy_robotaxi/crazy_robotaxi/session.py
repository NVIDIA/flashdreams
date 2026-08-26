# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Crazy Robotaxi V2 session with model and SlangPy UI loops."""

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
from omnidreams_game_engine.types import SceneDefinition

from crazy_robotaxi.factory import build_taxi_engine
from crazy_robotaxi.rules import TaxiGameSnapshot
from crazy_robotaxi.ui import (
    CrazyRobotaxiSlangPyUILoop,
    TaxiHudState,
    build_hud_frames,
)
from crazy_robotaxi.world_overlay import render_bev_overlay
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
    last_bev_overlay: torch.Tensor | None = None
    last_pose: np.ndarray | None = None
    blocks_generated: int = 0
    finished: bool = False
    realtime_miss_count: int = 0

    def ensure_rollout(self) -> WorldModelRollout:
        """Build renderer, PhysX, game, and cache on the model thread."""
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
        return self.rollout

    def reset(self) -> None:
        self.driver_input.reset()
        self.blocks_generated = 0
        self.finished = False
        self.realtime_miss_count = 0
        self.last_video = None
        self.last_bev_overlay = None
        self.last_pose = None
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
        step_wall_started = time.perf_counter()
        step_cpu_started = time.thread_time()
        state = self.state
        rollout = state.ensure_rollout()
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
            bev_cpu_started = time.thread_time()
            bev_overlay = (
                None
                if bev is None
                else render_bev_overlay(
                    bev,
                    width=state.session_desc.video_width,
                    height=state.session_desc.video_height,
                )
            )
            bev_overlay_cpu_ms = (time.thread_time() - bev_cpu_started) * 1000.0
            metrics = dict(generated.metrics)
            state.last_video = video[-1:].detach()
            state.last_bev_overlay = (
                None if bev_overlay is None else bev_overlay[-1:].detach()
            )
            state.last_pose = poses[-1].copy()
        else:
            if state.last_video is None or state.last_pose is None:
                raise RuntimeError("Terminal game state has no generated frame")
            video = state.last_video
            game_frames = (snapshot,)
            poses = state.last_pose[None, ...]
            bev_overlay = state.last_bev_overlay
            metrics = {}
            bev_overlay_cpu_ms = 0.0

        hud_frames = build_hud_frames(video, game_frames, poses)
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
                    "bev_overlay_cpu_ms": bev_overlay_cpu_ms,
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
        if bev_overlay is not None:
            results.append(
                StepResult(
                    step_index=step_index,
                    output=bev_overlay,
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
    """Register the model loop and Crazy Robotaxi SlangPy UI loop."""

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
        )
        ui_loop = self.register_ui_loop(
            CrazyRobotaxiSlangPyUILoop,
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
