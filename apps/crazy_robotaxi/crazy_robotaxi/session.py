# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Crazy Robotaxi V2 session with model and ImGui UI threads."""

from __future__ import annotations

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
    CrazyRobotaxiImGUIThread,
    TaxiHudState,
    build_hud_frames,
)
from crazy_robotaxi.world_overlay import render_waypoint_layers
from flashdreams.api_v2.session import ISession
from flashdreams.api_v2.thread import IThread, UIThread, invoke_async
from flashdreams.runtime_v2.session_desc import SessionDesc
from flashdreams.runtime_v2.step_result import StepResult
from flashdreams.runtime_v2.user_input_events import UserInputEvents
from flashdreams.runtime_v2.video_tensor import VideoTensorLayout

if TYPE_CHECKING:
    from crazy_robotaxi.application import ApplicationConfig


@dataclass(slots=True)
class ModelState:
    """All mutable state owned by the one V2 model thread."""

    pipeline: Any
    scene: SceneDefinition
    config: ApplicationConfig
    session_desc: SessionDesc
    driver_input: DriverInput
    ui_thread: UIThread[TaxiHudState]
    """UI-thread endpoint used only through ``invoke_async``."""

    rollout: WorldModelRollout | None = None
    last_video: torch.Tensor | None = None
    last_bev: torch.Tensor | None = None
    last_pose: np.ndarray | None = None
    blocks_generated: int = 0
    finished: bool = False

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
        self.last_video = None
        self.last_bev = None
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


class CrazyRobotaxiModelThread(IThread[ModelState]):
    """Run simulation, rules, conditioning, and generation in one V2 step."""

    def step(self, step_index: int, events: UserInputEvents) -> list[StepResult]:
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
            metrics = dict(generated.metrics)
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

        waypoint_layers = render_waypoint_layers(
            game_frames,
            poses,
            state.scene.selected_camera,
            width=state.session_desc.video_width,
            height=state.session_desc.video_height,
            device=video.device,
            dtype=video.dtype if video.is_floating_point() else torch.float32,
        )
        hud_frames = build_hud_frames(video, game_frames)
        invoke_async(
            state.ui_thread,
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
                output=waypoint_layers,
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
        self.state.close()


class CrazyRobotaxiSession(ISession):
    """Register the model thread and Crazy Robotaxi ImGui UI thread."""

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
        )
        ui_thread = self.register_ui_thread(
            CrazyRobotaxiImGUIThread,
            state=hud_state,
            width=self._session_desc.video_width,
            height=self._session_desc.video_height,
        )
        model_thread = self.register_model_thread(
            CrazyRobotaxiModelThread,
            state=ModelState(
                pipeline=self._pipeline,
                scene=self._scene,
                config=self._config,
                session_desc=self._session_desc,
                driver_input=DriverInput(self._config.driver_input),
                ui_thread=ui_thread,
            ),
        )
        hud_state.model_thread = model_thread
