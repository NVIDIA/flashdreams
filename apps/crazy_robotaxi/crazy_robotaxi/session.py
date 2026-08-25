# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Crazy Robotaxi V2 session and sole model-thread implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from crazy_robotaxi.factory import build_taxi_engine
from crazy_robotaxi.hud import render_hud
from crazy_robotaxi.rules import TaxiGameSnapshot
from flashdreams.api_v2.session import ISession
from flashdreams.api_v2.thread import IThread
from flashdreams.runtime_v2.session_desc import SessionDesc
from flashdreams.runtime_v2.step_result import StepResult
from flashdreams.runtime_v2.user_input_events import UserInputEvents
from flashdreams.runtime_v2.video_tensor import VideoTensorLayout
from omnidreams_game_engine.input import DriverInput
from omnidreams_game_engine.math3d import rig_pose_from_vehicle_state
from omnidreams_game_engine.model import WorldModelRollout
from omnidreams_game_engine.types import SceneDefinition

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


class CrazyRobotaxiModelThread(IThread[ModelState]):
    """Run simulation, rules, conditioning, and generation in one V2 step."""

    def step(self, step_index: int, events: UserInputEvents) -> list[StepResult]:
        state = self.state
        rollout = state.ensure_rollout()
        snapshot = rollout.engine.current_game_frame
        if not isinstance(snapshot, TaxiGameSnapshot):
            raise TypeError("Taxi engine returned a non-taxi game frame")
        accepting_text = snapshot.session_state == "awaiting_name"

        if snapshot.session_state == "playing":
            frame_count = rollout.frame_count(state.blocks_generated)
            input_batch = state.driver_input.reduce(
                events,
                frame_count=frame_count,
                frame_interval_s=(
                    1.0 / state.session_desc.frames_per_second_for_step
                ),
                accepting_text=False,
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
            input_batch = state.driver_input.reduce(
                events,
                frame_count=1,
                frame_interval_s=(
                    1.0 / state.session_desc.frames_per_second_for_step
                ),
                accepting_text=accepting_text,
            )
            if accepting_text and input_batch.submitted_text is not None:
                snapshot = rollout.engine.submit_text(input_batch.submitted_text)
            if state.last_video is None or state.last_pose is None:
                raise RuntimeError("Terminal game state has no generated frame")
            video = state.last_video
            game_frames = (snapshot,)
            poses = state.last_pose[None, ...]
            bev = state.last_bev
            metrics = {}

        overlay = render_hud(
            game_frames,
            rig_poses_world=poses,
            calibration=state.scene.selected_camera,
            bev_tchw=bev,
            width=int(video.shape[-1]),
            height=int(video.shape[-2]),
            device=video.device,
            dtype=video.dtype,
            player_name=state.driver_input.text,
        )
        latest = game_frames[-1]
        if isinstance(latest, TaxiGameSnapshot) and latest.session_state == "leaderboard":
            state.finished = True
        if (
            state.config.total_blocks is not None
            and state.blocks_generated >= state.config.total_blocks
        ):
            state.finished = True
        count = int(video.shape[0])
        return [
            StepResult(
                step_index=step_index,
                output=video,
                frame_count=count,
                output_layout=VideoTensorLayout.tchw,
                metrics=metrics,
            ),
            StepResult(
                step_index=step_index,
                output=overlay,
                frame_count=count,
                output_layout=VideoTensorLayout.tchw,
            ),
        ]

    def is_finished(self) -> bool:
        return self.state.finished

    def reset(self) -> None:
        self.state.reset()

    def close(self) -> None:
        self.state.close()


class CrazyRobotaxiSession(ISession):
    """Register one model thread and use V2's compositing UI thread."""

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
        self.register_model_thread(
            CrazyRobotaxiModelThread,
            state=ModelState(
                pipeline=self._pipeline,
                scene=self._scene,
                config=self._config,
                session_desc=self._session_desc,
                driver_input=DriverInput(self._config.driver_input),
            ),
        )
