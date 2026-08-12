# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Crazy Robotaxi application composition for Interactive Drive."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from typing import Any

from loguru import logger
from omnidreams.interactive_drive.app import InteractiveDriveApp
from omnidreams.interactive_drive.application import RolloutSpec
from omnidreams.interactive_drive.backends.base import RenderBackend
from omnidreams.interactive_drive.config import AppConfig
from omnidreams.interactive_drive.crazy_robotaxi.driving import (
    integrate_taxi_vehicle,
)
from omnidreams.interactive_drive.crazy_robotaxi.frame_alignment import (
    CausalFrameAlignmentPresenter,
)
from omnidreams.interactive_drive.crazy_robotaxi.game import (
    TaxiGameConfig,
    TaxiGameController,
)
from omnidreams.interactive_drive.crazy_robotaxi.high_scores import (
    default_high_scores_path,
)
from omnidreams.interactive_drive.crazy_robotaxi.input import (
    CrazyRobotaxiKeyboardState,
)
from omnidreams.interactive_drive.crazy_robotaxi.physics import (
    TaxiPhysicsWorld,
    step_taxi_physics_world,
)
from omnidreams.interactive_drive.crazy_robotaxi.scene import (
    load_scene_data,
)
from omnidreams.interactive_drive.simulation.ground_snap import GroundSnapper
from omnidreams.interactive_drive.simulation.map_bounds import MapBounds
from omnidreams.interactive_drive.types import (
    SceneBundle,
    TrajectoryChunk,
    VehicleState,
)

from flashdreams.serving.realtime.timing import TraceSink


class CrazyRobotaxiRuntime:
    """Game session advanced alongside one simulated rollout."""

    def __init__(
        self,
        controller: TaxiGameController,
        keyboard: CrazyRobotaxiKeyboardState,
    ) -> None:
        self._controller = controller
        self._keyboard = keyboard

    @property
    def is_running(self) -> bool:
        """Return whether the game is still accepting simulation chunks."""
        return self._controller.is_playing

    def process_events(self, state: VehicleState) -> None:
        """Consume a pending high-score name submission."""
        submitted_name = self._keyboard.consume_taxi_name_submission()
        if submitted_name is None:
            return
        try:
            self._controller.submit_high_score_name(submitted_name)
        except (RuntimeError, ValueError) as exc:
            logger.warning(f"[crazy-robotaxi] ignored high-score submission: {exc}")
        self.publish_boundary(state)

    def advance_frames(
        self, trajectory: TrajectoryChunk, frame_interval_s: float
    ) -> tuple[object | None, ...]:
        """Advance the game and return state synchronized to every frame."""
        return tuple(self._controller.advance_frames(trajectory, frame_interval_s))

    def publish_boundary(self, state: VehicleState) -> None:
        """Publish the latest vehicle and game state to presenters."""
        self._keyboard.update_runtime_state(state, self._controller.snapshot(state))


class CrazyRobotaxiApplication:
    """Taxi-specific policy injected into the shared Interactive Drive engine."""

    def __init__(
        self,
        config: TaxiGameConfig,
        keyboard: CrazyRobotaxiKeyboardState,
        presenter_config: Any,
    ) -> None:
        self._config = config
        self._keyboard = keyboard
        self._presenter_config = presenter_config
        self._reference_route_world: Any | None = None
        self._navigation_lanes: tuple[Any, ...] = ()
        self._ground_snapper: GroundSnapper | None = None
        self._map_bounds: MapBounds | None = None

    def configure_presenter(self, presenter: Any) -> None:
        """Configure application presentation before scene loading."""
        configure = getattr(presenter, "configure_taxi_hud", None)
        if callable(configure):
            configure(self._presenter_config)

    def load_scene(self, scene: SceneBundle, map_bounds: MapBounds | None) -> None:
        """Accept scene data already loaded by Interactive Drive."""
        scene_data = load_scene_data(scene)
        self._reference_route_world = scene_data.reference_route_world
        self._navigation_lanes = scene_data.navigation_lanes
        self._ground_snapper = _build_taxi_ground_snapper(scene)
        self._map_bounds = map_bounds

    def configure_scene_presenter(self, presenter: Any, scene: SceneBundle) -> None:
        """Publish camera calibration to an application-aware presenter."""
        configure = getattr(presenter, "configure_taxi_camera", None)
        if callable(configure):
            configure(scene.selected_camera)

    def rollout_spec(
        self,
        scene: SceneBundle,
        *,
        default_vehicle: Any,
        default_visual_flare_enabled: bool,
    ) -> RolloutSpec:
        """Return Crazy Robotaxi simulation policy for one rollout."""
        del default_vehicle, default_visual_flare_enabled
        return RolloutSpec(
            vehicle_config=self._config.vehicle,
            initial_speed_mps=0.0,
            integrate_fn=integrate_taxi_vehicle,
            physics_world_factory=lambda active_scene, vehicle: TaxiPhysicsWorld(
                active_scene,
                vehicle,
                traffic_density=self._config.traffic_density,
            ),
            physics_step_fn=step_taxi_physics_world,
            visual_flare_enabled=False,
            ground_snapper=self._ground_snapper,
            capture_physics_debug=self._config.alignment_diagnostics_enabled,
            include_initial_state_in_first_chunk=True,
        )

    def create_runtime(
        self, scene: SceneBundle, simulation: Any
    ) -> CrazyRobotaxiRuntime:
        """Create game state for a new rollout."""
        if self._reference_route_world is None:
            raise RuntimeError("Crazy Robotaxi scene data was not loaded")
        controller = TaxiGameController(
            scene_id=scene.scene_id,
            reference_route_world=self._reference_route_world,
            navigation_lanes=self._navigation_lanes,
            initial_state=simulation.current_state,
            config=self._config,
            initial_camera=scene.selected_camera,
            map_bounds=self._map_bounds,
        )
        return CrazyRobotaxiRuntime(controller, self._keyboard)


class CrazyRobotaxiApp(InteractiveDriveApp):
    """Interactive Drive engine configured with Crazy Robotaxi policy."""

    def __init__(
        self,
        config: AppConfig,
        taxi_config: TaxiGameConfig,
        backend: RenderBackend,
        presenter: Any | None = None,
        *,
        alignment_diagnostics_root: Path | None = None,
        trace_sink: TraceSink | None = None,
        close_presenter_on_exit: bool = True,
    ) -> None:
        keyboard = CrazyRobotaxiKeyboardState()
        if alignment_diagnostics_root is not None:
            from omnidreams.interactive_drive.crazy_robotaxi.alignment_diagnostics import (
                AlignmentDiagnosticPresenter,
            )

            if presenter is None:
                from omnidreams.interactive_drive.presenter import SlangPyPresenter

                presenter = SlangPyPresenter(config.raster, keyboard)
            presenter = AlignmentDiagnosticPresenter(
                presenter,
                alignment_diagnostics_root,
            )
            logger.info(
                f"[crazy-robotaxi] alignment diagnostics -> {presenter.output_dir}"
            )
        super().__init__(
            config=config,
            backend=backend,
            presenter=presenter,
            trace_sink=trace_sink,
            close_presenter_on_exit=close_presenter_on_exit,
            keyboard=keyboard,
            application=CrazyRobotaxiApplication(taxi_config, keyboard, config.bev),
        )
        self._presenter = CausalFrameAlignmentPresenter(self._presenter)


def taxi_config_from_args(args: argparse.Namespace) -> TaxiGameConfig:
    """Build Taxi-only configuration at the application composition root."""
    high_scores_path = (
        args.taxi_highscores.expanduser()
        if args.taxi_highscores is not None
        else default_high_scores_path()
    )
    return TaxiGameConfig(
        enabled=True,
        traffic_density=float(args.traffic_density),
        seed=None if args.taxi_seed is None else int(args.taxi_seed),
        high_scores_path=high_scores_path,
        alignment_diagnostics_enabled=(
            getattr(args, "taxi_alignment_diagnostics", None) is not None
        ),
    )


def _build_taxi_ground_snapper(scene: SceneBundle) -> GroundSnapper | None:
    if scene.ground_mesh_vertices is None or scene.ground_mesh_faces is None:
        return None
    return GroundSnapper(
        scene.ground_mesh_vertices,
        scene.ground_mesh_faces,
        max_absolute_rotation_deg=10.0,
        invalid_sample_handler=settle_invalid_ground_attitude,
    )


def settle_invalid_ground_attitude(state: VehicleState) -> VehicleState:
    """Ease stale ground attitude toward level after an invalid Taxi sample."""
    settle_fraction = 0.25
    pitch = state.pitch_rad * (1.0 - settle_fraction)
    roll = state.roll_rad * (1.0 - settle_fraction)
    if abs(pitch) < 1.0e-4:
        pitch = 0.0
    if abs(roll) < 1.0e-4:
        roll = 0.0
    return replace(state, pitch_rad=pitch, roll_rad=roll)
