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
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger
from omnidreams_game_engine.app import InteractiveDriveApp
from omnidreams_game_engine.application import (
    ApplicationChunkUpdate,
    RolloutSpec,
)
from omnidreams_game_engine.backends.base import RenderBackend
from omnidreams_game_engine.config import AppConfig
from omnidreams_game_engine.game_map.vicinity import GameMapVicinityResolver
from omnidreams_game_engine.simulation.ground_snap import GroundSnapper
from omnidreams_game_engine.types import (
    SceneBundle,
    TrajectoryChunk,
    VehicleState,
)

from crazy_robotaxi.driving import (
    integrate_taxi_vehicle,
)
from crazy_robotaxi.frame_alignment import (
    CausalFrameAlignmentPresenter,
)
from crazy_robotaxi.game import (
    TaxiGameConfig,
    TaxiGameController,
)
from crazy_robotaxi.game_settings import load_game_settings
from crazy_robotaxi.high_scores import (
    default_high_scores_path,
)
from crazy_robotaxi.input import (
    CrazyRobotaxiKeyboardState,
)
from crazy_robotaxi.live_edit.coin_ability import CoinAbility
from crazy_robotaxi.live_edit.config import LiveEditConfig
from crazy_robotaxi.navigation import NavigationLane
from crazy_robotaxi.passengers import (
    build_pickup_passenger_trajectories,
)
from crazy_robotaxi.physics import (
    TaxiPhysicsWorld,
    step_taxi_physics_world,
)
from crazy_robotaxi.scene import (
    load_scene_data,
)
from flashdreams.serving.realtime.timing import TraceSink


class CrazyRobotaxiRuntime:
    """Game session advanced alongside one simulated rollout."""

    def __init__(
        self,
        controller: TaxiGameController,
        keyboard: CrazyRobotaxiKeyboardState,
        *,
        style_ability: Any | None = None,
        coin_ability: CoinAbility | None = None,
        obstacle_ability: Any | None = None,
        item_ability: Any | None = None,
        item_effects: Any | None = None,
    ) -> None:
        self._controller = controller
        self._keyboard = keyboard
        self._style_ability = style_ability
        self._coin_ability = coin_ability
        self._obstacle_ability = obstacle_ability
        self._item_ability = item_ability
        self._item_effects = item_effects

    @property
    def is_running(self) -> bool:
        """Return whether the game is still accepting simulation chunks."""
        return self._controller.is_playing

    def process_events(self, state: VehicleState) -> None:
        """Drain live-edit key requests and a pending high-score name."""
        self._process_live_edit_requests()
        submitted_name = self._keyboard.consume_taxi_name_submission()
        if submitted_name is None:
            return
        try:
            self._controller.submit_high_score_name(submitted_name)
        except (RuntimeError, ValueError) as exc:
            logger.warning(f"[crazy-robotaxi] ignored high-score submission: {exc}")
        self.publish_boundary(state)

    def _process_live_edit_requests(self) -> None:
        """Consume rising-edge skin-cycle / coins-toggle key requests."""
        requests = getattr(self._keyboard, "live_edit", None)
        if requests is None:
            return
        if requests.consume_skin_cycle() and self._style_ability is not None:
            self._style_ability.request_cycle()
        if requests.consume_weather_cycle() and self._style_ability is not None:
            self._style_ability.request_weather_cycle()
        if requests.consume_obstacle_spawn() and self._obstacle_ability is not None:
            self._obstacle_ability.request_spawn()
        if requests.consume_coins_toggle() and self._coin_ability is not None:
            enabled = self._coin_ability.toggle()
            logger.info(f"[live-edit] coins {'on' if enabled else 'off'}")

    def advance_frames(
        self, trajectory: TrajectoryChunk, frame_interval_s: float
    ) -> ApplicationChunkUpdate:
        """Advance the game and add passengers synchronized to pickup state."""
        snapshots = tuple(self._controller.advance_frames(trajectory, frame_interval_s))
        if self._coin_ability is not None:
            picked = self._coin_ability.advance_frames(trajectory.vehicle_states)
            if picked:
                logger.info(
                    f"[live-edit] coin pickup +{picked} "
                    f"total={self._coin_ability.collected_count}"
                )
        if self._item_ability is not None and self._item_ability.enabled:
            # Pickup-driven effects: each pickup queues its effect on the
            # ability state machines, which land it at the next chunk
            # boundary — the same path the K/V key requests take.
            for item_type in self._item_ability.advance_frames(
                trajectory.vehicle_states
            ):
                label = (
                    self._item_effects.apply(item_type)
                    if self._item_effects is not None
                    else f"{item_type.upper()}!"
                )
                self._item_ability.flash(label)
                logger.info(f"[live-edit] item pickup {item_type} -> {label}")
        passengers = build_pickup_passenger_trajectories(
            snapshots, trajectory.timestamps_us
        )
        obstacles: tuple[Any, ...] = ()
        if self._obstacle_ability is not None:
            obstacles = self._obstacle_ability.advance_frames(trajectory)
        return ApplicationChunkUpdate(
            trajectory=replace(
                trajectory,
                dynamic_actors=(*trajectory.dynamic_actors, *passengers, *obstacles),
            ),
            frame_application_states=snapshots,
        )

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
        *,
        live_edit: LiveEditConfig | None = None,
        style_ability: Any | None = None,
    ) -> None:
        self._config = config
        self._keyboard = keyboard
        self._presenter_config = presenter_config
        self._reference_route_world: Any | None = None
        self._navigation_lanes: tuple[Any, ...] = ()
        self._fare_regions: tuple[Any, ...] = ()
        self._vicinity_resolver: GameMapVicinityResolver | None = None
        self._ground_snapper: GroundSnapper | None = None
        self._curb_segments_world = np.empty((0, 2, 3), dtype=np.float32)
        self._live_edit = live_edit or LiveEditConfig()
        self._style_ability = style_ability
        self._live_edit_presenter: Any | None = None
        self._coin_lanes: tuple[NavigationLane, ...] = ()
        self._nitro_ability: Any | None = None
        self._obstacle_ability: Any | None = None
        if self._live_edit.items.enabled:
            from crazy_robotaxi.live_edit.nitro_ability import NitroAbility

            # Application-lifetime (the integrate seam outlives one
            # rollout); reset per rollout in create_runtime.
            self._nitro_ability = NitroAbility(self._live_edit.items)

    def attach_live_edit_presenter(self, presenter: Any) -> None:
        """Bind the live-edit presenter so coin abilities reach the pixels."""
        self._live_edit_presenter = presenter

    def configure_presenter(self, presenter: Any) -> None:
        """Configure application presentation before scene loading."""
        configure = getattr(presenter, "configure_taxi_hud", None)
        if callable(configure):
            configure(self._presenter_config, self._config.vehicle)

    def load_scene(self, scene: SceneBundle) -> None:
        """Accept scene data already loaded by Interactive Drive."""
        scene_data = load_scene_data(scene)
        self._reference_route_world = scene_data.reference_route_world
        self._navigation_lanes = scene_data.navigation_lanes
        self._fare_regions = scene_data.fare_regions
        assert scene.game_map is not None
        self._vicinity_resolver = GameMapVicinityResolver(scene.game_map)
        self._curb_segments_world = scene_data.curb_segments_world
        self._ground_snapper = _build_taxi_ground_snapper(scene, self._config)
        if self._live_edit.coins.enabled or self._live_edit.items.enabled:
            # Lay coins/items along the driving-lane graph; legacy scenes
            # without mapped lanes fall back to the recorded ego route.
            self._coin_lanes = scene_data.navigation_lanes or (
                NavigationLane(
                    centerline_world=np.asarray(
                        scene_data.reference_route_world, dtype=np.float32
                    )
                ),
            )
        logger.info(
            "[crazy-robotaxi] compiled curb segments={}",
            len(scene_data.curb_segments_world),
        )

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
        integrate_fn = integrate_taxi_vehicle
        if self._nitro_ability is not None:
            from crazy_robotaxi.live_edit.nitro_ability import integrate_with_nitro

            # Nitro's physics seam: boost each tick's vehicle config while
            # a pickup is active (instant, no chunk-boundary handshake).
            integrate_fn = integrate_with_nitro(
                self._nitro_ability, integrate_taxi_vehicle
            )
        self._obstacle_ability = None
        if self._live_edit.obstacle.enabled:
            from crazy_robotaxi.live_edit.obstacle_ability import ObstacleAbility

            self._obstacle_ability = ObstacleAbility(
                self._live_edit.obstacle,
                game_map=scene.game_map,
                ground_vertices=scene.ground_mesh_vertices,
                vehicle=self._config.vehicle,
            )
        actor_controllers = (
            (self._obstacle_ability,)
            if self._obstacle_ability is not None and self._live_edit.obstacle.physics
            else ()
        )
        return RolloutSpec(
            vehicle_config=self._config.vehicle,
            initial_speed_mps=0.0,
            integrate_fn=integrate_fn,
            physics_world_factory=lambda active_scene, vehicle: TaxiPhysicsWorld(
                active_scene,
                vehicle,
                curb_segments_world=self._curb_segments_world,
                actor_controllers=actor_controllers,
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
            fare_regions=self._fare_regions,
            initial_state=simulation.current_state,
            config=self._config,
            initial_camera=scene.selected_camera,
            vicinity_resolver=self._vicinity_resolver,
        )
        coin_ability: CoinAbility | None = None
        if self._live_edit.coins.enabled and self._coin_lanes:
            # Rebuilt per rollout so a reset restores the full course.
            coin_ability = CoinAbility.from_lanes(
                self._coin_lanes, self._live_edit.coins
            )
            logger.info(
                f"[live-edit] coin course laid out: {coin_ability.remaining_count} coins"
            )
        item_ability: Any | None = None
        item_effects: Any | None = None
        if self._live_edit.items.enabled and self._coin_lanes:
            from crazy_robotaxi.live_edit.item_ability import (
                ItemAbility,
                ItemEffects,
            )

            # Rebuilt per rollout: a reset restores the course and re-seeds
            # the mystery RNG (reproducible captures).
            item_ability = ItemAbility.from_lanes(
                self._coin_lanes, self._live_edit.items
            )
            item_effects = ItemEffects(
                self._style_ability,
                self._live_edit.items,
                nitro_ability=self._nitro_ability,
            )
        if self._nitro_ability is not None:
            # A rollout reset always starts unboosted.
            self._nitro_ability.reset()
            logger.info(
                f"[live-edit] item course laid out: "
                f"{item_ability.remaining_count} items"
            )
        obstacle_ability = self._obstacle_ability
        if self._live_edit_presenter is not None:
            self._live_edit_presenter.set_coin_ability(coin_ability)
            self._live_edit_presenter.set_obstacle_ability(obstacle_ability)
            self._live_edit_presenter.set_item_ability(item_ability)
            self._live_edit_presenter.set_nitro_ability(self._nitro_ability)
        return CrazyRobotaxiRuntime(
            controller,
            self._keyboard,
            style_ability=self._style_ability,
            coin_ability=coin_ability,
            obstacle_ability=obstacle_ability,
            item_ability=item_ability,
            item_effects=item_effects,
        )


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
        live_edit_config: LiveEditConfig | None = None,
    ) -> None:
        keyboard = CrazyRobotaxiKeyboardState()
        live_edit = live_edit_config or LiveEditConfig()
        style_ability = None
        if live_edit.obstacle.enabled and live_edit.obstacle.guide_scale > 0.0:
            from crazy_robotaxi.live_edit.obstacle_ability import (
                install_obstacle_guidance_on_backend,
            )

            # Before the style install so both share one CUDA-graph-free
            # session swap (marker-coordinated) and one warmup wrap each.
            install_obstacle_guidance_on_backend(backend, live_edit.obstacle)
        if live_edit.style.enabled or live_edit.weather.enabled:
            from crazy_robotaxi.live_edit.style_ability import (
                StyleAbility,
                install_style_ability_on_backend,
            )

            # Must run before super().__init__ starts model warmup on the
            # pipeline worker: the corrector needs a CUDA-graph-free session
            # and the LoRA attach is deferred to the end of warmup_model.
            style_ability = StyleAbility(live_edit.style, live_edit.weather)
            install_style_ability_on_backend(backend, style_ability)
        if alignment_diagnostics_root is not None:
            from crazy_robotaxi.alignment_diagnostics import (
                AlignmentDiagnosticPresenter,
            )

            if presenter is None:
                from omnidreams_game_engine.presenter import SlangPyPresenter

                presenter = SlangPyPresenter(config.raster, keyboard)
            presenter = AlignmentDiagnosticPresenter(
                presenter,
                alignment_diagnostics_root,
            )
            logger.info(
                f"[crazy-robotaxi] alignment diagnostics -> {presenter.output_dir}"
            )
        application = CrazyRobotaxiApplication(
            taxi_config,
            keyboard,
            config.bev,
            live_edit=live_edit,
            style_ability=style_ability,
        )
        super().__init__(
            config=config,
            backend=backend,
            presenter=presenter,
            trace_sink=trace_sink,
            close_presenter_on_exit=close_presenter_on_exit,
            keyboard=keyboard,
            application=application,
        )
        inner_presenter = self._presenter
        if live_edit.any_enabled:
            from crazy_robotaxi.live_edit.presenter import LiveEditPresenter

            # Inside the alignment wrapper so composites see the
            # frame-synchronized rig_to_world matching the generated RGB.
            inner_presenter = LiveEditPresenter(
                inner_presenter, live_edit, style_ability=style_ability
            )
            application.attach_live_edit_presenter(inner_presenter)
        self._presenter = CausalFrameAlignmentPresenter(inner_presenter)


def taxi_config_from_args(args: argparse.Namespace) -> TaxiGameConfig:
    """Build Taxi-only configuration at the application composition root."""
    high_scores_path = (
        args.taxi_highscores.expanduser()
        if args.taxi_highscores is not None
        else default_high_scores_path()
    )
    config = getattr(args, "_game_settings", None) or load_game_settings(
        args.game_config
    )
    return replace(
        config,
        seed=None if args.taxi_seed is None else int(args.taxi_seed),
        high_scores_path=high_scores_path,
        alignment_diagnostics_enabled=(
            getattr(args, "taxi_alignment_diagnostics", None) is not None
        ),
    )


def _build_taxi_ground_snapper(
    scene: SceneBundle, config: TaxiGameConfig
) -> GroundSnapper | None:
    if scene.ground_mesh_vertices is None or scene.ground_mesh_faces is None:
        return None
    return GroundSnapper(
        scene.ground_mesh_vertices,
        scene.ground_mesh_faces,
        max_absolute_rotation_deg=config.ground_snap_max_absolute_rotation_deg,
        invalid_sample_handler=partial(
            settle_invalid_ground_attitude,
            settle_fraction=config.ground_snap_settle_fraction,
        ),
    )


def settle_invalid_ground_attitude(
    state: VehicleState, *, settle_fraction: float = 0.25
) -> VehicleState:
    """Ease stale ground attitude toward level after an invalid Taxi sample."""
    pitch = state.pitch_rad * (1.0 - settle_fraction)
    roll = state.roll_rad * (1.0 - settle_fraction)
    if abs(pitch) < 1.0e-4:
        pitch = 0.0
    if abs(roll) < 1.0e-4:
        roll = 0.0
    return replace(state, pitch_rad=pitch, roll_rad=roll)
