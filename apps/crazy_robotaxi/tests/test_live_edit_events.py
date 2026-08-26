# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""CPU-only tests for the weather and obstacle live-edit abilities."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from crazy_robotaxi.live_edit.config import (
    LiveEditObstacleConfig,
    LiveEditStyleConfig,
    LiveEditWeatherConfig,
    StyleSkin,
    WeatherPreset,
    add_live_edit_args,
    live_edit_config_from_args,
    weathers_starting_with,
)
from crazy_robotaxi.live_edit.input_hooks import LiveEditRequests
from crazy_robotaxi.live_edit.obstacle_ability import (
    OBSTACLE_ENTITY_PREFIX,
    ObstacleAbility,
    ObstacleGuidance,
    ObstaclePhase,
    ObstacleTemplate,
    build_obstacle_event,
    local_ground_z,
    road_ahead_pose,
)
from crazy_robotaxi.live_edit.style_ability import StyleAbility
from crazy_robotaxi.live_edit.weather_ability import compose_swap_target
from ludus_renderer import BodyState
from omnidreams_game_engine.types import (
    TrajectoryChunk,
    VehicleState,
)

pytestmark = pytest.mark.ci_cpu


## Fakes shared by the weather tests (mirrors test_live_edit._FakeSession)


class _FakeTransformer:
    def __init__(self) -> None:
        self._text_edit_lora: object | None = object()
        self.set_calls: list[object | None] = []

    def set_text_edit_lora(self, edit_lora: object | None) -> None:
        self.set_calls.append(edit_lora)
        self._text_edit_lora = edit_lora


class _FakePipeline:
    def __init__(self, transformer: _FakeTransformer | None = None) -> None:
        self.replace_text_calls: list[tuple] = []
        self.lora_present_during_call: list[bool] = []
        self._transformer = transformer

    def replace_text(self, cache: object, text: list, **kwargs: object) -> None:
        self.replace_text_calls.append((text, kwargs))
        if self._transformer is not None:
            self.lora_present_during_call.append(
                self._transformer._text_edit_lora is not None
            )

    def finalize(self, index: int, cache: object) -> None:
        pass


class _FakeSession:
    def __init__(self, transformer: _FakeTransformer | None = None) -> None:
        self.pipeline = _FakePipeline(transformer)
        self._cache = object()
        self._pending_finalization_index: int | None = None

    def start(self, initial_rgb: object, frames: list, prompt: str) -> list:
        return []

    def continue_generation(self, frames: list) -> list:
        return []

    def replace_prompt(
        self, prompt: str, *, guidance_scale: float = 1.0, guidance_chunks: int = 0
    ) -> None:
        # Mirrors FlashdreamsWorldModelSession.replace_prompt.
        if self._pending_finalization_index is not None:
            self.pipeline.finalize(self._pending_finalization_index, self._cache)
            self._pending_finalization_index = None
        self.pipeline.replace_text(
            self._cache,
            [[prompt]],
            guidance_scale=guidance_scale,
            guidance_chunks=guidance_chunks,
        )


_STYLE = LiveEditStyleConfig(
    enabled=True,
    lora_checkpoint=Path("/dev/null"),
    reswap_interval_chunks=0,
    skins=(StyleSkin("arcade", "arcade prompt"),),
)
_WEATHER = LiveEditWeatherConfig(
    enabled=True,
    weathers=(
        WeatherPreset("rain", "rain prompt"),
        WeatherPreset("snow", "snow prompt"),
    ),
)


def _weather_ability(
    style: LiveEditStyleConfig = _STYLE,
    *,
    transformer: _FakeTransformer | None = None,
    lora_attached: bool = False,
) -> tuple[StyleAbility, _FakeSession]:
    ability = StyleAbility(style, _WEATHER)
    ability._transformer = transformer
    ability._lora_attached = lora_attached
    session = _FakeSession(transformer)
    ability.hook_session(session)
    session.start(None, [], "scene prompt")
    return ability, session


class TestWeatherCycle:
    def test_cycles_clear_rain_snow_clear(self) -> None:
        ability, session = _weather_ability()
        names = [ability.active_weather_name]
        for _ in range(3):
            ability.request_weather_cycle()
            session.continue_generation([])
            names.append(ability.active_weather_name)

        assert names == ["clear", "rain", "snow", "clear"]
        texts = [text for text, _ in session.pipeline.replace_text_calls]
        assert texts == [[["rain prompt"]], [["snow prompt"]], [["scene prompt"]]]

    def test_weather_swap_uses_the_validated_guidance_kwargs(self) -> None:
        ability, session = _weather_ability()
        ability.request_weather_cycle()
        session.continue_generation([])

        _, kwargs = session.pipeline.replace_text_calls[-1]
        assert kwargs["guidance_scale"] == 2.5
        assert kwargs["guidance_chunks"] == 6

    def test_revert_to_clear_lands_guided(self) -> None:
        # Clear is itself a weather transition: a plain swap leaves the
        # precipitation running on KV-history momentum, so the wrap to
        # clear (and the timed auto-revert) uses the guided clear landing.
        ability, session = _weather_ability()
        for _ in range(3):
            ability.request_weather_cycle()
            session.continue_generation([])

        text, kwargs = session.pipeline.replace_text_calls[-1]
        assert text == [["scene prompt"]]
        assert kwargs["guidance_scale"] == _WEATHER.guidance_scale
        assert kwargs["guidance_chunks"] == _WEATHER.clear_guidance_chunks

    def test_weather_only_swap_bypasses_the_edit_lora(self) -> None:
        transformer = _FakeTransformer()
        ability, session = _weather_ability(transformer=transformer, lora_attached=True)
        ability.request_weather_cycle()
        session.continue_generation([])

        # Detached for the replace_text call, restored afterwards.
        assert session.pipeline.lora_present_during_call == [False]
        assert transformer._text_edit_lora is not None

    def test_weather_key_is_rejected_while_a_skin_is_active(self) -> None:
        ability, session = _weather_ability()
        ability.request_cycle()
        session.continue_generation([])  # arcade skin active
        ability.request_weather_cycle()  # base-only ability -> ignored
        session.continue_generation([])

        assert ability.active_skin_name == "arcade"
        assert ability.active_weather_name == "clear"
        assert len(session.pipeline.replace_text_calls) == 1

    def test_weather_key_is_rejected_while_a_skin_is_pending(self) -> None:
        ability, session = _weather_ability()
        ability.request_cycle()  # skin queued but not yet applied
        ability.request_weather_cycle()  # ignored against the pending skin
        session.continue_generation([])

        assert ability.active_skin_name == "arcade"
        assert ability.active_weather_name == "clear"

    def test_skin_activation_clears_an_active_weather(self) -> None:
        ability, session = _weather_ability()
        ability.request_weather_cycle()
        session.continue_generation([])  # rain over the base world
        ability.request_cycle()
        session.continue_generation([])  # skin wins; weather -> clear

        assert ability.active_skin_name == "arcade"
        assert ability.active_weather_name == "clear"
        text, _ = session.pipeline.replace_text_calls[-1]
        assert text == [["arcade prompt"]]

    def test_corrector_gain_follows_the_state(self) -> None:
        weather_config = LiveEditWeatherConfig(
            enabled=True, corrector_gain=0.1, weathers=_WEATHER.weathers
        )
        ability = StyleAbility(_STYLE, weather_config)
        session = _FakeSession()
        ability.hook_session(session)
        session.start(None, [], "scene prompt")
        gains: list[float] = []
        ability._set_corrector_gain = gains.append

        ability.request_weather_cycle()
        session.continue_generation([])  # weather -> reduced gain
        ability.request_cycle()
        session.continue_generation([])  # skin -> full style gain

        assert gains == [0.1, _STYLE.corrector_gain]

    def test_weather_key_is_rejected_while_a_timed_skin_is_active(self) -> None:
        from dataclasses import replace as dc_replace

        ability, session = _weather_ability(dc_replace(_STYLE, skin_duration_chunks=3))
        ability.request_cycle()
        session.continue_generation([])
        assert ability.active_skin_name == "arcade"

        ability.request_weather_cycle()  # mid-power-up V press
        session.continue_generation([])

        assert ability.active_weather_name == "clear"

    def test_weather_is_available_again_after_the_timed_skin_expires(self) -> None:
        from dataclasses import replace as dc_replace

        ability, session = _weather_ability(dc_replace(_STYLE, skin_duration_chunks=2))
        ability.request_cycle()
        session.continue_generation([])
        session.continue_generation([])
        session.continue_generation([])  # auto-revert to base
        assert ability.active_skin_name == "base"

        ability.request_weather_cycle()
        session.continue_generation([])

        assert ability.active_weather_name == "rain"

    def test_timed_skin_activation_still_clears_an_active_weather(self) -> None:
        from dataclasses import replace as dc_replace

        ability, session = _weather_ability(dc_replace(_STYLE, skin_duration_chunks=2))
        ability.request_weather_cycle()
        session.continue_generation([])
        assert ability.active_weather_name == "rain"

        ability.request_cycle()
        session.continue_generation([])
        assert ability.active_skin_name == "arcade"
        assert ability.active_weather_name == "clear"

        session.continue_generation([])
        session.continue_generation([])  # timer expires -> base, still clear
        assert ability.active_skin_name == "base"
        assert ability.active_weather_name == "clear"

    def test_weather_works_without_the_style_ability(self) -> None:
        ability = StyleAbility(LiveEditStyleConfig(), _WEATHER)
        session = _FakeSession()
        ability.hook_session(session)
        session.start(None, [], "scene prompt")
        ability.request_cycle()  # style disabled -> ignored
        ability.request_weather_cycle()
        session.continue_generation([])

        assert ability.active_skin_name == "base"
        assert ability.active_weather_name == "rain"
        text, _ = session.pipeline.replace_text_calls[-1]
        assert text == [["rain prompt"]]

    def test_weather_holds_unguided_without_reswaps(self) -> None:
        # Land-then-release: the skin re-swap interval must NOT re-open the
        # weather guidance window (a same-prompt re-swap has a zero guidance
        # direction and would pay 2x per chunk for nothing).
        ability = StyleAbility(LiveEditStyleConfig(reswap_interval_chunks=2), _WEATHER)
        session = _FakeSession()
        ability.hook_session(session)
        session.start(None, [], "scene prompt")
        ability.request_weather_cycle()
        session.continue_generation([])
        for _ in range(6):
            session.continue_generation([])

        assert len(session.pipeline.replace_text_calls) == 1
        assert ability.active_weather_name == "rain"

    def test_maintenance_pulse_rebases_then_reguides(self) -> None:
        weather = LiveEditWeatherConfig(
            enabled=True,
            maintain_interval_chunks=3,
            maintain_chunks=2,
            weathers=_WEATHER.weathers,
        )
        ability = StyleAbility(LiveEditStyleConfig(), weather)
        session = _FakeSession()
        ability.hook_session(session)
        session.start(None, [], "scene prompt")
        ability.request_weather_cycle()
        session.continue_generation([])  # rain lands (landing window)
        for _ in range(3):
            session.continue_generation([])  # interval reached -> pulse

        calls = session.pipeline.replace_text_calls
        assert len(calls) == 3  # landing + (rebase, pulse)
        rebase_text, rebase_kwargs = calls[1]
        pulse_text, pulse_kwargs = calls[2]
        assert rebase_text == [["scene prompt"]]
        assert rebase_kwargs["guidance_scale"] == 1.0
        assert rebase_kwargs["guidance_chunks"] == 0
        assert pulse_text == [["rain prompt"]]
        assert pulse_kwargs["guidance_scale"] == 2.5
        assert pulse_kwargs["guidance_chunks"] == 2
        assert ability.active_weather_name == "rain"

    def test_zero_maintain_interval_never_pulses(self) -> None:
        ability, session = _weather_ability()
        ability.request_weather_cycle()
        session.continue_generation([])
        for _ in range(30):
            session.continue_generation([])

        assert len(session.pipeline.replace_text_calls) == 1


class TestComposeSwapTarget:
    def test_base_state_is_a_plain_swap(self) -> None:
        target = compose_swap_target(
            base_prompt="base",
            skin=None,
            weather=None,
            style_config=_STYLE,
            weather_config=_WEATHER,
            lora_available=True,
        )
        assert target.prompt == "base"
        assert target.guidance_scale == 1.0
        assert not target.use_lora
        assert target.corrector_gain == 0.0

    def test_weather_only_forces_the_two_prompt_path(self) -> None:
        weather_config = LiveEditWeatherConfig(
            enabled=True, corrector_gain=0.1, weathers=_WEATHER.weathers
        )
        target = compose_swap_target(
            base_prompt="base",
            skin=None,
            weather=weather_config.weathers[0],
            style_config=_STYLE,
            weather_config=weather_config,
            lora_available=True,
        )
        assert target.prompt == "rain prompt"
        assert not target.use_lora
        assert target.corrector_gain == 0.1

    def test_skin_state_uses_the_full_style_gain(self) -> None:
        target = compose_swap_target(
            base_prompt="base",
            skin=_STYLE.skins[0],
            weather=None,
            style_config=_STYLE,
            weather_config=_WEATHER,
            lora_available=True,
        )
        assert target.prompt == "arcade prompt"
        assert target.use_lora
        assert target.corrector_gain == _STYLE.corrector_gain

    def test_skin_plus_weather_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="base-world-only"):
            compose_swap_target(
                base_prompt="base",
                skin=_STYLE.skins[0],
                weather=_WEATHER.weathers[0],
                style_config=_STYLE,
                weather_config=_WEATHER,
                lora_available=True,
            )


## Track-backed obstacle events


_OBSTACLE = LiveEditObstacleConfig(enabled=True, spawn_ahead_m=20.0, active_chunks=3)


def _obstacle_template(
    template_index: int,
    *,
    end_xy: tuple[float, float] = (0.0, 20.0),
    parked: bool = False,
) -> ObstacleTemplate:
    timestamps = np.linspace(0, 5_000_000, 16, dtype=np.int64)
    end = np.asarray((0.0, 0.0) if parked else end_xy, dtype=np.float32)
    translations = np.zeros((16, 3), dtype=np.float32)
    translations[:, :2] = np.linspace(np.zeros(2), end, 16, dtype=np.float32)
    yaw = 0.0 if parked else float(np.arctan2(end[1], end[0]))
    orientation = np.asarray(
        [0.0, 0.0, np.sin(yaw * 0.5), np.cos(yaw * 0.5)], dtype=np.float32
    )
    return ObstacleTemplate(
        template_index=template_index,
        object_type="Car",
        timestamps_us=timestamps,
        translations_local_m=translations,
        orientations_xyzw=np.repeat(orientation[None, :], 16, axis=0),
        dimensions_lwh=np.asarray([4.5, 1.8, 1.5], dtype=np.float32),
        source_ground_offset_m=0.75,
    )


_MOVING_TEMPLATES = (
    _obstacle_template(0, end_xy=(0.0, 20.0)),
    _obstacle_template(1, end_xy=(0.0, -20.0)),
    _obstacle_template(2, end_xy=(1.0, 20.0)),
    _obstacle_template(3, end_xy=(-1.0, -20.0)),
)
_PARKED_TEMPLATES = (_obstacle_template(4, parked=True),)


def _obstacle_ability(config: LiveEditObstacleConfig) -> ObstacleAbility:
    return ObstacleAbility(
        config,
        templates=_MOVING_TEMPLATES,
        parked_templates=_PARKED_TEMPLATES,
    )


def _chunk(start_us: int, *, frames: int = 4, ego_x: float = 0.0) -> TrajectoryChunk:
    from omnidreams_game_engine.math3d import rig_pose_from_vehicle_state

    ts = start_us + np.arange(frames, dtype=np.int64) * 33_333
    states = tuple(
        VehicleState(
            x_m=ego_x, y_m=0.0, z_m=0.0, yaw_rad=0.0, speed_mps=5.0, steer_rad=0.0
        )
        for _ in range(frames)
    )
    poses = np.stack([rig_pose_from_vehicle_state(state) for state in states])
    return TrajectoryChunk(
        timestamps_us=ts,
        rig_poses_world=poses,
        vehicle_states=states,
        boundary_state_after_chunk=states[-1],
    )


class TestTrackBackedObstacleEvents:
    def test_visual_event_uses_pr494_ego_relative_placement_by_default(self) -> None:
        ability = _obstacle_ability(_OBSTACLE)
        ability.request_spawn()

        (actor,) = ability.advance_frames(_chunk(1_000_000))

        assert actor.entity_id.startswith(OBSTACLE_ENTITY_PREFIX)
        assert actor.object_type == "Car"
        assert actor.translations_world[0, 0] == pytest.approx(20.0)
        assert actor.translations_world[0, 1] == pytest.approx(0.0)
        motion = actor.translations_world[-1] - actor.translations_world[0]
        assert abs(float(motion[1])) > abs(float(motion[0]))
        assert len(actor.timestamps_us) == 16
        assert np.diff(actor.timestamps_us) == pytest.approx(
            np.diff(_MOVING_TEMPLATES[0].timestamps_us)
        )
        assert actor.orientations_xyzw == pytest.approx(
            _MOVING_TEMPLATES[0].orientations_xyzw
        )
        assert actor.dimensions_lwh == pytest.approx([4.5, 1.8, 1.5])

    def test_visual_event_despawns_after_its_configured_chunks(self) -> None:
        ability = _obstacle_ability(_OBSTACLE)
        ability.request_spawn()
        assert ability.advance_frames(_chunk(0))
        assert ability.advance_frames(_chunk(133_332))
        assert ability.advance_frames(_chunk(266_664))
        assert not ability.active
        assert ability.advance_frames(_chunk(400_000)) == ()

    def test_visual_proximity_hit_is_logged_once(self) -> None:
        ability = _obstacle_ability(_OBSTACLE)
        ability.request_spawn()
        ability.advance_frames(_chunk(0))
        ability.advance_frames(_chunk(133_332, ego_x=20.0))
        ability.advance_frames(_chunk(266_664, ego_x=20.0))
        assert ability.hit_count == 1

    def test_static_roadblocks_start_in_chunk_zero_and_persist_until_reset(
        self,
    ) -> None:
        config = LiveEditObstacleConfig(
            enabled=True,
            static_count=2,
            static_ahead_m=28.0,
            static_lateral_m=2.8,
        )
        ability = _obstacle_ability(config)

        first = ability.advance_frames(_chunk(0))
        later = ability.advance_frames(_chunk(60_000_000))

        assert len(first) == len(later) == 2
        starts = [actor.translations_world[0] for actor in first]
        assert starts[0][:2] == pytest.approx([28.0, -2.8])
        assert starts[1][:2] == pytest.approx([36.0, 2.8])
        ability.reset()
        assert not ability.active

    def test_burst_count_and_stagger_are_preserved(self) -> None:
        config = LiveEditObstacleConfig(
            enabled=True,
            count=3,
            stagger_chunks=1,
            active_chunks=10,
            spawn_ahead_m=20.0,
        )
        ability = _obstacle_ability(config)
        ability.request_spawn()

        first = ability.advance_frames(_chunk(0))
        second = ability.advance_frames(_chunk(133_332))
        third = ability.advance_frames(_chunk(266_664))

        assert [len(first), len(second), len(third)] == [1, 2, 3]
        first_motion_y = (
            first[0].translations_world[-1, 1] - first[0].translations_world[0, 1]
        )
        second_motion_y = (
            second[1].translations_world[-1, 1] - second[1].translations_world[0, 1]
        )
        assert first_motion_y * second_motion_y < 0.0

    def test_physical_collision_detaches_without_npc_recovery(self) -> None:
        config = LiveEditObstacleConfig(
            enabled=True, physics=True, spawn_ahead_m=20.0, active_chunks=3
        )
        ability = _obstacle_ability(config)
        ability.request_spawn()
        ego = BodyState(
            position_m=np.asarray([0.0, 0.0, 0.8], dtype=np.float32),
            orientation_xyzw=np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
            linear_velocity_mps=np.zeros(3, dtype=np.float32),
            angular_velocity_radps=np.zeros(3, dtype=np.float32),
        )
        ability.prepare_topology(ego)
        targets = ability.prepare_step(ego, 1.0 / 30.0)
        (scene_object,) = ability.active_objects
        assert len(scene_object.timestamps_us) == 16
        assert ability.max_drive_speeds_mps[scene_object.object_id] == pytest.approx(
            4.0
        )
        assert tuple(target.object_id for target in targets) == (
            scene_object.object_id,
        )
        body = BodyState(
            position_m=scene_object.positions_m[0].copy(),
            orientation_xyzw=scene_object.orientations_xyzw[0].copy(),
            linear_velocity_mps=np.zeros(3, dtype=np.float32),
            angular_velocity_radps=np.zeros(3, dtype=np.float32),
        )

        decision = ability.observe_physics(
            scene_object.object_id, struck=True, body=body, dt_s=1.0 / 30.0
        )
        assert decision is not None
        assert not decision.drive_enabled
        assert decision.detached_from_track
        assert ability.events[0].phase is ObstaclePhase.DETACHED
        assert ability.hit_count == 1
        later = ability.observe_physics(
            scene_object.object_id, struck=False, body=body, dt_s=10.0
        )
        assert later == decision
        assert ability.events[0].phase is ObstaclePhase.DETACHED

    def test_road_ahead_walks_the_straightest_successor(self) -> None:
        first = SimpleNamespace(
            lane_id="first",
            centerline_world=np.asarray([[0, 0, 0], [10, 0, 0]], dtype=np.float32),
            successor_ids=("turn", "straight"),
        )
        turn = SimpleNamespace(
            lane_id="turn",
            centerline_world=np.asarray([[10, 0, 0], [10, 10, 0]], dtype=np.float32),
            successor_ids=(),
        )
        straight = SimpleNamespace(
            lane_id="straight",
            centerline_world=np.asarray([[10, 0, 0], [25, 0, 0]], dtype=np.float32),
            successor_ids=(),
        )
        oncoming = SimpleNamespace(
            lane_id="oncoming",
            centerline_world=np.asarray(
                [[25, -0.1, 0], [0, -0.1, 0]], dtype=np.float32
            ),
            successor_ids=(),
        )
        game_map = SimpleNamespace(lanes=(oncoming, first, turn, straight))
        ego = VehicleState(
            x_m=2.0,
            y_m=0.0,
            z_m=0.0,
            yaw_rad=0.0,
            speed_mps=0.0,
            steer_rad=0.0,
        )

        pose = road_ahead_pose(game_map, ego, 15.0)

        assert pose is not None
        position, heading = pose
        assert position[:2] == pytest.approx([17.0, 0.0])
        assert heading == pytest.approx(0.0)

    def test_road_ahead_rotates_the_source_track_across_the_lane(self) -> None:
        lane = SimpleNamespace(
            lane_id="lane",
            centerline_world=np.asarray([[0, 0, 0], [50, 0, 0]], dtype=np.float32),
            successor_ids=(),
        )
        game_map = SimpleNamespace(lanes=(lane,))
        config = LiveEditObstacleConfig(
            enabled=True, spawn_ahead_m=20.0, placement="road-ahead"
        )
        ego = VehicleState(
            x_m=0.0,
            y_m=0.0,
            z_m=0.0,
            yaw_rad=0.0,
            speed_mps=0.0,
            steer_rad=0.0,
        )

        event = build_obstacle_event(
            _obstacle_template(10, end_xy=(20.0, 0.0)),
            ego_state=ego,
            spawn_timestamp_us=0,
            config=config,
            entity_id=f"{OBSTACLE_ENTITY_PREFIX}-road-ahead",
            game_map=game_map,
        )

        motion = event.translations_world[-1] - event.translations_world[0]
        assert motion[:2] == pytest.approx([0.0, 20.0], abs=1.0e-5)
        assert event.translations_world[0, :2] == pytest.approx([20.0, 0.0])

    def test_physical_drive_speed_is_capped_at_fifteen_mph(self) -> None:
        fast = _obstacle_template(10, end_xy=(100.0, 0.0))
        ability = ObstacleAbility(
            LiveEditObstacleConfig(enabled=True, physics=True),
            templates=(fast,),
            parked_templates=(),
        )
        ability.request_spawn()
        ego = BodyState(
            position_m=np.asarray([0.0, 0.0, 0.8], dtype=np.float32),
            orientation_xyzw=np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
            linear_velocity_mps=np.zeros(3, dtype=np.float32),
            angular_velocity_radps=np.zeros(3, dtype=np.float32),
        )

        ability.prepare_topology(ego)

        (scene_object,) = ability.active_objects
        assert ability.max_drive_speeds_mps[scene_object.object_id] == pytest.approx(
            15.0 * 0.44704
        )

    def test_track_backed_event_grounds_the_car_center(self) -> None:
        config = LiveEditObstacleConfig(enabled=True, spawn_ahead_m=20.0)
        ego = VehicleState(
            x_m=0.0,
            y_m=0.0,
            z_m=0.0,
            yaw_rad=0.0,
            speed_mps=0.0,
            steer_rad=0.0,
        )
        event = build_obstacle_event(
            _MOVING_TEMPLATES[0],
            ego_state=ego,
            spawn_timestamp_us=0,
            config=config,
            entity_id=f"{OBSTACLE_ENTITY_PREFIX}-test",
            ground_vertices=np.asarray([[20.0, 0.0, 2.0]], dtype=np.float32),
        )
        assert event.translations_world[0, 2] == pytest.approx(2.75)

    def test_physics_and_placement_flags_round_trip(self) -> None:
        parser = argparse.ArgumentParser()
        add_live_edit_args(parser)
        args = parser.parse_args(
            [
                "--live-edit-obstacle",
                "--live-edit-obstacle-physics",
                "--live-edit-obstacle-placement",
                "road-ahead",
            ]
        )
        config = live_edit_config_from_args(args).obstacle
        assert config.enabled
        assert config.physics
        assert config.placement == "road-ahead"


## Obstacle guidance (model-free seams)


class TestObstacleGuidance:
    def test_alt_frames_strip_only_the_obstacle_actor(self) -> None:
        @dataclass
        class FakeRaster:
            calls: list[tuple] = field(default_factory=list)

            def render_chunk(self, **kwargs: Any) -> Any:
                self.calls.append(kwargs)

                @dataclass
                class Frame:
                    rgb_host_uint8: str

                @dataclass
                class Chunk:
                    frames: tuple

                return Chunk(frames=(Frame("nobox"),))

        ability = _obstacle_ability(_OBSTACLE)
        ability.request_spawn()
        (obstacle_actor,) = ability.advance_frames(_chunk(0))
        chunk = _chunk(0)
        from dataclasses import replace as dc_replace

        trajectory = dc_replace(chunk, dynamic_actors=(obstacle_actor,))

        guidance = ObstacleGuidance(3.0)
        raster = FakeRaster()
        guidance._stash_alt_frames(raster, trajectory)
        assert guidance._alt_frames == ["nobox"]
        assert raster.calls[0]["dynamic_actors"] == ()

        guidance._stash_alt_frames(raster, chunk)  # no obstacle -> no render
        assert guidance._alt_frames is None
        assert len(raster.calls) == 1

    def test_guided_predict_flow_combines_along_the_box_axis(self) -> None:
        class FakeTransformer:
            _finalizing_kv_cache = False

            def predict_flow(self, noisy, timestep, cache, input=None):
                return 10.0 if input == "box" else 4.0

        class FakeSession:
            def __init__(self) -> None:
                from types import SimpleNamespace

                self.pipeline = SimpleNamespace(
                    diffusion_model=SimpleNamespace(transformer=FakeTransformer())
                )

        session = FakeSession()
        transformer = session.pipeline.diffusion_model.transformer
        guidance = ObstacleGuidance(3.0)
        guidance._wrap_predict_flow(session)

        # No alt input -> passthrough.
        assert transformer.predict_flow(None, None, None, input="box") == 10.0
        guidance._alt_input = "nobox"
        # nobox + s * (box - nobox) = 4 + 3 * 6 = 22.
        assert transformer.predict_flow(None, None, None, input="box") == 22.0
        # Finalize forwards pass through untouched.
        transformer._finalizing_kv_cache = True
        assert transformer.predict_flow(None, None, None, input="box") == 10.0

    def test_requires_a_positive_scale(self) -> None:
        with pytest.raises(ValueError):
            ObstacleGuidance(0.0)

    def test_guard_accepts_cuda_graph_transformers(self) -> None:
        from types import SimpleNamespace

        transformer = SimpleNamespace(
            config=SimpleNamespace(use_cuda_graph=True, compile_network=True),
            _optimized_dit_executor=None,
        )
        session = SimpleNamespace(
            pipeline=SimpleNamespace(
                diffusion_model=SimpleNamespace(transformer=transformer)
            )
        )
        ObstacleGuidance._guard_transformer(session)  # must not raise

    def test_guard_rejects_the_native_executor(self) -> None:
        from types import SimpleNamespace

        transformer = SimpleNamespace(_optimized_dit_executor=object())
        session = SimpleNamespace(
            pipeline=SimpleNamespace(
                diffusion_model=SimpleNamespace(transformer=transformer)
            )
        )
        with pytest.raises(RuntimeError, match="native optimized-DiT"):
            ObstacleGuidance._guard_transformer(session)

    def test_eager_vae_scope_toggles_and_restores_the_graph_flag(self) -> None:
        from types import SimpleNamespace

        from crazy_robotaxi.live_edit.obstacle_ability import _eager_vae_scope

        vae = SimpleNamespace(_use_cuda_graph=True)
        encoder = SimpleNamespace(vae=vae)
        with _eager_vae_scope(encoder):
            assert vae._use_cuda_graph is False
        assert vae._use_cuda_graph is True

        # No-op for encoders without the knob (pixel shuffle, fakes).
        with _eager_vae_scope(SimpleNamespace()):
            pass


class TestRequestsAndConfig:
    def test_weather_and_obstacle_requests_are_one_shot(self) -> None:
        requests = LiveEditRequests()
        assert requests.consume_weather_cycle() is False
        requests.request_weather_cycle()
        requests.request_obstacle_spawn()
        assert requests.consume_weather_cycle() is True
        assert requests.consume_weather_cycle() is False
        assert requests.consume_obstacle_spawn() is True
        assert requests.consume_obstacle_spawn() is False

    def test_cli_round_trip(self) -> None:
        parser = argparse.ArgumentParser()
        add_live_edit_args(parser)
        args = parser.parse_args(
            [
                "--live-edit-weather",
                "--live-edit-weather-guidance",
                "3.0",
                "--live-edit-weather-corrector-gain",
                "0.1",
                "--live-edit-obstacle",
                "--live-edit-obstacle-ahead-m",
                "18",
                "--live-edit-obstacle-guide-scale",
                "3.0",
                "--live-edit-obstacle-annotate",
            ]
        )
        config = live_edit_config_from_args(args)
        assert config.weather.enabled
        assert config.weather.guidance_scale == 3.0
        assert config.weather.corrector_gain == 0.1
        assert config.obstacle.enabled
        assert config.obstacle.spawn_ahead_m == 18.0
        assert config.obstacle.guide_scale == 3.0
        assert config.obstacle.annotate
        assert config.any_enabled

    def test_default_weather_cycle_is_rain_snow_storm_hurricane(self) -> None:
        assert [w.name for w in weathers_starting_with(None)] == [
            "rain",
            "snow",
            "storm",
            "hurricane",
        ]

    def test_weather_first_rotates_the_cycle_for_direct_select(self) -> None:
        rotated = weathers_starting_with("snow")
        assert [w.name for w in rotated] == ["snow", "storm", "hurricane", "rain"]
        # Rotation reorders, never rewrites, the presets.
        assert {w.prompt for w in rotated} == {
            w.prompt for w in weathers_starting_with(None)
        }

    def test_hurricane_preset_escalates_storm_along_static_cues(self) -> None:
        by_name = {w.name: w for w in weathers_starting_with(None)}
        prompt = by_name["hurricane"].prompt
        # The cues that materialize: visibility collapse, spray walls,
        # debris lying on the road, black-green sky.
        assert "visibility" in prompt
        assert "walls of" in prompt
        assert "debris litter the flooded road" in prompt
        assert "black-green" in prompt
        # Dynamic wind effects never materialize; keep them out.
        assert "bend" not in prompt
        assert "fly" not in prompt

    def test_weather_first_direct_selects_hurricane(self) -> None:
        rotated = weathers_starting_with("hurricane")
        assert [w.name for w in rotated] == ["hurricane", "rain", "snow", "storm"]

    def test_weather_first_rejects_unknown_presets(self) -> None:
        with pytest.raises(ValueError, match="unknown weather preset"):
            weathers_starting_with("volcano")

    def test_weather_first_flag_reaches_the_config(self) -> None:
        parser = argparse.ArgumentParser()
        add_live_edit_args(parser)
        args = parser.parse_args(
            ["--live-edit-weather", "--live-edit-weather-first", "storm"]
        )
        config = live_edit_config_from_args(args)
        assert config.weather.weathers[0].name == "storm"

    def test_runtime_drains_weather_and_obstacle_requests(self) -> None:
        from crazy_robotaxi.app import CrazyRobotaxiRuntime
        from crazy_robotaxi.input import CrazyRobotaxiKeyboardState

        class FakeStyle:
            weather_cycles = 0

            def request_weather_cycle(self) -> None:
                self.weather_cycles += 1

        class FakeObstacle:
            spawns = 0

            def request_spawn(self) -> None:
                self.spawns += 1

        class FakeController:
            is_playing = True

        keyboard = CrazyRobotaxiKeyboardState()
        style = FakeStyle()
        obstacle = FakeObstacle()
        runtime = CrazyRobotaxiRuntime(
            FakeController(),
            keyboard,
            style_ability=style,
            obstacle_ability=obstacle,
        )
        keyboard.live_edit.request_weather_cycle()
        keyboard.live_edit.request_obstacle_spawn()
        state = VehicleState(
            x_m=0.0, y_m=0.0, z_m=0.0, yaw_rad=0.0, speed_mps=0.0, steer_rad=0.0
        )
        runtime.process_events(state)
        assert style.weather_cycles == 1
        assert obstacle.spawns == 1
        runtime.process_events(state)
        assert style.weather_cycles == 1
        assert obstacle.spawns == 1


## Fused per-state corrector dispatch (Task: real-time corrector in-game)


def _toy_attn_network():
    from torch import nn

    class Block(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.self_attn = nn.ModuleDict(
                {
                    n: nn.Linear(4, 4, bias=False)
                    for n in ("q_proj", "k_proj", "v_proj", "output_proj")
                }
            )
            self.cross_attn = nn.ModuleDict(
                {
                    n: nn.Linear(4, 4, bias=False)
                    for n in ("q_proj", "k_proj", "v_proj", "output_proj")
                }
            )

    return nn.Sequential(Block(), Block())


class _ToyDeployTransformer:
    """Deploy surface used by attach(): network + LoRA hook + finalize."""

    def __init__(self) -> None:
        import torch

        torch.manual_seed(0)
        self.network = _toy_attn_network()
        self._text_edit_lora: object | None = None

    def set_text_edit_lora(self, edit_lora: object | None) -> None:
        self._text_edit_lora = edit_lora

    def finalize_kv_cache(self, *args: object, **kwargs: object) -> None:
        return None


def _fused_session() -> tuple[_FakeSession, _ToyDeployTransformer]:
    from types import SimpleNamespace

    import torch

    transformer = _ToyDeployTransformer()
    session = _FakeSession()
    session.manifest = SimpleNamespace(native_dit_acceleration="disabled")
    session.pipeline.diffusion_model = SimpleNamespace(
        transformer=transformer,
        scheduler=SimpleNamespace(
            denoising_step_list=torch.tensor([1000.0, 803.0]), sample=None
        ),
        config=SimpleNamespace(context_noise=128),
    )
    return session, transformer


def _write_checkpoints(tmp_path, network) -> tuple[Path, Path]:
    import torch
    from omnidreams._drift_corrector import _target_linears as corrector_targets
    from omnidreams._edit_lora import _target_linears as edit_targets

    torch.manual_seed(3)
    lora_sd = {}
    for i, lin in enumerate(edit_targets(network)):
        lora_sd[2 * i] = 0.02 * torch.randn(2, lin.in_features)
        lora_sd[2 * i + 1] = 0.02 * torch.randn(lin.out_features, 2)
    lora_ckpt = tmp_path / "edit_lora.pt"
    torch.save({"lora": lora_sd}, lora_ckpt)

    corr_sd = {}
    for i, lin in enumerate(corrector_targets(network)):
        corr_sd[2 * i] = 0.02 * torch.randn(2, lin.in_features)
        corr_sd[2 * i + 1] = 0.02 * torch.randn(lin.out_features, 2)
    corr_ckpt = tmp_path / "corrector.pt"
    torch.save({"lora": corr_sd}, corr_ckpt)
    return lora_ckpt, corr_ckpt


def _fused_ability(
    tmp_path, *, weather_gain: float = 0.1, skin_duration_chunks: int = 0
) -> tuple[StyleAbility, _FakeSession, _ToyDeployTransformer]:
    session, transformer = _fused_session()
    lora_ckpt, corr_ckpt = _write_checkpoints(tmp_path, transformer.network)
    style = LiveEditStyleConfig(
        enabled=True,
        lora_checkpoint=lora_ckpt,
        corrector_checkpoint=corr_ckpt,
        corrector_mode="fused",
        reswap_interval_chunks=0,
        skin_duration_chunks=skin_duration_chunks,
        skins=(StyleSkin("arcade", "arcade prompt"),),
    )
    weather = LiveEditWeatherConfig(
        enabled=True,
        corrector_gain=weather_gain,
        weathers=(
            WeatherPreset("rain", "rain prompt"),
            WeatherPreset("snow", "snow prompt"),
        ),
    )
    ability = StyleAbility(style, weather)
    ability.attach(session)
    return ability, session, transformer


class TestFusedCorrectorDispatch:
    def test_attach_registers_base_skin_and_weather_states(self, tmp_path) -> None:
        ability, _, _ = _fused_ability(tmp_path)
        assert ability._dispatch is not None
        assert ability._corrector_states == {"base", "skin", "weather"}
        assert ability._dispatch.active_state == "base"

    def test_edit_lora_releases_the_self_attention_projections(self, tmp_path) -> None:
        import torch
        from omnidreams._drift_corrector import _target_linears as corrector_targets
        from omnidreams._edit_lora import _target_linears as edit_targets

        _ability, _, transformer = _fused_ability(tmp_path)
        edit_lora = transformer._text_edit_lora
        assert edit_lora is not None
        self_attn = corrector_targets(transformer.network)
        cross_attn = [
            lin for lin in edit_targets(transformer.network) if lin not in self_attn
        ]
        before_self = [lin.weight.detach().clone() for lin in self_attn]
        before_cross = [lin.weight.detach().clone() for lin in cross_attn]
        edit_lora.set_active(True)
        for lin, w in zip(self_attn, before_self):
            assert torch.equal(lin.weight, w)  # dispatch owns these now
        assert any(
            not torch.equal(lin.weight, w) for lin, w in zip(cross_attn, before_cross)
        )
        edit_lora.set_active(False)

    def test_state_machine_selects_the_dispatch_state_at_boundaries(
        self, tmp_path
    ) -> None:
        ability, session, _ = _fused_ability(tmp_path)
        session.start(None, [], "scene prompt")
        assert ability._dispatch.active_state == "base"

        ability.request_cycle()
        session.continue_generation([])
        assert ability._dispatch.active_state == "skin"

        ability.request_cycle()  # single skin -> back to base
        session.continue_generation([])
        assert ability._dispatch.active_state == "base"

        ability.request_weather_cycle()
        session.continue_generation([])
        assert ability._dispatch.active_state == "weather"
        ability.request_weather_cycle()  # rain -> snow stays weather
        session.continue_generation([])
        assert ability._dispatch.active_state == "weather"
        ability.request_weather_cycle()  # snow -> clear
        session.continue_generation([])
        assert ability._dispatch.active_state == "base"

    def test_skin_activation_moves_weather_state_back_through_skin(
        self, tmp_path
    ) -> None:
        ability, session, _ = _fused_ability(tmp_path)
        session.start(None, [], "scene prompt")
        ability.request_weather_cycle()
        session.continue_generation([])
        assert ability._dispatch.active_state == "weather"
        ability.request_cycle()  # skin clears weather (base-only rule)
        session.continue_generation([])
        assert ability._dispatch.active_state == "skin"

    def test_unregistered_weather_state_falls_back_to_base(self, tmp_path) -> None:
        ability, session, _ = _fused_ability(tmp_path, weather_gain=0.0)
        assert "weather" not in ability._corrector_states
        session.start(None, [], "scene prompt")
        ability.request_weather_cycle()
        session.continue_generation([])
        assert ability._dispatch.active_state == "base"

    def test_timed_skin_auto_revert_returns_the_dispatch_to_base(
        self, tmp_path
    ) -> None:
        ability, session, _ = _fused_ability(tmp_path, skin_duration_chunks=2)
        session.start(None, [], "scene prompt")
        ability.request_cycle()
        session.continue_generation([])
        assert ability._dispatch.active_state == "skin"
        session.continue_generation([])
        assert ability._dispatch.active_state == "skin"

        session.continue_generation([])  # timer expires at this boundary

        assert ability.active_skin_name == "base"
        assert ability._dispatch.active_state == "base"

    def test_session_restart_resets_the_dispatch_to_base(self, tmp_path) -> None:
        ability, session, _ = _fused_ability(tmp_path)
        session.start(None, [], "scene prompt")
        ability.request_cycle()
        session.continue_generation([])
        assert ability._dispatch.active_state == "skin"
        session.start(None, [], "scene prompt")
        assert ability._dispatch.active_state == "base"

    def test_fused_mode_accepts_an_accelerated_transformer(self, tmp_path) -> None:
        from types import SimpleNamespace

        session, transformer = _fused_session()
        transformer.config = SimpleNamespace(use_cuda_graph=True, compile_network=True)
        lora_ckpt, corr_ckpt = _write_checkpoints(tmp_path, transformer.network)
        style = LiveEditStyleConfig(
            enabled=True,
            lora_checkpoint=lora_ckpt,
            corrector_checkpoint=corr_ckpt,
            corrector_mode="fused",
            skins=(StyleSkin("arcade", "arcade prompt"),),
        )
        StyleAbility(style).attach(session)  # must not raise

    def test_unfused_mode_still_rejects_cuda_graphs(self, tmp_path) -> None:
        from types import SimpleNamespace

        session, transformer = _fused_session()
        transformer.config = SimpleNamespace(use_cuda_graph=True, compile_network=False)
        lora_ckpt, corr_ckpt = _write_checkpoints(tmp_path, transformer.network)
        style = LiveEditStyleConfig(
            enabled=True,
            lora_checkpoint=lora_ckpt,
            corrector_checkpoint=corr_ckpt,
            corrector_mode="unfused",
            skins=(StyleSkin("arcade", "arcade prompt"),),
        )
        with pytest.raises(RuntimeError, match="use_cuda_graph"):
            StyleAbility(style).attach(session)

    def test_backend_installer_keeps_the_accelerated_session_in_fused_mode(
        self, tmp_path
    ) -> None:
        from types import SimpleNamespace

        from crazy_robotaxi.live_edit.style_ability import (
            install_style_ability_on_backend,
        )

        lora_ckpt, corr_ckpt = _write_checkpoints(
            tmp_path, _ToyDeployTransformer().network
        )
        style = LiveEditStyleConfig(
            enabled=True,
            lora_checkpoint=lora_ckpt,
            corrector_checkpoint=corr_ckpt,
            corrector_mode="fused",
            skins=(StyleSkin("arcade", "arcade prompt"),),
        )
        ability = StyleAbility(style)
        session = SimpleNamespace(warmup_model=lambda: None)
        backend = SimpleNamespace(_session=session)
        install_style_ability_on_backend(backend, ability)
        assert backend._session is session  # no graph-free rebuild


class TestCorrectorModeConfig:
    def test_mode_flag_round_trip_and_new_checkpoint_flags(self) -> None:
        parser = argparse.ArgumentParser()
        add_live_edit_args(parser)
        args = parser.parse_args(
            [
                "--live-edit-style",
                "--live-edit-style-lora",
                "/tmp/lora.pt",
                "--live-edit-corrector-mode",
                "unfused",
                "--live-edit-base-corrector",
                "/tmp/photoreal.pt",
                "--live-edit-base-corrector-gain",
                "0.2",
                "--live-edit-weather",
                "--live-edit-weather-corrector",
                "/tmp/weather.pt",
            ]
        )
        config = live_edit_config_from_args(args)
        assert config.style.corrector_mode == "unfused"
        assert config.style.base_corrector_checkpoint == Path("/tmp/photoreal.pt")
        assert config.style.base_corrector_gain == 0.2
        assert config.weather.corrector_checkpoint == Path("/tmp/weather.pt")

    def test_mode_defaults_to_fused_and_honors_the_env_fallback(
        self, monkeypatch
    ) -> None:
        assert LiveEditStyleConfig().corrector_mode == "fused"
        monkeypatch.setenv("LIVE_EDIT_CORRECTOR_MODE", "unfused")
        parser = argparse.ArgumentParser()
        add_live_edit_args(parser)
        config = live_edit_config_from_args(parser.parse_args([]))
        assert config.style.corrector_mode == "unfused"

    def test_unknown_mode_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="corrector_mode"):
            LiveEditStyleConfig(corrector_mode="turbo")


class TestCorrectorModeOff:
    """`--live-edit-corrector-mode off`: no corrector machinery at all."""

    def _off_ability(
        self, tmp_path
    ) -> tuple[StyleAbility, _FakeSession, _ToyDeployTransformer]:
        session, transformer = _fused_session()
        lora_ckpt, corr_ckpt = _write_checkpoints(tmp_path, transformer.network)
        style = LiveEditStyleConfig(
            enabled=True,
            lora_checkpoint=lora_ckpt,
            corrector_checkpoint=corr_ckpt,
            base_corrector_checkpoint=corr_ckpt,
            corrector_mode="off",
            reswap_interval_chunks=0,
            skins=(StyleSkin("arcade", "arcade prompt"),),
        )
        weather = LiveEditWeatherConfig(
            enabled=True,
            corrector_gain=0.1,
            corrector_checkpoint=corr_ckpt,
            weathers=(WeatherPreset("rain", "rain prompt"),),
        )
        ability = StyleAbility(style, weather)
        return ability, session, transformer

    def test_off_ignores_configured_correctors_and_keeps_weights_pristine(
        self, tmp_path
    ) -> None:
        import torch
        from omnidreams._edit_lora import _target_linears as edit_targets

        ability, session, transformer = self._off_ability(tmp_path)
        before = [
            lin.weight.detach().clone() for lin in edit_targets(transformer.network)
        ]
        ability.attach(session)

        assert ability._dispatch is None
        # No gate driver installed: the scheduler hook stays untouched.
        assert session.pipeline.diffusion_model.scheduler.sample is None
        for lin, weight in zip(edit_targets(transformer.network), before):
            assert torch.equal(lin.weight, weight)
        # The edit LoRA is still attached for skins, but inactive (base
        # weights live) until a swap opens its window.
        assert transformer._text_edit_lora is not None
        assert not transformer._text_edit_lora.active

    def test_off_state_machine_runs_without_a_corrector(self, tmp_path) -> None:
        ability, session, _ = self._off_ability(tmp_path)
        ability.attach(session)
        session.start(None, [], "scene prompt")
        ability.request_cycle()
        session.continue_generation([])
        assert ability.active_skin_name == "arcade"
        ((text, _),) = session.pipeline.replace_text_calls
        assert text == [["arcade prompt"]]

    def test_weather_only_without_corrector_touches_no_weights(self) -> None:
        import torch

        session, transformer = _fused_session()
        weather = LiveEditWeatherConfig(
            enabled=True, weathers=(WeatherPreset("rain", "rain prompt"),)
        )
        ability = StyleAbility(LiveEditStyleConfig(), weather)
        before = [p.detach().clone() for p in transformer.network.parameters()]
        ability.attach(session)
        assert ability._dispatch is None
        assert transformer._text_edit_lora is None
        assert session.pipeline.diffusion_model.scheduler.sample is None
        for parameter, weight in zip(transformer.network.parameters(), before):
            assert torch.equal(parameter, weight)


class TestNativeDitGuard:
    """The native-DIT rejection is precise and actionable."""

    def _native_session(self) -> tuple[_FakeSession, _ToyDeployTransformer]:
        from types import SimpleNamespace

        session, transformer = _fused_session()
        session.manifest = SimpleNamespace(native_dit_acceleration="required")
        return session, transformer

    def test_prompt_swap_abilities_reject_native_dit_naming_the_flags(
        self, tmp_path
    ) -> None:
        ability, session, _ = TestCorrectorModeOff()._off_ability(tmp_path)
        from types import SimpleNamespace

        session.manifest = SimpleNamespace(native_dit_acceleration="required")
        with pytest.raises(RuntimeError) as excinfo:
            ability.attach(session)
        message = str(excinfo.value)
        assert "--live-edit-style" in message
        assert "--live-edit-weather" in message
        assert "native_dit_acceleration: disabled" in message
        # Corrector mode is off, so the error must not blame corrector flags.
        assert "corrector" not in message.lower()

    def test_enabled_corrector_adds_the_corrector_note(self, tmp_path) -> None:
        session, transformer = self._native_session()
        lora_ckpt, corr_ckpt = _write_checkpoints(tmp_path, transformer.network)
        style = LiveEditStyleConfig(
            enabled=True,
            lora_checkpoint=lora_ckpt,
            corrector_checkpoint=corr_ckpt,
            corrector_mode="fused",
            skins=(StyleSkin("arcade", "arcade prompt"),),
        )
        with pytest.raises(RuntimeError, match="--live-edit-corrector-mode off"):
            StyleAbility(style).attach(session)

    def test_disabled_native_dit_attaches_cleanly(self, tmp_path) -> None:
        ability, session, _ = TestCorrectorModeOff()._off_ability(tmp_path)
        ability.attach(session)  # manifest says disabled; must not raise
