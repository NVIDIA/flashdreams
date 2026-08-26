# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""CPU-only tests for the effect-pickup items and timed weather."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pytest
from crazy_robotaxi.live_edit.config import (
    ITEM_TYPES,
    LiveEditItemsConfig,
    LiveEditStyleConfig,
    LiveEditWeatherConfig,
    StyleSkin,
    WeatherPreset,
    add_live_edit_args,
    live_edit_config_from_args,
)
from crazy_robotaxi.live_edit.item_ability import (
    ItemAbility,
    ItemEffects,
    build_item_course,
)
from crazy_robotaxi.live_edit.style_ability import StyleAbility
from crazy_robotaxi.navigation import NavigationLane
from omnidreams_game_engine.config import VehicleConfig
from omnidreams_game_engine.types import CameraCalibration, VehicleState

pytestmark = pytest.mark.ci_cpu


def _straight_lane(length_m: float = 1000.0) -> NavigationLane:
    xs = np.linspace(0.0, length_m, int(length_m) + 1, dtype=np.float32)
    centerline = np.stack([xs, np.zeros_like(xs), np.zeros_like(xs)], axis=1)
    return NavigationLane(centerline_world=centerline)


def _test_calibration() -> CameraCalibration:
    return CameraCalibration(
        clipgt_name="camera:test",
        logical_name="camera_test",
        width=1280,
        height=704,
        cx=640.0,
        cy=352.0,
        polynomial=np.array([0.0, 0.002], dtype=np.float32),
        is_backward_polynomial=True,
        linear_cde=np.array([1.0, 0.0, 0.0], dtype=np.float32),
        sensor_to_rig_flu=np.eye(4, dtype=np.float32),
    )


def _items_config(**overrides: object) -> LiveEditItemsConfig:
    return LiveEditItemsConfig(enabled=True, **overrides)


def _ego_at(x_m: float, y_m: float = 0.0) -> VehicleState:
    return VehicleState(
        x_m=x_m, y_m=y_m, z_m=0.0, yaw_rad=0.0, speed_mps=5.0, steer_rad=0.0
    )


## Fakes mirroring test_live_edit._FakeSession


class _FakePipeline:
    def __init__(self) -> None:
        self.replace_text_calls: list[tuple] = []

    def replace_text(self, cache: object, text: list, **kwargs: object) -> None:
        self.replace_text_calls.append((text, kwargs))

    def finalize(self, index: int, cache: object) -> None:
        pass


class _FakeSession:
    def __init__(self) -> None:
        self.pipeline = _FakePipeline()
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


_SKINS = (
    StyleSkin("arcade", "arcade prompt"),
    StyleSkin("comic", "comic prompt"),
    StyleSkin("cyberpunk", "cyberpunk prompt"),
    StyleSkin("pixel", "pixel prompt"),
)
_WEATHERS = (
    WeatherPreset("rain", "rain prompt"),
    WeatherPreset("snow", "snow prompt"),
)


def _hooked_style_ability(
    *,
    skin_duration_chunks: int = 0,
    weather_duration_chunks: int = 0,
    style_enabled: bool = True,
    weather_enabled: bool = True,
) -> tuple[StyleAbility, _FakeSession]:
    style = LiveEditStyleConfig(
        enabled=style_enabled,
        lora_checkpoint=Path("/dev/null") if style_enabled else None,
        reswap_interval_chunks=0,
        skin_duration_chunks=skin_duration_chunks,
        skins=_SKINS,
    )
    weather = LiveEditWeatherConfig(
        enabled=weather_enabled,
        duration_chunks=weather_duration_chunks,
        weathers=_WEATHERS,
    )
    ability = StyleAbility(style, weather)
    session = _FakeSession()
    ability.hook_session(session)
    session.start(None, [], "scene prompt")
    return ability, session


class _ManualClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


class TestItemCourse:
    def test_items_are_sparse_at_the_configured_spacing(self) -> None:
        centers, types = build_item_course([_straight_lane(1000.0)], _items_config())

        assert len(centers) == 5  # 1000 m / 200 m spacing
        gaps = np.diff(centers[:, 0])
        assert np.allclose(gaps, 200.0, atol=1.0)
        assert len(types) == len(centers)

    def test_types_cycle_through_the_item_kinds(self) -> None:
        _, types = build_item_course([_straight_lane(1400.0)], _items_config())

        assert types[: len(ITEM_TYPES)] == ITEM_TYPES
        assert all(
            types[i] == ITEM_TYPES[i % len(ITEM_TYPES)] for i in range(len(types))
        )

    def test_items_hover_above_the_lane(self) -> None:
        centers, _ = build_item_course(
            [_straight_lane()], _items_config(hover_height_m=1.5)
        )
        assert np.allclose(centers[:, 2], 1.5)

    def test_short_lane_segments_still_get_items(self) -> None:
        # Real maps chop lanes into segments shorter than the item spacing
        # (the shipped suburb map): sparsity must be global, not per lane.
        segments = []
        for start in range(0, 1000, 40):
            xs = np.linspace(start, start + 40, 41, dtype=np.float32)
            centerline = np.stack([xs, np.zeros_like(xs), np.zeros_like(xs)], axis=1)
            segments.append(NavigationLane(centerline_world=centerline))

        centers, _ = build_item_course(segments, _items_config())

        assert 4 <= len(centers) <= 6  # ~1000 m / 200 m spacing
        order = np.sort(centers[:, 0])
        assert (np.diff(order) >= 200.0 - 1e-3).all()

    def test_each_driving_direction_gets_its_own_items(self) -> None:
        # An adjacent opposite-direction lane must not swallow a road's
        # items: drivers of either direction should encounter pickups.
        xs = np.linspace(0.0, 1000.0, 1001, dtype=np.float32)
        forward = np.stack([xs, np.zeros_like(xs), np.zeros_like(xs)], axis=1)
        backward = np.stack(
            [xs[::-1], np.full_like(xs, 3.5), np.zeros_like(xs)], axis=1
        )
        centers, _ = build_item_course(
            [
                NavigationLane(centerline_world=forward),
                NavigationLane(centerline_world=backward),
            ],
            _items_config(),
        )

        per_lane = [int((centers[:, 1] < 1.0).sum()), int((centers[:, 1] > 1.0).sum())]
        assert min(per_lane) >= 4  # ~5 each at 200 m spacing

    def test_overlapping_lanes_do_not_stack_items(self) -> None:
        lane = _straight_lane()
        centers, _ = build_item_course([lane, lane], _items_config())
        deduped, _ = build_item_course([lane], _items_config())
        assert len(centers) == len(deduped)

    def test_empty_lanes_are_rejected(self) -> None:
        with pytest.raises(ValueError):
            build_item_course([], _items_config())


class TestItemPickup:
    def test_pickup_returns_the_item_types_in_order(self) -> None:
        ability = ItemAbility(
            np.array([[10.0, 0.0, 1.0], [20.0, 0.0, 1.0]], dtype=np.float32),
            ("rain", "mystery"),
            _items_config(),
        )

        picked = ability.advance_frames([_ego_at(10.0), _ego_at(20.0)])

        assert picked == ("rain", "mystery")
        assert ability.collected_count == 2
        assert ability.remaining_count == 0

    def test_items_are_one_shot(self) -> None:
        ability = ItemAbility(
            np.array([[10.0, 0.0, 1.0]], dtype=np.float32),
            ("snow",),
            _items_config(),
        )

        assert ability.advance_frames([_ego_at(10.0)]) == ("snow",)
        assert ability.advance_frames([_ego_at(10.0)]) == ()

    def test_items_outside_pickup_radius_stay(self) -> None:
        ability = ItemAbility(
            np.array([[10.0, 0.0, 1.0]], dtype=np.float32),
            ("rain",),
            _items_config(pickup_radius_m=2.0),
        )

        assert ability.advance_frames([_ego_at(10.0, y_m=5.0)]) == ()
        assert ability.remaining_count == 1

    def test_type_length_mismatch_is_rejected(self) -> None:
        with pytest.raises(ValueError):
            ItemAbility(
                np.array([[10.0, 0.0, 1.0]], dtype=np.float32),
                ("rain", "snow"),
                _items_config(),
            )

    def test_unknown_types_are_rejected(self) -> None:
        with pytest.raises(ValueError):
            ItemAbility(
                np.array([[10.0, 0.0, 1.0]], dtype=np.float32),
                ("lava",),
                _items_config(),
            )


class TestItemSprites:
    def test_sprites_carry_their_type_and_do_not_spin(self) -> None:
        ability = ItemAbility(
            np.array([[15.0, 0.0, 1.0], [25.0, 1.0, 1.0]], dtype=np.float32),
            ("mystery", "rain"),
            _items_config(),
        )
        from omnidreams_game_engine.camera import FThetaCameraModel

        camera = FThetaCameraModel(_test_calibration())
        sprites = ability.visible_sprites(
            np.eye(4, dtype=np.float32),
            camera,
            image_width=1280,
            image_height=704,
        )

        assert {sprite.sprite_key for sprite in sprites} == {"mystery", "rain"}
        assert all(not sprite.spin for sprite in sprites)
        # Far-to-near painter's order.
        distances = [sprite.distance_m for sprite in sprites]
        assert distances == sorted(distances, reverse=True)

    def test_flash_expires_on_the_clock(self) -> None:
        clock = _ManualClock()
        ability = ItemAbility(
            np.array([[10.0, 0.0, 1.0]], dtype=np.float32),
            ("rain",),
            _items_config(flash_seconds=2.0),
            clock=clock,
        )

        assert ability.flash_label is None
        ability.flash("RAIN!")
        assert ability.flash_label == "RAIN!"
        clock.now = 1.9
        assert ability.flash_label == "RAIN!"
        clock.now = 2.0
        assert ability.flash_label is None


class TestItemEffects:
    def test_rain_pickup_lands_rain_at_the_next_chunk_boundary(self) -> None:
        style, session = _hooked_style_ability()
        effects = ItemEffects(style, _items_config())

        label = effects.apply("rain")
        assert label == "RAIN!"
        assert style.active_weather_name == "clear"  # not yet: boundary pending

        session.continue_generation([])
        assert style.active_weather_name == "rain"
        text, kwargs = session.pipeline.replace_text_calls[-1]
        assert text == [["rain prompt"]]
        assert kwargs["guidance_scale"] == 2.5

    def test_snow_pickup_lands_snow(self) -> None:
        style, session = _hooked_style_ability()
        effects = ItemEffects(style, _items_config())

        assert effects.apply("snow") == "SNOW!"
        session.continue_generation([])
        assert style.active_weather_name == "snow"

    def test_weather_pickup_is_ignored_while_a_skin_is_active(self) -> None:
        style, session = _hooked_style_ability()
        style.request_cycle()
        session.continue_generation([])
        assert style.active_skin_name == "arcade"
        swaps_before = len(session.pipeline.replace_text_calls)

        label = ItemEffects(style, _items_config()).apply("rain")

        assert "BLOCKED" in label
        session.continue_generation([])
        assert style.active_weather_name == "clear"
        assert len(session.pipeline.replace_text_calls) == swaps_before

    def test_repicking_the_active_weather_refreshes_its_timer(self) -> None:
        style, session = _hooked_style_ability(weather_duration_chunks=5)
        effects = ItemEffects(style, _items_config())
        effects.apply("rain")
        session.continue_generation([])  # lands rain; hold counter = 1
        for _ in range(3):
            session.continue_generation([])
        assert style.weather_chunks_remaining == 1
        swaps_before = len(session.pipeline.replace_text_calls)

        assert effects.apply("rain") == "RAIN!"
        session.continue_generation([])

        # Timer refreshed without a wasteful same-prompt guided re-swap.
        assert style.active_weather_name == "rain"
        assert len(session.pipeline.replace_text_calls) == swaps_before
        assert style.weather_chunks_remaining == 4  # one chunk generated since

    def test_mystery_grants_a_timed_skin_burst_even_in_hold_forever_mode(
        self,
    ) -> None:
        style, session = _hooked_style_ability(skin_duration_chunks=0)
        effects = ItemEffects(
            style, _items_config(mystery_seed=7, mystery_burst_chunks=11)
        )

        label = effects.apply("mystery")
        assert label.startswith("? ") and label.endswith(" BURST!")
        session.continue_generation([])
        granted = style.active_skin_name
        assert granted in style.skin_names
        assert style.skin_chunks_remaining == 10  # landing chunk already ran

        for _ in range(11):
            session.continue_generation([])
        assert style.active_skin_name == "base"
        # A later K press still uses the global (hold-forever) duration.
        style.request_cycle()
        session.continue_generation([])
        assert style.skin_chunks_remaining is None

    def test_mystery_roll_is_reproducible_for_a_seed(self) -> None:
        style_a, session_a = _hooked_style_ability()
        style_b, session_b = _hooked_style_ability()
        effects_a = ItemEffects(style_a, _items_config(mystery_seed=123))
        effects_b = ItemEffects(style_b, _items_config(mystery_seed=123))

        labels_a = []
        labels_b = []
        for _ in range(6):
            labels_a.append(effects_a.apply("mystery"))
            labels_b.append(effects_b.apply("mystery"))
            session_a.continue_generation([])
            session_b.continue_generation([])

        assert labels_a == labels_b
        # The roll actually varies across draws (not stuck on one skin).
        assert len(set(labels_a)) > 1

    def test_mystery_without_a_style_ability_degrades_to_a_hint(self) -> None:
        effects = ItemEffects(None, _items_config())
        assert effects.apply("mystery") == "? NO SKINS"

    def test_weather_item_without_a_weather_ability_degrades_to_a_hint(self) -> None:
        style, _ = _hooked_style_ability(weather_enabled=False)
        effects = ItemEffects(style, _items_config())
        assert effects.apply("rain") == "RAIN N/A"


class TestTimedWeather:
    def test_weather_auto_reverts_after_the_configured_duration(self) -> None:
        style, session = _hooked_style_ability(weather_duration_chunks=4)
        style.request_weather_cycle()
        session.continue_generation([])
        assert style.active_weather_name == "rain"

        for _ in range(3):
            session.continue_generation([])
        assert style.active_weather_name == "rain"  # 4 weather chunks generated
        session.continue_generation([])  # expiry boundary: clear lands
        assert style.active_weather_name == "clear"

    def test_the_auto_revert_lands_guided(self) -> None:
        style, session = _hooked_style_ability(weather_duration_chunks=2)
        style.request_weather_cycle()
        session.continue_generation([])
        for _ in range(3):
            session.continue_generation([])

        assert style.active_weather_name == "clear"
        text, kwargs = session.pipeline.replace_text_calls[-1]
        assert text == [["scene prompt"]]
        assert kwargs["guidance_scale"] == 2.5
        assert kwargs["guidance_chunks"] == 8  # clear_guidance_chunks default

    def test_zero_duration_holds_the_weather_forever(self) -> None:
        style, session = _hooked_style_ability(weather_duration_chunks=0)
        style.request_weather_cycle()
        session.continue_generation([])

        for _ in range(50):
            session.continue_generation([])
        assert style.active_weather_name == "rain"
        assert style.weather_chunks_remaining is None

    def test_remaining_chunks_count_down(self) -> None:
        style, session = _hooked_style_ability(weather_duration_chunks=6)
        assert style.weather_chunks_remaining is None
        style.request_weather_cycle()
        session.continue_generation([])  # lands rain; hold counter = 1
        assert style.weather_chunks_remaining == 5
        session.continue_generation([])
        assert style.weather_chunks_remaining == 4
        assert style.weather_seconds_remaining == pytest.approx(4 * 8.0 / 30.0)

    def test_skin_activation_cancels_the_weather_timer_cleanly(self) -> None:
        style, session = _hooked_style_ability(weather_duration_chunks=4)
        style.request_weather_cycle()
        session.continue_generation([])
        session.continue_generation([])

        style.request_cycle()  # K wins: clears weather + its timer
        session.continue_generation([])
        assert style.active_skin_name == "arcade"
        assert style.active_weather_name == "clear"
        assert style.weather_chunks_remaining is None

        # No stray weather revert fires later.
        swaps_before = len(session.pipeline.replace_text_calls)
        for _ in range(6):
            session.continue_generation([])
        assert len(session.pipeline.replace_text_calls) == swaps_before
        assert style.active_skin_name == "arcade"

    def test_both_trigger_paths_share_the_timer(self) -> None:
        # Pickup-triggered weather times out exactly like the V key path.
        style, session = _hooked_style_ability(weather_duration_chunks=3)
        ItemEffects(style, _items_config()).apply("snow")
        session.continue_generation([])
        assert style.active_weather_name == "snow"
        for _ in range(4):
            session.continue_generation([])
        assert style.active_weather_name == "clear"


class TestKeyAndPickupCoexistence:
    def test_mystery_burst_during_a_key_held_skin_switches_with_fresh_timer(
        self,
    ) -> None:
        style, session = _hooked_style_ability(skin_duration_chunks=0)
        style.request_cycle()  # K: arcade, hold-forever
        session.continue_generation([])
        assert style.active_skin_name == "arcade"
        assert style.skin_chunks_remaining is None

        effects = ItemEffects(
            style, _items_config(mystery_seed=1, mystery_burst_chunks=5)
        )
        effects.apply("mystery")
        session.continue_generation([])

        # Behaves like a K cycle: switch (possibly to the same skin) with a
        # fresh burst timer.
        assert style.active_skin_name in style.skin_names
        assert style.skin_chunks_remaining == 4  # landing chunk already ran
        for _ in range(5):
            session.continue_generation([])
        assert style.active_skin_name == "base"

    def test_key_cycle_still_works_after_pickup_driven_states(self) -> None:
        style, session = _hooked_style_ability(weather_duration_chunks=0)
        effects = ItemEffects(style, _items_config(mystery_seed=2))
        effects.apply("rain")
        session.continue_generation([])
        assert style.active_weather_name == "rain"

        style.request_cycle()  # K clears the pickup weather, lands a skin
        session.continue_generation([])
        assert style.active_skin_name == "arcade"
        assert style.active_weather_name == "clear"

        style.request_weather_cycle()  # V rejected while the skin holds
        session.continue_generation([])
        assert style.active_weather_name == "clear"

    def test_runtime_dispatches_key_requests_and_pickups_together(self) -> None:
        from crazy_robotaxi.app import CrazyRobotaxiRuntime
        from crazy_robotaxi.input import CrazyRobotaxiKeyboardState
        from omnidreams_game_engine.types import TrajectoryChunk

        style, session = _hooked_style_ability()
        items = ItemAbility(
            np.array([[10.0, 0.0, 1.0]], dtype=np.float32),
            ("rain",),
            _items_config(),
        )
        keyboard = CrazyRobotaxiKeyboardState()

        from types import SimpleNamespace

        class FakeController:
            is_playing = True

            def advance_frames(self, trajectory, interval: float) -> tuple:
                # One idle snapshot per frame (passenger builder contract).
                return tuple(
                    SimpleNamespace(session_state="menu")
                    for _ in trajectory.timestamps_us
                )

        runtime = CrazyRobotaxiRuntime(
            FakeController(),
            keyboard,
            style_ability=style,
            item_ability=items,
            item_effects=ItemEffects(style, _items_config()),
        )

        # Key path: K request drains into the style ability...
        keyboard.live_edit.request_skin_cycle()
        runtime.process_events(_ego_at(0.0))
        session.continue_generation([])
        assert style.active_skin_name == "arcade"
        # ... then back to base so the pickup path can land weather.
        for _ in range(len(_SKINS)):
            keyboard.live_edit.request_skin_cycle()
            runtime.process_events(_ego_at(0.0))
            session.continue_generation([])
        assert style.active_skin_name == "base"

        # Pickup path: driving over the rain item queues the weather.
        from omnidreams_game_engine.math3d import rig_pose_from_vehicle_state

        state = _ego_at(10.0)
        trajectory = TrajectoryChunk(
            timestamps_us=np.array([0], dtype=np.int64),
            rig_poses_world=rig_pose_from_vehicle_state(state)[None],
            vehicle_states=(state,),
            boundary_state_after_chunk=state,
        )
        runtime.advance_frames(trajectory, 1.0 / 30.0)
        assert items.flash_label == "RAIN!"
        session.continue_generation([])
        assert style.active_weather_name == "rain"


class TestItemsConfig:
    def test_cli_round_trip(self) -> None:
        parser = argparse.ArgumentParser()
        add_live_edit_args(parser)
        args = parser.parse_args(
            [
                "--live-edit-items",
                "--live-edit-item-spacing",
                "150",
                "--live-edit-item-rain-sprite",
                "/tmp/rain.png",
                "--live-edit-item-mystery-seed",
                "42",
                "--live-edit-item-mystery-burst-chunks",
                "9",
                "--live-edit-weather-duration-chunks",
                "45",
            ]
        )
        config = live_edit_config_from_args(args)

        assert config.items.enabled
        assert config.items.spacing_m == 150.0
        assert config.items.rain_sprite_path == Path("/tmp/rain.png")
        assert config.items.snow_sprite_path is None
        assert config.items.mystery_seed == 42
        assert config.items.mystery_burst_chunks == 9
        assert config.weather.duration_chunks == 45

    def test_items_default_disabled(self) -> None:
        parser = argparse.ArgumentParser()
        add_live_edit_args(parser)
        config = live_edit_config_from_args(parser.parse_args([]))
        assert not config.items.enabled
        assert not config.any_enabled

    def test_validation_rejects_bad_values(self) -> None:
        with pytest.raises(ValueError):
            LiveEditItemsConfig(spacing_m=0.0)
        with pytest.raises(ValueError):
            LiveEditItemsConfig(mystery_burst_chunks=-1)
        with pytest.raises(ValueError):
            LiveEditWeatherConfig(duration_chunks=-1)
        with pytest.raises(ValueError):
            LiveEditItemsConfig().sprite_path("lava")


class TestItemPresenterWiring:
    def _presented(self, item_ability: ItemAbility, config: object):
        from dataclasses import dataclass, field
        from typing import Any

        from crazy_robotaxi.live_edit.presenter import LiveEditPresenter
        from omnidreams_game_engine.types import PresentedFrame

        @dataclass
        class RecordingPresenter:
            frames: list[Any] = field(default_factory=list)

            def present_frame(self, frame: Any, view_mode: str) -> None:
                self.frames.append(frame)

        inner = RecordingPresenter()
        presenter = LiveEditPresenter(inner, config)
        presenter.set_item_ability(item_ability)
        presenter.configure_taxi_camera(_test_calibration())
        rgb = np.full((704, 1280, 3), 90, dtype=np.uint8)
        frame = PresentedFrame(
            timestamp_us=1,
            rgb_host_uint8=rgb,
            depth_host_f32=None,
            model_rgb_host_uint8=rgb.copy(),
            rig_to_world=np.eye(4, dtype=np.float32),
        )
        presenter.present_frame(frame, "model_rgb")
        return rgb, np.asarray(inner.frames[0].model_rgb_host_uint8)

    def test_composites_item_sprites_without_a_coin_ability(self) -> None:
        from crazy_robotaxi.live_edit.config import LiveEditConfig

        config = LiveEditConfig(items=_items_config())
        ability = ItemAbility(
            np.array([[15.0, 0.0, 1.0]], dtype=np.float32),
            ("mystery",),
            config.items,
        )

        rgb, out = self._presented(ability, config)

        assert not np.array_equal(out, rgb)

    def test_flash_label_shows_in_the_hud_chips(self) -> None:
        from crazy_robotaxi.live_edit.config import LiveEditConfig
        from crazy_robotaxi.live_edit.presenter import LiveEditPresenter

        config = LiveEditConfig(items=_items_config())
        ability = ItemAbility(
            np.array([[500.0, 0.0, 1.0]], dtype=np.float32),
            ("rain",),
            config.items,
        )
        presenter = LiveEditPresenter(object(), config)
        presenter.set_item_ability(ability)

        assert "RAIN!" not in presenter._hud_labels()
        ability.flash("RAIN!")
        assert "RAIN!" in presenter._hud_labels()

    def test_procedural_placeholders_load_when_no_sprites_configured(self) -> None:
        from crazy_robotaxi.live_edit.presenter import (
            LiveEditPresenter,
            procedural_item_sprite,
        )

        sprites = LiveEditPresenter._load_item_sprites(_items_config())

        assert set(sprites) == set(ITEM_TYPES)
        for item_type, sprite in sprites.items():
            assert sprite.mode == "RGBA"
            reference = procedural_item_sprite(item_type)
            assert np.array_equal(np.asarray(sprite), np.asarray(reference))
        # Distinct placeholder art per type.
        assert not np.array_equal(
            np.asarray(sprites["rain"]), np.asarray(sprites["snow"])
        )

    def test_configured_sprite_paths_override_the_placeholders(self, tmp_path) -> None:
        from crazy_robotaxi.live_edit.presenter import LiveEditPresenter
        from PIL import Image

        path = tmp_path / "rain.png"
        Image.new("RGBA", (12, 34), (1, 2, 3, 4)).save(path)
        sprites = LiveEditPresenter._load_item_sprites(
            _items_config(rain_sprite_path=path)
        )

        assert sprites["rain"].size == (12, 34)
        assert sprites["snow"].size != (12, 34)


class TestNitroAbility:
    def _nitro(self, **overrides: object):
        from crazy_robotaxi.live_edit.nitro_ability import NitroAbility

        return NitroAbility(_items_config(**overrides))

    def test_activation_starts_the_boost_and_counts_down_game_time(self) -> None:
        nitro = self._nitro(nitro_duration_s=1.0)
        vehicle = VehicleConfig()

        assert not nitro.active
        nitro.activate()
        assert nitro.active
        assert nitro.seconds_remaining == pytest.approx(1.0)
        for _ in range(10):
            nitro.vehicle_for_tick(vehicle, 0.1)
        assert not nitro.active
        assert nitro.seconds_remaining == 0.0

    def test_repickup_resets_the_timer_without_stacking(self) -> None:
        nitro = self._nitro(nitro_duration_s=4.0)
        vehicle = VehicleConfig()

        nitro.activate()
        for _ in range(20):
            nitro.vehicle_for_tick(vehicle, 0.1)  # burn 2 s
        assert nitro.seconds_remaining == pytest.approx(2.0)
        nitro.activate()
        assert nitro.seconds_remaining == pytest.approx(4.0)

    def test_boost_multiplies_accel_and_caps_max_speed_at_the_ceiling(self) -> None:
        nitro = self._nitro(nitro_boost=1.6, nitro_max_speed_mps=16.0)
        vehicle = VehicleConfig(max_speed_mps=31.2928, max_accel_mps2=10.0)

        boosted = nitro.boosted_vehicle(vehicle)

        assert boosted.max_accel_mps2 == pytest.approx(16.0)
        assert boosted.max_speed_mps == pytest.approx(16.0)  # ceiling binds

    def test_ceiling_yields_when_the_boosted_max_is_smaller(self) -> None:
        nitro = self._nitro(nitro_boost=1.5, nitro_max_speed_mps=100.0)
        vehicle = VehicleConfig(max_speed_mps=10.0)

        assert nitro.boosted_vehicle(vehicle).max_speed_mps == pytest.approx(15.0)

    def test_ticks_apply_and_revert_the_multiplier_exactly(self) -> None:
        nitro = self._nitro(nitro_boost=1.6, nitro_duration_s=0.2)
        vehicle = VehicleConfig()

        assert nitro.vehicle_for_tick(vehicle, 0.1) is vehicle  # inactive
        nitro.activate()
        boosted = nitro.vehicle_for_tick(vehicle, 0.1)
        assert boosted.max_accel_mps2 == pytest.approx(vehicle.max_accel_mps2 * 1.6)
        nitro.vehicle_for_tick(vehicle, 0.1)  # expires here
        after = nitro.vehicle_for_tick(vehicle, 0.1)
        assert after is vehicle  # exact base config, not a copy

    def test_reset_drops_an_active_boost(self) -> None:
        nitro = self._nitro()
        nitro.activate()
        nitro.reset()
        assert not nitro.active

    def test_integrated_speed_boosts_then_respects_the_ceiling(self) -> None:
        from crazy_robotaxi.driving import TaxiVehicleConfig, integrate_taxi_vehicle
        from crazy_robotaxi.live_edit.nitro_ability import integrate_with_nitro
        from omnidreams_game_engine.types import DriverCommand

        nitro = self._nitro(
            nitro_boost=1.6, nitro_duration_s=4.0, nitro_max_speed_mps=16.0
        )
        integrate = integrate_with_nitro(nitro, integrate_taxi_vehicle)
        vehicle = TaxiVehicleConfig()
        throttle = DriverCommand(throttle=1.0)

        state = _ego_at(0.0)
        state = integrate(state, throttle, 1.0 / 30.0, vehicle)
        base_gain = state.speed_mps - 5.0

        nitro.activate()
        boosted_state = integrate(_ego_at(0.0), throttle, 1.0 / 30.0, vehicle)
        assert boosted_state.speed_mps - 5.0 == pytest.approx(base_gain * 1.6)

        # Full throttle through the boost window never exceeds the ceiling.
        state = _ego_at(0.0)
        nitro.activate()
        top = 0.0
        while nitro.active:
            state = integrate(state, throttle, 1.0 / 30.0, vehicle)
            top = max(top, state.speed_mps)
        assert top <= 16.0 + 1e-6
        assert top > 10.0  # the boost actually reached past base speeds


class TestNitroDispatch:
    def test_nitro_pickup_activates_the_boost_immediately(self) -> None:
        from crazy_robotaxi.live_edit.nitro_ability import NitroAbility

        config = _items_config()
        nitro = NitroAbility(config)
        effects = ItemEffects(None, config, nitro_ability=nitro)

        assert not nitro.active
        assert effects.apply("nitro") == "NITRO!"
        assert nitro.active  # no chunk-boundary wait

    def test_nitro_without_an_ability_degrades_to_a_hint(self) -> None:
        effects = ItemEffects(None, _items_config())
        assert effects.apply("nitro") == "NITRO N/A"

    def test_nitro_composes_with_a_skin_burst(self) -> None:
        from crazy_robotaxi.live_edit.nitro_ability import NitroAbility

        style, session = _hooked_style_ability()
        config = _items_config(mystery_seed=7)
        nitro = NitroAbility(config)
        effects = ItemEffects(style, config, nitro_ability=nitro)

        effects.apply("mystery")
        session.continue_generation([])
        skin = style.active_skin_name
        assert skin in style.skin_names

        assert effects.apply("nitro") == "NITRO!"
        session.continue_generation([])
        # Physics-only: the skin state machine is untouched by nitro.
        assert style.active_skin_name == skin
        assert nitro.active

    def test_runtime_pickup_dispatches_nitro(self) -> None:
        from types import SimpleNamespace

        from crazy_robotaxi.app import CrazyRobotaxiRuntime
        from crazy_robotaxi.input import CrazyRobotaxiKeyboardState
        from crazy_robotaxi.live_edit.nitro_ability import NitroAbility
        from omnidreams_game_engine.math3d import rig_pose_from_vehicle_state
        from omnidreams_game_engine.types import TrajectoryChunk

        config = _items_config()
        nitro = NitroAbility(config)
        items = ItemAbility(
            np.array([[10.0, 0.0, 1.0]], dtype=np.float32), ("nitro",), config
        )

        class FakeController:
            is_playing = True

            def advance_frames(self, trajectory, interval: float) -> tuple:
                return tuple(
                    SimpleNamespace(session_state="menu")
                    for _ in trajectory.timestamps_us
                )

        runtime = CrazyRobotaxiRuntime(
            FakeController(),
            CrazyRobotaxiKeyboardState(),
            item_ability=items,
            item_effects=ItemEffects(None, config, nitro_ability=nitro),
        )
        state = _ego_at(10.0)
        trajectory = TrajectoryChunk(
            timestamps_us=np.array([0], dtype=np.int64),
            rig_poses_world=rig_pose_from_vehicle_state(state)[None],
            vehicle_states=(state,),
            boundary_state_after_chunk=state,
        )

        runtime.advance_frames(trajectory, 1.0 / 30.0)

        assert items.flash_label == "NITRO!"
        assert nitro.active


class TestNitroConfig:
    def test_nitro_cli_round_trip(self) -> None:
        parser = argparse.ArgumentParser()
        add_live_edit_args(parser)
        args = parser.parse_args(
            [
                "--live-edit-items",
                "--live-edit-item-nitro-sprite",
                "/tmp/gpu.png",
                "--live-edit-nitro-boost",
                "2.0",
                "--live-edit-nitro-duration-s",
                "6.5",
                "--live-edit-nitro-max-speed",
                "18.0",
                "--live-edit-item-types",
                "nitro,rain",
            ]
        )
        config = live_edit_config_from_args(args)

        assert config.items.nitro_sprite_path == Path("/tmp/gpu.png")
        assert config.items.nitro_boost == 2.0
        assert config.items.nitro_duration_s == 6.5
        assert config.items.nitro_max_speed_mps == 18.0
        assert config.items.item_types == ("nitro", "rain")

    def test_nitro_defaults(self) -> None:
        parser = argparse.ArgumentParser()
        add_live_edit_args(parser)
        config = live_edit_config_from_args(parser.parse_args([]))

        assert config.items.nitro_boost == 1.6
        assert config.items.nitro_duration_s == 4.0
        assert config.items.nitro_max_speed_mps == 16.0
        assert config.items.item_types == ITEM_TYPES
        assert "nitro" in ITEM_TYPES

    def test_nitro_validation_rejects_bad_values(self) -> None:
        with pytest.raises(ValueError):
            LiveEditItemsConfig(nitro_boost=0.9)
        with pytest.raises(ValueError):
            LiveEditItemsConfig(nitro_duration_s=0.0)
        with pytest.raises(ValueError):
            LiveEditItemsConfig(nitro_max_speed_mps=0.0)
        with pytest.raises(ValueError):
            LiveEditItemsConfig(item_types=("lava",))
        with pytest.raises(ValueError):
            LiveEditItemsConfig(item_types=())

    def test_course_mix_can_be_restricted_to_one_kind(self) -> None:
        _, types = build_item_course(
            [_straight_lane(1400.0)], _items_config(item_types=("nitro",))
        )
        assert types and set(types) == {"nitro"}

    def test_nitro_chip_shows_boost_and_countdown(self) -> None:
        from crazy_robotaxi.live_edit.config import LiveEditConfig
        from crazy_robotaxi.live_edit.nitro_ability import NitroAbility
        from crazy_robotaxi.live_edit.presenter import LiveEditPresenter

        config = LiveEditConfig(items=_items_config())
        nitro = NitroAbility(config.items)
        presenter = LiveEditPresenter(object(), config)
        presenter.set_nitro_ability(nitro)

        assert not any("NITRO" in label for label in presenter._hud_labels())
        nitro.activate()
        assert "NITRO x1.6 4.0s" in presenter._hud_labels()
