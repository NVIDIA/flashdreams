# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""CPU-only tests for the live-edit ability scaffolding."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pytest
from crazy_robotaxi.live_edit.coin_ability import CoinAbility, build_coin_course
from crazy_robotaxi.live_edit.config import (
    LiveEditCoinsConfig,
    LiveEditConfig,
    LiveEditStyleConfig,
    StyleSkin,
    add_live_edit_args,
    live_edit_config_from_args,
)
from crazy_robotaxi.live_edit.input_hooks import LiveEditRequests
from crazy_robotaxi.live_edit.presenter import LiveEditPresenter, unsharp_rgb
from crazy_robotaxi.live_edit.style_ability import StyleAbility
from crazy_robotaxi.navigation import NavigationLane
from omnidreams_game_engine.types import CameraCalibration, VehicleState

pytestmark = pytest.mark.ci_cpu


def _straight_lane(length_m: float = 200.0) -> NavigationLane:
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


def _coins_config(**overrides: object) -> LiveEditCoinsConfig:
    return LiveEditCoinsConfig(enabled=True, **overrides)


def _ego_at(x_m: float, y_m: float = 0.0) -> VehicleState:
    return VehicleState(
        x_m=x_m, y_m=y_m, z_m=0.0, yaw_rad=0.0, speed_mps=5.0, steer_rad=0.0
    )


class TestCoinCourse:
    def test_lays_out_lateral_groups_along_the_lane(self) -> None:
        config = _coins_config(spacing_m=50.0, group_offsets_m=(-1.0, 0.0, 1.0))
        coins = build_coin_course((_straight_lane(),), config)

        assert len(coins) % 3 == 0
        assert len(coins) >= 3
        # Lateral offsets sit right of the +X-directed lane (negative y is right).
        first_group = coins[:3]
        assert sorted(np.round(first_group[:, 1], 3).tolist()) == [-1.0, 0.0, 1.0]
        assert np.allclose(first_group[:, 2], config.hover_height_m)

    def test_includes_driving_lanes_without_stop_edges(self) -> None:
        # Lanes without a mapped roadside stopping edge are still driving
        # lanes; the course must cover them so pickups trigger on-route.
        lane = _straight_lane()
        no_stop_lane = NavigationLane(
            centerline_world=lane.centerline_world, allows_taxi_stops=False
        )
        coins = build_coin_course((no_stop_lane,), _coins_config())
        assert len(coins) >= 3

    def test_coins_reachable_within_pickup_radius_of_the_lane_center(self) -> None:
        config = _coins_config(pickup_radius_m=2.5)
        lane = NavigationLane(
            centerline_world=_straight_lane().centerline_world,
            allows_taxi_stops=False,
        )
        ability = CoinAbility(build_coin_course((lane,), config), config)
        # Drive straight down the lane center; every group must yield pickups.
        states = [_ego_at(float(x)) for x in np.arange(0.0, 200.0, 1.0)]
        assert ability.advance_frames(states) == ability.collected_count
        assert ability.collected_count > 0
        assert ability.remaining_count == 0


class TestCoinPickup:
    def test_collects_coins_within_pickup_radius(self) -> None:
        config = _coins_config(pickup_radius_m=2.5, points_per_coin=50)
        coins = np.array([[10.0, 0.0, 0.8], [50.0, 0.0, 0.8]], dtype=np.float32)
        ability = CoinAbility(coins, config)

        collected = ability.advance_frames([_ego_at(9.0)])

        assert collected == 1
        assert ability.collected_count == 1
        assert ability.score == 50
        assert ability.remaining_count == 1

    def test_ignores_coins_outside_pickup_radius(self) -> None:
        ability = CoinAbility(
            np.array([[10.0, 0.0, 0.8]], dtype=np.float32),
            _coins_config(pickup_radius_m=2.5),
        )

        assert ability.advance_frames([_ego_at(0.0), _ego_at(5.0)]) == 0
        assert ability.collected_count == 0

    def test_toggle_disables_collection(self) -> None:
        ability = CoinAbility(
            np.array([[10.0, 0.0, 0.8]], dtype=np.float32), _coins_config()
        )
        assert ability.toggle() is False
        assert ability.advance_frames([_ego_at(10.0)]) == 0


class TestCoinProjection:
    def test_projects_nearer_coins_larger_and_near_image_center(self) -> None:
        from omnidreams_game_engine.camera import FThetaCameraModel

        config = _coins_config(max_render_distance_m=120.0)
        coins = np.array([[10.0, 0.0, 0.0], [40.0, 0.0, 0.0]], dtype=np.float32)
        ability = CoinAbility(coins, config)
        camera = FThetaCameraModel(_test_calibration())

        sprites = ability.visible_sprites(
            np.eye(4, dtype=np.float32),
            camera,
            image_width=1280,
            image_height=704,
        )

        assert len(sprites) == 2
        # Far-to-near painter's order.
        assert sprites[0].distance_m > sprites[1].distance_m
        assert sprites[1].height_px > sprites[0].height_px
        for sprite in sprites:
            assert abs(sprite.center_uv[0] - 640.0) < 2.0

    def test_culls_collected_and_distant_coins(self) -> None:
        from omnidreams_game_engine.camera import FThetaCameraModel

        config = _coins_config(max_render_distance_m=50.0, fade_start_distance_m=40.0)
        coins = np.array([[10.0, 0.0, 0.0], [500.0, 0.0, 0.0]], dtype=np.float32)
        ability = CoinAbility(coins, config)
        ability.advance_frames([_ego_at(10.0)])
        camera = FThetaCameraModel(_test_calibration())

        sprites = ability.visible_sprites(
            np.eye(4, dtype=np.float32),
            camera,
            image_width=1280,
            image_height=704,
        )

        assert sprites == ()


class TestPresenter:
    def test_unsharp_increases_edge_contrast(self) -> None:
        frame = np.full((32, 32, 3), 100, dtype=np.uint8)
        frame[:, 16:] = 160

        sharpened = unsharp_rgb(frame, sigma=2.0, amount=0.8)

        original_step = int(frame[16, 16, 0]) - int(frame[16, 15, 0])
        sharpened_step = int(sharpened[16, 16, 0]) - int(sharpened[16, 15, 0])
        assert sharpened_step > original_step

    def test_zero_amount_is_identity(self) -> None:
        frame = np.random.default_rng(0).integers(
            0, 255, size=(16, 16, 3), dtype=np.uint8
        )
        assert unsharp_rgb(frame, sigma=2.0, amount=0.0) is frame

    def test_composites_coins_and_counter_into_the_model_frame(self) -> None:
        from dataclasses import dataclass, field
        from typing import Any

        from omnidreams_game_engine.types import PresentedFrame

        @dataclass
        class RecordingPresenter:
            frames: list[Any] = field(default_factory=list)

            def present_frame(self, frame: Any, view_mode: str) -> None:
                self.frames.append(frame)

            def close(self) -> None:
                pass

        config = LiveEditConfig(coins=_coins_config())
        ability = CoinAbility(
            np.array([[15.0, 0.0, 0.0]], dtype=np.float32), config.coins
        )
        inner = RecordingPresenter()
        presenter = LiveEditPresenter(inner, config, coin_ability=ability)
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

        assert len(inner.frames) == 1
        out = np.asarray(inner.frames[0].model_rgb_host_uint8)
        assert out.shape == rgb.shape
        assert not np.array_equal(out, rgb)
        # Counter chip occupies the top-left corner.
        assert not np.array_equal(out[:40, :120], rgb[:40, :120])

    def test_passes_frames_through_untouched_when_nothing_is_enabled(self) -> None:
        from dataclasses import dataclass, field
        from typing import Any

        from omnidreams_game_engine.types import PresentedFrame

        @dataclass
        class RecordingPresenter:
            frames: list[Any] = field(default_factory=list)

            def present_frame(self, frame: Any, view_mode: str) -> None:
                self.frames.append(frame)

        presenter = LiveEditPresenter(RecordingPresenter(), LiveEditConfig())
        frame = PresentedFrame(timestamp_us=1, rgb_host_uint8=None, depth_host_f32=None)
        presenter.present_frame(frame, "model_rgb")

        assert presenter._inner.frames[0] is frame


class _FakePipeline:
    def __init__(self) -> None:
        self.replace_text_calls: list[tuple] = []
        self.finalize_calls: list[int] = []
        self.call_order: list[str] = []

    def replace_text(self, cache: object, text: list, **kwargs: object) -> None:
        self.replace_text_calls.append((cache, text, kwargs))
        self.call_order.append("replace_text")

    def finalize(self, index: int, cache: object) -> None:
        self.finalize_calls.append(index)
        self.call_order.append("finalize")


class _FakeSession:
    def __init__(self) -> None:
        self.pipeline = _FakePipeline()
        self._cache = object()
        self._pending_finalization_index: int | None = None
        self.chunks: list[str] = []

    def start(self, initial_rgb: object, frames: list, prompt: str) -> list:
        self.chunks.append("start")
        self._pending_finalization_index = len(self.chunks) - 1
        return []

    def continue_generation(self, frames: list) -> list:
        self.chunks.append("continue")
        self._pending_finalization_index = len(self.chunks) - 1
        return []


def _hooked_style_ability(
    *, reswap_interval_chunks: int = 0
) -> tuple[StyleAbility, _FakeSession]:
    config = LiveEditStyleConfig(
        enabled=True,
        lora_checkpoint=Path("/dev/null"),
        reswap_interval_chunks=reswap_interval_chunks,
        skins=(
            StyleSkin("arcade", "arcade prompt"),
            StyleSkin("comic", "comic prompt"),
        ),
    )
    ability = StyleAbility(config)
    session = _FakeSession()
    ability.hook_session(session)
    return ability, session


class TestStyleAbility:
    def _ability(self) -> tuple[StyleAbility, _FakeSession]:
        return _hooked_style_ability()

    def test_cycle_swaps_prompt_on_the_next_chunk_boundary(self) -> None:
        ability, session = self._ability()
        session.start(None, [], "scene prompt")
        ability.request_cycle()

        assert ability.active_skin_name == "base"
        session.continue_generation([])

        assert ability.active_skin_name == "arcade"
        ((_, text, _),) = session.pipeline.replace_text_calls
        assert text == [["arcade prompt"]]

    def test_cycle_wraps_back_to_base_prompt(self) -> None:
        ability, session = self._ability()
        session.start(None, [], "scene prompt")
        for _ in range(3):
            ability.request_cycle()
            session.continue_generation([])

        assert ability.active_skin_name == "base"
        assert session.pipeline.replace_text_calls[-1][1] == [["scene prompt"]]

    def test_swap_before_first_chunk_is_ignored(self) -> None:
        ability, session = self._ability()
        ability.request_cycle()
        session.continue_generation([])

        assert session.pipeline.replace_text_calls == []
        assert ability.active_skin_name == "base"

    def test_skin_swap_opens_the_lora_window_with_configured_guidance(self) -> None:
        ability, session = self._ability()
        session.start(None, [], "scene prompt")
        ability.request_cycle()
        session.continue_generation([])

        ((_, _, kwargs),) = session.pipeline.replace_text_calls
        assert kwargs["guidance_scale"] == 2.5
        assert kwargs["guidance_chunks"] == 20

    def test_revert_to_base_is_a_plain_swap(self) -> None:
        ability, session = self._ability()
        session.start(None, [], "scene prompt")
        for _ in range(3):
            ability.request_cycle()
            session.continue_generation([])

        _, text, kwargs = session.pipeline.replace_text_calls[-1]
        assert text == [["scene prompt"]]
        assert kwargs["guidance_scale"] == 1.0
        assert kwargs["guidance_chunks"] == 0

    def test_pending_finalize_flushes_before_the_swap(self) -> None:
        ability, session = self._ability()
        session.start(None, [], "scene prompt")
        ability.request_cycle()
        session.continue_generation([])

        assert session.pipeline.finalize_calls == [0]
        assert session.pipeline.call_order == ["finalize", "replace_text"]
        # The next boundary without a queued request leaves finalize to the
        # session's own deferred path.
        session.continue_generation([])
        assert session.pipeline.finalize_calls == [0]

    def test_rollout_restart_resets_to_base(self) -> None:
        ability, session = self._ability()
        session.start(None, [], "scene prompt")
        ability.request_cycle()
        session.continue_generation([])
        session.start(None, [], "scene prompt")

        assert ability.active_skin_name == "base"


class TestDutyCycledReswap:
    def test_reissues_the_active_skin_every_interval(self) -> None:
        ability, session = _hooked_style_ability(reswap_interval_chunks=3)
        session.start(None, [], "scene prompt")
        ability.request_cycle()
        session.continue_generation([])  # swap lands -> arcade, count 1
        assert len(session.pipeline.replace_text_calls) == 1

        session.continue_generation([])  # count 2
        session.continue_generation([])  # count 3
        assert len(session.pipeline.replace_text_calls) == 1
        session.continue_generation([])  # count >= 3 -> re-swap
        assert len(session.pipeline.replace_text_calls) == 2

        _, text, kwargs = session.pipeline.replace_text_calls[-1]
        assert text == [["arcade prompt"]]
        assert kwargs["guidance_scale"] == 2.5
        assert kwargs["guidance_chunks"] == 20
        assert ability.active_skin_name == "arcade"

    def test_reswap_flushes_the_pending_finalize_first(self) -> None:
        ability, session = _hooked_style_ability(reswap_interval_chunks=1)
        session.start(None, [], "scene prompt")
        ability.request_cycle()
        session.continue_generation([])
        session.pipeline.call_order.clear()

        session.continue_generation([])  # re-swap due immediately

        assert session.pipeline.call_order == ["finalize", "replace_text"]

    def test_no_reswap_while_the_base_world_is_active(self) -> None:
        _, session = _hooked_style_ability(reswap_interval_chunks=1)
        session.start(None, [], "scene prompt")
        for _ in range(4):
            session.continue_generation([])

        assert session.pipeline.replace_text_calls == []

    def test_zero_interval_disables_the_refresh(self) -> None:
        ability, session = _hooked_style_ability(reswap_interval_chunks=0)
        session.start(None, [], "scene prompt")
        ability.request_cycle()
        session.continue_generation([])
        for _ in range(12):
            session.continue_generation([])

        assert len(session.pipeline.replace_text_calls) == 1

    def test_reswap_counter_restarts_after_a_manual_swap(self) -> None:
        ability, session = _hooked_style_ability(reswap_interval_chunks=3)
        session.start(None, [], "scene prompt")
        ability.request_cycle()
        session.continue_generation([])  # arcade, count 1
        session.continue_generation([])  # count 2
        ability.request_cycle()
        session.continue_generation([])  # comic swap resets the counter
        assert len(session.pipeline.replace_text_calls) == 2
        session.continue_generation([])
        session.continue_generation([])
        assert len(session.pipeline.replace_text_calls) == 2
        session.continue_generation([])
        assert len(session.pipeline.replace_text_calls) == 3
        assert session.pipeline.replace_text_calls[-1][1] == [["comic prompt"]]


class TestConfig:
    def test_style_requires_a_lora_checkpoint(self) -> None:
        with pytest.raises(ValueError):
            LiveEditStyleConfig(enabled=True)

    def test_cli_round_trip(self) -> None:
        parser = argparse.ArgumentParser()
        add_live_edit_args(parser)
        args = parser.parse_args(
            [
                "--live-edit-coins",
                "--live-edit-style",
                "--live-edit-style-lora",
                "/tmp/lora.pt",
            ]
        )
        config = live_edit_config_from_args(args)

        assert config.coins.enabled
        assert config.style.enabled
        assert config.style.lora_checkpoint == Path("/tmp/lora.pt")
        assert config.any_enabled

    def test_defaults_disable_everything(self) -> None:
        parser = argparse.ArgumentParser()
        add_live_edit_args(parser)
        config = live_edit_config_from_args(parser.parse_args([]))

        assert not config.any_enabled


class TestRequests:
    def test_requests_are_one_shot(self) -> None:
        requests = LiveEditRequests()
        assert requests.consume_skin_cycle() is False
        requests.request_skin_cycle()
        requests.request_coins_toggle()
        assert requests.consume_skin_cycle() is True
        assert requests.consume_skin_cycle() is False
        assert requests.consume_coins_toggle() is True
        assert requests.consume_coins_toggle() is False


class TestPresenterWiring:
    def _frame(self, rgb: np.ndarray):
        from omnidreams_game_engine.types import PresentedFrame

        return PresentedFrame(
            timestamp_us=1,
            rgb_host_uint8=rgb,
            depth_host_f32=None,
            model_rgb_host_uint8=rgb.copy(),
            rig_to_world=np.eye(4, dtype=np.float32),
        )

    def test_draws_the_skin_chip_when_a_skin_is_active(self) -> None:
        class RecordingPresenter:
            def __init__(self) -> None:
                self.frames: list = []

            def present_frame(self, frame, view_mode: str) -> None:
                self.frames.append(frame)

        class StyleStub:
            active_skin_name = "cyberpunk"

        inner = RecordingPresenter()
        config = LiveEditConfig(sharpen_amount=0.0)
        presenter = LiveEditPresenter(inner, config, style_ability=StyleStub())

        rgb = np.full((704, 1280, 3), 90, dtype=np.uint8)
        presenter.present_frame(self._frame(rgb), "model_rgb")

        out = np.asarray(inner.frames[0].model_rgb_host_uint8)
        # Only the chip pixels differ (sharpen disabled for isolation).
        assert not np.array_equal(out[:40, :120], rgb[:40, :120])
        assert np.array_equal(out[100:, :], rgb[100:, :])

    def test_set_coin_ability_binds_a_new_course(self) -> None:
        class RecordingPresenter:
            def __init__(self) -> None:
                self.frames: list = []

            def present_frame(self, frame, view_mode: str) -> None:
                self.frames.append(frame)

        inner = RecordingPresenter()
        config = LiveEditConfig(coins=_coins_config())
        presenter = LiveEditPresenter(inner, config)
        presenter.configure_taxi_camera(_test_calibration())

        rgb = np.full((704, 1280, 3), 90, dtype=np.uint8)
        presenter.present_frame(self._frame(rgb), "model_rgb")
        assert inner.frames[0] is not None
        assert np.array_equal(np.asarray(inner.frames[0].model_rgb_host_uint8), rgb)

        presenter.set_coin_ability(
            CoinAbility(np.array([[15.0, 0.0, 0.0]], dtype=np.float32), config.coins)
        )
        from dataclasses import replace

        frame = replace(self._frame(rgb), timestamp_us=2)
        presenter.present_frame(frame, "model_rgb")
        out = np.asarray(inner.frames[1].model_rgb_host_uint8)
        assert not np.array_equal(out, rgb)


class TestRuntimeWiring:
    def test_keyboard_owns_the_live_edit_request_channel(self) -> None:
        from crazy_robotaxi.input import CrazyRobotaxiKeyboardState

        keyboard = CrazyRobotaxiKeyboardState()
        assert isinstance(keyboard.live_edit, LiveEditRequests)

    def test_runtime_drains_requests_into_the_abilities(self) -> None:
        from crazy_robotaxi.app import CrazyRobotaxiRuntime
        from crazy_robotaxi.input import CrazyRobotaxiKeyboardState

        class FakeStyle:
            def __init__(self) -> None:
                self.cycles = 0

            def request_cycle(self) -> None:
                self.cycles += 1

        class FakeController:
            is_playing = True

        keyboard = CrazyRobotaxiKeyboardState()
        style = FakeStyle()
        coins = CoinAbility(
            np.array([[10.0, 0.0, 0.8]], dtype=np.float32), _coins_config()
        )
        runtime = CrazyRobotaxiRuntime(
            FakeController(), keyboard, style_ability=style, coin_ability=coins
        )

        keyboard.live_edit.request_skin_cycle()
        keyboard.live_edit.request_coins_toggle()
        runtime.process_events(_ego_at(0.0))
        assert style.cycles == 1
        assert coins.enabled is False

        # Requests are one-shot: a second drain without new keys is a no-op.
        runtime.process_events(_ego_at(0.0))
        assert style.cycles == 1
        assert coins.enabled is False


class TestReswapConfig:
    def test_reswap_flag_round_trip(self) -> None:
        parser = argparse.ArgumentParser()
        add_live_edit_args(parser)
        args = parser.parse_args(
            [
                "--live-edit-style",
                "--live-edit-style-lora",
                "/tmp/lora.pt",
                "--live-edit-style-reswap-chunks",
                "5",
            ]
        )
        config = live_edit_config_from_args(args)
        assert config.style.reswap_interval_chunks == 5

    def test_reswap_defaults_on_every_8_chunks(self) -> None:
        assert LiveEditStyleConfig().reswap_interval_chunks == 8

    def test_negative_reswap_interval_is_rejected(self) -> None:
        with pytest.raises(ValueError):
            LiveEditStyleConfig(reswap_interval_chunks=-1)
