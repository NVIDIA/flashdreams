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
from crazy_robotaxi.live_edit.presenter import (
    LiveEditPresenter,
    procedural_coin_sprite,
    scaled_sprite_size,
    unsharp_rgb,
)
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

    def test_default_sprite_is_the_procedural_coin(self) -> None:
        sprite = LiveEditPresenter._load_sprite(None)

        reference = procedural_coin_sprite()
        assert sprite.size == reference.size
        assert sprite.mode == "RGBA"
        assert np.array_equal(np.asarray(sprite), np.asarray(reference))

    def test_explicit_sprite_path_overrides_the_procedural_coin(self, tmp_path) -> None:
        from PIL import Image

        path = tmp_path / "custom.png"
        Image.new("RGBA", (10, 20), (1, 2, 3, 4)).save(path)

        sprite = LiveEditPresenter._load_sprite(path)

        assert sprite.size == (10, 20)

    def test_sprite_scaling_preserves_aspect_and_applies_squash(self) -> None:
        wide, tall = scaled_sprite_size((407, 491), height_px=50.0, squash=1.0)
        assert tall == 50
        assert wide == round(50.0 * 407 / 491)

        squashed_w, squashed_h = scaled_sprite_size((407, 491), 50.0, 0.5)
        assert squashed_h == 50
        assert squashed_w == round(50.0 * 407 / 491 * 0.5)

        # Tiny/edge cases stay drawable.
        assert scaled_sprite_size((407, 491), 0.5, 0.3) == (2, 3)

    def test_composites_a_coin_straddling_the_image_edge(self) -> None:
        from dataclasses import dataclass, field
        from typing import Any

        from omnidreams_game_engine.types import PresentedFrame

        @dataclass
        class RecordingPresenter:
            frames: list[Any] = field(default_factory=list)

            def present_frame(self, frame: Any, view_mode: str) -> None:
                self.frames.append(frame)

        config = LiveEditConfig(coins=_coins_config(coin_diameter_m=8.0))
        # Close and high coin: center projects in-image but the sprite
        # extends past the top edge; must clip, not raise.
        ability = CoinAbility(
            np.array([[4.0, 0.0, 2.5]], dtype=np.float32), config.coins
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

        out = np.asarray(inner.frames[0].model_rgb_host_uint8)
        assert not np.array_equal(out, rgb)

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
        assert kwargs["guidance_chunks"] == 6

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
        assert kwargs["guidance_chunks"] == 6
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


class _FakeTextEncoder:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def __call__(self, prompts: list[str]) -> object:
        import torch

        self.calls.append(list(prompts))
        return torch.full((len(prompts), 2, 2), float(len(self.calls)))


class _EmbeddingPipeline(_FakePipeline):
    """Fake pipeline with a resident text encoder + embedding swap path."""

    def __init__(self) -> None:
        super().__init__()
        self.text_encoder = _FakeTextEncoder()
        self.embedding_calls: list[tuple] = []

    def replace_text_from_embeddings(
        self, cache: object, embeddings: object, **kwargs: object
    ) -> None:
        self.embedding_calls.append((cache, embeddings, kwargs))
        self.call_order.append("replace_text_from_embeddings")


def _embedding_style_ability() -> tuple[StyleAbility, _FakeSession]:
    ability, session = _hooked_style_ability()
    session.pipeline = _EmbeddingPipeline()
    return ability, session


class TestPreEncodedSwaps:
    def test_precompute_encodes_each_configured_prompt_once(self) -> None:
        ability, session = _embedding_style_ability()
        ability._precompute_prompt_embeddings(session.pipeline)

        assert session.pipeline.text_encoder.calls == [
            ["arcade prompt"],
            ["comic prompt"],
        ]
        ability._precompute_prompt_embeddings(session.pipeline)
        assert len(session.pipeline.text_encoder.calls) == 2  # cache hit

    def test_swap_injects_cached_embeddings_without_reencoding(self) -> None:
        ability, session = _embedding_style_ability()
        ability._precompute_prompt_embeddings(session.pipeline)
        session.start(None, [], "scene prompt")  # also encodes the base prompt
        encoder_calls = len(session.pipeline.text_encoder.calls)

        ability.request_cycle()
        session.continue_generation([])

        assert session.pipeline.replace_text_calls == []
        ((_, embeddings, kwargs),) = session.pipeline.embedding_calls
        assert embeddings is ability._prompt_embeddings["arcade prompt"]
        assert kwargs["guidance_scale"] == 2.5
        assert len(session.pipeline.text_encoder.calls) == encoder_calls

    def test_revert_to_base_uses_the_embedding_cached_at_start(self) -> None:
        ability, session = _embedding_style_ability()
        ability._precompute_prompt_embeddings(session.pipeline)
        session.start(None, [], "scene prompt")
        for _ in range(3):  # arcade -> comic -> base
            ability.request_cycle()
            session.continue_generation([])

        assert session.pipeline.replace_text_calls == []
        _, embeddings, kwargs = session.pipeline.embedding_calls[-1]
        assert embeddings is ability._prompt_embeddings["scene prompt"]
        assert kwargs["guidance_scale"] == 1.0

    def test_uncached_prompt_falls_back_to_replace_text(self) -> None:
        ability, session = _embedding_style_ability()
        # No precompute and no encoder at start time -> nothing cached.
        session.pipeline.text_encoder = None
        session.start(None, [], "scene prompt")
        ability.request_cycle()
        session.continue_generation([])

        assert session.pipeline.embedding_calls == []
        ((_, text, _),) = session.pipeline.replace_text_calls
        assert text == [["arcade prompt"]]

    def test_pipeline_without_embedding_api_falls_back(self) -> None:
        ability, session = _hooked_style_ability()  # plain _FakePipeline
        session.start(None, [], "scene prompt")
        ability.request_cycle()
        session.continue_generation([])

        ((_, text, _),) = session.pipeline.replace_text_calls
        assert text == [["arcade prompt"]]


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


class TestGuidanceChunkFlags:
    def test_skin_and_weather_guidance_chunk_flags_round_trip(self) -> None:
        parser = argparse.ArgumentParser()
        add_live_edit_args(parser)
        args = parser.parse_args(
            [
                "--live-edit-style",
                "--live-edit-style-lora",
                "/tmp/lora.pt",
                "--live-edit-skin-guidance-chunks",
                "4",
                "--live-edit-weather",
                "--live-edit-weather-guidance-chunks",
                "12",
            ]
        )
        config = live_edit_config_from_args(args)
        assert config.style.guidance_chunks == 4
        assert config.weather.guidance_chunks == 12

    def test_skin_and_weather_landing_windows_default_short(self) -> None:
        parser = argparse.ArgumentParser()
        add_live_edit_args(parser)
        config = live_edit_config_from_args(parser.parse_args([]))
        assert config.style.guidance_chunks == 6
        assert config.weather.guidance_chunks == 6
        assert config.weather.maintain_interval_chunks == 0

    def test_weather_maintenance_flags_round_trip(self) -> None:
        parser = argparse.ArgumentParser()
        add_live_edit_args(parser)
        args = parser.parse_args(
            [
                "--live-edit-weather",
                "--live-edit-weather-maintain-interval",
                "16",
                "--live-edit-weather-maintain-chunks",
                "3",
            ]
        )
        config = live_edit_config_from_args(args)
        assert config.weather.maintain_interval_chunks == 16
        assert config.weather.maintain_chunks == 3

    def test_corrector_mode_off_round_trips(self) -> None:
        parser = argparse.ArgumentParser()
        add_live_edit_args(parser)
        args = parser.parse_args(["--live-edit-corrector-mode", "off"])
        config = live_edit_config_from_args(args)
        assert config.style.corrector_mode == "off"


class TestTorchCompositor:
    """Device-agnostic torch compositing (CPU tensors run the CUDA code)."""

    def _presenter_with_coin(self) -> LiveEditPresenter:
        class RecordingPresenter:
            def __init__(self) -> None:
                self.frames: list[object] = []

            def present_frame(self, frame: object, view_mode: str) -> None:
                self.frames.append(frame)

        config = LiveEditConfig(coins=_coins_config())
        ability = CoinAbility(
            np.array([[15.0, 0.0, 0.0]], dtype=np.float32), config.coins
        )
        presenter = LiveEditPresenter(
            RecordingPresenter(), config, coin_ability=ability
        )
        presenter.configure_taxi_camera(_test_calibration())
        presenter._allow_cpu_tensor_source = True
        return presenter

    class _FakeLazyTensorFrame:
        """LazyCudaFrame duck-type over a CPU torch tensor."""

        def __init__(self, tensor: object) -> None:
            self._tensor = tensor
            self.cuda_tensor_calls = 0

        @property
        def shape(self) -> tuple[int, ...]:
            return tuple(self._tensor.shape)

        def to_cuda_tensor(self) -> object:
            self.cuda_tensor_calls += 1
            return self._tensor

        def to_cuda_event(self) -> None:
            return None

        def to_numpy(self) -> object:
            raise AssertionError("torch path must not materialize the source")

    def _frame(self, source: object) -> object:
        from omnidreams_game_engine.types import PresentedFrame

        return PresentedFrame(
            timestamp_us=1,
            rgb_host_uint8=None,
            depth_host_f32=None,
            model_rgb_host_uint8=source,
            rig_to_world=np.eye(4, dtype=np.float32),
        )

    def test_tensor_source_composites_without_a_host_round_trip(self) -> None:
        import torch

        presenter = self._presenter_with_coin()
        base = torch.full((704, 1280, 3), 90, dtype=torch.uint8)
        source = self._FakeLazyTensorFrame(base.clone())
        presenter.present_frame(self._frame(source), "model_rgb")

        (out_frame,) = presenter._inner.frames
        out = out_frame.model_rgb_host_uint8
        assert callable(getattr(out, "to_cuda_tensor", None))
        composited = out.to_cuda_tensor()
        assert torch.is_tensor(composited)
        assert not torch.equal(composited, base)
        # Counter chip occupies the top-left corner.
        assert not torch.equal(composited[:40, :120], base[:40, :120])
        # A golden coin landed near the projected center.
        changed = (composited != base).any(dim=-1)
        assert changed[100:650, 200:1100].any()

    def test_speculative_prepare_and_present_share_one_composite(self) -> None:
        import torch

        presenter = self._presenter_with_coin()
        base = torch.full((704, 1280, 3), 90, dtype=torch.uint8)
        source = self._FakeLazyTensorFrame(base.clone())
        frame = self._frame(source)
        presenter.prepare_frame(frame, "model_rgb")
        presenter.present_frame(frame, "model_rgb")
        assert source.cuda_tensor_calls == 1

    def test_annotate_flag_forces_the_host_path(self) -> None:
        import torch
        from crazy_robotaxi.live_edit.config import LiveEditObstacleConfig

        class ObstacleStub:
            active = False
            hit_count = 0
            event = None
            events: tuple = ()

        config = LiveEditConfig(
            coins=_coins_config(),
            obstacle=LiveEditObstacleConfig(enabled=True, annotate=True),
        )
        presenter = LiveEditPresenter(object(), config)
        presenter._allow_cpu_tensor_source = True
        presenter.set_obstacle_ability(ObstacleStub())
        source = self._FakeLazyTensorFrame(torch.full((8, 8, 3), 90, dtype=torch.uint8))
        assert presenter._process_tensor(self._frame(source)) is None

    def test_alpha_blend_clips_at_canvas_edges(self) -> None:
        import torch
        from crazy_robotaxi.live_edit.gpu_compositor import alpha_blend_

        canvas = torch.zeros((10, 10, 3), dtype=torch.uint8)
        rgb = torch.full((3, 4, 4), 200.0)
        alpha = torch.ones((1, 4, 4))
        alpha_blend_(canvas, rgb, alpha, -2, -2)  # top-left overhang
        alpha_blend_(canvas, rgb, alpha, 8, 8)  # bottom-right overhang
        alpha_blend_(canvas, rgb, alpha, 20, 20)  # fully outside: no-op
        assert canvas[0, 0].tolist() == [200, 200, 200]
        assert canvas[1, 1].tolist() == [200, 200, 200]
        assert canvas[9, 9].tolist() == [200, 200, 200]
        assert canvas[5, 5].tolist() == [0, 0, 0]

    def test_chip_textures_are_cached_per_label(self) -> None:
        import torch
        from crazy_robotaxi.live_edit.gpu_compositor import LiveEditFrameCompositor

        compositor = LiveEditFrameCompositor(procedural_coin_sprite())
        device = torch.device("cpu")
        first = compositor._chip("COINS 3", device)
        again = compositor._chip("COINS 3", device)
        other = compositor._chip("COINS 4", device)
        assert first[0] is again[0]
        assert other[0] is not first[0]

    def test_torch_unsharp_increases_edge_contrast(self) -> None:
        import torch
        from crazy_robotaxi.live_edit.gpu_compositor import LiveEditFrameCompositor

        frame = np.full((32, 32, 3), 40, dtype=np.uint8)
        frame[:, 16:] = 200
        compositor = LiveEditFrameCompositor(procedural_coin_sprite())
        sharpened = compositor.unsharp(
            torch.from_numpy(frame.copy()), sigma=2.0, amount=0.8
        ).numpy()
        assert int(sharpened[16, 15].mean()) < 40
        assert int(sharpened[16, 16].mean()) > 200
        identity = compositor.unsharp(
            torch.from_numpy(frame.copy()), sigma=2.0, amount=0.0
        ).numpy()
        assert np.array_equal(identity, frame)

    def test_torch_and_host_paths_draw_the_coin_in_the_same_region(self) -> None:
        import torch

        # Host reference.
        host_presenter = self._presenter_with_coin()
        host_presenter._allow_cpu_tensor_source = False
        rgb = np.full((704, 1280, 3), 90, dtype=np.uint8)
        host_presenter.present_frame(self._frame(rgb.copy()), "model_rgb")
        host_out = np.asarray(host_presenter._inner.frames[0].model_rgb_host_uint8)

        torch_presenter = self._presenter_with_coin()
        source = self._FakeLazyTensorFrame(torch.from_numpy(rgb.copy()))
        torch_presenter.present_frame(self._frame(source), "model_rgb")
        torch_out = (
            torch_presenter._inner.frames[0].model_rgb_host_uint8.to_cuda_tensor()
        ).numpy()

        host_mask = (host_out != rgb).any(axis=-1)
        torch_mask = (torch_out != rgb).any(axis=-1)
        # Skip the chip corner; compare the coin footprints.
        host_mask[:60, :200] = False
        torch_mask[:60, :200] = False
        assert host_mask.any() and torch_mask.any()
        overlap = (host_mask & torch_mask).sum()
        union = (host_mask | torch_mask).sum()
        assert overlap / union > 0.5  # same place, resampler differences ok


class TestCoinSpatialWindow:
    """O(nearby) windowed culling must match the brute-force reference."""

    def _course(self) -> np.ndarray:
        xs = np.arange(0.0, 2000.0, 5.0, dtype=np.float32)
        near_lane = np.stack([xs, np.zeros_like(xs), np.full_like(xs, 0.8)], axis=1)
        far_lane = np.stack([xs, np.full_like(xs, 60.0), np.full_like(xs, 0.8)], axis=1)
        return np.concatenate([near_lane, far_lane])

    def _brute_force(
        self, coins: np.ndarray, config: LiveEditCoinsConfig
    ) -> CoinAbility:
        """Reference ability whose grid degenerates to one all-coins cell."""
        from crazy_robotaxi.live_edit import coin_ability as module

        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(module, "_GRID_CELL_M", 1.0e9)
            return CoinAbility(coins, config)

    def test_windowed_projection_matches_brute_force_across_cells(self) -> None:
        from omnidreams_game_engine.camera import FThetaCameraModel

        config = _coins_config()
        coins = self._course()
        windowed = CoinAbility(coins, config)
        reference = self._brute_force(coins, config)
        camera = FThetaCameraModel(_test_calibration())
        for x in (0.0, 31.9, 32.1, 63.9, 100.0, 500.5, 1999.0):
            rig = np.eye(4, dtype=np.float32)
            rig[0, 3] = np.float32(x)
            kwargs = {"image_width": 1280, "image_height": 704}
            assert windowed.visible_sprites(
                rig, camera, **kwargs
            ) == reference.visible_sprites(rig, camera, **kwargs), f"x={x}"

    def test_pickup_matches_brute_force_across_cell_boundaries(self) -> None:
        config = _coins_config(pickup_radius_m=2.5)
        coins = np.array([[32.1, 0.0, 0.8], [31.9, 0.0, 0.8]], dtype=np.float32)
        windowed = CoinAbility(coins, config)
        reference = self._brute_force(coins, config)
        states = [_ego_at(30.0), _ego_at(33.0)]
        assert windowed.advance_frames(states) == reference.advance_frames(states) == 2

    def test_window_is_a_superset_of_the_exact_radius(self) -> None:
        from crazy_robotaxi.live_edit.coin_ability import _CoinGrid

        rng = np.random.default_rng(7)
        coins_xy = rng.uniform(-500.0, 500.0, size=(4400, 2)).astype(np.float32)
        grid = _CoinGrid(coins_xy)
        for x, y, radius in ((0.0, 0.0, 120.0), (-431.0, 250.0, 2.5)):
            window = set(grid.near(x, y, radius).tolist())
            exact = np.flatnonzero(
                np.linalg.norm(coins_xy - np.array([x, y], dtype=np.float32), axis=1)
                <= radius
            )
            assert set(exact.tolist()) <= window

    def test_window_query_is_cached_per_cell(self) -> None:
        from crazy_robotaxi.live_edit.coin_ability import _CoinGrid

        grid = _CoinGrid(np.zeros((10, 2), dtype=np.float32))
        first = grid.near(1.0, 1.0, 120.0)
        again = grid.near(5.0, 9.0, 120.0)  # same 32 m cell
        assert again is first


class TestRoiCompositor:
    """The uint8 ROI path must match the float full-frame blends."""

    def _sprites(self):
        from crazy_robotaxi.live_edit.coin_ability import CoinSprite

        return (
            CoinSprite(
                center_uv=(200.0, 300.0),
                height_px=40.0,
                alpha=1.0,
                distance_m=20.0,
                spin_phase=0.0,
            ),
            # Overlaps the first coin and carries a distance fade.
            CoinSprite(
                center_uv=(215.0, 310.0),
                height_px=30.0,
                alpha=0.4,
                distance_m=90.0,
                spin_phase=1.2,
            ),
            # Straddles the frame edge.
            CoinSprite(
                center_uv=(2.0, 700.0),
                height_px=24.0,
                alpha=0.8,
                distance_m=50.0,
                spin_phase=2.4,
            ),
        )

    def test_roi_path_matches_float_path_within_one_lsb(self) -> None:
        import torch
        from crazy_robotaxi.live_edit.gpu_compositor import (
            LiveEditFrameCompositor,
            _blend_float_,
        )

        rng = np.random.default_rng(3)
        frame = torch.from_numpy(
            rng.integers(0, 256, size=(704, 1280, 3), dtype=np.uint8)
        )
        compositor = LiveEditFrameCompositor(procedural_coin_sprite())
        compositor._roi_blends = True
        sprites = self._sprites()
        labels = ["COINS 7", "SKIN ARCADE"]

        out = compositor.composite(frame, sprites=sprites, frame_index=5, labels=labels)

        reference = frame.to(torch.float32)
        compositor._blend_coins(reference, sprites, 5, _blend_float_)
        compositor._blend_chips(reference, labels, _blend_float_)
        reference = reference.round_().clamp_(0.0, 255.0).to(torch.uint8)
        diff = (out.int() - reference.int()).abs()
        assert diff.max().item() <= 1
        # Blends changed something.
        assert not torch.equal(out, frame)

    def test_composite_without_edits_copies_and_leaves_the_source_alone(self) -> None:
        import torch
        from crazy_robotaxi.live_edit.gpu_compositor import LiveEditFrameCompositor

        frame = torch.full((32, 32, 3), 90, dtype=torch.uint8)
        compositor = LiveEditFrameCompositor(procedural_coin_sprite())
        compositor._roi_blends = True
        out = compositor.composite(frame)
        assert torch.equal(out, frame)
        assert out is not frame

    def test_roi_path_does_not_mutate_the_source_frame(self) -> None:
        import torch
        from crazy_robotaxi.live_edit.gpu_compositor import LiveEditFrameCompositor

        frame = torch.full((704, 1280, 3), 90, dtype=torch.uint8)
        pristine = frame.clone()
        compositor = LiveEditFrameCompositor(procedural_coin_sprite())
        compositor._roi_blends = True
        out = compositor.composite(frame, sprites=self._sprites(), labels=["COINS 1"])
        assert torch.equal(frame, pristine)
        assert not torch.equal(out, pristine)

    def test_env_switch_selects_the_blend_path(self, monkeypatch) -> None:
        from crazy_robotaxi.live_edit.gpu_compositor import LiveEditFrameCompositor

        sprite = procedural_coin_sprite()
        monkeypatch.delenv("LIVE_EDIT_COMPOSITOR", raising=False)
        assert LiveEditFrameCompositor(sprite)._roi_blends is False
        monkeypatch.setenv("LIVE_EDIT_COMPOSITOR", "roi")
        assert LiveEditFrameCompositor(sprite)._roi_blends is True


class TestPerfLog:
    def test_perf_log_reports_percentiles_every_n_frames(self) -> None:
        import torch
        from loguru import logger
        from omnidreams_game_engine.types import PresentedFrame

        class RecordingPresenter:
            def __init__(self) -> None:
                self.frames: list[object] = []

            def present_frame(self, frame: object, view_mode: str) -> None:
                self.frames.append(frame)

        class LazyTensorFrame:
            def __init__(self, tensor: object) -> None:
                self._tensor = tensor

            def to_cuda_tensor(self) -> object:
                return self._tensor

            def to_cuda_event(self) -> None:
                return None

        config = LiveEditConfig(coins=_coins_config(), perf_log_every_frames=2)
        ability = CoinAbility(
            np.array([[15.0, 0.0, 0.0]], dtype=np.float32), config.coins
        )
        presenter = LiveEditPresenter(
            RecordingPresenter(), config, coin_ability=ability
        )
        presenter.configure_taxi_camera(_test_calibration())
        presenter._allow_cpu_tensor_source = True

        messages: list[str] = []
        sink_id = logger.add(lambda message: messages.append(str(message)))
        try:
            for timestamp in (1, 2):
                frame = PresentedFrame(
                    timestamp_us=timestamp,
                    rgb_host_uint8=None,
                    depth_host_f32=None,
                    model_rgb_host_uint8=LazyTensorFrame(
                        torch.full((704, 1280, 3), 90, dtype=torch.uint8)
                    ),
                    rig_to_world=np.eye(4, dtype=np.float32),
                )
                presenter.present_frame(frame, "model_rgb")
        finally:
            logger.remove(sink_id)
        perf_lines = [line for line in messages if "[live-edit] perf over 2" in line]
        assert len(perf_lines) == 1
        assert "coin_cpu_ms p50=" in perf_lines[0]
        assert "compositor_enqueue_cpu_ms p50=" in perf_lines[0]

    def test_perf_log_flag_round_trip_and_env_default(self, monkeypatch) -> None:
        parser = argparse.ArgumentParser()
        add_live_edit_args(parser)
        args = parser.parse_args(["--live-edit-perf-log", "240"])
        assert live_edit_config_from_args(args).perf_log_every_frames == 240

        monkeypatch.setenv("LIVE_EDIT_PERF_LOG", "120")
        env_parser = argparse.ArgumentParser()
        add_live_edit_args(env_parser)
        env_args = env_parser.parse_args([])
        assert live_edit_config_from_args(env_args).perf_log_every_frames == 120

    def test_negative_perf_log_interval_is_rejected(self) -> None:
        with pytest.raises(ValueError):
            LiveEditConfig(perf_log_every_frames=-1)


class TestSpriteCap:
    def test_cap_keeps_the_nearest_coins_in_painter_order(self) -> None:
        from omnidreams_game_engine.camera import FThetaCameraModel

        config = _coins_config(max_render_distance_m=120.0, max_visible_sprites=2)
        coins = np.array(
            [[10.0, 0.0, 0.0], [40.0, 0.0, 0.0], [80.0, 0.0, 0.0]],
            dtype=np.float32,
        )
        ability = CoinAbility(coins, config)
        camera = FThetaCameraModel(_test_calibration())

        sprites = ability.visible_sprites(
            np.eye(4, dtype=np.float32), camera, image_width=1280, image_height=704
        )

        assert len(sprites) == 2
        # The farthest coin (80 m) dropped; order stays far-to-near.
        assert sprites[0].distance_m == pytest.approx(40.0)
        assert sprites[1].distance_m == pytest.approx(10.0)

    def test_zero_cap_disables_the_limit(self) -> None:
        from omnidreams_game_engine.camera import FThetaCameraModel

        config = _coins_config(max_visible_sprites=0)
        coins = np.array(
            [[10.0, 0.0, 0.0], [40.0, 0.0, 0.0], [80.0, 0.0, 0.0]],
            dtype=np.float32,
        )
        ability = CoinAbility(coins, config)
        camera = FThetaCameraModel(_test_calibration())
        sprites = ability.visible_sprites(
            np.eye(4, dtype=np.float32), camera, image_width=1280, image_height=704
        )
        assert len(sprites) == 3

    def test_cap_flag_round_trip(self) -> None:
        parser = argparse.ArgumentParser()
        add_live_edit_args(parser)
        args = parser.parse_args(["--live-edit-coin-max-visible", "32"])
        assert live_edit_config_from_args(args).coins.max_visible_sprites == 32
        assert (
            live_edit_config_from_args(parser.parse_args([])).coins.max_visible_sprites
            == 64
        )
