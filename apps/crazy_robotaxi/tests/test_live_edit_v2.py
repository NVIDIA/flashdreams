# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""CPU regression tests for API-v2 live-edit composition."""

from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import crazy_robotaxi.live_edit.config as live_edit_config
import numpy as np
import pytest
import torch
from crazy_robotaxi.live_edit.config import (
    LiveEditCoinsConfig,
    LiveEditConfig,
    LiveEditItemsConfig,
    LiveEditMapContextConfig,
    LiveEditObstacleConfig,
    LiveEditStyleConfig,
    LiveEditWeatherConfig,
    StyleSkin,
    WeatherPreset,
    add_live_edit_args,
    live_edit_config_from_args,
    resolve_live_edit_assets,
)
from crazy_robotaxi.live_edit.item_ability import ItemEffects
from crazy_robotaxi.live_edit.nitro_ability import NitroAbility
from crazy_robotaxi.live_edit.obstacle_events import (
    ObstacleAbility,
    ObstacleEvent,
    ObstaclePhase,
)
from crazy_robotaxi.live_edit.obstacle_templates import load_obstacle_template_catalog
from crazy_robotaxi.live_edit.runtime_v2 import LiveEditGameplay, LiveEditGameRules
from crazy_robotaxi.live_edit.style_ability import StyleAbility
from crazy_robotaxi.live_edit.weather_ability import compose_swap_target
from crazy_robotaxi.navigation import NavigationLane
from ludus_renderer import SceneObject
from omnidreams_game_engine.config import VehicleConfig
from omnidreams_game_engine.contracts import GameRules, GameUpdate
from omnidreams_game_engine.game_map import load_game_map
from omnidreams_game_engine.types import (
    CameraCalibration,
    DynamicActorTrajectory,
    SceneDefinition,
    TrajectoryChunk,
    VehicleState,
)
from PIL import Image

pytestmark = pytest.mark.ci_cpu


def _scene(*, game_map: Any = None) -> SceneDefinition:
    calibration = CameraCalibration(
        clipgt_name="camera_front_wide_120fov",
        logical_name="camera_front_wide_120fov",
        width=3848,
        height=2168,
        cx=1924.0,
        cy=1084.0,
        polynomial=np.asarray([0.0, 1.0], dtype=np.float32),
        is_backward_polynomial=False,
        linear_cde=np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
        sensor_to_rig_flu=np.eye(4, dtype=np.float32),
    )
    return cast(
        SceneDefinition,
        SimpleNamespace(
            selected_camera=calibration,
            initial_rgb=np.zeros((640, 1168, 3), dtype=np.uint8),
            game_map=game_map,
            ground_mesh_vertices=None,
        ),
    )


def test_style_assets_download_only_when_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FLASHDREAMS_CACHE_DIR", str(tmp_path))
    assets = {
        live_edit_config._STYLE_LORA_URL: tmp_path / "style.pt",
        live_edit_config._STYLE_CORRECTOR_URL: tmp_path / "corrector.pt",
        live_edit_config._STYLE_GATE_URL: tmp_path / "gate.json",
        live_edit_config._BASE_CORRECTOR_URL: tmp_path / "base.pt",
    }
    downloads: list[tuple[str, Path]] = []

    def fake_download(url: str, *, cache_dir: Path) -> Path:
        downloads.append((url, cache_dir))
        return assets[url]

    monkeypatch.setattr(live_edit_config, "download_to_cache", fake_download)
    resolved = resolve_live_edit_assets(
        LiveEditConfig(style=LiveEditStyleConfig(enabled=True))
    )

    assert resolved.style.lora_checkpoint == tmp_path / "style.pt"
    assert resolved.style.corrector_checkpoint == tmp_path / "corrector.pt"
    assert resolved.style.gate_alpha_json == tmp_path / "gate.json"
    assert resolved.style.base_corrector_checkpoint == tmp_path / "base.pt"
    assert downloads
    assert all(
        directory == tmp_path / "crazy-robotaxi" / "live_edit"
        for _, directory in downloads
    )
    initial_downloads = downloads.copy()
    assert resolve_live_edit_assets(resolved) == resolved
    assert downloads == initial_downloads


@pytest.mark.parametrize("gain", (0.0, 0.1))
def test_weather_downloads_corrector_only_for_nonzero_gain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, gain: float
) -> None:
    assets = {
        live_edit_config._STYLE_CORRECTOR_URL: tmp_path / "corrector.pt",
        live_edit_config._STYLE_GATE_URL: tmp_path / "gate.json",
    }
    downloads: list[str] = []

    def fake_download(url: str, *, cache_dir: Path) -> Path:
        downloads.append(url)
        return assets[url]

    monkeypatch.setattr(live_edit_config, "download_to_cache", fake_download)
    resolved = resolve_live_edit_assets(
        LiveEditConfig(
            weather=LiveEditWeatherConfig(enabled=True, corrector_gain=gain)
        ),
        cache_dir=tmp_path,
    )

    assert resolved.style.lora_checkpoint is None
    assert resolved.style.base_corrector_checkpoint is None
    if gain == 0.0:
        assert downloads == []
        assert resolved.style.corrector_checkpoint is None
        assert resolved.style.gate_alpha_json is None
    else:
        assert resolved.style.corrector_checkpoint == tmp_path / "corrector.pt"
        assert resolved.style.gate_alpha_json == tmp_path / "gate.json"
        assert set(downloads) == set(assets)


def test_style_and_weather_items_request_named_effects() -> None:
    requests: list[tuple[str, str]] = []
    style = SimpleNamespace(
        skin_names=("comic",),
        weather_names=("rain", "snow"),
        request_skin=lambda name: requests.append(("style", name)) or name,
        request_weather=lambda name: requests.append(("weather", name)) or True,
    )
    effects = ItemEffects(style, LiveEditItemsConfig(mystery_seed=0))

    effects.apply("mystery")
    effects.apply("rain")
    assert requests == [("style", "comic"), ("weather", "rain")]


def test_hud_status_snapshots_live_gameplay_values() -> None:
    gameplay = LiveEditGameplay.__new__(LiveEditGameplay)
    gameplay.config = LiveEditConfig(
        weather=LiveEditWeatherConfig(enabled=True),
    )
    gameplay.style = SimpleNamespace(active_weather_name="rain")
    gameplay.coins = SimpleNamespace(enabled=True, collected_count=3)
    gameplay.nitro = NitroAbility(LiveEditItemsConfig(nitro_duration_s=2.0))
    gameplay.items = None
    gameplay.obstacles = None
    gameplay.nitro.activate()

    first = gameplay.hud_status()
    gameplay.coins.collected_count += 1
    gameplay.style.active_weather_name = "snow"
    gameplay.nitro.vehicle_for_tick(VehicleConfig(), 0.5)
    second = gameplay.hud_status()

    assert (first.coins_collected, first.weather_name) == (3, "rain")
    assert first.nitro_seconds_remaining == pytest.approx(2.0)
    assert (second.coins_collected, second.weather_name) == (4, "snow")
    assert second.nitro_seconds_remaining == pytest.approx(1.5)


def test_manual_actions_dispatch_to_the_requested_ability() -> None:
    requests: list[str] = []
    gameplay = LiveEditGameplay.__new__(LiveEditGameplay)
    gameplay.style = SimpleNamespace(
        request_cycle=lambda: requests.append("style"),
        request_weather_cycle=lambda: requests.append("weather"),
    )
    gameplay.coins = SimpleNamespace(toggle=lambda: requests.append("coins"))
    gameplay.obstacles = SimpleNamespace(
        request_spawn=lambda: requests.append("obstacle")
    )

    for action in ("style", "weather", "coins", "obstacle"):
        gameplay.request_action(action)
        assert requests == [action]
        requests.clear()


def test_live_edit_rules_preserve_actors_and_complete_frame_capture() -> None:
    completed: list[int] = []
    inner_actor = cast(DynamicActorTrajectory, object())
    live_edit_actor = cast(DynamicActorTrajectory, object())

    def advance_inner(_trajectory: object, _frame_interval_s: float) -> GameUpdate:
        return GameUpdate(frames=("frame",), dynamic_actors=(inner_actor,))

    inner = SimpleNamespace(is_running=True, advance_frames=advance_inner)
    gameplay = SimpleNamespace(
        begin_advance=lambda _trajectory: (live_edit_actor,),
        complete_advance=completed.append,
    )
    rules = LiveEditGameRules(
        cast(GameRules, inner),
        cast(LiveEditGameplay, gameplay),
    )

    update = rules.advance_frames(cast(Any, object()), 0.1)

    assert update.frames == ("frame",)
    assert update.dynamic_actors == (inner_actor, live_edit_actor)
    assert completed == [1]


def test_nitro_boosts_and_expires_on_game_time() -> None:
    config = LiveEditItemsConfig(
        enabled=True,
        nitro_boost=2.0,
        nitro_duration_s=0.2,
        nitro_max_speed_mps=16.0,
    )
    nitro = NitroAbility(config)
    vehicle = VehicleConfig(max_speed_mps=10.0, max_accel_mps2=3.0)
    nitro.activate()

    boosted = nitro.vehicle_for_tick(vehicle, 0.1)
    nitro.vehicle_for_tick(vehicle, 0.1)

    assert boosted.max_speed_mps == 16.0
    assert boosted.max_accel_mps2 == 3.0
    assert (
        nitro.boosted_vehicle(VehicleConfig(max_speed_mps=20.0)).max_speed_mps == 20.0
    )
    assert not nitro.active
    assert nitro.consume_frame_seconds(2) == pytest.approx((0.1, 0.0))


def test_v2_live_edit_camera_uses_generated_frame_size() -> None:
    gameplay = LiveEditGameplay(LiveEditConfig(), _scene(), (), vehicle=VehicleConfig())

    assert (gameplay._camera.output_width, gameplay._camera.output_height) == (
        1168,
        640,
    )


def test_v2_live_edit_loads_configured_sprites(tmp_path) -> None:
    coin_path = tmp_path / "coin.png"
    nitro_path = tmp_path / "nitro.png"
    Image.new("RGBA", (4, 4), (10, 20, 30, 255)).save(coin_path)
    Image.new("RGBA", (4, 4), (40, 50, 60, 255)).save(nitro_path)
    config = LiveEditConfig(
        coins=LiveEditCoinsConfig(enabled=True, sprite_path=coin_path),
        items=LiveEditItemsConfig(
            enabled=True,
            item_types=("nitro",),
            nitro_sprite_path=nitro_path,
        ),
    )
    lane = NavigationLane(
        np.asarray([[0.0, 0.0, 0.0], [100.0, 0.0, 0.0]], dtype=np.float32)
    )

    gameplay = LiveEditGameplay(config, _scene(), (lane,), vehicle=VehicleConfig())

    assert gameplay._compositor.sprite_image("coin").getpixel((0, 0)) == (
        10,
        20,
        30,
        255,
    )
    assert gameplay._compositor.sprite_image("nitro").getpixel((0, 0)) == (
        40,
        50,
        60,
        255,
    )


def test_physical_obstacle_lifetime_uses_relative_track_clock() -> None:
    relative_timestamps = np.asarray([0, 4_000_000], dtype=np.int64)
    event = ObstacleEvent(
        entity_id="live-edit-obstacle-test",
        object_type="Car",
        timestamps_us=relative_timestamps,
        translations_world=np.zeros((2, 3), dtype=np.float32),
        orientations_xyzw=np.tile(
            np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32), (2, 1)
        ),
        dimensions_lwh=np.asarray([4.0, 2.0, 1.5], dtype=np.float32),
        template_index=0,
        drive_speed_mps=4.0,
        scene_object=cast(
            SceneObject, SimpleNamespace(timestamps_us=relative_timestamps)
        ),
    )
    obstacle = ObstacleAbility.__new__(ObstacleAbility)
    obstacle._config = LiveEditObstacleConfig(
        enabled=True, physics=True, active_chunks=10
    )
    obstacle._events = [event]
    obstacle._chunk_index = 0
    state = VehicleState(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    trajectory = TrajectoryChunk(
        timestamps_us=np.asarray([1_000_000_000], dtype=np.int64),
        rig_poses_world=np.eye(4, dtype=np.float32)[None],
        vehicle_states=(state,),
        boundary_state_after_chunk=state,
    )

    obstacle.advance_frames(trajectory)

    assert event.phase is ObstaclePhase.SCRIPTED

    event.logical_timestamp_us = 4_000_000.0
    obstacle.advance_frames(trajectory)

    assert event.phase is ObstaclePhase.EXPIRED


def test_bundled_obstacle_catalog_has_usable_tracks() -> None:
    catalog = load_obstacle_template_catalog()

    assert catalog.moving(
        min_drift_m=15.0,
        min_coverage_s=4.0,
        length_range_m=(3.4, 5.6),
    )
    assert catalog.parked(length_range_m=(3.4, 5.6))


def _map_prompt_ability() -> tuple[StyleAbility, SimpleNamespace, list[Any]]:
    ability = StyleAbility(
        LiveEditStyleConfig(),
        map_context_config=LiveEditMapContextConfig(enabled=True),
    )
    session = SimpleNamespace(
        _cache=SimpleNamespace(transformer_cache=SimpleNamespace())
    )
    targets: list[Any] = []
    ability._session = session
    ability._base_prompt = "A sunny suburb."
    setattr(
        ability, "_replace_text", lambda active_session, target: targets.append(target)
    )
    return ability, session, targets


def test_weather_suffix_composes_without_changing_style_prompt() -> None:
    style_config = LiveEditStyleConfig(
        enabled=True,
        skins=(StyleSkin("comic", "Comic-book visuals."),),
        guidance_chunks=3,
    )
    weather_config = LiveEditWeatherConfig(
        enabled=True,
        weathers=(WeatherPreset("rain", "Heavy rain falls."),),
    )
    common = {
        "base_prompt": "A city road.",
        "map_prompt_suffix": "The taxi is driving forward.",
        "style_config": style_config,
        "weather_config": weather_config,
        "lora_available": True,
    }

    base = compose_swap_target(skin=None, weather=None, **common)
    style = compose_swap_target(skin=style_config.skins[0], weather=None, **common)
    weather = compose_swap_target(
        skin=None, weather=weather_config.weathers[0], **common
    )

    assert base.prompt == "A city road. The taxi is driving forward."
    assert style.prompt == "Comic-book visuals. The taxi is driving forward."
    assert style.use_lora is True
    assert style.guidance_chunks == 3
    assert weather.prompt == (
        "A city road. The taxi is driving forward. Heavy rain falls."
    )


def test_map_prompt_change_is_plain_and_deferred_during_guidance() -> None:
    ability, session, targets = _map_prompt_ability()
    ability._pending_map_suffix = "The taxi is driving forward."
    session._cache.transformer_cache.text_edit_guidance = SimpleNamespace(
        chunks_remaining=1
    )

    ability.before_v2_chunk()
    assert targets == []

    session._cache.transformer_cache.text_edit_guidance.chunks_remaining = 0
    ability.before_v2_chunk()

    assert len(targets) == 1
    target = targets[0]
    assert target.prompt == "A sunny suburb. The taxi is driving forward."
    assert target.guidance_scale == 1.0
    assert target.guidance_chunks == 0
    assert target.use_lora is False


def test_map_state_is_applied_at_the_post_simulation_model_boundary() -> None:
    state = VehicleState(1.0, 2.0, 0.0, 0.0, 3.0, 0.0)
    calls: list[object] = []
    gameplay = LiveEditGameplay.__new__(LiveEditGameplay)
    gameplay.style = SimpleNamespace(
        update_map_context=lambda value: calls.append(("update", value)),
        before_v2_chunk=lambda: calls.append("apply"),
    )
    gameplay.coins = None
    gameplay.items = None
    gameplay.effects = None
    gameplay.nitro = None
    gameplay.obstacles = None
    gameplay.guidance = None
    gameplay._frame_statuses = []

    gameplay.begin_advance(
        SimpleNamespace(boundary_state_after_chunk=state, vehicle_states=(state,))
    )
    gameplay.prepare_model_step(None, None, None, 0)

    assert calls == [("update", state), "apply"]


def test_visual_swap_absorbs_pending_map_change_once() -> None:
    style = LiveEditStyleConfig(
        enabled=True,
        lora_checkpoint=Path("/tmp/style.pt"),
        skins=(StyleSkin("comic", "Comic-book visuals."),),
        reswap_interval_chunks=0,
    )
    ability = StyleAbility(
        style,
        map_context_config=LiveEditMapContextConfig(enabled=True),
    )
    session = SimpleNamespace(
        _cache=SimpleNamespace(transformer_cache=SimpleNamespace())
    )
    targets: list[Any] = []
    ability._session = session
    ability._base_prompt = "A sunny suburb."
    ability._pending_index = 0
    ability._pending_map_suffix = "The taxi is driving forward."
    setattr(
        ability, "_replace_text", lambda active_session, target: targets.append(target)
    )

    ability.before_v2_chunk()

    assert len(targets) == 1
    assert targets[0].prompt == (
        f"{style.skins[0].prompt} The taxi is driving forward."
    )
    assert ability._pending_map_suffix is None
    assert ability.active_prompt == targets[0].prompt

    ability.before_v2_chunk()
    assert len(targets) == 1


def test_combined_map_prompts_are_encoded_lazily() -> None:
    ability = StyleAbility(
        LiveEditStyleConfig(
            enabled=True,
            skins=(StyleSkin("comic", "Comic-book visuals."),),
        ),
        map_context_config=LiveEditMapContextConfig(enabled=True),
    )
    prompts = (
        "A sunny suburb. The taxi is stationary.",
        "A sunny suburb. The taxi moves.",
    )
    embeddings = {
        prompts[0]: torch.ones((1, 2, 3)),
        prompts[1]: torch.full((1, 2, 3), 2.0),
    }
    encoded: list[str] = []
    replaced: list[torch.Tensor] = []

    def encode(texts: list[str]) -> torch.Tensor:
        assert len(texts) == 1
        encoded.append(texts[0])
        return embeddings[texts[0]]

    pipeline = SimpleNamespace(
        text_encoder=encode,
        replace_text_from_embeddings=lambda cache, value, **kwargs: replaced.append(
            value
        ),
    )
    session = SimpleNamespace(
        pipeline=pipeline,
        _cache=object(),
        _pending_finalization_index=None,
        replace_prompt=lambda *args, **kwargs: pytest.fail("expected embedding swap"),
    )
    ability._base_prompt = "A sunny suburb."
    ability._precompute_prompt_embeddings(pipeline)
    assert encoded == []

    for prompt in (prompts[0], prompts[0], prompts[1]):
        ability._replace_text(
            session,
            SimpleNamespace(
                prompt=prompt,
                guidance_scale=1.0,
                guidance_chunks=0,
                use_lora=False,
            ),
        )

    assert encoded == list(prompts)
    assert len(replaced) == 3
    assert replaced[0] is replaced[1]
    torch.testing.assert_close(replaced[0], embeddings[prompts[0]].unsqueeze(0))
    torch.testing.assert_close(replaced[2], embeddings[prompts[1]].unsqueeze(0))


def test_map_only_postprocessing_returns_original_video() -> None:
    scene = _scene(
        game_map=load_game_map(
            Path(__file__).parent / "maps" / "intersection_geometry.robotaxi.yaml"
        )
    )
    gameplay = LiveEditGameplay(
        LiveEditConfig(dynamic_prompts=LiveEditMapContextConfig(enabled=True)),
        scene,
        (),
        vehicle=VehicleConfig(),
    )
    video = object()

    assert gameplay.postprocess_video(cast(Any, video), None) is video


def test_dynamic_prompts_cli_enables_live_edit_runtime() -> None:
    parser = argparse.ArgumentParser()
    add_live_edit_args(parser)

    defaults = live_edit_config_from_args(parser.parse_args([]))
    assert not defaults.dynamic_prompts.enabled
    assert not defaults.any_enabled

    config = live_edit_config_from_args(
        parser.parse_args(["--live-edit-dynamic-prompts"])
    )

    assert config.dynamic_prompts.enabled
    assert config.any_enabled


def test_dynamic_prompts_need_enabled_and_an_included_fragment() -> None:
    inclusions_disabled = LiveEditMapContextConfig(
        enabled=True,
        include_current_road_context=False,
        include_next_map_node_type=False,
        include_next_map_node_context=False,
        include_upcoming_road_curve_direction=False,
        include_taxi_motion_state=False,
    )
    config = LiveEditConfig(dynamic_prompts=inclusions_disabled)

    assert not inclusions_disabled.active
    assert not config.any_enabled
    assert not config.requires_python_dit
