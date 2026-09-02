# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""CPU regression tests for API-v2 live-edit composition."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import crazy_robotaxi.live_edit.config as live_edit_config
import numpy as np
import pytest
from crazy_robotaxi.live_edit.config import (
    LiveEditCoinsConfig,
    LiveEditConfig,
    LiveEditItemsConfig,
    LiveEditMapContextConfig,
    LiveEditObstacleConfig,
    LiveEditStyleConfig,
    LiveEditWeatherConfig,
    add_live_edit_args,
    live_edit_config_from_args,
    resolve_live_edit_assets,
)
from crazy_robotaxi.live_edit.nitro_ability import NitroAbility
from crazy_robotaxi.live_edit.item_ability import ItemEffects
from crazy_robotaxi.live_edit.obstacle_events import (
    ObstacleAbility,
    ObstacleEvent,
    ObstaclePhase,
)
from crazy_robotaxi.live_edit.obstacle_templates import load_obstacle_template_catalog
from crazy_robotaxi.live_edit.runtime_v2 import LiveEditGameplay, LiveEditGameRules
from crazy_robotaxi.live_edit.style_ability import StyleAbility
from crazy_robotaxi.navigation import NavigationLane
from ludus_renderer import SceneObject
from omnidreams_game_engine.config import VehicleConfig
from omnidreams_game_engine.contracts import GameUpdate
from omnidreams_game_engine.game_map import load_game_map
from omnidreams_game_engine.types import (
    CameraCalibration,
    SceneDefinition,
    TrajectoryChunk,
    VehicleState,
)
from PIL import Image

pytestmark = pytest.mark.ci_cpu


class _StyleRequests:
    def __init__(self) -> None:
        self.skin_cycles = 0
        self.weather_cycles = 0
        self.active_skin_name = "comic"
        self.active_weather_name = "rain"

    def request_cycle(self) -> None:
        self.skin_cycles += 1

    def request_weather_cycle(self) -> None:
        self.weather_cycles += 1


class _Coins:
    def __init__(self) -> None:
        self.toggles = 0
        self.enabled = True
        self.collected_count = 3

    def toggle(self) -> bool:
        self.toggles += 1
        return True


class _Obstacles:
    def __init__(self) -> None:
        self.spawns = 0
        self.events = (object(), object())

    def request_spawn(self) -> None:
        self.spawns += 1


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
    downloads: list[str] = []

    def fake_download(url: str, *, cache_dir: Path) -> Path:
        downloads.append(url)
        return cache_dir / Path(url).name

    monkeypatch.setattr(live_edit_config, "download_to_cache", fake_download)

    resolved = resolve_live_edit_assets(
        LiveEditConfig(style=LiveEditStyleConfig(enabled=True)),
        cache_dir=tmp_path,
    )

    paths = (
        resolved.style.lora_checkpoint,
        resolved.style.corrector_checkpoint,
        resolved.style.gate_alpha_json,
        resolved.style.base_corrector_checkpoint,
    )
    assert [path.name for path in paths if path is not None] == [
        "lora_style_v6_step1600.pt",
        "lora_style_corrector_v5_valpeak.pt",
        "gate_style_v5.json",
        "lora_v2_v3_valpeak.pt",
    ]
    assert len(downloads) == 4


def test_explicit_style_assets_skip_download(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        live_edit_config,
        "download_to_cache",
        lambda *args, **kwargs: pytest.fail("explicit assets must not download"),
    )
    style = LiveEditStyleConfig(
        enabled=True,
        lora_checkpoint=tmp_path / "style.pt",
        corrector_checkpoint=tmp_path / "corrector.pt",
        gate_alpha_json=tmp_path / "gate.json",
        base_corrector_checkpoint=tmp_path / "base.pt",
    )
    config = LiveEditConfig(style=style)

    assert resolve_live_edit_assets(config, cache_dir=tmp_path) == config


def test_weather_downloads_corrector_only_for_nonzero_gain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    downloads: list[str] = []

    def fake_download(url: str, *, cache_dir: Path) -> Path:
        downloads.append(url)
        return cache_dir / Path(url).name

    monkeypatch.setattr(live_edit_config, "download_to_cache", fake_download)
    resolved = resolve_live_edit_assets(
        LiveEditConfig(weather=LiveEditWeatherConfig(enabled=True, corrector_gain=0.1)),
        cache_dir=tmp_path,
    )

    assert resolved.style.lora_checkpoint is None
    assert resolved.style.base_corrector_checkpoint is None
    assert resolved.style.corrector_checkpoint is not None
    assert resolved.style.gate_alpha_json is not None
    assert resolved.style.corrector_checkpoint.name == (
        "lora_style_corrector_v5_valpeak.pt"
    )
    assert resolved.style.gate_alpha_json.name == "gate_style_v5.json"
    assert len(downloads) == 2


def test_v2_manual_actions_share_keyboard_dispatch() -> None:
    gameplay = LiveEditGameplay.__new__(LiveEditGameplay)
    gameplay.style = _StyleRequests()
    gameplay.coins = _Coins()
    gameplay.obstacles = _Obstacles()

    for action in ("style", "weather", "coins", "obstacle"):
        gameplay.request_action(action)

    assert gameplay.style.skin_cycles == 1
    assert gameplay.style.weather_cycles == 1
    assert gameplay.coins.toggles == 1
    assert gameplay.obstacles.spawns == 1


def test_v2_hud_status_snapshots_enabled_abilities() -> None:
    gameplay = LiveEditGameplay.__new__(LiveEditGameplay)
    gameplay.config = LiveEditConfig(
        style=LiveEditStyleConfig(enabled=True),
        weather=LiveEditWeatherConfig(enabled=True),
        coins=LiveEditCoinsConfig(enabled=True),
        obstacle=LiveEditObstacleConfig(enabled=True),
    )
    gameplay.style = _StyleRequests()
    gameplay.coins = _Coins()
    gameplay.nitro = SimpleNamespace(active=True, seconds_remaining=4.0)
    gameplay.items = SimpleNamespace(flash_label="NITRO BOOST")
    gameplay.obstacles = _Obstacles()

    status = gameplay.hud_status()

    assert status.skin_name == "comic"
    assert status.weather_name == "rain"
    assert status.coins_enabled
    assert status.coins_collected == 3
    assert status.nitro_seconds_remaining == 4.0
    assert status.item_flash == "NITRO BOOST"
    assert status.obstacle_count == 2


def test_style_and_weather_items_request_persistent_selections() -> None:
    requests: list[tuple[str, str]] = []
    style = SimpleNamespace(
        skin_names=("comic",),
        weather_names=("rain", "snow"),
        request_skin=lambda name: requests.append(("style", name)) or name,
        request_weather=lambda name: requests.append(("weather", name)) or True,
    )
    effects = ItemEffects(style, LiveEditItemsConfig(mystery_seed=0))

    assert effects.apply("mystery") == "? COMIC!"
    assert effects.apply("rain") == "RAIN!"
    assert requests == [("style", "comic"), ("weather", "rain")]


def test_live_edit_rules_award_coins_before_taxi_snapshots() -> None:
    awarded: list[int] = []

    def advance_inner(_trajectory: object, _frame_interval_s: float) -> GameUpdate:
        return GameUpdate(frames=(sum(awarded),))

    inner = SimpleNamespace(is_running=True, advance_frames=advance_inner)
    gameplay = SimpleNamespace(advance=lambda _trajectory: (("actor",), 2))
    rules = LiveEditGameRules(
        inner,
        gameplay,
        coin_collected=lambda count: awarded.append(count * 100),
    )

    update = rules.advance_frames(cast(Any, object()), 0.1)

    assert update.frames == (200,)
    assert update.dynamic_actors == ("actor",)


@pytest.mark.parametrize(
    "config",
    [
        LiveEditConfig(style=LiveEditStyleConfig(enabled=True)),
        LiveEditConfig(weather=LiveEditWeatherConfig(enabled=True)),
        LiveEditConfig(obstacle=LiveEditObstacleConfig(enabled=True, guide_scale=1.0)),
    ],
)
def test_prompt_live_edit_requires_python_dit(config: LiveEditConfig) -> None:
    assert config.requires_python_dit


@pytest.mark.parametrize(
    "config",
    [
        LiveEditConfig(coins=LiveEditCoinsConfig(enabled=True)),
        LiveEditConfig(items=LiveEditItemsConfig(enabled=True)),
        LiveEditConfig(obstacle=LiveEditObstacleConfig(enabled=True, guide_scale=0.0)),
    ],
)
def test_pixel_live_edit_keeps_native_dit_compatible(config: LiveEditConfig) -> None:
    assert not config.requires_python_dit


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


def test_v2_live_edit_camera_uses_generated_frame_size() -> None:
    gameplay = LiveEditGameplay(LiveEditConfig(), _scene(), (), vehicle=VehicleConfig())

    assert (gameplay._camera.output_width, gameplay._camera.output_height) == (
        1168,
        640,
    )
    assert gameplay._compositor.sprite_image("coin").getpixel((15, 15)) == (
        0,
        0,
        0,
        0,
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


def test_bundled_obstacle_catalog_matches_source_branch() -> None:
    catalog = load_obstacle_template_catalog()

    assert len(catalog.templates) == 668
    assert (
        len(
            catalog.moving(
                min_drift_m=15.0,
                min_coverage_s=4.0,
                length_range_m=(3.4, 5.6),
            )
        )
        == 63
    )
    assert len(catalog.parked(length_range_m=(3.4, 5.6))) == 236


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
    gameplay.obstacles = None
    gameplay.guidance = None

    gameplay.advance(SimpleNamespace(boundary_state_after_chunk=state))
    gameplay.prepare_model_step(None, None, None, 0)

    assert calls == [("update", state), "apply"]


def test_visual_swap_absorbs_pending_map_change_once() -> None:
    style = replace(
        LiveEditStyleConfig(),
        enabled=True,
        lora_checkpoint=Path("/tmp/style.pt"),
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
    assert targets[0].prompt.endswith("The taxi is driving forward.")
    assert ability._pending_map_suffix is None


def test_combined_map_prompts_are_encoded_lazily() -> None:
    ability = StyleAbility(
        LiveEditStyleConfig(),
        map_context_config=LiveEditMapContextConfig(enabled=True),
    )
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
    pipeline = SimpleNamespace(
        replace_text_from_embeddings=lambda *args, **kwargs: calls.append(
            (args, kwargs)
        )
    )
    session = SimpleNamespace(
        pipeline=pipeline,
        _cache=object(),
        _pending_finalization_index=None,
        replace_prompt=lambda *args, **kwargs: pytest.fail("expected cached swap"),
    )
    setattr(
        ability,
        "_encode_prompt",
        lambda active_pipeline, prompt: ability._prompt_embeddings.__setitem__(
            prompt, "cached"
        ),
    )
    target = SimpleNamespace(
        prompt="A sunny suburb. The taxi is stationary.",
        guidance_scale=1.0,
        guidance_chunks=0,
        use_lora=False,
    )

    ability._replace_text(session, target)

    assert target.prompt in ability._prompt_embeddings
    assert calls[0][0][1] == "cached"


def test_map_only_postprocessing_returns_original_video() -> None:
    scene = _scene(
        game_map=load_game_map(
            Path(__file__).parent / "maps" / "intersection_geometry.robotaxi.yaml"
        )
    )
    gameplay = LiveEditGameplay(
        LiveEditConfig(map_context=LiveEditMapContextConfig(enabled=True)),
        scene,
        (),
        vehicle=VehicleConfig(),
    )
    video = object()

    assert gameplay.postprocess_video(cast(Any, video), None) is video


def test_map_context_cli_enables_live_edit_runtime() -> None:
    parser = argparse.ArgumentParser()
    add_live_edit_args(parser)

    config = live_edit_config_from_args(parser.parse_args(["--live-edit-map-context"]))

    assert config.map_context.enabled
    assert config.any_enabled
