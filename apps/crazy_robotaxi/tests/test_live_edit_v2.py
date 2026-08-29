# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""CPU regression tests for API-v2 live-edit composition."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import cast

import crazy_robotaxi.live_edit.config as live_edit_config
import numpy as np
import pytest
from crazy_robotaxi.live_edit.config import (
    LiveEditCoinsConfig,
    LiveEditConfig,
    LiveEditItemsConfig,
    LiveEditObstacleConfig,
    LiveEditStyleConfig,
    LiveEditWeatherConfig,
    resolve_live_edit_assets,
)
from crazy_robotaxi.live_edit.nitro_ability import NitroAbility
from crazy_robotaxi.live_edit.obstacle_events import (
    ObstacleAbility,
    ObstacleEvent,
    ObstaclePhase,
)
from crazy_robotaxi.live_edit.obstacle_templates import load_obstacle_template_catalog
from crazy_robotaxi.live_edit.runtime_v2 import LiveEditGameplay
from crazy_robotaxi.navigation import NavigationLane
from ludus_renderer import SceneObject
from omnidreams_game_engine.config import VehicleConfig
from omnidreams_game_engine.types import (
    CameraCalibration,
    SceneDefinition,
    TrajectoryChunk,
    VehicleState,
)
from PIL import Image

from flashdreams.runtime_v2.user_input_event import (
    KeyboardInputState,
    KeyboardUserInputEvent,
)
from flashdreams.runtime_v2.user_input_events import UserInputEvents

pytestmark = pytest.mark.ci_cpu


class _StyleRequests:
    def __init__(self) -> None:
        self.skin_cycles = 0
        self.weather_cycles = 0

    def request_cycle(self) -> None:
        self.skin_cycles += 1

    def request_weather_cycle(self) -> None:
        self.weather_cycles += 1


class _Coins:
    def __init__(self) -> None:
        self.toggles = 0

    def toggle(self) -> bool:
        self.toggles += 1
        return True


class _Obstacles:
    def __init__(self) -> None:
        self.spawns = 0

    def request_spawn(self) -> None:
        self.spawns += 1


def _scene() -> SceneDefinition:
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
            game_map=None,
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


def test_v2_ability_keys_are_consumed_on_pressed_edges() -> None:
    gameplay = LiveEditGameplay.__new__(LiveEditGameplay)
    gameplay.style = _StyleRequests()
    gameplay.coins = _Coins()
    gameplay.obstacles = _Obstacles()
    events = UserInputEvents(
        [
            KeyboardUserInputEvent(
                timestamp=np.uint64(index),
                key=key,
                state=state,
            )
            for index, (key, state) in enumerate(
                (
                    ("k", KeyboardInputState.PRESSED),
                    ("k", KeyboardInputState.RELEASED),
                    ("v", KeyboardInputState.PRESSED),
                    ("c", KeyboardInputState.PRESSED),
                    ("o", KeyboardInputState.PRESSED),
                )
            )
        ]
    )

    gameplay.process_events(events)

    assert gameplay.style.skin_cycles == 1
    assert gameplay.style.weather_cycles == 1
    assert gameplay.coins.toggles == 1
    assert gameplay.obstacles.spawns == 1


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
    assert boosted.max_accel_mps2 == 6.0
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
