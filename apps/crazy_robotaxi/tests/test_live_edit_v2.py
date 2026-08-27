# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""CPU regression tests for API-v2 live-edit composition."""

from __future__ import annotations

import numpy as np
import pytest
from crazy_robotaxi.live_edit.config import LiveEditItemsConfig
from crazy_robotaxi.live_edit.nitro_ability import NitroAbility
from crazy_robotaxi.live_edit.obstacle_templates import load_obstacle_template_catalog
from crazy_robotaxi.live_edit.runtime_v2 import LiveEditGameplay
from flashdreams.runtime_v2.user_input_event import (
    KeyboardInputState,
    KeyboardUserInputEvent,
)
from flashdreams.runtime_v2.user_input_events import UserInputEvents
from omnidreams_game_engine.config import VehicleConfig

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
