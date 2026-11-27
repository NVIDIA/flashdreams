# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for gameplay-owned PhysX track progress."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import patch

import pytest
from ludus_renderer import PhysXWorld
from omnidreams_game_engine.simulation.actor_controller import ActorTrackTarget
from omnidreams_game_engine.simulation.gameplay_physx import GameplayPhysXWorld

pytestmark = pytest.mark.ci_cpu


def test_actor_targets_use_one_batched_track_progress_update() -> None:
    world = object.__new__(GameplayPhysXWorld)
    targets = (
        ActorTrackTarget(
            object_id="traffic-a",
            timestamp_us=250_000,
            velocity_scale=0.75,
        ),
        ActorTrackTarget(
            object_id="obstacle-b",
            timestamp_us=500_000,
            velocity_scale=1.0,
        ),
    )

    with patch.object(PhysXWorld, "apply_track_progress") as apply_track_progress:
        world.apply_actor_track_targets(targets)

    apply_track_progress.assert_called_once_with(
        (
            ("traffic-a", 250_000, 0.75),
            ("obstacle-b", 500_000, 1.0),
        )
    )


def test_backend_prepares_native_module_before_constructing_world(monkeypatch) -> None:
    from omnidreams_game_engine.simulation import gameplay_physx

    calls = []
    monkeypatch.setattr(
        gameplay_physx, "prepare_physx", lambda: calls.append("prepare")
    )

    def initialize(world, *args, **kwargs):
        assert calls == ["prepare"]
        calls.append("world")
        world._scene = SimpleNamespace(set_body_track_progress=lambda: None)

    monkeypatch.setattr(PhysXWorld, "__init__", initialize)
    GameplayPhysXWorld(cast(Any, object()), cast(Any, object()))
    assert calls == ["prepare", "world"]
