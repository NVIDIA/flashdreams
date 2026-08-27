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

"""CPU tests for gameplay-owned PhysX track progress."""

from __future__ import annotations

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
