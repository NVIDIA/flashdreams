# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU CI regressions for native barrier queries on synthetic geometry."""

import math
import os
from dataclasses import replace

import numpy as np
import pytest
from ludus_renderer import (
    BodyState,
    InvisibleBarrier,
    PhysicsObjectGraph,
    PhysXWorld,
    RigidBodyModel,
    prepare_physx,
)
from ludus_renderer.physx import _EGO_BODY_ID, _state_arrays
from omnidreams_game_engine.config import VehicleConfig
from omnidreams_game_engine.simulation.game_physics import _ego_model

pytestmark = pytest.mark.ci_cpu


@pytest.fixture(scope="module", autouse=True)
def native_physics() -> None:
    if os.environ.get("FLASHDREAMS_TEST_NATIVE_PHYSX") != "1":
        pytest.skip("Native SDK preparation is enabled separately in CPU CI")
    prepare_physx()


@pytest.mark.parametrize("height", [-4.0, 0.0, 8.0])
@pytest.mark.parametrize("nearby_count", [0, 80])
def test_barrier_query_keeps_contacts_after_topology_changes(
    height: float, nearby_count: int
) -> None:
    model = _ego_model(VehicleConfig())
    yaw = 0.9
    support = (
        math.sin(yaw) * model.half_extents_m[0]
        + math.cos(yaw) * model.half_extents_m[1]
    )
    radius = math.hypot(*model.half_extents_m[:2])
    contact = InvisibleBarrier(
        (-10, support + 0.03),
        (10, support + 0.03),
        barrier_id="contact",
        start_z_m=height,
        end_z_m=height,
    )
    # More than a fixed 32/64-hit buffer can hold, but outside contact distance.
    barriers = tuple(
        InvisibleBarrier(
            (-10, radius + 0.25 + i * 0.0001),
            (10, radius + 0.25 + i * 0.0001),
            barrier_id=f"near-{i}",
            start_z_m=height,
            end_z_m=height,
        )
        for i in range(nearby_count)
    ) + (
        InvisibleBarrier(
            contact.start_xy_m,
            contact.end_xy_m,
            barrier_id="above",
            start_z_m=height + 20,
            end_z_m=height + 20,
        ),
    )
    world = PhysXWorld(PhysicsObjectGraph(barriers=barriers), model)
    try:
        world.set_ground_mesh(
            np.asarray(
                [
                    [-30, -30, height],
                    [30, -30, height],
                    [30, 30, height],
                    [-30, 30, height],
                ],
                dtype=np.float32,
            ),
            np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int32),
        )
        body = BodyState(
            position_m=np.asarray(
                [0, 0, height + model.half_extents_m[2]], dtype=np.float32
            ),
            orientation_xyzw=np.asarray(
                [0, 0, math.sin(yaw / 2), math.cos(yaw / 2)], dtype=np.float32
            ),
            linear_velocity_mps=np.zeros(3, dtype=np.float32),
            angular_velocity_radps=np.zeros(3, dtype=np.float32),
        )
        for tick, touching in enumerate((False, True, False, True)):
            if touching:
                world.add_barrier("contact", contact)
            else:
                world.remove_barrier("contact")
            world._scene.update_body(_EGO_BODY_ID, *_state_arrays(body), False)
            resolved = world.step_compact(body, tick * 33333, 1 / 30).ego
            actual_yaw = 2 * math.atan2(
                resolved.orientation_xyzw[2], resolved.orientation_xyzw[3]
            )
            assert actual_yaw == pytest.approx(
                math.radians(25) if touching else yaw, abs=1e-5
            )
    finally:
        world.close()


def test_barrier_query_preserves_thick_wall_endpoint_rebound() -> None:
    model = RigidBodyModel(mass_kg=1500, half_extents_m=(1.4, 0.7, 0.5))
    wall = InvisibleBarrier((0, 2), (0, 4), thickness_m=3, barrier_id="endpoint")
    world = PhysXWorld(PhysicsObjectGraph(barriers=(wall,)), model)
    body = BodyState(
        position_m=np.asarray([0, 0, 0.5], dtype=np.float32),
        orientation_xyzw=np.asarray([0, 0, 0, 1], dtype=np.float32),
        linear_velocity_mps=np.asarray([0, 5, 0], dtype=np.float32),
        angular_velocity_radps=np.zeros(3, dtype=np.float32),
    )
    try:
        rebound = world.step_compact(body, 0, 1 / 30).ego
        assert rebound.linear_velocity_mps[1] <= -1.0
        world.remove_barrier("endpoint")
        world._scene.update_body(_EGO_BODY_ID, *_state_arrays(body), False)
        clear = world.step_compact(body, 33333, 1 / 30).ego
        assert clear.linear_velocity_mps[1] > 0
    finally:
        world.close()


@pytest.mark.parametrize("vehicle", [False, True])
def test_competing_barriers_preserve_contact_order(vehicle: bool) -> None:
    model = (
        _ego_model(VehicleConfig())
        if vehicle
        else RigidBodyModel(mass_kg=1500, half_extents_m=(2, 2, 0.5))
    )
    model = replace(model, half_extents_m=(2, 2, model.half_extents_m[2]))
    # Equal clearance at opposite 45-degree walls makes heading ties observable.
    walls = {
        "first": InvisibleBarrier(
            (0, 4.2), (4.2, 0), thickness_m=0.2, barrier_id="first"
        ),
        "second": InvisibleBarrier(
            (-4.2, 0), (0, 4.2), thickness_m=0.2, barrier_id="second"
        ),
    }
    world = PhysXWorld(PhysicsObjectGraph(barriers=tuple(walls.values())), model)
    body = BodyState(
        position_m=np.asarray([0, 0, model.half_extents_m[2]], dtype=np.float32),
        orientation_xyzw=np.asarray([0, 0, 0, 1], dtype=np.float32),
        linear_velocity_mps=np.asarray([0, 0 if vehicle else 5, 0], dtype=np.float32),
        angular_velocity_radps=np.zeros(3, dtype=np.float32),
    )
    try:
        # These signs match the original barrier scan, including reinsertion.
        for tick, (reinsert, sign) in enumerate(
            ((None, 1), ("first", -1), ("second", 1), ("first", -1))
        ):
            if reinsert is not None:
                world.remove_barrier(reinsert)
                world.add_barrier(reinsert, walls[reinsert])
            world._scene.update_body(_EGO_BODY_ID, *_state_arrays(body), False)
            # Keep the tied geometry intact through a single short solver step.
            resolved = world.step_compact(body, tick * 333, 1 / 3000).ego
            if vehicle:
                yaw = 2 * math.atan2(
                    resolved.orientation_xyzw[2], resolved.orientation_xyzw[3]
                )
                assert yaw == pytest.approx(sign * math.radians(20), abs=1e-5)
            else:
                assert resolved.linear_velocity_mps[:2] == pytest.approx(
                    [sign * 3.05, 1.95], abs=1e-5
                )
    finally:
        world.close()
