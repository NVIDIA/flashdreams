# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native CPU solver checks for authored ground and elevated walls.

These manual tests build the pinned PhysX SDK on first use. They run without
CUDA, model weights, or a renderer.
"""

from dataclasses import replace

import numpy as np
import pytest
from ludus_renderer import (
    BodyState,
    InvisibleBarrier,
    PhysicsObjectGraph,
    PhysXWorld,
    RigidBodyModel,
    SceneObject,
    prepare_physx,
)

pytestmark = pytest.mark.manual


@pytest.fixture(scope="module", autouse=True)
def native_physx() -> None:
    prepare_physx()


def _plane(height: float) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.asarray(
            [
                [-20, -20, height],
                [20, -20, height],
                [20, 20, height],
                [-20, 20, height],
            ],
            dtype=np.float32,
        ),
        np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int32),
    )


def _body(height: float) -> BodyState:
    return BodyState(
        position_m=np.asarray([0, 0, height], dtype=np.float32),
        orientation_xyzw=np.asarray([0, 0, 0, 1], dtype=np.float32),
        linear_velocity_mps=np.zeros(3, dtype=np.float32),
        angular_velocity_radps=np.zeros(3, dtype=np.float32),
    )


@pytest.mark.parametrize("height", [8.0, -4.0])
def test_native_body_lands_on_authored_floor(height: float) -> None:
    model = RigidBodyModel(mass_kg=1500, half_extents_m=(1.4, 0.7, 0.5))
    world = PhysXWorld(PhysicsObjectGraph(), model)
    try:
        world.set_ground_mesh(*_plane(height))
        body = _body(max(height + 2, 2))
        for tick in range(240):
            body = world.step_compact(body, tick * 33333, 1 / 30).ego
        assert body.position_m[2] == pytest.approx(height + 0.51, abs=0.03)
        assert abs(body.linear_velocity_mps[2]) < 0.05
    finally:
        world.close()


@pytest.mark.parametrize("actor_height", [0.51, 8.51])
def test_actor_impact_detection_respects_height(actor_height: float) -> None:
    model = RigidBodyModel(mass_kg=1500, half_extents_m=(1.4, 0.7, 0.5))
    actor = SceneObject(
        "other-car",
        "Car",
        model,
        np.asarray([0, 1000000], dtype=np.int64),
        np.asarray([[1.5, 0, actor_height], [1.5, 0, actor_height]], dtype=np.float32),
        np.asarray([[0, 0, 0, 1], [0, 0, 0, 1]], dtype=np.float32),
    )
    world = PhysXWorld(PhysicsObjectGraph(objects=(actor,)), model)
    try:
        world.set_ground_mesh(*_plane(0))
        body = replace(
            _body(0.51), linear_velocity_mps=np.asarray([5, 0, 0], dtype=np.float32)
        )
        result = world.step_compact(body, 0, 1 / 30)
        assert result.impact is (actor_height < 1)
        assert ("other-car" in result.struck_object_ids) is (actor_height < 1)
    finally:
        world.close()


def test_invalid_mesh_preserves_existing_ground() -> None:
    world = PhysXWorld(
        PhysicsObjectGraph(),
        RigidBodyModel(mass_kg=1500, half_extents_m=(1.4, 0.7, 0.5)),
    )
    try:
        vertices, faces = _plane(8)
        world.set_ground_mesh(vertices, faces)
        invalid = vertices.copy()
        invalid[0, 2] = np.nan
        for points, indices in (
            (invalid, faces),
            (vertices, np.asarray([[0, 1, 99]], dtype=np.int32)),
            (vertices, faces.astype(np.float32)),
            (vertices[:, :2], faces),
        ):
            with pytest.raises(ValueError):
                world.set_ground_mesh(points, indices)
        body = _body(10)
        for tick in range(180):
            body = world.step_compact(body, tick * 33333, 1 / 30).ego
        assert body.position_m[2] == pytest.approx(8.51, abs=0.03)
    finally:
        world.close()


@pytest.mark.parametrize("floor", [0.0, 8.0])
def test_elevated_barrier_contacts_only_its_height_range(floor: float) -> None:
    wall = InvisibleBarrier((-10, 0), (10, 0), start_z_m=8, end_z_m=8)
    world = PhysXWorld(
        PhysicsObjectGraph(barriers=(wall,)),
        RigidBodyModel(mass_kg=1500, half_extents_m=(1.4, 0.7, 0.5)),
    )
    try:
        world.set_ground_mesh(*_plane(floor))
        body = replace(
            _body(floor + 0.51),
            position_m=np.asarray([0, 3, floor + 0.51], dtype=np.float32),
        )
        for tick in range(120):
            intent = replace(
                body, linear_velocity_mps=np.asarray([0, -5, 0], dtype=np.float32)
            )
            body = world.step_compact(intent, tick * 33333, 1 / 30).ego
        if floor == 8:
            assert body.position_m[1] > 0.7
        else:
            assert body.position_m[1] < -2
    finally:
        world.close()
