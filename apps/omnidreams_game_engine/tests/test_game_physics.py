# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU parity tests for game-specific PhysX bridge helpers."""

from __future__ import annotations

import math
from types import SimpleNamespace
from typing import cast
from unittest.mock import MagicMock

import numpy as np
import pytest
from ludus_renderer import BodyState, InvisibleBarrier, RigidBodyModel
from ludus_renderer.physx import PhysXStepTimings, _CompactPhysicsStep
from omnidreams_game_engine.config import VehicleConfig
from omnidreams_game_engine.simulation import game_physics
from omnidreams_game_engine.simulation.game_physics import (
    _BARRIER_CONTACT_SLOP_M,
    _BarrierReboundIndex,
    _reinforce_static_barrier_rebound,
)
from omnidreams_game_engine.simulation.ground_snap import GroundSnapper
from omnidreams_game_engine.types import SceneDefinition, VehicleState

pytestmark = pytest.mark.ci_cpu


def _body_state(
    *,
    position_xy: tuple[float, float],
    velocity_xy: tuple[float, float],
    yaw_rad: float,
) -> BodyState:
    half_yaw = yaw_rad * 0.5
    return BodyState(
        position_m=np.asarray([*position_xy, 0.5], dtype=np.float32),
        orientation_xyzw=np.asarray(
            [0.0, 0.0, math.sin(half_yaw), math.cos(half_yaw)],
            dtype=np.float32,
        ),
        linear_velocity_mps=np.asarray([*velocity_xy, 0.0], dtype=np.float32),
        angular_velocity_radps=np.zeros(3, dtype=np.float32),
    )


def _index(barriers: tuple[InvisibleBarrier, ...]) -> _BarrierReboundIndex:
    segments = np.asarray(
        [[barrier.start_xy_m, barrier.end_xy_m] for barrier in barriers],
        dtype=np.float32,
    ).reshape((-1, 2, 2))
    thicknesses = np.asarray(
        [barrier.thickness_m for barrier in barriers], dtype=np.float32
    )
    return _BarrierReboundIndex.from_arrays(segments, thicknesses)


def _scalar_rebound(
    barriers: tuple[InvisibleBarrier, ...],
    requested_ego: BodyState,
    resolved_velocity_mps: np.ndarray,
    ego_model: RigidBodyModel,
    restitution: float,
) -> tuple[np.ndarray, bool]:
    position = np.asarray(requested_ego.position_m[:2], dtype=np.float32)
    incoming_velocity = np.asarray(requested_ego.linear_velocity_mps, dtype=np.float32)
    reinforced = np.asarray(resolved_velocity_mps, dtype=np.float32).copy()
    x, y, z, w = [float(value) for value in requested_ego.orientation_xyzw]
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    forward = np.asarray([math.cos(yaw), math.sin(yaw)], dtype=np.float32)
    left = np.asarray([-forward[1], forward[0]], dtype=np.float32)
    contact_detected = False

    for barrier in barriers:
        start = np.asarray(barrier.start_xy_m, dtype=np.float32)
        end = np.asarray(barrier.end_xy_m, dtype=np.float32)
        segment = end - start
        length_squared = float(np.dot(segment, segment))
        if length_squared <= 1.0e-8:
            continue
        alpha = float(
            np.clip(np.dot(position - start, segment) / length_squared, 0.0, 1.0)
        )
        offset = position - (start + segment * alpha)
        distance = float(np.linalg.norm(offset))
        if distance > 1.0e-6:
            normal = offset / distance
        else:
            speed = float(np.linalg.norm(incoming_velocity[:2]))
            normal = (
                -incoming_velocity[:2] / speed
                if speed > 1.0e-6
                else np.asarray([1.0, 0.0], dtype=np.float32)
            )
        support = (
            abs(float(np.dot(normal, forward))) * ego_model.half_extents_m[0]
            + abs(float(np.dot(normal, left))) * ego_model.half_extents_m[1]
        )
        if distance > support + barrier.thickness_m * 0.5 + _BARRIER_CONTACT_SLOP_M:
            continue
        incoming_normal_speed = float(np.dot(incoming_velocity[:2], normal))
        if incoming_normal_speed >= 0.0:
            continue
        contact_detected = True
        target_outward_speed = -restitution * incoming_normal_speed
        resolved_normal_speed = float(np.dot(reinforced[:2], normal))
        if resolved_normal_speed < target_outward_speed:
            reinforced[:2] += normal * (target_outward_speed - resolved_normal_speed)
    return reinforced, contact_detected


def test_debug_observation_preserves_native_velocity_before_curb_rebound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    native = MagicMock()
    native.collider_state_arrays.return_value = (
        (),
        np.empty((0, 3)),
        np.empty((0, 4)),
        np.empty((0, 3)),
    )
    monkeypatch.setattr(game_physics, "GameplayPhysXWorld", lambda *a, **kw: native)
    scene = SimpleNamespace(
        game_map=None,
        line_layers=(),
        ground_mesh_vertices=None,
        ground_mesh_faces=None,
        initial_rig_to_world=np.eye(4),
    )
    world = game_physics.GamePhysicsWorld(
        cast(SceneDefinition, scene),
        VehicleConfig(static_collision_enabled=True),
        static_barrier_segments_world=np.asarray([[[-5, 0, 0], [5, 0, 0]]]),
        static_barrier_restitution=0.5,
    )
    body = _body_state(position_xy=(0, 0.2), velocity_xy=(2, -1), yaw_rad=0)
    body.position_m[2] = world._ego_model.half_extents_m[2]
    native.step_compact.return_value = _CompactPhysicsStep(
        ego=body,
        actor_samples=(),
        track_samples=(),
        struck_object_ids=frozenset(),
        impact=False,
        timings=PhysXStepTimings(0, 0, 0, 0, 0, 0, 0),
    )
    state = VehicleState(0, 0.2, 0, 0, 2, 0, velocity_x_mps=2, velocity_y_mps=-6)
    resolved, _ = world.step(state, 0, 1 / 30)
    assert resolved.velocity_y_mps == pytest.approx(3)
    body.linear_velocity_mps[:] = 99
    debug = world.debug_frame(resolved)
    assert debug.static_barrier_collision
    np.testing.assert_array_equal(debug.ego_linear_velocity_mps, [2, -1, 0])
    assert debug.vehicle_state is not None
    assert debug.vehicle_state.velocity_y_mps == -1


def test_vectorized_barrier_rebound_matches_scalar_reference() -> None:
    rng = np.random.default_rng(20260827)
    random_endpoints = rng.uniform(-6.0, 6.0, size=(96, 2, 2)).astype(np.float32)
    random_endpoints[0, 1] = random_endpoints[0, 0]
    barriers = tuple(
        InvisibleBarrier(
            (float(endpoints[0, 0]), float(endpoints[0, 1])),
            (float(endpoints[1, 0]), float(endpoints[1, 1])),
            thickness_m=float(rng.uniform(0.1, 0.8)),
        )
        for endpoints in random_endpoints
    )
    index = _index(barriers)
    ego_model = RigidBodyModel(mass_kg=1_500.0, half_extents_m=(1.4, 0.7, 0.5))

    for _ in range(128):
        requested = _body_state(
            position_xy=tuple(rng.uniform(-4.0, 4.0, size=2)),
            velocity_xy=tuple(rng.uniform(-15.0, 15.0, size=2)),
            yaw_rad=float(rng.uniform(-math.pi, math.pi)),
        )
        resolved = rng.uniform(-15.0, 15.0, size=3).astype(np.float32)
        restitution = float(rng.uniform(0.0, 1.0))

        expected_velocity, expected_contact = _scalar_rebound(
            barriers,
            requested,
            resolved,
            ego_model,
            restitution,
        )
        actual_velocity, actual_contact = _reinforce_static_barrier_rebound(
            index,
            requested,
            resolved,
            ego_model,
            restitution,
        )

        assert actual_contact is expected_contact
        np.testing.assert_allclose(
            actual_velocity,
            expected_velocity,
            rtol=2.0e-6,
            atol=2.0e-6,
        )


def test_vectorized_barrier_rebound_reinforces_single_contact() -> None:
    barriers = (InvisibleBarrier((-5.0, 0.0), (5.0, 0.0), thickness_m=0.3),)
    requested = _body_state(
        position_xy=(0.0, 0.2),
        velocity_xy=(2.0, -6.0),
        yaw_rad=0.0,
    )
    resolved = np.asarray([2.0, -1.0, 0.0], dtype=np.float32)
    ego_model = RigidBodyModel(mass_kg=1_500.0, half_extents_m=(1.4, 0.7, 0.5))

    actual, contacted = _reinforce_static_barrier_rebound(
        _index(barriers), requested, resolved, ego_model, restitution=0.5
    )

    assert contacted
    np.testing.assert_allclose(actual, np.asarray([2.0, 3.0, 0.0], np.float32))


def test_empty_barrier_index_preserves_resolved_velocity() -> None:
    requested = _body_state(
        position_xy=(0.0, 0.0),
        velocity_xy=(1.0, 2.0),
        yaw_rad=0.0,
    )
    resolved = np.asarray([3.0, 4.0, 5.0], dtype=np.float32)
    ego_model = RigidBodyModel(mass_kg=1_500.0, half_extents_m=(1.4, 0.7, 0.5))

    actual, contacted = _reinforce_static_barrier_rebound(
        _index(()), requested, resolved, ego_model, restitution=0.5
    )

    assert not contacted
    np.testing.assert_array_equal(actual, resolved)
    assert actual is not resolved


def test_ground_index_populates_only_queried_cells_and_preserves_face_order() -> None:
    vertices = np.asarray(
        [
            [-400, -400, 0],
            [400, -400, 0],
            [400, 400, 0],
            [-400, 400, 0],
            [0, 0, 8],
            [2, 0, 8],
            [0, 2, 8],
            [0, 0, 10],
            [2, 0, 10],
            [0, 2, 10],
            [4, 4, 0],
            [4, 4, 5],
            [4, 6, 0],
        ],
        dtype=np.float32,
    )
    faces = np.asarray([[0, 1, 2], [0, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
    sampler = GroundSnapper(vertices, faces)
    assert not sampler._cell_candidates
    points = np.asarray(
        [
            [-400, -400, 0],
            [400, 400, 0],
            [1, 1, 9],
            [2, 0, 8],
            [4, 4, 0],
            [401, 0, 0],
            [-401, 0, 0],
        ],
        dtype=np.float64,
    )
    heights = sampler._raycast(points)
    np.testing.assert_array_equal(heights, [0, 0, 8, 8, 0, np.nan, np.nan])
    assert len(sampler._cell_candidates) == 5
    for (i, j), candidates in sampler._cell_candidates.items():
        expected = [
            face
            for face in range(len(faces))
            if sampler._cell_x(sampler._tri_xy_min[face, 0])
            <= i
            <= sampler._cell_x(sampler._tri_xy_max[face, 0])
            and sampler._cell_y(sampler._tri_xy_min[face, 1])
            <= j
            <= sampler._cell_y(sampler._tri_xy_max[face, 1])
        ]
        np.testing.assert_array_equal(candidates, expected)
