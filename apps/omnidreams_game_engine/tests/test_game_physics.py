# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU parity tests for game-specific PhysX bridge helpers."""

from __future__ import annotations

import math

import numpy as np
import pytest
from ludus_renderer import BodyState, InvisibleBarrier, RigidBodyModel
from omnidreams_game_engine.simulation.game_physics import (
    _BARRIER_CONTACT_SLOP_M,
    _BarrierReboundIndex,
    _reinforce_static_barrier_rebound,
)

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


def test_vectorized_barrier_rebound_preserves_corner_response_order() -> None:
    barriers = (
        InvisibleBarrier((-5.0, 0.0), (5.0, 0.0), thickness_m=0.3),
        InvisibleBarrier((0.0, -5.0), (0.0, 5.0), thickness_m=0.3),
    )
    requested = _body_state(
        position_xy=(0.2, 0.2),
        velocity_xy=(-8.0, -6.0),
        yaw_rad=0.35,
    )
    resolved = np.asarray([-1.0, -2.0, 0.0], dtype=np.float32)
    ego_model = RigidBodyModel(mass_kg=1_500.0, half_extents_m=(1.4, 0.7, 0.5))

    expected = _scalar_rebound(
        barriers, requested, resolved, ego_model, restitution=0.6
    )
    actual = _reinforce_static_barrier_rebound(
        _index(barriers), requested, resolved, ego_model, restitution=0.6
    )

    assert actual[1] is expected[1]
    np.testing.assert_array_equal(actual[0], expected[0])


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
