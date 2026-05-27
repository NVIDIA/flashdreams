# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Tests for the ground-snap physics step.

Covers ``interactive_drive.ply_io``, ``interactive_drive.physics.GroundSnapper``, and the
integration into ``interactive_drive.control.sample_chunk_trajectory``.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from interactive_drive.config import ChunkConfig, VehicleConfig
from interactive_drive.control import sample_chunk_trajectory
from interactive_drive.physics import GroundSnapper
from interactive_drive.ply_io import load_mesh_vf, save_mesh_vf
from interactive_drive.scene_fixture import build_synthetic_scene_usdz
from interactive_drive.scene_loader import load_scene_bundle
from interactive_drive.types import DriverCommand, VehicleState

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _flat_ground(z: float = 0.0, half_extent: float = 50.0) -> tuple[np.ndarray, np.ndarray]:
    """Two-triangle quad ground plane, square in xy at the given z."""
    vertices = np.array(
        [
            [-half_extent, -half_extent, z],
            [half_extent, -half_extent, z],
            [half_extent, half_extent, z],
            [-half_extent, half_extent, z],
        ],
        dtype=np.float32,
    )
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    return vertices, faces


def _sloped_ground(pitch_deg: float, half_extent: float = 50.0) -> tuple[np.ndarray, np.ndarray]:
    """Ground tilted around the y axis: z = x * tan(pitch_deg).

    Positive ``pitch_deg`` means "uphill in +x direction", so the body's
    pitch after snap should be the *negative* of that (math3d's pitch
    convention is nose-down-positive; uphill = nose-up = negative pitch).
    """
    slope = math.tan(math.radians(pitch_deg))
    vertices = np.array(
        [
            [-half_extent, -half_extent, -half_extent * slope],
            [half_extent, -half_extent, half_extent * slope],
            [half_extent, half_extent, half_extent * slope],
            [-half_extent, half_extent, -half_extent * slope],
        ],
        dtype=np.float32,
    )
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    return vertices, faces


def _state(
    *,
    x: float = 0.0,
    y: float = 0.0,
    z: float = 0.0,
    yaw: float = 0.0,
    pitch: float = 0.0,
    roll: float = 0.0,
) -> VehicleState:
    return VehicleState(
        x_m=x,
        y_m=y,
        z_m=z,
        yaw_rad=yaw,
        speed_mps=0.0,
        steer_rad=0.0,
        pitch_rad=pitch,
        roll_rad=roll,
    )


# ---------------------------------------------------------------------------
# ply_io
# ---------------------------------------------------------------------------


def test_ply_io_round_trip_binary() -> None:
    vertices = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.5], [0.0, 1.0, -0.25], [1.0, 1.0, 0.75]],
        dtype=np.float32,
    )
    faces = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.int32)

    blob = save_mesh_vf(vertices, faces)
    decoded_v, decoded_f = load_mesh_vf(blob)

    assert decoded_v.shape == vertices.shape
    assert decoded_f.shape == faces.shape
    np.testing.assert_allclose(decoded_v, vertices, atol=1e-6)
    np.testing.assert_array_equal(decoded_f, faces)


def test_ply_io_rejects_non_ply_data() -> None:
    with pytest.raises(ValueError):
        load_mesh_vf(b"not a ply file at all")


# ---------------------------------------------------------------------------
# GroundSnapper construction
# ---------------------------------------------------------------------------


def test_ground_snapper_rejects_bad_shapes() -> None:
    with pytest.raises(ValueError):
        GroundSnapper(np.zeros((3, 4), dtype=np.float32), np.zeros((1, 3), dtype=np.int32))
    with pytest.raises(ValueError):
        GroundSnapper(np.zeros((3, 3), dtype=np.float32), np.zeros((1, 4), dtype=np.int32))


def test_ground_snapper_rejects_out_of_range_face_indices() -> None:
    vertices = np.zeros((3, 3), dtype=np.float32)
    faces = np.array([[0, 1, 7]], dtype=np.int32)
    with pytest.raises(ValueError):
        GroundSnapper(vertices, faces)


# ---------------------------------------------------------------------------
# Snap behavior
# ---------------------------------------------------------------------------


def test_snap_flat_ground_calibrates_anchor_and_keeps_z() -> None:
    snapper = GroundSnapper(*_flat_ground(z=0.0))
    initial = _state(z=5.0)

    snapped = snapper.snap(initial, VehicleConfig())

    assert snapper.anchor_offset_m == pytest.approx(5.0, abs=1e-3)
    assert snapped.z_m == pytest.approx(5.0, abs=1e-3)
    assert snapped.pitch_rad == pytest.approx(0.0, abs=1e-4)
    assert snapped.roll_rad == pytest.approx(0.0, abs=1e-4)
    # x, y, yaw, speed, steer pass through unchanged.
    assert snapped.x_m == initial.x_m
    assert snapped.y_m == initial.y_m
    assert snapped.yaw_rad == initial.yaw_rad
    assert snapped.steer_rad == initial.steer_rad


def test_snap_flat_ground_holds_height_when_x_changes() -> None:
    snapper = GroundSnapper(*_flat_ground(z=0.0))
    vehicle = VehicleConfig()

    snapper.snap(_state(z=2.5), vehicle)  # calibrate at origin.
    moved = snapper.snap(_state(x=10.0, y=3.0, z=2.5), vehicle)

    assert moved.z_m == pytest.approx(2.5, abs=1e-3)
    assert moved.pitch_rad == pytest.approx(0.0, abs=1e-4)
    assert moved.roll_rad == pytest.approx(0.0, abs=1e-4)


def test_snap_sloped_ground_updates_pitch_and_z() -> None:
    pitch_deg = 5.0
    snapper = GroundSnapper(*_sloped_ground(pitch_deg=pitch_deg))
    vehicle = VehicleConfig()

    initial = _state(x=0.0, z=2.0)  # ground_z(0,0)=0, so anchor offset ~ 2.0
    snapper.snap(initial, vehicle)

    advanced = _state(x=10.0, z=2.0)  # state.z_m hasn't tracked the slope yet
    snapped = snapper.snap(advanced, vehicle)

    expected_z = 10.0 * math.tan(math.radians(pitch_deg)) + 2.0
    assert snapped.z_m == pytest.approx(expected_z, abs=5e-3)
    # math3d convention: positive pitch_deg in the slope (uphill +x) -> body
    # pitch is the negative (nose-up). See module docstring of physics.py.
    assert snapped.pitch_rad == pytest.approx(-math.radians(pitch_deg), abs=5e-3)
    assert snapped.roll_rad == pytest.approx(0.0, abs=5e-3)


def test_snap_off_mesh_returns_input_unchanged() -> None:
    snapper = GroundSnapper(*_flat_ground(z=0.0, half_extent=2.0))
    state = _state(x=500.0, y=500.0, z=10.0)

    out = snapper.snap(state, VehicleConfig())

    assert out == state
    assert snapper.anchor_offset_m is None


def test_snap_translation_threshold_rejects_jump() -> None:
    snapper = GroundSnapper(*_flat_ground(z=0.0), max_translation_m=0.05)
    vehicle = VehicleConfig()
    snapper.snap(_state(z=2.0), vehicle)  # calibrate; anchor_offset = 2.0

    # State teleported to z=10 where ground would still pull us back to ~2.0;
    # delta_z >> 0.05 -> reject.
    teleported = _state(z=10.0)
    out = snapper.snap(teleported, vehicle)
    assert out == teleported


def test_snap_rotation_threshold_rejects_steep_slope() -> None:
    snapper = GroundSnapper(*_sloped_ground(pitch_deg=20.0), max_rotation_deg=5.0)
    vehicle = VehicleConfig()

    # First call calibrates with delta_rot = 20deg vs starting pitch=0, which
    # exceeds the 5deg cap, so the snapper bails out and leaves us unchanged.
    initial = _state(z=0.0)
    out = snapper.snap(initial, vehicle)
    assert out == initial


# ---------------------------------------------------------------------------
# Wiring into sample_chunk_trajectory
# ---------------------------------------------------------------------------


def test_sample_chunk_trajectory_without_snapper_is_unchanged() -> None:
    """Regression: passing ``ground_snapper=None`` (the default) preserves the
    pre-physics behaviour of constant z, pitch, roll over the chunk."""
    state = _state(z=4.2)
    chunk = sample_chunk_trajectory(
        start_state=state,
        start_timestamp_us=0,
        command=DriverCommand(throttle=1.0),
        chunk_size=10,
        chunk_config=ChunkConfig(fps=30),
        vehicle_config=VehicleConfig(),
        ground_snapper=None,
    )

    # Every pose's z translation in the rendered transform stays at the input.
    np.testing.assert_allclose(chunk.rig_poses_world[:, 2, 3], 4.2, atol=1e-5)
    assert chunk.boundary_state_after_chunk.z_m == pytest.approx(4.2, abs=1e-5)
    assert chunk.boundary_state_after_chunk.pitch_rad == 0.0
    assert chunk.boundary_state_after_chunk.roll_rad == 0.0


def test_sample_chunk_trajectory_with_snapper_follows_slope() -> None:
    snapper = GroundSnapper(*_sloped_ground(pitch_deg=3.0))
    state = _state(x=0.0, z=1.5)
    chunk_cfg = ChunkConfig(fps=30)
    vehicle = VehicleConfig(max_speed_mps=20.0, max_accel_mps2=10.0)

    chunk = sample_chunk_trajectory(
        start_state=state,
        start_timestamp_us=0,
        command=DriverCommand(throttle=1.0),
        chunk_size=30,  # ~1s of driving
        chunk_config=chunk_cfg,
        vehicle_config=vehicle,
        ground_snapper=snapper,
    )

    final = chunk.boundary_state_after_chunk
    assert final.x_m > 0.0  # we actually moved
    expected_anchor = 1.5  # ground_z(0,0)=0, initial z=1.5 -> offset 1.5
    expected_z = math.tan(math.radians(3.0)) * final.x_m + expected_anchor
    assert final.z_m == pytest.approx(expected_z, abs=2e-2)
    assert final.pitch_rad == pytest.approx(-math.radians(3.0), abs=5e-3)


# ---------------------------------------------------------------------------
# scene_loader integration via the synthetic USDZ fixture
# ---------------------------------------------------------------------------


def test_synthetic_fixture_ships_ground_mesh(tmp_path: Path) -> None:
    """The synthetic fixture must include ``mesh_ground.ply`` so legacy
    fixture-based tests still exercise the new ground-snap code path."""
    from interactive_drive.config import RasterConfig

    fixture_path = build_synthetic_scene_usdz(tmp_path / "synthetic.usdz")
    bundle = load_scene_bundle(
        scene_path=fixture_path,
        camera_name="camera_front_wide_120fov",
        variant="default",
        prompt_override=None,
        raster=RasterConfig(width=320, height=176),
    )

    assert bundle.ground_mesh_vertices is not None
    assert bundle.ground_mesh_faces is not None
    assert bundle.ground_mesh_vertices.shape[1] == 3
    assert bundle.ground_mesh_faces.shape[1] == 3
    assert bundle.ground_mesh_faces.shape[0] >= 2

    # The ground mesh must be usable as input to GroundSnapper.
    snapper = GroundSnapper(bundle.ground_mesh_vertices, bundle.ground_mesh_faces)
    snapped = snapper.snap(_state(x=0.0, y=0.0, z=1.5), VehicleConfig())
    # Synthetic ground is flat at z=0; anchor offset = 1.5, so z stays 1.5.
    assert snapped.z_m == pytest.approx(1.5, abs=1e-3)
