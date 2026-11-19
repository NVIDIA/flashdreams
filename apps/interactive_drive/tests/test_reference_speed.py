# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check stable reference speed, distance preservation, and native replay."""

import csv
import json
from dataclasses import asdict, replace

import numpy as np
import pytest
from interactive_drive.comparison import load_source, stabilize_reference_speed
from interactive_drive.config import VehicleConfig
from interactive_drive.simulation.ego_vehicle_kinematics import integrate_vehicle
from interactive_drive.types import DriverCommand, VehicleState

pytestmark = pytest.mark.ci_cpu
pytest.importorskip("scipy")


def test_fit_reduces_jitter_without_losing_distance_on_irregular_samples():
    times = np.r_[0, np.cumsum(np.random.default_rng(4).uniform(0.07, 0.15, 120))]
    positions = np.column_stack(
        (6 * times + 6 * (1 - np.cos(times * 0.2)), np.zeros_like(times))
    )
    actual = 6 + 1.2 * np.sin(times * 0.2)
    noisy = actual + 0.3 * np.sin(times * 9) + 0.15 * np.cos(times * 15)
    fitted = stabilize_reference_speed(times, noisy, positions, VehicleConfig())
    dt = np.diff(times)
    before = np.diff(np.diff(noisy) / dt) / ((dt[:-1] + dt[1:]) / 2)
    after = np.diff(np.diff(fitted) / dt) / ((dt[:-1] + dt[1:]) / 2)
    assert np.sqrt(np.mean(after**2)) < np.sqrt(np.mean(before**2)) * 0.1
    assert np.sqrt(np.mean((fitted - actual) ** 2)) < 0.08
    travelled = np.sum((fitted[1:] + fitted[:-1]) * dt / 2)
    assert abs(travelled - positions[-1, 0]) < 1e-7
    with pytest.raises(ValueError, match="increasing timestamps"):
        stabilize_reference_speed(
            np.zeros_like(times), noisy, positions, VehicleConfig()
        )


def test_fitted_acceleration_and_braking_replay_in_native_dynamics():
    times = np.arange(61) / 10
    positions = np.column_stack((5 * times + 1 - np.cos(times), np.zeros_like(times)))
    observed = 5 + np.sin(times)
    observed[20] += 4
    vehicle = replace(VehicleConfig(), max_accel_mps2=0.5, max_brake_mps2=0.8)
    fitted = stabilize_reference_speed(times, observed, positions, vehicle)
    acceleration = np.diff(fitted) / np.diff(times)
    assert acceleration.min() >= -vehicle.max_brake_mps2 - 1e-7
    assert acceleration.max() <= vehicle.max_accel_mps2 + 1e-7
    state = VehicleState(0, 0, 0, 0, float(fitted[0]), 0)
    for i, (a, dt) in enumerate(zip(acceleration, np.diff(times), strict=True)):
        command = DriverCommand(
            throttle=max(1e-10, a / vehicle.max_accel_mps2) if a >= 0 else 0,
            brake=max(0, -a / vehicle.max_brake_mps2),
        )
        assert 0 <= command.throttle <= 1 + 1e-7 and 0 <= command.brake <= 1 + 1e-7
        state = integrate_vehicle(state, command, float(dt), vehicle)
        assert abs(state.speed_mps - fitted[i + 1]) < 1e-7


def test_raw_reader_uses_one_stable_speed_acceleration_and_pedal_timeline(tmp_path):
    raw, scene = tmp_path / "raw", tmp_path / "scene"
    raw.mkdir()
    scene.mkdir()

    def write_json(path, value):
        path.write_text(json.dumps(value))

    def write_csv(name, rows):
        with (raw / name).open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    write_json(raw / "manifest.json", {})
    write_json(
        raw / "sample.json",
        {
            "format": "flashdreams.jlu-raw-multiview",
            "multiview": {
                "views": [{"camera_key": "CAM_FRONT", "timestamps_path": "images.csv"}]
            },
            "estimated_controls_path": "controls.csv",
            "estimated_vehicle_state_path": "states.csv",
            "calibration_path": "calibration.json",
        },
    )
    write_json(raw / "dynamics_model.json", {"vehicle_config": asdict(VehicleConfig())})
    write_json(
        raw / "calibration.json",
        {"cameras": {"CAM_FRONT": {"sensor_to_ego_matrix": np.eye(4).tolist()}}},
    )
    write_csv(
        "images.csv",
        [
            {"index_unix_timestamp_ns": i * 100_000_000, "image_path": f"{i}.jpg"}
            for i in range(21)
        ],
    )
    write_csv("controls.csv", [{"front_wheel_angle_est_rad": 0} for _ in range(21)])
    write_csv(
        "states.csv",
        [
            {
                "timestamp_ns": i * 100_000_000,
                "speed_est_mps": 5,
                "x_fit_m": i * 0.5,
                "y_fit_m": 0,
            }
            for i in range(22)
        ],
    )
    write_csv(
        "ego_poses.csv",
        [
            {"timestamp_ns": i * 100_000_000, "metadata_speed_mps": 3 if i % 2 else 9}
            for i in range(21)
        ],
    )
    write_json(scene / "scene.json", {"start_us": 0, "fps": 10})
    write_json(
        scene / "poses.json", {"camera_front_wide_120fov": [np.eye(4).tolist()] * 21}
    )
    source = load_source(raw, scene)
    frames = source["frames"]
    assert np.std([f["speed_raw"] for f in frames]) > 2
    np.testing.assert_allclose([f["speed"] for f in frames], 5, atol=1e-5)
    np.testing.assert_allclose([f["acceleration"] for f in frames], 0, atol=1e-5)
    assert max(f["throttle"] for f in frames) < 1e-5
    assert max(f["brake"] for f in frames) < 1e-5
