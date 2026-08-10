# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import threading

import numpy as np
from omnidreams.interactive_drive.camera import FThetaCameraModel
from omnidreams.interactive_drive.config import BevConfig
from omnidreams.interactive_drive.input.keyboard import KeyboardState
from omnidreams.interactive_drive.streaming_presenter import (
    _INDEX_HTML,
    MJPEGStreamingPresenter,
    _as_rgb_host_uint8,
    _publish_if_open,
    _wait_for_bus_frame,
)
from omnidreams.interactive_drive.taxi_game import TaxiGameSnapshot
from omnidreams.interactive_drive.types import (
    CameraCalibration,
    PresentedFrame,
    VehicleState,
)

from flashdreams.serving.realtime.frame_bus import LatestFrameBus


def test_streaming_page_contains_taxi_name_and_leaderboard_controls() -> None:
    assert 'id="name-entry"' in _INDEX_HTML
    assert 'id="player-name"' in _INDEX_HTML
    assert "'/taxi/name'" in _INDEX_HTML
    assert 'id="score-rows"' in _INDEX_HTML
    assert 'id="new-game"' in _INDEX_HTML


def test_streaming_presenter_materializes_lazy_rgba_frames() -> None:
    class LazyFrame:
        def to_numpy(self) -> np.ndarray:
            return np.array(
                [[[1, 2, 3, 255], [4, 5, 6, 255]]],
                dtype=np.uint8,
            )

    frame = _as_rgb_host_uint8(LazyFrame())

    assert frame.flags.c_contiguous
    np.testing.assert_array_equal(
        frame,
        np.array([[[1, 2, 3], [4, 5, 6]]], dtype=np.uint8),
    )


def test_browser_taxi_arrow_has_a_visible_shaft() -> None:
    assert "L42 68 L22 68 L22 36" in _INDEX_HTML
    assert "&#9650;" not in _INDEX_HTML


def test_streaming_presenter_draws_visible_world_marker() -> None:
    calibration = CameraCalibration(
        clipgt_name="camera:test",
        logical_name="camera_test",
        width=100,
        height=80,
        cx=50.0,
        cy=40.0,
        polynomial=np.array([0.0, 0.01], dtype=np.float32),
        is_backward_polynomial=True,
        linear_cde=np.array([1.0, 0.0, 0.0], dtype=np.float32),
        sensor_to_rig_flu=np.eye(4, dtype=np.float32),
    )
    taxi = TaxiGameSnapshot(
        phase="seeking_pickup",
        target_xyz_m=(10.0, 0.0, 0.0),
        distance_m=10.0,
        relative_bearing_rad=0.0,
        target_radius_m=2.0,
        remaining_time_s=None,
        score=0,
    )
    frame = PresentedFrame(
        timestamp_us=0,
        rgb_host_uint8=np.zeros((80, 100, 3), dtype=np.uint8),
        depth_host_f32=None,
        rig_to_world=np.eye(4, dtype=np.float32),
        taxi_game_snapshot=taxi,
    )
    presenter = MJPEGStreamingPresenter.__new__(MJPEGStreamingPresenter)
    presenter._taxi_camera_calibration = calibration
    presenter._taxi_camera_models = {(100, 80): FThetaCameraModel(calibration)}

    marked = presenter._with_taxi_world_marker(frame.rgb_host_uint8, frame)

    assert np.count_nonzero(marked) > 0
    assert np.any(np.all(marked == np.array([118, 185, 0]), axis=2))


def test_streaming_presenter_publishes_jpeg_on_latest_frame_bus() -> None:
    bus = LatestFrameBus[bytes]()

    _publish_if_open(bus, b"jpeg", stop_event=threading.Event())

    latest = bus.latest()
    assert latest is not None
    assert latest.payload == b"jpeg"
    assert latest.count == 1


def test_streaming_presenter_frame_wait_returns_none_after_bus_close() -> None:
    bus = LatestFrameBus[bytes]()
    bus.publish(b"old")
    bus.close()

    frame = _wait_for_bus_frame(
        bus,
        last_seen_count=1,
        stop_event=threading.Event(),
    )

    assert frame is None


def test_streaming_state_snapshot_includes_taxi_payload() -> None:
    keyboard = KeyboardState()
    vehicle = VehicleState(0.0, 0.0, 0.0, 0.0, 3.0, 0.0)
    taxi = TaxiGameSnapshot(
        phase="to_dropoff",
        target_xyz_m=(10.0, 0.0, 0.0),
        distance_m=10.0,
        relative_bearing_rad=0.0,
        target_radius_m=6.0,
        remaining_time_s=12.0,
        score=100,
        high_score=500,
    )
    keyboard.update_runtime_state(vehicle, taxi)
    presenter = MJPEGStreamingPresenter.__new__(MJPEGStreamingPresenter)
    presenter._keyboard = keyboard
    presenter._bev_config = BevConfig(tilt_deg=0.0)

    snapshot = presenter._state_snapshot()

    assert snapshot["speed_mps"] == 3.0
    assert isinstance(snapshot["taxi"], dict)
    assert snapshot["taxi"]["phase"] == "to_dropoff"
    assert snapshot["taxi"]["session_state"] == "playing"
    assert snapshot["taxi"]["high_score"] == 500
    assert snapshot["taxi"]["global_remaining_time_s"] == 0.0
    assert snapshot["taxi"]["bev_target"]["visible"] is True
