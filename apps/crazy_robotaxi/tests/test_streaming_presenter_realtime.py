# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import threading
from dataclasses import replace

import numpy as np
import pytest
from crazy_robotaxi.game import TaxiGameSnapshot
from crazy_robotaxi.input import (
    CrazyRobotaxiKeyboardState,
)
from crazy_robotaxi.race import RaceGameSnapshot
from crazy_robotaxi.streaming_presenter import (
    _INDEX_HTML,
    MJPEGStreamingPresenter,
    _as_rgb_host_uint8,
    _downscale_rgb,
    _publish_if_open,
    _scaled_dims,
    _StreamClientStats,
    _validate_extra_key_handlers,
    _wait_for_bus_frame,
)
from omnidreams_game_engine.camera import FThetaCameraModel
from omnidreams_game_engine.config import BevConfig, RasterConfig
from omnidreams_game_engine.input.keyboard import KeyboardState
from omnidreams_game_engine.math3d import rig_pose_from_vehicle_state
from omnidreams_game_engine.streaming_presenter import (
    MJPEGStreamingPresenter as BaseMJPEGStreamingPresenter,
)
from omnidreams_game_engine.types import (
    CameraCalibration,
    PresentedFrame,
    VehicleState,
)

from flashdreams.serving.realtime.frame_bus import LatestFrameBus


def _race_snapshot(*, checkpoint_markers: bool) -> RaceGameSnapshot:
    return RaceGameSnapshot(
        map_id="map",
        course_id="course",
        session_state="racing",
        target_kind="checkpoint",
        target_element_id="checkpoint",
        target_xyz_m=(10.0, 0.0, 0.0),
        gate_start_xyz_m=(10.0, -2.0, 0.0),
        gate_end_xyz_m=(10.0, 2.0, 0.0),
        checkpoint_markers=checkpoint_markers,
        distance_m=10.0,
        relative_bearing_rad=0.0,
        checkpoint_index=0,
        checkpoint_count=2,
        completed_laps=0,
        lap_count=0,
        elapsed_time_us=1_000_000,
        best_time_us=None,
    )


def test_streaming_page_contains_taxi_name_and_leaderboard_controls() -> None:
    assert 'id="name-entry"' in _INDEX_HTML
    assert 'id="player-name"' in _INDEX_HTML
    assert "'/taxi/name'" in _INDEX_HTML
    assert 'id="score-rows"' in _INDEX_HTML
    assert 'id="new-game"' in _INDEX_HTML
    assert "race-gate" in _INDEX_HTML
    assert "bev-edge-arrow" in _INDEX_HTML
    assert "appendBevEdgeArrow(taxi.bev_arrow)" in _INDEX_HTML
    assert "taxi.elapsed_time}" in _INDEX_HTML
    assert "entry.elapsed_time" in _INDEX_HTML
    assert "taxi.elapsed_time_s.toFixed(3)" not in _INDEX_HTML
    assert 'id="taxi-boundaries"' not in _INDEX_HTML
    assert 'id="scene-picker"' not in _INDEX_HTML
    assert "fetchScenes" not in _INDEX_HTML


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
        application_state=taxi,
    )
    presenter = MJPEGStreamingPresenter.__new__(MJPEGStreamingPresenter)
    presenter._taxi_camera_calibration = calibration
    presenter._taxi_camera_models = {(100, 80): FThetaCameraModel(calibration)}

    marked = presenter._with_taxi_world_marker(frame.rgb_host_uint8, frame)

    assert np.count_nonzero(marked) > 0
    assert np.any(np.all(marked == np.array([118, 185, 0]), axis=2))


@pytest.mark.parametrize("checkpoint_markers", [False, True])
def test_streaming_race_gate_respects_camera_marker_setting(
    checkpoint_markers: bool,
) -> None:
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
    frame = PresentedFrame(
        timestamp_us=0,
        rgb_host_uint8=np.zeros((80, 100, 3), dtype=np.uint8),
        depth_host_f32=None,
        rig_to_world=np.eye(4, dtype=np.float32),
        application_state=_race_snapshot(checkpoint_markers=checkpoint_markers),
    )
    presenter = MJPEGStreamingPresenter.__new__(MJPEGStreamingPresenter)
    presenter._taxi_camera_calibration = calibration
    presenter._taxi_camera_models = {(100, 80): FThetaCameraModel(calibration)}

    marked = presenter._with_taxi_world_marker(frame.rgb_host_uint8, frame)

    assert bool(np.count_nonzero(marked)) is checkpoint_markers


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
    keyboard = CrazyRobotaxiKeyboardState()
    vehicle = VehicleState(0.0, 0.0, 0.0, 0.0, 3.0, 0.0)
    future_vehicle = VehicleState(20.0, 5.0, 0.0, 1.0, 30.0, 0.4)
    taxi = TaxiGameSnapshot(
        phase="seeking_pickup",
        target_xyz_m=(10.0, 0.0, 0.0),
        distance_m=10.0,
        relative_bearing_rad=0.0,
        target_radius_m=6.0,
        remaining_time_s=None,
        score=100,
        high_score=500,
        pickup_targets_xyz_m=(
            (10.0, 0.0, 0.0),
            (20.0, 0.0, 0.0),
            (30.0, 0.0, 0.0),
            (40.0, 0.0, 0.0),
        ),
    )
    keyboard.update_runtime_state(future_vehicle, taxi)
    presenter = MJPEGStreamingPresenter.__new__(MJPEGStreamingPresenter)
    presenter._keyboard = keyboard
    presenter._taxi_enabled = True
    presenter._bev_config = BevConfig(tilt_deg=0.0)
    presenter._latest_presented_frame = PresentedFrame(
        timestamp_us=0,
        rgb_host_uint8=np.zeros((1, 1, 3), dtype=np.uint8),
        depth_host_f32=None,
        vehicle_state=vehicle,
        application_state=taxi,
        bev_rig_to_world=rig_pose_from_vehicle_state(vehicle),
    )

    snapshot = presenter._state_snapshot()

    assert snapshot["speed_mps"] == 3.0
    assert isinstance(snapshot["taxi"], dict)
    assert snapshot["taxi"]["phase"] == "seeking_pickup"
    assert snapshot["taxi"]["session_state"] == "playing"
    assert snapshot["taxi"]["high_score"] == 500
    assert snapshot["taxi"]["global_remaining_time_s"] == 0.0
    assert len(snapshot["taxi"]["bev_targets"]) == 4
    assert all(target["visible"] for target in snapshot["taxi"]["bev_targets"])
    assert "bev_arrow" not in snapshot["taxi"]
    assert "bev_enclosure_segments" not in snapshot["taxi"]

    presenter._latest_presented_frame.application_state = replace(
        taxi,
        phase="to_dropoff",
        remaining_time_s=20.0,
        pickup_targets_xyz_m=(),
    )
    dropoff_snapshot = presenter._state_snapshot()

    assert dropoff_snapshot["taxi"]["bev_arrow"] == {"u": 0.5, "v": 0.0}


def test_streaming_state_always_includes_race_bev_gate() -> None:
    keyboard = CrazyRobotaxiKeyboardState()
    vehicle = VehicleState(0.0, 0.0, 0.0, 0.0, 3.0, 0.0)
    race = _race_snapshot(checkpoint_markers=False)
    presenter = MJPEGStreamingPresenter.__new__(MJPEGStreamingPresenter)
    presenter._keyboard = keyboard
    presenter._taxi_enabled = True
    presenter._bev_config = BevConfig(tilt_deg=0.0)
    presenter._latest_presented_frame = PresentedFrame(
        timestamp_us=0,
        rgb_host_uint8=np.zeros((1, 1, 3), dtype=np.uint8),
        depth_host_f32=None,
        vehicle_state=vehicle,
        application_state=race,
        bev_rig_to_world=rig_pose_from_vehicle_state(vehicle),
    )

    snapshot = presenter._state_snapshot()

    assert snapshot["taxi"]["checkpoint_markers"] is False
    assert snapshot["taxi"]["target_label"] == "CHECKPOINT"
    assert snapshot["taxi"]["elapsed_time"] == "0:01.000"
    assert snapshot["taxi"]["bev_arrow"] == {"u": 0.5, "v": 0.0}
    assert snapshot["taxi"]["bev_gate"]["start"]["visible"] is True
    assert snapshot["taxi"]["bev_gate"]["end"]["visible"] is True


def test_streaming_state_snapshot_keeps_upstream_shape_outside_taxi() -> None:
    keyboard = KeyboardState()
    keyboard.update_telemetry(VehicleState(0.0, 0.0, 0.0, 0.5, 3.0, 0.25))
    presenter = BaseMJPEGStreamingPresenter.__new__(BaseMJPEGStreamingPresenter)
    presenter._keyboard = keyboard
    presenter._taxi_enabled = False

    snapshot = presenter._state_snapshot()

    assert snapshot == {
        "speed_mps": 3.0,
        "steer_rad": 0.25,
        "yaw_rad": 0.5,
    }


def _bare_presenter(*, jpeg_quality: int = 85, scale: float = 1.0):
    presenter = MJPEGStreamingPresenter.__new__(MJPEGStreamingPresenter)
    presenter._jpeg_quality = jpeg_quality
    presenter._stream_scale = scale
    presenter._stop_event = threading.Event()
    presenter._frame_bus = LatestFrameBus[bytes]()
    return presenter


def test_stream_stats_counts_skipped_bus_frames_as_drops() -> None:
    stats = _StreamClientStats("/stream test", log_interval_s=1000.0)

    stats.record(frame_count=1, last_seen_count=0, nbytes=100, now=0.0)
    stats.record(frame_count=2, last_seen_count=1, nbytes=100, now=0.1)
    # Slow client: five frames published while the write blocked.
    stats.record(frame_count=7, last_seen_count=2, nbytes=100, now=0.5)

    assert stats.sent_frames == 3
    assert stats.dropped_frames == 4
    assert stats.sent_bytes == 300


def test_stream_stats_first_frame_never_counts_history_as_drops() -> None:
    stats = _StreamClientStats("/stream test", log_interval_s=1000.0)

    # A client connecting mid-session first sees a large frame_count.
    stats.record(frame_count=5000, last_seen_count=0, nbytes=100, now=0.0)

    assert stats.dropped_frames == 0


def test_stream_stats_emits_periodic_log_lines_with_rates() -> None:
    stats = _StreamClientStats("/stream 10.0.0.1", log_interval_s=5.0)

    assert stats.record(frame_count=1, last_seen_count=0, nbytes=1024, now=0.0) is None
    assert stats.record(frame_count=2, last_seen_count=1, nbytes=1024, now=1.0) is None
    line = stats.record(frame_count=4, last_seen_count=2, nbytes=1024, now=6.0)

    assert line is not None
    assert "sent=3" in line
    assert "dropped=1" in line
    assert "/stream 10.0.0.1" in line
    assert "KiB/s" in line
    # Next interval starts fresh from the last log time.
    assert stats.record(frame_count=5, last_seen_count=4, nbytes=1024, now=7.0) is None


def test_scaled_dims_are_even_and_clamped() -> None:
    assert _scaled_dims(1280, 704, 0.5) == (640, 352)
    assert _scaled_dims(1280, 704, 1.0) == (1280, 704)
    assert _scaled_dims(1280, 704, 0.33) == (422, 232)
    assert _scaled_dims(4, 4, 0.1) == (2, 2)


def test_downscale_rgb_halves_frame_and_is_identity_at_full_scale() -> None:
    frame = np.random.default_rng(0).integers(
        0, 255, size=(704, 1280, 3), dtype=np.uint8
    )

    assert _downscale_rgb(frame, 1.0) is frame
    half = _downscale_rgb(frame, 0.5)
    assert half.shape == (352, 640, 3)


def test_publish_respects_stream_scale_and_quality() -> None:
    import io

    from PIL import Image

    frame = np.random.default_rng(1).integers(
        0, 255, size=(704, 1280, 3), dtype=np.uint8
    )
    scaled = _bare_presenter(jpeg_quality=60, scale=0.5)
    full = _bare_presenter(jpeg_quality=85, scale=1.0)

    scaled._publish(frame)
    full._publish(frame)

    scaled_jpeg = scaled._frame_bus.latest().payload
    full_jpeg = full._frame_bus.latest().payload
    assert Image.open(io.BytesIO(scaled_jpeg)).size == (640, 352)
    assert Image.open(io.BytesIO(full_jpeg)).size == (1280, 704)
    assert len(scaled_jpeg) < len(full_jpeg) / 3


def test_presenter_rejects_out_of_range_stream_knobs() -> None:
    import pytest

    raster = RasterConfig()
    keyboard = KeyboardState()
    with pytest.raises(ValueError, match="scale"):
        MJPEGStreamingPresenter(raster, keyboard, "127.0.0.1", 0, stream_scale=0.05)
    with pytest.raises(ValueError, match="quality"):
        MJPEGStreamingPresenter(raster, keyboard, "127.0.0.1", 0, jpeg_quality=0)


def test_streaming_extra_key_handler_fires_on_keydown_case_insensitively() -> None:
    presenter = MJPEGStreamingPresenter.__new__(MJPEGStreamingPresenter)
    presenter._keyboard = CrazyRobotaxiKeyboardState()
    fired: list[str] = []
    presenter._extra_key_handlers = {"k": lambda: fired.append("k")}

    presenter._apply_control("k", True)
    presenter._apply_control("K", True)  # Shift held: browser posts uppercase
    presenter._apply_control("k", False)  # keyup must not re-fire

    assert fired == ["k", "k"]


def test_streaming_extra_key_handlers_reject_reserved_browser_keys() -> None:
    with pytest.raises(ValueError, match="reserved"):
        _validate_extra_key_handlers({"R": lambda: None})
