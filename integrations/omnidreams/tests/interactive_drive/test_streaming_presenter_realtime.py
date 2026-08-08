# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import threading

import numpy as np
from omnidreams.interactive_drive.config import BevConfig
from omnidreams.interactive_drive.input.keyboard import KeyboardState
from omnidreams.interactive_drive.streaming_presenter import (
    MJPEGStreamingPresenter,
    _as_rgb_host_uint8,
    _publish_if_open,
    _wait_for_bus_frame,
)
from omnidreams.interactive_drive.taxi_game import TaxiGameSnapshot
from omnidreams.interactive_drive.types import VehicleState

from flashdreams.serving.realtime.frame_bus import LatestFrameBus


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
        remaining_time_s=12.0,
        score=100,
    )
    keyboard.update_runtime_state(vehicle, taxi)
    presenter = MJPEGStreamingPresenter.__new__(MJPEGStreamingPresenter)
    presenter._keyboard = keyboard
    presenter._bev_config = BevConfig(tilt_deg=0.0)

    snapshot = presenter._state_snapshot()

    assert snapshot["speed_mps"] == 3.0
    assert isinstance(snapshot["taxi"], dict)
    assert snapshot["taxi"]["phase"] == "to_dropoff"
    assert snapshot["taxi"]["bev_target"]["visible"] is True
