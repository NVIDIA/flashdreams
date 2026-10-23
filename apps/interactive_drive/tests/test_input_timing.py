# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Vehicle motion from keyboard transitions batched during inference."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from interactive_drive import InteractiveDriveApplication, InteractiveDriveModelLoop
from interactive_drive.config import ChunkConfig, VehicleConfig
from interactive_drive.simulation.ego_vehicle_kinematics import sample_chunk_trajectory
from interactive_drive.types import DriverCommand, FrameChunk, VehicleState

from flashdreams.runtime_v2.user_input_event import (
    FocusUserInputEvent,
    GameWheelUserInputEvent,
    KeyboardInputState,
    KeyboardUserInputEvent,
)
from flashdreams.runtime_v2.user_input_events import UserInputEvents

pytestmark = pytest.mark.ci_cpu


def _key(timestamp_s: float, key: str, pressed: bool) -> KeyboardUserInputEvent:
    return KeyboardUserInputEvent(
        timestamp=np.uint64(round(timestamp_s * 1_000_000)),
        key=key,
        state=KeyboardInputState.PRESSED if pressed else KeyboardInputState.RELEASED,
    )


def _loop(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> InteractiveDriveModelLoop:
    scene = tmp_path / "scene.usdz"
    scene.touch()
    app = InteractiveDriveApplication()
    app.init(["--scene", str(scene), "--total-blocks", "0"])
    session = app.create_session(app.session_desc())
    session.init()
    loop = session.model_loop
    assert isinstance(loop, InteractiveDriveModelLoop)
    loop.state.scene = object()
    loop.state.vehicle = VehicleState(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    loop.state.backend = SimpleNamespace(
        initial_chunk_frames=12,
        chunk_frames=12,
        render_first_chunk=lambda trajectory: FrameChunk(
            (), trajectory.boundary_state_after_chunk, "stub"
        ),
        render_next_chunk=lambda trajectory: FrameChunk(
            (), trajectory.boundary_state_after_chunk, "stub"
        ),
        finalize=lambda: None,
    )
    monkeypatch.setattr(loop, "_frame_chunk_tensors", lambda chunk: [])
    return loop


def test_batched_taps_restart_a_stopped_car_repeatedly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    loop = _loop(tmp_path, monkeypatch)
    for cycle in range(3):
        start_s = 10.0 * (cycle + 1)
        x_before = loop.state.vehicle.x_m
        loop.step(
            cycle * 3,
            UserInputEvents(
                [_key(start_s, "w", True), _key(start_s + 0.1, "w", False)]
            ),
        )
        assert loop.state.vehicle.x_m > x_before
        assert loop.state.vehicle.speed_mps > 0.0
        loop.step(cycle * 3 + 1, UserInputEvents([_key(start_s + 1.0, " ", True)]))
        assert loop.state.vehicle.speed_mps == 0.0
        loop.step(cycle * 3 + 2, UserInputEvents([_key(start_s + 2.0, " ", False)]))
        assert loop.state.vehicle.speed_mps == 0.0


@pytest.mark.parametrize("frames", [1, 12])
@pytest.mark.parametrize("key, direction", [("w", 1.0), ("s", -1.0)])
def test_late_tap_moves_the_car_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    frames: int,
    key: str,
    direction: float,
) -> None:
    loop = _loop(tmp_path, monkeypatch)
    loop.step(0, UserInputEvents([]))
    loop.state.backend.chunk_frames = frames
    loop.step(1, UserInputEvents([_key(0.1, key, True), _key(0.2, key, False)]))
    assert loop.state.vehicle.x_m * direction > 0.0
    assert loop.state.keyboard_track.pending_event_count == 0
    speed = loop.state.vehicle.speed_mps
    loop.step(2, UserInputEvents([]))
    drag = frames / 30 * loop.state.config.app.vehicle.drag_mps2
    assert abs(loop.state.vehicle.speed_mps) == pytest.approx(max(0, abs(speed) - drag))


def test_late_brake_tap_brakes_then_releases(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    loop = _loop(tmp_path, monkeypatch)
    loop.state.vehicle.speed_mps = 5.0
    loop.step(0, UserInputEvents([]))
    before = loop.state.vehicle.speed_mps
    loop.step(1, UserInputEvents([_key(0.1, " ", True), _key(0.2, " ", False)]))
    speed = loop.state.vehicle.speed_mps
    assert speed < before - 0.5
    loop.step(2, UserInputEvents([]))
    assert loop.state.vehicle.speed_mps == pytest.approx(speed - 0.28)


def test_mixed_late_and_current_taps_keep_their_order_and_spacing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    late = _loop(tmp_path, monkeypatch)
    reference = _loop(tmp_path, monkeypatch)
    for loop, offset_s in [(late, 0.0), (reference, 0.3)]:
        loop.step(0, UserInputEvents([]))
        loop.step(
            1,
            UserInputEvents(
                [
                    FocusUserInputEvent(timestamp=np.uint64(0), focused=True),
                    _key(0.0, "unmapped", True),
                    _key(0.1 + offset_s, "w", True),
                    _key(0.2 + offset_s, "w", False),
                    _key(0.6 + offset_s, "a", True),
                    _key(0.7 + offset_s, "a", False),
                ]
            ),
        )
    assert late.state.vehicle.x_m > 0.0
    for field in ["x_m", "y_m", "yaw_rad", "speed_mps"]:
        assert getattr(late.state.vehicle, field) == pytest.approx(
            getattr(reference.state.vehicle, field)
        )
    assert late.state.keyboard_track.pending_event_count == 0


@pytest.mark.parametrize("focus_loss", [False, True])
def test_late_release_stops_a_previously_sampled_held_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, focus_loss: bool
) -> None:
    loop = _loop(tmp_path, monkeypatch)
    loop.step(0, UserInputEvents([_key(0.0, "w", True)]))
    speed = loop.state.vehicle.speed_mps
    release = (
        FocusUserInputEvent(timestamp=np.uint64(100_000), focused=False)
        if focus_loss
        else _key(0.1, "w", False)
    )
    loop.step(1, UserInputEvents([release]))
    assert loop.state.vehicle.speed_mps == pytest.approx(speed - 0.28)
    loop.step(2, UserInputEvents([]))
    assert loop.state.vehicle.speed_mps == pytest.approx(max(0, speed - 0.56))


def test_batched_space_tap_brakes_then_releases(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    loop = _loop(tmp_path, monkeypatch)
    loop.state.vehicle.speed_mps = 5.0
    loop.step(0, UserInputEvents([_key(10.0, " ", True), _key(10.1, " ", False)]))
    speed = loop.state.vehicle.speed_mps
    assert 3.0 < speed < 4.0
    loop.step(1, UserInputEvents([]))
    assert loop.state.vehicle.speed_mps == pytest.approx(speed - 0.28)


@pytest.mark.parametrize("key, direction", [("a", 1.0), ("d", -1.0)])
def test_batched_steering_tap_changes_heading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, key: str, direction: float
) -> None:
    loop = _loop(tmp_path, monkeypatch)
    loop.state.vehicle.speed_mps = 5.0
    loop.step(0, UserInputEvents([_key(10.0, key, True), _key(10.1, key, False)]))
    assert loop.state.vehicle.yaw_rad * direction > 0.0


@pytest.mark.parametrize("frames", [1, 12])
@pytest.mark.parametrize("key, direction", [("w", 1.0), ("s", -1.0)])
def test_long_batch_preserves_early_throttle_and_consumes_releases(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    frames: int,
    key: str,
    direction: float,
) -> None:
    loop = _loop(tmp_path, monkeypatch)
    loop.state.backend.initial_chunk_frames = frames
    loop.step(
        0,
        UserInputEvents(
            [
                _key(10.0, key, True),
                _key(10.1, key, False),
                _key(10.8, "a", True),
                _key(10.9, "a", False),
            ]
        ),
    )
    assert loop.state.vehicle.x_m * direction > 0.0
    assert loop.state.keyboard_track.pending_event_count == 0
    # A fresh input must act in the next chunk, without replaying old controls.
    loop.step(1, UserInputEvents([_key(11.0, " ", True)]))
    assert loop.state.vehicle.speed_mps == 0.0
    stopped_x = loop.state.vehicle.x_m
    loop.step(2, UserInputEvents([_key(11.1, " ", False)]))
    assert loop.state.vehicle.x_m == stopped_x


@pytest.mark.parametrize("key, direction", [("a", 1.0), ("d", -1.0)])
def test_long_batch_preserves_early_steering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, key: str, direction: float
) -> None:
    loop = _loop(tmp_path, monkeypatch)
    loop.state.vehicle.speed_mps = 5.0
    loop.step(
        0,
        UserInputEvents(
            [_key(10.0, key, True), _key(10.1, key, False), _key(10.9, "w", True)]
        ),
    )
    assert loop.state.vehicle.yaw_rad * direction > 0.0
    speed = loop.state.vehicle.speed_mps
    loop.step(1, UserInputEvents([]))
    assert loop.state.vehicle.speed_mps > speed


def test_long_batch_preserves_braking_between_frame_samples(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    loop = _loop(tmp_path, monkeypatch)
    loop.state.vehicle.speed_mps = 5.0
    loop.step(
        0,
        UserInputEvents(
            [
                _key(10.0, "a", True),
                _key(10.01, "a", False),
                _key(10.11, " ", True),
                _key(10.12, " ", False),
                _key(10.9, "d", True),
            ]
        ),
    )
    coast_speed = 5.0 - 0.4 * loop.state.config.app.vehicle.drag_mps2
    assert loop.state.vehicle.speed_mps < coast_speed - 0.02
    speed = loop.state.vehicle.speed_mps
    loop.step(1, UserInputEvents([]))
    assert loop.state.vehicle.speed_mps == pytest.approx(speed - 0.28)


def test_focus_loss_releases_throttle_after_a_long_batch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    loop = _loop(tmp_path, monkeypatch)
    loop.step(
        0,
        UserInputEvents(
            [
                _key(10.0, "w", True),
                FocusUserInputEvent(timestamp=np.uint64(10_900_000), focused=False),
            ]
        ),
    )
    speed = loop.state.vehicle.speed_mps
    assert speed > 0.0
    loop.step(1, UserInputEvents([]))
    assert loop.state.vehicle.speed_mps == pytest.approx(max(0, speed - 0.28))


def test_controller_override_consumes_batched_keyboard_taps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    loop = _loop(tmp_path, monkeypatch)
    loop.step(
        0,
        UserInputEvents(
            [
                GameWheelUserInputEvent(timestamp=np.uint64(9_000_000), action="state"),
                _key(10.0, "w", True),
                _key(10.1, "w", False),
                _key(10.8, "a", True),
                _key(10.9, "a", False),
            ]
        ),
    )
    assert loop.state.vehicle.x_m == 0.0
    loop.step(
        1,
        UserInputEvents(
            [
                GameWheelUserInputEvent(
                    timestamp=np.uint64(11_000_000), action="disconnected"
                )
            ]
        ),
    )
    assert loop.state.vehicle.x_m == 0.0


def test_timed_controls_keep_whole_chunk_and_per_frame_commands_compatible() -> None:
    command = DriverCommand(throttle=1.0, steer=0.5)
    config = ChunkConfig(fps=30)
    arguments = dict(
        start_state=VehicleState(0.0, 0.0, 0.0, 0.0, 5.0, 0.0),
        start_timestamp_us=100_000,
        chunk_size=12,
        chunk_config=config,
        vehicle_config=VehicleConfig(),
        ground_snapper=None,
    )
    whole = sample_chunk_trajectory(command=command, **arguments)
    per_frame = sample_chunk_trajectory(command=(command,) * 12, **arguments)
    timed = sample_chunk_trajectory(
        command=(command,), command_times_s=(0.0,), **arguments
    )
    for trajectory in (per_frame, timed):
        np.testing.assert_array_equal(trajectory.timestamps_us, whole.timestamps_us)
        np.testing.assert_allclose(trajectory.rig_poses_world, whole.rig_poses_world)
        assert trajectory.boundary_state_after_chunk.speed_mps == pytest.approx(
            whole.boundary_state_after_chunk.speed_mps
        )
