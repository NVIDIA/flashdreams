from __future__ import annotations

import numpy as np
from lingbot.webrtc.controls import CameraPoseIntegrator, KeyboardState


def test_keyboard_state_keydown_keyup_roundtrip() -> None:
    state = KeyboardState()
    assert state.apply_event(event="keydown", key="w")
    assert "w" in state.snapshot()

    assert state.apply_event(event="keyup", key="w")
    assert "w" not in state.snapshot()


def test_keyboard_state_rejects_unknown_key() -> None:
    state = KeyboardState()
    assert not state.apply_event(event="keydown", key="x")
    assert len(state.snapshot()) == 0


def test_keyboard_state_latest_turn_key_takes_precedence() -> None:
    state = KeyboardState()
    assert state.apply_event(event="keydown", key="a")
    assert state.apply_event(event="keydown", key="d")
    assert state.resolved_effective_keys() == frozenset({"d"})


def test_keyboard_state_release_restores_previous_turn_key() -> None:
    state = KeyboardState()
    assert state.apply_event(event="keydown", key="a")
    assert state.apply_event(event="keydown", key="d")
    assert state.apply_event(event="keyup", key="d")
    assert state.resolved_effective_keys() == frozenset({"a"})


def test_pose_integrator_idle_keeps_pose_constant() -> None:
    integrator = CameraPoseIntegrator()
    chunk = integrator.next_pose_chunk(num_frames=3, pressed_keys=frozenset())
    assert chunk.shape == (3, 4, 4)
    assert np.allclose(chunk[0], np.eye(4), atol=1e-6)
    assert np.allclose(chunk[1], np.eye(4), atol=1e-6)
    assert np.allclose(chunk[2], np.eye(4), atol=1e-6)


def test_pose_integrator_forward_advances_z_axis() -> None:
    integrator = CameraPoseIntegrator(move_speed=0.5, rotate_speed_rad=0.0)
    chunk = integrator.next_pose_chunk(num_frames=2, pressed_keys=frozenset({"w"}))
    assert np.isclose(chunk[0, 2, 3], 0.5)
    assert np.isclose(chunk[1, 2, 3], 1.0)


def test_pose_integrator_yaw_changes_rotation() -> None:
    integrator = CameraPoseIntegrator(move_speed=0.0, rotate_speed_rad=0.1)
    chunk = integrator.next_pose_chunk(num_frames=1, pressed_keys=frozenset({"a"}))
    assert not np.isclose(chunk[0, 0, 0], 1.0)
    assert np.isclose(chunk[0, 0, 2], -np.sin(0.1), atol=1e-5)


def test_pose_integrator_strafe_moves_along_x() -> None:
    integrator = CameraPoseIntegrator(move_speed=0.5, rotate_speed_rad=0.0)
    chunk = integrator.next_pose_chunk(num_frames=2, pressed_keys=frozenset({"e"}))
    assert np.isclose(chunk[0, 0, 3], 0.5)
    assert np.isclose(chunk[1, 0, 3], 1.0)
