# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared participant calibration for the two Isaac Teleop input paths."""

from __future__ import annotations

import copy
import threading

import numpy as np
import pytest
from handdreams.controllers.isaac_teleop_calibration import (
    HANDDREAMS_PARTICIPANT_WRIST_POSE_CONVENTION,
    POSITION_SCALING_CONVENTION,
    ParticipantPoseCalibrator,
    quaternion_to_matrix,
    rotation_to_quaternion,
)
from handdreams.rotations import quat_xyzw_to_matrix

pytestmark = pytest.mark.ci_cpu


def _quaternion(axis: list[float], degrees: float) -> list[float]:
    vector = np.asarray(axis, dtype=np.float64)
    vector /= np.linalg.norm(vector)
    half = np.deg2rad(degrees) * 0.5
    return [*(vector * np.sin(half)), np.cos(half)]


def _frame() -> dict:
    return {
        "left": {
            "position": [-0.21, 1.08, -0.44],
            "orientation": _quaternion([1.0, 2.0, -0.5], 37.0),
            "trigger": 0.2,
            "grip": 0.3,
            "tracked": True,
            "axes": [0.1, -0.4],
            "battery": 0.8,
            "custom": {"nested": [1, 2]},
        },
        "right": {
            "position": [0.28, 1.15, -0.39],
            "orientation": _quaternion([-0.2, 1.0, 0.4], -61.0),
            "trigger": 0.7,
            "grip": 0.9,
            "tracked": True,
        },
        "client_capture_ns": 123456789,
        "relay_timestamp": 12.5,
        "top_level_metadata": {"source": "test"},
    }


def _basis() -> np.ndarray:
    # A nontrivial proper rotation, not merely the default OpenXR axis flip.
    return quat_xyzw_to_matrix(_quaternion([0.3, -1.0, 0.8], 73.0))


@pytest.mark.parametrize(
    "axis", [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, -2.0, 3.0]]
)
@pytest.mark.parametrize("degrees", [0.0, 37.0, 179.999, 180.0, 180.001])
def test_shared_rotation_helpers_preserve_precision_and_canonical_sign(
    axis: list[float], degrees: float
) -> None:
    quaternion = np.asarray(_quaternion(axis, degrees))
    matrix = quaternion_to_matrix(3.0 * quaternion, name="controller orientation")
    result = rotation_to_quaternion(matrix)

    assert matrix.dtype == np.float64
    assert result.dtype == np.float32
    assert result[3] >= 0.0
    assert np.linalg.norm(result) == pytest.approx(1.0)
    assert matrix == pytest.approx(quat_xyzw_to_matrix(quaternion), abs=1e-6)
    assert quaternion_to_matrix(result, name="controller orientation") == pytest.approx(
        matrix, abs=1e-6
    )
    assert quaternion_to_matrix(
        -quaternion, name="controller orientation"
    ) == pytest.approx(matrix)


@pytest.mark.parametrize(
    "quaternion", [[0.0] * 4, [0.0] * 3, [float("nan"), 0.0, 0.0, 1.0]]
)
def test_shared_rotation_helper_rejects_invalid_quaternions(
    quaternion: list[float],
) -> None:
    with pytest.raises(ValueError, match="controller orientation"):
        quaternion_to_matrix(quaternion, name="controller orientation")


def test_configure_maps_neutral_positions_and_rotations_exactly() -> None:
    frame = _frame()
    basis = _basis()
    calibrator = ParticipantPoseCalibrator()

    epoch, anchor = calibrator.configure_with_epoch(frame, basis)

    assert calibrator.active
    assert epoch == 1
    assert calibrator.epoch == 1
    assert calibrator.pose_convention == HANDDREAMS_PARTICIPANT_WRIST_POSE_CONVENTION
    assert calibrator.position_gain == 1.0
    assert POSITION_SCALING_CONVENTION == "neutral-relative"
    for side in ("left", "right"):
        expected_position = basis @ np.asarray(frame[side]["position"])
        assert anchor[side]["position"] == pytest.approx(expected_position)
        assert quat_xyzw_to_matrix(anchor[side]["orientation"]) == pytest.approx(
            np.eye(3), abs=1e-6
        )


def test_position_gain_scales_displacement_from_neutral_not_absolute_position() -> None:
    neutral = _frame()
    current = _frame()
    current["left"]["position"] = [-0.12, 1.22, -0.31]
    current["right"]["position"] = [0.37, 0.96, -0.55]
    basis = _basis()
    gain = 1.5
    calibrator = ParticipantPoseCalibrator()

    anchor = calibrator.configure(neutral, basis, position_gain=gain)
    transformed = calibrator.transform(current)

    assert calibrator.position_gain == gain
    for side in ("left", "right"):
        center = basis @ np.asarray(neutral[side]["position"])
        mapped = basis @ np.asarray(current[side]["position"])
        assert anchor[side]["position"] == pytest.approx(center, abs=1e-6)
        assert transformed[side]["position"] == pytest.approx(
            center + gain * (mapped - center), abs=1e-6
        )


def test_later_pose_uses_the_documented_participant_calibration_formula() -> None:
    neutral = _frame()
    basis = _basis()
    current = _frame()
    current["left"]["position"] = [-0.12, 1.22, -0.31]
    current["left"]["orientation"] = _quaternion([0.5, 0.4, 1.0], 112.0)
    current["right"]["position"] = [0.37, 0.96, -0.55]
    current["right"]["orientation"] = _quaternion([1.0, -0.6, 0.2], 26.0)
    calibrator = ParticipantPoseCalibrator()
    calibrator.configure(neutral, basis)

    transformed = calibrator.transform(current)

    for side in ("left", "right"):
        neutral_rotation = quat_xyzw_to_matrix(neutral[side]["orientation"])
        current_rotation = quat_xyzw_to_matrix(current[side]["orientation"])
        expected_rotation = basis @ current_rotation @ (basis @ neutral_rotation).T
        assert transformed[side]["position"] == pytest.approx(
            basis @ np.asarray(current[side]["position"]), abs=1e-6
        )
        assert quat_xyzw_to_matrix(transformed[side]["orientation"]) == pytest.approx(
            expected_rotation, abs=1e-6
        )


def test_transform_retains_non_pose_fields_without_mutating_the_input() -> None:
    frame = _frame()
    original = copy.deepcopy(frame)
    calibrator = ParticipantPoseCalibrator()
    calibrator.configure(frame, np.eye(3))

    transformed = calibrator.transform(frame)
    transformed["left"]["custom"]["nested"].append(3)
    transformed["top_level_metadata"]["source"] = "changed"

    assert frame == original
    assert transformed["left"]["trigger"] == frame["left"]["trigger"]
    assert transformed["left"]["grip"] == frame["left"]["grip"]
    assert transformed["left"]["axes"] == frame["left"]["axes"]
    assert transformed["left"]["battery"] == frame["left"]["battery"]
    assert frame["left"]["custom"] == {"nested": [1, 2]}
    assert frame["top_level_metadata"] == {"source": "test"}


def test_calibration_does_not_apply_another_grip_to_wrist_offset() -> None:
    frame = _frame()
    # Treat this as the already-offset wrist proxy emitted by Se3AbsRetargeter.
    frame["left"]["position"] = [0.05, 0.0, 0.0]
    frame["left"]["orientation"] = _quaternion([0.0, 1.0, 0.0], 90.0)
    calibrator = ParticipantPoseCalibrator()

    anchor = calibrator.configure(frame, np.eye(3))

    assert anchor["left"]["position"] == pytest.approx([0.05, 0.0, 0.0])


def test_reset_restores_copying_pass_through_and_advances_epoch() -> None:
    frame = _frame()
    calibrator = ParticipantPoseCalibrator()
    calibrator.configure(frame, _basis())

    assert calibrator.reset() == 2
    passthrough = calibrator.transform(frame)

    assert not calibrator.active
    assert calibrator.epoch == 2
    assert calibrator.position_gain is None
    assert passthrough == frame
    assert passthrough is not frame
    assert passthrough["left"] is not frame["left"]


def test_reconfigure_replaces_instead_of_composing_the_previous_calibration() -> None:
    neutral = _frame()
    current = _frame()
    current["left"]["position"] = [0.4, 0.7, -0.2]
    second_basis = quat_xyzw_to_matrix(_quaternion([0.0, 1.0, 0.0], -41.0))
    calibrator = ParticipantPoseCalibrator()
    calibrator.configure(neutral, _basis())

    second_anchor = calibrator.configure(current, second_basis)

    assert calibrator.epoch == 2
    assert second_anchor["left"]["position"] == pytest.approx(
        second_basis @ np.asarray(current["left"]["position"]), abs=1e-6
    )
    assert quat_xyzw_to_matrix(second_anchor["left"]["orientation"]) == pytest.approx(
        np.eye(3), abs=1e-6
    )


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda frame: frame.pop("left"), "left controller"),
        (
            lambda frame: frame["right"].update(orientation=[0.0, 0.0, 0.0, 0.0]),
            "zero norm",
        ),
        (
            lambda frame: frame["left"].update(position=[float("nan"), 0.0, 0.0]),
            "finite values",
        ),
    ],
)
def test_invalid_reconfiguration_is_atomic(mutate, message: str) -> None:
    calibrator = ParticipantPoseCalibrator()
    valid = _frame()
    calibrator.configure(valid, np.eye(3))
    invalid = _frame()
    mutate(invalid)

    with pytest.raises(ValueError, match=message):
        calibrator.configure(invalid, _basis())

    assert calibrator.active
    assert calibrator.epoch == 1
    transformed = calibrator.transform(valid)
    assert transformed["left"]["position"] == pytest.approx(valid["left"]["position"])


def test_invalid_basis_does_not_replace_an_existing_calibration() -> None:
    calibrator = ParticipantPoseCalibrator()
    frame = _frame()
    calibrator.configure(frame, np.eye(3))

    with pytest.raises(ValueError, match="right-handed and orthonormal"):
        calibrator.configure(frame, np.eye(3) * 2.0)

    assert calibrator.epoch == 1


@pytest.mark.parametrize(
    "gain",
    [
        True,
        False,
        0.0,
        -1.0,
        float("inf"),
        float("-inf"),
        float("nan"),
        "1.5",
        None,
    ],
)
def test_invalid_position_gain_does_not_replace_existing_calibration(
    gain: object,
) -> None:
    calibrator = ParticipantPoseCalibrator()
    frame = _frame()
    calibrator.configure(frame, np.eye(3), position_gain=1.25)

    with pytest.raises(ValueError, match="finite positive"):
        calibrator.configure(frame, _basis(), position_gain=gain)

    assert calibrator.active
    assert calibrator.epoch == 1
    assert calibrator.position_gain == 1.25
    assert calibrator.transform(frame)["left"]["position"] == pytest.approx(
        frame["left"]["position"]
    )


def test_tolerated_basis_roundoff_is_projected_before_state_is_installed() -> None:
    frame = _frame()
    basis = np.diag([1.001, 0.999, 1.0 / (1.0 - 0.001**2)])
    calibrator = ParticipantPoseCalibrator()

    anchor = calibrator.configure(frame, basis)
    later = calibrator.transform(frame)

    assert calibrator.active
    assert calibrator.epoch == 1
    for output in (anchor, later):
        for side in ("left", "right"):
            rotation = quat_xyzw_to_matrix(output[side]["orientation"])
            assert rotation == pytest.approx(np.eye(3), abs=1e-6)


def test_transform_many_can_require_an_active_calibration() -> None:
    calibrator = ParticipantPoseCalibrator()

    with pytest.raises(RuntimeError, match="not active"):
        calibrator.transform_many([_frame()], require_active=True)


def test_transform_many_with_epoch_returns_its_atomic_snapshot_generation() -> None:
    frame = _frame()
    calibrator = ParticipantPoseCalibrator()
    calibrator.configure(frame, _basis())

    epoch, window = calibrator.transform_many_with_epoch(
        [frame, frame], require_active=True
    )

    assert epoch == 1
    assert window == calibrator.transform_many([frame, frame], require_active=True)

    assert calibrator.reset() == 2
    reset_epoch, passthrough = calibrator.transform_many_with_epoch([frame])
    assert reset_epoch == 2
    assert passthrough == [frame]


def test_concurrent_reconfigure_never_splits_a_window_between_states() -> None:
    frame = _frame()
    first_basis = np.eye(3)
    second_basis = _basis()
    first = ParticipantPoseCalibrator()
    first.configure(frame, first_basis)
    expected_first = first.transform(frame)
    second = ParticipantPoseCalibrator()
    second.configure(frame, second_basis)
    expected_second = second.transform(frame)
    calibrator = ParticipantPoseCalibrator()
    calibrator.configure(frame, first_basis)
    start = threading.Event()
    observed: list[tuple[int, list[dict]]] = []

    def reconfigure() -> None:
        start.wait()
        for index in range(200):
            calibrator.configure(frame, first_basis if index % 2 else second_basis)

    worker = threading.Thread(target=reconfigure)
    worker.start()
    start.set()
    for _ in range(200):
        observed.append(
            calibrator.transform_many_with_epoch(
                [frame, frame, frame], require_active=True
            )
        )
    worker.join(timeout=2.0)

    assert not worker.is_alive()
    for epoch, window in observed:
        for result in window:
            matches_first = all(
                np.allclose(result[side]["position"], expected_first[side]["position"])
                and np.allclose(
                    quat_xyzw_to_matrix(result[side]["orientation"]),
                    quat_xyzw_to_matrix(expected_first[side]["orientation"]),
                )
                for side in ("left", "right")
            )
            matches_second = all(
                np.allclose(result[side]["position"], expected_second[side]["position"])
                and np.allclose(
                    quat_xyzw_to_matrix(result[side]["orientation"]),
                    quat_xyzw_to_matrix(expected_second[side]["orientation"]),
                )
                for side in ("left", "right")
            )
            assert matches_first or matches_second
        assert all(window[index] == window[0] for index in range(1, len(window)))
        expected = expected_first if epoch % 2 else expected_second
        for result in window:
            for side in ("left", "right"):
                assert result[side]["position"] == pytest.approx(
                    expected[side]["position"]
                )
