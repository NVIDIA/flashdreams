# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Transactional tests for the application-owned Isaac wrist-pose pipeline."""

from __future__ import annotations

import copy
from typing import Any, cast

import numpy as np
import pytest
from flashdreams_controller_input.controller_source_v2_window import (
    controller_frame_event,
)
from flashdreams_controller_input.controller_types import Observation
from flashdreams_controller_input.frames import SOURCE_CLIENT_CAPTURE_NS_KEY
from handdreams.controllers import quest_controller
from handdreams.controllers.controller_types import LiveCalibrationState
from handdreams.controllers.isaac_teleop_calibration import (
    HANDDREAMS_PARTICIPANT_WRIST_POSE_CONVENTION,
)
from handdreams.controllers.isaac_teleop_calibration_client import (
    PreparedControlWindow,
)
from handdreams.controllers.isaac_teleop_client import IsaacTeleopBridge
from handdreams.controllers.isaac_teleop_control import (
    HANDDREAMS_ABSOLUTE_COMMAND_TARGET_CONVENTION,
)
from handdreams.controllers.quest_pose import (
    HANDDREAMS_WRIST_PROXY_POSE_CONVENTION,
    OPENXR_LOCAL_GRIP_POSE_CONVENTION,
)
from handdreams.input import controller_frame
from handdreams.teleop import IsaacTeleopControllerPipeline

pytestmark = pytest.mark.ci_cpu


def _frame(capture_ns: int = 1) -> dict[str, Any]:
    return {
        "client_capture_ns": capture_ns,
        "pose_convention": HANDDREAMS_WRIST_PROXY_POSE_CONVENTION,
        "left": {
            "position": [-0.2, 1.0, -0.5],
            "orientation": [0.0, 0.0, 0.0, 1.0],
            "trigger": 0.0,
            "grip": 0.0,
            "tracked": True,
        },
        "right": {
            "position": [0.2, 1.0, -0.5],
            "orientation": [0.0, 0.0, 0.0, 1.0],
            "trigger": 0.0,
            "grip": 0.0,
            "tracked": True,
        },
    }


class _Bridge:
    instance: "_Bridge"

    def __init__(
        self,
        *,
        python_executable: str,
        startup_timeout_s: float,
    ) -> None:
        del python_executable, startup_timeout_s
        self.starts = 0
        self.stops = 0
        self.commits: list[object] = []
        self.aborts: list[object] = []
        self.controls: list[dict[str, Any] | None] = []
        self.calibration_payloads: list[dict[str, Any] | None] = []
        _Bridge.instance = self

    def start(self) -> None:
        self.starts += 1

    def stop(self) -> None:
        self.stops += 1

    def configure_calibration(
        self,
        calibration: dict[str, Any] | None,
        *,
        control: dict[str, Any] | None = None,
    ) -> LiveCalibrationState | None:
        self.calibration_payloads.append(copy.deepcopy(calibration))
        self.controls.append(control)
        if control is None:
            return None
        return LiveCalibrationState(
            pose_space=HANDDREAMS_PARTICIPANT_WRIST_POSE_CONVENTION,
            epoch=len(self.controls),
            control_space=HANDDREAMS_ABSOLUTE_COMMAND_TARGET_CONVENTION,
        )

    def prepare_window(
        self,
        frames: list[dict[str, Any]],
        *,
        command_sequence: int,
    ) -> Any:
        targets: list[dict[str, Any]] = []
        for frame in frames:
            target = {"client_capture_ns": frame["client_capture_ns"]}
            for side in ("left", "right"):
                target[side] = {
                    "position": list(frame[side]["position"]),
                    "orientation": list(frame[side]["orientation"]),
                    "gripper_open": 0.0,
                }
            targets.append(target)
        return PreparedControlWindow(
            frames=targets,
            epoch=1,
            command_sequence=command_sequence,
            token=f"token-{command_sequence}",
        )

    def commit_window(self, receipt: object) -> None:
        self.commits.append(receipt)

    def abort_window(self, receipt: object) -> None:
        self.aborts.append(receipt)


def _pipeline(
    *, slew: float = 0.25
) -> tuple[IsaacTeleopControllerPipeline, dict[str, Any]]:
    pipeline = IsaacTeleopControllerPipeline(
        python_executable="unused",
        startup_timeout_s=1.0,
        bridge_factory=cast(type[IsaacTeleopBridge], _Bridge),
    )
    pipeline.start()
    wrist = pipeline.transform(_frame())
    pipeline.calibrate(
        wrist,
        scene_gripper_open={"left": 1.0, "right": 1.0},
        position_gain=1.2,
        grip_threshold=0.0,
        stale_ms=500.0,
        gripper_slew_per_action=slew,
    )
    return pipeline, wrist


def _grippers(prepared: Any, side: str = "left") -> list[float]:
    return [float(frame[side]["gripper_open"]) for frame in prepared.frames]


def _through_atomic_event(
    frame: dict[str, Any], *, frame_id: int = 0
) -> dict[str, Any]:
    event = controller_frame_event(
        Observation(frame=frame, capture_ns=frame["client_capture_ns"], source="test"),
        session_origin_ns=0,
        frame_id=frame_id,
        pose_convention=frame["pose_convention"],
        source_name="test",
    )
    assert event is not None
    return controller_frame(event)


def test_app_never_reapplies_the_source_wrist_offset() -> None:
    pipeline, wrist = _pipeline()

    assert wrist["left"]["position"][2] == pytest.approx(-0.5)
    assert wrist["pose_convention"] == HANDDREAMS_WRIST_PROXY_POSE_CONVENTION
    prepared = pipeline.prepare_window(
        [copy.deepcopy(wrist), copy.deepcopy(wrist)],
        command_sequence=1,
    )
    assert prepared.frames[0]["left"]["position"][2] == pytest.approx(-0.5)
    pipeline.close()


@pytest.mark.parametrize("atomic_input", [False, True])
def test_iwer_wrist_pose_bypasses_raw_grip_transform(atomic_input: bool) -> None:
    pipeline = IsaacTeleopControllerPipeline(
        python_executable="unused",
        startup_timeout_s=1.0,
        bridge_factory=cast(type[IsaacTeleopBridge], _Bridge),
    )
    pipeline.start()
    source = _frame()
    source["pose_convention"] = HANDDREAMS_WRIST_PROXY_POSE_CONVENTION
    source["source_frame_id"] = 17

    if atomic_input:
        source = _through_atomic_event(source, frame_id=17)
    wrist = pipeline.transform(source)

    assert wrist is not source
    assert wrist["left"]["position"][2] == pytest.approx(-0.5)
    assert wrist["source_frame_id"] == 17
    assert wrist["pose_convention"] == HANDDREAMS_WRIST_PROXY_POSE_CONVENTION
    wrist["left"]["position"][0] = 99
    assert source["left"]["position"][0] == -0.2
    pipeline.close()


@pytest.mark.parametrize("atomic_input", [False, True])
def test_raw_grip_pose_is_rejected_instead_of_silently_reinterpreted(
    atomic_input: bool,
) -> None:
    pipeline = IsaacTeleopControllerPipeline(
        python_executable="unused",
        startup_timeout_s=1.0,
        bridge_factory=cast(type[IsaacTeleopBridge], _Bridge),
    )
    pipeline.start()
    source = _frame()
    source["pose_convention"] = OPENXR_LOCAL_GRIP_POSE_CONVENTION
    if atomic_input:
        source = _through_atomic_event(source)

    with pytest.raises(ValueError, match="unsupported controller pose convention"):
        pipeline.transform(source)
    pipeline.close()


def test_missing_pose_convention_is_rejected() -> None:
    pipeline, _ = _pipeline()
    source = _frame()
    del source["pose_convention"]
    with pytest.raises(ValueError, match="unsupported controller pose convention"):
        pipeline.transform(source)
    pipeline.close()


def test_remote_isaac_wrist_frames_keep_dgx_calibration_without_another_offset() -> (
    None
):
    pipeline = IsaacTeleopControllerPipeline(
        python_executable="unused",
        startup_timeout_s=1.0,
        bridge_factory=cast(type[IsaacTeleopBridge], _Bridge),
    )
    pipeline.start()
    wrists = []
    for frame_id, source_ns in enumerate((1_000_000_000, 1_011_111_111)):
        source = _frame(capture_ns=2_000_000_000 + frame_id)
        source[SOURCE_CLIENT_CAPTURE_NS_KEY] = source_ns
        for side in ("left", "right"):
            source[side]["position"][2] += 0.05
        event = controller_frame_event(
            Observation(
                frame=source,
                capture_ns=source["client_capture_ns"],
                source="remote Isaac Teleop",
            ),
            session_origin_ns=2_000_000_000,
            frame_id=frame_id,
            pose_convention=HANDDREAMS_WRIST_PROXY_POSE_CONVENTION,
            source_name="remote Isaac Teleop",
            client_capture_keys=(SOURCE_CLIENT_CAPTURE_NS_KEY, "client_capture_ns"),
        )
        wrists.append(pipeline.transform(controller_frame(event)))
    pipeline.calibrate(
        wrists[0],
        scene_gripper_open={"left": 1.0, "right": 1.0},
        position_gain=1.2,
        grip_threshold=0.0,
        stale_ms=500.0,
        gripper_slew_per_action=0.25,
    )
    prepared = pipeline.prepare_window(wrists, command_sequence=1)
    pipeline.commit(prepared)
    assert len(_Bridge.instance.calibration_payloads) == 1
    assert len(_Bridge.instance.commits) == 1
    assert [frame["client_capture_ns"] for frame in prepared.frames] == [
        1_000_000_000,
        1_011_111_111,
    ]
    assert all(
        frame["left"]["position"][2] == pytest.approx(-0.45)
        for frame in prepared.frames
    )
    pipeline.close()


def test_calibration_uses_frozen_off_axis_participant_basis() -> None:
    pipeline = IsaacTeleopControllerPipeline(
        python_executable="unused",
        startup_timeout_s=1.0,
        bridge_factory=cast(type[IsaacTeleopBridge], _Bridge),
    )
    pipeline.start()
    raw = _frame()
    yaw = np.deg2rad(30.0)
    orientation = [0.0, float(np.sin(yaw / 2.0)), 0.0, float(np.cos(yaw / 2.0))]
    for side in ("left", "right"):
        raw[side]["orientation"] = orientation
    wrist = pipeline.transform(raw)
    frozen_basis = quest_controller.controller_separation_basis(wrist)
    np.testing.assert_allclose(frozen_basis, np.diag([1.0, -1.0, -1.0]))

    pipeline.calibrate(
        wrist,
        scene_gripper_open={"left": 1.0, "right": 1.0},
        position_gain=1.2,
        grip_threshold=0.0,
        stale_ms=500.0,
        gripper_slew_per_action=0.25,
        participant_basis=frozen_basis,
    )
    v2_calibration = _Bridge.instance.calibration_payloads[-1]
    assert v2_calibration is not None

    np.testing.assert_allclose(
        v2_calibration["participant_basis"],
        frozen_basis,
    )
    pipeline.close()


@pytest.mark.parametrize("yaw_degrees", [30.0, -45.0, 180.0])
def test_default_calibration_and_recenter_ignore_controller_pointing(
    yaw_degrees: float,
) -> None:
    pipeline, wrist = _pipeline()
    initial = _Bridge.instance.calibration_payloads[-1]
    assert initial is not None
    expected_basis = np.diag([1.0, -1.0, -1.0])
    np.testing.assert_allclose(initial["participant_basis"], expected_basis)

    yaw = np.deg2rad(yaw_degrees)
    for side in ("left", "right"):
        wrist[side]["orientation"] = [0.0, np.sin(yaw / 2), 0.0, np.cos(yaw / 2)]
    pipeline.calibrate(
        wrist,
        scene_gripper_open={"left": 1.0, "right": 1.0},
        position_gain=1.2,
        grip_threshold=0.0,
        stale_ms=500.0,
        gripper_slew_per_action=0.25,
    )
    recentered = _Bridge.instance.calibration_payloads[-1]
    assert recentered is not None
    np.testing.assert_allclose(recentered["participant_basis"], expected_basis)
    for side in ("left", "right"):
        np.testing.assert_allclose(
            recentered[side]["orientation"], wrist[side]["orientation"], atol=1e-7
        )
    assert recentered["position_gain"] == 1.2
    pipeline.close()


def test_gripper_slew_commits_transactionally_and_abort_rolls_back() -> None:
    pipeline, wrist = _pipeline(slew=0.25)
    frames = [copy.deepcopy(wrist) for _ in range(3)]

    first = pipeline.prepare_window(frames, command_sequence=1)
    assert _grippers(first) == pytest.approx([1.0, 0.75, 0.5])
    pipeline.abort(first)

    retry = pipeline.prepare_window(frames, command_sequence=2)
    assert _grippers(retry) == pytest.approx([1.0, 0.75, 0.5])
    pipeline.commit(retry)

    following = pipeline.prepare_window(frames, command_sequence=3)
    assert _grippers(following) == pytest.approx([0.5, 0.25, 0.0])
    assert len(_Bridge.instance.aborts) == 1
    assert len(_Bridge.instance.commits) == 1


def test_zero_gripper_slew_applies_desired_target_immediately() -> None:
    pipeline, wrist = _pipeline(slew=0.0)
    prepared = pipeline.prepare_window(
        [copy.deepcopy(wrist) for _ in range(3)],
        command_sequence=1,
    )

    assert _grippers(prepared) == pytest.approx([1.0, 0.0, 0.0])
