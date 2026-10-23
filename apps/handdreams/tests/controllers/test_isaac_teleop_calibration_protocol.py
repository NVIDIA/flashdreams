# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Calibration RPC contract tests for the isolated Isaac Teleop bridge."""

from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
from typing import Any

import pytest
from handdreams.controllers.isaac_teleop_calibration import (
    CONTROLLER_CALIBRATION_EPOCH_KEY,
    HANDDREAMS_PARTICIPANT_WRIST_POSE_CONVENTION,
    POSITION_SCALING_CONVENTION,
)
from handdreams.controllers.isaac_teleop_calibration_protocol import (
    CALIBRATION_REQUEST_TYPES,
    CALIBRATION_RPC_VERSION,
    CalibrationRequestHandler,
    calibration_error,
    calibration_hello,
)
from handdreams.controllers.isaac_teleop_control import (
    HANDDREAMS_ABSOLUTE_COMMAND_TARGET_CONVENTION,
)
from handdreams.controllers.quest_pose import (
    HANDDREAMS_WRIST_PROXY_POSE_CONVENTION,
)

pytestmark = pytest.mark.ci_cpu


def _frame(*, offset: float = 0.0, capture_ns: int = 123) -> dict[str, Any]:
    return {
        "left": {
            "position": [1.0 + offset, 2.0, 3.0],
            "orientation": [0.0, 0.0, 0.0, 1.0],
            "trigger": 0.25,
            "metadata": {"samples": [1, 2]},
        },
        "right": {
            "position": [-1.0, 1.5 + offset, 0.5],
            "orientation": [0.0, 0.0, 0.0, 1.0],
            "grip": 0.75,
        },
        "client_capture_ns": capture_ns,
    }


def _calibration() -> dict[str, Any]:
    calibration = _frame()
    calibration["participant_basis"] = [
        [0.0, -1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ]
    calibration["position_gain"] = 1.5
    return calibration


def _control() -> dict[str, Any]:
    """Build a minimal control configuration normalized by the sidecar."""
    return {
        "scene_gripper_open": {"left": 0.8, "right": 0.7},
        "calibration_trigger": {"left": 0.25, "right": 0.0},
    }


def _configure_control(
    handler: CalibrationRequestHandler, *, request_id: int = 1
) -> dict[str, Any]:
    return handler.handle(
        {
            "type": "calibration_configure",
            "request_id": request_id,
            "calibration": _calibration(),
            "control": _control(),
        }
    )


def test_calibration_hello_advertises_the_pose_contract() -> None:
    assert calibration_hello("controller-frames") == {
        "type": "calibration_hello",
        "protocol": CALIBRATION_RPC_VERSION,
        "mode": "controller-frames",
        "input_pose_convention": HANDDREAMS_WRIST_PROXY_POSE_CONVENTION,
        "output_pose_convention": HANDDREAMS_PARTICIPANT_WRIST_POSE_CONVENTION,
        "output_control_convention": (HANDDREAMS_ABSOLUTE_COMMAND_TARGET_CONVENTION),
        "position_scaling": POSITION_SCALING_CONVENTION,
        "atomic_window_transform": True,
    }
    assert CALIBRATION_REQUEST_TYPES == {
        "calibration_configure",
        "calibration_reset",
        "control_prepare_window",
        "control_commit_window",
        "control_abort_window",
    }

    with pytest.raises(ValueError, match="nonempty"):
        calibration_hello("")


def test_configure_and_prepare_window_round_trip_metadata_without_mutation() -> None:
    handler = CalibrationRequestHandler()

    configured = _configure_control(handler)

    assert configured["type"] == "calibration_configured"
    assert configured["request_id"] == 1
    assert configured["pose_space"] == HANDDREAMS_PARTICIPANT_WRIST_POSE_CONVENTION
    assert configured["epoch"] == 1
    assert configured["position_gain"] == 1.5
    assert configured["control_space"] == HANDDREAMS_ABSOLUTE_COMMAND_TARGET_CONVENTION
    frames = [
        _frame(offset=0.1, capture_ns=100),
        _frame(offset=0.2, capture_ns=200),
    ]
    original = copy.deepcopy(frames)

    response = handler.handle(
        {
            "type": "control_prepare_window",
            "request_id": 2,
            "expected_epoch": 1,
            "command_sequence": 1,
            "frames": frames,
        }
    )

    assert response["type"] == "control_window_prepared"
    assert response["request_id"] == 2
    assert response["pose_space"] == HANDDREAMS_PARTICIPANT_WRIST_POSE_CONVENTION
    assert response["epoch"] == 1
    assert frames == original
    assert response["frames"] is not frames
    for original_frame, transformed in zip(original, response["frames"], strict=True):
        assert transformed[CONTROLLER_CALIBRATION_EPOCH_KEY] == 1
        assert transformed["left"]["position"] == pytest.approx(
            [
                -original_frame["left"]["position"][1],
                1.0 + 1.5 * (original_frame["left"]["position"][0] - 1.0),
                3.0,
            ]
        )
        assert transformed["left"]["trigger"] == 0.25
        assert transformed["left"]["metadata"] == {"samples": [1, 2]}
        assert transformed["left"]["metadata"] is not original_frame["left"]["metadata"]


def test_prepare_calls_one_atomic_active_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    handler = CalibrationRequestHandler()
    _configure_control(handler)
    original_transform = handler.calibrator.transform_many_with_epoch
    calls: list[tuple[object, bool]] = []

    def tracked_transform(frames, *, require_active=False):
        calls.append((frames, require_active))
        return original_transform(frames, require_active=require_active)

    monkeypatch.setattr(
        handler.calibrator,
        "transform_many_with_epoch",
        tracked_transform,
    )
    frames = [
        _frame(capture_ns=100),
        _frame(offset=0.1, capture_ns=200),
        _frame(offset=0.2, capture_ns=300),
    ]

    response = handler.handle(
        {
            "type": "control_prepare_window",
            "request_id": 2,
            "expected_epoch": 1,
            "command_sequence": 1,
            "frames": frames,
        }
    )

    assert calls == [(frames, True)]
    assert {
        frame[CONTROLLER_CALIBRATION_EPOCH_KEY] for frame in response["frames"]
    } == {1}


def test_configure_installs_and_echoes_normalized_control_shaping() -> None:
    handler = CalibrationRequestHandler()

    response = _configure_control(handler)

    assert response["control_space"] == (HANDDREAMS_ABSOLUTE_COMMAND_TARGET_CONVENTION)
    assert response["control"]["scene_gripper_open"] == {
        "left": 0.8,
        "right": 0.7,
    }
    assert response["control"]["calibration_trigger"] == {
        "left": 0.25,
        "right": 0.0,
    }
    assert response["control"]["grip_threshold"] == 0.0
    assert response["control"]["gripper_slew_per_action"] == 0.25
    assert handler.control_shaper.active


def test_control_transaction_prepare_and_commit_are_exactly_idempotent() -> None:
    handler = CalibrationRequestHandler()
    _configure_control(handler)
    base_token = handler.control_shaper.snapshot().token
    frames = [
        _frame(offset=0.1, capture_ns=100),
        _frame(offset=0.2, capture_ns=200),
    ]
    original = copy.deepcopy(frames)
    prepare = {
        "type": "control_prepare_window",
        "request_id": 2,
        "expected_epoch": 1,
        "command_sequence": 7,
        "frames": frames,
    }

    prepared = handler.handle(prepare)

    assert handler.control_shaper.snapshot().token == base_token
    assert frames == original
    assert prepared["type"] == "control_window_prepared"
    assert prepared["control_space"] == (HANDDREAMS_ABSOLUTE_COMMAND_TARGET_CONVENTION)
    assert prepared["command_sequence"] == 7
    assert isinstance(prepared["token"], str) and prepared["token"]
    assert len(prepared["frames"]) == 2
    assert {
        frame[CONTROLLER_CALIBRATION_EPOCH_KEY] for frame in prepared["frames"]
    } == {1}
    assert all(
        frame["left"]["gripper_open"] == pytest.approx(0.8)
        and frame["right"]["gripper_open"] == pytest.approx(0.7)
        for frame in prepared["frames"]
    )

    retry = handler.handle({**prepare, "request_id": 3})
    assert retry == {**prepared, "request_id": 3}
    assert handler.control_shaper.snapshot().token == base_token

    with pytest.raises(ValueError, match="reused with different input"):
        handler.handle(
            {
                **prepare,
                "request_id": 4,
                "frames": [_frame(offset=0.3, capture_ns=100)],
            }
        )

    commit = {
        "type": "control_commit_window",
        "request_id": 5,
        "expected_epoch": 1,
        "command_sequence": 7,
        "token": prepared["token"],
    }
    committed = handler.handle(commit)
    committed_token = handler.control_shaper.snapshot().token

    assert committed == {
        "type": "control_window_committed",
        "request_id": 5,
        "control_space": HANDDREAMS_ABSOLUTE_COMMAND_TARGET_CONVENTION,
        "epoch": 1,
        "command_sequence": 7,
        "token": prepared["token"],
    }
    assert committed_token != base_token
    assert handler.handle({**commit, "request_id": 6}) == {
        **committed,
        "request_id": 6,
    }
    assert handler.control_shaper.snapshot().token == committed_token


def test_control_transaction_abort_is_idempotent_and_does_not_advance_state() -> None:
    handler = CalibrationRequestHandler()
    _configure_control(handler)
    base_token = handler.control_shaper.snapshot().token
    prepared = handler.handle(
        {
            "type": "control_prepare_window",
            "request_id": 2,
            "expected_epoch": 1,
            "command_sequence": 10,
            "frames": [_frame(capture_ns=100)],
        }
    )
    abort = {
        "type": "control_abort_window",
        "request_id": 3,
        "expected_epoch": 1,
        "command_sequence": 10,
        "token": prepared["token"],
    }

    aborted = handler.handle(abort)

    assert aborted["type"] == "control_window_aborted"
    assert handler.control_shaper.snapshot().token == base_token
    assert handler.handle({**abort, "request_id": 4}) == {
        **aborted,
        "request_id": 4,
    }
    with pytest.raises(ValueError, match="already aborted"):
        handler.handle(
            {
                **abort,
                "type": "control_commit_window",
                "request_id": 5,
            }
        )
    replacement = handler.handle(
        {
            "type": "control_prepare_window",
            "request_id": 6,
            "expected_epoch": 1,
            "command_sequence": 11,
            "frames": [_frame(capture_ns=200)],
        }
    )
    assert replacement["type"] == "control_window_prepared"


def test_control_prepare_requires_the_previous_transaction_to_finish() -> None:
    handler = CalibrationRequestHandler()
    _configure_control(handler)
    handler.handle(
        {
            "type": "control_prepare_window",
            "request_id": 2,
            "expected_epoch": 1,
            "command_sequence": 1,
            "frames": [_frame(capture_ns=100)],
        }
    )

    with pytest.raises(RuntimeError, match="committed or aborted"):
        handler.handle(
            {
                "type": "control_prepare_window",
                "request_id": 3,
                "expected_epoch": 1,
                "command_sequence": 2,
                "frames": [_frame(capture_ns=200)],
            }
        )


def test_invalid_control_config_does_not_publish_calibration_or_control() -> None:
    handler = CalibrationRequestHandler()

    with pytest.raises(ValueError, match="calibration_trigger"):
        handler.handle(
            {
                "type": "calibration_configure",
                "request_id": 1,
                "calibration": _calibration(),
                "control": {"scene_gripper_open": {"left": 0.8, "right": 0.7}},
            }
        )

    assert not handler.calibrator.active
    assert not handler.control_shaper.active


@pytest.mark.parametrize(
    ("update", "message"),
    [
        ({"expected_epoch": True}, "positive integer"),
        ({"command_sequence": True}, "non-negative integer"),
        ({"frames": []}, "nonempty list"),
        ({"frames": [None]}, "each control frame"),
    ],
)
def test_control_prepare_strictly_validates_transaction_fields(
    update: dict[str, Any], message: str
) -> None:
    handler = CalibrationRequestHandler()
    _configure_control(handler)
    request = {
        "type": "control_prepare_window",
        "request_id": 2,
        "expected_epoch": 1,
        "command_sequence": 1,
        "frames": [_frame(capture_ns=100)],
        **update,
    }

    with pytest.raises(ValueError, match=message):
        handler.handle(request)

    assert handler.control_shaper.snapshot().last_capture_ns is None


def test_control_finish_rejects_wrong_identity_without_advancing_state() -> None:
    handler = CalibrationRequestHandler()
    _configure_control(handler)
    base_token = handler.control_shaper.snapshot().token
    prepared = handler.handle(
        {
            "type": "control_prepare_window",
            "request_id": 2,
            "expected_epoch": 1,
            "command_sequence": 3,
            "frames": [_frame(capture_ns=100)],
        }
    )

    for request_id, update, message in (
        (3, {"expected_epoch": 2}, "expected_epoch"),
        (4, {"command_sequence": 4}, "command_sequence"),
        (5, {"token": "wrong"}, "token"),
    ):
        request = {
            "type": "control_commit_window",
            "request_id": request_id,
            "expected_epoch": 1,
            "command_sequence": 3,
            "token": prepared["token"],
            **update,
        }
        with pytest.raises(ValueError, match=message):
            handler.handle(request)
        assert handler.control_shaper.snapshot().token == base_token

    handler.handle(
        {
            "type": "control_commit_window",
            "request_id": 6,
            "expected_epoch": 1,
            "command_sequence": 3,
            "token": prepared["token"],
        }
    )
    assert handler.control_shaper.snapshot().token != base_token


def test_reset_disables_calibration_and_advances_the_epoch() -> None:
    handler = CalibrationRequestHandler()
    _configure_control(handler)

    assert handler.handle({"type": "calibration_reset", "request_id": 2}) == {
        "type": "calibration_reset_done",
        "request_id": 2,
        "epoch": 2,
        "control_space": None,
    }
    assert not handler.calibrator.active
    assert not handler.control_shaper.active

    with pytest.raises(RuntimeError, match="not configured"):
        handler.handle(
            {
                "type": "control_prepare_window",
                "request_id": 3,
                "expected_epoch": 2,
                "command_sequence": 1,
                "frames": [_frame()],
            }
        )


def test_request_ids_are_strictly_increasing_non_boolean_integers() -> None:
    handler = CalibrationRequestHandler()
    handler.handle({"type": "calibration_reset", "request_id": 10})

    for request_id in (True, 10, 9, 1.0, None):
        with pytest.raises(ValueError, match="request_id"):
            handler.handle({"type": "calibration_reset", "request_id": request_id})

    assert (
        handler.handle({"type": "calibration_reset", "request_id": 11})["request_id"]
        == 11
    )


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"type": "unknown", "request_id": 1}, "unknown calibration"),
        (
            {"type": "calibration_configure", "request_id": 1},
            "object calibration",
        ),
        (
            {
                "type": "calibration_configure",
                "request_id": 1,
                "calibration": _frame(),
            },
            "participant_basis",
        ),
        (
            {
                "type": "calibration_configure",
                "request_id": 1,
                "calibration": {
                    **_frame(),
                    "participant_basis": [
                        [1.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0],
                        [0.0, 0.0, 1.0],
                    ],
                },
            },
            "position_gain",
        ),
        (
            {
                "type": "control_prepare_window",
                "request_id": 1,
                "expected_epoch": True,
                "frames": [],
            },
            "positive integer",
        ),
        (
            {
                "type": "control_prepare_window",
                "request_id": 1,
                "expected_epoch": 1,
                "command_sequence": 1,
                "frames": (),
            },
            "nonempty list",
        ),
    ],
)
def test_malformed_requests_raise(payload: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        CalibrationRequestHandler().handle(payload)


@pytest.mark.parametrize("gain", [True, 0.0, -1.0, float("inf"), float("nan"), "1.5"])
def test_configure_rejects_invalid_position_gain(gain: object) -> None:
    calibration = _calibration()
    calibration["position_gain"] = gain
    handler = CalibrationRequestHandler()

    with pytest.raises(ValueError, match="finite positive"):
        handler.handle(
            {
                "type": "calibration_configure",
                "request_id": 1,
                "calibration": calibration,
            }
        )

    assert not handler.calibrator.active


def test_non_object_request_raises() -> None:
    with pytest.raises(TypeError, match="must be an object"):
        CalibrationRequestHandler().handle([])  # ty: ignore[invalid-argument-type] - exercise input validation.


def test_epoch_mismatch_rejects_the_whole_window() -> None:
    handler = CalibrationRequestHandler()
    _configure_control(handler)
    _configure_control(handler, request_id=2)
    frames = [_frame(capture_ns=100), _frame(offset=0.1, capture_ns=200)]
    original = copy.deepcopy(frames)
    baseline = handler.control_shaper.snapshot()

    with pytest.raises(ValueError, match="expected 1, active 2"):
        handler.handle(
            {
                "type": "control_prepare_window",
                "request_id": 3,
                "expected_epoch": 1,
                "command_sequence": 1,
                "frames": frames,
            }
        )

    assert frames == original
    assert handler.control_shaper.snapshot() is baseline
    assert (
        handler.handle(
            {
                "type": "control_prepare_window",
                "request_id": 4,
                "expected_epoch": 2,
                "command_sequence": 1,
                "frames": frames,
            }
        )["epoch"]
        == 2
    )


def test_valid_id_is_consumed_when_request_dispatch_fails() -> None:
    handler = CalibrationRequestHandler()

    with pytest.raises(ValueError, match="unknown"):
        handler.handle({"type": "unknown", "request_id": 5})
    with pytest.raises(ValueError, match="increase strictly"):
        handler.handle({"type": "calibration_reset", "request_id": 5})

    assert (
        handler.handle({"type": "calibration_reset", "request_id": 6})["request_id"]
        == 6
    )


def test_calibration_error_echoes_only_a_valid_request_id() -> None:
    error = RuntimeError("window rejected")

    assert calibration_error({"request_id": 7}, error) == {
        "type": "error",
        "request_id": 7,
        "error": "RuntimeError: window rejected",
    }
    assert calibration_error({"request_id": True}, error)["request_id"] is None
    assert calibration_error([], error)["request_id"] is None


def test_module_supports_standalone_sibling_imports(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module_path = (
        Path(__file__).parents[2]
        / "handdreams"
        / "controllers"
        / "isaac_teleop_calibration_protocol.py"
    )
    monkeypatch.syspath_prepend(str(module_path.parent))
    spec = importlib.util.spec_from_file_location(
        "standalone_isaac_teleop_calibration_protocol",
        module_path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)

    spec.loader.exec_module(module)

    assert module.calibration_hello("standalone")["protocol"] == CALIBRATION_RPC_VERSION
    assert module.CalibrationRequestHandler is not None
