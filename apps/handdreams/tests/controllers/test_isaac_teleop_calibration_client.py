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

"""Host validation for Isaac Teleop calibration RPC and socket framing."""

from __future__ import annotations

import copy
import socket
import threading
from collections.abc import Callable
from typing import Any

import pytest
from handdreams.controllers.isaac_teleop_calibration import (
    CONTROLLER_CALIBRATION_EPOCH_KEY,
    HANDDREAMS_PARTICIPANT_WRIST_POSE_CONVENTION,
)
from handdreams.controllers.isaac_teleop_calibration_client import (
    CalibrationRpcClient,
    JsonSocketConnection,
)
from handdreams.controllers.isaac_teleop_calibration_protocol import (
    CALIBRATION_RPC_VERSION,
    CalibrationRequestHandler,
    calibration_error,
    calibration_hello,
)
from handdreams.controllers.isaac_teleop_control import (
    HANDDREAMS_ABSOLUTE_COMMAND_TARGET_CONVENTION,
)

pytestmark = pytest.mark.ci_cpu

_ResponseMutator = Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]]


def _frame(capture_ns: int = 123) -> dict[str, Any]:
    """Build one valid canonical wrist-proxy frame."""
    return {
        "left": {
            "position": [-0.2, 1.0, -0.5],
            "orientation": [0.0, 0.0, 0.0, 1.0],
            "tracked": True,
            "trigger": 0.1,
            "grip": 0.2,
        },
        "right": {
            "position": [0.2, 1.1, -0.4],
            "orientation": [0.0, 2**-0.5, 0.0, 2**-0.5],
            "tracked": True,
            "trigger": 0.3,
            "grip": 0.4,
        },
        "client_capture_ns": capture_ns,
        "relay_timestamp": 12.5,
    }


def _calibration() -> dict[str, Any]:
    """Build the neutral frame and participant basis sent by HandDreams."""
    return {
        **_frame(),
        "participant_basis": [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        "position_gain": 1.5,
    }


def _control() -> dict[str, Any]:
    """Build a minimal sidecar control configuration."""
    return {
        "scene_gripper_open": {"left": 0.8, "right": 0.7},
        "calibration_trigger": {"left": 0.1, "right": 0.3},
    }


def _client(
    mutator: _ResponseMutator | None = None,
) -> tuple[
    CalibrationRpcClient,
    CalibrationRequestHandler,
    list[dict[str, Any]],
]:
    """Build a client whose callback executes the real protocol handler."""
    handler = CalibrationRequestHandler()
    requests: list[dict[str, Any]] = []

    def round_trip(request: dict[str, Any]) -> dict[str, Any]:
        requests.append(copy.deepcopy(request))
        try:
            response = handler.handle(request)
        except Exception as error:  # noqa: BLE001 - emulate sidecar boundary
            response = calibration_error(request, error)
        if mutator is not None:
            response = mutator(request, copy.deepcopy(response))
        return response

    return (
        CalibrationRpcClient(
            hello=calibration_hello("test-mode"),
            expected_mode="test-mode",
            round_trip=round_trip,
        ),
        handler,
        requests,
    )


def test_client_runs_the_full_lifecycle_through_the_real_handler() -> None:
    client, _, requests = _client()
    frame = _frame(456)
    original = copy.deepcopy(frame)

    state = client.configure(_calibration(), control=_control())
    prepared = client.prepare_window([frame], command_sequence=1)
    client.commit_window(prepared)
    reset = client.configure(None)
    with pytest.raises(RuntimeError, match="stale epoch"):
        client.commit_window(prepared)
    replacement = client.configure(_calibration(), control=_control())

    assert state is not None
    assert state.pose_space == HANDDREAMS_PARTICIPANT_WRIST_POSE_CONVENTION
    assert state.epoch == 1
    assert state.control_space == HANDDREAMS_ABSOLUTE_COMMAND_TARGET_CONVENTION
    assert frame == original
    assert prepared.frames[0]["client_capture_ns"] == 456
    assert prepared.frames[0][CONTROLLER_CALIBRATION_EPOCH_KEY] == 1
    assert reset is None
    assert replacement is not None and replacement.epoch == 3
    assert client.state == replacement
    assert client.ready
    assert [request["type"] for request in requests] == [
        "calibration_configure",
        "control_prepare_window",
        "control_commit_window",
        "calibration_reset",
        "calibration_configure",
    ]
    assert [request["request_id"] for request in requests] == [0, 1, 2, 3, 4]
    assert requests[0]["calibration"]["position_gain"] == 1.5
    assert requests[0]["control"]["scene_gripper_open"] == {"left": 0.8, "right": 0.7}
    assert requests[1]["expected_epoch"] == 1


def test_client_rejects_a_window_stale_across_sidecar_reset_and_reconfigure() -> None:
    client, handler, _ = _client()
    state = client.configure(_calibration(), control=_control())
    assert state is not None and state.epoch == 1
    handler.calibrator.reset()
    handler.calibrator.configure(
        _calibration(),
        _calibration()["participant_basis"],
    )

    with pytest.raises(RuntimeError, match="failed.*epoch mismatch"):
        client.prepare_window([_frame()], command_sequence=1)

    assert client.state == state
    assert client.ready


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("type", "hello"),
        ("protocol", 99),
        ("protocol", CALIBRATION_RPC_VERSION - 1),
        ("mode", "other-mode"),
        ("input_pose_convention", "wrong-input"),
        ("output_pose_convention", "wrong-output"),
        ("output_control_convention", "wrong-control"),
        ("position_scaling", "absolute-origin"),
        ("atomic_window_transform", False),
    ],
)
def test_client_rejects_an_incompatible_hello(field: str, value: object) -> None:
    hello = calibration_hello("test-mode")
    hello[field] = value

    with pytest.raises(RuntimeError, match=field):
        CalibrationRpcClient(
            hello=hello,
            expected_mode="test-mode",
            round_trip=lambda _request: {},
        )


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("error", "failed.*deliberate"),
        ("request-id", "out-of-order"),
        ("type", "unexpected response"),
        ("pose-space", "incompatible pose space"),
        ("control-space", "incompatible control space"),
        ("control-config", "different control configuration"),
        ("position-gain", "invalid position_gain"),
        ("position-gain-mismatch", "returned position_gain"),
        ("epoch", "invalid calibration epoch"),
    ],
)
def test_configure_rejects_malformed_responses(case: str, message: str) -> None:
    def mutate(request: dict[str, Any], response: dict[str, Any]) -> dict[str, Any]:
        if case == "error":
            return calibration_error(request, RuntimeError("deliberate rejection"))
        if case == "request-id":
            response["request_id"] += 1
        elif case == "type":
            response["type"] = "wrong-response"
        elif case == "pose-space":
            response["pose_space"] = "wrong-space"
        elif case == "control-space":
            response["control_space"] = "wrong-control"
        elif case == "control-config":
            response["control"] = {}
        elif case == "position-gain":
            response["position_gain"] = True
        elif case == "position-gain-mismatch":
            response["position_gain"] = 1.25
        elif case == "epoch":
            response["epoch"] = True
        return response

    client, _, _ = _client(mutate)

    with pytest.raises(RuntimeError, match=message):
        client.configure(_calibration())

    assert client.state is None
    assert client.ready is (case == "error")


@pytest.mark.parametrize(
    "gain", [None, True, 0.0, -1.0, float("inf"), float("nan"), "1.5"]
)
def test_client_rejects_invalid_requested_position_gain_without_an_exchange(
    gain: object,
) -> None:
    client, _, requests = _client()
    calibration = _calibration()
    calibration["position_gain"] = gain

    with pytest.raises(ValueError, match="finite positive"):
        client.configure(calibration)

    assert requests == []
    assert client.ready


@pytest.mark.parametrize("operation", ["reset", "reconfigure"])
def test_client_rejects_a_nonadvancing_lifecycle_epoch(operation: str) -> None:
    def mutate(request: dict[str, Any], response: dict[str, Any]) -> dict[str, Any]:
        if request["request_id"] == 1:
            response["epoch"] = 1
        return response

    client, _, _ = _client(mutate)
    client.configure(_calibration())

    with pytest.raises(
        RuntimeError,
        match=f"{'configure' if operation == 'reconfigure' else operation} did not advance",
    ):
        client.configure(None if operation == "reset" else _calibration())

    assert client.state is None
    assert not client.ready


def test_transport_failure_poisoning_prevents_another_exchange() -> None:
    calls = 0

    def fail(_request: dict[str, Any]) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        raise TimeoutError("deliberate timeout")

    client = CalibrationRpcClient(
        hello=calibration_hello("test-mode"),
        expected_mode="test-mode",
        round_trip=fail,
    )

    with pytest.raises(RuntimeError, match="unusable.*deliberate timeout"):
        client.configure(_calibration())
    with pytest.raises(RuntimeError, match="unusable.*deliberate timeout"):
        client.configure(_calibration())

    assert calls == 1
    assert client.state is None
    assert not client.ready


def test_transport_timeout_names_the_calibration_operation() -> None:
    """Make a fatal watchdog useful without hiding its transport cause."""

    def fail(_request: dict[str, Any]) -> dict[str, Any]:
        raise TimeoutError("sidecar timed out")

    client = CalibrationRpcClient(
        hello=calibration_hello("test-mode"),
        expected_mode="test-mode",
        round_trip=fail,
    )

    with pytest.raises(
        RuntimeError,
        match=r"calibration_configure.*sidecar timed out",
    ):
        client.configure(_calibration())

    assert client.state is None
    assert not client.ready


def test_socket_round_trip_encodes_one_json_record() -> None:
    parent, peer = socket.socketpair()
    connection = JsonSocketConnection(parent, timeout_s=0.5, peer_name="test peer")
    failures: list[BaseException] = []

    def respond() -> None:
        try:
            assert peer.recv(4096).endswith(b"\n")
            peer.sendall(b'{"type":"ok"}\n')
        except BaseException as error:  # pragma: no cover - asserted below
            failures.append(error)

    worker = threading.Thread(target=respond, daemon=True)
    worker.start()
    try:
        assert connection.round_trip({"type": "request"}) == {"type": "ok"}
    finally:
        connection.close()
        peer.close()
        worker.join(timeout=1.0)

    assert failures == []
    assert not worker.is_alive()
    assert not connection.ready


def test_socket_receive_times_out() -> None:
    parent, peer = socket.socketpair()
    connection = JsonSocketConnection(parent, timeout_s=0.01, peer_name="test peer")
    try:
        with pytest.raises(TimeoutError, match="test peer timed out"):
            connection.receive()
    finally:
        connection.close()
        peer.close()


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (b"{not-json}\n", "invalid JSON"),
        (b"[]\n", "must be an object"),
    ],
)
def test_socket_rejects_malformed_responses(payload: bytes, message: str) -> None:
    parent, peer = socket.socketpair()
    connection = JsonSocketConnection(parent, timeout_s=0.5, peer_name="test peer")
    try:
        peer.sendall(payload)
        with pytest.raises(RuntimeError, match=message):
            connection.receive()
    finally:
        connection.close()
        peer.close()


def test_socket_reports_peer_eof() -> None:
    parent, peer = socket.socketpair()
    connection = JsonSocketConnection(parent, timeout_s=0.5, peer_name="test peer")
    peer.close()
    try:
        with pytest.raises(RuntimeError, match="disconnected"):
            connection.receive()
    finally:
        connection.close()


def test_socket_bounds_requests_and_responses() -> None:
    parent, peer = socket.socketpair()
    connection = JsonSocketConnection(parent, timeout_s=1.0, peer_name="test peer")

    with pytest.raises(ValueError, match="request exceeds 1 MiB"):
        connection.round_trip({"blob": "x" * 1_100_000})

    def send_oversized_response() -> None:
        try:
            peer.sendall(b"x" * 1_100_000)
        except OSError:
            pass

    worker = threading.Thread(target=send_oversized_response, daemon=True)
    worker.start()
    try:
        with pytest.raises(RuntimeError, match="response exceeds 1 MiB"):
            connection.receive()
    finally:
        connection.close()
        peer.close()
        worker.join(timeout=1.0)

    assert not worker.is_alive()


def test_client_control_transaction_lifecycle_uses_explicit_receipts() -> None:
    client, handler, requests = _client()
    state = client.configure(_calibration(), control=_control())
    assert state is not None
    assert state.control_space == HANDDREAMS_ABSOLUTE_COMMAND_TARGET_CONVENTION
    base_token = handler.control_shaper.snapshot().token
    frames = [_frame(456), _frame(789)]

    prepared = client.prepare_window(frames, command_sequence=9)

    assert prepared.epoch == state.epoch
    assert prepared.command_sequence == 9
    assert prepared.token
    assert len(prepared.frames) == 2
    assert all(
        frame["left"]["gripper_open"] == pytest.approx(0.8)
        and frame["right"]["gripper_open"] == pytest.approx(0.7)
        for frame in prepared.frames
    )
    assert handler.control_shaper.snapshot().token == base_token

    client.commit_window(prepared)
    committed_token = handler.control_shaper.snapshot().token
    client.commit_window(prepared)

    assert committed_token != base_token
    assert handler.control_shaper.snapshot().token == committed_token
    assert [request["type"] for request in requests] == [
        "calibration_configure",
        "control_prepare_window",
        "control_commit_window",
        "control_commit_window",
    ]
    assert requests[0]["control"]["scene_gripper_open"] == {
        "left": 0.8,
        "right": 0.7,
    }
    assert requests[1]["command_sequence"] == 9
    assert requests[2]["token"] == prepared.token


def test_client_abort_is_idempotent_and_does_not_publish_candidate() -> None:
    client, handler, requests = _client()
    client.configure(_calibration(), control=_control())
    base_token = handler.control_shaper.snapshot().token
    prepared = client.prepare_window([_frame(456)], command_sequence=2)

    client.abort_window(prepared)
    client.abort_window(prepared)

    assert handler.control_shaper.snapshot().token == base_token
    assert [request["type"] for request in requests[-2:]] == [
        "control_abort_window",
        "control_abort_window",
    ]


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("pose-space", "incompatible pose space"),
        ("control-space", "incompatible control space"),
        ("epoch", "unexpected epoch"),
        ("sequence", "unexpected command_sequence"),
        ("token", "invalid token"),
        ("window-length", "malformed window"),
        ("timestamp", "changed a frame timestamp"),
        ("invalid-frame", "invalid frame"),
        ("epoch-metadata", "omitted the window epoch"),
        ("gripper-open", "omitted gripper_open"),
    ],
)
def test_prepare_rejects_malformed_responses(case: str, message: str) -> None:
    def mutate(request: dict[str, Any], response: dict[str, Any]) -> dict[str, Any]:
        if request["type"] != "control_prepare_window":
            return response
        if case == "pose-space":
            response["pose_space"] = "wrong-space"
        elif case == "control-space":
            response["control_space"] = "wrong-space"
        elif case == "epoch":
            response["epoch"] += 1
        elif case == "sequence":
            response["command_sequence"] += 1
        elif case == "token":
            response["token"] = True
        elif case == "window-length":
            response["frames"] = []
        elif case == "timestamp":
            response["frames"][0]["client_capture_ns"] += 1
        elif case == "invalid-frame":
            response["frames"][0]["left"].pop("orientation")
        elif case == "epoch-metadata":
            response["frames"][0].pop(CONTROLLER_CALIBRATION_EPOCH_KEY)
        elif case == "gripper-open":
            response["frames"][0]["left"].pop("gripper_open")
        return response

    client, _, _ = _client(mutate)
    client.configure(_calibration(), control=_control())

    with pytest.raises(RuntimeError, match=message):
        client.prepare_window([_frame(456)], command_sequence=1)

    assert client.state is None
    assert not client.ready


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("control-space", "incompatible control space"),
        ("epoch", "unexpected epoch"),
        ("sequence", "unexpected command_sequence"),
        ("token", "unexpected token"),
    ],
)
def test_commit_rejects_malformed_responses(case: str, message: str) -> None:
    def mutate(request: dict[str, Any], response: dict[str, Any]) -> dict[str, Any]:
        if request["type"] != "control_commit_window":
            return response
        if case == "control-space":
            response["control_space"] = "wrong-space"
        elif case == "epoch":
            response["epoch"] += 1
        elif case == "sequence":
            response["command_sequence"] += 1
        elif case == "token":
            response["token"] = "wrong-token"
        return response

    client, _, _ = _client(mutate)
    client.configure(_calibration(), control=_control())
    prepared = client.prepare_window([_frame(456)], command_sequence=1)

    with pytest.raises(RuntimeError, match=message):
        client.commit_window(prepared)

    assert client.state is None
    assert not client.ready


def test_client_rejects_invalid_control_config_without_an_exchange() -> None:
    client, _, requests = _client()

    with pytest.raises(ValueError, match="calibration_trigger"):
        client.configure(
            _calibration(),
            control={"scene_gripper_open": {"left": 0.8, "right": 0.7}},
        )
    with pytest.raises(ValueError, match="requires participant calibration"):
        client.configure(None, control=_control())

    assert requests == []
    assert client.ready


@pytest.mark.parametrize(
    ("frames", "sequence", "message"),
    [
        ([], 1, "nonempty list"),
        ([None], 1, "each control frame"),
        ([_frame(456)], True, "non-negative integer"),
        ([_frame(456)], -1, "non-negative integer"),
    ],
)
def test_client_validates_prepare_arguments_without_an_exchange(
    frames: Any, sequence: object, message: str
) -> None:
    client, _, requests = _client()
    client.configure(_calibration(), control=_control())
    exchanged = len(requests)

    with pytest.raises(ValueError, match=message):
        client.prepare_window(frames, command_sequence=sequence)  # ty: ignore[invalid-argument-type] - exercise input validation.

    assert len(requests) == exchanged
    assert client.ready
