# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Calibration sidecar transport, transaction, and lifecycle coverage."""

from __future__ import annotations

import copy
import json
import math
import os
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, cast

import pytest
from handdreams.controllers import isaac_teleop_client
from handdreams.controllers.isaac_teleop_calibration import (
    CONTROLLER_CALIBRATION_EPOCH_KEY,
    HANDDREAMS_PARTICIPANT_WRIST_POSE_CONVENTION,
)
from handdreams.controllers.isaac_teleop_calibration_client import (
    JsonSocketConnection,
)
from handdreams.controllers.isaac_teleop_calibration_protocol import (
    CalibrationRequestHandler,
    calibration_error,
    calibration_hello,
)
from handdreams.controllers.isaac_teleop_client import IsaacTeleopBridge
from handdreams.controllers.isaac_teleop_control import (
    HANDDREAMS_ABSOLUTE_COMMAND_TARGET_CONVENTION,
)

pytestmark = pytest.mark.ci_cpu

_TIMEOUT_S = 3.0


def test_bridge_preserves_a_virtualenv_python_symlink(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    venv_python = tmp_path / ".venv" / "bin" / "python"
    venv_python.parent.mkdir(parents=True)
    venv_python.symlink_to(sys.executable)
    monkeypatch.chdir(tmp_path)

    resolved = Path(isaac_teleop_client._resolve_python(".venv/bin/python"))

    assert resolved == venv_python.absolute()
    assert resolved.is_symlink()


def _frame() -> dict[str, Any]:
    return {
        side: {
            "position": [-0.25 if side == "left" else 0.25, 1.0, -0.5],
            "orientation": [0.0, 0.0, 0.0, 1.0],
            "trigger": 0.1,
            "grip": 0.5,
            "tracked": True,
            "battery": 0.75,
        }
        for side in ("left", "right")
    } | {"client_capture_ns": 123_456_789, "relay_timestamp": 9876.5}


def _calibration(frame: dict[str, Any]) -> dict[str, Any]:
    return copy.deepcopy(frame) | {
        "participant_basis": [
            [0.0, 0.0, -1.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
        ],
        "position_gain": 1.0,
    }


class _FakeSidecarProcess:
    """Thread-backed stand-in for the inherited-socket child process."""

    def __init__(self, connection: socket.socket, handler: Any) -> None:
        self.args: list[str] = []
        self.pid = 4242
        self.returncode: int | None = None
        self._connection = connection
        self._handler = handler
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        try:
            self._handler(self._connection)
        except (BrokenPipeError, ConnectionError, OSError):
            pass
        finally:
            self._connection.close()
            if self.returncode is None:
                self.returncode = 0

    def poll(self) -> int | None:
        return self.returncode

    def wait(self, timeout: float | None = None) -> int:
        self._thread.join(timeout=timeout)
        if self._thread.is_alive():
            raise subprocess.TimeoutExpired(
                self.args, 0.0 if timeout is None else timeout
            )
        return int(self.returncode or 0)

    def terminate(self) -> None:
        self.returncode = -15
        try:
            self._connection.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        self._connection.close()

    def kill(self) -> None:
        self.terminate()
        self.returncode = -9


def _send_record(connection: socket.socket, payload: object) -> None:
    connection.sendall((json.dumps(payload, separators=(",", ":")) + "\n").encode())


def _sidecar_handler(responder: Any, *, hello: object | None = None) -> Any:
    handshake = calibration_hello("controller-frames") if hello is None else hello

    def handle(connection: socket.socket) -> None:
        _send_record(connection, handshake)
        with connection.makefile("r", encoding="utf-8") as reader:
            while line := reader.readline():
                response = responder(json.loads(line))
                if response is None:
                    return
                _send_record(connection, response)

    return handle


def _install_sidecar(monkeypatch: pytest.MonkeyPatch, handler: Any) -> dict[str, Any]:
    launched: dict[str, Any] = {}

    def popen(command: list[str], **kwargs: Any) -> _FakeSidecarProcess:
        passed = tuple(kwargs["pass_fds"])
        assert len(passed) == 1
        connection = socket.socket(fileno=os.dup(passed[0]))
        process = _FakeSidecarProcess(connection, handler)
        process.args = list(command)
        launched.update(command=list(command), kwargs=dict(kwargs), process=process)
        return process

    monkeypatch.setattr(isaac_teleop_client.subprocess, "Popen", popen)
    return launched


@pytest.mark.parametrize("timeout", [0.0, -1.0, math.inf, math.nan])
def test_bridge_rejects_invalid_startup_timeout(timeout: float) -> None:
    with pytest.raises(ValueError, match="startup timeout"):
        IsaacTeleopBridge(python_executable=sys.executable, startup_timeout_s=timeout)


def test_bridge_calibrates_and_resolves_ordered_transactions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests: list[dict[str, Any]] = []
    handler = CalibrationRequestHandler()

    def respond(request: dict[str, Any]) -> dict[str, Any]:
        requests.append(copy.deepcopy(request))
        try:
            return handler.handle(request)
        except Exception as error:  # noqa: BLE001 - fake RPC error boundary
            return calibration_error(request, error)

    launched = _install_sidecar(monkeypatch, _sidecar_handler(respond))
    bridge = IsaacTeleopBridge(python_executable=sys.executable, startup_timeout_s=0.5)
    frame = _frame()
    original = copy.deepcopy(frame)
    with pytest.raises(RuntimeError, match="not running"):
        bridge.configure_calibration(None)

    try:
        bridge.start()
        process = launched["process"]
        bridge.start()
        assert launched["process"] is process
        state = bridge.configure_calibration(
            _calibration(frame),
            control={
                "scene_gripper_open": {"left": 0.9, "right": 0.7},
                "calibration_trigger": {"left": 0.1, "right": 0.1},
                "grip_threshold": 0.2,
            },
        )
        assert state is not None
        assert state.pose_space == HANDDREAMS_PARTICIPANT_WRIST_POSE_CONVENTION
        assert state.control_space == HANDDREAMS_ABSOLUTE_COMMAND_TARGET_CONVENTION
        prepared = bridge.prepare_window([frame], command_sequence=31)
        calibrated = prepared.frames[0]
        assert calibrated is not frame
        assert calibrated[CONTROLLER_CALIBRATION_EPOCH_KEY] == state.epoch
        assert calibrated["left"]["position"] == pytest.approx([0.5, 1.0, -0.25])
        assert calibrated["right"]["position"] == pytest.approx([0.5, 1.0, 0.25])
        assert calibrated["left"]["gripper_open"] == pytest.approx(0.9)
        assert calibrated["left"]["battery"] == frame["left"]["battery"]
        assert "recenter" not in calibrated["right"]
        assert calibrated["relay_timestamp"] == frame["relay_timestamp"]
        assert frame == original
        retried = bridge.prepare_window([frame], command_sequence=31)
        assert retried == prepared
        bridge.commit_window(prepared)
        bridge.commit_window(prepared)
        aborted = bridge.prepare_window([frame], command_sequence=32)
        bridge.abort_window(aborted)
        assert bridge.configure_calibration(None) is None
        with pytest.raises(RuntimeError, match="not active"):
            bridge.prepare_window([frame], command_sequence=33)
    finally:
        bridge.stop()
        bridge.stop()

    assert launched["process"].wait(timeout=_TIMEOUT_S) == 0
    with pytest.raises(RuntimeError, match="not running"):
        bridge.configure_calibration(None)
    assert [request["request_id"] for request in requests] == list(range(8))
    assert [request["type"] for request in requests] == [
        "calibration_configure",
        "control_prepare_window",
        "control_prepare_window",
        "control_commit_window",
        "control_commit_window",
        "control_prepare_window",
        "control_abort_window",
        "calibration_reset",
    ]
    command = launched["command"]
    assert command[2] == "calibrate"
    assert command[3] == "--fd"
    package = Path(isaac_teleop_client.flashdreams_controller_input.__file__).parent
    assert command[-2:] == ["--controller-input-package", str(package)]
    environment = launched["kwargs"]["env"]
    assert environment["PYTHONNOUSERSITE"] == "1"
    assert "PYTHONPATH" not in environment
    assert "PYTHONHOME" not in environment
    assert launched["kwargs"]["stdin"] is subprocess.DEVNULL
    assert launched["kwargs"]["start_new_session"] is True


def test_calibration_exchanges_share_one_transaction_lock() -> None:
    class _RunningProcess:
        @staticmethod
        def poll() -> None:
            return None

    bridge = IsaacTeleopBridge(python_executable=sys.executable, startup_timeout_s=0.5)
    bridge._process = cast(subprocess.Popen[Any], _RunningProcess())
    sent: list[int] = []
    first_sent = threading.Event()
    release_first = threading.Event()
    second_started = threading.Event()
    errors: list[BaseException] = []

    class _Connection:
        def round_trip(self, payload: dict[str, Any]) -> dict[str, Any]:
            sent.append(payload["request_id"])
            if payload["request_id"] == 0:
                first_sent.set()
                if not release_first.wait(timeout=_TIMEOUT_S):
                    raise TimeoutError("test did not release the first transaction")
            return {"request_id": payload["request_id"]}

    bridge._connection = cast(JsonSocketConnection, _Connection())

    def request(request_id: int) -> None:
        if request_id == 1:
            second_started.set()
        try:
            bridge._round_trip({"type": "calibration_reset", "request_id": request_id})
        except BaseException as error:  # noqa: BLE001 - asserted across thread
            errors.append(error)

    first = threading.Thread(target=request, args=(0,))
    second = threading.Thread(target=request, args=(1,))
    first.start()
    try:
        assert first_sent.wait(timeout=_TIMEOUT_S)
        second.start()
        assert second_started.wait(timeout=_TIMEOUT_S)
        time.sleep(0.05)
        assert sent == [0]
    finally:
        release_first.set()
        first.join(timeout=_TIMEOUT_S)
        if second.ident is not None:
            second.join(timeout=_TIMEOUT_S)
        bridge._connection = None
        bridge._process = None

    assert not first.is_alive()
    assert not second.is_alive()
    assert errors == []
    assert sent == [0, 1]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("type", "hello"),
        ("protocol", 99),
        ("mode", "cloudxr"),
        ("input_pose_convention", "raw-grip"),
        ("output_pose_convention", "raw-wrist"),
        ("output_control_convention", "raw-controller"),
        ("position_scaling", "origin-relative"),
        ("atomic_window_transform", False),
    ],
)
def test_bridge_rejects_invalid_handshake_and_closes_child(
    monkeypatch: pytest.MonkeyPatch, field: str, value: Any
) -> None:
    hello = calibration_hello("controller-frames") | {field: value}
    launched = _install_sidecar(
        monkeypatch, _sidecar_handler(lambda _request: None, hello=hello)
    )
    bridge = IsaacTeleopBridge(python_executable=sys.executable, startup_timeout_s=0.5)

    with pytest.raises(RuntimeError, match=f"incompatible {field}"):
        bridge.start()

    assert launched["process"].wait(timeout=_TIMEOUT_S) == 0
    with pytest.raises(RuntimeError, match="not running"):
        bridge.configure_calibration(None)


def test_bridge_rejects_out_of_order_response_and_poisons_later_requests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests: list[dict[str, Any]] = []

    def respond(request: dict[str, Any]) -> dict[str, Any]:
        requests.append(request)
        return {
            "type": "calibration_reset_done",
            "request_id": request["request_id"] + 1,
            "epoch": 1,
        }

    _install_sidecar(monkeypatch, _sidecar_handler(respond))
    bridge = IsaacTeleopBridge(python_executable=sys.executable, startup_timeout_s=0.5)
    try:
        bridge.start()
        with pytest.raises(RuntimeError, match="out-of-order"):
            bridge.configure_calibration(None)
        with pytest.raises(RuntimeError, match="unusable"):
            bridge.configure_calibration(None)
    finally:
        bridge.stop()
    assert len(requests) == 1


def test_bridge_surfaces_request_error_and_allows_a_later_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def respond(request: dict[str, Any]) -> dict[str, Any]:
        if request["request_id"] == 0:
            return calibration_error(request, ValueError("malformed calibration"))
        return {
            "type": "calibration_reset_done",
            "request_id": request["request_id"],
            "epoch": 1,
            "control_space": None,
        }

    _install_sidecar(monkeypatch, _sidecar_handler(respond))
    bridge = IsaacTeleopBridge(python_executable=sys.executable, startup_timeout_s=0.5)
    try:
        bridge.start()
        with pytest.raises(RuntimeError, match="malformed calibration"):
            bridge.configure_calibration(None)
        assert bridge.configure_calibration(None) is None
    finally:
        bridge.stop()


@pytest.mark.parametrize("stalled", [False, True])
def test_bridge_bounds_transport_failure_and_poisons_later_requests(
    monkeypatch: pytest.MonkeyPatch, stalled: bool
) -> None:
    def respond(_request: dict[str, Any]) -> None:
        if stalled:
            time.sleep(0.15)

    monkeypatch.setattr(isaac_teleop_client, "_REQUEST_TIMEOUT_S", 0.02)
    _install_sidecar(monkeypatch, _sidecar_handler(respond))
    bridge = IsaacTeleopBridge(python_executable=sys.executable, startup_timeout_s=0.5)
    try:
        bridge.start()
        started = time.monotonic()
        with pytest.raises(
            RuntimeError, match="timed out" if stalled else "disconnected"
        ):
            bridge.configure_calibration(None)
        assert time.monotonic() - started < 0.5
        with pytest.raises(RuntimeError, match="unusable"):
            bridge.configure_calibration(None)
    finally:
        bridge.stop()


@pytest.mark.parametrize("variable", ["PYTHONHOME", "PYTHONPATH"])
def test_calibration_ignores_parent_python_environment_in_real_child(
    variable, tmp_path, monkeypatch
):
    shadow = tmp_path / "main-environment"
    shadow.mkdir()
    (shadow / "numpy.py").write_text("raise RuntimeError('main NumPy leaked')\n")
    monkeypatch.setenv(variable, str(shadow))
    parent_environment = os.environ.copy()
    bridge = IsaacTeleopBridge(python_executable=sys.executable, startup_timeout_s=5.0)
    try:
        bridge.start()
        assert bridge.configure_calibration(None) is None
    finally:
        bridge.stop()
    assert os.environ == parent_environment
