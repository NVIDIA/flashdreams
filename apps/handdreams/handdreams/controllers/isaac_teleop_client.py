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

"""Host client for isolated HandDreams calibration and control transactions."""

from __future__ import annotations

import math
import os
import signal
import socket
import subprocess
import threading
from contextlib import suppress
from pathlib import Path
from typing import Any

import flashdreams_controller_input
from flashdreams_controller_input.isaac_teleop_source import (
    isolated_python_environment,
    sidecar_recovery,
)
from flashdreams_controller_input.isaac_teleop_source import (
    resolve_python as _resolve_python,
)

from handdreams.controllers.controller_types import (
    LiveCalibrationState,
)
from handdreams.controllers.isaac_teleop_calibration_client import (
    CalibrationRpcClient,
    JsonSocketConnection,
    PreparedControlWindow,
)

_REQUEST_TIMEOUT_S = 0.05
_STOP_GRACE_S = 3.0


class IsaacTeleopBridge:
    """Own one isolated calibration process and its private socket."""

    def __init__(
        self,
        *,
        python_executable: str,
        startup_timeout_s: float,
    ) -> None:
        if not math.isfinite(startup_timeout_s) or startup_timeout_s <= 0.0:
            raise ValueError("Isaac Teleop startup timeout must be finite and positive")
        self.python_executable = str(python_executable)
        self.startup_timeout_s = float(startup_timeout_s)
        self._process: subprocess.Popen[Any] | None = None
        self._connection: JsonSocketConnection | None = None
        self._request_lock = threading.Lock()
        self._calibration_client: CalibrationRpcClient | None = None

    def start(self) -> None:
        """Start the calibration process and validate its protocol handshake."""
        if self._process is not None:
            return
        python = _resolve_python(self.python_executable)
        parent, child = socket.socketpair()
        bridge = Path(__file__).with_name("isaac_teleop_bridge.py")
        package_directory = Path(flashdreams_controller_input.__file__).parent
        command = [
            python,
            str(bridge),
            "calibrate",
            "--fd",
            str(child.fileno()),
            "--controller-input-package",
            str(package_directory),
        ]
        try:
            process = subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                pass_fds=(child.fileno(),),
                start_new_session=True,
                env=isolated_python_environment(),
            )
        except BaseException as error:
            parent.close()
            child.close()
            if isinstance(error, Exception):
                raise RuntimeError(
                    f"Isaac Teleop calibration startup failed: {error}. "
                    + sidecar_recovery(self.python_executable)
                ) from error
            raise
        child.close()
        self._process = process
        self._connection = JsonSocketConnection(
            parent,
            timeout_s=self.startup_timeout_s,
            peer_name="Isaac Teleop calibration service",
        )
        try:
            hello = self._connection.receive()
            self._calibration_client = CalibrationRpcClient(
                hello=hello,
                expected_mode="controller-frames",
                round_trip=self._round_trip,
            )
            self._connection.set_timeout(_REQUEST_TIMEOUT_S)
        except BaseException as error:
            self.stop()
            if isinstance(error, Exception):
                raise RuntimeError(
                    f"Isaac Teleop calibration handshake failed: {error}. "
                    + sidecar_recovery(self.python_executable)
                ) from error
            raise
        print(
            "[isaac-teleop] isolated calibration/control service ready",
            flush=True,
        )

    def _require_running(self) -> JsonSocketConnection:
        """Raise when the calibration process can no longer answer a request."""
        process = self._process
        connection = self._connection
        if process is None or process.poll() is not None or connection is None:
            status = process.poll() if process is not None else None
            raise RuntimeError(
                f"Isaac Teleop calibration service is not running (status {status}). "
                + sidecar_recovery(self.python_executable)
            )
        return connection

    def _round_trip(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Serialize one request/reply exchange on the private socket."""
        with self._request_lock:
            return self._require_running().round_trip(payload)

    def configure_calibration(
        self,
        calibration: dict[str, Any] | None,
        *,
        control: dict[str, Any] | None = None,
    ) -> LiveCalibrationState | None:
        """Install or reset participant calibration and source-owned control."""
        client = self._calibration_client
        if client is None:
            raise RuntimeError("Isaac Teleop calibration RPC is not running")
        return client.configure(calibration, control=control)

    def prepare_window(
        self, frames: list[dict[str, Any]], *, command_sequence: int
    ) -> PreparedControlWindow:
        """Prepare one command transaction in the calibration process."""
        client = self._calibration_client
        if client is None:
            raise RuntimeError("Isaac Teleop calibration RPC is not running")
        return client.prepare_window(frames, command_sequence=command_sequence)

    def commit_window(self, prepared: PreparedControlWindow) -> None:
        """Commit one accepted source-owned command transaction."""
        client = self._calibration_client
        if client is None:
            raise RuntimeError("Isaac Teleop calibration RPC is not running")
        client.commit_window(prepared)

    def abort_window(self, prepared: PreparedControlWindow) -> None:
        """Abort one rejected source-owned command transaction."""
        client = self._calibration_client
        if client is None:
            raise RuntimeError("Isaac Teleop calibration RPC is not running")
        client.abort_window(prepared)

    def stop(self) -> None:
        """Close IPC and terminate the isolated process; remain idempotent."""
        with self._request_lock:
            self._calibration_client = None
            connection, self._connection = self._connection, None
        if connection is not None:
            connection.close()
        process, self._process = self._process, None
        if process is not None and process.poll() is None:
            try:
                process.wait(timeout=_STOP_GRACE_S)
            except subprocess.TimeoutExpired:
                process.terminate()
                try:
                    process.wait(timeout=_STOP_GRACE_S)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(int(process.pid), signal.SIGKILL)
                    except (AttributeError, OSError):
                        process.kill()
                    with suppress(subprocess.TimeoutExpired):
                        process.wait(timeout=1.0)
