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

"""CPU coverage for the isolated app-owned calibration service."""

from __future__ import annotations

import copy
import os
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any

import flashdreams_controller_input
import pytest
from handdreams.controllers import isaac_teleop_bridge as bridge
from handdreams.controllers.isaac_teleop_calibration import (
    CONTROLLER_CALIBRATION_EPOCH_KEY,
)
from handdreams.controllers.isaac_teleop_calibration_client import (
    JsonSocketConnection,
)
from handdreams.controllers.isaac_teleop_calibration_protocol import (
    calibration_hello,
)

pytestmark = pytest.mark.ci_cpu

_TIMEOUT_S = 10.0


def _frame() -> dict[str, Any]:
    return {
        side: {
            "position": [-0.2 if side == "left" else 0.2, 1.0, -0.5],
            "orientation": [0.0, 0.0, 0.0, 1.0],
            "tracked": True,
            "trigger": 0.1,
            "grip": 0.2,
        }
        for side in ("left", "right")
    } | {"client_capture_ns": 123}


def test_subprocess_calibrates_and_resolves_transactions_without_sdk_or_torch(
    tmp_path: Path,
) -> None:
    package = Path(flashdreams_controller_input.__file__).parent
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(tmp_path / "unrelated-python-path")
    parent, child = socket.socketpair()
    connection = JsonSocketConnection(
        parent, timeout_s=_TIMEOUT_S, peer_name="test calibration service"
    )
    try:
        process = subprocess.Popen(
            [
                sys.executable,
                "-c",
                """
import builtins
from pathlib import Path
import runpy
import sys

package = Path(sys.argv[sys.argv.index('--controller-input-package') + 1])
sys.path[:] = [path for path in sys.path if Path(path) != package.parent]
class NoDevicePackageDiscovery:
    def find_spec(self, fullname, *args):
        if fullname == 'flashdreams_controller_input':
            raise AssertionError('device package must be loaded from its selected path')
sys.meta_path.insert(0, NoDevicePackageDiscovery())
original_import = builtins.__import__
def isolated_import(name, *args, **kwargs):
    if name.split('.')[0] in {'handdreams', 'flashdreams', 'torch', 'isaacteleop'}:
        raise AssertionError(f'unexpected application or SDK import: {name}')
    return original_import(name, *args, **kwargs)
builtins.__import__ = isolated_import
sys.argv = sys.argv[1:]
sys.path.insert(0, str(Path(sys.argv[0]).parent))
runpy.run_path(sys.argv[0], run_name='__main__')
""",
                str(Path(bridge.__file__)),
                "calibrate",
                "--fd",
                str(child.fileno()),
                "--controller-input-package",
                str(package),
            ],
            pass_fds=(child.fileno(),),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=tmp_path,
            env=environment,
            text=True,
        )
    except BaseException:
        connection.close()
        raise
    finally:
        child.close()

    try:
        assert connection.receive() == calibration_hello("controller-frames")
        rejected = connection.round_trip(
            {"type": "controller_frame", "request_id": 0, "frame": _frame()}
        )
        assert rejected["type"] == "error"
        assert "unknown calibration request type" in rejected["error"]

        frame = _frame()
        calibration = copy.deepcopy(frame) | {
            "participant_basis": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            "position_gain": 1.0,
        }
        configured = connection.round_trip(
            {
                "type": "calibration_configure",
                "request_id": 1,
                "calibration": calibration,
                "control": {
                    "scene_gripper_open": {"left": 0.8, "right": 0.7},
                    "calibration_trigger": {"left": 0.1, "right": 0.1},
                },
            }
        )
        assert configured["type"] == "calibration_configured"
        request = {
            "type": "control_prepare_window",
            "request_id": 2,
            "expected_epoch": configured["epoch"],
            "command_sequence": 1,
            "frames": [frame],
        }
        prepared = connection.round_trip(request)
        assert prepared["type"] == "control_window_prepared"
        assert prepared["frames"][0][CONTROLLER_CALIBRATION_EPOCH_KEY] == 1
        assert prepared["frames"][0]["left"]["gripper_open"] == pytest.approx(0.8)
        repeated = connection.round_trip(request | {"request_id": 3})
        assert repeated == prepared | {"request_id": 3}
        transaction = {
            "type": "control_commit_window",
            "request_id": 4,
            "expected_epoch": configured["epoch"],
            "command_sequence": 1,
            "token": prepared["token"],
        }
        assert connection.round_trip(transaction)["type"] == "control_window_committed"
        next_prepared = connection.round_trip(
            request | {"request_id": 5, "command_sequence": 2}
        )
        aborted = connection.round_trip(
            transaction
            | {
                "type": "control_abort_window",
                "request_id": 6,
                "command_sequence": 2,
                "token": next_prepared["token"],
            }
        )
        assert aborted["type"] == "control_window_aborted"
        reset = connection.round_trip({"type": "calibration_reset", "request_id": 7})
        assert reset["type"] == "calibration_reset_done"
        assert reset["epoch"] == 2
    finally:
        connection.close()
        try:
            stdout, stderr = process.communicate(timeout=_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            process.kill()
            process.communicate(timeout=_TIMEOUT_S)
            raise
    assert process.returncode == 0, stdout + stderr
