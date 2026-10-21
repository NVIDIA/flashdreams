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


"""Isolated HandDreams calibration and control transaction service."""

from __future__ import annotations

import argparse
import importlib.util
import socket
import sys
from pathlib import Path

if not __package__:
    # Load only the discovered device package, never the main interpreter's
    # site-packages directory: its NumPy/SDK binaries may use a different ABI.
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--controller-input-package", type=Path)
    package_directory = bootstrap.parse_known_args()[0].controller_input_package
    if package_directory is not None:
        spec = importlib.util.spec_from_file_location(
            "flashdreams_controller_input", package_directory / "__init__.py"
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"invalid controller-input package: {package_directory}")
        device_package = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = device_package
        spec.loader.exec_module(device_package)

from flashdreams_controller_input.isaac_teleop_bridge import (
    receive_message as _receive,
)
from flashdreams_controller_input.isaac_teleop_bridge import (
    send_message as _send,
)

if __package__:
    from handdreams.controllers.isaac_teleop_calibration_protocol import (
        CalibrationRequestHandler,
        calibration_error,
        calibration_hello,
    )
else:
    # The host supplies the installed device package path to this interpreter;
    # import local calibration helpers without loading the application stack.
    from isaac_teleop_calibration_protocol import (  # ty: ignore[unresolved-import]
        CalibrationRequestHandler,
        calibration_error,
        calibration_hello,
    )


def _calibrate(args: argparse.Namespace) -> int:
    """Serve calibration and control requests over an inherited socket."""
    calibration = CalibrationRequestHandler()

    with socket.socket(fileno=args.fd) as connection:
        with (
            connection.makefile("r", encoding="utf-8") as reader,
            connection.makefile("w", encoding="utf-8") as writer,
        ):
            _send(writer, calibration_hello("controller-frames"))
            while (request := _receive(reader)) is not None:
                try:
                    response = calibration.handle(request)
                except Exception as error:  # noqa: BLE001 - report across IPC
                    response = calibration_error(request, error)
                _send(writer, response)
    return 0


def main(argv: list[str] | None = None) -> int:
    """Run the isolated application calibration and control service."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("calibrate",))
    parser.add_argument("--fd", type=int, required=True)
    parser.add_argument("--controller-input-package", type=Path)
    args = parser.parse_args(argv)
    try:
        return _calibrate(args)
    except (BrokenPipeError, ConnectionError):
        return 0
    except Exception as error:  # noqa: BLE001 - isolated diagnostic process
        print(
            f"[isaac-teleop] calibration service failed: {type(error).__name__}: {error}",
            file=sys.stderr,
            flush=True,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
