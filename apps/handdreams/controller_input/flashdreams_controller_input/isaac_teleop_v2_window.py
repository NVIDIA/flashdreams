# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""V2 WebRTC output paired with Isaac Teleop WebXR controller input."""

from __future__ import annotations

import argparse
import math
from typing import TYPE_CHECKING, cast

from flashdreams.api_v2.client_window import IClientWindow
from flashdreams.runtime_v2.client_window_factory import ClientWindowMode
from flashdreams_controller_input.controller_source_v2_window import (
    ControllerSourceWindow,
)
from flashdreams_controller_input.controller_types import CLOUDXR_EULA_HELP
from flashdreams_controller_input.isaac_teleop_source import (
    DEFAULT_ISAAC_TELEOP_PYTHON,
    IsaacTeleopSource,
)
from flashdreams_controller_input.pose import (
    DEFAULT_GRIP_TO_WRIST_OFFSET_M,
    WRIST_PROXY_POSE_CONVENTION,
)

if TYPE_CHECKING:
    from flashdreams.runtime_v2.webrtc_client_window import WebRTCClientWindow


_DEFAULT_SOURCE_RATE_HZ = 120.0
_DEFAULT_TRACKING_RATE_HZ = 72.0


class IsaacTeleopWebRTCWindow(ControllerSourceWindow):
    """Standard V2 browser presentation with Isaac Teleop WebXR input."""

    def __init__(
        self,
        *,
        host: str,
        port: int,
        python_executable: str,
        bridge_port: int,
        source_rate_hz: float,
        tracking_rate_hz: float,
        wrist_offset_m: float,
        startup_timeout_s: float | None,
        install_dir: str,
        env_config: str | None,
        device_profile: str,
        accept_eula: bool,
        verbose: bool,
        oob_hub: bool = False,
        _webrtc_window: WebRTCClientWindow | None = None,
        _isaac_source: IsaacTeleopSource | None = None,
    ) -> None:
        source = _isaac_source or IsaacTeleopSource(
            python_executable=python_executable,
            port=bridge_port,
            rate_hz=source_rate_hz,
            tracking_rate_hz=tracking_rate_hz,
            wrist_offset_m=wrist_offset_m,
            startup_timeout_s=startup_timeout_s,
            install_dir=install_dir,
            env_config=env_config,
            device_profile=device_profile,
            accept_eula=accept_eula,
            oob_hub=oob_hub,
            launch_runtime=True,
            verbose=verbose,
            wait_for_controllers=False,
        )
        if _webrtc_window is None:
            from flashdreams.runtime_v2.webrtc_client_window import WebRTCClientWindow

            _webrtc_window = WebRTCClientWindow(host=host, port=port)
        super().__init__(
            output_window=_webrtc_window,
            source=source,
            source_name="Isaac Teleop",
            pose_convention=WRIST_PROXY_POSE_CONVENTION,
            source_thread_name="isaac-teleop-v2-input",
            source_failure_message="Isaac Teleop input failed",
            # Local freshness uses the bridge receiver's monotonic clock.
            client_capture_keys=("client_capture_ns",),
        )


class IsaacTeleopWebRTCMode(ClientWindowMode):
    """Create browser output paired with Isaac Teleop/IWER controller input."""

    name = "isaac-teleop-webrtc"

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        group = parser.add_argument_group("Isaac Teleop WebXR input")
        group.add_argument(
            "--isaac-teleop-python",
            default=DEFAULT_ISAAC_TELEOP_PYTHON,
            help=(
                "Python interpreter for the isolated Isaac Teleop sidecar; "
                f"default: {DEFAULT_ISAAC_TELEOP_PYTHON}."
            ),
        )
        group.add_argument(
            "--isaac-teleop-port",
            type=int,
            default=0,
            help="Loopback sidecar receiver port; 0 chooses an unused port.",
        )
        group.add_argument(
            "--isaac-teleop-source-rate-hz",
            type=float,
            default=_DEFAULT_SOURCE_RATE_HZ,
            help="Maximum controller event rate exposed to V2.",
        )
        group.add_argument(
            "--isaac-teleop-tracking-hz",
            type=float,
            default=_DEFAULT_TRACKING_RATE_HZ,
            help="OpenXR sampling rate inside the isolated sidecar.",
        )
        group.add_argument(
            "--isaac-teleop-wrist-offset-m",
            type=float,
            default=DEFAULT_GRIP_TO_WRIST_OFFSET_M,
            help="Controller-local +Z grip-to-wrist offset applied once by Teleop.",
        )
        group.add_argument(
            "--isaac-teleop-startup-timeout-s",
            type=float,
            default=None,
            help=(
                "Seconds to wait for CloudXR's first compatible controller "
                "frame, including partial or untracked hands; "
                "by default, wait until connected or stopped."
            ),
        )
        group.add_argument(
            "--isaac-teleop-install-dir",
            default="~/.cloudxr",
            help="CloudXR runtime state and cache directory.",
        )
        group.add_argument(
            "--isaac-teleop-env-config",
            default=None,
            help="Optional CloudXR KEY=value environment configuration file.",
        )
        group.add_argument(
            "--isaac-teleop-device-profile",
            default="Quest3",
            help="CloudXR device profile.",
        )
        group.add_argument(
            "--isaac-teleop-accept-eula",
            action="store_true",
            help=CLOUDXR_EULA_HELP,
        )
        group.add_argument(
            "--isaac-teleop-oob-hub",
            action="store_true",
            help=(
                "Enable the CloudXR browser status hub for a remote Quest "
                "companion without local ADB or USB automation."
            ),
        )
        group.add_argument(
            "--isaac-teleop-verbose",
            action="store_true",
            help="Enable verbose Isaac Teleop session logging.",
        )

    def check_arguments(self, parsed_args: argparse.Namespace) -> None:
        if not 0 <= parsed_args.isaac_teleop_port <= 65535:
            raise ValueError("--isaac-teleop-port must be between 0 and 65535")
        for name, value in (
            ("--isaac-teleop-source-rate-hz", parsed_args.isaac_teleop_source_rate_hz),
            ("--isaac-teleop-tracking-hz", parsed_args.isaac_teleop_tracking_hz),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        timeout_s = parsed_args.isaac_teleop_startup_timeout_s
        if timeout_s is not None and (not math.isfinite(timeout_s) or timeout_s <= 0.0):
            raise ValueError(
                "--isaac-teleop-startup-timeout-s must be finite and positive"
            )
        offset = parsed_args.isaac_teleop_wrist_offset_m
        if not math.isfinite(offset) or offset < 0.0:
            raise ValueError(
                "--isaac-teleop-wrist-offset-m must be finite and non-negative"
            )
        for name, value in (
            ("--isaac-teleop-python", parsed_args.isaac_teleop_python),
            ("--isaac-teleop-install-dir", parsed_args.isaac_teleop_install_dir),
            ("--isaac-teleop-device-profile", parsed_args.isaac_teleop_device_profile),
        ):
            if not str(value).strip():
                raise ValueError(f"{name} must be non-empty")

    def create(self, parsed_args: argparse.Namespace) -> IClientWindow:
        self.check_arguments(parsed_args)
        return IsaacTeleopWebRTCWindow(
            host=parsed_args.host,
            port=parsed_args.port,
            python_executable=str(parsed_args.isaac_teleop_python).strip(),
            bridge_port=parsed_args.isaac_teleop_port,
            source_rate_hz=parsed_args.isaac_teleop_source_rate_hz,
            tracking_rate_hz=parsed_args.isaac_teleop_tracking_hz,
            wrist_offset_m=parsed_args.isaac_teleop_wrist_offset_m,
            startup_timeout_s=parsed_args.isaac_teleop_startup_timeout_s,
            install_dir=str(parsed_args.isaac_teleop_install_dir).strip(),
            env_config=parsed_args.isaac_teleop_env_config,
            device_profile=str(parsed_args.isaac_teleop_device_profile).strip(),
            accept_eula=parsed_args.isaac_teleop_accept_eula,
            oob_hub=parsed_args.isaac_teleop_oob_hub,
            verbose=parsed_args.isaac_teleop_verbose,
        )

    def starting(self, client_window: IClientWindow) -> str | None:
        window = cast("IsaacTeleopWebRTCWindow", client_window)
        output = cast("WebRTCClientWindow", window.output_window)
        return f"Open {output.server.url} in a browser."


ISAAC_TELEOP_WEBRTC_MODE = IsaacTeleopWebRTCMode()
"""Client-window entry point loaded by ``flashdreams-run-v2``."""


__all__ = [
    "ISAAC_TELEOP_WEBRTC_MODE",
    "IsaacTeleopWebRTCMode",
    "IsaacTeleopWebRTCWindow",
]
