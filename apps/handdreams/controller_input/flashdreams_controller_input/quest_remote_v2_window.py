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

"""V2 browser or native output with direct or relayed Isaac controller input."""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
from typing import Literal, cast
from urllib.parse import urlencode

from flashdreams.api_v2.client_window import IClientWindow
from flashdreams.runtime_v2.client_window_factory import (
    ClientWindowMode,
    client_window_mode,
)
from flashdreams_controller_input import credentials
from flashdreams_controller_input.controller_source_v2_window import (
    ControllerSourceWindow,
)
from flashdreams_controller_input.controller_types import CLOUDXR_EULA_HELP, InputSource
from flashdreams_controller_input.frames import (
    SOURCE_CLIENT_CAPTURE_NS_KEY,
)
from flashdreams_controller_input.isaac_teleop_setup import prepare_web_client
from flashdreams_controller_input.isaac_teleop_source import IsaacTeleopSource
from flashdreams_controller_input.isaac_teleop_v2_window import ISAAC_TELEOP_WEBRTC_MODE
from flashdreams_controller_input.pose import WRIST_PROXY_POSE_CONVENTION

_CAPTURE_KEYS = (SOURCE_CLIENT_CAPTURE_NS_KEY, "client_capture_ns")


class QuestRemoteMode(ClientWindowMode):
    """Pair direct or relayed controller input with a built-in V2 display."""

    def __init__(self, output_mode: Literal["webrtc", "native-window"]) -> None:
        self.output_mode = output_mode
        self.name = f"quest-remote-{output_mode}"

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        # All modes share one parser; register controller arguments only once.
        if self.output_mode != "webrtc":
            return
        parser.add_argument(
            "--controller-host",
            default="127.0.0.1",
            help="Concrete controller/CloudXR IPv4 interface or hostname; no wildcard binds.",
        )
        parser.add_argument("--controller-port", type=int, default=8445)
        parser.add_argument(
            "--local-cloudxr",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Start local CloudXR for direct Quest/IWER input; disable to accept only the remote controller relay.",
        )
        parser.add_argument(
            "--controller-cert",
            type=Path,
            default=None,
            help="Optional TLS certificate; otherwise create a private managed server identity.",
        )
        parser.add_argument(
            "--controller-key",
            type=Path,
            default=None,
            help="Private key for the controller TLS listener.",
        )
        group = parser.add_argument_group("Server-attached Quest WebXR")
        group.add_argument(
            "--quest-device",
            help="Select a USB Quest; otherwise discover the only authorized local headset.",
        )
        group.add_argument(
            "--quest-require-physical-wake",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Wait for a sleep-to-wake transition before opening Quest Browser.",
        )
        group.add_argument(
            "--quest-accept-server-certificate",
            action="store_true",
            help="Allow this server's CloudXR certificate in the managed Quest tab.",
        )
        group.add_argument(
            "--quest-keep-browser-open-on-error",
            action="store_true",
            help="Keep the managed Quest tab and USB helpers after a connection failure until retry or exit.",
        )
        group.add_argument("--quest-cloudxr-port", type=int, default=48322)

    def check_arguments(self, parsed_args: argparse.Namespace) -> None:
        if parsed_args.local_cloudxr:
            from flashdreams_controller_input.quest_webxr import _validate_options

            if not getattr(parsed_args, "isaac_teleop_accept_eula", False):
                raise ValueError(
                    "The shared controller server requires --isaac-teleop-accept-eula. "
                    + CLOUDXR_EULA_HELP
                )
            ISAAC_TELEOP_WEBRTC_MODE.check_arguments(parsed_args)
            if (
                parsed_args.quest_device is not None
                and not parsed_args.quest_device.strip()
            ):
                raise ValueError("--quest-device must be non-empty")
            _validate_options(self._quest_options(parsed_args))
        offset = parsed_args.isaac_teleop_wrist_offset_m
        if not math.isfinite(offset) or offset < 0.0:
            raise ValueError(
                "--isaac-teleop-wrist-offset-m must be finite and non-negative"
            )
        if not parsed_args.controller_host.strip():
            raise ValueError("--controller-host must be non-empty")
        credentials.resolve_server_address(parsed_args.controller_host)
        if (
            parsed_args.local_cloudxr
            and os.environ.get("TELEOP_STREAM_SERVER_IP") == "0.0.0.0"
        ):
            raise ValueError(
                "Set TELEOP_STREAM_SERVER_IP to a reachable server IPv4 address"
            )
        if not 1 <= parsed_args.controller_port <= 65535:
            raise ValueError("--controller-port must be between 1 and 65535")
        supplied = (
            parsed_args.controller_cert is not None,
            parsed_args.controller_key is not None,
            bool(os.environ.get("FLASHDREAMS_CONTROLLER_TOKEN")),
        )
        if not any(supplied):
            return
        if not all(supplied):
            raise ValueError(
                "Supply --controller-cert, --controller-key, and "
                "FLASHDREAMS_CONTROLLER_TOKEN together, or omit all three "
                "to use a managed server identity"
            )
        for name in ("controller_cert", "controller_key"):
            path = getattr(parsed_args, name)
            if path is None or not path.is_file():
                raise ValueError(
                    f"--{name.replace('_', '-')} must name an existing file"
                )
        if not 32 <= len(os.environ.get("FLASHDREAMS_CONTROLLER_TOKEN", "")) <= 256:
            raise ValueError(
                "Set FLASHDREAMS_CONTROLLER_TOKEN to a shared random token of 32–256 characters"
            )

    @staticmethod
    def _quest_options(args: argparse.Namespace) -> argparse.Namespace:
        return argparse.Namespace(
            quest_device=args.quest_device,
            require_physical_wake=args.quest_require_physical_wake,
            accept_server_certificate=args.quest_accept_server_certificate,
            keep_browser_open_on_error=args.quest_keep_browser_open_on_error,
            cloudxr_port=args.quest_cloudxr_port,
            isaac_teleop_python=args.isaac_teleop_python,
            isaac_teleop_accept_eula=args.isaac_teleop_accept_eula,
            startup_timeout=args.isaac_teleop_startup_timeout_s or 120.0,
        )

    def create(self, parsed_args: argparse.Namespace) -> IClientWindow:
        from flashdreams_controller_input.remote_quest import RemoteQuestSource

        self.check_arguments(parsed_args)
        if parsed_args.local_cloudxr:
            prepare_web_client(python=str(parsed_args.isaac_teleop_python))
        certfile, keyfile = parsed_args.controller_cert, parsed_args.controller_key
        token = os.environ.get("FLASHDREAMS_CONTROLLER_TOKEN", "")
        if certfile is None:
            directory = credentials.server_credentials(parsed_args.controller_host)
            certfile, keyfile = directory / "server.crt", directory / "server.key"
            token = credentials.read_token(directory)
        remote = RemoteQuestSource(
            host=parsed_args.controller_host,
            port=parsed_args.controller_port,
            certfile=certfile,
            keyfile=keyfile,
            token=token,
            wrist_offset_m=parsed_args.isaac_teleop_wrist_offset_m,
        )
        source: InputSource = remote
        status_provider = None
        restart_headset = None
        if parsed_args.local_cloudxr:
            from flashdreams_controller_input.quest_webxr import QuestWebXRProducer
            from flashdreams_controller_input.unified_source import (
                UnifiedControllerSource,
            )

            cloudxr_host = (
                os.environ.get("TELEOP_STREAM_SERVER_IP") or parsed_args.controller_host
            )
            local = IsaacTeleopSource(
                python_executable=str(parsed_args.isaac_teleop_python).strip(),
                port=parsed_args.isaac_teleop_port,
                rate_hz=parsed_args.isaac_teleop_source_rate_hz,
                tracking_rate_hz=parsed_args.isaac_teleop_tracking_hz,
                wrist_offset_m=parsed_args.isaac_teleop_wrist_offset_m,
                startup_timeout_s=parsed_args.isaac_teleop_startup_timeout_s,
                install_dir=str(parsed_args.isaac_teleop_install_dir).strip(),
                env_config=parsed_args.isaac_teleop_env_config,
                device_profile=str(parsed_args.isaac_teleop_device_profile).strip(),
                accept_eula=parsed_args.isaac_teleop_accept_eula,
                host_client=True,
                offline_web_client=True,
                oob_hub=True,
                verbose=parsed_args.isaac_teleop_verbose,
                wait_for_controllers=False,
                cloudxr_host=cloudxr_host,
                cloudxr_port=parsed_args.quest_cloudxr_port,
                cloudxr_proxy_bind=credentials.resolve_server_address(
                    parsed_args.controller_host
                ),
            )
            query = urlencode(
                {
                    "serverIP": cloudxr_host,
                    "port": parsed_args.quest_cloudxr_port,
                    "oobEnable": "1",
                }
            )
            headset = QuestWebXRProducer(
                self._quest_options(parsed_args),
                source=local,
                shared_runtime=True,
                hub_host=cloudxr_host,
                can_connect=lambda: unified.can_connect_headset(),
            )
            unified = UnifiedControllerSource(
                remote=remote,
                local=local,
                headset=headset,
                web_client_url=(
                    f"https://{cloudxr_host}:{parsed_args.quest_cloudxr_port}/client/?{query}"
                ),
                certificate_url=f"https://{cloudxr_host}:{parsed_args.quest_cloudxr_port}/",
            )
            source = unified
            status_provider = unified.connection_status
            restart_headset = headset.restart
        return ControllerSourceWindow(
            output_window=client_window_mode(
                self.output_mode,
                modes=getattr(parsed_args, "_client_window_modes", None),
            ).create(parsed_args),
            source=source,
            status_provider=status_provider,
            restart_headset=restart_headset,
            source_name="Isaac Teleop",
            pose_convention=WRIST_PROXY_POSE_CONVENTION,
            source_thread_name="isaac-server-v2-input",
            source_failure_message="Isaac Teleop input failed",
            initial_window_size=(
                (1280, 940) if self.output_mode == "native-window" else None
            ),
            # Remote input carries its original producer clock separately.
            client_capture_keys=_CAPTURE_KEYS,
        )

    def starting(self, client_window: IClientWindow) -> str | None:
        window = cast("ControllerSourceWindow", client_window)
        return client_window_mode(self.output_mode).starting(window.output_window)


QUEST_REMOTE_WEBRTC_MODE = QuestRemoteMode("webrtc")
"""Remote controller source registered through the existing V2 plugin API."""

QUEST_REMOTE_NATIVE_WINDOW_MODE = QuestRemoteMode("native-window")
"""The same controller source with the standard V2 native presentation backend."""
