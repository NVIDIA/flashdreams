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

"""Isaac Teleop controller bridge and installation diagnostics."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import signal
import socket
import sys
import time
from contextlib import ExitStack, contextmanager, nullcontext
from ipaddress import IPv4Address
from pathlib import Path
from types import FrameType
from typing import Any, Iterator, TextIO

import numpy as np

if __package__:
    from flashdreams_controller_input.cloudxr_64k_compat import (
        cloudxr_64k_workaround_required,
        prepare_cloudxr_64k_preload,
    )
    from flashdreams_controller_input.cloudxr_client_identity import (
        CloudXRClientIdentity,
    )
    from flashdreams_controller_input.controller_types import (
        CLOUDXR_EULA_HELP,
        ISAAC_TELEOP_BRIDGE_PROTOCOL_VERSION,
        SUPPORTED_ISAAC_TELEOP_VERSION,
    )
    from flashdreams_controller_input.pose import (
        DEFAULT_GRIP_TO_WRIST_OFFSET_M,
        OPENXR_STAGE_GRIP_POSE_CONVENTION,
        WRIST_PROXY_POSE_CONVENTION,
    )
    from flashdreams_controller_input.quest_usb_local import cached_web_client_dir
else:
    # Direct script execution keeps the SDK interpreter independent of the
    # application's environment, including the FlashDreams runtime.
    from cloudxr_64k_compat import (  # ty: ignore[unresolved-import]
        cloudxr_64k_workaround_required,
        prepare_cloudxr_64k_preload,
    )
    from cloudxr_client_identity import (  # ty: ignore[unresolved-import]
        CloudXRClientIdentity,
    )
    from controller_types import (  # ty: ignore[unresolved-import]
        CLOUDXR_EULA_HELP,
        ISAAC_TELEOP_BRIDGE_PROTOCOL_VERSION,
        SUPPORTED_ISAAC_TELEOP_VERSION,
    )
    from pose import (  # ty: ignore[unresolved-import]
        DEFAULT_GRIP_TO_WRIST_OFFSET_M,
        OPENXR_STAGE_GRIP_POSE_CONVENTION,
        WRIST_PROXY_POSE_CONVENTION,
    )
    from quest_usb_local import cached_web_client_dir  # ty: ignore[unresolved-import]

_SIDES = ("left", "right")


def _build_wrist_retargeters(
    controller_outputs: dict[str, Any],
    *,
    wrist_offset_m: float,
    node_prefix: str,
) -> dict[str, Any]:
    """Attach the canonical Isaac SE(3) grip-to-wrist transform to both hands."""
    if not np.isfinite(wrist_offset_m) or wrist_offset_m < 0.0:
        raise ValueError("wrist offset must be finite and non-negative")

    from isaacteleop.retargeters.se3_retargeter import (  # ty: ignore[unresolved-import]
        Se3AbsRetargeter,
        Se3RetargeterConfig,
    )

    wrist_nodes = {}
    for side in _SIDES:
        config = Se3RetargeterConfig(
            input_device=f"controller_{side}",
            zero_out_xy_rotation=False,
            use_wrist_position=True,
            use_wrist_rotation=True,
            target_offset_x=0.0,
            target_offset_y=0.0,
            target_offset_z=float(wrist_offset_m),
            target_offset_roll=0.0,
            target_offset_pitch=0.0,
            target_offset_yaw=0.0,
        )
        wrist_nodes[side] = Se3AbsRetargeter(
            config, f"{node_prefix}_{side}_grip_to_wrist"
        ).connect({f"controller_{side}": controller_outputs[side]})
    return wrist_nodes


def _build_controller_pipeline(
    wrist_offset_m: float = DEFAULT_GRIP_TO_WRIST_OFFSET_M,
) -> tuple[Any, Any]:
    """Build the canonical Isaac grip-to-wrist graph for both controllers.

    Returns:
        Pipeline graph and controller-index enum from the installed SDK.
    """
    from isaacteleop.retargeting_engine.deviceio_source_nodes import (  # ty: ignore[unresolved-import]
        ControllersSource,
    )
    from isaacteleop.retargeting_engine.interface import (  # ty: ignore[unresolved-import]
        OutputCombiner,
    )
    from isaacteleop.retargeting_engine.tensor_types.indices import (  # ty: ignore[unresolved-import]
        ControllerInputIndex,
    )

    controllers = ControllersSource(name="controllers")
    controller_outputs = {
        "left": controllers.output(ControllersSource.LEFT),
        "right": controllers.output(ControllersSource.RIGHT),
    }
    wrist_nodes = _build_wrist_retargeters(
        controller_outputs,
        wrist_offset_m=wrist_offset_m,
        node_prefix="webxr",
    )
    pipeline = OutputCombiner(
        {
            **{f"controller_{side}": controller_outputs[side] for side in _SIDES},
            **{f"wrist_{side}": wrist_nodes[side].output("ee_pose") for side in _SIDES},
        }
    )
    return pipeline, ControllerInputIndex


def require_supported_version() -> str:
    """Return the installed SDK version after enforcing the pinned API."""
    version = importlib.metadata.version("isaacteleop")
    if version != SUPPORTED_ISAAC_TELEOP_VERSION and not version.startswith(
        f"{SUPPORTED_ISAAC_TELEOP_VERSION}+"
    ):
        raise RuntimeError(
            f"isaacteleop {version} is unsupported; install "
            f"isaacteleop=={SUPPORTED_ISAAC_TELEOP_VERSION}"
        )
    return version


def _as_float(value: Any) -> float:
    """Return one controller scalar as a finite Python float."""
    result = float(value)
    return result if np.isfinite(result) else 0.0


def _controller_payload(
    controller: Any,
    wrist_pose: Any,
    indices: Any,
) -> dict[str, Any] | None:
    """Convert one Isaac controller tensor group to the controller schema.

    Args:
        controller: Optional raw Isaac Teleop controller tensor group.
        wrist_pose: Retargeted canonical wrist-proxy pose tensor.
        indices: Installed SDK's ``ControllerInputIndex`` enum.

    Returns:
        Canonical wrist-proxy pose and input state, or ``None`` when absent.
    """
    if controller.is_none:
        return None

    if not bool(controller[indices.GRIP_IS_VALID]):
        return None

    position, orientation = _retargeted_pose(
        wrist_pose,
        context="Isaac Teleop WebXR bridge",
    )

    primary = _as_float(controller[indices.PRIMARY_CLICK]) >= 0.5
    secondary = _as_float(controller[indices.SECONDARY_CLICK]) >= 0.5
    return {
        "position": position.tolist(),
        # Isaac Teleop uses OpenXR's XYZW convention.
        "orientation": orientation.tolist(),
        "trigger": float(
            np.clip(_as_float(controller[indices.TRIGGER_VALUE]), 0.0, 1.0)
        ),
        "grip": float(np.clip(_as_float(controller[indices.SQUEEZE_VALUE]), 0.0, 1.0)),
        "tracked": True,
        "primary_button": primary,
        "secondary_button": secondary,
        "axes": [
            float(np.clip(_as_float(controller[indices.THUMBSTICK_X]), -1.0, 1.0)),
            float(np.clip(_as_float(controller[indices.THUMBSTICK_Y]), -1.0, 1.0)),
        ],
    }


def controller_result_to_message(
    result: dict[str, Any], indices: Any, *, capture_time: float
) -> dict[str, Any]:
    """Convert an Isaac Teleop result to one XR-frame wire message.

    Args:
        result: Output from ``TeleopSession.step``.
        indices: Installed SDK's ``ControllerInputIndex`` enum.
        capture_time: Monotonic host time for this sample.

    Returns:
        JSON-serializable absolute controller observation.
    """
    return {
        "type": "xr_frame",
        "t_client": float(capture_time),
        "controllers": {
            side: _controller_payload(
                result[f"controller_{side}"],
                result[f"wrist_{side}"],
                indices,
            )
            for side in _SIDES
        },
    }


def _retargeted_pose(
    wrist_pose: Any,
    *,
    context: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate and normalize one absolute wrist pose emitted by Isaac."""
    pose = np.asarray(wrist_pose[0], dtype=np.float32).reshape(-1)
    if pose.shape != (7,) or not np.isfinite(pose).all():
        raise ValueError(f"{context} requires a finite wrist pose[7]")
    position = pose[:3]
    orientation = pose[3:]
    norm = float(np.linalg.norm(orientation))
    if norm < 1e-8:
        raise ValueError(f"{context} returned a zero quaternion")
    return position, orientation / norm


def receive_message(
    stream: TextIO, *, limit: int = 1024 * 1024
) -> dict[str, Any] | None:
    """Read one bounded newline-delimited JSON object, or ``None`` at EOF."""
    line = stream.readline(limit + 1)
    if not line:
        return None
    if len(line) > limit:
        raise ValueError("Isaac Teleop adapter message exceeds 1 MiB")
    payload = json.loads(line)
    if not isinstance(payload, dict):
        raise ValueError("Isaac Teleop adapter message must be an object")
    return payload


def send_message(stream: TextIO, payload: dict[str, Any]) -> None:
    """Write and flush one newline-delimited JSON record."""
    stream.write(json.dumps(payload, separators=(",", ":")) + "\n")
    stream.flush()


def _prepare_cloudxr_64k_preload() -> Path | None:
    """Return a generated CloudXR server preload when this host needs one."""
    if not cloudxr_64k_workaround_required():
        return None
    from isaacteleop.cloudxr.runtime import (  # ty: ignore[unresolved-import]
        get_sdk_path,
    )

    return prepare_cloudxr_64k_preload(get_sdk_path())


@contextmanager
def _temporary_ld_preload(library: Path | None) -> Iterator[None]:
    """Expose *library* only while CloudXR constructs its worker environment."""
    if library is None:
        yield
        return
    value = str(library)
    if os.pathsep in value or any(character.isspace() for character in value):
        raise RuntimeError(f"CloudXR preload path cannot be represented: {value!r}")
    previous = os.environ.get("LD_PRELOAD")
    previous_entries = [] if previous is None else previous.replace(":", " ").split()
    if previous is not None and value in previous_entries:
        preload_value = previous
    elif previous:
        preload_value = f"{value} {previous}"
    else:
        preload_value = value
    os.environ["LD_PRELOAD"] = preload_value
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("LD_PRELOAD", None)
        else:
            os.environ["LD_PRELOAD"] = previous


def _bind_cloudxr_proxy(address: str) -> None:
    """Bind the proxy to an explicit IPv4 address and USB loopback."""
    host = str(IPv4Address(address))
    if host == "0.0.0.0":
        raise ValueError("CloudXR proxy requires a concrete IPv4 bind address")
    hosts = host if host == "127.0.0.1" else [host, "127.0.0.1"]
    from isaacteleop.cloudxr import wss  # ty: ignore[unresolved-import]

    # SDK 1.4.142 advertises TELEOP_PROXY_HOST but binds to all interfaces;
    # it has no bind-address option. Only replace its module-local factory.
    serve = wss.ws_serve

    def serve_bound(*args: Any, **kwargs: Any) -> Any:
        kwargs["host"] = hosts
        kwargs["family"] = socket.AF_INET
        return serve(*args, **kwargs)

    # Keep this installed: the SDK thread can bind after construction returns
    # and may outlive its bounded shutdown wait.
    wss.ws_serve = serve_bound


def _cloudxr_context(args: argparse.Namespace) -> Any:
    """Construct the optional CloudXR runtime context."""
    if not args.launch_cloudxr_runtime:
        return nullcontext(None)

    if not args.accept_eula:
        raise ValueError(f"CloudXR startup requires --accept-eula. {CLOUDXR_EULA_HELP}")

    # The SDK proxy reads this asynchronously in the isolated sidecar.
    os.environ.pop("CONTROL_TOKEN", None)
    overrides = {}
    if args.cloudxr_env_config:
        from isaacteleop.cloudxr.env_config import (  # ty: ignore[unresolved-import]
            EnvConfig,
        )

        overrides = EnvConfig._load_env_file(args.cloudxr_env_config)
        if "CONTROL_TOKEN" in overrides:
            raise ValueError(
                "CloudXR CONTROL_TOKEN authentication is unsupported; "
                "remove CONTROL_TOKEN from --cloudxr-env-config"
            )

    if args.offline_web_client:
        if not args.host_client:
            raise ValueError("--offline-web-client requires --host-client")
        cached_web_client_dir()
        if "TELEOP_WEB_CLIENT_STATIC_DIR" in overrides and overrides[
            "TELEOP_WEB_CLIENT_STATIC_DIR"
        ] != os.environ.get("TELEOP_WEB_CLIENT_STATIC_DIR"):
            raise ValueError(
                "--offline-web-client cannot override TELEOP_WEB_CLIENT_STATIC_DIR "
                "through --cloudxr-env-config"
            )

    if args.oob_hub:
        if overrides.get("TELEOP_OOB_HUB_ONLY", "1") != "1":
            raise ValueError(
                "--oob-hub requires TELEOP_OOB_HUB_ONLY=1; remove the "
                "conflicting setting from --cloudxr-env-config"
            )
        # The managed browser uses its parent's endpoint. The SDK's env-file
        # precedence must not move its hub elsewhere.
        for key in (
            "TELEOP_PROXY_HOST",
            "PROXY_PORT",
            "TELEOP_STREAM_SERVER_IP",
            "TELEOP_STREAM_PORT",
        ):
            if (
                key in overrides
                and key in os.environ
                and overrides[key] != os.environ[key]
            ):
                raise ValueError(
                    f"--oob-hub requires {key} to match the parent environment; "
                    "remove the conflicting setting from --cloudxr-env-config"
                )
        # This module runs in the isolated sidecar. Keep the flag set for its
        # lifetime: the SDK's proxy thread checks it after construction to skip
        # all local ADB automation while retaining the browser status hub.
        os.environ["TELEOP_OOB_HUB_ONLY"] = "1"

    from isaacteleop.cloudxr import CloudXRLauncher  # ty: ignore[unresolved-import]

    proxy_bind = args.cloudxr_proxy_bind
    if proxy_bind is not None or args.offline_web_client:
        _bind_cloudxr_proxy(str(proxy_bind) if proxy_bind is not None else "127.0.0.1")

    preload = _prepare_cloudxr_64k_preload()
    if preload is not None:
        print(
            "[isaac-teleop] using guarded ARM64/64K CloudXR compatibility "
            f"library {preload}",
            flush=True,
        )
    # The pinned CloudXR launcher snapshots os.environ into a dedicated worker
    # subprocess. Restore the bridge environment immediately after its
    # constructor returns so the preload does not leak to the WSS proxy or
    # application process.
    with _temporary_ld_preload(preload):
        return CloudXRLauncher(
            install_dir=args.cloudxr_install_dir,
            env_config=args.cloudxr_env_config,
            device_profile=args.cloudxr_device_profile,
            accept_eula=args.accept_eula,
            setup_oob=args.oob_hub,
            usb_local=False,
            host_client=args.host_client,
        )


def _serve(args: argparse.Namespace) -> int:
    """Run the isolated CloudXR/OpenXR controller bridge."""
    from isaacteleop.teleop_session_manager import (  # ty: ignore[unresolved-import]
        TeleopSession,
        TeleopSessionConfig,
    )

    version = require_supported_version()
    pipeline, indices = _build_controller_pipeline(args.wrist_offset_m)
    session_config = TeleopSessionConfig(
        app_name="FlashDreamsIsaacTeleop",
        pipeline=pipeline,
        verbose=bool(args.verbose),
    )
    stopping = False

    def request_stop(_signum: int, _frame: FrameType | None) -> None:
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    with socket.create_connection(
        (args.host, args.port), timeout=args.connect_timeout_s
    ) as connection:
        connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        with connection.makefile("w", encoding="utf-8") as stream:
            send_message(
                stream,
                {
                    "type": "hello",
                    "protocol": ISAAC_TELEOP_BRIDGE_PROTOCOL_VERSION,
                    "version": version,
                    "mode": "cloudxr-webxr",
                    "input_pose_convention": OPENXR_STAGE_GRIP_POSE_CONVENTION,
                    "output_pose_convention": WRIST_PROXY_POSE_CONVENTION,
                    "pose_convention": WRIST_PROXY_POSE_CONVENTION,
                    "wrist_offset_m": float(args.wrist_offset_m),
                },
            )
            send_message(
                stream,
                {
                    "type": "status",
                    "stage": "connected",
                    "detail": "starting CloudXR/OpenXR",
                },
            )

            with ExitStack() as resources:
                identity = resources.enter_context(
                    CloudXRClientIdentity(
                        enabled=args.oob_hub and args.launch_cloudxr_runtime
                    )
                )
                launcher = resources.enter_context(_cloudxr_context(args))
                resources.callback(identity.close)
                send_message(
                    stream,
                    {
                        "type": "status",
                        "stage": "runtime",
                        "detail": "waiting for a WebXR client",
                    },
                )
                with TeleopSession(session_config) as session:
                    send_message(
                        stream,
                        {
                            "type": "status",
                            "stage": "session",
                            "detail": "waiting for both controllers",
                        },
                    )
                    interval = 1.0 / args.rate_hz
                    deadline = time.monotonic()
                    while not stopping:
                        if launcher is not None:
                            launcher.health_check()
                        result = session.step()
                        captured = time.monotonic()
                        message = controller_result_to_message(
                            result, indices, capture_time=captured
                        )
                        client_id, is_emulated = identity.sample()
                        message["client_connection_id"] = client_id
                        message["is_emulated"] = is_emulated
                        send_message(stream, message)
                        deadline = max(deadline + interval, captured)
                        remaining = deadline - time.monotonic()
                        if remaining > 0.0:
                            time.sleep(remaining)
    return 0


def _doctor() -> int:
    """Verify the installed wheel and controller graph without opening XR."""
    version = require_supported_version()
    importlib.import_module("isaacteleop.cloudxr")
    importlib.import_module("isaacteleop.viz")

    preload = _prepare_cloudxr_64k_preload()
    pipeline, indices = _build_controller_pipeline()
    pipeline.output_types()
    required = (
        "GRIP_POSITION",
        "GRIP_ORIENTATION",
        "GRIP_IS_VALID",
        "TRIGGER_VALUE",
        "SQUEEZE_VALUE",
    )
    missing = [name for name in required if not hasattr(indices, name)]
    if missing:
        raise RuntimeError(f"Isaac Teleop controller schema lacks {missing}")
    print(f"isaacteleop={version}")
    print(f"machine={platform.machine()}")
    print("cloudxr=ok")
    print(
        "cloudxr_64k="
        + (f"prepared:{preload}" if preload is not None else "not-needed")
    )
    print("televiz=ok")
    print("controller_pipeline=ok")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the bridge command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser(
        "doctor", help="check the installed SDK without starting CloudXR"
    )
    serve = subparsers.add_parser("serve", help="stream controller poses to a receiver")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, required=True)
    serve.add_argument("--rate-hz", type=float, default=72.0)
    serve.add_argument(
        "--wrist-offset-m",
        type=float,
        default=DEFAULT_GRIP_TO_WRIST_OFFSET_M,
    )
    serve.add_argument("--connect-timeout-s", type=float, default=10.0)
    serve.add_argument("--cloudxr-install-dir", default="~/.cloudxr")
    serve.add_argument("--cloudxr-env-config", default=None)
    serve.add_argument(
        "--cloudxr-proxy-bind",
        type=IPv4Address,
        default=None,
        help="Proxy IPv4 interface; USB loopback remains available on this port.",
    )
    serve.add_argument("--cloudxr-device-profile", default="Quest3")
    serve.add_argument("--accept-eula", action="store_true", help=CLOUDXR_EULA_HELP)
    serve.add_argument("--host-client", action="store_true")
    serve.add_argument("--offline-web-client", action="store_true")
    serve.add_argument(
        "--oob-hub",
        action="store_true",
        help="Enable the browser status hub without local ADB or USB automation.",
    )
    serve.add_argument(
        "--no-launch-cloudxr-runtime",
        dest="launch_cloudxr_runtime",
        action="store_false",
        default=True,
    )
    serve.add_argument("--verbose", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run diagnostics or the controller bridge."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "serve" and (
        not np.isfinite(args.wrist_offset_m) or args.wrist_offset_m < 0.0
    ):
        parser.error("--wrist-offset-m must be finite and non-negative")
    if args.command == "doctor":
        return _doctor()
    if args.launch_cloudxr_runtime and not args.accept_eula:
        parser.error(f"CloudXR startup requires --accept-eula. {CLOUDXR_EULA_HELP}")
    if args.rate_hz <= 0.0:
        parser.error("--rate-hz must be positive")
    try:
        return _serve(args)
    except (BrokenPipeError, ConnectionError):
        return 0
    except Exception as error:  # noqa: BLE001 - subprocess reports actionable cause
        print(
            f"[isaac-teleop] bridge failed: {type(error).__name__}: {error}",
            file=sys.stderr,
            flush=True,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
