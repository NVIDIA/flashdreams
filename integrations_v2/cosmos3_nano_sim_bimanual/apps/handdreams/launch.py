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

"""Unified HandDreams server and remote Quest client launchers."""

from __future__ import annotations

import argparse
import filecmp
import importlib
import ipaddress
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
from pathlib import Path
from urllib.parse import urlencode

REPOSITORY = Path(__file__).resolve().parents[4]
MODEL_PACKAGE = "flashdreams-cosmos3-nano-sim-bimanual"


def _controller_module(name: str):
    path = REPOSITORY / "apps/handdreams/controller_input"
    # Source checkout support without syncing the application's GPU dependencies.
    sys.path.insert(0, str(path))
    try:
        return importlib.import_module(f"flashdreams_controller_input.{name}")
    finally:
        sys.path.pop(0)


_credentials = _controller_module("credentials")


def _ssh_target(value: str) -> tuple[str, str]:
    user, separator, host = value.partition("@")
    if not separator or not re.fullmatch(r"[a-zA-Z0-9_][a-zA-Z0-9_.-]*", user):
        raise argparse.ArgumentTypeError(
            "--server must be user@host with an existing trusted SSH identity"
        )
    host = host.removeprefix("[").removesuffix("]")
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9.-]*", host):
            raise argparse.ArgumentTypeError(
                "--server host must be an IP address or DNS name"
            ) from None
    else:
        if address.version != 4:
            raise argparse.ArgumentTypeError(
                "The controller transport requires an IPv4 server address"
            )
    return f"{user}@{host}", host


def _client_web_assets(target: str) -> Path:
    """Fill missing USB-local assets from the trusted server before launching VR."""
    directory = (
        Path(
            os.environ.get("TELEOP_WEB_CLIENT_STATIC_DIR", "").strip()
            or "~/.cloudxr/static-client"
        )
        .expanduser()
        .resolve()
    )
    required = ("index.html", "bundle.js", "bundle.emulator.js")
    missing = [
        name
        for name in required
        if not (directory / name).is_file() or (directory / name).stat().st_size == 0
    ]
    if not missing:
        return directory
    if shutil.which("scp") is None:
        raise ValueError("USB-local asset preparation requires scp (OpenSSH client)")
    directory.mkdir(parents=True, exist_ok=True)
    print(
        f"Preparing USB-local web client from {target} into {directory} ...", flush=True
    )
    # Stage the complete bundle to detect mismatched partial caches before publication.
    with tempfile.TemporaryDirectory(prefix=".web-client-", dir=directory) as temporary:
        staging = Path(temporary)
        try:
            subprocess.run(
                [
                    "scp",
                    *_credentials.SSH_OPTIONS,
                    "--",
                    *(f"{target}:.cloudxr/static-client/{name}" for name in required),
                    str(staging),
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=300,
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
            raise ValueError(
                f"Could not prepare the USB-local web client through verified SSH to {target}. "
                "Check SSH access and the server's ~/.cloudxr/static-client cache "
                "(index.html, bundle.js, bundle.emulator.js), or set "
                "TELEOP_WEB_CLIENT_STATIC_DIR to a complete matching Isaac Teleop 1.4 cache. "
                "Existing local assets were preserved; retry after fixing the source."
            ) from error
        if any(
            not (staging / name).is_file() or (staging / name).stat().st_size == 0
            for name in required
        ):
            raise ValueError(
                "The server returned incomplete USB-local web client assets; "
                "existing local assets were preserved. Check its ~/.cloudxr/static-client cache."
            )
        missing = []
        for name in required:
            destination = directory / name
            if destination.is_symlink() or (
                destination.exists() and not destination.is_file()
            ):
                raise ValueError(
                    f"USB-local cache entry is not a regular file: {destination}"
                )
            if destination.is_file() and destination.stat().st_size > 0:
                if not filecmp.cmp(destination, staging / name, shallow=False):
                    raise ValueError(
                        f"Partial USB-local cache at {directory} does not match {target}. "
                        "Existing assets were preserved; set TELEOP_WEB_CLIENT_STATIC_DIR "
                        "to an empty directory or a complete matching cache."
                    )
            else:
                missing.append(name)
        for name in missing:
            destination = directory / name
            if destination.exists():
                if (
                    destination.is_symlink()
                    or not destination.is_file()
                    or destination.stat().st_size != 0
                ):
                    raise ValueError(
                        f"USB-local cache changed during preparation: {destination}; retry"
                    )
                os.replace(staging / name, destination)
            else:
                os.link(staging / name, destination)
    return directory


def _address(value: str) -> str:
    try:
        address = ipaddress.ip_address(value)
    except ValueError:
        raise argparse.ArgumentTypeError("Use an IPv4 address") from None
    if address.version != 4 or address.is_unspecified or address.is_multicast:
        raise argparse.ArgumentTypeError(
            "Use a LAN or loopback IPv4 address, not a wildcard or multicast address"
        )
    return str(address)


def _controller_address(host: str) -> str:
    """Resolve one IPv4 identity for certificate verification and relay dialing."""
    try:
        ipaddress.ip_address(host)
    except ValueError:
        try:
            addresses = {
                result[4][0]
                for result in socket.getaddrinfo(
                    host, None, family=socket.AF_INET, type=socket.SOCK_STREAM
                )
            }
        except socket.gaierror:
            raise ValueError(
                f"Cannot resolve {host!r} to IPv4; use the server's LAN IPv4 address. "
                "SSH-only aliases are not controller addresses."
            ) from None
        if len(addresses) != 1:
            raise ValueError(
                f"{host!r} must resolve to exactly one IPv4 address; "
                "use the server's LAN IPv4 address explicitly."
            )
        host = str(addresses.pop())
    try:
        address = _address(host)
    except argparse.ArgumentTypeError as error:
        raise ValueError(str(error)) from None
    if ipaddress.ip_address(address).is_loopback:
        raise ValueError("Use the remote server's LAN IPv4 address, not loopback")
    return address


def _port(value: str) -> int:
    port = int(value)
    if not 1 <= port <= 65535:
        raise argparse.ArgumentTypeError("Port must be between 1 and 65535")
    return port


def _setup_module():
    return _controller_module("isaac_teleop_setup")


def _add_scene_source_arguments(parser: argparse.ArgumentParser) -> None:
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--sample-dir", type=Path, help="External scenario catalog directory"
    )
    source.add_argument("--scene-path", type=Path, help="External scene directory")


def _validate_scene_source(args: argparse.Namespace, extra: list[str]) -> list[str]:
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    _add_scene_source_arguments(parser)
    supplied = []
    for option in ("sample_dir", "scene_path"):
        value = getattr(args, option)
        if value is not None:
            supplied.extend(["--" + option.replace("_", "-"), str(value)])
    source, remaining = parser.parse_known_args([*supplied, *extra])
    path = source.scene_path or source.sample_dir
    if path is None:
        # The adapter prepares its default catalog with the user's HF credentials.
        return remaining
    path = path.expanduser().resolve()
    if not path.is_dir():
        raise ValueError(f"Scene directory does not exist: {path}")
    scenes = (
        (path,)
        if source.scene_path
        else tuple(item.parent for item in path.rglob("scene.json"))
    )
    if not scenes or any(
        not (scene / filename).is_file()
        for scene in scenes
        for filename in ("scene.json", "first_frame.jpg")
    ):
        raise ValueError(
            f"Scene source requires scene.json / first_frame.jpg pairs: {path}"
        )
    args.scene_path = path if source.scene_path is not None else None
    args.sample_dir = path if source.sample_dir is not None else None
    return remaining


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser(
        "setup-isaac-teleop", help="Install the hash-locked Isaac Teleop sidecar"
    )
    for name in ("server", "remote-vr"):
        command = commands.add_parser(name)
        if name != "remote-vr":
            command.add_argument(
                "--address",
                type=_address,
                default="127.0.0.1",
                help="Controller address: loopback for local use (default); set a LAN IPv4 address for remote clients",
            )
            command.add_argument(
                "--port", type=_port, default=8081, help="WebRTC operator viewer port"
            )
            command.add_argument(
                "--mode",
                choices=("webrtc", "native-window"),
                default="webrtc",
                help="Operator display: browser video or a native window on the model host (default: %(default)s)",
            )
            command.add_argument(
                "--gpu", help="CUDA_VISIBLE_DEVICES for this process only"
            )
            command.add_argument(
                "--checkpoint", help="Optional existing export or checkpoint alias"
            )
            _add_scene_source_arguments(command)
            command.add_argument(
                "--replay",
                action="store_true",
                help="Replay only Fold the T-shirt using recorded actions without Isaac Teleop or controllers",
            )
            command.add_argument(
                "--prewarm-all-scenes",
                action="store_true",
                help="Compatibility option; every available scene is prepared during startup",
            )
        command.add_argument("--controller-port", type=_port, default=8445)
        command.add_argument(
            "--isaac-teleop-python",
            default=str(
                _controller_module("controller_types").DEFAULT_ISAAC_TELEOP_DIRECTORY
                / "bin/python"
            ),
            help="Isolated interpreter created by setup-isaac-teleop",
        )
        command.add_argument(
            "--accept-eula",
            action="store_true",
            help=(
                "Accept the NVIDIA CloudXR EULA for Isaac Teleop 1.4.142: "
                "https://github.com/NVIDIA/IsaacCapture/blob/main/deps/cloudxr/CLOUDXR_LICENSE"
            ),
        )
        command.add_argument("--cloudxr-port", type=_port, default=48322)
        command.add_argument(
            "--quest-device",
            metavar="SERIAL",
            help="Select a USB Quest; otherwise detect the single authorized headset at startup",
        )
        command.add_argument(
            "--require-physical-wake",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Wait for a physical headset wake before connecting (default: enabled)",
        )
        command.add_argument(
            "--accept-server-certificate",
            action="store_true",
            help="Accept certificate errors only for this host's CloudXR origin in the managed Quest tab",
        )
        command.add_argument(
            "--keep-browser-open-on-error",
            action="store_true",
            help="Keep the managed Quest tab and USB helpers after a connection failure until retry or exit",
        )
        if name == "remote-vr":
            command.add_argument(
                "--server",
                type=_ssh_target,
                required=True,
                metavar="USER@HOST",
                help="SSH user and server IPv4 or hostname resolving to one IPv4 address",
            )
    return parser


def _command(
    args: argparse.Namespace, extra: list[str], credentials: Path | None
) -> list[str]:
    command: list[str]
    replay = getattr(args, "replay", False)
    if not replay and not args.accept_eula:
        raise ValueError("CloudXR startup requires --accept-eula.")
    if not replay and credentials is None:
        raise ValueError("Live control requires prepared transport credentials.")
    if args.command == "remote-vr":
        assert credentials is not None
        package = REPOSITORY / "apps/handdreams/controller_input"
        command = "uv run --no-project --python 3.12 --with-editable".split() + [
            f"{package}[webxr]",
            "flashdreams-quest-relay",
            "--server",
            args.server[1],
            "--port",
            str(args.controller_port),
            "--ca-cert",
            str(credentials / "server.crt"),
            "--cloudxr-port",
            str(args.cloudxr_port),
            "--isaac-teleop-accept-eula",
            "--isaac-teleop-python",
            args.isaac_teleop_python,
        ]
        if args.quest_device:
            command.extend(["--quest-device", args.quest_device])
        command.append(
            "--require-physical-wake"
            if args.require_physical_wake
            else "--no-require-physical-wake"
        )
        if args.accept_server_certificate:
            command.append("--accept-server-certificate")
        if args.keep_browser_open_on_error:
            command.append("--keep-browser-open-on-error")
        return command + extra
    application = "handdreams-replay" if replay else "handdreams"
    command = [
        "uv",
        "run",
        "--package",
        MODEL_PACKAGE,
        "flashdreams-run-v2",
        application,
    ]
    command.extend(["--mode", args.mode])
    if not replay:
        command.extend(["--fps", "36"])
    if args.mode == "webrtc":
        viewer_host = (
            args.address
            if ipaddress.ip_address(args.address).is_loopback
            else "0.0.0.0"
        )
        command.extend(["--host", viewer_host, "--port", str(args.port)])
    if replay:
        command.append("--")
        if args.checkpoint:
            command.extend(["--checkpoint", args.checkpoint])
        return command + extra
    assert credentials is not None
    command.extend(
        [
            "--controller-host",
            args.address,
            "--controller-port",
            str(args.controller_port),
            "--controller-cert",
            str(credentials / "server.crt"),
            "--controller-key",
            str(credentials / "server.key"),
            "--isaac-teleop-accept-eula",
            "--isaac-teleop-python",
            args.isaac_teleop_python,
            "--quest-cloudxr-port",
            str(args.cloudxr_port),
            "--quest-require-physical-wake"
            if args.require_physical_wake
            else "--no-quest-require-physical-wake",
        ]
    )
    if args.quest_device:
        command.extend(["--quest-device", args.quest_device])
    if args.accept_server_certificate:
        command.append("--quest-accept-server-certificate")
    if args.keep_browser_open_on_error:
        command.append("--quest-keep-browser-open-on-error")
    command.append("--")
    command.extend(["--isaac-teleop-python", args.isaac_teleop_python])
    if args.checkpoint:
        command.extend(["--checkpoint", args.checkpoint])
    if args.sample_dir is not None:
        command.extend(["--sample-dir", str(args.sample_dir)])
    if args.scene_path is not None:
        command.extend(["--scene-path", str(args.scene_path)])
    if args.prewarm_all_scenes:
        command.append("--prewarm-all-scenes")
    return command + extra


def main(argv: list[str] | None = None) -> None:
    """Prepare private transport state and replace this process with the V2 command."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments[:1] == ["setup-isaac-teleop"]:
        _setup_module().main(arguments[1:])
        return
    separator = arguments.index("--") if "--" in arguments else len(arguments)
    parser = _parser()
    args = parser.parse_args(arguments[:separator])
    extra = arguments[separator + 1 :]
    replay = getattr(args, "replay", False)
    if replay:
        incompatible = ("--sample-dir", "--scene-path")
        if (
            args.sample_dir is not None
            or args.scene_path is not None
            or any(option.partition("=")[0] in incompatible for option in extra)
        ):
            parser.error(
                "--replay uses the public dataset scenarios; custom --sample-dir "
                "and --scene-path do not provide matching recorded actions"
            )
        extra = [option for option in extra if option != "--prewarm-all-scenes"]
    if not replay and not args.accept_eula:
        parser.error(
            "Review the NVIDIA CloudXR EULA for Isaac Teleop 1.4.142 at "
            "https://github.com/NVIDIA/IsaacCapture/blob/main/deps/cloudxr/CLOUDXR_LICENSE"
            "; then pass --accept-eula to accept it"
        )
    if not (REPOSITORY / "pyproject.toml").is_file():
        parser.error("Run this launcher from a FlashDreams source checkout")
    for executable in (
        "uv",
        *(("openssl",) if not replay else ()),
        *(("ssh",) if args.command == "remote-vr" else ()),
    ):
        if shutil.which(executable) is None:
            parser.error(f"Required command is not installed: {executable}")
    try:
        if args.command == "server" and not replay:
            extra = _validate_scene_source(args, extra)
        if args.command == "remote-vr":
            target, host = args.server
            args.server = (target, _controller_address(host))
        environment = os.environ.copy()
        credentials = None
        if not replay:
            credentials = (
                _credentials.server_credentials(args.address)
                if args.command == "server"
                else _credentials.client_credentials(*args.server)
            )
            environment["FLASHDREAMS_CONTROLLER_TOKEN"] = _credentials.read_token(
                credentials
            )
            if args.command == "remote-vr":
                environment["TELEOP_WEB_CLIENT_STATIC_DIR"] = str(
                    _client_web_assets(args.server[0])
                )
            else:
                environment["TELEOP_WEB_CLIENT_STATIC_DIR"] = str(
                    _setup_module().prepare_web_client(python=args.isaac_teleop_python)
                )
        if getattr(args, "gpu", None):
            environment["CUDA_VISIBLE_DEVICES"] = args.gpu
        if args.command == "server" and not replay:
            environment.update(
                TELEOP_PROXY_HOST=args.address,
                TELEOP_STREAM_SERVER_IP=args.address,
                TELEOP_STREAM_PORT=str(args.cloudxr_port),
                PROXY_PORT=str(args.cloudxr_port),
            )
        command = _command(args, extra, credentials)
        if args.command == "server":
            if args.mode == "webrtc":
                print(
                    f"Operator viewer: http://{args.address}:{args.port}/", flush=True
                )
            else:
                print("Operator viewer: native window on the model host", flush=True)
            if not replay:
                print(
                    f"CloudXR certificate: https://{args.address}:{args.cloudxr_port}/",
                    flush=True,
                )
                query = urlencode(
                    {
                        "serverIP": args.address,
                        "port": args.cloudxr_port,
                        "oobEnable": "1",
                    }
                )
                print(
                    "Isaac Teleop / IWER: "
                    f"https://{args.address}:{args.cloudxr_port}/client/?{query}",
                    flush=True,
                )
        os.chdir(REPOSITORY)
        os.execvpe(command[0], command, environment)
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
