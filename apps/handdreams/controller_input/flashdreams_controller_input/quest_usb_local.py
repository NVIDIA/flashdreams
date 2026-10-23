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

"""Owned ADB reverses and local TURN relay for Quest USB streaming."""

from __future__ import annotations

import ipaddress
import os
import secrets
import shutil
import socket
import subprocess
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import BinaryIO

_BACKEND_PORT = 49100
"""CloudXR's local WebSocket backend, reached through its browser proxy."""

_STARTUP_TIMEOUT_S = 10.0
"""Bound local TURN startup independently of the browser's startup timeout."""


def cached_web_client_dir() -> Path:
    """Require the complete preinstalled WebXR client without downloading assets."""
    directory = (
        Path(
            os.environ.get("TELEOP_WEB_CLIENT_STATIC_DIR", "").strip()
            or "~/.cloudxr/static-client"
        )
        .expanduser()
        .resolve()
    )
    required = ("index.html", "bundle.js", "bundle.emulator.js")
    try:
        missing = [
            name
            for name in required
            if not (directory / name).is_file()
            or (directory / name).stat().st_size == 0
        ]
    except OSError as error:
        raise RuntimeError(
            f"Cannot inspect cached WebXR client at {directory}: {error}"
        ) from error
    if missing:
        raise RuntimeError(
            f"USB local mode needs a cached WebXR client at {directory}; missing or "
            f"empty: {', '.join(missing)}. Preinstall the Isaac Teleop WebXR client "
            "or set TELEOP_WEB_CLIENT_STATIC_DIR to its complete directory before "
            "starting offline."
        )
    return directory


class USBLocalTransport:
    """One worker's local TURN process and selected-device ADB reverse rules."""

    def __init__(self, adb: Callable[..., str], proxy_port: int) -> None:
        """Configure transport without starting processes or changing ADB rules.

        Args:
            adb: Bounded ADB command already bound to the selected USB device.
            proxy_port: Local CloudXR HTTPS proxy port.
        """
        if type(proxy_port) is not int or not 1 <= proxy_port <= 65535:
            raise ValueError("CloudXR proxy port must be an integer from 1 to 65535")
        if proxy_port == _BACKEND_PORT:
            raise ValueError("CloudXR proxy port conflicts with backend port 49100")
        self._adb = adb
        self._proxy_port = proxy_port
        self._turn_port = 0
        self._credential = secrets.token_hex(24)
        self._process: subprocess.Popen | None = None
        self._directory: tempfile.TemporaryDirectory | None = None
        self._log: BinaryIO | None = None
        self._reverses: list[str] = []
        self._stopped = False

    @property
    def turn_port(self) -> int:
        """Return the owned TURN TCP listener port after startup."""
        return self._turn_port

    @property
    def credential(self) -> str:
        """Return this run's TURN password for the ``cloudxr`` user."""
        return self._credential

    def start(self, check_alive: Callable[[], None]) -> None:
        """Start the TURN relay and reverse its endpoints after physical wake.

        Args:
            check_alive: Raise when the owning producer is cancelled or unhealthy.
        """
        self.check_alive()
        check_alive()
        if self._process is not None:
            return
        try:
            if not self._adb("get-devpath").startswith("usb:"):
                raise RuntimeError("USB local mode requires a USB-connected ADB device")
            check_alive()
            addresses = self._adb("shell", "ip", "-o", "-4", "addr", "show", "up")
            usable = False
            for line in addresses.splitlines():
                fields = line.split()
                if len(fields) < 4 or fields[2] != "inet":
                    continue
                try:
                    address = ipaddress.IPv4Interface(fields[3]).ip
                except ValueError:
                    continue
                if not (
                    address.is_loopback
                    or address.is_unspecified
                    or address.is_multicast
                ):
                    usable = True
                    break
            if not usable:
                raise RuntimeError(
                    "USB local mode requires an active nonloopback IPv4 address on "
                    "the Quest for WebRTC; connect it to a network (internet and "
                    "the host's LAN are not required)."
                )
            executable = shutil.which("turnserver")
            if executable is None:
                raise RuntimeError(
                    "USB local mode requires coturn; install turnserver on PATH"
                )
            check_alive()
            for _ in range(8):
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
                    listener.bind(("127.0.0.1", 0))
                    self._turn_port = listener.getsockname()[1]
                if self._turn_port not in (self._proxy_port, _BACKEND_PORT):
                    break
            else:
                raise RuntimeError("Could not allocate a distinct local TURN TCP port")
            self._directory = tempfile.TemporaryDirectory(
                prefix="flashdreams-quest-usb-"
            )
            directory = Path(self._directory.name)
            config = directory / "turnserver.conf"
            config.write_text(
                "\n".join(
                    (
                        f"listening-port={self._turn_port}",
                        "listening-ip=127.0.0.1",
                        "relay-ip=127.0.0.1",
                        "external-ip=127.0.0.1",
                        "lt-cred-mech",
                        f"user=cloudxr:{self._credential}",
                        "realm=cloudxr",
                        "fingerprint",
                        "allow-loopback-peers",
                        "no-cli",
                        "no-tls",
                        "no-dtls",
                        "no-udp",
                        "no-tcp-relay",
                        "simple-log",
                        "log-file=stdout",
                        f"pidfile={directory / 'turnserver.pid'}",
                        "",
                    )
                ),
                encoding="utf-8",
            )
            config.chmod(0o600)
            self._log = (directory / "turnserver.log").open("wb")
            (directory / "turnserver.log").chmod(0o600)
            check_alive()
            try:
                self._process = subprocess.Popen(
                    [executable, "-c", str(config)],
                    stdin=subprocess.DEVNULL,
                    stdout=self._log,
                    stderr=subprocess.STDOUT,
                )
            except OSError as error:
                raise RuntimeError(
                    self._diagnostic(f"cannot launch turnserver: {error}")
                ) from None
            deadline = time.monotonic() + _STARTUP_TIMEOUT_S
            while True:
                check_alive()
                self.check_alive()
                try:
                    with socket.create_connection(
                        ("127.0.0.1", self._turn_port), timeout=0.1
                    ):
                        pass
                    self.check_alive()
                    break
                except OSError:
                    if time.monotonic() >= deadline:
                        raise RuntimeError(
                            self._diagnostic("TURN TCP listener did not become ready")
                        ) from None
                    time.sleep(0.05)
            for port in (self._proxy_port, _BACKEND_PORT, self._turn_port):
                check_alive()
                self.check_alive()
                endpoint = f"tcp:{port}"
                self._adb("reverse", "--no-rebind", endpoint, endpoint)
                self._reverses.append(endpoint)
            check_alive()
            self.check_alive()
        except BaseException:
            self.stop()
            raise

    def _diagnostic(self, cause: str) -> str:
        details = ""
        if self._directory is not None:
            directory = Path(self._directory.name)
            try:
                with (directory / "turnserver.log").open("rb") as log:
                    log.seek(0, os.SEEK_END)
                    size = log.tell()
                    log.seek(0)
                    if size <= 8192:
                        output = (
                            log.read()
                            .decode("utf-8", errors="replace")
                            .replace(self._credential, "<redacted>")
                        )
                    else:
                        # Read across the truncation boundaries before redacting.
                        width = 4096 + len(self._credential)
                        head = (
                            log.read(width)
                            .decode("utf-8", errors="replace")
                            .replace(self._credential, "<redacted>")
                        )
                        log.seek(size - width)
                        tail = (
                            log.read(width)
                            .decode("utf-8", errors="replace")
                            .replace(self._credential, "<redacted>")
                        )
                        output = head[:4096] + "\n...\n" + tail[-4096:]
                config = (directory / "turnserver.conf").read_text(encoding="utf-8")
                details = f"\nTURN log:\n{output}\nTURN configuration:\n{config}"
            except OSError as error:
                details = f"; TURN diagnostic files could not be read ({type(error).__name__})"
        return f"USB local transport failed: {cause}{details}".replace(
            self._credential, "<redacted>"
        )

    def check_alive(self) -> None:
        """Raise if this transport was stopped or its owned TURN process exited."""
        if self._stopped:
            raise RuntimeError("USB local transport is stopped")
        if self._process is not None and (status := self._process.poll()) is not None:
            raise RuntimeError(
                self._diagnostic(f"turnserver exited with status {status}")
            )

    def stop(self) -> None:
        """Remove owned rules and reap the owned process, tolerating repeated calls."""
        self._stopped = True
        reverses, self._reverses = self._reverses, []
        for endpoint in reversed(reverses):
            try:
                rules = self._adb("reverse", "--list").splitlines()
                if any(line.split()[1:] == [endpoint, endpoint] for line in rules):
                    self._adb("reverse", "--remove", endpoint)
            except Exception as error:
                print(
                    f"[quest-usb] could not remove owned reverse {endpoint} ({type(error).__name__})",
                    flush=True,
                )
        process, self._process = self._process, None
        if process is not None:
            try:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=3)
            except OSError as error:
                print(
                    f"[quest-usb] could not stop owned turnserver ({type(error).__name__})",
                    flush=True,
                )
            except subprocess.TimeoutExpired:
                print(
                    "[quest-usb] owned turnserver did not exit after kill", flush=True
                )
        if self._log is not None:
            try:
                self._log.close()
            except OSError as error:
                print(
                    f"[quest-usb] could not close owned TURN log ({type(error).__name__})",
                    flush=True,
                )
            self._log = None
        if self._directory is not None:
            try:
                self._directory.cleanup()
            except OSError as error:
                print(
                    f"[quest-usb] could not remove owned TURN directory ({type(error).__name__})",
                    flush=True,
                )
            self._directory = None
