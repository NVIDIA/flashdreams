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

"""Private controller relay credentials shared by local and remote launch paths."""

from __future__ import annotations

import ipaddress
import json
import os
import secrets
import shlex
import socket
import ssl
import stat
import subprocess
import tempfile
from pathlib import Path

SSH_OPTIONS = [
    "-o",
    "StrictHostKeyChecking=yes",
    "-o",
    "UpdateHostKeys=no",
    "-o",
    "BatchMode=yes",
    "-o",
    "ConnectTimeout=10",
]
"""Reuse established SSH trust without changing host keys or prompting for login."""


def _state_root() -> Path:
    return Path.home() / ".config" / "flashdreams"


def _check_private(
    path: Path, *, directory: bool = False, public: bool = False
) -> None:
    info = path.lstat()
    expected = stat.S_ISDIR if directory else stat.S_ISREG
    if (
        not expected(info.st_mode)
        or info.st_uid != os.getuid()
        or info.st_mode & (0o022 if public else 0o077)
    ):
        raise ValueError(
            f"Unsafe ownership, type, or permissions: {path}. Existing state was preserved."
        )


def _private_directory(path: Path) -> None:
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    _check_private(path, directory=True, public=True)


def _existing(directory: Path, names: tuple[str, ...]) -> bool:
    if not directory.exists() and not directory.is_symlink():
        return False
    _check_private(directory, directory=True)
    if not all((directory / name).is_file() for name in names):
        raise ValueError(
            f"Incomplete credentials in {directory}; inspect them before retrying. Nothing was replaced."
        )
    for name in names:
        _check_private(directory / name, public=name == "server.crt")
    return True


def _publish(staging: Path, directory: Path, names: tuple[str, ...]) -> None:
    # Exclusive creation and hard links cannot replace an existing trust file.
    directory.mkdir(mode=0o700)
    for name in names:
        os.link(staging / name, directory / name)


def _write_private(path: Path, value: str) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        stream.write(value)


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(command, check=True, capture_output=True, text=True)
    except FileNotFoundError:
        raise ValueError(f"Required command is not installed: {command[0]}") from None


def resolve_server_address(host: str) -> str:
    """Resolve a concrete controller IPv4 interface without a wildcard bind."""
    try:
        address = ipaddress.IPv4Address(socket.gethostbyname(host))
    except (OSError, ValueError):
        raise ValueError(
            "--controller-host must resolve to a concrete IPv4 interface"
        ) from None
    if address.is_unspecified or address.is_multicast:
        raise ValueError(
            "--controller-host must select a concrete IPv4 interface, not a "
            "wildcard or multicast address; use 127.0.0.1 for local-only "
            "operation or a LAN address"
        )
    return str(address)


def check_certificate(certificate: Path, address: str) -> None:
    """Validate a trusted server certificate against its explicit identity."""
    ssl.create_default_context(cafile=str(certificate))
    try:
        ipaddress.ip_address(address)
    except ValueError:
        identity_option = "-verify_hostname"
    else:
        identity_option = "-verify_ip"
    try:
        _run(
            [
                "openssl",
                "verify",
                "-CAfile",
                str(certificate),
                "-purpose",
                "sslserver",
                identity_option,
                address,
                str(certificate),
            ]
        )
    except subprocess.CalledProcessError:
        raise ValueError(
            f"Certificate {certificate} is expired or does not cover {address}; existing trust was preserved."
        ) from None


def read_token(directory: Path) -> str:
    """Read a validated token without exposing it in diagnostics."""
    value = (directory / "token").read_text(encoding="utf-8").strip()
    if (
        not 32 <= len(value) <= 256
        or not value.isascii()
        or any(c.isspace() for c in value)
    ):
        raise ValueError(
            f"Invalid controller token in {directory}; nothing was replaced."
        )
    return value


def server_credentials(address: str) -> Path:
    """Create or validate the selected server identity without replacing trust."""
    address = resolve_server_address(address)
    root = _state_root()
    _private_directory(root)
    # Local runs must not replace the LAN identity trusted by remote clients.
    directory = root / (
        f"quest-relay-{address}"
        if ipaddress.ip_address(address).is_loopback
        else "quest-relay"
    )
    names = ("server.crt", "server.key", "token")
    if not _existing(directory, names):
        with tempfile.TemporaryDirectory(prefix=".quest-relay-", dir=root) as temporary:
            staging = Path(temporary)
            _run(
                "openssl req -x509 -newkey rsa:3072 -sha256 -nodes -days 365".split()
                + [
                    "-keyout",
                    str(staging / "server.key"),
                    "-out",
                    str(staging / "server.crt"),
                    "-subj",
                    f"/CN={address}",
                    "-addext",
                    f"subjectAltName=IP:{address}",
                ]
            )
            for name in names[:2]:
                (staging / name).chmod(0o600)
            _write_private(staging / "token", secrets.token_hex(32))
            _publish(staging, directory, names)
    check_certificate(directory / "server.crt", address)
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(
        str(directory / "server.crt"),
        str(directory / "server.key"),
        password=lambda: "",
    )
    read_token(directory)
    return directory


def client_credentials(target: str, address: str) -> Path:
    """Cache the server's public certificate and token through verified SSH."""
    root = _state_root()
    _private_directory(root)
    clients = root / "quest-relay-clients"
    _private_directory(clients)
    directory = clients / target
    names = ("server.crt", "token")
    if not _existing(directory, names):
        script = (
            "import json; from pathlib import Path; "
            "p=Path.home()/'.config/flashdreams/quest-relay'; "
            "print(json.dumps({n:(p/n).read_text() for n in ('server.crt','token')}))"
        )
        try:
            response = _run(
                [
                    "ssh",
                    *SSH_OPTIONS,
                    "--",
                    target,
                    "python3 -c " + shlex.quote(script),
                ]
            )
        except subprocess.CalledProcessError:
            raise ValueError(
                f"Could not obtain controller credentials through verified SSH to {target}. "
                "Start the server, verify its SSH host identity and existing login access, then retry. "
                "No SSH keys or trust settings were changed."
            ) from None
        try:
            values = json.loads(response.stdout)
            if set(values) != set(names) or not all(
                isinstance(values[name], str) for name in names
            ):
                raise ValueError
        except (ValueError, TypeError):
            raise ValueError(
                "SSH returned an invalid controller credential bundle"
            ) from None
        with tempfile.TemporaryDirectory(
            prefix=".quest-relay-", dir=clients
        ) as temporary:
            staging = Path(temporary)
            for name in names:
                _write_private(staging / name, values[name])
            check_certificate(staging / "server.crt", address)
            read_token(staging)
            _publish(staging, directory, names)
    check_certificate(directory / "server.crt", address)
    read_token(directory)
    return directory
