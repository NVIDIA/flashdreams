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

"""Hash-verified installation of the isolated Isaac Teleop environment."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from flashdreams_controller_input.controller_types import (
    CLOUDXR_EULA_URL,
    DEFAULT_ISAAC_TELEOP_DIRECTORY,
    ISAAC_TELEOP_REQUIREMENT,
)

REQUIREMENTS = Path(__file__).resolve().parents[1] / "isaac_teleop"
MARKER = ".flashdreams-isaac-teleop.json"


def _run(command: list[str], *, timeout: float = 120.0) -> str:
    environment = os.environ.copy()
    # Do not let the main application's Python or pip configuration leak in.
    for name in tuple(environment):
        if name.startswith(("PYTHON", "PIP_")):
            environment.pop(name)
    environment["PIP_CONFIG_FILE"] = os.devnull
    try:
        result = subprocess.run(
            command,
            check=True,
            text=True,
            capture_output=True,
            env=environment,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as error:
        raise ValueError(
            f"Command timed out after {timeout:g}s: {shlex.join(command)}"
        ) from error
    except subprocess.CalledProcessError as error:
        raise ValueError(
            f"Command failed: {shlex.join(command)}\n{error.stdout}{error.stderr}"
        ) from error
    return result.stdout


def _platform(python: str | None) -> str:
    """Validate the host and, when supplied, its native CPython interpreter."""
    if platform.system() != "Linux" or platform.machine() not in ("x86_64", "aarch64"):
        raise ValueError("Isaac Teleop setup requires Linux x86_64 or aarch64.")
    if python is None:
        return platform.machine()
    info = json.loads(
        _run(
            [
                python,
                "-I",
                "-c",
                "import json,platform,sys; print(json.dumps([list(sys.version_info[:2]), "
                "platform.system(), platform.machine(), platform.python_implementation(), "
                "list(platform.libc_ver())]))",
            ]
        )
    )
    if info[:4] != [[3, 12], "Linux", platform.machine(), "CPython"]:
        raise ValueError(
            "Select a native Linux CPython 3.12 interpreter with --python python3.12."
        )
    libc, version = info[4]
    if libc != "glibc" or tuple(int(n) for n in version.split(".")[:2]) < (2, 35):
        raise ValueError("Isaac Teleop 1.4.142 wheels require glibc 2.35 or newer.")
    return platform.machine()


def _versions(requirements: Path) -> dict[str, str]:
    return {
        re.sub(r"[-_.]+", "-", name).lower(): version
        for name, version in re.findall(
            r"^([A-Za-z0-9_.-]+)(?:\[[^]]+\])?==([^\s;]+)",
            requirements.read_text(),
            re.MULTILINE,
        )
    }


def _validate(directory: Path, requirements: Path) -> None:
    python = str(directory / "bin/python")
    _platform(python)
    installed = json.loads(_run([python, "-I", "-m", "pip", "list", "--format=json"]))
    actual = {
        re.sub(r"[-_.]+", "-", item["name"]).lower(): item["version"]
        for item in installed
        if item["name"].lower() != "pip"
    }
    if actual != _versions(requirements):
        raise ValueError("Installed packages do not match the reviewed sidecar lock.")
    print("Checking sidecar dependencies...", flush=True)
    _run([python, "-I", "-m", "pip", "check"])
    print("Running Isaac Teleop doctor...", flush=True)
    # Script mode exposes only the bridge's local helpers and the sidecar's deps.
    print(
        _run(
            [python, str(Path(__file__).with_name("isaac_teleop_bridge.py")), "doctor"]
        ),
        end="",
    )


def prepare_web_client(*, python: str) -> Path:
    """Cache the SDK web client during launch setup, after EULA acceptance."""
    from flashdreams_controller_input.quest_usb_local import cached_web_client_dir

    try:
        return cached_web_client_dir()
    except RuntimeError:
        pass
    print("Preparing the local Isaac Teleop Web Client...", flush=True)
    try:
        # The pinned wheel's IsaacTeleop/v1.4.142 asset URL is not published.
        _run(
            [
                os.path.expanduser(python.strip()),
                "-I",
                "-c",
                (
                    "from isaacteleop.cloudxr import oob_teleop_env as client; "
                    "client.default_web_client_origin = lambda: "
                    "'https://nvidia.github.io/IsaacCapture/client/release-1.4.x/'; "
                    "client.require_web_client_static_dir()"
                ),
            ],
            timeout=400.0,
        )
        # The SDK treats the emulator bundle as optional; IWER requires it.
        return cached_web_client_dir()
    except (OSError, RuntimeError, ValueError) as error:
        raise ValueError(
            "Could not prepare the local Isaac Teleop Web Client. "
            "Connect to the internet once and retry, or set "
            "TELEOP_WEB_CLIENT_STATIC_DIR to a complete cached client "
            "(index.html, bundle.js, bundle.emulator.js). "
            f"Existing assets were preserved. {error}"
        ) from error


def setup(*, directory: Path, python: str) -> None:
    """Validate and publish an environment without replacing existing state."""
    machine = _platform(None)
    requirements = REQUIREMENTS / f"requirements-linux-{machine}-py312.txt"
    expected = {
        "requirements_sha256": hashlib.sha256(requirements.read_bytes()).hexdigest(),
        "machine": machine,
        "python": "3.12",
        "package": ISAAC_TELEOP_REQUIREMENT,
    }
    directory = directory.expanduser().absolute()
    recovery = (
        f"Existing environment at {directory} was preserved. Inspect it and move it "
        "aside before retrying, or select another --directory and pass its bin/python "
        "to the launcher with --isaac-teleop-python."
    )
    if directory.exists() or directory.is_symlink():
        try:
            if json.loads((directory / MARKER).read_text()) != expected:
                raise ValueError("The setup record does not match the selected lock.")
            _validate(directory, requirements)
        except (OSError, ValueError) as error:
            raise ValueError(f"{error}\n{recovery}") from error
        print(f"Isaac Teleop environment already matches: {directory}")
        return

    _platform(python)
    directory.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{directory.name.lstrip('.')}-", dir=directory.parent)
    )
    published = False
    try:
        try:
            print(f"Creating isolated environment in {staging}...", flush=True)
            _run([python, "-I", "-m", "venv", str(staging)])
        except ValueError as error:
            raise ValueError(
                f"{error}\nInstall the venv/ensurepip support for Python 3.12 "
                "(python3.12-venv on Debian/Ubuntu), then retry."
            ) from error
        print(f"Installing hash-locked wheels from {requirements.name}...", flush=True)
        _run(
            [
                str(staging / "bin/python"),
                "-I",
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                "--no-input",
                "--require-hashes",
                "--only-binary=:all:",
                "--index-url",
                "https://pypi.org/simple",
                "-r",
                str(requirements),
            ],
            timeout=1800.0,
        )
        _validate(staging, requirements)
        (staging / MARKER).write_text(json.dumps(expected, indent=2) + "\n")
        # Venv scripts contain absolute paths. Publish a link so they stay valid;
        # exclusive symlink creation also preserves a concurrently created env.
        directory.symlink_to(staging.name, target_is_directory=True)
        published = True
    except (OSError, ValueError) as error:
        raise ValueError(
            f"Isaac Teleop setup failed: {error}\nNo existing environment was replaced. "
            "Resolve the error above, then rerun "
            "flashdreams-controller-input setup isaac-teleop."
        ) from error
    finally:
        if not published:
            # A signal can arrive after symlink creation but before assignment.
            try:
                published = directory.readlink() == Path(staging.name)
            except OSError:
                pass
            if not published:
                shutil.rmtree(staging)
    print(f"Isaac Teleop environment ready: {directory}")
    print(
        "Launch with --accept-eula to explicitly accept the CloudXR terms for that run."
    )


def main(argv: list[str] | None = None) -> None:
    """Display package terms and obtain acceptance before installing the sidecar."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--python",
        default="python3.12",
        help="Native CPython 3.12 interpreter for creating a new environment",
    )
    parser.add_argument(
        "--directory",
        type=Path,
        default=DEFAULT_ISAAC_TELEOP_DIRECTORY,
        help=f"Sidecar environment (default: {DEFAULT_ISAAC_TELEOP_DIRECTORY})",
    )
    parser.add_argument(
        "--accept-eula",
        action="store_true",
        help=f"Explicitly accept the CloudXR EULA for {ISAAC_TELEOP_REQUIREMENT}: {CLOUDXR_EULA_URL}",
    )
    args = parser.parse_args(argv)
    print(f"Public PyPI package: {ISAAC_TELEOP_REQUIREMENT}", flush=True)
    print("Package: https://pypi.org/project/isaacteleop/1.4.142/", flush=True)
    print(
        f"NVIDIA CloudXR EULA (applicable to Isaac Teleop 1.4.142): {CLOUDXR_EULA_URL}",
        flush=True,
    )
    print(
        "Setup acceptance does not enable CloudXR startup; each launch requires explicit acceptance.",
        flush=True,
    )
    if not args.accept_eula:
        if not sys.stdin.isatty():
            parser.error(
                "Review the terms above and pass --accept-eula to accept them explicitly."
            )
        try:
            answer = input("Do you accept these CloudXR terms? Type yes to continue: ")
        except (EOFError, KeyboardInterrupt):
            parser.exit(1, "\nSetup cancelled; no environment was created.\n")
        if answer.strip().lower() != "yes":
            parser.exit(1, "Setup declined; no environment was created.\n")
    try:
        setup(directory=args.directory, python=args.python)
    except KeyboardInterrupt:
        parser.exit(
            130,
            "\nSetup interrupted. Completed environments were preserved; "
            "rerun flashdreams-controller-input setup isaac-teleop "
            "to validate or retry.\n",
        )
    except (OSError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
