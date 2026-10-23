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

"""Isolated Isaac Teleop controller source for FlashDreams."""

from __future__ import annotations

import math
import os
import shlex
import shutil
import signal
import subprocess
import threading
import time
from collections.abc import Callable
from contextlib import suppress
from ipaddress import IPv4Address
from pathlib import Path

from flashdreams_controller_input.controller_server import (
    ControllerServer,
)
from flashdreams_controller_input.controller_types import (
    CLOUDXR_EULA_HELP,
    DEFAULT_ISAAC_TELEOP_DIRECTORY,
    ISAAC_TELEOP_BRIDGE_PROTOCOL_VERSION,
    Observation,
)
from flashdreams_controller_input.frames import SOURCE_CLIENT_CAPTURE_NS_KEY
from flashdreams_controller_input.pose import (
    DEFAULT_GRIP_TO_WRIST_OFFSET_M,
    OPENXR_STAGE_GRIP_POSE_CONVENTION,
    WRIST_PROXY_POSE_CONVENTION,
)
from flashdreams_controller_input.quest_source import FrameStore

ISAAC_TELEOP_SOURCE = "isaac-teleop"
"""Input-source name selecting NVIDIA Isaac Teleop and CloudXR."""

DEFAULT_ISAAC_TELEOP_PYTHON = str(DEFAULT_ISAAC_TELEOP_DIRECTORY / "bin/python")
"""User-local interpreter containing the separately installed SDK."""

# The pinned launcher may spend 5 s joining its WSS proxy, then 10 s each on
# runtime SIGTERM and SIGKILL escalation. Let that owner finish before killing
# the bridge, whose child runtime deliberately lives in a separate process group.
_STOP_GRACE_S = 30.0
_ISAAC_TELEOP_STAGE_HELP = {
    "": "the Isaac Teleop sidecar has not connected",
    "connected": "the sidecar is starting CloudXR/OpenXR",
    "runtime": (
        "CloudXR is waiting; open the Isaac Teleop Web Client link shown by the demo"
    ),
    "session": (
        "the host OpenXR session is ready; waiting for tracked controllers "
        "from the WebXR client"
    ),
    "failed": "the Isaac Teleop sidecar failed",
}


def sidecar_recovery(executable: str) -> str:
    """Return setup and diagnostic commands for the selected SDK interpreter."""
    doctor = shlex.join(
        [
            os.path.expanduser(str(executable)),
            "-E",
            "-s",
            str(Path(__file__).with_name("isaac_teleop_bridge.py")),
            "doctor",
        ]
    )
    return (
        "Run `flashdreams-controller-input setup isaac-teleop` to prepare "
        "the isolated environment, then retry. For an existing environment run "
        f"`{doctor}` and inspect the sidecar output above. "
        "For a custom environment, pass its Python with --isaac-teleop-python."
    )


def isolated_python_environment() -> dict[str, str]:
    """Keep the selected interpreter independent of parent Python configuration."""
    environment = {
        name: value
        for name, value in os.environ.items()
        if not name.startswith("PYTHON")
    }
    environment["PYTHONNOUSERSITE"] = "1"
    return environment


def resolve_python(executable: str) -> str:
    """Resolve an isolated interpreter without following its venv symlink."""
    value = os.path.expanduser(str(executable))
    if os.sep in value:
        candidate = Path(value)
        if not candidate.is_absolute():
            candidate = Path.cwd() / candidate
        if candidate.is_file():
            # Invoking the venv path is what makes Python discover pyvenv.cfg.
            return str(candidate.absolute())
    else:
        resolved = shutil.which(value)
        if resolved is not None:
            return resolved
    raise RuntimeError(
        f"Isaac Teleop Python was not found at {executable!r}. "
        + sidecar_recovery(executable)
    )


class IsaacTeleopSource:
    """Receive absolute Quest poses from an isolated Isaac Teleop process.

    The sidecar owns CloudXR and OpenXR in a separate environment. Its output is
    converted to the canonical wrist-proxy schema. Participant calibration and
    command shaping belong to the application's controller pipeline.
    """

    name = ISAAC_TELEOP_SOURCE

    def __init__(
        self,
        *,
        python_executable: str = DEFAULT_ISAAC_TELEOP_PYTHON,
        port: int = 0,
        rate_hz: float = 36.0,
        tracking_rate_hz: float = 72.0,
        wrist_offset_m: float = DEFAULT_GRIP_TO_WRIST_OFFSET_M,
        startup_timeout_s: float | None = 300.0,
        host: str = "127.0.0.1",
        install_dir: str = "~/.cloudxr",
        env_config: str | None = None,
        device_profile: str = "Quest3",
        accept_eula: bool = False,
        host_client: bool = False,
        offline_web_client: bool = False,
        oob_hub: bool = False,
        launch_runtime: bool = True,
        verbose: bool = False,
        wait_for_controllers: bool = True,
        on_message: Callable[[str | bytes], None] | None = None,
        on_disconnect: Callable[[], None] | None = None,
        cloudxr_host: str | None = None,
        cloudxr_port: int = 48322,
        cloudxr_proxy_bind: str | None = None,
    ) -> None:
        """Configure the sidecar without importing Isaac Teleop.

        Args:
            python_executable: Interpreter containing ``isaacteleop``.
            port: Loopback receiver port; ``0`` selects an unused port.
            rate_hz: Maximum controller rate returned by :meth:`poll`.
            tracking_rate_hz: OpenXR sampling rate inside the sidecar.
            wrist_offset_m: Controller-local positive-Z grip-to-wrist offset.
            startup_timeout_s: Time to wait for the readiness selected by
                ``wait_for_controllers``, or ``None`` to wait until stopped or failed.
            host: Receiver interface passed to the sidecar.
            install_dir: CloudXR runtime state and cache directory.
            env_config: Optional CloudXR ``KEY=value`` configuration file.
            device_profile: CloudXR device profile.
            accept_eula: Accept the CloudXR EULA non-interactively.
            host_client: Serve the WebXR client from the CloudXR proxy.
            offline_web_client: Require cached web assets before launch; never
                start the local web client with missing assets.
            oob_hub: Enable the browser status hub without local ADB automation.
            launch_runtime: Start CloudXR in the sidecar; false uses an existing
                runtime configured in the process environment.
            verbose: Enable verbose Isaac Teleop session logging.
            wait_for_controllers: Await a tracked pair before returning from
                startup. False awaits the first compatible fresh frame instead,
                allowing setup UI to display partial observations.
            on_message: Receive original bridge records after local validation,
                including handshake and status messages. Runs on the receiver
                thread; finish promptly and do not call :meth:`stop`.
            on_disconnect: Receive a connection-close notification on the
                receiver thread, subject to the same callback restrictions.
            cloudxr_host: Override proxy and stream advertising in the child
                environment only. ``None`` preserves inherited SDK settings.
            cloudxr_port: Proxy port used with ``cloudxr_host``.
            cloudxr_proxy_bind: Explicit proxy listener IPv4 address, independent
                of the advertised address. Also retains loopback for USB clients.
                ``None`` uses loopback for offline hosting, otherwise SDK defaults.
        """
        if not math.isfinite(rate_hz) or rate_hz <= 0.0:
            raise ValueError("Isaac Teleop source rate must be finite and positive")
        if not math.isfinite(tracking_rate_hz) or tracking_rate_hz <= 0.0:
            raise ValueError("Isaac Teleop tracking rate must be finite and positive")
        if not math.isfinite(wrist_offset_m) or wrist_offset_m < 0.0:
            raise ValueError(
                "Isaac Teleop wrist offset must be finite and non-negative"
            )
        if startup_timeout_s is not None and (
            not math.isfinite(startup_timeout_s) or startup_timeout_s <= 0.0
        ):
            raise ValueError(
                "Isaac Teleop startup timeout must be finite and positive or None"
            )
        if cloudxr_host is not None and (
            not isinstance(cloudxr_port, int)
            or isinstance(cloudxr_port, bool)
            or not 1 <= cloudxr_port <= 65535
        ):
            raise ValueError("CloudXR proxy port must be an integer from 1 to 65535")
        self.python_executable = str(python_executable)
        self.port = int(port)
        self.host = str(host)
        self.install_dir = str(install_dir)
        self.env_config = None if env_config is None else str(env_config)
        self.device_profile = str(device_profile)
        self.accept_eula = bool(accept_eula)
        self.host_client = bool(host_client)
        if offline_web_client and not host_client:
            raise ValueError("offline_web_client requires host_client=True")
        self.offline_web_client = bool(offline_web_client)
        self.oob_hub = bool(oob_hub)
        self.launch_runtime = bool(launch_runtime)
        self.verbose = bool(verbose)
        self.tracking_rate_hz = float(tracking_rate_hz)
        self.wrist_offset_m = float(wrist_offset_m)
        self._interval_ns = int(1e9 / rate_hz)
        self._startup_timeout_s = (
            None if startup_timeout_s is None else float(startup_timeout_s)
        )
        self._wait_for_controllers = wait_for_controllers
        self._on_message = on_message
        self._on_disconnect = on_disconnect
        self._cloudxr_host = cloudxr_host
        self._cloudxr_port = cloudxr_port
        self._cloudxr_proxy_bind = (
            None if cloudxr_proxy_bind is None else str(IPv4Address(cloudxr_proxy_bind))
        )
        self._store = FrameStore(
            expected_protocol=ISAAC_TELEOP_BRIDGE_PROTOCOL_VERSION,
            expected_pose_convention=WRIST_PROXY_POSE_CONVENTION,
            expected_hello_fields={
                "mode": "cloudxr-webxr",
                "input_pose_convention": OPENXR_STAGE_GRIP_POSE_CONVENTION,
                "output_pose_convention": WRIST_PROXY_POSE_CONVENTION,
                "wrist_offset_m": self.wrist_offset_m,
            },
            stage_help=_ISAAC_TELEOP_STAGE_HELP,
            peer_name="Isaac Teleop bridge",
            compatibility_hint=sidecar_recovery(self.python_executable),
        )
        self._server: ControllerServer | None = None
        self._process: subprocess.Popen | None = None
        self._stopped = False
        self._starting = False
        self._lifecycle_lock = threading.Lock()
        self._stop_lock = threading.Lock()
        self._next_ns = 0
        self._last_capture_ns = 0
        self._client_identity: tuple[str | None, bool] | None = None
        self._generation = 0

    def _bridge_command(self, *, python: str, port: int) -> list[str]:
        """Build the isolated bridge command."""
        bridge = Path(__file__).with_name("isaac_teleop_bridge.py")
        command = [
            python,
            str(bridge),
            "serve",
            "--host",
            self.host,
            "--port",
            str(port),
            "--rate-hz",
            str(self.tracking_rate_hz),
            "--wrist-offset-m",
            str(self.wrist_offset_m),
            "--cloudxr-install-dir",
            self.install_dir,
            "--cloudxr-device-profile",
            self.device_profile,
        ]
        if self.env_config is not None:
            command.extend(("--cloudxr-env-config", self.env_config))
        if self._cloudxr_proxy_bind is not None:
            command.extend(("--cloudxr-proxy-bind", self._cloudxr_proxy_bind))
        for enabled, option in (
            (self.accept_eula, "--accept-eula"),
            (self.host_client, "--host-client"),
            (self.offline_web_client, "--offline-web-client"),
            (self.oob_hub, "--oob-hub"),
            (not self.launch_runtime, "--no-launch-cloudxr-runtime"),
            (self.verbose, "--verbose"),
        ):
            if enabled:
                command.append(option)
        return command

    def start(self, *, wait_for_frame: bool = True) -> None:
        """Start the sidecar and optionally await observation readiness.

        Args:
            wait_for_frame: Await the configured controller readiness. False
                returns after child launch so a browser can connect before the
                first OpenXR frame; the caller must check runtime readiness.
        """
        if self.launch_runtime and not self.accept_eula:
            raise ValueError(
                f"CloudXR startup requires explicit acceptance. {CLOUDXR_EULA_HELP}. "
                "Pass --isaac-teleop-accept-eula (or --accept-eula on the launcher)."
            )
        with self._lifecycle_lock:
            if self._stopped:
                raise RuntimeError("Isaac Teleop source was stopped before starting")
            if self._process is not None:
                return
            if self._starting:
                raise RuntimeError("Isaac Teleop source is already starting")
            self._starting = True
        try:
            python = resolve_python(self.python_executable)
            static_dir = None
            if self.offline_web_client:
                from flashdreams_controller_input.quest_usb_local import (
                    cached_web_client_dir,
                )

                static_dir = cached_web_client_dir()
            startup_deadline = (
                None
                if self._startup_timeout_s is None
                else time.monotonic() + self._startup_timeout_s
            )
            server = ControllerServer(
                on_message=self._consume,
                on_disconnect=self._on_disconnect,
                host=self.host,
                port=self.port,
                peer_name="Isaac Teleop bridge",
            )
            with self._lifecycle_lock:
                if self._stopped:
                    raise RuntimeError("Isaac Teleop source was stopped while starting")
                self._server = server
            server.start()
            if self._stopped:
                raise RuntimeError("Isaac Teleop source was stopped while starting")
            command = self._bridge_command(python=python, port=server.port)
            # CUDA masks use renumbered ordinals while Vulkan retains physical GPU
            # identities. CloudXR needs the complete CUDA set to match its Vulkan
            # device; the parent mask remains intact for the application.
            sidecar_environment = isolated_python_environment()
            sidecar_environment.pop("CUDA_VISIBLE_DEVICES", None)
            sidecar_environment.pop("CONTROL_TOKEN", None)
            if static_dir is not None:
                sidecar_environment["TELEOP_WEB_CLIENT_STATIC_DIR"] = str(static_dir)
            if self._cloudxr_host is not None:
                sidecar_environment.update(
                    TELEOP_PROXY_HOST=self._cloudxr_host,
                    TELEOP_STREAM_SERVER_IP=self._cloudxr_host,
                    TELEOP_STREAM_PORT=str(self._cloudxr_port),
                    PROXY_PORT=str(self._cloudxr_port),
                )
            print(
                "[isaac-teleop] starting isolated CloudXR/OpenXR controller bridge",
                flush=True,
            )
            process = subprocess.Popen(
                command,
                stdin=None,
                text=True,
                start_new_session=True,
                env=sidecar_environment,
            )
            with self._lifecycle_lock:
                cancelled = self._stopped
                if not cancelled:
                    self._process = process
            if cancelled:
                # Startup owns a child returned after stop detached the receiver.
                self._stop_process(process)
                raise RuntimeError("Isaac Teleop source was stopped while starting")
            if wait_for_frame:
                self._await_controllers(deadline=startup_deadline)
        except BaseException as error:
            self.stop()
            if isinstance(error, OSError) and not isinstance(error, TimeoutError):
                raise RuntimeError(
                    f"Isaac Teleop startup failed: {error}. "
                    + sidecar_recovery(self.python_executable)
                ) from error
            raise
        finally:
            with self._lifecycle_lock:
                self._starting = False

    def _consume(self, raw: str | bytes) -> None:
        """Validate locally and forward the unchanged bridge record."""
        if self._stopped:
            return
        self._store.consume(raw)
        if self._on_message is not None and not self._stopped:
            self._on_message(raw)

    def _await_controllers(self, *, deadline: float | None) -> None:
        """Wait for compatible fresh input or a sidecar failure."""
        next_report = 0.0
        while deadline is None or time.monotonic() < deadline:
            if self._stopped:
                raise RuntimeError("Isaac Teleop source was stopped while starting")
            process = self._process
            if process is not None and (status := process.poll()) is not None:
                raise RuntimeError(
                    f"Isaac Teleop sidecar exited before tracking (status {status}). "
                    + sidecar_recovery(self.python_executable)
                )
            compatibility_error = self._store.compatibility_error()
            if compatibility_error is not None:
                raise RuntimeError(compatibility_error)
            if self._store.stage == "failed":
                raise RuntimeError(
                    f"{self._store.explain()}. "
                    + sidecar_recovery(self.python_executable)
                )
            if self._store.ready(require_tracked=self._wait_for_controllers):
                self._next_ns = time.perf_counter_ns()
                expected = (
                    "both tracked controllers"
                    if self._wait_for_controllers
                    else "controller observations"
                )
                print(f"[isaac-teleop] receiving {expected}", flush=True)
                return
            now = time.monotonic()
            if now >= next_report:
                print(f"[isaac-teleop] waiting: {self._store.explain()}", flush=True)
                next_report = now + 10.0
            time.sleep(0.1)
        expected = (
            "tracked Isaac Teleop controllers"
            if self._wait_for_controllers
            else "fresh Isaac Teleop controller frame"
        )
        raise TimeoutError(
            f"no {expected} within "
            f"{self._startup_timeout_s:.0f}s: {self._store.explain()}. "
            + sidecar_recovery(self.python_executable)
        )

    def stop(self) -> None:
        """Terminate the sidecar and receiver. Remain safe when repeated."""
        self._stopped = True
        # Concurrent callers also wait for receiver callbacks to finish.
        with self._stop_lock:
            with self._lifecycle_lock:
                process, self._process = self._process, None
                server, self._server = self._server, None
            try:
                self._stop_process(process)
            finally:
                if server is not None:
                    server.stop()

    @staticmethod
    def _stop_process(process: subprocess.Popen | None) -> None:
        """Stop a published or late-created sidecar process."""
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=_STOP_GRACE_S)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(int(process.pid), signal.SIGKILL)
                except (AttributeError, OSError):
                    process.kill()
                with suppress(subprocess.TimeoutExpired):
                    process.wait(timeout=3.0)

    @property
    def url(self) -> str:
        """Return the local bridge receiver address for diagnostics."""
        server = self._server
        return server.url if server is not None else ""

    @property
    def ready_for_clients(self) -> bool:
        """Report compatible OpenXR readiness without consuming controller input."""
        process = self._process
        return (
            not self._stopped
            and process is not None
            and process.poll() is None
            and self._store.protocol is not None
            and self._store.compatibility_error() is None
            and self._store.stage in ("session", "streaming")
        )

    def check_alive(self) -> None:
        """Validate sidecar and bridge health without consuming an observation."""
        process = self._process
        if process is not None and (status := process.poll()) is not None:
            raise RuntimeError(
                f"Isaac Teleop sidecar exited while running (status {status}). "
                + sidecar_recovery(self.python_executable)
            )
        compatibility_error = self._store.compatibility_error()
        if compatibility_error is not None:
            raise RuntimeError(compatibility_error)
        if self._store.stage == "failed":
            raise RuntimeError(
                f"{self._store.explain()}. " + sidecar_recovery(self.python_executable)
            )

    def poll(self) -> Observation | None:
        """Return the newest absolute controller frame without blocking."""
        self.check_alive()
        if not self._wait_for_controllers and not self._store.ready(
            require_tracked=False
        ):
            return None
        now = time.perf_counter_ns()
        if now < self._next_ns:
            return None
        frame = self._store.snapshot()
        if frame is None:
            return None
        capture_ns = int(frame["client_capture_ns"])
        if capture_ns <= self._last_capture_ns:
            return None
        self._next_ns = max(now, self._next_ns + self._interval_ns)
        self._last_capture_ns = capture_ns
        identity = (frame.get("client_connection_id"), frame.get("is_emulated", False))
        if self._client_identity is not None and identity != self._client_identity:
            self._generation += 1
        self._client_identity = identity
        # Direct observations use the receiver clock; raw relay records retain theirs.
        frame.pop(SOURCE_CLIENT_CAPTURE_NS_KEY, None)
        return Observation(
            frame=frame,
            capture_ns=capture_ns,
            source=self.name,
            generation=self._generation,
        )
