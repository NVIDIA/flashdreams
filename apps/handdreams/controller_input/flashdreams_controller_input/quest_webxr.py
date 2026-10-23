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

"""Physical Quest WebXR sessions managed over local ADB."""

from __future__ import annotations

import argparse
import http.client
import json
import math
import os
import re
import shlex
import shutil
import ssl
import subprocess
import threading
import time
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Protocol
from urllib.parse import urlencode, urlsplit, urlunsplit
from urllib.request import urlopen

from flashdreams_controller_input.controller_types import CLOUDXR_EULA_HELP, Observation
from flashdreams_controller_input.isaac_teleop_source import DEFAULT_ISAAC_TELEOP_PYTHON
from flashdreams_controller_input.quest_source import ManagedHeadset
from flashdreams_controller_input.quest_usb_local import (
    USBLocalTransport,
    cached_web_client_dir,
)

if TYPE_CHECKING:
    from flashdreams_controller_input.isaac_teleop_source import IsaacTeleopSource

_SOCKET = "localabstract:chrome_devtools_remote"
_PAGE_STATE = """(() => {
  const button = document.getElementById('startButton');
  const box = document.getElementById('errorMessageBox');
  const error = box?.classList.contains('show') &&
    !box.classList.contains('success') && !box.classList.contains('info');
  const rect = button?.getBoundingClientRect();
  return {ready: document.readyState === 'complete' && !!button &&
      !button.disabled && button.textContent.trim().toUpperCase() === 'CONNECT',
    x: rect ? rect.left + rect.width / 2 : 0,
    y: rect ? rect.top + rect.height / 2 : 0,
    error: error ? document.getElementById('errorMessageText')?.textContent?.slice(0, 4096) : ''};
})()"""


def _find_adb() -> str:
    """Resolve ADB in the headset launcher's override, PATH, and SDK order."""
    explicit = os.environ.get("QUEST_ADB")
    if explicit and (executable := shutil.which(os.path.abspath(explicit))):
        return executable
    if executable := shutil.which("adb"):
        return os.path.abspath(executable)
    for root in (
        os.environ.get("ANDROID_SDK_ROOT"),
        os.environ.get("ANDROID_HOME"),
        str(Path.home() / "Android" / "Sdk"),
    ):
        if root and (
            executable := shutil.which(str(Path(root) / "platform-tools" / "adb"))
        ):
            return os.path.abspath(executable)
    raise FileNotFoundError(
        "ADB not found; set QUEST_ADB, add adb to PATH, or set ANDROID_SDK_ROOT/ANDROID_HOME"
    )


def _adb(executable: str, serial: str | None, *args: str) -> str:
    command = [executable]
    if serial is not None:
        command.extend(("-s", serial))
    # adb joins shell arguments without escaping them on the remote shell.
    command.extend(("shell", shlex.join(args[1:])) if args[0] == "shell" else args)
    result = subprocess.run(command, capture_output=True, text=True, timeout=5)
    if result.returncode:
        raise RuntimeError(
            f"ADB {args[0]} failed; check USB authorization and the headset"
        )
    return result.stdout.strip()


def _device(serial: str | None, adb_executable: str) -> str:
    if not serial:
        devices = [
            line.split()[0]
            for line in _adb(adb_executable, None, "devices").splitlines()
            if len(line.split()) == 2 and line.split()[1] == "device"
        ]
        if len(devices) != 1:
            raise ValueError("select one authorized headset with --quest-device")
        serial = devices[0]
    if _adb(adb_executable, serial, "get-state") != "device":
        raise RuntimeError("Quest is not authorized for ADB")
    if not _adb(
        adb_executable, serial, "shell", "pm", "path", "com.oculus.browser"
    ).startswith("package:"):
        raise RuntimeError("Meta Quest Browser is not installed")
    return serial


def _discover_device(adb_executable: str) -> tuple[str | None, str]:
    """Find an authorized USB Quest without selecting unrelated Android devices."""
    candidates = []
    for line in _adb(adb_executable, None, "devices").splitlines():
        fields = line.split()
        if len(fields) != 2 or fields[1] != "device":
            continue
        serial = fields[0]
        try:
            if not _adb(adb_executable, serial, "get-devpath").startswith("usb:"):
                continue
            if _adb(
                adb_executable, serial, "shell", "pm", "path", "com.oculus.browser"
            ).startswith("package:"):
                candidates.append(serial)
        except (RuntimeError, subprocess.TimeoutExpired):
            # A non-Quest package probe or unplugged candidate is not fatal.
            continue
    if len(candidates) == 1:
        return candidates[0], "Detected"
    if candidates:
        return None, "Multiple headsets; select one with --quest-device"
    return None, "No authorized USB headset detected"


def _https_origin(url: str) -> str:
    parts = urlsplit(url)
    if (
        parts.scheme not in ("https", "wss")
        or not parts.hostname
        or parts.username
        or parts.password
    ):
        return ""
    return f"https://{parts.hostname}:{parts.port or 443}"


class _CDP:
    """Issue bounded CDP calls and observe this page's OOB stream status."""

    def __init__(self, websocket, server_origin: str = "") -> None:
        self.websocket = websocket
        self.server_origin = _https_origin(server_origin)
        self.sequence = 0
        self.oob_socket: str | None = None
        self.registered = False
        self.streaming = False

    def _send(self, method: str, params: dict) -> int:
        self.sequence += 1
        self.websocket.send(
            json.dumps({"id": self.sequence, "method": method, "params": params})
        )
        return self.sequence

    def call(self, method: str, **params) -> dict:
        request_id = self._send(method, params)
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            message = json.loads(
                self.websocket.recv(timeout=max(0.001, deadline - time.monotonic()))
            )
            if message.get("id") == request_id:
                if "error" in message:
                    raise RuntimeError(
                        f"Quest Browser rejected {method}: {message['error'].get('code')}"
                    )
                return message.get("result", {})
            self._event(message)
        raise TimeoutError(f"Quest Browser did not respond to {method}")

    def _event(self, message: dict) -> None:
        method, params = message.get("method"), message.get("params", {})
        if method == "Security.certificateError":
            origin = _https_origin(params.get("requestURL", ""))
            self._send(
                "Security.handleCertificateError",
                {
                    "eventId": params["eventId"],
                    "action": "continue"
                    if origin and origin == self.server_origin
                    else "cancel",
                },
            )
        elif method == "Network.webSocketCreated":
            url = urlsplit(params.get("url", ""))
            if (
                url.scheme == "wss"
                and _https_origin(url.geturl()) == self.server_origin
                and url.path == "/oob/v1/ws"
            ):
                self.oob_socket = params["requestId"]
                self.registered = self.streaming = False
        elif params.get("requestId") == self.oob_socket and self.oob_socket is not None:
            if method == "Network.webSocketClosed":
                self.registered = self.streaming = False
            elif method in (
                "Network.webSocketFrameSent",
                "Network.webSocketFrameReceived",
            ):
                try:
                    data = json.loads(params["response"]["payloadData"])
                    if not isinstance(data, dict):
                        return
                    payload = data.get("payload", {})
                    if not isinstance(payload, dict):
                        return
                    if method.endswith("Received") and data.get("type") == "hello":
                        self.registered = bool(payload.get("clientId"))
                    elif method.endswith("Sent") and data.get("type") == "streamStatus":
                        self.streaming = payload.get("streaming") is True
                except (KeyError, ValueError, TypeError):
                    pass


def _json(port: int, path: str) -> dict:
    with urlopen(f"http://127.0.0.1:{port}/{path}", timeout=5) as response:
        return json.load(response)


def _debugger_url(url: str, port: int) -> str:
    parts = urlsplit(url)
    return urlunsplit(("ws", f"127.0.0.1:{port}", parts.path, parts.query, ""))


class _HeadsetHealth(Protocol):
    def check_alive(self) -> None: ...


@contextmanager
def _page(
    serial: str,
    headset: _HeadsetHealth,
    timeout: float,
    server_origin: str,
    adb_executable: str,
):
    from websockets.sync.client import connect

    headset.check_alive()
    # A new tab cannot release an exhausted browser process's service bindings.
    print(
        "[quest-webxr] restarting Quest Browser; existing browser sessions will close (saved data and permissions are preserved)",
        flush=True,
    )
    _adb(adb_executable, serial, "shell", "am", "force-stop", "com.oculus.browser")
    headset.check_alive()
    _adb(
        adb_executable,
        serial,
        "shell",
        "am",
        "start",
        "-n",
        "com.oculus.browser/.OculusLauncherActivity",
    )
    deadline = time.monotonic() + timeout
    while "@chrome_devtools_remote" not in _adb(
        adb_executable, serial, "shell", "cat", "/proc/net/unix"
    ):
        headset.check_alive()
        if time.monotonic() >= deadline:
            raise TimeoutError(
                "Quest Browser did not expose DevTools; check the headset for a lock/privacy screen"
            )
        time.sleep(0.25)
    port = int(_adb(adb_executable, serial, "forward", "--no-rebind", "tcp:0", _SOCKET))
    try:
        version = _json(port, "json/version")
        with connect(
            _debugger_url(version["webSocketDebuggerUrl"], port),
            open_timeout=5,
            close_timeout=2,
        ) as websocket:
            browser = _CDP(websocket)
            target = browser.call("Target.createTarget", url="about:blank")["targetId"]
            try:
                with connect(
                    f"ws://127.0.0.1:{port}/devtools/page/{target}",
                    open_timeout=5,
                    close_timeout=2,
                ) as page:
                    yield _CDP(page, server_origin)
            finally:
                try:
                    browser.call("Target.closeTarget", targetId=target)
                except Exception as error:
                    print(
                        f"[quest-webxr] could not close the owned browser tab ({type(error).__name__}); exit VR in the headset",
                        flush=True,
                    )
    finally:
        try:
            rule = f"{serial} tcp:{port} {_SOCKET}"
            if rule in _adb(adb_executable, serial, "forward", "--list").splitlines():
                _adb(adb_executable, serial, "forward", "--remove", f"tcp:{port}")
        except Exception as error:
            print(
                f"[quest-webxr] could not remove ADB forward tcp:{port} ({type(error).__name__})",
                flush=True,
            )


def _page_error_summary(banner: str) -> str:
    """Summarize known errors without echoing credential-bearing page text."""
    banner = banner[:4096]
    summary = "Isaac Teleop reported a connection error"
    if "could not establish the signaling websocket connection" in banner.lower():
        summary += " (CloudXR signaling WebSocket connection failed)"
    if code := re.search(r"\bError code:\s*(0x[0-9a-fA-F]{1,8})\b", banner):
        summary += f"; error code {code[1]}"
    return summary + "; inspect its error banner in the headset"


def _run_page(
    cdp: _CDP,
    headset: _HeadsetHealth,
    args: argparse.Namespace,
    url: str,
    *,
    can_connect: Callable[[], bool] = lambda: True,
    on_status: Callable[[str], None] = lambda _: None,
) -> None:
    cdp.call("Network.enable")
    if args.accept_server_certificate:
        cdp.call("Security.enable")
        cdp.call("Security.setOverrideCertificateErrors", override=True)
    try:
        cdp.call("Page.bringToFront")
        cdp.call("Page.navigate", url=url)
        deadline = time.monotonic() + args.startup_timeout
        clicked = active = False
        ready_at = None
        while True:
            headset.check_alive()
            if not clicked and not can_connect():
                on_status("Waiting for the active client to disconnect")
                ready_at = None
                deadline = time.monotonic() + args.startup_timeout
                time.sleep(0.25)
                continue
            state = (
                cdp.call("Runtime.evaluate", expression=_PAGE_STATE, returnByValue=True)
                .get("result", {})
                .get("value", {})
            )
            if state.get("error"):
                raise RuntimeError(_page_error_summary(state["error"]))
            if not clicked and state.get("ready"):
                if ready_at is None:
                    ready_at = time.monotonic()
                elif time.monotonic() - ready_at >= 2:
                    for event in ("mousePressed", "mouseReleased"):
                        cdp.call(
                            "Input.dispatchMouseEvent",
                            type=event,
                            x=state["x"],
                            y=state["y"],
                            button="left",
                            clickCount=1,
                        )
                    cdp.call(
                        "Runtime.evaluate",
                        expression="if (!document.getElementById('startButton')?.disabled && document.getElementById('startButton')?.textContent.trim().toUpperCase() === 'CONNECT') document.getElementById('startButton').click()",
                    )
                    clicked = True
                    on_status("Starting immersive VR; check the headset for Allow")
                    print(
                        "[quest-webxr] CONNECT requested; waiting for streaming. If Quest shows Allow, approve immersive VR in the headset.",
                        flush=True,
                    )
            elif not clicked:
                ready_at = None
            streaming = cdp.registered and cdp.streaming
            if streaming and not active:
                active = True
                on_status("Streaming over USB")
                print(
                    "[quest-webxr] streaming confirmed; Ctrl-C closes this VR tab and releases the headset",
                    flush=True,
                )
            elif active and not streaming:
                raise RuntimeError(
                    "the Quest WebXR stream disconnected; restart the headset route to reconnect"
                )
            if not active and time.monotonic() >= deadline:
                raise TimeoutError(
                    "WebXR did not start streaming. Check Quest Allow/lock screens, "
                    "TLS, the OOB hub, and USB/TURN connectivity "
                    "(Quest Wi-Fi must retain an IP address)"
                )
            time.sleep(0.25)
    finally:
        if args.accept_server_certificate:
            try:
                cdp.call("Security.setOverrideCertificateErrors", override=False)
            except Exception:
                pass  # Closing the owned target also removes this session's override.


def _validate_options(args: argparse.Namespace) -> None:
    if not 1 <= args.cloudxr_port <= 65535:
        raise ValueError("--cloudxr-port must be between 1 and 65535")
    if not math.isfinite(args.startup_timeout) or args.startup_timeout <= 0:
        raise ValueError("--startup-timeout must be finite and positive")


def _urls(args: argparse.Namespace) -> tuple[str, str]:
    _validate_options(args)
    params = {
        "oobEnable": "1",
        "headless": "true",
        "panelHiddenAtStart": "true",
        "serverIP": "127.0.0.1",
        "port": str(args.cloudxr_port),
        "turnServer": f"turn:127.0.0.1:{args.usb_turn_port}?transport=tcp",
        "turnUsername": "cloudxr",
        "turnCredential": args.usb_turn_credential,
        "iceRelayOnly": "1",
    }
    origin = f"https://127.0.0.1:{args.cloudxr_port}"
    return origin, origin + "/client/?" + urlencode(params)


def add_relay_arguments(parser: argparse.ArgumentParser) -> None:
    """Add laptop CloudXR options to the existing controller relay command."""
    group = parser.add_argument_group("Laptop Isaac Teleop / Quest WebXR")
    group.add_argument("--cloudxr-port", type=int, default=48322)
    group.add_argument("--isaac-teleop-python", default=DEFAULT_ISAAC_TELEOP_PYTHON)
    group.add_argument(
        "--isaac-teleop-accept-eula",
        action="store_true",
        help=CLOUDXR_EULA_HELP,
    )
    group.add_argument(
        "--accept-server-certificate",
        action="store_true",
        help="Accept TLS certificate errors only for this laptop CloudXR origin in the managed Quest tab.",
    )
    group.add_argument(
        "--keep-browser-open-on-error",
        action="store_true",
        help="Keep the managed Quest tab and USB transport open after a browser failure for inspection; Ctrl-C still cleans up.",
    )
    group.add_argument(
        "--startup-timeout",
        type=float,
        default=120,
        help="Timeout for each CloudXR/browser startup stage, excluding physical wake.",
    )


class QuestWebXRProducer:
    """Own local CloudXR and Quest browser input for direct use or a relay.

    Startup runs on a worker so controller polling remains responsive during
    physical wake, runtime loading, and browser setup. Relay callbacks forward
    raw sidecar records; direct consumers receive physical observations.
    """

    name = "quest-webxr"
    """Physical Quest input identity used by direct controller consumers."""

    def __init__(
        self,
        args: argparse.Namespace,
        *,
        source: IsaacTeleopSource | None = None,
        shared_runtime: bool = False,
        hub_host: str = "127.0.0.1",
        can_connect: Callable[[], bool] = lambda: True,
    ) -> None:
        """Configure browser ownership and an optional shared sidecar source.

        Args:
            args: Quest browser and local CloudXR launch options.
            source: Preconfigured source; ``None`` creates the relay's own source.
            shared_runtime: Borrow the source without starting, polling or stopping
                it; headset failure leaves browser input available.
            hub_host: Expected hub advertisement; USB URLs always use loopback.
            can_connect: Whether controller ownership permits automatic connection.
        """
        if not args.isaac_teleop_accept_eula:
            raise ValueError(
                f"WebXR requires explicit --isaac-teleop-accept-eula. {CLOUDXR_EULA_HELP}"
            )
        self._args = argparse.Namespace(**vars(args))
        _validate_options(self._args)
        if shared_runtime and source is None:
            raise ValueError("a shared runtime requires an existing Isaac source")
        self._shared_runtime = shared_runtime
        self._hub_host = hub_host
        self._can_connect = can_connect
        self._status = "Checking for a local headset..."
        self._stopped = threading.Event()
        self._restart = threading.Event()
        self._debug_hold = threading.Event()
        self._lifecycle_lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._headset: ManagedHeadset | None = None
        self._usb: USBLocalTransport | None = None
        self._usb_started = False
        self._source = source
        self._source_started = False
        self._failure: Exception | None = None

    @property
    def status_message(self) -> str:
        """Return actionable headset status without ADB or network calls."""
        if self._stopped.is_set():
            return "Stopped"
        if self._restart.is_set():
            return "Restarting local headset..."
        if self._debug_hold.is_set():
            return (
                "Connection failed; browser kept open; press B to retry"
                if self._shared_runtime
                else "Connection failed; browser kept open; Ctrl-C to close"
            )
        if self._failure is not None:
            return (
                "Unavailable; check controller battery and press B to retry"
                if self._shared_runtime
                else "Unavailable; see server log"
            )
        headset = self._headset
        if headset is not None and headset.waiting_for_wake:
            return "Press the headset power button"
        with self._lifecycle_lock:
            return self._status

    def _set_status(self, message: str) -> None:
        with self._lifecycle_lock:
            self._status = message

    def _connection_available(self) -> bool:
        if not self._shared_runtime:
            return True
        if not self._can_connect():
            return False
        state = self._await_hub()
        self.check_alive()
        # Ownership may change while a transient hub failure is being retried.
        return self._can_connect() and not any(
            peer.get("streaming") is True for peer in state.get("headsets", [])
        )

    @property
    def wake_ready(self) -> bool:
        """Report whether this run has passed the headset's wake barrier."""
        return (
            not self._stopped.is_set()
            and not self._debug_hold.is_set()
            and self._headset is not None
            and self._headset.wake_ready
        )

    def start(
        self,
        on_message: Callable[[str | bytes], None] | None = None,
        on_disconnect: Callable[[], None] | None = None,
        *,
        wait_for_frame: bool = False,
    ) -> None:
        """Start the worker without waiting for physical wake or observations.

        Args:
            on_message: Receive raw records from an internally created source;
                ``None`` discards records.
            on_disconnect: Receive input suspension or worker completion;
                ``None`` does nothing.
            wait_for_frame: Accept the shared input-source startup interface.
                Startup always returns immediately so physical wake can follow.
        """
        with self._lifecycle_lock:
            if self._stopped.is_set() or self._thread is not None:
                return
            self._thread = threading.Thread(
                target=self._run,
                args=(
                    on_message or (lambda raw: None),
                    on_disconnect or (lambda: None),
                ),
                name="quest-webxr",
            )
            self._thread.start()

    def restart(self) -> None:
        """Request a fresh physical-wake cycle without blocking the operator UI."""
        if not self._shared_runtime:
            raise RuntimeError("only the shared-runtime headset helper can restart")
        with self._lifecycle_lock:
            if not self._stopped.is_set():
                self._restart.set()

    def check_alive(self) -> None:
        """Check cancellation, device ownership, and local sidecar health."""
        if self._stopped.is_set():
            raise RuntimeError("local Quest WebXR input stopped")
        if self._restart.is_set():
            raise RuntimeError("local Quest WebXR restart requested")
        failure = self._failure
        if self._debug_hold.is_set():
            return
        if failure is not None:
            raise RuntimeError("local Quest WebXR input failed") from failure
        if self._headset is not None:
            self._headset.check_alive()
        if self._usb_started:
            assert self._usb is not None
            self._usb.check_alive()
        if self._source_started:
            assert self._source is not None
            self._source.check_alive()

    def poll(self) -> Observation | None:
        """Return the latest physical observation after wake and sidecar launch."""
        if self._shared_runtime:
            raise RuntimeError("poll the shared Isaac source, not its headset helper")
        self.check_alive()
        if not self.wake_ready or not self._source_started:
            return None
        assert self._source is not None
        observation = self._source.poll()
        return None if observation is None else replace(observation, source=self.name)

    def stop(self) -> None:
        """Cancel startup and join cleanup of the browser, sidecar, and headset."""
        with self._lifecycle_lock:
            self._stopped.set()
            self._restart.set()
            thread = self._thread
        if self._source is not None and not self._shared_runtime:
            self._source.stop()
        if thread is not None and thread is not threading.current_thread():
            thread.join()

    def _run(self, on_message, on_disconnect) -> None:
        require_physical_wake = self._args.require_physical_wake
        while not self._stopped.is_set():
            with self._lifecycle_lock:
                if self._stopped.is_set():
                    return
                require_physical_wake |= self._restart.is_set()
                self._restart.clear()
                self._failure = None
                self._debug_hold.clear()
                self._headset = None
                self._usb = None
                self._status = "Checking for a local headset..."
            self._run_once(on_message, on_disconnect, require_physical_wake)
            if not self._shared_runtime:
                return
            # Keep the shared Isaac runtime alive while waiting for an explicit retry.
            self._restart.wait()

    def _run_once(self, on_message, on_disconnect, require_physical_wake) -> None:
        try:
            try:
                # Validate optional imports before touching headset settings.
                from websockets.sync.client import connect  # noqa: F401

                from flashdreams_controller_input.isaac_teleop_source import (
                    IsaacTeleopSource,
                )

                try:
                    adb_executable = _find_adb()
                except FileNotFoundError:
                    if not self._shared_runtime or self._args.quest_device:
                        raise
                    self._set_status("No ADB installed; browser input is available")
                    return
                if self._shared_runtime and not self._args.quest_device:
                    serial, message = _discover_device(adb_executable)
                    self._set_status(message)
                    if serial is None:
                        return
                else:
                    serial = _device(self._args.quest_device, adb_executable)
                cached_web_client_dir()
                if not shutil.which("turnserver"):
                    raise RuntimeError(
                        "Quest USB streaming requires coturn on this machine: sudo apt-get install coturn"
                    )
                if os.environ.get("QUEST_META_PIN"):
                    raise ValueError(
                        "Unset QUEST_META_PIN for Quest USB streaming; Meta PIN-based settings may require internet"
                    )
                self.check_alive()
                self._set_status("Preparing headset...")
                self._headset = ManagedHeadset(
                    device_serial=serial,
                    require_physical_wake=require_physical_wake,
                    adb_executable=adb_executable,
                )
                self.check_alive()
                self._headset.start()
                while not self.wake_ready:
                    self.check_alive()
                    self._stopped.wait(0.1)
                self.check_alive()
                self._set_status("Starting USB transport...")
                self._usb = USBLocalTransport(
                    lambda *parts: _adb(adb_executable, serial, *parts),
                    self._args.cloudxr_port,
                )
                self._usb.start(self.check_alive)
                self._usb_started = True
                self._args.usb_turn_port = self._usb.turn_port
                self._args.usb_turn_credential = self._usb.credential
                print(
                    "[quest-webxr] USB-local transport ready; Quest streaming uses USB",
                    flush=True,
                )
                self.check_alive()
                origin, url = _urls(self._args)
                if self._source is None:
                    self._source = IsaacTeleopSource(
                        python_executable=self._args.isaac_teleop_python,
                        accept_eula=self._args.isaac_teleop_accept_eula,
                        oob_hub=True,
                        wait_for_controllers=False,
                        startup_timeout_s=self._args.startup_timeout,
                        cloudxr_host="127.0.0.1",
                        cloudxr_port=self._args.cloudxr_port,
                        host_client=True,
                        offline_web_client=True,
                        on_message=on_message,
                        on_disconnect=on_disconnect,
                    )
                self.check_alive()
                if not self._shared_runtime:
                    self._source.start(wait_for_frame=False)
                self._source_started = True
                self._await_hub()
                self.check_alive()
                self._set_status("Opening immersive VR...")
                print(
                    "[quest-webxr] local CloudXR ready; opening Quest WebXR",
                    flush=True,
                )
                with _page(
                    serial, self, self._args.startup_timeout, origin, adb_executable
                ) as cdp:
                    try:
                        _run_page(
                            cdp,
                            self,
                            self._args,
                            url,
                            can_connect=self._connection_available,
                            on_status=self._set_status,
                        )
                    except Exception as error:
                        if (
                            getattr(self._args, "keep_browser_open_on_error", False)
                            and not self._stopped.is_set()
                            and not self._restart.is_set()
                        ):
                            # Pause inside the owned tab's context so its USB
                            # transport and awake headset remain inspectable.
                            self._debug_hold.set()
                            self._failure = error
                            on_disconnect()
                            print(f"[quest-webxr] input failed: {error}", flush=True)
                            retry = (
                                "press B to clean up and retry, or Ctrl-C to exit"
                                if self._shared_runtime
                                else "Ctrl-C to clean up, then rerun to retry"
                            )
                            print(
                                f"[quest-webxr] browser kept open for inspection; {retry}",
                                flush=True,
                            )
                            # stop() also signals restart. Keep the hold through
                            # teardown; the next startup clears it.
                            self._restart.wait()
                        raise
            finally:
                self._source_started = False
                try:
                    if self._source is not None and not self._shared_runtime:
                        self._source.stop()
                finally:
                    try:
                        self._usb_started = False
                        if self._usb is not None:
                            self._usb.stop()
                    finally:
                        if self._headset is not None:
                            self._headset.stop()
        except Exception as error:
            if not self._stopped.is_set() and not self._restart.is_set():
                self._failure = error
                print(f"[quest-webxr] input failed: {error}", flush=True)
        finally:
            on_disconnect()

    def _await_hub(self) -> dict:
        """Wait for a validated local OOB snapshot without a WebXR sample."""
        deadline = time.monotonic() + self._args.startup_timeout
        while time.monotonic() < deadline:
            self.check_alive()
            try:
                return self._hub_snapshot()
            except (
                OSError,
                http.client.IncompleteRead,
                http.client.BadStatusLine,
                http.client.LineTooLong,
            ):
                self._stopped.wait(0.1)
        raise TimeoutError("local CloudXR OOB hub did not become ready")

    def _hub_snapshot(self) -> dict:
        """Read the owned hub on loopback and validate its advertised endpoint."""
        # This credential-free loopback probe checks availability, not ownership.
        # Controller-relay TLS and the browser's certificate opt-in are separate.
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        context.check_hostname = False
        context.verify_mode = ssl.CERT_NONE
        connection = http.client.HTTPSConnection(
            "127.0.0.1", self._args.cloudxr_port, timeout=1, context=context
        )
        try:
            connection.request("GET", "/api/oob/v1/state")
            response = connection.getresponse()
            if response.status != 200:
                raise RuntimeError("local CloudXR port does not provide an OOB hub")
            state = json.loads(response.read(64 * 1024 + 1))
            config = state.get("config", {}) if isinstance(state, dict) else {}
            if not isinstance(config, dict) or (
                config.get("serverIP"),
                config.get("port"),
            ) != (self._hub_host, self._args.cloudxr_port):
                raise RuntimeError(
                    "local OOB hub advertises the wrong CloudXR address; check for another running demo"
                )
            peers = state.get("headsets", [])
            if not isinstance(peers, list) or not all(
                isinstance(peer, dict) for peer in peers
            ):
                raise RuntimeError("local OOB hub returned invalid client state")
            return state
        finally:
            connection.close()
