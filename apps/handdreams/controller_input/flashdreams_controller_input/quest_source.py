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

"""Controller frame validation and managed Quest browser wake state."""

from __future__ import annotations

import json
import math
import os
import signal
import subprocess
import threading
import time
from contextlib import suppress
from pathlib import Path

from flashdreams_controller_input import frames

_LAUNCHER = "quest-offhead-teleop.sh"
_STALE_MS = 500.0
_STOP_GRACE_S = 15.0

# A malformed producer emits bad frames at full rate, so reports are throttled.
_REJECT_LOG_INTERVAL_S = 5.0

_STAGE_HELP = {
    "": "the controller producer has not connected",
    "connected": "the controller producer connected and is starting tracking",
    "session": "the controller session is waiting for tracked controllers",
    "failed": "the controller producer could not start tracking",
}


_MISSING_HELLO_FIELD = object()


def _hello_field_matches(actual: object, expected: object) -> bool:
    """Compare one advertised handshake value without accepting bools as floats."""
    if isinstance(expected, float):
        return (
            not isinstance(actual, bool)
            and isinstance(actual, (int, float))
            and math.isfinite(float(actual))
            and math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=1e-9)
        )
    return actual == expected


def _launcher_script() -> Path:
    """Return the packaged headset launcher."""
    script = Path(__file__).resolve().with_name(_LAUNCHER)
    if not script.is_file():
        raise RuntimeError(f"the managed Quest launcher is missing: {script}")
    return script


class FrameStore:
    """Holds the newest validated controller frame, from a controller producer.

    Args:
        expected_protocol: Required controller protocol; ``None`` accepts a
            producer with no handshake.
        expected_pose_convention: Required pose convention after a
            compatible protocol handshake. ``None`` skips this check.
        expected_hello_fields: Additional required fields in a compatible
            handshake. Numeric float values must be finite and equal.
        stage_help: Startup-stage explanations for the selected transport.
        peer_name: Producer name used in handshake diagnostics.
        compatibility_hint: Recovery instructions for a protocol mismatch.
    """

    def __init__(
        self,
        *,
        expected_protocol: int | None = None,
        expected_pose_convention: str | None = None,
        expected_hello_fields: dict[str, object] | None = None,
        stage_help: dict[str, str] | None = None,
        peer_name: str = "controller producer",
        compatibility_hint: str = "restart the bridge with matching Isaac settings",
    ) -> None:
        self._lock = threading.Lock()
        self._latest: dict | None = None
        self._received = 0
        self._rejected = 0
        self._next_reject_log = 0.0
        self._stage = ""
        self._stage_detail = ""
        self._expected_protocol = expected_protocol
        self._expected_pose_convention = expected_pose_convention
        self._expected_hello_fields = dict(expected_hello_fields or {})
        self._protocol: int | None = None
        self._pose_convention: str | None = None
        self._peer_version = ""
        self._hello_fields: dict[str, object] = {}
        self._stage_help = dict(stage_help or _STAGE_HELP)
        self._peer_name = str(peer_name)
        self._compatibility_hint = str(compatibility_hint)

    @property
    def received(self) -> int:
        """Return the number of accepted controller frames."""
        with self._lock:
            return self._received

    @property
    def rejected(self) -> int:
        """Return the number of malformed messages or frames."""
        with self._lock:
            return self._rejected

    @property
    def stage(self) -> str:
        """Return the producer's last reported startup stage."""
        with self._lock:
            return self._stage

    @property
    def protocol(self) -> int | None:
        """Return the controller protocol announced by the peer, when present."""
        with self._lock:
            return self._protocol

    @property
    def pose_convention(self) -> str | None:
        """Return the pose convention announced by the controller producer."""
        with self._lock:
            return self._pose_convention

    def compatibility_error(self) -> str | None:
        """Return a protocol or advertised-semantics mismatch explanation."""
        with self._lock:
            protocol = self._protocol
            peer_version = self._peer_version
            expected = self._expected_protocol
            pose_convention = self._pose_convention
            expected_pose_convention = self._expected_pose_convention
            hello_fields = dict(self._hello_fields)
            expected_hello_fields = dict(self._expected_hello_fields)
        if protocol is None:
            return None
        version = f" version {peer_version}" if peer_version else ""
        if expected is not None and protocol != expected:
            return (
                f"{self._peer_name}{version} uses protocol {protocol}, but the laptop "
                f"requires {expected}; {self._compatibility_hint}"
            )
        if (
            expected_pose_convention is not None
            and pose_convention != expected_pose_convention
        ):
            actual = repr(pose_convention) if pose_convention is not None else "none"
            return (
                f"{self._peer_name}{version} reports pose convention {actual}, but "
                f"the laptop requires {expected_pose_convention!r}; "
                f"{self._compatibility_hint}"
            )
        for field, required in expected_hello_fields.items():
            actual = hello_fields.get(field, _MISSING_HELLO_FIELD)
            if not _hello_field_matches(actual, required):
                advertised = (
                    "<missing>" if actual is _MISSING_HELLO_FIELD else repr(actual)
                )
                return (
                    f"{self._peer_name}{version} reports hello {field}={advertised}, "
                    f"but the laptop requires {required!r}; "
                    f"{self._compatibility_hint}"
                )
        return None

    def explain(self) -> str:
        """Describe what the selected controller producer is waiting for."""
        compatibility_error = self.compatibility_error()
        if compatibility_error is not None:
            return compatibility_error
        with self._lock:
            stage = self._stage
            detail = self._stage_detail
            stage_help = dict(self._stage_help)
        message = stage_help.get(stage, f"the controller source reported {stage!r}")
        return f"{message} ({detail})" if detail else message

    def snapshot(self) -> dict | None:
        """Return a shallow copy of the newest frame, when available."""
        with self._lock:
            return None if self._latest is None else dict(self._latest)

    def age_ms(self) -> float | None:
        """Return the newest frame's host-side age in milliseconds."""
        frame = self.snapshot()
        if frame is None:
            return None
        return (time.perf_counter_ns() - int(frame["client_capture_ns"])) / 1e6

    def ready(
        self, stale_ms: float = _STALE_MS, *, require_tracked: bool = True
    ) -> bool:
        """Report whether a compatible, fresh frame meets the tracking requirement."""
        age = self.age_ms()
        compatibility_ok = self.compatibility_error() is None
        with self._lock:
            protocol_ok = self._expected_protocol is None or (
                self._protocol == self._expected_protocol
            )
            pose_convention_ok = self._expected_pose_convention is None or (
                self._pose_convention == self._expected_pose_convention
            )
            hello_fields_present = not self._expected_hello_fields or bool(
                self._hello_fields
            )
        return (
            compatibility_ok
            and protocol_ok
            and pose_convention_ok
            and hello_fields_present
            and age is not None
            and age <= stale_ms
            and (
                not require_tracked or frames.both_controllers_tracked(self.snapshot())
            )
        )

    def consume(self, message: str | bytes) -> None:
        """Take one message off the wire, tolerating malformed input.

        A bad frame must not drop the socket. Otherwise one unparseable message
        becomes a reconnect loop when the producer sends it again.
        """
        try:
            payload = json.loads(message)
        except (TypeError, ValueError) as error:
            self._note_rejected(error)
            return
        if isinstance(payload, dict) and payload.get("type") == "hello":
            self._note_hello(payload)
            return
        if isinstance(payload, dict) and payload.get("type") == "status":
            self._note_status(payload)
            return
        try:
            frame = frames.parse_xr_frame(payload)
        except (TypeError, ValueError) as error:
            self._note_rejected(error)
            return
        if frame is not None:
            self._store(frame)

    def _note_hello(self, payload: dict) -> None:
        """Record a controller producer handshake."""
        protocol = payload.get("protocol")
        if isinstance(protocol, bool) or not isinstance(protocol, int):
            self._note_rejected(
                ValueError("controller hello requires an integer protocol")
            )
            return
        version = str(payload.get("version", ""))
        pose_convention_value = payload.get("pose_convention")
        pose_convention = (
            pose_convention_value if isinstance(pose_convention_value, str) else None
        )
        with self._lock:
            self._hello_fields = dict(payload)
            self._protocol = protocol
            self._peer_version = version
            self._pose_convention = pose_convention
        suffix = f" ({version})" if version else ""
        print(f"[quest] {self._peer_name} protocol {protocol}{suffix}", flush=True)

    def _note_status(self, payload: dict) -> None:
        """Record a producer startup status."""
        stage = str(payload.get("stage", ""))
        detail = str(payload.get("detail", ""))
        with self._lock:
            changed = stage != self._stage or detail != self._stage_detail
            self._stage = stage
            self._stage_detail = detail
        if changed:
            print(
                f"[quest] source reports {stage}{f': {detail}' if detail else ''}",
                flush=True,
            )

    def _store(self, frame: dict) -> None:
        """Assign source identity and hold missing controllers as untracked."""
        with self._lock:
            first = self._received == 0
            frame["source_frame_index"] = self._received
            previous = self._latest
            if previous is not None:
                # Preserve pose continuity while exposing the tracking loss.
                for side in frames.SIDES:
                    if frame.get(side) is None and previous.get(side) is not None:
                        held = dict(previous[side])
                        held["tracked"] = False
                        frame[side] = held
            self._latest = frame
            self._received += 1
        if first:
            print("[quest] first controller frame received", flush=True)

    def _note_rejected(self, error: Exception) -> None:
        """Count a rejected message, logging at most once per interval."""
        with self._lock:
            self._rejected += 1
            count = self._rejected
            now = time.monotonic()
            report = now >= self._next_reject_log
            if report:
                self._next_reject_log = now + _REJECT_LOG_INTERVAL_S
        if report:
            print(
                f"[quest] ignoring malformed frame ({error}); {count} rejected so far",
                flush=True,
            )


class ManagedHeadset:
    """Prepares the headset over ADB for the life of the demo.

    Args:
        device_serial: Explicit ADB device serial, or ``None`` to let the
            launcher select the only authorized device.
        require_physical_wake: Wait for an operator wake from confirmed sleep
            before starting browser input, without blocking the host transport.
        adb_executable: Resolved ADB executable, or ``None`` for launcher discovery.
    """

    def __init__(
        self,
        *,
        device_serial: str | None = None,
        require_physical_wake: bool = False,
        adb_executable: str | None = None,
    ) -> None:
        self.device_serial = device_serial
        self._require_physical_wake = require_physical_wake
        self._adb_executable = adb_executable
        self._wake_ready = False
        self._waiting_for_wake = False
        self._wake_ready_fd: int | None = None
        self._process: subprocess.Popen | None = None
        self._lifecycle_lock = threading.Lock()
        self._stopped = False

    @property
    def wake_ready(self) -> bool:
        """Report wake readiness, after wake and hold setup."""
        with self._lifecycle_lock:
            self._poll_wake_state()
            return self._wake_ready

    @property
    def waiting_for_wake(self) -> bool:
        """Report confirmed sleep until wake and hold setup complete."""
        with self._lifecycle_lock:
            self._poll_wake_state()
            return self._waiting_for_wake

    def _poll_wake_state(self) -> None:
        # Both properties share one pipe and are called under the lifecycle lock.
        while self._wake_ready_fd is not None:
            try:
                marker = os.read(self._wake_ready_fd, 1)
            except BlockingIOError:
                return
            if marker == b"W":
                self._waiting_for_wake = True
                continue
            self._wake_ready = marker == b"1"
            self._waiting_for_wake = False
            os.close(self._wake_ready_fd)
            self._wake_ready_fd = None

    def start(self) -> None:
        """Launch the headset setup script."""
        with self._lifecycle_lock:
            if self._stopped:
                raise RuntimeError(
                    "the managed Quest headset was stopped before starting"
                )
        script = _launcher_script()
        environment = os.environ.copy()
        environment["QUEST_OWNER_PID"] = str(os.getpid())
        environment["QUEST_REQUIRE_PHYSICAL_WAKE"] = (
            "1" if self._require_physical_wake else "0"
        )
        environment.pop("QUEST_WAKE_READY_FD", None)
        if self._adb_executable is not None:
            environment["QUEST_ADB"] = self._adb_executable
        if self.device_serial is not None:
            environment["QUEST_SERIAL"] = self.device_serial
        print("[quest] preparing the headset for browser input", flush=True)
        wake_read_fd = wake_write_fd = None
        try:
            wake_read_fd, wake_write_fd = os.pipe()
            os.set_blocking(wake_read_fd, False)
            environment["QUEST_WAKE_READY_FD"] = str(wake_write_fd)
            # A new session keeps ADB helpers in one process group for cleanup.
            process = subprocess.Popen(
                ["bash", str(script)],
                cwd=script.parent,
                env=environment,
                stdin=subprocess.DEVNULL,
                text=True,
                start_new_session=True,
                pass_fds=() if wake_write_fd is None else (wake_write_fd,),
            )
        except BaseException:
            if wake_read_fd is not None:
                os.close(wake_read_fd)
            raise
        finally:
            if wake_write_fd is not None:
                os.close(wake_write_fd)
        with self._lifecycle_lock:
            if not self._stopped:
                self._process = process
                self._wake_ready_fd = wake_read_fd
                return
        # Stop may win while Popen is creating the child. The starting thread
        # retains cleanup ownership until the child has been published.
        if wake_read_fd is not None:
            os.close(wake_read_fd)
        self._stop_process(process)
        raise RuntimeError("the managed Quest headset was stopped while starting")

    def check_alive(self) -> None:
        """Raise when the managed launcher exits before its owner stops it."""
        with self._lifecycle_lock:
            process = self._process
        if process is None:
            return
        status = process.poll()
        if status is not None:
            raise RuntimeError(
                f"managed Quest launcher exited unexpectedly (status {status})"
            )

    def stop(self) -> None:
        """Terminate the stack, killing the process group if it hangs."""
        with self._lifecycle_lock:
            self._stopped = True
            self._wake_ready = False
            self._waiting_for_wake = False
            if self._wake_ready_fd is not None:
                os.close(self._wake_ready_fd)
                self._wake_ready_fd = None
            process, self._process = self._process, None
        self._stop_process(process)

    @staticmethod
    def _stop_process(process: subprocess.Popen | None) -> None:
        if process is None or process.poll() is not None:
            return
        print("[quest] restoring the headset", flush=True)
        process.terminate()
        try:
            process.wait(timeout=_STOP_GRACE_S)
            return
        except subprocess.TimeoutExpired:
            print("[quest] cleanup timed out; killing the process group", flush=True)
        try:
            os.killpg(int(process.pid), signal.SIGKILL)
        except (AttributeError, OSError):
            process.kill()
        with suppress(subprocess.TimeoutExpired):
            process.wait(timeout=3)
