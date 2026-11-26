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

"""Authenticated remote Quest input with a validated Isaac bridge contract."""

from __future__ import annotations

import hmac
import json
import math
import secrets
import socket
import ssl
import threading
import time
import weakref
from collections import deque
from collections.abc import Callable, Iterator
from contextlib import contextmanager, suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO

from flashdreams_controller_input.controller_types import (
    ISAAC_TELEOP_BRIDGE_PROTOCOL_VERSION,
    SUPPORTED_ISAAC_TELEOP_VERSION,
    Observation,
)
from flashdreams_controller_input.frames import SOURCE_CLIENT_CAPTURE_NS_KEY
from flashdreams_controller_input.pose import (
    DEFAULT_GRIP_TO_WRIST_OFFSET_M,
    OPENXR_STAGE_GRIP_POSE_CONVENTION,
    WRIST_PROXY_POSE_CONVENTION,
)
from flashdreams_controller_input.quest_source import FrameStore

REMOTE_PORT = 8445
MAX_RECORD_BYTES = 256 * 1024
"""Maximum encoded TLS record, including the JSON-escaped controller messages."""

MAX_BATCH_MESSAGES = 32
"""Maximum controller records per pull, limiting memory and receive work."""

MAX_PENDING_FRAMES = 256
"""Queue ceiling on both hosts; overflow inserts a tracking-loss boundary."""

MAX_FRAME_AGE_S = 0.25
"""Maximum laptop queue residence before a controller message is discarded."""

MAX_ROUND_TRIP_S = 0.5
"""Maximum pull round trip and DGX queue residence, measured on the DGX clock."""

IO_TIMEOUT_S = 3.0
"""Bound authentication and idle reads; slow replies are rejected before ingestion."""

CODE_LIFETIME_S = 5 * 60
"""An unclaimed viewer code expires; a claimed laptop stays paired until revoked."""

MAX_CODE_ATTEMPTS = 5
"""Wrong entries before a code is voided, so six symbols resist guessing."""

_CODE_ALPHABET = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"
"""Crockford base32: no I, L, O or U, so a code survives being read aloud."""


@dataclass(eq=False)
class _ProducerAuthority:
    viewer: object
    key: str = field(repr=False)
    epoch: int
    """Bumped at each session boundary so earlier-session poses lose authority."""


@dataclass(eq=False)
class _PairingCode:
    viewer: object
    code: str = field(repr=False)
    expires_at: float
    attempts_left: int = MAX_CODE_ATTEMPTS


def _normalize_code(value: str) -> str:
    """Accept lower case, spaces, dashes and the digits' look-alike letters."""
    value = value.upper().translate(str.maketrans("OIL", "011", " -"))
    return value[:16]


def validate_token(token: str, *, name: str = "FLASHDREAMS_CONTROLLER_TOKEN") -> bytes:
    """Require a nonempty, sufficiently long shared secret without printing it."""
    if not isinstance(token, str) or not 32 <= len(token) <= 256:
        raise ValueError(f"{name} must contain 32–256 characters")
    return token.encode("utf-8")


def _reject_constant(_value: str) -> None:
    raise ValueError("remote controller JSON contains a non-finite number")


def read_record(stream: BinaryIO) -> dict:
    """Read one bounded JSON object, failing closed on truncation or excess data."""
    line = stream.readline(MAX_RECORD_BYTES + 1)
    if not line:
        raise EOFError("remote controller connection closed")
    if len(line) > MAX_RECORD_BYTES or not line.endswith(b"\n"):
        raise ValueError("remote controller record is oversized or truncated")
    value = json.loads(line, parse_constant=_reject_constant)
    if not isinstance(value, dict):
        raise ValueError("remote controller record must be an object")
    return value


def encode_record(value: dict) -> bytes:
    """Encode a bounded NDJSON record without allowing non-finite numbers."""
    data = (json.dumps(value, separators=(",", ":"), allow_nan=False) + "\n").encode()
    if len(data) > MAX_RECORD_BYTES:
        raise ValueError("remote controller record is oversized")
    return data


@dataclass(frozen=True, slots=True)
class _AuthorityCheck:
    """One epoch check without retaining a receiver or its pose history."""

    source: weakref.ReferenceType[RemoteQuestSource]
    """Weak reference to the process-local authority owner."""
    epoch: int
    """Issued authorization epoch; a later session or pairing never revives it."""

    def __call__(self) -> bool:
        source = self.source()
        if source is None:
            return False
        with source.input_guard():
            return source._authorized_locked(self.epoch)


class RemoteQuestSource:
    """Receive one explicitly paired laptop's raw controller pairs over TLS.

    The viewer shows a short code (P); the laptop that enters it receives a
    reconnect key and stays paired across scenes, viewer reloads and network
    drops until R, another laptop claims a new code, or the demo stops. The
    source owns no ADB state. Network disconnects produce tracking-loss
    boundaries; reconnects preserve frame identity and the headset capture clock.
    A reset capture clock requires restarting the demo and recalibrating, or a
    deliberate new pairing for a replacement producer and tracking origin.
    The bridge handshake fixes the timestamp units and pose convention; Isaac
    wrist-proxy poses are already retargeted and must not receive another offset.
    """

    name = "remote-quest"

    def __init__(
        self,
        *,
        host: str = "0.0.0.0",
        port: int = REMOTE_PORT,
        certfile: str | Path,
        keyfile: str | Path,
        token: str,
        wrist_offset_m: float = DEFAULT_GRIP_TO_WRIST_OFFSET_M,
    ) -> None:
        self._token = validate_token(token)
        if not host or not 0 <= port <= 65535:
            raise ValueError("remote controller host and port are invalid")
        if not math.isfinite(wrist_offset_m) or wrist_offset_m < 0.0:
            raise ValueError(
                "Isaac Teleop wrist offset must be finite and non-negative"
            )
        self._context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        self._context.minimum_version = ssl.TLSVersion.TLSv1_2
        self._context.load_cert_chain(certfile, keyfile)
        self._host = host
        self._port = port
        self._capture_scale_ns = 1_000_000_000
        self._expected_sdk_version = SUPPORTED_ISAAC_TELEOP_VERSION
        self._store = FrameStore(
            expected_protocol=ISAAC_TELEOP_BRIDGE_PROTOCOL_VERSION,
            expected_pose_convention=WRIST_PROXY_POSE_CONVENTION,
            expected_hello_fields={
                "mode": "cloudxr-webxr",
                "input_pose_convention": OPENXR_STAGE_GRIP_POSE_CONVENTION,
                "output_pose_convention": WRIST_PROXY_POSE_CONVENTION,
                "wrist_offset_m": float(wrist_offset_m),
            },
            peer_name="remote Isaac Teleop bridge",
            compatibility_hint="restart the laptop bridge with matching Isaac settings",
        )
        self._pending: deque[Observation] = deque()
        self._lock = threading.RLock()
        self._viewer: object | None = None
        self._authority: _ProducerAuthority | None = None
        self._code: _PairingCode | None = None
        self._authority_epoch = 0
        self._clear_input: Callable[[], None] | None = None
        self._lifecycle_lock = threading.Lock()
        self._stopped = threading.Event()
        self._listener: socket.socket | None = None
        self._connection: socket.socket | None = None
        self._thread: threading.Thread | None = None
        self._generation = 0
        self._ever_received = False
        self._last_capture_ns = -1
        self._failure: RuntimeError | None = None

    def register_input_clear(self, callback: Callable[[], None]) -> None:
        """Clear the window's queued poses under the same authority fence."""
        with self._lock:
            self._clear_input = callback

    def set_viewer(self, viewer: object | None) -> None:
        """Follow the server-observed viewer, never producer metadata.

        No viewer pauses remote input; a different viewer revokes the pairing.
        """
        with self._lock:
            self._viewer = viewer
            if viewer is None:
                self._boundary_locked()
            elif any(
                grant is not None and grant.viewer is not viewer
                for grant in (self._authority, self._code)
            ):
                self._invalidate_locked()

    def begin_session(self) -> None:
        """Keep the paired laptop across scenes; earlier poses lose authority."""
        with self._lock:
            self._boundary_locked()
            if self._authority is not None:
                self._authority_epoch += 1
                self._authority.epoch = self._authority_epoch

    def pair(self) -> str:
        """End the current pairing and return a new code to show in the viewer."""
        with self._lock:
            if self._viewer is None or self._stopped.is_set():
                raise ValueError("remote pairing requires a connected viewer")
            # One relay is served at a time: the next laptop needs the slot now.
            self._invalidate_locked()
            code = "".join(secrets.choice(_CODE_ALPHABET) for _ in range(6))
            self._code = _PairingCode(
                self._viewer, code, time.monotonic() + CODE_LIFETIME_S
            )
            return f"{code[:3]}-{code[3:]}"

    @property
    def pairing_code(self) -> str:
        """Return the unclaimed code for the viewer to display, or ``""``."""
        with self.input_guard():
            code = self._code
            return "" if code is None else f"{code.code[:3]}-{code.code[3:]}"

    def revoke(self) -> None:
        """Revoke connected and reconnecting producers and discard queued poses."""
        with self._lock:
            self._invalidate_locked()

    def _invalidate_locked(self) -> None:
        self._authority_epoch += 1
        self._authority = None
        self._code = None
        self._boundary_locked()
        if self._clear_input is not None:
            self._clear_input()
        if self._connection is not None:
            with suppress(OSError):
                self._connection.shutdown(socket.SHUT_RDWR)

    def _expire_locked(self) -> None:
        if self._code is not None and time.monotonic() >= self._code.expires_at:
            self._code = None

    def _claim_locked(self, request: dict) -> _ProducerAuthority:
        """Accept the current key, or exchange the viewer's code for a new key."""
        key, code = request.get("key"), request.get("code")
        authority = self._authority
        if isinstance(key, str):
            if authority is not None and hmac.compare_digest(
                key.encode("utf-8"), authority.key.encode()
            ):
                return authority
            raise ValueError("remote controller key is no longer paired")
        pending = self._code
        if not isinstance(code, str) or pending is None:
            raise ValueError("no laptop code is waiting; press P in the viewer")
        if pending.viewer is not self._viewer:
            raise ValueError("laptop code belongs to a disconnected viewer")
        if not hmac.compare_digest(
            _normalize_code(code).encode("utf-8"), pending.code.encode()
        ):
            pending.attempts_left -= 1
            if pending.attempts_left <= 0:
                self._code = None
                raise ValueError("laptop code voided; press P for a new code")
            raise ValueError(
                f"laptop code rejected; {pending.attempts_left} attempts left"
            )
        # P already revoked any previous laptop, so a code never coexists
        # with a key; a fresh epoch keeps earlier events from reviving.
        self._code = None
        self._authority_epoch += 1
        self._authority = _ProducerAuthority(
            pending.viewer, secrets.token_urlsafe(32), self._authority_epoch
        )
        self._last_capture_ns = -1
        print(
            "[quest] laptop paired; it stays paired until R or a new code", flush=True
        )
        return self._authority

    @contextmanager
    def input_guard(self) -> Iterator[None]:
        """Serialize pose enqueue/delivery with revoke, viewer switch and expiry."""
        with self._lock:
            self._expire_locked()
            yield

    def allows(self, observation: Observation) -> bool:
        """Recheck remote observations even after they leave the transport queue."""
        with self.input_guard():
            epoch = observation.authority_epoch
            return epoch is None or self._authorized_locked(epoch)

    def _authorized_locked(self, epoch: int) -> bool:
        # Paused while the paired viewer is away; the reconnect key stays valid.
        authority = self._authority
        return (
            authority is not None
            and authority.epoch == epoch
            and self._viewer is authority.viewer
        )

    def authority_check(self, epoch: int) -> Callable[[], bool]:
        """Return a weak check that retains only one issued epoch."""
        return _AuthorityCheck(weakref.ref(self), epoch)

    def _require_authority_locked(self, grant: _ProducerAuthority) -> None:
        if self._authority is not grant:
            raise ValueError("remote controller pairing ended")

    @property
    def port(self) -> int:
        """Return the listening port after startup."""
        return self._port

    def start(self) -> None:
        """Start a bounded TLS listener without waiting for a laptop or controller."""
        with self._lifecycle_lock:
            if self._stopped.is_set():
                raise RuntimeError("remote Quest source was stopped before starting")
            if self._listener is not None:
                return
            listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            try:
                listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                listener.bind((self._host, self._port))
                listener.listen(4)
                listener.settimeout(0.2)
            except BaseException:
                listener.close()
                raise
            self._listener = listener
            self._port = listener.getsockname()[1]
            self._thread = threading.Thread(
                target=self._serve, args=(listener,), name="remote-quest", daemon=True
            )
            self._thread.start()
        print(
            f"[quest] remote controller TLS listener on {self._host}:{self.port}; "
            "in the viewer: P shows a laptop code, R revokes",
            flush=True,
        )

    def stop(self) -> None:
        """Close the producer connection and join the receiver exactly once."""
        self._stopped.set()
        self.revoke()
        with self._lifecycle_lock:
            listener, self._listener = self._listener, None
            with self._lock:
                connection = self._connection
            for stream in (connection, listener):
                if stream is not None:
                    with suppress(OSError):
                        stream.shutdown(socket.SHUT_RDWR)
                    stream.close()
            thread, self._thread = self._thread, None
            if thread is not None:
                thread.join()
            with self._lock:
                self._pending.clear()

    def poll(self) -> Observation | None:
        """Return the next fresh observation or explicit disconnect boundary."""
        with self.input_guard():
            if self._failure is not None:
                raise self._failure
            if not self._pending:
                return None
            now = time.perf_counter_ns()
            if now - self._pending[0].capture_ns > MAX_ROUND_TRIP_S * 1e9:
                self._pending.clear()
                return Observation(None, now, self.name, self._generation)
            return self._pending.popleft()

    def _boundary_locked(self) -> None:
        self._pending.clear()
        self._pending.append(
            Observation(None, time.perf_counter_ns(), self.name, self._generation)
        )

    def _boundary(self, grant: _ProducerAuthority) -> None:
        with self._lock:
            self._require_authority_locked(grant)
            self._boundary_locked()

    def _serve(self, listener: socket.socket) -> None:
        # One connection is handled at a time, bounding unauthenticated work and
        # preventing two producers from interleaving a participant's poses.
        while not self._stopped.is_set():
            connection = None
            grant: _ProducerAuthority | None = None
            try:
                raw, _ = listener.accept()
                connection = raw
                raw.settimeout(IO_TIMEOUT_S)
                raw.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                with self._lock:
                    self._connection = raw
                if self._stopped.is_set():
                    break
                connection = self._context.wrap_socket(
                    raw, server_side=True, do_handshake_on_connect=False
                )
                with self._lock:
                    self._connection = connection
                if self._stopped.is_set():
                    break
                connection.do_handshake()
                with connection.makefile("rb") as stream:
                    request = read_record(stream)
                    supplied = request.get("token")
                    # The SSH-delivered token gates who may try codes at all.
                    if not (
                        request.get("type") == "auth"
                        and type(request.get("version")) is int
                        and request["version"] == 3
                        and isinstance(supplied, str)
                        and hmac.compare_digest(supplied.encode("utf-8"), self._token)
                    ):
                        raise ValueError("remote controller authentication rejected")
                    try:
                        with self.input_guard():
                            grant = self._claim_locked(request)
                    except ValueError:
                        connection.sendall(encode_record({"type": "rejected"}))
                        raise
                    connection.sendall(
                        encode_record({"type": "ready", "key": grant.key})
                    )
                    self._boundary(grant)
                    self._receive(connection, stream, grant)
            except socket.timeout:
                pass
            except (
                OSError,
                EOFError,
                ValueError,
                OverflowError,
                RecursionError,
            ) as error:
                if not self._stopped.is_set():
                    # Messages are server-authored; escape and bound any
                    # producer-derived text they quote.
                    reason = str(error) or type(error).__name__
                    reason = reason.encode("ascii", "backslashreplace").decode()
                    print(
                        f"[quest] remote controller connection ended: {reason[:200]}",
                        flush=True,
                    )
            finally:
                if connection is not None:
                    connection.close()
                with self._lock:
                    self._connection = None
                if grant is not None and not self._stopped.is_set():
                    with self._lock:
                        if self._authority is grant:
                            self._boundary_locked()

    def _receive(
        self, connection: ssl.SSLSocket, stream: BinaryIO, grant: _ProducerAuthority
    ) -> None:
        sequence = 0
        hello_seen = False
        producer_generation = None
        while not self._stopped.is_set():
            with self._lock:
                self._require_authority_locked(grant)
                epoch = grant.epoch
            sent = time.monotonic()
            connection.sendall(encode_record({"type": "pull", "seq": sequence}))
            reply = read_record(stream)
            if time.monotonic() - sent > MAX_ROUND_TRIP_S:
                raise ValueError(
                    "remote controller round trip exceeded freshness limit; restart the laptop relay"
                )
            if (
                reply.get("type") != "batch"
                or type(reply.get("seq")) is not int
                or reply["seq"] != sequence
            ):
                raise ValueError("remote controller reply is out of sequence")
            messages = reply.get("messages")
            if not isinstance(messages, list) or len(messages) > MAX_BATCH_MESSAGES:
                raise ValueError("remote controller batch is malformed")
            if type(reply.get("disconnected")) is not bool:
                raise ValueError("remote controller batch requires a disconnect flag")
            generation = reply.get("generation")
            if type(generation) is not int or generation < 0:
                raise ValueError(
                    "remote controller batch requires a controller generation"
                )
            new_producer = (
                producer_generation is None or generation > producer_generation
            )
            if producer_generation is not None and generation < producer_generation:
                raise ValueError("remote controller generation went backwards")
            if new_producer:
                hello_seen = False
            if reply["disconnected"]:
                self._boundary(grant)
                hello_seen = False
            for message in messages:
                if not isinstance(message, str) or len(message.encode()) > 64 * 1024:
                    raise ValueError(
                        "remote controller record is malformed or oversized"
                    )
                payload = json.loads(message, parse_constant=_reject_constant)
                kind = payload.get("type") if isinstance(payload, dict) else None
                if kind not in ("hello", "status", "xr_frame"):
                    raise ValueError("unexpected remote controller message type")
                if kind == "hello":
                    version = payload.get("version")
                    if not isinstance(version, str) or (
                        version != self._expected_sdk_version
                        and not version.startswith(f"{self._expected_sdk_version}+")
                    ):
                        raise ValueError(
                            "remote Isaac Teleop bridge requires SDK "
                            f"{self._expected_sdk_version}"
                        )
                if kind == "xr_frame" and (not hello_seen or "t_client" not in payload):
                    raise ValueError(
                        "remote controller frame requires a handshake and capture timestamp"
                    )
                if kind == "xr_frame":
                    timestamp = payload["t_client"]
                    if (
                        type(timestamp) not in (int, float)
                        or not math.isfinite(timestamp)
                        or not 0 <= timestamp <= (2**64 - 1) / self._capture_scale_ns
                    ):
                        raise ValueError(
                            "remote controller frame has an invalid capture timestamp"
                        )
                    capture_ns = int(round(float(timestamp) * self._capture_scale_ns))
                    if capture_ns > 2**64 - 1:
                        raise ValueError(
                            "remote controller capture timestamp exceeds uint64"
                        )
                    if capture_ns == self._last_capture_ns:
                        continue
                    controllers = payload.get("controllers")
                    if isinstance(controllers, dict):
                        for side in ("left", "right"):
                            if isinstance(controller := controllers.get(side), dict):
                                for key in ("trigger", "grip"):
                                    value = controller.get(key, 0.0)
                                    if (
                                        type(value) not in (int, float)
                                        or not 0 <= value <= 1
                                    ):
                                        raise ValueError(
                                            "remote controller analog values must be finite numbers in [0, 1]"
                                        )
                with self._lock:
                    self._require_authority_locked(grant)
                    rejected = self._store.rejected
                    self._store.consume(message)
                if self._store.rejected != rejected:
                    raise ValueError(
                        "remote Quest rejected a malformed controller frame"
                    )
                if error := self._store.compatibility_error():
                    raise ValueError(error)
                if kind == "hello":
                    if new_producer:
                        with self._lock:
                            self._require_authority_locked(grant)
                            if self._ever_received:
                                self._generation += 1
                        self._boundary(grant)
                        producer_generation = generation
                        new_producer = False
                    hello_seen = True
                if kind == "xr_frame":
                    self._accept_frame(capture_ns, grant, epoch)
            sequence += 1

    def _accept_frame(
        self, capture_ns: int, grant: _ProducerAuthority, epoch: int
    ) -> None:
        with self._lock:
            self._require_authority_locked(grant)
            # A reply to a pull sent before a scene change carries old-scene poses.
            if grant.epoch == epoch:
                self._accept_frame_locked(capture_ns, epoch)

    def _accept_frame_locked(self, capture_ns: int, epoch: int) -> None:
        frame = self._store.snapshot()
        assert frame is not None
        # The remote-vr relay owns a physical ADB headset, never a desktop emulator.
        frame["is_emulated"] = False
        # Keep producer identity distinct from DGX receive-time freshness.
        frame[SOURCE_CLIENT_CAPTURE_NS_KEY] = capture_ns
        for side in ("left", "right"):
            if controller := frame.get(side):
                if not all(
                    math.isfinite(controller[key]) for key in ("trigger", "grip")
                ):
                    raise ValueError(
                        "remote controller frame has non-finite controller values"
                    )
                if math.hypot(*controller["orientation"]) < 1e-8:
                    raise ValueError(
                        "remote controller frame has an invalid quaternion"
                    )
        if capture_ns == self._last_capture_ns:
            return
        if capture_ns < self._last_capture_ns:
            with self._lock:
                self._failure = RuntimeError(
                    "remote Quest capture clock restarted or went backwards; "
                    "restart the demo and recalibrate before reconnecting"
                )
            raise ValueError("remote Quest capture clock is not increasing")
        self._last_capture_ns = capture_ns
        observation = Observation(
            frame, frame["client_capture_ns"], self.name, self._generation, epoch
        )
        with self._lock:
            self._ever_received = True
            if len(self._pending) >= MAX_PENDING_FRAMES:
                self._pending.clear()
                self._pending.append(
                    Observation(
                        None, observation.capture_ns, self.name, self._generation
                    )
                )
            self._pending.append(observation)
