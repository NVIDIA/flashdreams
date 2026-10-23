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

"""Laptop WebXR companion for authenticated remote Isaac controller input."""

from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import ssl
import threading
import time
from collections import deque
from contextlib import suppress
from pathlib import Path
from typing import TYPE_CHECKING

from flashdreams_controller_input.remote_quest import (
    IO_TIMEOUT_S,
    MAX_BATCH_MESSAGES,
    MAX_FRAME_AGE_S,
    MAX_PENDING_FRAMES,
    REMOTE_PORT,
    encode_record,
    read_record,
    validate_token,
)

if TYPE_CHECKING:
    from flashdreams_controller_input.quest_webxr import QuestWebXRProducer


class QuestRelay:
    """Forward raw paired messages while owning the laptop headset lifecycle.

    The server pulls bounded batches, so a stalled network cannot accumulate an
    unbounded TCP replay. Any network failure closes the local producer; rerun the
    companion to reconnect without restarting the DGX application.
    """

    def __init__(
        self,
        *,
        server: str,
        port: int = REMOTE_PORT,
        ca_cert: str | Path,
        token: str,
        webxr: QuestWebXRProducer,
    ) -> None:
        validate_token(token)
        if not server or not 1 <= port <= 65535:
            raise ValueError("remote controller address or port is invalid")
        self._context = ssl.create_default_context(cafile=ca_cert)
        self._context.minimum_version = ssl.TLSVersion.TLSv1_2
        self._server_host = server
        self._port = port
        self._token = token
        self._condition = threading.Condition()
        self._pending: deque[tuple[float, str]] = deque()
        self._hello: str | None = None
        self._disconnected = False
        self._producer_generation = -1
        self._stopped = threading.Event()
        self._connection: socket.socket | None = None
        self._webxr = webxr

    def run(self) -> None:
        """Authenticate before starting input and restore it on every exit path."""
        try:
            if self._stopped.is_set():
                return
            raw = socket.create_connection(
                (self._server_host, self._port), timeout=IO_TIMEOUT_S
            )
            self._connection = raw
            if self._stopped.is_set():
                return
            raw.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            connection = self._context.wrap_socket(
                raw, server_hostname=self._server_host
            )
            self._connection = connection
            if self._stopped.is_set():
                return
            with connection.makefile("rb") as stream:
                connection.sendall(
                    encode_record({"type": "auth", "version": 1, "token": self._token})
                )
                try:
                    ready = read_record(stream)
                except EOFError:
                    raise ValueError(
                        "remote controller authentication failed; check the shared token"
                    ) from None
                if ready != {"type": "ready"}:
                    raise ValueError("remote controller authentication failed")
                if self._stopped.is_set():
                    return
                self._webxr.start(self._consume, self._disconnect)
                print(
                    "[quest-relay] authenticated; waiting for controller input (Ctrl-C to stop)",
                    flush=True,
                )
                while not self._stopped.is_set():
                    self._webxr.check_alive()
                    request = read_record(stream)
                    if (
                        request.get("type") != "pull"
                        or type(request.get("seq")) is not int
                    ):
                        raise ValueError("unexpected remote controller request")
                    connection.sendall(encode_record(self._batch(request["seq"])))
        finally:
            self.stop()
            self._webxr.stop()

    def stop(self) -> None:
        """Wake network and producer waits so ``run`` can restore the headset."""
        self._stopped.set()
        with self._condition:
            self._condition.notify_all()
        connection = self._connection
        if connection is not None:
            with suppress(OSError):
                connection.shutdown(socket.SHUT_RDWR)
            connection.close()

    def _consume(self, raw: str | bytes) -> None:
        # Only inspect the envelope here; FrameStore on the DGX validates and
        # converts each frame exactly once, preserving its producer clock.
        try:
            message = raw.decode("utf-8") if isinstance(raw, bytes) else raw
        except UnicodeDecodeError:
            message = "{invalid UTF-8 controller record"
        with self._condition:
            if self._stopped.is_set() or not self._wake_ready():
                return
            try:
                payload = json.loads(message)
                if isinstance(payload, dict) and payload.get("type") == "hello":
                    self._producer_generation += 1
                    self._hello = message
                    self._pending.clear()
                    self._disconnected = True
            except (ValueError, RecursionError):
                pass  # Preserve malformed records for the receiver's validation.
            if len(self._pending) >= MAX_PENDING_FRAMES:
                self._pending.clear()
                self._disconnected = True
                if self._hello is not None:
                    self._pending.append((time.monotonic(), self._hello))
            self._pending.append((time.monotonic(), message))
            self._condition.notify_all()

    def _wake_ready(self) -> bool:
        return self._webxr.wake_ready

    def _disconnect(self) -> None:
        with self._condition:
            self._pending.clear()
            self._hello = None
            self._disconnected = True
            self._condition.notify_all()

    def _batch(self, sequence: int) -> dict:
        with self._condition:
            if not self._pending and not self._stopped.is_set():
                self._condition.wait(timeout=0.01)
            now = time.monotonic()
            dropped = False
            while self._pending and now - self._pending[0][0] > MAX_FRAME_AGE_S:
                self._pending.popleft()
                dropped = True
            if dropped:
                self._disconnected = True
                if self._hello is not None:
                    self._pending.appendleft((now, self._hello))
            result = {
                "type": "batch",
                "seq": sequence,
                "disconnected": self._disconnected,
                "generation": max(0, self._producer_generation),
                "messages": [],
            }
            # ponytail: repeated encoding is O(n²), capped at 32 records; use an
            # incremental encoded-byte budget if larger batches become necessary.
            for _ in range(min(MAX_BATCH_MESSAGES, len(self._pending))):
                result["messages"].append(self._pending[0][1])
                try:
                    encode_record(result)
                except ValueError:
                    result["messages"].pop()
                    if not result["messages"]:
                        raise ValueError(
                            "controller record does not fit a remote batch"
                        ) from None
                    break
                self._pending.popleft()
            self._disconnected = False
            return result


def main() -> None:
    """Run laptop controller input without loading FlashDreams or a model."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--server", required=True, help="DGX address matching its TLS certificate"
    )
    parser.add_argument("--port", type=int, default=REMOTE_PORT)
    parser.add_argument(
        "--ca-cert", required=True, help="Trusted DGX certificate or issuing CA PEM"
    )
    parser.add_argument("--quest-device", default=os.environ.get("ANDROID_SERIAL"))
    parser.add_argument(
        "--require-physical-wake",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Confirm headset sleep, then wait for the user to wake it before starting input (default: enabled).",
    )
    from flashdreams_controller_input.quest_webxr import add_relay_arguments

    add_relay_arguments(parser)
    args = parser.parse_args()
    try:
        from flashdreams_controller_input.quest_webxr import QuestWebXRProducer

        webxr = QuestWebXRProducer(args)
        relay = QuestRelay(
            server=args.server,
            port=args.port,
            ca_cert=args.ca_cert,
            token=os.environ.get("FLASHDREAMS_CONTROLLER_TOKEN", ""),
            webxr=webxr,
        )
    except (OSError, ValueError) as error:
        parser.error(str(error))
    previous = signal.signal(signal.SIGTERM, lambda *_: relay.stop())
    try:
        relay.run()
    except KeyboardInterrupt:
        pass
    except EOFError:
        print("[quest-relay] DGX connection closed; headset restored", flush=True)
    except (OSError, ValueError, RuntimeError) as error:
        raise SystemExit(
            f"Quest relay stopped: {error}; rerun the laptop command to reconnect"
        ) from error
    finally:
        relay.stop()
        signal.signal(signal.SIGTERM, previous)


if __name__ == "__main__":
    main()
