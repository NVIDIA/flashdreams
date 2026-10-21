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

"""Isaac controller stream server."""

from __future__ import annotations

import socket
import socketserver
import threading
from collections.abc import Callable
from contextlib import suppress
from typing import Any

from flashdreams_controller_input.controller_types import DEFAULT_PORT

_MAX_MESSAGE_BYTES = 64 * 1024
"""Maximum NDJSON record size accepted from the headset."""


class _ThreadingTCPServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True


class ControllerServer:
    """Receives newline-delimited controller messages from the Isaac Teleop bridge.

    Stopping is terminal for this instance and waits for callbacks to return.
    Callbacks must therefore finish promptly and must not call ``stop``.

    Args:
        on_message: Called with every complete message on a connection thread.
        on_disconnect: Called once when a connection that delivered a message
            closes, including during shutdown. Empty receiver probes are ignored.
        host: Interface to bind; loopback keeps the local bridge private.
        port: TCP port, or ``0`` to let the OS choose and read :attr:`port`.
        peer_name: Producer name used in diagnostics.
    """

    def __init__(
        self,
        *,
        on_message: Callable[[str | bytes], None],
        on_disconnect: Callable[[], None] | None = None,
        host: str = "127.0.0.1",
        port: int = DEFAULT_PORT,
        peer_name: str = "Isaac Teleop bridge",
    ) -> None:
        self._on_message = on_message
        self._on_disconnect_callback = on_disconnect
        self._host = str(host)
        self._requested_port = int(port)
        self._peer_name = str(peer_name)
        self._server: _ThreadingTCPServer | None = None
        self._thread: threading.Thread | None = None
        self._port = 0
        self._connections = 0
        self._lock = threading.Lock()
        self._clients: set[socket.socket] = set()
        self._stopped = threading.Event()
        self._lifecycle_lock = threading.Lock()

    @property
    def port(self) -> int:
        """Return the bound host port."""
        return self._port

    @property
    def connections(self) -> int:
        """Return how many controller producer connections have been accepted."""
        with self._lock:
            return self._connections

    @property
    def url(self) -> str:
        """Return the diagnostic address of the bridge receiver."""
        return f"tcp://localhost:{self._port}"

    def start(self) -> None:
        """Bind and serve on background connection threads."""
        owner = self

        class Handler(socketserver.StreamRequestHandler):
            def handle(self) -> None:
                with owner._lock:
                    if owner._stopped.is_set():
                        return
                    owner._clients.add(self.request)
                try:
                    owner._handle(self.rfile)
                finally:
                    with owner._lock:
                        owner._clients.discard(self.request)

        with self._lifecycle_lock:
            if self._stopped.is_set() or self._server is not None:
                return
            server = _ThreadingTCPServer(
                (self._host, self._requested_port),
                Handler,
                bind_and_activate=True,
            )
            # A stop can arrive while binding; it must also cancel late startup.
            if self._stopped.is_set():
                server.server_close()
                return
            self._server = server
            self._port = int(server.server_address[1])
            self._thread = threading.Thread(
                target=server.serve_forever,
                name="controller-server",
                daemon=True,
            )
            self._thread.start()
        print(f"[quest] accepting the {self._peer_name} at {self.url}", flush=True)

    def stop(self) -> None:
        """Cancel startup and join active connections, including their callbacks."""
        self._stopped.set()
        with self._lifecycle_lock:
            server, self._server = self._server, None
            if server is None:
                return
            with self._lock:
                clients = tuple(self._clients)
            for connection in clients:
                with suppress(OSError):
                    connection.shutdown(socket.SHUT_RDWR)
            server.shutdown()
            # Non-daemon handlers are joined after shutdown wakes idle reads.
            server.server_close()
            thread, self._thread = self._thread, None
            if thread is not None:
                thread.join()

    def _handle(self, stream: Any) -> None:
        """Pump one controller connection into the message callback."""
        announced = False
        try:
            while not self._stopped.is_set() and (
                line := stream.readline(_MAX_MESSAGE_BYTES + 1)
            ):
                if self._stopped.is_set():
                    break
                if len(line) > _MAX_MESSAGE_BYTES:
                    self._discard_record_tail(stream, line)
                    if not self._stopped.is_set():
                        self._on_message(b"{oversized controller message")
                    continue
                message = line.rstrip(b"\r\n")
                if message:
                    if not announced:
                        with self._lock:
                            self._connections += 1
                        print(f"[quest] the {self._peer_name} connected", flush=True)
                        announced = True
                    self._on_message(message)
        except (ConnectionError, OSError):
            pass
        finally:
            if announced:
                print(f"[quest] the {self._peer_name} disconnected", flush=True)
                if self._on_disconnect_callback is not None:
                    self._on_disconnect_callback()

    @staticmethod
    def _discard_record_tail(stream: Any, first: bytes) -> None:
        """Consume the remainder of an oversized NDJSON record."""
        if first.endswith(b"\n"):
            return
        while tail := stream.readline(_MAX_MESSAGE_BYTES + 1):
            if tail.endswith(b"\n"):
                return
