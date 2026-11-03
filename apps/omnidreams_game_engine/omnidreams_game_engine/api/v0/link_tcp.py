# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TCP transport: the host listens, one engine connects out."""

from __future__ import annotations

import contextlib
import logging
import queue
import socket
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from omnidreams_game_engine.api.v0.framing import (
    encode_message,
    recv_message,
    send_message,
)
from omnidreams_game_engine.api.v0.messages import (
    REFUSAL_CODES,
    Error,
    Hello,
    Message,
    Ping,
    ProtocolError,
    Status,
    check_api,
    from_header,
)
from omnidreams_game_engine.api.v0.recorder import TX, PacketRecorder, now_us
from omnidreams_game_engine.api.v0.source import (
    LOST_TYPE,
    Inbound,
    MessageEngineSource,
)
from omnidreams_game_engine.api.v0.state import StateDecoder

_LOGGER = logging.getLogger(__name__)

HELLO_TIMEOUT_S = 10.0
_ACCEPT_POLL_S = 0.2
_JOIN_TIMEOUT_S = 2.0


@dataclass(slots=True)
class _Connection:
    sock: socket.socket
    session: str
    last_rx: float = field(default_factory=time.monotonic)
    outbox: queue.Queue[tuple[dict[str, Any], bytes] | None] = field(
        default_factory=queue.Queue
    )
    closed: threading.Event = field(default_factory=threading.Event)
    close_reason: str = ""
    threads: list[threading.Thread] = field(default_factory=list)

    def close(self, reason: str) -> None:
        if self.closed.is_set():
            return
        self.close_reason = self.close_reason or reason
        self.closed.set()
        self.outbox.put(None)
        with contextlib.suppress(OSError):
            self.sock.shutdown(socket.SHUT_RDWR)
        _close_quietly(self.sock)


class TcpEngineLink(MessageEngineSource):
    """An :class:`~.external.EngineSource` serving one live engine over TCP.

    The link listens on ``host`` and ``port``. ``host`` defaults to
    ``127.0.0.1``, so only an engine on the same machine can connect. The
    protocol has no authentication: pass ``host`` explicitly (for example
    ``"0.0.0.0"``) to accept an engine on another machine, and only on a
    trusted network. ``port=0`` picks a free port; :attr:`address` reports it.

    Each connection has a reader thread, which only queues messages, and a
    writer thread; a monitor thread pings the engine every
    ``ping_interval_s``, sends the host's status and drops an engine that has
    been silent for ``silence_s``.
    """

    def __init__(
        self,
        decoder: StateDecoder,
        *,
        host: str = "127.0.0.1",
        port: int = 0,
        recorder: PacketRecorder | None = None,
        ping_interval_s: float = 0.5,
        silence_s: float = 10.0,
        status_interval_s: float = 1.0,
        status_provider: Callable[[], Mapping[str, Any]] | None = None,
    ) -> None:
        super().__init__(decoder, recorder=recorder)
        self.ping_interval_s = float(ping_interval_s)
        self.silence_s = float(silence_s)
        self.status_interval_s = float(status_interval_s)
        self.status_provider = status_provider
        self.connects = 0
        self.refusals: list[str] = []
        self._inbox: queue.Queue[Inbound] = queue.Queue()
        self._lock = threading.Lock()
        self._connection: _Connection | None = None
        self._stop = threading.Event()
        self._listener = socket.create_server((host, port))
        self._listener.settimeout(_ACCEPT_POLL_S)
        self._threads = [
            threading.Thread(
                target=self._accept_loop, name="engine-link-accept", daemon=True
            ),
            threading.Thread(
                target=self._monitor_loop, name="engine-link-monitor", daemon=True
            ),
        ]
        for thread in self._threads:
            thread.start()

    @property
    def address(self) -> tuple[str, int]:
        """Host and port the link listens on."""
        host, port = self._listener.getsockname()[:2]
        return str(host), int(port)

    def drop_connection(self, reason: str = "host") -> None:
        """Close the engine's connection, as a network failure would."""
        with self._lock:
            connection = self._connection
        if connection is not None:
            connection.close(reason)

    def close(self) -> None:
        self._stop.set()
        self.drop_connection("host closed")
        _close_quietly(self._listener)
        for thread in self._threads:
            thread.join(timeout=_JOIN_TIMEOUT_S)
        with self._lock:
            connection = self._connection
        if connection is not None:
            for thread in connection.threads:
                thread.join(timeout=_JOIN_TIMEOUT_S)
        super().close()

    ## Transport

    def _fetch(self, timeout_s: float) -> list[Inbound]:
        items: list[Inbound] = []
        try:
            if timeout_s > 0.0:
                items.append(self._inbox.get(timeout=timeout_s))
            else:
                items.append(self._inbox.get_nowait())
        except queue.Empty:
            return items
        while True:
            try:
                items.append(self._inbox.get_nowait())
            except queue.Empty:
                return items

    def _transmit(self, message: Message, payload: bytes) -> None:
        with self._lock:
            connection = self._connection
        if connection is not None and not connection.closed.is_set():
            connection.outbox.put((message.to_header(), payload))

    ## Threads

    def _accept_loop(self) -> None:
        while not self._stop.is_set():
            try:
                sock, peer = self._listener.accept()
            except TimeoutError:
                continue
            except OSError:
                if self._stop.is_set():
                    return
                continue
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            thread = threading.Thread(
                target=self._serve,
                args=(sock, f"{peer[0]}:{peer[1]}"),
                name="engine-link-reader",
                daemon=True,
            )
            thread.start()

    def _serve(self, sock: socket.socket, peer: str) -> None:
        connection = self._handshake(sock, peer)
        if connection is None:
            return
        try:
            while not connection.closed.is_set():
                header, payload = recv_message(connection.sock)
                connection.last_rx = time.monotonic()
                self._inbox.put(Inbound(header, payload, now_us()))
        except Exception as error:  # any read error ends the connection
            reason = connection.close_reason or f"connection: {error}"
        else:
            reason = connection.close_reason or "closed"
        connection.close(reason)
        # Under the lock, so that a reconnect's hello never precedes this LOST.
        with self._lock:
            if self._connection is connection:
                self._connection = None
                if not self._stop.is_set():
                    self._inbox.put(
                        Inbound({"type": LOST_TYPE, "reason": reason}, b"", now_us())
                    )

    def _handshake(self, sock: socket.socket, peer: str) -> _Connection | None:
        sock.settimeout(HELLO_TIMEOUT_S)
        try:
            header, payload = recv_message(sock)
            arrived = now_us()
            message = from_header(header)
            if not isinstance(message, Hello):
                raise ProtocolError("The first message must be a hello")
            check_api(message.api)
            expected = self.decoder.state_format
            if message.state_format is not None and message.state_format != expected:
                raise ProtocolError(
                    f"State format {message.state_format!r} is not {expected!r}",
                    code="state_format",
                )
        except ProtocolError as error:
            self._refuse(sock, error.code, str(error))
            return None
        except Exception as error:  # a failed handshake drops the socket
            _LOGGER.warning("Engine handshake from %s failed: %s", peer, error)
            _close_quietly(sock)
            return None
        sock.settimeout(None)
        connection = _Connection(sock=sock, session=message.session)
        with self._lock:
            previous = self._connection
            busy = (
                previous is not None
                and not previous.closed.is_set()
                and previous.session != message.session
            )
            accepted = not busy and not self._stop.is_set()
            if accepted:
                self._connection = connection
                self.connects += 1
                self._inbox.put(Inbound(header, payload, arrived))
        if busy:
            assert previous is not None
            self._refuse(
                sock, "busy", f"Engine session {previous.session!r} is connected"
            )
            return None
        if not accepted:
            _close_quietly(sock)
            return None
        if previous is not None:
            previous.close("replaced by a reconnect")
        writer = threading.Thread(
            target=self._write_loop,
            args=(connection,),
            name="engine-link-writer",
            daemon=True,
        )
        connection.threads.append(writer)
        writer.start()
        return connection

    def _refuse(self, sock: socket.socket, code: str, text: str) -> None:
        self.refusals.append(code)
        _LOGGER.warning("Refused an engine (%s): %s", code, text)
        header = Error(code=code, message=text).to_header()
        with contextlib.suppress(OSError):
            send_message(sock, header)
            if self.recorder is not None:
                self.recorder.record(TX, header)
        _close_quietly(sock)

    def _write_loop(self, connection: _Connection) -> None:
        while True:
            item = connection.outbox.get()
            if item is None or connection.closed.is_set():
                return
            header, payload = item
            try:
                send_message(connection.sock, header, payload)
            except Exception as error:  # a send error ends the connection
                connection.close(f"send failed: {error}")
                return
            if self.recorder is not None:
                self.recorder.record(TX, header, payload)
            if header["type"] == "error" and header.get("code") in REFUSAL_CODES:
                self.refusals.append(header["code"])
                connection.close(f"refused: {header['code']}")
                return

    def _monitor_loop(self) -> None:
        next_status = time.monotonic() + self.status_interval_s
        while not self._stop.wait(self.ping_interval_s):
            with self._lock:
                connection = self._connection
            if connection is None or connection.closed.is_set():
                continue
            now = time.monotonic()
            if now - connection.last_rx > self.silence_s:
                _LOGGER.warning(
                    "Engine %s silent for %.1f s; dropping it",
                    connection.session,
                    now - connection.last_rx,
                )
                connection.close("silence")
                continue
            connection.outbox.put((Ping(t_host_us=now_us()).to_header(), b""))
            if self.status_provider is not None and now >= next_status:
                next_status = now + self.status_interval_s
                try:
                    header = Status(extra=dict(self.status_provider())).to_header()
                    encode_message(header)
                except Exception:  # a bad status must not stop pings
                    _LOGGER.exception("Host status provider failed")
                    continue
                connection.outbox.put((header, b""))


def _close_quietly(sock: socket.socket) -> None:
    with contextlib.suppress(OSError):
        sock.close()


__all__ = ["HELLO_TIMEOUT_S", "TcpEngineLink"]
