# SPDX-FileCopyrightText: Copyright (c) 2026 FlashDreams contributors
# SPDX-License-Identifier: Apache-2.0
"""Local Interactive Drive wire protocol, shared with the ROS Python environment.

Each record starts with a big-endian uint32 UTF-8 JSON length, then the header
and its tightly packed RGB8 images in descriptor order. One client sends one
request at a time. The model thread alone owns the listener and step cursor.
"""

from __future__ import annotations

import json
import os
import socket
import struct
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from uuid import UUID

NIL_UUID = str(UUID(int=0))
MAX_HEADER = 256 * 1024
MAX_FRAME_BYTES = 256 * 1024 * 1024


def _read_exact(
    conn: socket.socket, size: int, stopping: Callable[[], bool] | None
) -> bytes:
    data = bytearray(size)
    target = memoryview(data)
    while target:
        if stopping is not None and stopping():
            raise InterruptedError("Drive session stopped")
        try:
            count = conn.recv_into(target)
        except TimeoutError:
            if stopping is None:
                raise
            continue
        if count == 0:
            raise EOFError("Drive peer disconnected")
        target = target[count:]
    return bytes(data)


def _image_sizes(header: dict) -> list[int]:
    images = header.get("images", [])
    if not isinstance(images, list) or len(images) > 33:
        raise ValueError("Invalid image descriptor list")
    if len(images) == 33 and (
        not isinstance(images[-1], dict) or images[-1].get("role") != "condition"
    ):
        raise ValueError("At most 32 cameras plus one conditioning image")
    sizes = []
    names = set()
    for item in images:
        if not isinstance(item, dict) or not isinstance(item.get("name"), str):
            raise ValueError("Missing camera name")
        if not item["name"] or item["name"] in names:
            raise ValueError("Empty or duplicate camera name")
        names.add(item["name"])
        width, height, size = (item.get(k) for k in ("width", "height", "nbytes"))
        if not all(type(v) is int and v > 0 for v in (width, height, size)):
            raise ValueError("Invalid image dimensions")
        if item.get("encoding", "rgb8") != "rgb8" or width * height * 3 != size:
            raise ValueError("Expected tightly packed RGB8 pixels")
        sizes.append(size)
    if sum(sizes) > MAX_FRAME_BYTES:
        raise ValueError("Drive frame exceeds the size limit")
    return sizes


def read_message(
    conn: socket.socket,
    *,
    stopping: Callable[[], bool] | None = None,
    allow_images: bool = True,
) -> tuple[dict, list[bytes]]:
    size = struct.unpack("!I", _read_exact(conn, 4, stopping))[0]
    if not 0 < size <= MAX_HEADER:
        raise ValueError("Invalid drive header size")
    header = json.loads(_read_exact(conn, size, stopping).decode("utf-8"))
    if not isinstance(header, dict) or not isinstance(header.get("type"), str):
        raise ValueError("Expected a typed drive message")
    sizes = _image_sizes(header)
    if sizes and not allow_images:
        raise ValueError("Drive requests cannot contain images")
    return header, [_read_exact(conn, n, stopping) for n in sizes]


def write_message(
    conn: socket.socket,
    header: dict,
    images: Sequence = (),
    *,
    stopping: Callable[[], bool] | None = None,
) -> None:
    body = json.dumps(header, separators=(",", ":"), allow_nan=False).encode("utf-8")
    if len(body) > MAX_HEADER:
        raise ValueError("Drive header exceeds the size limit")
    buffers = [memoryview(img).cast("B") for img in images]
    if [buf.nbytes for buf in buffers] != _image_sizes(header):
        raise ValueError("Image descriptors do not match the supplied RGB buffers")
    for buf in [memoryview(struct.pack("!I", len(body)) + body), *buffers]:
        while buf:
            if stopping is not None and stopping():
                raise InterruptedError("Drive session stopped")
            try:
                count = conn.send(buf)
            except TimeoutError:
                if stopping is None:
                    raise
                continue
            if count == 0:
                raise EOFError("Drive peer disconnected")
            buf = buf[count:]


def _uuid(request: dict, field: str) -> str:
    value = request.get(field)
    if not isinstance(value, str) or str(UUID(value)) != value:
        raise ValueError(f"{field} must be a canonical UUID")
    return value


def validate_request(request: dict) -> None:
    """Check service input on either side of the stream, before admission."""
    if _uuid(request, "episode_id") == NIL_UUID:
        raise ValueError("episode_id must be nonzero")
    fields = {"type", "episode_id"}
    if request.get("type") == "reset":
        _uuid(request, "previous_episode_id")
        fields.add("previous_episode_id")
    elif request.get("type") == "step":
        step = request.get("expected_step")
        if type(step) is not int or not 0 <= step < 2**64:
            raise ValueError("expected_step must be uint64")
        if type(request.get("reverse")) is not bool:
            raise ValueError("reverse must be boolean")
        fields.update(
            ("expected_step", "reverse", "steering_angle_rad", "throttle", "brake")
        )
        if "front_condition" in request:
            if type(request["front_condition"]) is not bool:
                raise ValueError("front_condition must be boolean")
            fields.add("front_condition")
        if "manual_gear" in request:
            gear = request["manual_gear"]
            if gear not in ("P", "R", "N", "D") or (gear == "R") != request["reverse"]:
                raise ValueError("manual_gear must be P/R/N/D and agree with reverse")
            fields.add("manual_gear")
        for key in ("steering_angle_rad", "throttle", "brake"):
            value = request.get(key)
            if (
                type(value) not in (int, float)
                or not -sys.float_info.max <= value <= sys.float_info.max
            ):
                raise ValueError(f"{key} must be finite")
            if key != "steering_angle_rad" and not 0 <= value <= 1:
                raise ValueError(f"{key} must be between zero and one")
    else:
        raise ValueError("Expected a reset or step request")
    if request.keys() != fields:
        raise ValueError("Unexpected drive request fields")


def _error(code: str, message: str) -> dict:
    return {"type": "error", "code": code, "message": message}


class DriveEndpoint:
    """Serve one local bridge with an interruptible, serialized model-side loop."""

    def __init__(self, path: str | Path, *, front_condition: bool = False):
        self._path = Path(path)
        self._front_condition = front_condition
        self._listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._owned_inode = None
        self._conn: socket.socket | None = None
        self._stopping: Callable[[], bool] | None = None
        self._episode = NIL_UUID
        self._next = 0
        self._needs_reset = True
        self._pending: dict | None = None
        self._sent_frame = False
        self._receipts: dict[str, tuple[dict, dict]] = {}
        try:
            self._listener.bind(os.fspath(self._path))
            self._owned_inode = self._path.lstat().st_ino
            self._path.chmod(0o600)
            self._listener.listen(1)
            self._listener.settimeout(0.2)
        except BaseException:
            self.close()
            raise

    def _drop_client(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def close(self) -> None:
        self._drop_client()
        self._listener.close()
        if self._owned_inode is not None:
            try:
                if self._path.lstat().st_ino == self._owned_inode:
                    self._path.unlink()
            except FileNotFoundError:
                pass
            self._owned_inode = None

    def _write(self, header: dict, images: Sequence = ()) -> None:
        if self._conn is not None:
            try:
                write_message(self._conn, header, images, stopping=self._stopping)
            except (OSError, EOFError):
                self._drop_client()
                if self._pending is not None:
                    self._needs_reset = True

    def _admit(self, request: dict) -> dict | None:
        kind = request["type"]
        receipt = self._receipts.get(kind)
        if request.get("front_condition") and not self._front_condition:
            return _error("INVALID_REQUEST", "Endpoint has no front conditioning image")
        # Viewer subscriptions can change between retries of the same control.
        if receipt and (request | {"front_condition": False}) == (
            receipt[0] | {"front_condition": False}
        ):
            if self._needs_reset:
                return _error("RESET_REQUIRED", "Observations were interrupted")
            return {**receipt[1], "replayed": True}
        episode = request["episode_id"]
        if kind == "reset":
            if self._episode == episode:
                return _error("CONFLICT", "Reset requires a new episode UUID")
            if request["previous_episode_id"] != self._episode:
                return _error(
                    "STALE_EPISODE", "Reset parent differs from current episode"
                )
        elif episode != self._episode:
            return _error("STALE_EPISODE", "Step episode differs from current episode")
        elif self._needs_reset:
            return _error("RESET_REQUIRED", "Reset after interrupted observations")
        elif request["expected_step"] != self._next:
            code = (
                "CONFLICT"
                if receipt and request["expected_step"] == receipt[0]["expected_step"]
                else "INVALID_STEP"
            )
            return _error(code, f"Expected step {self._next}")
        return None

    def receive(self, stopping: Callable[[], bool]) -> dict | None:
        """Return the next admitted request, answering protocol errors and retries here."""
        if self._pending is not None:
            raise RuntimeError("Previous drive request has no result")
        self._stopping = stopping
        while not stopping() and self._listener.fileno() != -1:
            if self._conn is None:
                try:
                    self._conn, _ = self._listener.accept()
                except TimeoutError:
                    continue
                except OSError:
                    if self._listener.fileno() == -1:
                        return None
                    raise
                self._conn.settimeout(0.2)
                self._write(
                    {
                        "type": "hello",
                        "version": 1,
                        "episode_id": self._episode,
                        "next_step": self._next,
                        "needs_reset": self._needs_reset,
                        "features": ["front_condition"] if self._front_condition else [],
                    }
                )
                continue
            try:
                request, _ = read_message(
                    self._conn, stopping=stopping, allow_images=False
                )
            except (ValueError, RecursionError, EOFError, OSError):
                # A truncated or invalid frame loses synchronization with the stream.
                self._drop_client()
                continue
            try:
                validate_request(request)
                reply = self._admit(request)
            except ValueError as exc:
                reply = _error("INVALID_REQUEST", str(exc))
            if reply is not None:
                self._write(reply)
                continue
            self._pending = request
            self._sent_frame = False
            return request
        return None

    def send(self, header: dict, images: Sequence = ()) -> None:
        """Emit observations or finish the current operation, including after a disconnect."""
        request = self._pending
        if request is None or header["type"] not in ("frame", "done", "error"):
            raise RuntimeError("Unexpected drive response")
        if header["type"] != "error":
            step = request.get("expected_step", -1)
            if (
                header.get("episode_id") != request["episode_id"]
                or header.get("step") != step
            ):
                raise ValueError("Drive response does not match the admitted request")
            if header["type"] == "done" and header.get("next_step") != step + 1:
                raise ValueError("Drive response has an inconsistent step cursor")
        if header["type"] == "frame":
            self._sent_frame = True
            self._write(header, images)
            return
        if header["type"] == "done":
            if request["type"] == "reset":
                self._episode = request["episode_id"]
                self._needs_reset = self._conn is None
                self._receipts.clear()
            self._next = header["next_step"]
            self._receipts[request["type"]] = (request, dict(header))
        elif self._sent_frame or header.get("fatal"):
            self._needs_reset = True
        self._write(header, images)
        self._pending = None
