# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Transport-neutral MJPEG streaming helpers."""

from __future__ import annotations

from collections.abc import Callable
from http import HTTPStatus
from typing import Protocol

from flashdreams.serving.realtime.frame_bus import LatestFrameBus

DEFAULT_MJPEG_BOUNDARY = "frame"
WaitForMjpegFrame = Callable[[int], tuple[bytes, int] | None]


class StopEvent(Protocol):
    def is_set(self) -> bool: ...


class MjpegResponse(Protocol):
    def send_response(self, code: int | HTTPStatus) -> None: ...

    def send_header(self, keyword: str, value: str) -> None: ...

    def end_headers(self) -> None: ...


class MjpegWriter(Protocol):
    def write(self, data: bytes, /) -> object: ...

    def flush(self) -> object: ...


def mjpeg_content_type(boundary: str = DEFAULT_MJPEG_BOUNDARY) -> str:
    return f"multipart/x-mixed-replace; boundary={boundary}"


def format_mjpeg_part(jpeg: bytes, *, boundary: str = DEFAULT_MJPEG_BOUNDARY) -> bytes:
    """Format one JPEG payload as a multipart MJPEG response part."""
    return (
        (
            f"--{boundary}\r\n"
            "Content-Type: image/jpeg\r\n"
            f"Content-Length: {len(jpeg)}\r\n\r\n"
        ).encode("ascii")
        + jpeg
        + b"\r\n"
    )


def send_mjpeg_response_headers(
    response: MjpegResponse, *, boundary: str = DEFAULT_MJPEG_BOUNDARY
) -> None:
    """Send generic no-cache multipart MJPEG response headers."""
    response.send_response(HTTPStatus.OK)
    response.send_header(
        "Cache-Control", "no-store, no-cache, must-revalidate, max-age=0"
    )
    response.send_header("Pragma", "no-cache")
    response.send_header("Content-Type", mjpeg_content_type(boundary))
    response.end_headers()


def write_mjpeg_stream(
    writer: MjpegWriter,
    wait_for_frame: WaitForMjpegFrame,
    *,
    boundary: str = DEFAULT_MJPEG_BOUNDARY,
) -> None:
    """Write MJPEG parts until ``wait_for_frame`` returns ``None``."""
    last_seen = 0
    try:
        while True:
            result = wait_for_frame(last_seen)
            if result is None:
                return
            jpeg, last_seen = result
            writer.write(format_mjpeg_part(jpeg, boundary=boundary))
            writer.flush()
    except (BrokenPipeError, ConnectionResetError):
        return


def publish_latest_jpeg(
    bus: LatestFrameBus[bytes], jpeg: bytes, *, stop_event: StopEvent
) -> None:
    """Publish ``jpeg`` unless shutdown is already in progress."""
    if stop_event.is_set():
        return
    try:
        bus.publish(jpeg)
    except RuntimeError:
        if not stop_event.is_set():
            raise


def wait_for_latest_jpeg(
    bus: LatestFrameBus[bytes],
    *,
    last_seen_count: int,
    stop_event: StopEvent,
    poll_timeout_s: float = 1.0,
) -> tuple[bytes, int] | None:
    """Wait for a JPEG newer than ``last_seen_count`` or return ``None`` on close."""
    while not stop_event.is_set():
        frame = bus.wait_for_frame(
            last_seen_count=last_seen_count, timeout_s=poll_timeout_s
        )
        if frame is not None:
            return frame.payload, frame.count
        if bus.closed:
            return None
    return None
