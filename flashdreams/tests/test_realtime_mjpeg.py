# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import threading
from http import HTTPStatus

import pytest

from flashdreams.serving.realtime.frame_bus import LatestFrameBus
from flashdreams.serving.realtime.mjpeg import (
    format_mjpeg_part,
    mjpeg_content_type,
    publish_latest_jpeg,
    send_mjpeg_response_headers,
    wait_for_latest_jpeg,
    write_mjpeg_stream,
)

pytestmark = pytest.mark.ci_cpu


class _Response:
    def __init__(self) -> None:
        self.status: int | HTTPStatus | None = None
        self.headers: list[tuple[str, str]] = []
        self.ended = False

    def send_response(self, code: int | HTTPStatus) -> None:
        self.status = code

    def send_header(self, keyword: str, value: str) -> None:
        self.headers.append((keyword, value))

    def end_headers(self) -> None:
        self.ended = True


class _Writer:
    def __init__(self) -> None:
        self.parts: list[bytes] = []
        self.flushes = 0

    def write(self, data: bytes) -> object:
        self.parts.append(data)
        return len(data)

    def flush(self) -> object:
        self.flushes += 1
        return None


def test_format_mjpeg_part_includes_boundary_headers_and_payload() -> None:
    part = format_mjpeg_part(b"\xff\xd8jpeg\xff\xd9", boundary="test-boundary")

    assert part == (
        b"--test-boundary\r\n"
        b"Content-Type: image/jpeg\r\n"
        b"Content-Length: 8\r\n\r\n"
        b"\xff\xd8jpeg\xff\xd9\r\n"
    )


def test_send_mjpeg_response_headers_uses_no_cache_multipart_type() -> None:
    response = _Response()

    send_mjpeg_response_headers(response, boundary="test-boundary")

    assert response.status == HTTPStatus.OK
    assert ("Pragma", "no-cache") in response.headers
    assert (
        "Content-Type",
        "multipart/x-mixed-replace; boundary=test-boundary",
    ) in response.headers
    assert response.ended


def test_mjpeg_content_type_uses_boundary() -> None:
    assert mjpeg_content_type("custom") == "multipart/x-mixed-replace; boundary=custom"


def test_write_mjpeg_stream_writes_until_waiter_returns_none() -> None:
    frames = [(b"first", 1), (b"second", 2)]
    seen_counts: list[int] = []

    def wait_for_frame(last_seen_count: int) -> tuple[bytes, int] | None:
        seen_counts.append(last_seen_count)
        if not frames:
            return None
        return frames.pop(0)

    writer = _Writer()

    write_mjpeg_stream(writer, wait_for_frame, boundary="test")

    assert seen_counts == [0, 1, 2]
    assert writer.parts == [
        format_mjpeg_part(b"first", boundary="test"),
        format_mjpeg_part(b"second", boundary="test"),
    ]
    assert writer.flushes == 2


def test_publish_latest_jpeg_ignores_stopped_publish() -> None:
    bus = LatestFrameBus[bytes]()
    stop_event = threading.Event()
    stop_event.set()

    publish_latest_jpeg(bus, b"jpeg", stop_event=stop_event)

    assert bus.latest() is None


def test_wait_for_latest_jpeg_returns_frame_and_count() -> None:
    bus = LatestFrameBus[bytes]()
    stop_event = threading.Event()
    bus.publish(b"old")
    bus.publish(b"new")

    frame = wait_for_latest_jpeg(
        bus,
        last_seen_count=1,
        stop_event=stop_event,
        poll_timeout_s=0.01,
    )

    assert frame == (b"new", 2)


def test_wait_for_latest_jpeg_returns_none_after_bus_close() -> None:
    bus = LatestFrameBus[bytes]()
    stop_event = threading.Event()
    bus.publish(b"old")
    bus.close()

    frame = wait_for_latest_jpeg(
        bus,
        last_seen_count=1,
        stop_event=stop_event,
        poll_timeout_s=0.01,
    )

    assert frame is None
