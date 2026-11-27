# SPDX-FileCopyrightText: Copyright (c) 2026 FlashDreams contributors
# SPDX-License-Identifier: Apache-2.0

import socket
import struct
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from uuid import uuid4

import pytest
from interactive_drive.ros_endpoint import (
    MAX_HEADER,
    NIL_UUID,
    DriveEndpoint,
    read_message,
    write_message,
)

pytestmark = pytest.mark.ci_cpu


def test_single_execution_disconnect_recovery_and_interrupt(tmp_path):
    path = tmp_path / "drive.sock"
    endpoint = DriveEndpoint(path, front_condition=True)
    stop = Event()
    admitted = []
    step_started, proceed = Event(), Event()
    payload = b"\x01\x02\x03"

    def model():
        while (request := endpoint.receive(stop.is_set)) is not None:
            admitted.append(request)
            step = request.get("expected_step", -1)
            tags = {"episode_id": request["episode_id"], "step": step}
            if step == 1:
                step_started.set()
                assert proceed.wait(timeout=4)
            endpoint.send(
                {
                    "type": "frame",
                    **tags,
                    "images": [{"name": "front", "width": 1, "height": 1, "nbytes": 3}],
                },
                [payload],
            )
            endpoint.send({"type": "done", **tags, "next_step": step + 1})

    def connect():
        conn = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        conn.settimeout(4)
        conn.connect(str(path))
        header, _ = read_message(conn)
        assert header["type"] == "hello" and header["version"] == 1
        assert "front_condition" in header["features"]
        return conn, header

    def advance(conn, request):
        write_message(conn, request)
        header, images = read_message(conn)
        assert header["type"] == "frame" and images == [payload]
        assert read_message(conn)[0]["type"] == "done"

    episode = str(uuid4())
    reset = {"type": "reset", "episode_id": episode, "previous_episode_id": NIL_UUID}
    step = {
        "type": "step",
        "episode_id": episode,
        "expected_step": 0,
        "steering_angle_rad": 0.1,
        "throttle": 1.0,
        "brake": 0.0,
        "reverse": False,
    }
    with ThreadPoolExecutor(max_workers=1) as pool:
        result = pool.submit(model)
        try:
            conn, hello = connect()
            with conn:
                assert hello["episode_id"] == NIL_UUID
                advance(conn, reset)
                advance(conn, step)
                for original in (reset, step, {**step, "front_condition": True}):
                    write_message(conn, original)
                    assert read_message(conn)[0]["replayed"] is True
                for req, error in (
                    ({**step, "throttle": 0.2}, "CONFLICT"),
                    ({**step, "expected_step": 4}, "INVALID_STEP"),
                    ({**step, "expected_step": True}, "INVALID_REQUEST"),
                    ({**step, "front_condition": 1}, "INVALID_REQUEST"),
                    ({**step, "brake": -1}, "INVALID_REQUEST"),
                    ({**step, "manual_gear": "R"}, "INVALID_REQUEST"),
                    ({**step, "manual_gear": "X"}, "INVALID_REQUEST"),
                    ({**step, "episode_id": str(uuid4())}, "STALE_EPISODE"),
                ):
                    write_message(conn, req)
                    assert read_message(conn)[0]["code"] == error
                write_message(conn, {**step, "expected_step": 1, "manual_gear": "D"})
                assert step_started.wait(timeout=4)
            # Drop the peer after admission, before the model writes any image.
            proceed.set()
            conn, hello = connect()
            with conn:
                assert hello["needs_reset"] and hello["next_step"] == 2
                write_message(conn, {**step, "expected_step": 2})
                assert read_message(conn)[0]["code"] == "RESET_REQUIRED"
                advance(
                    conn,
                    {
                        **reset,
                        "previous_episode_id": episode,
                        "episode_id": str(uuid4()),
                    },
                )
                assert len(admitted) == 4
                conn.sendall(struct.pack("!I", 128) + b"{")
                stop.set()
                result.result(timeout=2)
        finally:
            stop.set()
            proceed.set()
            endpoint.close()
    assert not path.exists()


def test_wire_bounds_and_no_socket_clobber(tmp_path):
    sender, receiver = socket.socketpair()
    with sender, receiver:
        sender.sendall(struct.pack("!I", MAX_HEADER + 1))
        with pytest.raises(ValueError, match="header size"):
            read_message(receiver)
        with pytest.raises(ValueError, match="RGB8"):
            write_message(
                sender,
                {
                    "type": "frame",
                    "images": [{"name": "front", "width": 1, "height": 1, "nbytes": 2}],
                },
                [b"ab"],
            )
    path = tmp_path / "owned"
    path.write_text("keep")
    with pytest.raises(OSError):
        DriveEndpoint(path)
    assert path.read_text() == "keep"
