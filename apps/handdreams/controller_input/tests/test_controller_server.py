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

"""Controller TCP transport, protocol handshake, and managed lifecycle."""

from __future__ import annotations

import json
import os
import socket
import subprocess
import threading
import time

import pytest
from flashdreams_controller_input import controller_server as server_module
from flashdreams_controller_input.controller_server import (
    ControllerServer,
)
from flashdreams_controller_input.controller_types import (
    ISAAC_TELEOP_BRIDGE_PROTOCOL_VERSION,
)
from flashdreams_controller_input.pose import WRIST_PROXY_POSE_CONVENTION
from flashdreams_controller_input.quest_source import (
    _STAGE_HELP,
    FrameStore,
    ManagedHeadset,
)

pytestmark = pytest.mark.ci_cpu

_TIMEOUT_S = 10.0


def _wait(predicate, timeout: float = _TIMEOUT_S) -> bool:
    """Wait until ``predicate`` succeeds or the test deadline expires."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


def _frame() -> dict:
    """Return a minimal Isaac controller frame."""

    def side(x: float, *, toggle: bool = False) -> dict:
        return {
            "position": [x, 1.0, -0.5],
            "orientation": [0.0, 0.0, 0.0, 1.0],
            "trigger": 0.2,
            "grip": 0.3,
            "axes": [0.1, -0.1],
            "battery": 0.75 if x < 0 else 0.5,
            "tracked": True,
            "primary_button": toggle,
            "secondary_button": False,
        }

    return {
        "type": "xr_frame",
        "t_client": 123.5,
        "controllers": {
            "left": side(-0.22),
            "right": side(0.22, toggle=True),
        },
    }


@pytest.fixture
def controller_server():
    store = FrameStore(
        expected_protocol=ISAAC_TELEOP_BRIDGE_PROTOCOL_VERSION,
        expected_pose_convention=WRIST_PROXY_POSE_CONVENTION,
        stage_help=_STAGE_HELP,
    )
    server = ControllerServer(on_message=store.consume, port=0)
    server.start()
    try:
        yield server, store
    finally:
        server.stop()


def test_controller_records_reach_the_shared_frame_store(controller_server) -> None:
    server, store = controller_server
    message = _frame()
    message["controllers"]["left"].update(primary_button=True, secondary_button=True)
    message["controllers"]["right"]["secondary_button"] = True
    hello = {
        "type": "hello",
        "protocol": ISAAC_TELEOP_BRIDGE_PROTOCOL_VERSION,
        "version": "0.1.5",
        "pose_convention": WRIST_PROXY_POSE_CONVENTION,
    }
    before_capture_ns = time.perf_counter_ns()
    with socket.create_connection(
        ("127.0.0.1", server.port), timeout=_TIMEOUT_S
    ) as connection:
        connection.sendall(
            (json.dumps(hello) + "\n" + json.dumps(message) + "\n").encode()
        )
        assert _wait(store.ready)

    snapshot = store.snapshot()
    assert snapshot is not None
    assert before_capture_ns <= snapshot["client_capture_ns"] <= time.perf_counter_ns()
    assert snapshot["source_frame_index"] == 0
    for side, controller in message["controllers"].items():
        for key in ("position", "orientation", "trigger", "grip"):
            assert snapshot[side][key] == pytest.approx(controller[key])
        assert snapshot[side]["tracked"] is controller["tracked"]
        assert snapshot[side]["primary_button"] is controller["primary_button"]
        assert snapshot[side]["secondary_button"] is controller["secondary_button"]
        assert not {
            "next_scene",
            "previous_scene",
            "toggle_live",
            "recenter",
        }.intersection(snapshot[side])
    assert snapshot["right"]["primary_button"] is True
    assert snapshot["left"]["axes"] == pytest.approx([0.1, -0.1])
    assert snapshot["left"]["battery"] == pytest.approx(0.75)
    assert snapshot["right"]["battery"] == pytest.approx(0.5)
    assert snapshot["source_client_capture_ns"] == 123_500_000_000
    assert store.protocol == ISAAC_TELEOP_BRIDGE_PROTOCOL_VERSION
    assert store.pose_convention == WRIST_PROXY_POSE_CONVENTION
    assert server.connections == 1


@pytest.mark.parametrize(
    "t_client", [True, -1.0, float("nan"), "bad", 1e308, 10**400, (2**64 - 1) / 1e9]
)
def test_controller_rejects_malformed_client_capture_time(t_client: object) -> None:
    store = FrameStore()
    frame = _frame()
    frame["t_client"] = t_client

    store.consume(json.dumps(frame))

    assert store.received == 0
    assert store.rejected == 1


def test_controller_transport_handles_fragmented_and_multiple_records(
    controller_server,
) -> None:
    server, store = controller_server
    hello = json.dumps(
        {
            "type": "hello",
            "protocol": ISAAC_TELEOP_BRIDGE_PROTOCOL_VERSION,
            "pose_convention": WRIST_PROXY_POSE_CONVENTION,
        }
    ).encode()
    frame = json.dumps(_frame()).encode()
    with socket.create_connection(
        ("127.0.0.1", server.port), timeout=_TIMEOUT_S
    ) as connection:
        connection.sendall(hello[:8])
        connection.sendall(hello[8:] + b"\n{not json\n" + frame + b"\n")
        assert _wait(lambda: store.rejected == 1 and store.received == 1)
    assert store.ready()


def test_controller_receiver_probe_is_not_reported_as_a_producer(
    controller_server, capsys
) -> None:
    server, _ = controller_server
    capsys.readouterr()
    with socket.create_connection(("127.0.0.1", server.port), timeout=_TIMEOUT_S):
        pass
    time.sleep(0.05)
    assert server.connections == 0
    output = capsys.readouterr().out
    assert "Isaac Teleop bridge connected" not in output
    assert "Isaac Teleop bridge disconnected" not in output


@pytest.mark.parametrize("pending", [b"", b"partial", b"x" * (64 * 1024 + 32)])
def test_controller_stop_closes_idle_and_partial_connections(pending: bytes) -> None:
    messages: list[str | bytes] = []
    server = ControllerServer(on_message=messages.append, port=0)
    server.start()
    with socket.create_connection(("127.0.0.1", server.port), timeout=2) as peer:
        try:
            peer.sendall(b"first\n" + pending)
            assert _wait(lambda: messages == [b"first"])
            assert server._server is not None
            handlers: tuple[threading.Thread, ...] = tuple(
                getattr(server._server, "_threads")
            )
            stopper = threading.Thread(target=server.stop, daemon=True)
            stopper.start()
            stopper.join(timeout=2)
            assert not stopper.is_alive()
            assert not server._clients
            assert all(not thread.is_alive() for thread in handlers)
            assert peer.recv(1) == b""
            try:
                peer.sendall(b"after stop\n")
            except OSError:
                pass
            assert messages == [b"first"]
        finally:
            peer.close()
            server.stop()


def test_controller_disconnect_notifies_once_per_producer_not_empty_probe() -> None:
    messages: list[str | bytes] = []
    disconnects: list[None] = []
    server = ControllerServer(
        on_message=messages.append,
        on_disconnect=lambda: disconnects.append(None),
        port=0,
    )
    server.start()
    try:
        with socket.create_connection(("127.0.0.1", server.port), timeout=2):
            pass
        for index in range(2):
            with socket.create_connection(
                ("127.0.0.1", server.port), timeout=2
            ) as peer:
                peer.sendall(b"first\nsecond\n")
                assert _wait(lambda: len(messages) == (index + 1) * 2)
            assert _wait(lambda: len(disconnects) == index + 1)
    finally:
        server.stop()
    assert disconnects == [None, None]


def test_controller_stop_joins_the_disconnect_callback() -> None:
    messages: list[str | bytes] = []
    entered = threading.Event()
    release = threading.Event()

    def disconnect() -> None:
        entered.set()
        assert release.wait(timeout=5)

    server = ControllerServer(
        on_message=messages.append, on_disconnect=disconnect, port=0
    )
    server.start()
    stopper = threading.Thread(target=server.stop)
    try:
        with socket.create_connection(("127.0.0.1", server.port), timeout=2) as peer:
            peer.sendall(b"first\n")
            assert _wait(lambda: messages == [b"first"])
            stopper.start()
            assert entered.wait(timeout=2)
            assert stopper.is_alive()
            release.set()
            stopper.join(timeout=2)
            assert not stopper.is_alive()
    finally:
        release.set()
        if stopper.ident is not None:
            stopper.join(timeout=2)
        server.stop()


def test_controller_stop_waits_for_in_flight_callback(monkeypatch) -> None:
    entered = threading.Event()
    release = threading.Event()
    listener_stopped = threading.Event()
    messages: list[str | bytes] = []

    def consume(message: str | bytes) -> None:
        entered.set()
        assert release.wait(timeout=5)
        messages.append(message)

    server = ControllerServer(on_message=consume, port=0)
    server.start()
    assert server._server is not None
    shutdown = server._server.shutdown

    def shutdown_listener() -> None:
        shutdown()
        listener_stopped.set()

    monkeypatch.setattr(server._server, "shutdown", shutdown_listener)
    with socket.create_connection(("127.0.0.1", server.port), timeout=2) as peer:
        stopper = threading.Thread(target=server.stop, daemon=True)
        try:
            peer.sendall(b"first\nsecond\n")
            assert entered.wait(timeout=2)
            stopper.start()
            assert listener_stopped.wait(timeout=2)
            assert stopper.is_alive()
            release.set()
            stopper.join(timeout=2)
            assert not stopper.is_alive()
            assert messages == [b"first"]
        finally:
            release.set()
            if stopper.ident is not None:
                stopper.join(timeout=2)
            server.stop()


def test_controller_stop_cancels_a_concurrent_bind(monkeypatch) -> None:
    bound = threading.Event()
    release = threading.Event()
    sockets = []
    server_type = server_module._ThreadingTCPServer

    class DelayedServer(server_type):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            sockets.append(self.socket)
            bound.set()
            assert release.wait(timeout=5)

    monkeypatch.setattr(server_module, "_ThreadingTCPServer", DelayedServer)
    server = ControllerServer(on_message=lambda _: None, port=0)
    starter = threading.Thread(target=server.start, daemon=True)
    stopper = threading.Thread(target=server.stop, daemon=True)
    starter.start()
    try:
        assert bound.wait(timeout=2)
        stopper.start()
        assert server._stopped.wait(timeout=2)
        release.set()
        starter.join(timeout=2)
        stopper.join(timeout=2)
        assert not starter.is_alive() and not stopper.is_alive()
        assert sockets[0].fileno() == -1
        assert server._server is None and server._thread is None
        server.start()
        assert len(sockets) == 1
    finally:
        release.set()
        starter.join(timeout=2)
        if stopper.ident is not None:
            stopper.join(timeout=2)
        server.stop()


def test_controller_protocol_mismatch_never_becomes_ready() -> None:
    store = FrameStore(expected_protocol=2, stage_help=_STAGE_HELP)
    store.consume(json.dumps({"type": "hello", "protocol": 1, "version": "old"}))
    store.consume(json.dumps(_frame()))
    assert not store.ready()
    explanation = store.explain()
    assert "protocol 1" in explanation
    assert "requires 2" in explanation
    assert "restart the bridge with matching Isaac settings" in explanation


@pytest.mark.parametrize(
    "pose_convention",
    [None, "openxr-local-grip-v1"],
)
def test_controller_pose_convention_mismatch_never_becomes_ready(
    pose_convention: str | None,
) -> None:
    store = FrameStore(
        expected_protocol=ISAAC_TELEOP_BRIDGE_PROTOCOL_VERSION,
        expected_pose_convention=WRIST_PROXY_POSE_CONVENTION,
        stage_help=_STAGE_HELP,
    )
    hello = {
        "type": "hello",
        "protocol": ISAAC_TELEOP_BRIDGE_PROTOCOL_VERSION,
        "version": "incompatible",
    }
    if pose_convention is not None:
        hello["pose_convention"] = pose_convention
    store.consume(json.dumps(hello))
    store.consume(json.dumps(_frame()))

    assert not store.ready()
    explanation = store.explain()
    assert "pose convention" in explanation
    assert WRIST_PROXY_POSE_CONVENTION in explanation
    assert "restart the bridge with matching Isaac settings" in explanation


@pytest.mark.parametrize("protocol", ["3", True])
def test_controller_hello_requires_an_integer_protocol(protocol: object) -> None:
    store = FrameStore(expected_protocol=ISAAC_TELEOP_BRIDGE_PROTOCOL_VERSION)
    store.consume(json.dumps({"type": "hello", "protocol": protocol}))
    assert store.protocol is None
    assert store.rejected == 1


@pytest.mark.parametrize("require_physical_wake", [False, True])
@pytest.mark.parametrize("adb_executable", [None, "/resolved/Android SDK/adb"])
def test_managed_headset_passes_controller_receiver_configuration(
    monkeypatch, require_physical_wake, adb_executable
) -> None:
    launched: dict = {}
    monkeypatch.delenv("QUEST_META_PIN", raising=False)
    monkeypatch.setenv("QUEST_REQUIRE_PHYSICAL_WAKE", "inherited-value")
    monkeypatch.setenv("QUEST_ADB", "/inherited/adb")

    class _Process:
        stdin = None

        @staticmethod
        def poll():
            return 0

    def popen(*args, **kwargs):
        launched["args"] = args
        launched["kwargs"] = kwargs
        return _Process()

    monkeypatch.setattr("subprocess.Popen", popen)
    headset = ManagedHeadset(
        device_serial="quest-explicit",
        require_physical_wake=require_physical_wake,
        adb_executable=adb_executable,
    )
    headset.start()
    environment = launched["kwargs"]["env"]
    assert environment["QUEST_OWNER_PID"] == str(os.getpid())
    assert environment["QUEST_SERIAL"] == "quest-explicit"
    assert environment["QUEST_ADB"] == (adb_executable or "/inherited/adb")
    assert os.environ["QUEST_ADB"] == "/inherited/adb"
    assert environment["QUEST_REQUIRE_PHYSICAL_WAKE"] == (
        "1" if require_physical_wake else "0"
    )
    assert launched["kwargs"]["pass_fds"]
    assert not headset.wake_ready
    assert "QUEST_META_PIN" not in environment
    assert launched["kwargs"]["stdin"] is subprocess.DEVNULL
    headset.stop()


def test_stopping_a_controller_server_that_never_started_is_safe() -> None:
    ControllerServer(on_message=lambda _: None, port=0).stop()
