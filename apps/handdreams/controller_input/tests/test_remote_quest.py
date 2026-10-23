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

"""CPU transport, freshness, identity, and WebXR relay lifecycle regressions."""

from __future__ import annotations

import io
import json
import shutil
import socket
import ssl
import subprocess
import threading
import time
from contextlib import contextmanager
from typing import Any, Callable, cast

import pytest
from flashdreams_controller_input import quest_relay, remote_quest
from flashdreams_controller_input.controller_types import (
    ISAAC_TELEOP_BRIDGE_PROTOCOL_VERSION,
    SUPPORTED_ISAAC_TELEOP_VERSION,
)
from flashdreams_controller_input.frames import SOURCE_CLIENT_CAPTURE_NS_KEY
from flashdreams_controller_input.pose import (
    DEFAULT_GRIP_TO_WRIST_OFFSET_M,
    OPENXR_LOCAL_GRIP_POSE_CONVENTION,
    OPENXR_STAGE_GRIP_POSE_CONVENTION,
    WRIST_PROXY_POSE_CONVENTION,
)
from flashdreams_controller_input.quest_source import FrameStore
from flashdreams_controller_input.remote_quest import (
    RemoteQuestSource,
    encode_record,
    read_record,
)

pytestmark = pytest.mark.ci_cpu
_TOKEN = "test-controller-token-never-use-in-deployment"
_HELLO = json.dumps(
    {
        "type": "hello",
        "protocol": ISAAC_TELEOP_BRIDGE_PROTOCOL_VERSION,
        "version": SUPPORTED_ISAAC_TELEOP_VERSION,
        "mode": "cloudxr-webxr",
        "pose_convention": WRIST_PROXY_POSE_CONVENTION,
        "input_pose_convention": OPENXR_STAGE_GRIP_POSE_CONVENTION,
        "output_pose_convention": WRIST_PROXY_POSE_CONVENTION,
        "wrist_offset_m": DEFAULT_GRIP_TO_WRIST_OFFSET_M,
    }
)


@pytest.fixture(scope="module")
def certificate(tmp_path_factory):
    if not shutil.which("openssl"):
        pytest.skip("TLS integration test requires the openssl executable")
    directory = tmp_path_factory.mktemp("remote-controller-tls")
    cert, key = directory / "server.pem", directory / "key.pem"
    subprocess.run(
        [
            "openssl",
            "req",
            "-x509",
            "-newkey",
            "rsa:2048",
            "-nodes",
            "-days",
            "1",
            "-subj",
            "/CN=localhost",
            "-addext",
            "subjectAltName=DNS:localhost,IP:127.0.0.1",
            "-keyout",
            str(key),
            "-out",
            str(cert),
        ],
        check=True,
        capture_output=True,
    )
    return str(cert), str(key)


@pytest.fixture
def source(certificate):
    cert, key = certificate
    source = RemoteQuestSource(
        host="127.0.0.1",
        port=0,
        certfile=cert,
        keyfile=key,
        token=_TOKEN,
    )
    source.start()
    try:
        yield source
    finally:
        source.stop()


def _frame(capture_ms=1.0, **overrides):
    side = {
        "position": [0.2, 1.0, -0.3],
        "orientation": [0, 0.3, 0, 0.9],
        "trigger": 0.25,
        "grip": 0.6,
        "tracked": True,
        "axes": [0.1, -0.2],
        "battery": 0.75,
        "buttons": {"a": False},
    }
    return json.dumps(
        {
            "type": "xr_frame",
            "t_client": capture_ms,
            "controllers": {"left": side, "right": side},
            **overrides,
        }
    )


def _wait(predicate, timeout=5):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = predicate()
        if result:
            return result
        time.sleep(0.005)
    raise AssertionError("condition was not reached")


@contextmanager
def _peer(source, certificate, token=_TOKEN, hostname="localhost"):
    context = ssl.create_default_context(cafile=certificate[0])
    with socket.create_connection(("127.0.0.1", source.port), timeout=2) as raw:
        with context.wrap_socket(raw, server_hostname=hostname) as connection:
            with connection.makefile("rb") as stream:
                connection.sendall(
                    encode_record({"type": "auth", "version": 1, "token": token})
                )
                assert read_record(stream) == {"type": "ready"}
                yield connection, stream


def _send(peer, messages, *, disconnected=False, generation=0):
    connection, stream = peer
    request = read_record(stream)
    assert request["type"] == "pull"
    connection.sendall(
        encode_record(
            {
                "type": "batch",
                "seq": request["seq"],
                "messages": messages,
                "disconnected": disconnected,
                "generation": generation,
            }
        )
    )


def _next_frame(source):
    item = source.poll()
    return item if item is not None and item.frame is not None else None


def test_remote_vr_remains_physical_even_with_an_emulation_marker(source, certificate):
    with _peer(source, certificate) as peer:
        _send(peer, [_HELLO, _frame(is_emulated=True, client_connection_id="iwer")])
        observation = _wait(lambda: _next_frame(source))
        assert observation.frame["is_emulated"] is False


@pytest.mark.parametrize("version_suffix", ["", "+local"])
def test_isaac_transport_preserves_wrist_poses_and_uses_seconds(
    source, certificate, version_suffix
):
    hello = json.loads(_HELLO)
    hello["version"] += version_suffix
    message = _frame(123.456)
    local = FrameStore()
    local.consume(message)
    with _peer(source, certificate) as peer:
        before = time.perf_counter_ns()
        _send(peer, [json.dumps(hello), message])
        observation = _wait(lambda: _next_frame(source))
        after = time.perf_counter_ns()
        assert before <= observation.capture_ns <= after
        assert observation.frame["client_capture_ns"] == observation.capture_ns
        assert observation.frame[SOURCE_CLIENT_CAPTURE_NS_KEY] == 123456000000
        local_frame = local.snapshot()
        for side in ("left", "right"):
            assert local_frame is not None
            assert observation.frame[side] == local_frame[side]
        assert source._store.pose_convention == WRIST_PROXY_POSE_CONVENTION
        assert source.poll() is None


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("protocol", 3),
        ("protocol", None),
        ("version", "1.5.0"),
        ("version", None),
        ("mode", "native-apk"),
        ("mode", None),
        ("pose_convention", OPENXR_LOCAL_GRIP_POSE_CONVENTION),
        ("pose_convention", None),
        ("input_pose_convention", OPENXR_LOCAL_GRIP_POSE_CONVENTION),
        ("input_pose_convention", None),
        ("output_pose_convention", OPENXR_LOCAL_GRIP_POSE_CONVENTION),
        ("output_pose_convention", None),
        ("wrist_offset_m", 0.075),
        ("wrist_offset_m", None),
        ("wrist_offset_m", True),
    ],
)
def test_isaac_receiver_requires_its_explicit_handshake(
    source, certificate, field, value
):
    hello = json.loads(_HELLO)
    if value is None:
        hello.pop(field)
    else:
        hello[field] = value
    with _peer(source, certificate) as peer:
        _send(peer, [json.dumps(hello), _frame()])
        with pytest.raises(EOFError):
            read_record(peer[1])
        assert _next_frame(source) is None
        assert source._store.received == 0
    with _peer(source, certificate) as peer:
        _send(peer, [_HELLO, _frame(2)])
        assert _wait(lambda: _next_frame(source))


@pytest.mark.parametrize(
    "timestamp", [None, True, -1.0, 1e300, (2**64) / 1e9, float("nan")]
)
def test_isaac_receiver_rejects_invalid_source_timestamps(
    source, certificate, timestamp
):
    with _peer(source, certificate) as peer:
        _send(peer, [_HELLO, _frame(t_client=timestamp)])
        with pytest.raises(EOFError):
            read_record(peer[1])
        assert source._store.received == 0
        assert _next_frame(source) is None


def test_isaac_receiver_accepts_the_configured_wrist_offset(certificate):
    source = RemoteQuestSource(
        host="127.0.0.1",
        port=0,
        certfile=certificate[0],
        keyfile=certificate[1],
        token=_TOKEN,
        wrist_offset_m=0.075,
    )
    hello = json.loads(_HELLO)
    hello["wrist_offset_m"] = 0.075
    source.start()
    try:
        with _peer(source, certificate) as peer:
            _send(peer, [json.dumps(hello), _frame()])
            assert _wait(lambda: _next_frame(source))
            assert source.poll() is None
    finally:
        source.stop()


@pytest.mark.parametrize("offset", [-0.01, float("nan"), float("inf")])
def test_isaac_receiver_rejects_invalid_wrist_offsets(certificate, offset):
    with pytest.raises(ValueError, match="wrist offset"):
        RemoteQuestSource(
            certfile=certificate[0],
            keyfile=certificate[1],
            token=_TOKEN,
            wrist_offset_m=offset,
        )


def test_wrong_token_never_produces_input_or_logs_secret(source, certificate, capsys):
    wrong = "wrong-secret-" * 4
    with pytest.raises(EOFError), _peer(source, certificate, token=wrong):
        pass
    assert source.poll() is None
    assert wrong not in capsys.readouterr().out
    # A rejected client does not take ownership of the source.
    with _peer(source, certificate) as peer:
        _send(peer, [_HELLO, _frame()])
        _wait(lambda: _next_frame(source))


def test_wrong_tls_hostname_is_rejected(source, certificate):
    with (
        pytest.raises(ssl.SSLCertVerificationError),
        _peer(source, certificate, hostname="wrong.example"),
    ):
        pass
    assert source.poll() is None


def test_untrusted_certificate_is_rejected(source):
    context = ssl.create_default_context()
    with socket.create_connection(("127.0.0.1", source.port), timeout=2) as raw:
        with pytest.raises(ssl.SSLCertVerificationError):
            context.wrap_socket(raw, server_hostname="localhost")


@pytest.mark.parametrize(
    "line", [b"[]\n", b"{}", b"x" * (remote_quest.MAX_RECORD_BYTES + 1), b"{broken}\n"]
)
def test_wire_records_are_bounded_objects(line):
    with pytest.raises(ValueError):
        read_record(io.BytesIO(line))


def test_reconnect_preserves_frame_ids_and_emits_disconnect(source, certificate):
    with _peer(source, certificate) as peer:
        _send(peer, [_HELLO, _frame(10)])
        first = _wait(lambda: _next_frame(source))
    boundary = _wait(source.poll)
    assert boundary.frame is None
    with _peer(source, certificate) as peer:
        _send(peer, [_HELLO, _frame(20)])
        second = _wait(lambda: _next_frame(source))
    assert second.frame["source_frame_index"] > first.frame["source_frame_index"]
    scale = 1_000_000_000
    assert second.frame[SOURCE_CLIENT_CAPTURE_NS_KEY] == 20 * scale
    assert first.generation == 0
    assert second.generation == 1


def test_capture_clock_rollback_requires_recalibration(source, certificate):
    with _peer(source, certificate) as peer:
        _send(peer, [_HELLO, _frame(20)])
        _wait(lambda: _next_frame(source))
        _send(peer, [_frame(10)])
        _wait(lambda: source._failure is not None)
        with pytest.raises(RuntimeError, match="restart the demo and recalibrate"):
            source.poll()


def test_reconnect_does_not_reset_the_capture_clock(source, certificate):
    with _peer(source, certificate) as peer:
        _send(peer, [_HELLO, _frame(20)])
        _wait(lambda: _next_frame(source))
    with _peer(source, certificate) as peer:
        _send(peer, [_HELLO, _frame(10)])
        _wait(lambda: source._failure is not None)
        with pytest.raises(RuntimeError, match="restart the demo and recalibrate"):
            source.poll()


def test_duplicate_capture_is_ignored_and_newer_sample_recovers(source, certificate):
    with _peer(source, certificate) as peer:
        _send(peer, [_HELLO, _frame(20)])
        first = _wait(lambda: _next_frame(source))
        _send(peer, [_frame(20), _frame(21)])
        second = _wait(lambda: _next_frame(source))
        assert source._store.received == 2
        assert source.poll() is None
        assert second.frame["source_frame_index"] > first.frame["source_frame_index"]


def test_producer_generation_changes_but_cached_hello_does_not_reset(
    source, certificate
):
    with _peer(source, certificate) as peer:
        _send(peer, [_HELLO, _frame(1)])
        first = _wait(lambda: _next_frame(source))
        _send(peer, [_HELLO, _frame(2)], disconnected=True)
        second = _wait(lambda: _next_frame(source))
        assert first.generation == second.generation == 0
        _send(peer, [_HELLO, _frame(3)], disconnected=True, generation=1)
        third = _wait(lambda: _next_frame(source))
        assert third.generation == 1


def test_reauthenticated_source_does_not_replay_old_frames(source, certificate):
    with _peer(source, certificate) as peer:
        _send(peer, [_HELLO, _frame(1)])
        _wait(lambda: _next_frame(source))
    with _peer(source, certificate) as peer:
        assert read_record(peer[1])["type"] == "pull"
        boundary = source.poll()
        assert boundary is not None and boundary.frame is None
        assert source.poll() is None


@pytest.mark.parametrize(
    "messages",
    [
        [_frame()],
        [_HELLO, _frame(t_client=None)],
        [_HELLO, _frame(t_client=1e300)],
        [
            _HELLO,
            _frame(
                controllers={
                    "left": {
                        "position": [1, 2, 3],
                        "orientation": [0, 0, 0, 1],
                        "trigger": "nan",
                    }
                }
            ),
        ],
        [
            _HELLO,
            _frame(
                controllers={
                    "left": {
                        "position": [1, 2, 3],
                        "orientation": [0, 0, 0, 1],
                        "trigger": 2,
                    }
                }
            ),
        ],
        ["[" * 2000 + "]" * 2000],
        [_HELLO, _frame(controllers={"left": {"position": [1, 2]}})],
        [json.dumps({"type": "hello", "protocol": 999})],
        [_HELLO, "x" * (64 * 1024 + 1)],
    ],
)
def test_malformed_or_incompatible_producer_disconnects_without_frames(
    source, certificate, messages
):
    with _peer(source, certificate) as peer:
        _send(peer, messages)
        with pytest.raises(EOFError):
            read_record(peer[1])
        assert _next_frame(source) is None
    with _peer(source, certificate) as peer:
        _send(peer, [_HELLO, _frame(1000)])
        _wait(lambda: _next_frame(source))


def test_stalled_reply_is_not_replayed(source, certificate, monkeypatch):
    monkeypatch.setattr(remote_quest, "MAX_ROUND_TRIP_S", 0.02)
    with _peer(source, certificate) as peer:
        request = read_record(peer[1])
        time.sleep(0.03)
        peer[0].sendall(
            encode_record(
                {
                    "type": "batch",
                    "seq": request["seq"],
                    "messages": [_HELLO, _frame()],
                    "disconnected": False,
                    "generation": 0,
                }
            )
        )
        with pytest.raises(EOFError):
            read_record(peer[1])
        assert source._store.received == 0
        assert _next_frame(source) is None


def test_receiver_discards_stale_queue_and_emits_disconnect(
    source, certificate, monkeypatch
):
    now = time.perf_counter_ns()
    monkeypatch.setattr(remote_quest.time, "perf_counter_ns", lambda: now)
    with _peer(source, certificate) as peer:
        _send(peer, [_HELLO, _frame(1), _frame(2)])
        assert read_record(peer[1])["type"] == "pull"
        now += int(remote_quest.MAX_ROUND_TRIP_S * 1e9) + 1
        boundary = source.poll()
        assert boundary is not None and boundary.frame is None
        assert boundary.capture_ns == now
        assert source.poll() is None


def test_receiver_overflow_places_boundary_before_equal_timestamp_suffix(
    source, certificate, monkeypatch
):
    monkeypatch.setattr(remote_quest, "MAX_PENDING_FRAMES", 4)
    now = time.perf_counter_ns()
    monkeypatch.setattr(remote_quest.time, "perf_counter_ns", lambda: now)
    with _peer(source, certificate) as peer:
        _send(peer, [_HELLO] + [_frame(i) for i in range(6)])
        _wait(lambda: source._store.received == 6)
        observations = [source.poll() for _ in range(4)]
        assert observations[0].frame is None
        scale = 1_000_000_000
        assert [
            item.frame[SOURCE_CLIENT_CAPTURE_NS_KEY] for item in observations[1:]
        ] == [3 * scale, 4 * scale, 5 * scale]
        assert all(item.capture_ns == now for item in observations)


def test_stop_wakes_authenticated_peer_and_is_idempotent(source, certificate):
    with _peer(source, certificate) as peer:
        read_record(peer[1])
        source.stop()
        with pytest.raises(EOFError):
            read_record(peer[1])
    source.stop()
    assert source.poll() is None


def _relay(source, certificate, *, require_physical_wake=False):
    return quest_relay.QuestRelay(
        server="localhost",
        port=source.port,
        ca_cert=certificate[0],
        token=_TOKEN,
        webxr=cast(Any, _FakeWebXRProducer(wake_ready=not require_physical_wake)),
    )


def test_relay_drops_old_backlog_and_emits_boundary(source, certificate, monkeypatch):
    relay = _relay(source, certificate)
    relay._consume(_HELLO)
    relay._consume(_frame(1))
    future = time.monotonic() + 1
    monkeypatch.setattr(quest_relay.time, "monotonic", lambda: future)
    relay._consume(_frame(2))
    batch = relay._batch(0)
    assert batch["disconnected"]
    assert batch["messages"] == [_HELLO, _frame(2)]


def test_relay_suspension_cannot_requeue_an_inflight_controller_record(
    source, certificate
):
    relay = _relay(source, certificate)
    relay._consume(_HELLO)
    relay._consume(_frame(1))

    class InflightRecord(bytes):
        def decode(self, *args, **kwargs):
            # Suspend between receiving the record and locking its queue.
            relay._webxr.wake_ready = False
            relay._disconnect()
            return super().decode(*args, **kwargs)

    relay._consume(InflightRecord(_frame(2).encode()))
    relay._consume(_frame(3))
    batch = relay._batch(0)
    assert batch["disconnected"]
    assert batch["messages"] == []
    assert relay._hello is None


def test_relay_overflow_is_bounded_and_preserves_fresh_suffix(
    source, certificate, monkeypatch
):
    monkeypatch.setattr(quest_relay, "MAX_PENDING_FRAMES", 4)
    relay = _relay(source, certificate)
    relay._consume(_HELLO)
    for capture in range(6):
        relay._consume(_frame(capture))
    batch = relay._batch(0)
    assert batch["disconnected"]
    assert batch["messages"] == [_HELLO, _frame(3), _frame(4), _frame(5)]


class _FakeWebXRProducer:
    def __init__(self, *, wake_ready=True):
        self.wake_ready = wake_ready
        self.lifecycle = []
        self.on_message: Callable[[str | bytes], None] | None = None
        self.on_disconnect: Callable[[], None] | None = None

    def start(self, on_message, on_disconnect):
        self.on_message = on_message
        self.on_disconnect = on_disconnect
        self.lifecycle.append("start")

    def check_alive(self):
        pass

    def stop(self):
        self.lifecycle.append("stop")


@pytest.mark.parametrize("wait_for_wake", [False, True])
def test_webxr_relay_forwards_raw_isaac_frames_and_stops_on_dgx_disconnect(
    source, certificate, monkeypatch, wait_for_wake
):
    producer = _FakeWebXRProducer(wake_ready=not wait_for_wake)
    relay = quest_relay.QuestRelay(
        server="localhost",
        port=source.port,
        ca_cert=certificate[0],
        token=_TOKEN,
        webxr=cast(Any, producer),
    )

    batches = []
    original_batch = relay._batch

    def batch(sequence):
        result = original_batch(sequence)
        batches.append(result)
        return result

    monkeypatch.setattr(relay, "_batch", batch)
    received = []
    original_consume = source._store.consume

    def consume(message):
        received.append(message)
        original_consume(message)

    monkeypatch.setattr(source._store, "consume", consume)
    errors = []

    def run():
        try:
            relay.run()
        except (EOFError, OSError) as error:
            errors.append(error)

    thread = threading.Thread(target=run)
    thread.start()
    try:
        _wait(lambda: producer.lifecycle == ["start"] and len(batches) >= 5)
        assert producer.on_message is not None
        if wait_for_wake:
            producer.on_message(_HELLO)
            producer.on_message(_frame(12.25))
            _wait(lambda: len(batches) >= 15)
            assert thread.is_alive()
            assert not errors
            assert all(not item["messages"] for item in batches)
            assert received == []
            assert _next_frame(source) is None
            producer.wake_ready = True
        message = _frame(12.5)
        producer.on_message(_HELLO.encode())
        producer.on_message(message)
        observation = _wait(lambda: _next_frame(source))
        assert received == [_HELLO, message]
        assert observation.frame[SOURCE_CLIENT_CAPTURE_NS_KEY] == 12_500_000_000
        assert source._store.protocol == ISAAC_TELEOP_BRIDGE_PROTOCOL_VERSION
        source.stop()
        thread.join(timeout=5)
        assert not thread.is_alive()
        assert producer.lifecycle == ["start", "stop"]
    finally:
        relay.stop()
        thread.join(timeout=5)


def test_webxr_relay_auth_failure_never_starts_producer(source, certificate):
    producer = _FakeWebXRProducer()
    relay = quest_relay.QuestRelay(
        server="localhost",
        port=source.port,
        ca_cert=certificate[0],
        token="wrong" * 12,
        webxr=cast(Any, producer),
    )
    with pytest.raises(ValueError, match="authentication failed"):
        relay.run()
    assert producer.lifecycle == ["stop"]
    assert producer.on_message is None
    assert source._store.received == 0


@pytest.mark.parametrize(
    "wake_options,require_physical_wake",
    [
        ([], True),
        (["--require-physical-wake"], True),
        (["--no-require-physical-wake"], False),
    ],
)
def test_relay_cli_passes_physical_wake_option(
    monkeypatch, wake_options, require_physical_wake
):
    arguments = [
        "flashdreams-quest-relay",
        "--server",
        "dgx",
        "--ca-cert",
        "ca.pem",
        "--cloudxr-port",
        "48323",
        "--isaac-teleop-accept-eula",
        *wake_options,
    ]
    monkeypatch.setattr("sys.argv", arguments)
    monkeypatch.setenv("FLASHDREAMS_CONTROLLER_TOKEN", _TOKEN)
    options = {}

    class Relay:
        def __init__(self, **kwargs):
            options.update(kwargs)

        def run(self):
            pass

        def stop(self):
            pass

    monkeypatch.setattr(quest_relay, "QuestRelay", Relay)
    quest_relay.main()
    assert options["webxr"]._args.require_physical_wake is require_physical_wake
    assert options["webxr"]._args.cloudxr_port == 48323


def test_relay_stop_during_connect_does_not_launch_usb(
    source, certificate, monkeypatch
):
    relay = _relay(source, certificate)
    original = socket.create_connection

    def stopped_connect(*args, **kwargs):
        connection = original(*args, **kwargs)
        relay.stop()
        return connection

    monkeypatch.setattr(quest_relay.socket, "create_connection", stopped_connect)
    relay.run()
    assert relay._connection.fileno() == -1
