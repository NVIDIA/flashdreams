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

import gc
import io
import json
import queue
import shutil
import socket
import ssl
import subprocess
import threading
import time
import weakref
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any, Callable, cast

import pytest
from flashdreams_controller_input import quest_relay, remote_quest
from flashdreams_controller_input.controller_source_v2_window import (
    ControllerSourceWindow,
)
from flashdreams_controller_input.controller_types import (
    ISAAC_TELEOP_BRIDGE_PROTOCOL_VERSION,
    SUPPORTED_ISAAC_TELEOP_VERSION,
)
from flashdreams_controller_input.events import (
    ControllerFrameEvent,
    ControllerStatusEvent,
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

from flashdreams.runtime_v2.session_desc import SessionDesc
from flashdreams.runtime_v2.user_input_events import UserInputEvents

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


def _start_paired(source, certificate):
    """Show a code to one viewer and claim it like a laptop entering it."""
    source.begin_session()
    source.set_viewer(object())
    code = source.pair()
    source.start()
    with _peer(source, certificate, code=code):
        pass
    _wait(lambda: source._connection is None)
    while source.poll() is not None:
        pass


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
    try:
        _start_paired(source, certificate)
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
def _connect(source, certificate, credential, token=_TOKEN, hostname="localhost"):
    context = ssl.create_default_context(cafile=certificate[0])
    with socket.create_connection(("127.0.0.1", source.port), timeout=2) as raw:
        with context.wrap_socket(raw, server_hostname=hostname) as connection:
            with connection.makefile("rb") as stream:
                connection.sendall(
                    encode_record(
                        {"type": "auth", "version": 3, "token": token, **credential}
                    )
                )
                yield connection, stream


def _reply(source, certificate, **credential):
    with _connect(source, certificate, credential) as (_, stream):
        return read_record(stream)


@contextmanager
def _peer(source, certificate, token=_TOKEN, hostname="localhost", **credential):
    """Authenticate with ``code=`` or ``key=``; default to the laptop's saved key."""
    credential = credential or {"key": source._test_key}
    with _connect(source, certificate, credential, token, hostname) as peer:
        reply = read_record(peer[1])
        assert set(reply) == {"type", "key"} and reply["type"] == "ready"
        source._test_key = reply["key"]
        yield peer


def _send(peer, messages, *, disconnected=False, generation=0, request=None):
    connection, stream = peer
    request = request or read_record(stream)
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
    try:
        _start_paired(source, certificate)
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


def _relay(source, certificate, *, require_physical_wake=False, **options):
    options.setdefault("code", "unused")
    return quest_relay.QuestRelay(
        server="localhost",
        port=source.port,
        ca_cert=certificate[0],
        token=_TOKEN,
        webxr=cast(Any, _FakeWebXRProducer(wake_ready=not require_physical_wake)),
        **options,
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
def test_webxr_relay_forwards_raw_isaac_frames_and_retries_after_dgx_disconnect(
    source, certificate, monkeypatch, wait_for_wake
):
    producer = _FakeWebXRProducer(wake_ready=not wait_for_wake)
    relay = quest_relay.QuestRelay(
        server="localhost",
        port=source.port,
        ca_cert=certificate[0],
        token=_TOKEN,
        code=source.pair(),
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
        # A paired relay keeps the headset and retries until the operator stops it.
        time.sleep(quest_relay.RECONNECT_DELAY_S * 1.5)
        assert thread.is_alive() and producer.lifecycle == ["start"]
        relay.stop()
        thread.join(timeout=5)
        assert not thread.is_alive()
        assert not errors
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
        code=source.pair(),
        webxr=cast(Any, producer),
    )
    with pytest.raises(ValueError, match="authentication failed"):
        relay.run()
    assert producer.lifecycle == ["stop"]
    assert producer.on_message is None
    assert source._store.received == 0


def test_relay_reconnects_with_its_key_and_asks_for_the_next_code(
    source, certificate, monkeypatch
):
    monkeypatch.setattr(quest_relay, "RECONNECT_DELAY_S", 0.01)
    codes: queue.Queue[str] = queue.Queue()
    asked = []

    def ask_code():
        asked.append(True)
        return codes.get(timeout=5)

    relay = _relay(source, certificate, code=source.pair(), ask_code=ask_code)
    thread = threading.Thread(target=relay.run)
    thread.start()
    try:
        first = _wait(lambda: source._authority is not None and source._connection)
        key = relay._key
        assert key is not None
        first.shutdown(socket.SHUT_RDWR)  # A network drop.
        _wait(lambda: source._connection not in (None, first))
        assert asked == [] and relay._key == key
        codes.put(source.pair())  # The operator shows the next code.
        _wait(lambda: relay._key not in (None, key) and source._authority)
        assert asked == [True]
        assert relay._webxr.lifecycle == ["start"]
    finally:
        relay.stop()
        codes.put("")
        thread.join(timeout=5)
    assert not thread.is_alive()


def test_paired_relay_retries_a_handshake_the_server_drops(
    source, certificate, monkeypatch, capsys
):
    monkeypatch.setattr(quest_relay, "RECONNECT_DELAY_S", 0.01)
    relay = _relay(source, certificate, code=source.pair())
    thread = threading.Thread(target=relay.run)
    output: list[str] = []

    def printed(text: str, count: int = 1) -> bool:
        output.append(capsys.readouterr().out)
        return "".join(output).count(text) >= count

    thread.start()
    try:
        first = _wait(lambda: source._authority is not None and source._connection)
        key, token = relay._key, source._token
        # The server closes reconnects before answering, as while it restarts.
        monkeypatch.setattr(source, "_token", b"0" * len(token))
        first.shutdown(socket.SHUT_RDWR)
        _wait(lambda: printed("authentication rejected"))
        monkeypatch.setattr(source, "_token", token)
        _wait(lambda: printed("[quest-relay] paired", 2))
        assert relay._key == key and thread.is_alive()
    finally:
        relay.stop()
        thread.join(timeout=5)
    assert not thread.is_alive()


def test_relay_without_a_terminal_needs_a_code(source, certificate):
    relay = _relay(source, certificate, code="")
    with pytest.raises(ValueError, match="pass --code"):
        relay.run()
    assert relay._webxr.lifecycle == ["stop"]


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


def _claim_offline(source):
    code = source.pair()
    with source.input_guard():
        return source._claim_locked({"code": code})


def test_shared_token_alone_cannot_control_and_codes_need_a_viewer(
    source, certificate, capsys
):
    assert _reply(source, certificate) == {"type": "rejected"}
    assert _reply(source, certificate, key="guessed-" * 5) == {"type": "rejected"}
    source.set_viewer(None)
    with pytest.raises(ValueError, match="connected viewer"):
        source.pair()
    assert _next_frame(source) is None
    output = capsys.readouterr()
    assert source._test_key not in output.out + output.err
    assert _TOKEN not in output.out + output.err


def test_code_is_single_use_and_voided_after_too_many_wrong_entries(
    source, certificate, capsys
):
    code = source.pair()
    wrong = "ZZZ-ZZZ" if code != "ZZZ-ZZZ" else "YYY-YYY"
    for _ in range(remote_quest.MAX_CODE_ATTEMPTS - 1):
        assert _reply(source, certificate, code=wrong) == {"type": "rejected"}
    # Lower case and look-alike letters still match the displayed code.
    loose = code.lower().replace("0", "o").replace("1", "l")
    with _peer(source, certificate, code=loose):
        pass
    assert source.pairing_code == ""
    assert _reply(source, certificate, code=code) == {"type": "rejected"}
    code = source.pair()
    for _ in range(remote_quest.MAX_CODE_ATTEMPTS):
        assert _reply(source, certificate, code=wrong) == {"type": "rejected"}
    assert source.pairing_code == ""
    assert _reply(source, certificate, code=code) == {"type": "rejected"}
    assert "press P for a new code" in capsys.readouterr().out
    assert code not in capsys.readouterr().out


def test_unclaimed_code_expires(source, certificate, monkeypatch):
    code = source.pair()
    expires = source._code.expires_at
    monkeypatch.setattr(remote_quest.time, "monotonic", lambda: expires)
    assert source.pairing_code == ""
    assert _reply(source, certificate, code=code) == {"type": "rejected"}


@pytest.mark.parametrize("transition", ["revoke", "viewer", "new code"])
def test_authority_transition_rejects_reconnect_and_clears_pending(
    source, certificate, transition
):
    with _peer(source, certificate) as peer:
        _send(peer, [_HELLO, _frame(20)])
        _wait(lambda: source._store.received == 1)
        if transition == "revoke":
            source.revoke()
        elif transition == "viewer":
            source.set_viewer(object())
        else:
            source.pair()
        with pytest.raises((EOFError, OSError)):
            # A pull may already have been sent before revocation.
            while True:
                read_record(peer[1])
        assert _next_frame(source) is None
    assert _reply(source, certificate, key=source._test_key) == {"type": "rejected"}


def test_scene_change_keeps_the_laptop_paired_but_fences_old_poses(source, certificate):
    with _peer(source, certificate) as peer:
        _send(peer, [_HELLO, _frame(20)])
        earlier = _wait(lambda: _next_frame(source))
        inflight = read_record(peer[1])  # The next pull left before the change.
        source.begin_session()
        assert not source.allows(earlier)
        _send(peer, [_frame(21)], request=inflight)  # An old-scene pose: dropped.
        _send(peer, [_frame(22)])
        later = _wait(lambda: _next_frame(source))
        assert later.frame[SOURCE_CLIENT_CAPTURE_NS_KEY] * 20 == (
            earlier.frame[SOURCE_CLIENT_CAPTURE_NS_KEY] * 22
        )
        assert source.allows(later)
    with _peer(source, certificate) as peer:
        _send(peer, [_HELLO, _frame(23)])
        assert source.allows(_wait(lambda: _next_frame(source)))


def test_missing_viewer_pauses_input_and_only_a_new_viewer_revokes(source, certificate):
    viewer = source._viewer
    with _peer(source, certificate) as peer:
        _send(peer, [_HELLO, _frame(20)])
        paused = _wait(lambda: _next_frame(source))
        check = source.authority_check(paused.authority_epoch)
        source.set_viewer(None)
        # Input buffered before the disconnect cannot reach the model either.
        assert not source.allows(paused) and not check()
        source.set_viewer(viewer)  # The same paired browser reloaded.
        assert source.allows(paused) and check()
        _send(peer, [_frame(21)])
        assert source.allows(_wait(lambda: _next_frame(source)))
    source.set_viewer(object())
    assert _reply(source, certificate, key=source._test_key) == {"type": "rejected"}


def test_switching_laptops_locks_out_the_old_key_and_new_laptop_reconnects(
    source, certificate
):
    with _peer(source, certificate) as laptop_a:
        _send(laptop_a, [_HELLO, _frame(200)])
        previous = _wait(lambda: _next_frame(source))
    key_a = source._test_key
    code = source.pair()
    assert not source.allows(previous)
    # Laptop A still has SSH access and its key, but no longer the authority.
    assert _reply(source, certificate, key=key_a) == {"type": "rejected"}
    with _peer(source, certificate, code=code) as laptop_b:
        _send(laptop_b, [_HELLO, _frame(1, session_id="producer-chosen-session")])
        current = _wait(lambda: _next_frame(source))
        assert current.generation > previous.generation
        assert current.authority_epoch != previous.authority_epoch
        assert source.allows(current)
    key_b = source._test_key
    assert key_b != key_a
    assert _reply(source, certificate, key=key_a) == {"type": "rejected"}
    # B reconnects with its key: no new code and no demo restart.
    with _peer(source, certificate, key=key_b) as laptop_b:
        _send(laptop_b, [_HELLO, _frame(2)])
        assert source.allows(_wait(lambda: _next_frame(source)))


@pytest.mark.asyncio
@pytest.mark.parametrize("bearer", [True, False])
async def test_browser_reload_keeps_a_bearer_pairing_but_not_a_no_login_one(
    certificate, monkeypatch, bearer
):
    pytest.importorskip("aiortc")
    from flashdreams.runtime_v2.serving import webrtc_server

    monkeypatch.delenv("FLASHDREAMS_WEBRTC_TOKEN_FILE", raising=False)
    monkeypatch.delenv("FLASHDREAMS_WEBRTC_ALLOWED_ORIGINS", raising=False)
    monkeypatch.setattr(
        webrtc_server.WebRTCServer, "_run_server", lambda self: self._started.set()
    )
    login: dict[str, Any] = (
        {"auth_token": "A" * 43, "allowed_origins": ("https://viewer.example",)}
        if bearer
        else {}
    )
    server = webrtc_server.WebRTCServer(port=8080, **login)
    source = RemoteQuestSource(
        host="127.0.0.1",
        port=0,
        certfile=certificate[0],
        keyfile=certificate[1],
        token=_TOKEN,
    )
    server.register_viewer_callback(source.set_viewer)
    server._set_viewer_connected(True)
    code = source.pair()
    source.start()
    try:
        with _peer(source, certificate, code=code):
            pass
        server._set_viewer_connected(False)
        server._set_viewer_connected(True)
        if bearer:
            with _peer(source, certificate) as peer:
                _send(peer, [_HELLO, _frame(3)])
                assert source.allows(_wait(lambda: _next_frame(source)))
        else:
            reply = _reply(source, certificate, key=cast(Any, source)._test_key)
            assert reply == {"type": "rejected"}
    finally:
        source.stop()
        server.close()


def test_retained_authority_check_is_weak_and_cannot_revive(certificate):
    cert, key = certificate
    owner = RemoteQuestSource(certfile=cert, keyfile=key, token=_TOKEN)
    owner.set_viewer(object())
    grant = _claim_offline(owner)
    check = owner.authority_check(grant.epoch)
    assert check()
    owner.revoke()
    assert not check()
    grant = _claim_offline(owner)
    assert not check()
    current = owner.authority_check(grant.epoch)
    assert current()
    reference = weakref.ref(owner)
    del owner, grant
    gc.collect()
    assert reference() is None
    assert not current()


def test_inflight_pose_cannot_repopulate_revoked_queue(
    source, certificate, monkeypatch
):
    entered, resume = threading.Event(), threading.Event()
    accept = source._accept_frame

    def paused_accept(*args):
        entered.set()
        assert resume.wait(2)
        accept(*args)

    monkeypatch.setattr(source, "_accept_frame", paused_accept)
    try:
        with _peer(source, certificate) as peer:
            _send(peer, [_HELLO, _frame()])
            assert entered.wait(2)
            source.revoke()
            resume.set()
            with pytest.raises(EOFError):
                read_record(peer[1])
            assert _next_frame(source) is None
    finally:
        resume.set()


def test_false_authorized_pose_reaches_input_but_revocation_fences_retained_events(
    source, certificate
):
    viewer_callbacks = []
    output: Any = SimpleNamespace(
        register_viewer_callback=viewer_callbacks.append,
        open=lambda _: None,
        get_user_input_events=lambda: UserInputEvents([]),
        close=lambda: None,
    )
    window = ControllerSourceWindow(
        output_window=output,
        source=source,
        remote_control=source,
        source_name="Quest",
        pose_convention=WRIST_PROXY_POSE_CONVENTION,
        source_thread_name="authority-test",
        source_failure_message="input failed",
    )
    try:
        window.open(SessionDesc())
        viewer_callbacks[0](object())
        code = source.pair()
        false_pose = json.loads(_frame())
        false_pose["controllers"]["left"]["position"] = [1.25, 1.0, -0.3]
        false_pose["controllers"]["left"]["grip"] = 1.0
        with _peer(source, certificate, code=code) as peer:
            _send(peer, [_HELLO, json.dumps(false_pose)])
            _wait(lambda: any(item.frame is not None for item in window._observations))
            events = window.get_user_input_events().get_events()
            retained = next(
                item
                for item in events
                if isinstance(item, ControllerFrameEvent) and item.action == "state"
            )
            before = retained.to_controller_frame()
            assert before["left"]["position"] == pytest.approx([1.25, 1.0, -0.3])
            assert before["left"]["grip"] == 1.0
            assert before["left"]["tracked"]
            _send(peer, [_frame(2)])
            _wait(lambda: any(item.frame is not None for item in window._observations))
            source.revoke()
            assert not any(item.frame is not None for item in window._observations)
            assert not retained.to_controller_frame()["left"]["tracked"]
            assert all(
                item.action == "disconnected"
                for item in window.get_user_input_events().get_events()
                if isinstance(item, ControllerFrameEvent)
            )
    finally:
        window.close()


def test_operator_controls_and_revoke_preserve_local_input(source):
    from flashdreams_controller_input.controller_types import Observation
    from numpy import uint64

    from flashdreams.runtime_v2.user_input_event import (
        KeyboardInputState,
        KeyboardUserInputEvent,
    )

    queued = []
    viewer_callbacks = []
    output: Any = SimpleNamespace(
        register_viewer_callback=viewer_callbacks.append,
        open=lambda _: None,
        get_user_input_events=lambda: UserInputEvents(queued.copy()),
        close=lambda: None,
    )
    idle: Any = SimpleNamespace(
        start=lambda: None, poll=lambda: None, stop=lambda: None
    )
    window = ControllerSourceWindow(
        output_window=output,
        source=idle,
        remote_control=source,
        source_name="Quest",
        pose_convention=WRIST_PROXY_POSE_CONVENTION,
        source_thread_name="operator-test",
        source_failure_message="input failed",
    )
    try:
        window.open(SessionDesc())
        viewer_callbacks[0](object())
        queued.append(
            KeyboardUserInputEvent(
                timestamp=uint64(0), key="p", state=KeyboardInputState.PRESSED
            )
        )
        events = window.get_user_input_events().get_events()
        code = source.pairing_code
        assert code
        # The viewer HUD receives the code through the connection status.
        assert any(
            isinstance(item, ControllerStatusEvent) and item.status.pairing_code == code
            for item in events
        )
        window.get_user_input_events()
        assert source.pairing_code == code  # Held P does not continually rotate.
        window._source_generation = 7
        window._remote_active = True
        queued[:] = [
            KeyboardUserInputEvent(
                timestamp=uint64(0), key="r", state=KeyboardInputState.PRESSED
            )
        ]
        events = window.get_user_input_events().get_events()
        assert source._authority is None and source.pairing_code == ""
        assert any(
            isinstance(item, ControllerFrameEvent) and item.action == "disconnected"
            for item in events
        )
        queued.clear()
        window._remote_active = False  # A server-local source now owns input.
        local = Observation(
            {"left": None, "right": None}, time.perf_counter_ns(), "local", 7
        )
        with window._lock:
            window._observations.append(local)
        source.revoke()
        with window._lock:
            assert list(window._observations) == [local]
        viewer_callbacks[0](object())
        queued.append(
            KeyboardUserInputEvent(
                timestamp=uint64(0), key="p", state=KeyboardInputState.PRESSED
            )
        )
        window.get_user_input_events()
        assert source.pairing_code  # New viewer's first P works.
        expires = source._code.expires_at
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(remote_quest.time, "monotonic", lambda: expires)
            window.get_user_input_events()
            assert source.pairing_code == ""  # Held P cannot silently renew.
    finally:
        window.close()
