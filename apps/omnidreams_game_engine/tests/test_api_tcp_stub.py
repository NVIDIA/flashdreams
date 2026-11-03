# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for a host session driven by the stub engine over TCP on localhost."""

from __future__ import annotations

import contextlib
import json
import socket
import struct
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
import torch
from omnidreams_game_engine.api import v0
from omnidreams_game_engine.api.v0 import framing, link_tcp, stub_engine
from omnidreams_game_engine.api.v0.driver import DISCONNECTED, IDLE, SEEDING
from omnidreams_game_engine.api.v0.recorder import TX
from omnidreams_game_engine.api.v0.source import LOST_TYPE
from omnidreams_game_engine.api.v0.stub_engine import StubEngine, StubScript
from omnidreams_game_engine.api.v0.testing import (
    FakePipeline,
    PoseConditionRenderer,
    SeedSceneProvider,
)

pytestmark = pytest.mark.ci_cpu

LEAD = 2
DEADLINE_S = 20.0


def _driver(source: v0.EngineSource) -> v0.ExternalSessionDriver:
    return v0.ExternalSessionDriver(
        source=source,
        pipeline=FakePipeline(),
        scene_provider=SeedSceneProvider(),
        condition_renderer=PoseConditionRenderer(),
        config=v0.DriverConfig(lead_chunks=LEAD),
    )


@dataclass
class Host:
    link: v0.TcpEngineLink
    driver: v0.ExternalSessionDriver
    steps: list[v0.DriverStep] = field(default_factory=list)
    held_seen: bool = False

    def run(self, until: Callable[[], bool], deadline_s: float = DEADLINE_S) -> None:
        end = time.monotonic() + deadline_s
        while time.monotonic() < end:
            step = self.driver.poll(0.02)
            if step is not None:
                self.steps.append(step)
            self.held_seen |= self.driver.engine_held
            if until():
                return
        raise AssertionError(f"Host loop timed out in state {self.driver.state}")


@contextmanager
def _host(**link_options: Any) -> Iterator[Host]:
    link = v0.TcpEngineLink(v0.FluStateCodec(), port=0, **link_options)
    driver = _driver(link)
    try:
        yield Host(link=link, driver=driver)
    finally:
        driver.close()
        link.close()


@contextmanager
def _stub(host: Host, script: StubScript) -> Iterator[StubEngine]:
    address, port = host.link.address
    stub = StubEngine(address, port, script=script).start()
    try:
        yield stub
    finally:
        stub.stop()
        assert stub.join(5.0), "stub engine did not stop"
        assert stub.failure is None, stub.failure


def _finished(host: Host, stub: StubEngine) -> Callable[[], bool]:
    return lambda: (
        stub.join(0.0)
        and host.driver.stats.ends >= 1
        and host.driver.state in (IDLE, DISCONNECTED)
    )


def _frame_ids(stub: StubEngine) -> dict[int, list[int]]:
    ids: dict[int, list[int]] = {}
    for epoch, _chunk, g, _tick, _bytes in stub.summary.gen_frames:
        ids.setdefault(epoch, []).append(g)
    return ids


def _assert_grants_within_lead(stub: StubEngine) -> None:
    leads = [chunk - complete for _e, chunk, _g, _t, complete in stub.summary.grants]
    assert leads, "no grants"
    assert max(leads) == LEAD


def test_stub_session_config_grants_and_frames_in_order() -> None:
    with _host() as host, _stub(host, StubScript(end_after_frames=60)) as stub:
        host.run(_finished(host, stub))
    config = stub.summary.configs[0]
    assert config["api"] == v0.PROTOCOL
    assert (config["model_fps"], config["frame_ticks"]) == (30, 2)
    assert config["chunk_plan"] == [5, 8]
    assert config["lead_chunks"] == LEAD
    assert config["return_encoding"] == "raw"
    assert stub.summary.seeds == [(1, 0)]
    _assert_grants_within_lead(stub)
    # Every generated frame comes back once, in order, with its id and its tick.
    ids = _frame_ids(stub)
    assert list(ids) == [1]
    assert ids[1] == list(range(len(ids[1])))
    # The engine runs at most LEAD chunks ahead of generation when it ends.
    assert len(ids[1]) >= 60 - (LEAD + 1) * 8
    timeline = host.driver.timeline
    assert timeline is not None
    for _epoch, chunk, g, tick, size in stub.summary.gen_frames:
        assert tick == 2 * g
        assert chunk == timeline.chunk_of(g)
        assert size == 8 * 4 * 3
    assert [list(step.chunk.g_range) for step in host.steps[:2]] == [
        list(range(5)),
        list(range(5, 13)),
    ]
    assert host.driver.stats.ends == 1


def test_hold_and_release_continue_the_same_epoch() -> None:
    script = StubScript(end_after_frames=70, hold_at_tick=30, hold_s=0.3)
    with _host() as host, _stub(host, script) as stub:
        host.run(_finished(host, stub))
    assert stub.summary.holds == [30]
    assert host.held_seen
    assert not host.driver.engine_held
    assert {step.epoch for step in host.steps} == {1}
    ids = _frame_ids(stub)[1]
    assert ids == list(range(len(ids))) and len(ids) > 30


def test_restart_starts_a_new_epoch_with_a_new_seed() -> None:
    script = StubScript(end_after_frames=90, restart_at_ticks=(40,))
    with _host() as host, _stub(host, script) as stub:
        host.run(_finished(host, stub))
    assert stub.summary.restarts == [40]
    assert stub.summary.seeds == [(1, 0), (2, 40)]
    assert 1 in stub.summary.cancels
    epoch_two = [step for step in host.steps if step.epoch == 2]
    assert epoch_two[0].ar_index == 0
    assert epoch_two[0].chunk.ticks == (40, 42, 44, 46, 48)
    ids = _frame_ids(stub)
    assert ids[2] == list(range(len(ids[2])))
    assert host.driver.stats.seeds == 2
    # One model cache per seed; the pipeline stays loaded.
    assert host.driver.pipeline.caches == 2
    _assert_grants_within_lead(stub)


def test_reconnect_after_a_dropped_connection_starts_a_new_epoch() -> None:
    script = StubScript(end_after_frames=80, drop_at_tick=30)
    with _host() as host, _stub(host, script) as stub:
        host.run(_finished(host, stub))
    assert stub.summary.connects == 2
    assert host.link.connects == 2
    # Either the host noticed the drop first (lost) or the reconnect replaced it.
    assert host.driver.stats.connects == 2
    assert [seed[0] for seed in stub.summary.seeds] == [1, 2]
    assert stub.summary.seeds[1][1] >= 30
    # Epoch 1 may lose its first chunk to the drop; epoch 2 starts from frame 0.
    ids = _frame_ids(stub)
    assert ids[2] == list(range(len(ids[2]))) and len(ids[2]) > 8


def test_a_silent_engine_is_dropped_and_reconnects() -> None:
    script = StubScript(end_after_frames=80, silent_at_tick=30, silent_s=1.5)
    with (
        _host(ping_interval_s=0.05, silence_s=0.6) as host,
        _stub(host, script) as stub,
    ):
        host.run(_finished(host, stub))
    assert "silence" in host.driver.stats.lost_reasons
    assert stub.summary.connects == 2
    assert [seed[0] for seed in stub.summary.seeds] == [1, 2]


def test_the_link_listens_on_loopback_by_default() -> None:
    link = v0.TcpEngineLink(v0.FluStateCodec(), port=0)
    try:
        assert link.address[0] == "127.0.0.1"
    finally:
        link.close()


def _connect(host: Host) -> socket.socket:
    sock = socket.create_connection(host.link.address, timeout=5.0)
    sock.settimeout(5.0)
    return sock


def _raw_hello(address: tuple[str, int], **fields: object) -> dict[str, object]:
    with socket.create_connection(address, timeout=5.0) as sock:
        sock.settimeout(5.0)
        framing.send_message(sock, {"type": "hello", "session": "raw", **fields})
        header, _ = framing.recv_message(sock)
        return header


def _read_to_eof(sock: socket.socket) -> None:
    while sock.recv(1 << 16):
        pass


def _wait_until(condition: Callable[[], bool]) -> None:
    deadline = time.monotonic() + DEADLINE_S
    while not condition():
        assert time.monotonic() < deadline, "timed out"
        time.sleep(0.01)


def test_wrong_version_state_format_and_a_second_engine_are_refused() -> None:
    with _host() as host:
        address = host.link.address
        with socket.create_connection(address, timeout=5.0) as sock:
            sock.settimeout(5.0)
            framing.send_message(sock, {"type": "start", "tick": 0})
            assert framing.recv_message(sock)[0]["code"] == "protocol"
        assert _raw_hello(address, api="omnidreams-engine/1")["code"] == "version"
        assert _raw_hello(address)["code"] == "version"  # a hello without api
        refused = _raw_hello(address, api=v0.PROTOCOL, state_format="other/6")
        assert refused["code"] == "state_format"
        with _stub(host, StubScript()) as stub:
            host.run(
                lambda: host.driver.state != DISCONNECTED and stub.summary.seeds != []
            )
            busy = _raw_hello(address, api=v0.PROTOCOL, session="another-engine")
            assert busy["type"] == "error" and busy["code"] == "busy"
        assert host.link.refusals == [
            "protocol",
            "version",
            "version",
            "state_format",
            "busy",
        ]


def test_a_timing_refusal_closes_the_connection() -> None:
    # No pings, so only the refusal can end the connection.
    with _host(ping_interval_s=60.0) as host:
        with _connect(host) as sock:
            hello = {
                "type": "hello",
                "session": "raw",
                "api": v0.PROTOCOL,
                "sim_hz": 50,
            }
            framing.send_message(sock, hello)
            host.run(lambda: host.link.refusals == ["timing"])
            assert framing.recv_message(sock)[0]["code"] == "timing"
            _read_to_eof(sock)
        with _stub(host, StubScript()) as stub:
            host.run(lambda: stub.summary.seeds != [])
        assert stub.summary.errors == []


def test_the_same_session_replaces_its_open_connection() -> None:
    hello = {"type": "hello", "session": "s", "api": v0.PROTOCOL}
    with _host() as host, _connect(host) as first, _connect(host) as second:
        framing.send_message(first, hello)
        host.run(lambda: host.driver.stats.connects == 1)
        assert framing.recv_message(first)[0]["type"] == "config"
        framing.send_message(second, hello)
        host.run(lambda: host.driver.stats.connects == 2)
        assert framing.recv_message(second)[0]["type"] == "config"
        _read_to_eof(first)
        host.run(lambda: True)
        assert host.link.connects == 2
        assert host.driver.stats.lost_reasons == []
        assert host.driver.state == IDLE


def test_close_closes_a_connection_that_waits_for_its_hello() -> None:
    link = v0.TcpEngineLink(v0.FluStateCodec(), port=0)
    with socket.create_connection(link.address, timeout=5.0) as sock:
        sock.settimeout(5.0)
        _wait_until(lambda: len(link._pending) == 1)
        link.close()
        _read_to_eof(sock)
    assert link.connects == 0


def test_a_hello_that_does_not_arrive_in_time_is_dropped() -> None:
    with _host(hello_timeout_s=0.3) as host, _connect(host) as sock:
        # A header that never completes, sent a byte at a time, each in time
        # for the next read: only a deadline for the whole hello ends it.
        sock.sendall(struct.pack(">I", 1000))
        with pytest.raises((BrokenPipeError, ConnectionResetError)):
            for _ in range(100):
                sock.sendall(b" ")
                time.sleep(0.05)
    assert host.link.connects == 0


def test_connections_that_wait_for_their_hello_are_limited() -> None:
    with _host() as host, contextlib.ExitStack() as sockets:
        waiting = [
            sockets.enter_context(_connect(host))
            for _ in range(link_tcp.MAX_PENDING_HELLOS)
        ]
        _read_to_eof(sockets.enter_context(_connect(host)))
        for sock in waiting:
            sock.settimeout(0.1)
            with pytest.raises(TimeoutError):
                sock.recv(1)


def test_an_engine_that_floods_a_busy_host_is_dropped() -> None:
    hello = {"type": "hello", "session": "s", "api": v0.PROTOCOL}
    with _host(max_inbox_bytes=4096) as host, _connect(host) as sock:
        # The host does not poll while the engine sends.
        with contextlib.suppress(BrokenPipeError, ConnectionResetError):
            framing.send_message(sock, hello)
            for _ in range(10):
                framing.send_message(sock, {"type": "status"}, bytes(1000))
            _read_to_eof(sock)
        host.run(lambda: host.driver.stats.lost_reasons != [])
    assert host.driver.stats.lost_reasons[0].startswith("inbox full")


def test_messages_of_a_replaced_connection_are_dropped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The old connection's reader holds its `end` until the replacement runs.
    read = link_tcp._recv_framed
    release = threading.Event()
    holders: list[threading.Thread] = []

    def read_end_late(sock: socket.socket) -> tuple[dict[str, Any], bytes, int]:
        message = read(sock)
        if message[0]["type"] == "end":
            holders.append(threading.current_thread())
            release.wait(DEADLINE_S)
        return message

    monkeypatch.setattr(link_tcp, "_recv_framed", read_end_late)
    hello = {"type": "hello", "session": "s", "api": v0.PROTOCOL}
    with _host() as host, _connect(host) as first, _connect(host) as second:
        framing.send_message(first, hello)
        host.run(lambda: host.driver.stats.connects == 1)
        framing.send_message(first, {"type": "end"})
        _wait_until(lambda: holders != [])
        framing.send_message(second, hello)
        framing.send_message(second, {"type": "start", "tick": 0})
        host.run(lambda: host.driver.state == SEEDING)
        release.set()
        holders[0].join(5.0)
        host.run(lambda: True)
        assert (host.driver.state, host.driver.stats.ends) == (SEEDING, 0)


def test_an_engine_message_of_the_lost_type_is_ignored() -> None:
    hello = {"type": "hello", "session": "s", "api": v0.PROTOCOL}
    with _host() as host, _connect(host) as sock:
        for header in (hello, {"type": LOST_TYPE}, {"type": "start", "tick": 0}):
            framing.send_message(sock, header)
        host.run(lambda: host.driver.state == SEEDING)
    assert host.driver.stats.lost_reasons == []


def test_a_failed_recording_closes_the_connection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recorder = v0.PacketRecorder(tmp_path)
    record = recorder.record

    def record_engine_messages_only(direction: str, *args: Any, **kwargs: Any) -> None:
        if direction == TX:
            raise OSError("disk full")
        record(direction, *args, **kwargs)

    monkeypatch.setattr(recorder, "record", record_engine_messages_only)
    hello = {"type": "hello", "session": "s", "api": v0.PROTOCOL}
    with _host(recorder=recorder) as host, _connect(host) as sock:
        framing.send_message(sock, hello)
        host.run(lambda: host.driver.stats.lost_reasons != [])
    assert "disk full" in host.driver.stats.lost_reasons[0]


def test_a_status_that_cannot_be_encoded_is_skipped() -> None:
    statuses = iter([{"x": float("nan")}])
    options = {"ping_interval_s": 0.02, "status_interval_s": 0.02}
    with (
        _host(status_provider=lambda: next(statuses, {"x": 1}), **options) as host,
        _stub(host, StubScript(realtime=True)) as stub,
    ):
        host.run(lambda: stub.summary.host_messages.get("status", 0) >= 2)
    assert host.driver.stats.lost_reasons == []


def test_host_refresh_and_forwarded_input_reach_the_engine() -> None:
    with _host() as host, _stub(host, StubScript(end_after_frames=90)) as stub:
        host.run(lambda: host.driver.stats.chunks >= 3)
        last_tick = host.driver.last_tick
        assert last_tick is not None
        host.driver.forward_input("throttle", "down")
        assert host.driver.request_refresh()
        host.run(_finished(host, stub))
    assert stub.summary.host_messages.get("input") == 1
    assert stub.summary.host_messages.get("restart") == 1
    assert [epoch for epoch, _ in stub.summary.seeds] == [1, 2]
    assert stub.summary.seeds[1][1] >= last_tick + 2
    assert 1 in stub.summary.cancels
    epoch_two = [step for step in host.steps if step.epoch == 2]
    assert epoch_two and epoch_two[0].chunk.ticks[0] == stub.summary.seeds[1][1]


def test_map_view_frames_follow_views_and_match_the_generated_frames() -> None:
    script = StubScript(end_after_frames=80, views_at_ticks=((20, ("map",)), (90, ())))
    with _host() as host, _stub(host, script) as stub:
        host.run(_finished(host, stub))
    views = stub.summary.view_frames
    assert views, "no map frames"
    assert {view for view, *_ in views} == {"map"}
    generated = {
        (epoch, g): tick for epoch, _chunk, g, tick, _size in stub.summary.gen_frames
    }
    for _view, epoch, g, tick, size in views:
        # Each map frame names the generated frame it conditioned, with the same tick.
        assert generated[(epoch, g)] == tick
        assert size == 8 * 4 * 3
    # Asked for at tick 20 and stopped at tick 90: later chunks carry no map frames.
    assert max(g for _v, _e, g, _t, _s in views) < max(
        g for _e, _c, g, _t, _s in stub.summary.gen_frames
    )
    assert host.driver.stats.view_frames_sent == len(views)
    assert sum(step.views_ms > 0.0 for step in host.steps) >= 1


def test_unknown_host_views_are_ignored() -> None:
    script = StubScript(end_after_frames=60, views_at_ticks=((20, ("thermal", "map")),))
    with _host() as host, _stub(host, script) as stub:
        host.run(_finished(host, stub))
    assert {view for view, *_ in stub.summary.view_frames} == {"map"}


def _without_stamps(header: dict) -> dict:
    return {k: v for k, v in header.items() if k not in ("t_host_us", "bytes")}


def test_a_recorded_session_replays_without_the_engine(tmp_path: Path) -> None:
    recorder = v0.PacketRecorder(tmp_path, meta={"test": "replay"})
    script = StubScript(
        end_after_frames=110, restart_at_ticks=(44,), hold_at_tick=90, hold_s=0.2
    )
    with _host(recorder=recorder) as live, _stub(live, script) as stub:
        live.run(_finished(live, stub))
    recorder.close()
    assert {step.epoch for step in live.steps} == {1, 2}

    log = v0.PacketLogReader(tmp_path)
    assert log.meta["protocol"] == v0.PROTOCOL and log.meta["test"] == "replay"
    received = log.received()
    assert [r.header["type"] for r in received[:2]] == ["hello", "start"]
    seeds = [r for r in received if r.header.get("seed")]
    assert [len(r.payload) for r in seeds] == [8 * 4 * 3, 8 * 4 * 3]
    assert all(r.host_sends is not None for r in received)
    assert any(r.header["type"] == "hold" for r in received)

    replay = v0.PacketLogSource(log, v0.FluStateCodec())
    driver = _driver(replay)
    steps: list[v0.DriverStep] = []
    for _ in range(10_000):
        step = driver.poll(0.0)
        if step is not None:
            steps.append(step)
        elif replay.exhausted:
            break
    driver.close()
    assert replay.exhausted, "the replay stalled"

    # Same chunks of the same epochs, from the same frames and ticks.
    def boundaries(steps: list[v0.DriverStep]) -> list[tuple]:
        return [(s.epoch, s.ar_index, s.chunk.ticks) for s in steps]

    assert boundaries(steps) == boundaries(live.steps)
    # The model received the same conditioning.
    inputs = driver.pipeline.inputs
    assert len(inputs) == len(live.driver.pipeline.inputs)
    for got, want in zip(inputs, live.driver.pipeline.inputs, strict=True):
        assert torch.equal(got, want)
    # And the host sent the same config, grants, seed requests and frames.
    kept = ("frame_req", "seed_req", "gen_frame", "cancel", "config")
    live_sent = [
        _without_stamps(r.header) for r in log.sent() if r.header["type"] in kept
    ]
    replay_sent = [_without_stamps(h) for h, _size in replay.sent if h["type"] in kept]
    # The live link stops sending once the engine has gone; the replay sends everything.
    assert len(live_sent) > 100
    assert replay_sent[: len(live_sent)] == live_sent


def test_recorded_states_are_looked_up_by_the_nearest_tick(tmp_path: Path) -> None:
    path = tmp_path / "state.jsonl"
    lines = ['{"tick": 0, "v": 0}', "", '{"frame": 10, "v": 1}', '{"tick": 20, "v": 2}']
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    state = stub_engine.jsonl_states(path)
    assert [state(tick)["v"] for tick in (0, 4, 6, 10, 19, 99)] == [0, 0, 1, 1, 2, 2]


def test_the_stub_command_line_runs_a_session(tmp_path: Path) -> None:
    summary = tmp_path / "summary.json"
    with _host() as host:
        args = ["--port", str(host.link.address[1]), "--frames", "20"]
        thread = threading.Thread(
            target=stub_engine.main,
            args=([*args, "--summary-json", str(summary)],),
            daemon=True,
        )
        thread.start()
        host.run(lambda: not thread.is_alive())
    assert json.loads(summary.read_text(encoding="utf-8"))["ended"]
