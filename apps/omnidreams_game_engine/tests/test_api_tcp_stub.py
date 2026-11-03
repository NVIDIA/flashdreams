# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A session driven by the stub external engine over TCP on localhost.

The host side is the real layer (TcpEngineLink, ExternalSessionDriver,
ExternalEngineSession, WorldModelRollout) with a CPU pipeline and renderer.
"""

from __future__ import annotations

import socket
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import pytest
from omnidreams_game_engine.api import v0
from omnidreams_game_engine.api.v0 import framing
from omnidreams_game_engine.api.v0.driver import DISCONNECTED, IDLE
from omnidreams_game_engine.api.v0.stub_engine import StubEngine, StubScript
from omnidreams_game_engine.api.v0.testing import (
    FakePipeline,
    PoseConditionRenderer,
    SeedSceneProvider,
)

pytestmark = pytest.mark.ci_cpu

LEAD = 2
DEADLINE_S = 20.0


@dataclass
class Host:
    link: v0.TcpEngineLink
    driver: v0.ExternalSessionDriver
    pipeline: FakePipeline
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
    pipeline = FakePipeline()
    driver = v0.ExternalSessionDriver(
        source=link,
        pipeline=pipeline,
        scene_provider=SeedSceneProvider(),
        condition_renderer=PoseConditionRenderer(),
        config=v0.DriverConfig(lead_chunks=LEAD),
    )
    try:
        yield Host(link=link, driver=driver, pipeline=pipeline)
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
    # Every generated frame comes back once, in order, with its id and the tick of its state.
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
    assert host.pipeline.caches == 2
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


def test_a_hello_after_close_is_not_accepted() -> None:
    link = v0.TcpEngineLink(v0.FluStateCodec(), port=0)
    with socket.create_connection(link.address, timeout=5.0) as sock:
        sock.settimeout(5.0)
        deadline = time.monotonic() + 5.0
        while not any(t.name == "engine-link-reader" for t in threading.enumerate()):
            assert time.monotonic() < deadline, "the link did not accept"
            time.sleep(0.01)
        link.close()
        framing.send_message(
            sock, {"type": "hello", "session": "s", "api": v0.PROTOCOL}
        )
        _read_to_eof(sock)


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
    # Asked for at tick 20 and stopped at tick 90: only the chunks generated in between carry map frames.
    assert len(views) < len(stub.summary.gen_frames)
    assert host.driver.stats.view_frames_sent == len(views)
    assert sum(step.views_ms > 0.0 for step in host.steps) >= 1


def test_unknown_host_views_are_ignored() -> None:
    script = StubScript(end_after_frames=40, views_at_ticks=((0, ("thermal",)),))
    with _host() as host, _stub(host, script) as stub:
        host.run(_finished(host, stub))
    assert stub.summary.view_frames == []
    assert host.driver.views == frozenset()
