# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Record a live stub session, then replay it without the engine."""

from __future__ import annotations

import time
from pathlib import Path

import pytest
import torch
from omnidreams_game_engine.api import v0
from omnidreams_game_engine.api.v0.driver import DISCONNECTED, IDLE
from omnidreams_game_engine.api.v0.recorder import RX, TX
from omnidreams_game_engine.api.v0.stub_engine import StubEngine, StubScript
from omnidreams_game_engine.api.v0.testing import (
    FakePipeline,
    PoseConditionRenderer,
    SeedSceneProvider,
)

pytestmark = pytest.mark.ci_cpu


def _driver(
    source: v0.EngineSource, pipeline: FakePipeline
) -> v0.ExternalSessionDriver:
    return v0.ExternalSessionDriver(
        source=source,
        pipeline=pipeline,
        scene_provider=SeedSceneProvider(),
        condition_renderer=PoseConditionRenderer(),
    )


def _boundaries(
    steps: list[v0.DriverStep],
) -> list[tuple[int, int, tuple[int, ...], tuple[int, ...]]]:
    return [
        (step.epoch, step.ar_index, tuple(step.chunk.g_range), step.chunk.ticks)
        for step in steps
    ]


def _without_stamps(header: dict) -> dict:
    return {k: v for k, v in header.items() if k not in ("t_host_us", "bytes")}


def _record_live_session(directory: Path) -> tuple[list[v0.DriverStep], FakePipeline]:
    recorder = v0.PacketRecorder(directory, meta={"test": "replay"})
    link = v0.TcpEngineLink(
        v0.FluStateCodec(), host="127.0.0.1", port=0, recorder=recorder
    )
    pipeline = FakePipeline()
    driver = _driver(link, pipeline)
    script = StubScript(
        end_after_frames=110, restart_at_ticks=(44,), hold_at_tick=90, hold_s=0.2
    )
    stub = StubEngine(*link.address, script=script).start()
    steps: list[v0.DriverStep] = []
    try:
        deadline = time.monotonic() + 20.0
        while time.monotonic() < deadline:
            step = driver.poll(0.02)
            if step is not None:
                steps.append(step)
            if (
                stub.join(0.0)
                and driver.stats.ends == 1
                and driver.state in (IDLE, DISCONNECTED)
            ):
                break
        else:
            raise AssertionError("live session did not finish")
        assert stub.failure is None
    finally:
        stub.stop()
        stub.join(5.0)
        driver.close()
        link.close()
        recorder.close()
    return steps, pipeline


def test_replay_reproduces_the_live_session_without_the_engine(tmp_path: Path) -> None:
    live_steps, live_pipeline = _record_live_session(tmp_path)
    assert {step.epoch for step in live_steps} == {1, 2}

    log = v0.PacketLogReader(tmp_path)
    assert log.meta["protocol"] == v0.PROTOCOL and log.meta["test"] == "replay"
    received = log.received()
    assert [r.header["type"] for r in received[:2]] == ["hello", "start"]
    seeds = [r for r in received if r.header.get("seed")]
    assert [len(r.payload) for r in seeds] == [8 * 4 * 3, 8 * 4 * 3]
    assert all(r.host_sends is not None for r in received if r.header["type"] != "pong")
    assert any(r.header["type"] == "hold" for r in received)

    replay = v0.PacketLogSource(log, v0.FluStateCodec())
    replay_pipeline = FakePipeline()
    driver = _driver(replay, replay_pipeline)
    replay_steps: list[v0.DriverStep] = []
    for _ in range(10_000):
        step = driver.poll(0.0)
        if step is not None:
            replay_steps.append(step)
        elif replay.exhausted:
            break
    driver.close()
    assert replay.exhausted, "the replay stalled"

    # Same chunks of the same epochs, from the same frames and ticks.
    assert _boundaries(replay_steps) == _boundaries(live_steps)
    # The model received the same conditioning.
    assert len(replay_pipeline.inputs) == len(live_pipeline.inputs)
    for got, want in zip(replay_pipeline.inputs, live_pipeline.inputs, strict=True):
        assert torch.equal(got, want)
    # And the host sent the same grants, seeds requests and frames.
    live_sent = [
        _without_stamps(r.header)
        for r in log.sent()
        if r.header["type"]
        in ("frame_req", "seed_req", "gen_frame", "cancel", "config")
    ]
    replay_sent = [
        _without_stamps(header)
        for header, _size in replay.sent
        if header["type"] in ("frame_req", "seed_req", "gen_frame", "cancel", "config")
    ]
    # The live link stops sending once the engine has gone; the replay sends everything.
    assert len(live_sent) > 100
    assert replay_sent[: len(live_sent)] == live_sent


def test_a_log_is_readable_before_close_and_skips_an_unfinished_line(
    tmp_path: Path,
) -> None:
    recorder = v0.PacketRecorder(tmp_path)
    recorder.record(RX, {"type": "frame"}, b"seed", host_sends=0)
    with (tmp_path / "packets.jsonl").open("a", encoding="utf-8") as index:
        index.write('{"seq": 1, "dir": "rx", "hea')
    records = list(v0.PacketLogReader(tmp_path))
    assert [(r.header["type"], r.payload) for r in records] == [("frame", b"seed")]
    recorder.close()


def test_replay_refuses_a_log_with_a_host_initiated_restart(tmp_path: Path) -> None:
    recorder = v0.PacketRecorder(tmp_path)
    recorder.record(TX, {"type": "restart", "reason": "refresh"})
    recorder.close()
    with pytest.raises(ValueError, match="request_refresh"):
        v0.PacketLogSource(tmp_path, v0.FluStateCodec())


def test_recorder_writes_both_directions_and_omits_tx_payloads_by_default(
    tmp_path: Path,
) -> None:
    recorder = v0.PacketRecorder(tmp_path)
    recorder.record(
        RX, {"type": "frame", "epoch": 1, "g": 0, "tick": 0}, b"seed", host_sends=0
    )
    recorder.record(TX, {"type": "gen_frame"}, b"pixels")
    recorder.close()
    records = list(v0.PacketLogReader(tmp_path))
    assert [(r.direction, r.payload, r.host_sends) for r in records] == [
        (RX, b"seed", 0),
        (TX, b"", None),
    ]
    full = v0.PacketRecorder(tmp_path / "full", record_tx_payloads=True)
    full.record(TX, {"type": "gen_frame"}, b"pixels")
    full.close()
    assert [r.payload for r in v0.PacketLogReader(tmp_path / "full")] == [b"pixels"]
