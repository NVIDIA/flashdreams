# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for the host session flow, driven by scripted engine messages."""

from __future__ import annotations

from collections import deque
from dataclasses import replace
from typing import Any

import pytest
from omnidreams_game_engine.api import v0
from omnidreams_game_engine.api.v0.driver import DISCONNECTED, IDLE, RUNNING, SEEDING
from omnidreams_game_engine.api.v0.messages import PROTOCOL, Input, Message
from omnidreams_game_engine.api.v0.source import Inbound, MessageEngineSource
from omnidreams_game_engine.api.v0.stub_engine import ScriptedDrive
from omnidreams_game_engine.api.v0.testing import (
    FakePipeline,
    PoseConditionRenderer,
    SeedSceneProvider,
)

pytestmark = pytest.mark.ci_cpu

SEED_IMAGE = bytes(8 * 4 * 3)


class _Script(MessageEngineSource):
    """In-memory engine: each fetch delivers the next fed batch."""

    _fetch_blocks = False

    def __init__(self) -> None:
        super().__init__(v0.FluStateCodec())
        self.batches: deque[list[Inbound]] = deque()
        self.sent: list[Message] = []

    def feed(self, *items: dict[str, Any] | tuple[dict[str, Any], bytes]) -> None:
        self.batches.append(
            [
                Inbound(*item) if isinstance(item, tuple) else Inbound(item)
                for item in items
            ]
        )

    def _fetch(self, timeout_s: float) -> list[Inbound]:
        del timeout_s
        return self.batches.popleft() if self.batches else []

    def _transmit(self, message: Message, payload: bytes) -> None:
        del payload
        self.sent.append(message)

    def types(self) -> list[str]:
        return [message.TYPE for message in self.sent]


def _frame(epoch: int, g: int, t0: int = 0, **fields: Any) -> dict[str, Any]:
    tick = t0 + 2 * g
    state = ScriptedDrive().state(tick)
    return {
        "type": "frame",
        "epoch": epoch,
        "g": g,
        "tick": tick,
        "state": state,
        **fields,
    }


def _seed(epoch: int, t0: int = 0, image: bytes = SEED_IMAGE) -> tuple[dict, bytes]:
    header = _frame(epoch, 0, t0, seed=True, encoding="raw", width=8, height=4)
    return header, image


def _running(**config: Any) -> tuple[_Script, v0.ExternalSessionDriver, FakePipeline]:
    script = _Script()
    pipeline = FakePipeline()
    driver = v0.ExternalSessionDriver(
        source=script,
        pipeline=pipeline,
        scene_provider=SeedSceneProvider(),
        condition_renderer=PoseConditionRenderer(),
        config=v0.DriverConfig(**config),
    )
    script.feed(
        {"type": "hello", "session": "s", "api": PROTOCOL}, {"type": "start", "tick": 0}
    )
    driver.poll(0.0)
    script.feed(_seed(1))
    driver.poll(0.0)
    assert driver.state == RUNNING
    return script, driver, pipeline


def _complete_chunk_0(script: _Script, driver: v0.ExternalSessionDriver) -> None:
    script.feed(*(_frame(1, g) for g in range(1, 5)))
    assert driver.poll(0.0) is not None


def test_lead_chunks_must_be_at_least_one() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        v0.DriverConfig(lead_chunks=0)


def test_an_unreadable_frame_is_answered_and_re_anchors_the_epoch() -> None:
    script, driver, _ = _running()
    script.feed(_frame(1, 1, state={"rig_to_world": [0.0] * 15, "vehicle": {}}))
    assert driver.poll(0.0) is None
    assert script.stats.bad_states == 1
    assert script.types()[-3:] == ["error", "cancel", "seed_req"]
    assert (driver.state, driver.epoch) == (SEEDING, 2)


@pytest.mark.parametrize(
    "header",
    [
        {"type": "frame", "epoch": 1, "g": "x", "tick": 2, "state": {}},
        _frame(1, 5, seed=True),
    ],
)
def test_a_malformed_frame_or_seed_re_anchors(header: dict[str, Any]) -> None:
    script, driver, _ = _running()
    script.feed(header)
    driver.poll(0.0)
    assert script.stats.bad_messages == 1
    assert script.types()[-3:] == ["error", "cancel", "seed_req"]
    assert driver.epoch == 2


def test_a_malformed_message_or_stale_frame_is_answered_without_re_anchoring() -> None:
    script, driver, _ = _running()
    script.feed({"type": "views", "views": [{}]}, _frame(0, 1, state={"vehicle": {}}))
    driver.poll(0.0)
    assert (script.stats.bad_messages, script.stats.bad_states) == (1, 1)
    assert script.types()[-2:] == ["error", "error"]
    assert (driver.state, driver.epoch) == (RUNNING, 1)
    _complete_chunk_0(script, driver)


@pytest.mark.parametrize("seed", [_seed(1, image=b"short"), (_frame(1, 0), b"")])
def test_an_unusable_seed_or_frame_0_is_answered_and_requested_again(
    seed: tuple[dict, bytes],
) -> None:
    script = _Script()
    driver = v0.ExternalSessionDriver(
        source=script,
        pipeline=FakePipeline(),
        scene_provider=SeedSceneProvider(),
        condition_renderer=PoseConditionRenderer(),
    )
    script.feed(
        {"type": "hello", "session": "s", "api": PROTOCOL}, {"type": "start", "tick": 0}
    )
    driver.poll(0.0)
    script.feed(seed)
    driver.poll(0.0)
    assert script.types()[-3:] == ["error", "cancel", "seed_req"]
    assert (driver.state, driver.epoch, driver.rollout) == (SEEDING, 2, None)


def test_a_failed_generation_raises_and_re_anchors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    script, driver, pipeline = _running()

    def fail(**kwargs: Any) -> None:
        raise RuntimeError("generation failed")

    monkeypatch.setattr(pipeline, "generate", fail)
    script.feed(*(_frame(1, g) for g in range(1, 5)))
    with pytest.raises(RuntimeError, match="generation failed"):
        driver.poll(0.0)
    assert script.types()[-3:] == ["restart", "cancel", "seed_req"]
    assert (driver.state, driver.epoch) == (SEEDING, 2)
    monkeypatch.undo()
    script.feed(_seed(2, t0=10))
    driver.poll(0.0)
    script.feed(*(_frame(2, g, t0=10) for g in range(1, 5)))
    step = driver.poll(0.0)
    assert step is not None and (step.epoch, step.ar_index) == (2, 0)


def test_a_failed_model_cache_raises_and_re_anchors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    script, driver, pipeline = _running()
    script.feed({"type": "restart", "tick": 10})
    driver.poll(0.0)

    def fail(**kwargs: Any) -> None:
        raise RuntimeError("cache failed")

    monkeypatch.setattr(pipeline, "initialize_cache", fail)
    script.feed(_seed(2, t0=10))
    with pytest.raises(RuntimeError, match="cache failed"):
        driver.poll(0.0)
    assert script.types()[-3:] == ["restart", "cancel", "seed_req"]
    assert (driver.state, driver.epoch, driver.rollout) == (SEEDING, 3, None)


def test_a_seed_with_another_camera_of_the_same_map_reloads_the_scene() -> None:
    script = _Script()
    scenes = SeedSceneProvider()
    renderer = PoseConditionRenderer()
    driver = v0.ExternalSessionDriver(
        source=script,
        pipeline=FakePipeline(),
        scene_provider=scenes,
        condition_renderer=renderer,
    )
    script.feed(
        {"type": "hello", "session": "s", "api": PROTOCOL}, {"type": "start", "tick": 0}
    )
    driver.poll(0.0)
    camera = scenes.template.selected_camera
    rear = "camera:rear:wide:120fov"
    for epoch, name in enumerate((camera.clipgt_name, camera.clipgt_name, rear), 1):
        t0 = 10 * (epoch - 1)
        scenes.template = replace(
            scenes.template, selected_camera=replace(camera, clipgt_name=name)
        )
        if epoch > 1:
            script.feed({"type": "restart", "tick": t0})
            driver.poll(0.0)
        script.feed(_seed(epoch, t0=t0))
        driver.poll(0.0)
        assert (driver.state, driver.epoch) == (RUNNING, epoch)
    # The same map and camera keep the loaded scene; another camera reloads it.
    assert renderer.scenes == ["external-test", "external-test"]


def test_a_lost_engine_releases_the_rollout() -> None:
    script, driver, _ = _running()
    rollout = driver.rollout
    assert rollout is not None
    script.feed({"type": "_lost", "reason": "eof"})
    driver.poll(0.0)
    assert (driver.state, driver.rollout) == (DISCONNECTED, None)
    assert not rollout.is_running
    assert driver.stats.lost_reasons == ["eof"]


def test_views_take_effect_before_the_next_chunk() -> None:
    script, driver, _ = _running()
    # The views arrive with the frames that complete chunk 0, while the
    # driver waits for that chunk.
    script.feed()
    script.feed(
        *(_frame(1, g) for g in range(1, 5)), {"type": "views", "views": ["map"]}
    )
    assert driver.poll(0.0) is None
    step = driver.poll(0.0)
    assert step is not None and step.ar_index == 0
    assert driver.stats.view_frames_sent == 5


def test_session_paths_restart_end_start_error_and_a_stale_seed() -> None:
    script, driver, _ = _running()
    _complete_chunk_0(script, driver)
    assert driver.last_tick == 8
    # A restart without a tick re-seeds at the last tick.
    script.feed({"type": "restart"}, {"type": "error", "code": "x"})
    driver.poll(0.0)
    assert (driver.state, driver.epoch) == (SEEDING, 2)
    assert script.sent[-1] == v0.messages.SeedReq(epoch=2, tick=8, reason="restart")
    # A seed of an older epoch is ignored.
    script.feed(_seed(1))
    driver.poll(0.0)
    assert (driver.state, driver.stats.seeds) == (SEEDING, 1)
    script.feed({"type": "end"})
    driver.poll(0.0)
    assert (driver.state, driver.rollout) == (IDLE, None)
    assert not driver.request_refresh()
    # A later start on the same connection opens a new epoch.
    script.feed({"type": "start", "tick": 40})
    driver.poll(0.0)
    assert (driver.state, driver.epoch) == (SEEDING, 3)


def test_forwarded_input_is_not_counted_for_replay() -> None:
    script = _Script()
    script.send(Input(action="horn", phase="down"))
    assert script.sends == 0 and script.types() == ["input"]
