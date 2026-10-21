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

"""CPU ownership, clock-boundary, and lifecycle contracts for unified input."""

from __future__ import annotations

import threading
from collections import deque
from typing import Any, cast

import pytest
from flashdreams_controller_input.controller_types import InputSource, Observation
from flashdreams_controller_input.unified_source import UnifiedControllerSource

pytestmark = pytest.mark.ci_cpu


class _Clock:
    def __init__(self):
        self.now = 10_000_000_000

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += int(seconds * 1e9)


class _Source:
    status_message: str

    def __init__(self, name, events):
        self.name = name
        self.events = events
        self.pending = deque()
        self.active = False
        self.poll_calls = 0
        self.start_error = None
        self.poll_error = None
        self.stop_error = None
        self.start_entered = None
        self.start_release = None
        self.continuous = None
        self.ready_for_clients = True

    def start(self, **options):
        self.events.append((self.name, "start", options))
        if self.start_entered is not None:
            self.start_entered.set()
            assert self.start_release is not None
            assert self.start_release.wait(3)
        self.active = True
        if self.start_error:
            raise self.start_error

    def stop(self):
        self.events.append((self.name, "stop"))
        self.active = False
        if self.stop_error:
            raise self.stop_error

    def poll(self):
        self.poll_calls += 1
        if self.poll_error:
            raise self.poll_error
        if self.pending:
            return self.pending.popleft()
        return self.continuous


def _setup():
    clock, events = _Clock(), []
    remote = _Source("remote-quest", events)
    local = _Source("isaac-teleop", events)
    source = UnifiedControllerSource(remote, cast(Any, local), _clock_ns=clock)
    return source, remote, local, clock, events


def _sample(
    clock, producer, *, tracked=(True, True), generation=0, index=0, capture=90
):
    frame: dict[str, Any] = {
        side: {
            "position": [0.1, 1.2, -0.3],
            "orientation": [0.0, 0.0, 0.0, 1.0],
            "trigger": 0.75,
            "grip": 0.25,
            "axes": [0.2, -0.4],
            "tracked": present,
        }
        for side, present in zip(("left", "right"), tracked, strict=True)
    }
    frame.update(
        client_capture_ns=clock.now,
        source_client_capture_ns=capture,
        source_frame_index=index,
    )
    return Observation(frame, clock.now, producer.name, generation)


def test_start_is_nonblocking_for_browser_and_stop_is_idempotent():
    source, remote, local, _, events = _setup()
    assert isinstance(source, InputSource)
    assert source.poll() is None
    source.start()
    source.start()
    assert events == [
        ("remote-quest", "start", {}),
        ("isaac-teleop", "start", {"wait_for_frame": False}),
    ]
    source.stop()
    source.stop()
    assert not remote.active and not local.active
    assert source.poll() is None
    with pytest.raises(RuntimeError, match="stopped before starting"):
        source.start()


@pytest.mark.parametrize(
    "headset_status",
    ["Detected; press the power button", "Headset setup failed; reconnect USB"],
)
def test_headset_setup_does_not_block_shared_browser_input_or_consume_poses(
    headset_status,
):
    _, remote, local, clock, events = _setup()
    headset = _Source("headset-setup", events)
    headset.status_message = headset_status
    source = UnifiedControllerSource(
        remote,
        local,
        headset=cast(Any, headset),
        web_client_url="https://teleop.example/?serverIP=10.0.0.1",
        certificate_url="https://10.0.0.1:48322/",
        _clock_ns=clock,
    )
    local.ready_for_clients = False
    assert source.connection_status().web_client == "Starting..."
    assert source.connection_status().web_client_url == ""
    source.start()
    assert [event[:2] for event in events] == [
        (remote.name, "start"),
        (local.name, "start"),
        (headset.name, "start"),
    ]
    assert source.can_connect_headset()
    local.ready_for_clients = True
    sample = _sample(clock, local)
    local.pending.append(sample)
    before = local.poll_calls, remote.poll_calls, headset.poll_calls
    status = source.connection_status()
    assert before == (local.poll_calls, remote.poll_calls, headset.poll_calls)
    assert list(local.pending) == [sample]
    assert status.headset == headset_status
    assert status.web_client == "Ready to connect"
    assert status.web_client_url == "https://teleop.example/?serverIP=10.0.0.1"
    assert status.certificate_url == "https://10.0.0.1:48322/"
    assert status.active_input == "Waiting for controllers"
    observation = source.poll()
    assert observation is not None
    assert observation.frame is sample.frame
    assert source.connection_status().active_input == "Isaac Teleop"
    assert not source.can_connect_headset()
    assert headset.poll_calls == 0
    source.stop()
    assert not source.can_connect_headset()
    assert not headset.active and not local.active and not remote.active


@pytest.mark.parametrize("tracked", [(False, False), (True, False), (False, True)])
def test_idle_or_partial_local_input_cannot_reserve_ownership(tracked):
    source, remote, local, clock, _ = _setup()
    source.start()
    local.pending.extend(_sample(clock, local, tracked=tracked) for _ in range(20))
    assert source.poll() is None
    assert not local.pending
    expected = _sample(clock, remote)
    remote.pending.append(expected)
    result = source.poll()
    assert result.source == "remote-quest" and result.generation == 0
    assert result.frame is expected.frame


@pytest.mark.parametrize("first", ["remote", "local", "tie"])
def test_first_host_capture_wins_and_inactive_history_is_drained(first):
    source, remote, local, clock, _ = _setup()
    source.start()
    earlier = clock.now - 100_000_000
    remote_sample = _sample(clock, remote)
    local_sample = _sample(clock, local)
    if first == "remote":
        remote_sample = Observation(remote_sample.frame, earlier, remote.name)
    if first == "local":
        local_sample = Observation(local_sample.frame, earlier, local.name)
    remote.pending.append(remote_sample)
    local.pending.append(local_sample)
    winner, loser = (local, remote) if first == "local" else (remote, local)
    following = [_sample(clock, winner, index=index) for index in range(1, 4)]
    winner.pending.extend(following)
    loser.pending.extend(_sample(clock, loser) for _ in range(50))
    selected = source.poll()
    assert selected.source == winner.name
    assert not loser.pending
    assert list(winner.pending) == following
    for expected in following:
        result = source.poll()
        assert result.frame is expected.frame
        assert result.capture_ns == expected.capture_ns
        assert result.generation == selected.generation


def test_active_source_never_mixes_with_simultaneous_other_input():
    source, remote, local, clock, _ = _setup()
    source.start()
    remote.pending.append(_sample(clock, remote))
    assert source.poll().source == "remote-quest"
    for index in range(5):
        clock.advance(0.1)
        local.pending.extend(_sample(clock, local) for _ in range(10))
        expected = _sample(clock, remote, index=index)
        remote.pending.append(expected)
        result = source.poll()
        assert result.frame is expected.frame and result.source == "remote-quest"
        assert result.generation == 0
        assert not local.pending


@pytest.mark.parametrize("identity", ["quest-webxr", "isaac-teleop"])
def test_direct_connection_preserves_identity_instead_of_assuming_iwer(identity):
    source, _, local, clock, _ = _setup()
    local.name = identity
    source.start()
    local.pending.append(_sample(clock, local))
    result = source.poll()
    assert result.source == identity


def test_continuous_empty_local_runtime_cannot_reserve_or_block_ownership():
    source, remote, local, clock, _ = _setup()
    source.start()
    local.continuous = Observation(
        {"left": None, "right": None, "client_capture_ns": clock.now},
        clock.now,
        local.name,
    )
    assert source.poll() is None
    assert local.poll_calls == 256
    remote.pending.append(_sample(clock, remote))
    result = source.poll()
    assert result.source == remote.name and result.generation == 0


@pytest.mark.parametrize("missing_hand", [False, True])
def test_tracking_loss_and_one_hand_recovery_keep_the_same_origin(missing_hand):
    source, remote, local, clock, _ = _setup()
    source.start()
    remote.pending.append(_sample(clock, remote))
    assert source.poll().generation == 0
    clock.advance(0.3)
    boundary = Observation(None, clock.now, remote.name)
    remote.pending.append(boundary)
    local.pending.append(_sample(clock, local))
    result = source.poll()
    assert result.frame is None and result.generation == 0
    for tracked in ((True, False), (False, True), (True, True)):
        clock.advance(0.6)
        sample = _sample(clock, remote, tracked=tracked)
        if missing_hand:
            for side, present in zip(("left", "right"), tracked, strict=True):
                if not present:
                    sample.frame[side] = None
        remote.pending.append(sample)
        local.pending.append(_sample(clock, local))
        result = source.poll()
        assert result.frame is sample.frame
        assert result.generation == 0 and result.source == remote.name
        assert not local.pending


@pytest.mark.parametrize("loss", ["silence", "untracked"])
@pytest.mark.parametrize("next_owner", ["remote", "local"])
def test_bounded_loss_releases_owner_and_reacquisition_resets_clock_generation(
    loss, next_owner
):
    source, remote, local, clock, _ = _setup()
    source.start()
    remote.pending.append(_sample(clock, remote, index=500, capture=90_000_000_000))
    first = source.poll()
    for _ in range(4):
        clock.advance(0.2)
        if loss == "untracked":
            remote.pending.append(_sample(clock, remote, tracked=(False, False)))
        local.pending.append(_sample(clock, local))
        result = source.poll()
        assert result is None or result.generation == first.generation
        assert not local.pending
    clock.advance(0.2)
    local.pending.append(_sample(clock, local))
    boundary = source.poll()
    assert boundary.frame is None and boundary.source == remote.name
    assert boundary.generation == first.generation
    assert not local.pending
    assert source.poll() is None
    winner = remote if next_owner == "remote" else local
    expected = _sample(clock, winner, index=0, capture=10)
    winner.pending.append(expected)
    result = source.poll()
    assert result.generation == first.generation + 1
    assert result.source == winner.name
    assert result.frame is expected.frame and result.capture_ns == expected.capture_ns
    assert result.frame["source_client_capture_ns"] == 10
    assert result.frame["source_frame_index"] == 0


def test_remote_generation_change_is_forwarded_before_new_origin_frames():
    source, remote, _, clock, _ = _setup()
    source.start()
    remote.pending.append(_sample(clock, remote, generation=8, index=500))
    assert source.poll().generation == 0
    remote.pending.extend(
        [
            Observation(None, clock.now, remote.name, generation=9),
            _sample(clock, remote, generation=9, index=0, capture=1),
        ]
    )
    boundary = source.poll()
    assert boundary.frame is None and boundary.generation == 1
    result = source.poll()
    assert result.generation == 1 and result.frame["source_frame_index"] == 0


def test_paced_inactive_sample_from_before_release_cannot_acquire_later():
    source, remote, local, clock, _ = _setup()
    source.start()
    remote.pending.append(_sample(clock, remote))
    assert source.poll().source == remote.name
    clock.advance(0.9)
    local.pending.append(_sample(clock, local))
    next_local_poll_ns = clock.now + 200_000_000
    original_poll = local.poll

    def paced_poll():
        return None if clock.now < next_local_poll_ns else original_poll()

    local.poll = paced_poll
    clock.advance(0.1)
    assert source.poll().frame is None
    assert local.pending
    clock.advance(0.1)
    assert source.poll() is None
    assert not local.pending
    local.pending.append(_sample(clock, local))
    result = source.poll()
    assert result.source == local.name and result.generation == 1


def test_stale_frames_cannot_claim_or_keep_ownership():
    source, remote, local, clock, _ = _setup()
    source.start()
    remote.pending.append(_sample(clock, remote))
    clock.advance(0.6)
    assert source.poll() is None
    local.pending.append(_sample(clock, local))
    assert source.poll().source == local.name
    stale = _sample(clock, local)
    clock.advance(0.6)
    local.pending.append(stale)
    result = source.poll()
    assert result.frame is None
    clock.advance(0.4)
    assert source.poll().frame is None
    remote.pending.append(_sample(clock, remote))
    assert source.poll().generation == 1


def test_inactive_continuous_producer_has_bounded_poll_work():
    source, remote, local, clock, _ = _setup()
    source.start()
    remote.pending.append(_sample(clock, remote))
    assert source.poll().source == remote.name
    local.continuous = _sample(clock, local)
    before = local.poll_calls
    remote.pending.append(_sample(clock, remote))
    assert source.poll().source == remote.name
    assert local.poll_calls - before == 256


@pytest.mark.parametrize("failed_source", ["remote", "local"])
def test_start_failure_cleans_both_sources(failed_source):
    source, remote, local, _, events = _setup()
    failing = remote if failed_source == "remote" else local
    failing.start_error = RuntimeError("startup failed")
    with pytest.raises(RuntimeError, match="startup failed"):
        source.start()
    assert not remote.active and not local.active
    assert (remote.name, "stop") in events and (local.name, "stop") in events
    assert source.poll() is None


@pytest.mark.parametrize("failed_source", ["remote", "local"])
def test_runtime_failure_of_either_source_is_fatal_and_cleans_both(failed_source):
    source, remote, local, clock, _ = _setup()
    source.start()
    remote.pending.append(_sample(clock, remote))
    source.poll()
    failing = remote if failed_source == "remote" else local
    failing.poll_error = RuntimeError("runtime failed")
    with pytest.raises(RuntimeError, match="runtime failed"):
        source.poll()
    assert not remote.active and not local.active
    assert source.poll() is None


def test_cleanup_failure_still_stops_the_other_source():
    source, remote, local, _, _ = _setup()
    source.start()
    local.stop_error = RuntimeError("local cleanup failed")
    with pytest.raises(RuntimeError, match="local cleanup failed"):
        source.stop()
    assert not remote.active and not local.active


@pytest.mark.parametrize("blocked_source", ["remote", "local"])
def test_stop_during_startup_reaps_late_resources(blocked_source):
    source, remote, local, _, events = _setup()
    blocked = remote if blocked_source == "remote" else local
    blocked.start_entered = threading.Event()
    blocked.start_release = threading.Event()
    errors = []

    def start():
        try:
            source.start()
        except BaseException as error:
            errors.append(error)

    thread = threading.Thread(target=start)
    thread.start()
    try:
        assert blocked.start_entered.wait(3)
        source.stop()
    finally:
        blocked.start_release.set()
        thread.join(timeout=3)
    assert not thread.is_alive()
    assert len(errors) == 1 and "stopped during startup" in str(errors[0])
    assert not remote.active and not local.active
    if blocked_source == "remote":
        assert not any(event[:2] == (local.name, "start") for event in events)
    assert source.poll() is None


@pytest.mark.parametrize("timeout", [0, -1, float("inf"), float("nan")])
def test_invalid_ownership_timeout_is_rejected(timeout):
    _, remote, local, _, _ = _setup()
    with pytest.raises(ValueError, match="finite and positive"):
        UnifiedControllerSource(remote, local, ownership_timeout_s=timeout)
