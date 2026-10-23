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

"""Controller frame-store and managed headset lifecycle regressions."""

from __future__ import annotations

import json
import os
import subprocess
import threading
import time
from types import SimpleNamespace

import pytest
from flashdreams_controller_input import frames as quest_controller
from flashdreams_controller_input.quest_source import (
    FrameStore,
    ManagedHeadset,
)

pytestmark = pytest.mark.ci_cpu
_TIMEOUT_S = 10.0


def synthetic_frame(index: int, *, capture_ns: int) -> dict:
    """Build a controller sample without importing the V1 synthetic source."""
    return {**_controllers(), "client_capture_ns": capture_ns}


def _controllers() -> dict:
    """A minimal well-formed pair of controller payloads."""

    def side(x: float) -> dict:
        return {
            "position": [x, 1.0, -0.5],
            "orientation": [0.0, 0.0, 0.0, 1.0],
            "trigger": 0.0,
            "grip": 0.0,
            "tracked": True,
        }

    return {"left": side(-0.22), "right": side(0.22)}


def _wait(predicate, timeout: float = _TIMEOUT_S) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


def test_a_dropped_controller_is_held_as_untracked() -> None:
    store = FrameStore()
    store._store(synthetic_frame(0, capture_ns=1_000_000))
    dropped = synthetic_frame(1, capture_ns=2_000_000)
    dropped["left"] = None
    store._store(dropped)

    latest = store.snapshot()
    assert latest is not None
    assert latest["left"] is not None, "the last known pose should be retained"
    assert latest["left"]["tracked"] is False
    assert set(quest_controller.controller_sides(latest)) == {"left", "right"}
    assert not quest_controller.both_controllers_tracked(latest)


def test_unparseable_json_is_skipped() -> None:
    store = FrameStore()
    store.consume("{not json")
    assert store.rejected == 1


def test_an_unknown_message_is_ignored_without_counting_as_bad() -> None:
    store = FrameStore()
    store.consume(json.dumps({"type": "pong"}))
    assert store.rejected == 0
    assert store.received == 0


def test_rejection_logging_is_throttled(capsys) -> None:
    store = FrameStore()
    for _ in range(50):
        store.consume("{not json")
    assert store.rejected == 50
    # A page emitting bad frames does so at full rate; one report, not fifty.
    assert capsys.readouterr().out.count("malformed frame") == 1


def test_staleness_is_reported() -> None:
    store = FrameStore()
    assert store.age_ms() is None
    assert not store.ready()
    store._store(synthetic_frame(0, capture_ns=time.perf_counter_ns()))
    assert store.ready(stale_ms=1000.0)
    store._store(synthetic_frame(1, capture_ns=1))
    assert not store.ready(stale_ms=100.0)


def test_partial_readiness_still_requires_a_compatible_fresh_frame() -> None:
    store = FrameStore(expected_protocol=3)
    assert not store.ready(require_tracked=False)
    frame = synthetic_frame(0, capture_ns=time.perf_counter_ns())
    frame["left"] = None
    store._store(frame)
    assert not store.ready(require_tracked=False)
    store.consume(json.dumps({"type": "hello", "protocol": 3}))
    assert store.ready(require_tracked=False)
    assert not store.ready()
    store._store({**frame, "client_capture_ns": 1})
    assert not store.ready(require_tracked=False)


def test_empty_axes_count_as_absent_rather_than_malformed() -> None:
    # WebXR reports `gamepad.axes: []` for a controller it has not finished
    # enumerating, which used to be rejected and killed the connection.
    controllers = _controllers()
    controllers["left"]["axes"] = []
    controllers["right"]["axes"] = []
    frame = quest_controller.parse_xr_frame(
        {"type": "xr_frame", "controllers": controllers}
    )
    assert frame is not None
    assert "axes" not in frame["left"]
    assert "axes" not in frame["right"]
    assert quest_controller.both_controllers_tracked(frame) is True


def test_named_control_defaults_preserve_missing_axes() -> None:
    raw = synthetic_frame(0, capture_ns=123)
    for side in quest_controller.SIDES:
        for key in ("trigger", "grip", "tracked"):
            del raw[side][key]
    frame = quest_controller.parse_xr_frame({"type": "xr_frame", "controllers": raw})
    assert frame is not None
    for side in quest_controller.SIDES:
        assert frame[side]["trigger"] == frame[side]["grip"] == 0.0
        assert frame[side]["tracked"] is True
        assert "axes" not in frame[side]
        for key in ("primary_button", "secondary_button"):
            assert frame[side][key] is False


@pytest.mark.parametrize(
    ("metadata", "expected"),
    [({}, False), ({"is_emulated": False}, False), ({"is_emulated": True}, True)],
)
def test_client_identity_is_explicit_and_preserves_capture_clocks(
    metadata: dict, expected: bool
) -> None:
    frame = quest_controller.parse_xr_frame(
        {
            "type": "xr_frame",
            "controllers": _controllers(),
            "t_client": 2.5,
            "client_connection_id": "browser:stream-1",
            **metadata,
        },
        received_ns=123,
    )
    assert frame is not None
    assert frame["is_emulated"] is expected
    assert frame["client_connection_id"] == "browser:stream-1"
    assert frame["client_capture_ns"] == 123
    assert frame[quest_controller.SOURCE_CLIENT_CAPTURE_NS_KEY] == 2_500_000_000


@pytest.mark.parametrize(
    ("field", "value"),
    [("is_emulated", value) for value in (None, 0, 1, "true", "false")]
    + [("client_connection_id", value) for value in (False, 1, "", "a" * 513)],
)
def test_client_identity_rejects_ambiguous_metadata(field: str, value: object) -> None:
    with pytest.raises(ValueError, match=field):
        quest_controller.parse_xr_frame(
            {"type": "xr_frame", "controllers": _controllers(), field: value}
        )


@pytest.mark.parametrize("physical_fields", [True])
def test_physical_buttons_preserve_both_producer_schemas(physical_fields: bool) -> None:
    controllers = _controllers()
    for side, controller in controllers.items():
        if physical_fields:
            controller.update(primary_button=True, secondary_button=True)
        elif side == "left":
            controller.update(next_scene=True, previous_scene=True)
        else:
            controller.update(toggle_live=True, recenter=True)
        controller["gripper_open"] = 0.25
    frame = quest_controller.parse_xr_frame(
        {"type": "xr_frame", "t_client": 123.5, "controllers": controllers},
        received_ns=987_654_321,
    )
    assert frame is not None
    assert frame["client_capture_ns"] == 987_654_321
    assert frame["source_client_capture_ns"] == 123_500_000_000
    for side in ("left", "right"):
        assert frame[side]["primary_button"] is True
        assert frame[side]["secondary_button"] is True
        assert frame[side]["tracked"] is True
        assert frame[side]["position"] == pytest.approx(controllers[side]["position"])
        assert frame[side]["orientation"] == controllers[side]["orientation"]
        assert not {
            "next_scene",
            "previous_scene",
            "recenter",
            "toggle_live",
            "gripper_open",
        }.intersection(frame[side])


def test_physical_buttons_override_legacy_names() -> None:
    controllers = _controllers()
    controllers["left"].update(
        primary_button=False,
        secondary_button=False,
        next_scene=True,
        previous_scene=True,
    )
    controllers["right"].update(
        primary_button=False, secondary_button=False, toggle_live=True, recenter=True
    )
    frame = quest_controller.parse_xr_frame(
        {"type": "xr_frame", "controllers": controllers}
    )
    assert frame is not None
    for side in ("left", "right"):
        assert frame[side]["primary_button"] is False
        assert frame[side]["secondary_button"] is False


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("position", [0.0, 1.0]),
        ("position", [0.0, float("nan"), 0.0]),
        ("orientation", [0.0, 0.0, 0.0, 0.0]),
        ("axes", [0.0]),
        ("axes", [0.0, 0.0, 0.0]),
        ("axes", [0.0, float("inf")]),
        ("battery", 1.1),
    ],
)
def test_named_controller_validation_rejects_malformed_values(key, value) -> None:
    controllers = _controllers()
    controllers["left"][key] = value
    store = FrameStore()
    store.consume(json.dumps({"type": "xr_frame", "controllers": controllers}))
    assert store.received == 0
    assert store.rejected == 1
    assert store.snapshot() is None


def test_stopping_a_headset_that_never_started_is_safe() -> None:
    ManagedHeadset().stop()


@pytest.mark.parametrize("require_physical_wake", [False, True])
def test_headset_stop_during_popen_reaps_the_late_child(
    monkeypatch, require_physical_wake
) -> None:
    spawning = threading.Event()
    finish_spawn = threading.Event()
    pipe_fds = []
    original_pipe = os.pipe

    def pipe():
        result = original_pipe()
        pipe_fds.extend(result)
        return result

    monkeypatch.setattr(os, "pipe", pipe)

    class Process:
        terminated = 0
        waited = 0

        def poll(self):
            return None if not self.terminated else 0

        def terminate(self):
            self.terminated += 1

        def wait(self, *, timeout):
            self.waited += 1
            return 0

    process = Process()

    def popen(*_args, **_kwargs):
        spawning.set()
        assert finish_spawn.wait(5.0)
        return process

    monkeypatch.setattr("subprocess.Popen", popen)
    headset = ManagedHeadset(require_physical_wake=require_physical_wake)
    errors: list[BaseException] = []

    def start():
        try:
            headset.start()
        except BaseException as error:
            errors.append(error)

    worker = threading.Thread(target=start, daemon=True)
    worker.start()
    try:
        assert spawning.wait(5.0)
        headset.stop()
    finally:
        finish_spawn.set()
        worker.join(timeout=5.0)

    assert not worker.is_alive()
    assert process.terminated == process.waited == 1
    assert headset._process is None
    assert not headset.wake_ready
    for fd in pipe_fds:
        with pytest.raises(OSError):
            os.fstat(fd)
    assert len(errors) == 1 and "stopped while starting" in str(errors[0])
    headset.stop()
    assert process.terminated == 1
    with pytest.raises(RuntimeError, match="stopped before starting"):
        headset.start()


@pytest.mark.parametrize("marker", ["1", "W1", "W", "Wx", "x", ""])
def test_headset_wake_readiness_requires_the_launcher_marker(
    monkeypatch, tmp_path, marker
) -> None:
    script = tmp_path / "launcher.sh"
    script.write_text(
        'while [[ ! -f "$QUEST_TEST_GATE" ]]; do sleep 0.01; done\n'
        'printf "%s" "$QUEST_TEST_MARKER" >&"$QUEST_WAKE_READY_FD"\n'
    )
    gate = tmp_path / "gate"
    monkeypatch.setenv("QUEST_TEST_GATE", str(gate))
    monkeypatch.setenv("QUEST_TEST_MARKER", marker)
    monkeypatch.setattr(
        "flashdreams_controller_input.quest_source._launcher_script", lambda: script
    )
    original_popen = subprocess.Popen
    write_fds = []

    def popen(*args, **kwargs):
        write_fd = int(kwargs["env"]["QUEST_WAKE_READY_FD"])
        assert kwargs["pass_fds"] == (write_fd,)
        write_fds.append(write_fd)
        return original_popen(*args, **kwargs)

    monkeypatch.setattr(subprocess, "Popen", popen)
    headset = ManagedHeadset(require_physical_wake=True)
    assert not headset.wake_ready
    try:
        headset.start()
        read_fd = headset._wake_ready_fd
        assert read_fd is not None
        assert not os.get_blocking(read_fd)
        assert not headset.wake_ready
        with pytest.raises(OSError):
            os.fstat(write_fds[0])
        gate.touch()
        process = headset._process
        assert process is not None
        assert _wait(lambda: process.poll() is not None)
        assert headset.wake_ready is (marker in {"1", "W1"})
        assert headset.wake_ready is (marker in {"1", "W1"})
        assert not headset.waiting_for_wake
        with pytest.raises(OSError):
            os.fstat(read_fd)
    finally:
        headset.stop()
    assert not headset.wake_ready


@pytest.mark.parametrize("complete", [True, False])
def test_headset_wait_status_begins_after_sleep_and_clears_on_ready_or_stop(
    monkeypatch, complete
) -> None:
    write_fds = []

    def popen(*args, **kwargs):
        write_fds.append(os.dup(int(kwargs["env"]["QUEST_WAKE_READY_FD"])))
        return SimpleNamespace(poll=lambda: 0)

    monkeypatch.setattr(subprocess, "Popen", popen)
    headset = ManagedHeadset(require_physical_wake=True)
    try:
        headset.start()
        assert not headset.waiting_for_wake
        assert not headset.wake_ready
        os.write(write_fds[0], b"W")
        assert headset.waiting_for_wake
        assert not headset.wake_ready
        assert headset.waiting_for_wake
        if complete:
            os.write(write_fds[0], b"1")
            assert not headset.waiting_for_wake
            assert headset.wake_ready
    finally:
        headset.stop()
        for fd in write_fds:
            os.close(fd)
    assert not headset.waiting_for_wake
    assert not headset.wake_ready


def test_headset_stop_closes_an_unread_wake_pipe(monkeypatch) -> None:
    monkeypatch.setattr(
        subprocess, "Popen", lambda *args, **kwargs: SimpleNamespace(poll=lambda: 0)
    )
    headset = ManagedHeadset(require_physical_wake=True)
    headset.start()
    read_fd = headset._wake_ready_fd
    assert read_fd is not None
    headset.stop()
    headset.stop()
    assert not headset.wake_ready
    with pytest.raises(OSError):
        os.fstat(read_fd)


@pytest.mark.parametrize("failure", ["set_blocking", "Popen"])
def test_failed_headset_start_closes_both_wake_pipe_ends(monkeypatch, failure) -> None:
    pipe_fds = []
    original_pipe = os.pipe

    def pipe():
        result = original_pipe()
        pipe_fds.extend(result)
        return result

    def fail(*args, **kwargs):
        raise OSError("test startup failure")

    monkeypatch.setattr(os, "pipe", pipe)
    monkeypatch.setattr(os if failure == "set_blocking" else subprocess, failure, fail)
    headset = ManagedHeadset(require_physical_wake=True)
    with pytest.raises(OSError, match="test startup failure"):
        headset.start()
    assert not headset.wake_ready
    assert headset._process is None
    assert len(pipe_fds) == 2
    for fd in pipe_fds:
        with pytest.raises(OSError):
            os.fstat(fd)
    headset.stop()


def test_ungated_headset_ignores_an_inherited_wake_pipe(monkeypatch) -> None:
    monkeypatch.setenv("QUEST_WAKE_READY_FD", "999")

    def popen(*args, **kwargs):
        assert kwargs["env"]["QUEST_WAKE_READY_FD"] != "999"
        assert kwargs["pass_fds"]
        return SimpleNamespace(poll=lambda: 0)

    monkeypatch.setattr(subprocess, "Popen", popen)
    headset = ManagedHeadset()
    assert not headset.wake_ready
    headset.start()
    assert not headset.wake_ready
    headset.stop()
    assert not headset.wake_ready
