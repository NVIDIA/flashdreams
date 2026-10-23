# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for the Isaac Teleop V2 client-window plugin."""

from __future__ import annotations

import argparse
import threading
import time
from collections import deque
from importlib.metadata import entry_points
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from flashdreams_controller_input import (
    isaac_teleop_v2_window as window_module,
)
from flashdreams_controller_input.controller_source_v2_window import (
    ControllerSourceWindow,
)
from flashdreams_controller_input.controller_types import Observation
from flashdreams_controller_input.events import ControllerFrameEvent
from flashdreams_controller_input.isaac_teleop_source import IsaacTeleopSource
from flashdreams_controller_input.isaac_teleop_v2_window import (
    ISAAC_TELEOP_WEBRTC_MODE,
    IsaacTeleopWebRTCWindow,
)
from flashdreams_controller_input.pose import (
    WRIST_PROXY_POSE_CONVENTION,
)
from numpy import uint64

from flashdreams.runtime_v2.session_desc import SessionDesc
from flashdreams.runtime_v2.user_input_event import (
    ResetUserInputEvent,
    XRControllerUserInputEvent,
)
from flashdreams.runtime_v2.user_input_events import UserInputEvents
from flashdreams.runtime_v2.webrtc_client_window import WebRTCClientWindow

pytestmark = pytest.mark.ci_cpu


def _wait(predicate, timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.005)
    raise AssertionError("condition was not met before the test timeout")


def _frame(capture_ns: int, frame_id: int) -> dict[str, Any]:
    return {
        "client_capture_ns": capture_ns,
        # This producer timestamp intentionally uses a different clock.
        "source_client_capture_ns": 77,
        "source_frame_index": frame_id,
        "left": {
            "position": [-0.2, 1.0, -0.5],
            "orientation": [0.0, 0.0, 0.0, 1.0],
            "trigger": 0.25,
            "grip": 0.75,
            "axes": [0.1, -0.2],
            "tracked": True,
            "primary_button": True,
        },
        "right": {
            "position": [0.2, 1.0, -0.5],
            "orientation": [0.0, 0.0, 0.0, 1.0],
            "trigger": 0.9,
            "grip": 0.1,
            "axes": [-0.3, 0.4],
            "tracked": True,
            "secondary_button": True,
        },
    }


class _FakeWebRTCWindow:
    def __init__(self) -> None:
        self.server = SimpleNamespace(url="http://127.0.0.1:54321")
        self.opened: list[SessionDesc] = []
        self.close_count = 0
        self.input_events = UserInputEvents([])

    def open(self, session_desc: SessionDesc) -> None:
        self.opened.append(session_desc)

    def get_user_input_events(self) -> UserInputEvents:
        events = self.input_events
        self.input_events = UserInputEvents([])
        return events

    def request_hide_cursor(self, value: bool) -> None:
        del value

    def request_lock_cursor_to_window(self, value: bool) -> None:
        del value

    def write(self, result: object) -> None:
        del result

    def metrics_snapshot(self) -> dict[str, int]:
        return {"sent": 0}

    def close(self) -> None:
        self.close_count += 1


class _FakeIsaacSource:
    name = "isaac-teleop"

    def __init__(self) -> None:
        self.started = threading.Event()
        self._observations: deque[Observation] = deque()
        self._lock = threading.Lock()
        self.poll_count = 0
        self.stop_count = 0

    def start(self) -> None:
        self.started.set()

    def poll(self) -> Observation | None:
        with self._lock:
            if not self._observations:
                return None
            self.poll_count += 1
            return self._observations.popleft()

    def push(self, observation: Observation) -> None:
        with self._lock:
            self._observations.append(observation)

    def stop(self) -> None:
        self.stop_count += 1


def _window(
    source: _FakeIsaacSource,
    browser: _FakeWebRTCWindow,
) -> IsaacTeleopWebRTCWindow:
    return IsaacTeleopWebRTCWindow(
        host="127.0.0.1",
        port=0,
        python_executable="teleop-python",
        bridge_port=0,
        source_rate_hz=120.0,
        tracking_rate_hz=72.0,
        wrist_offset_m=0.05,
        startup_timeout_s=30.0,
        install_dir="~/.cloudxr",
        env_config=None,
        device_profile="Quest3",
        accept_eula=True,
        verbose=False,
        _webrtc_window=cast(WebRTCClientWindow, browser),
        _isaac_source=cast(IsaacTeleopSource, source),
    )


def test_iwer_frame_keeps_wrist_convention_and_client_capture_clock() -> None:
    source = _FakeIsaacSource()
    browser = _FakeWebRTCWindow()
    window = _window(source, browser)
    assert source.started.wait(timeout=1.0)
    window.open(SessionDesc())

    capture_ns = time.perf_counter_ns() + 1_000_000
    source.push(
        Observation(
            frame=_frame(capture_ns, 19),
            capture_ns=capture_ns,
            source="isaac-teleop",
        )
    )
    _wait(lambda: source.poll_count == 1)

    events = window.get_user_input_events().get_events()
    controllers = [event for event in events if isinstance(event, ControllerFrameEvent)]
    assert len(controllers) == 1
    assert controllers[0].left is not None and controllers[0].right is not None
    assert {event.frame_id for event in controllers} == {19}
    assert {event.client_capture_ns for event in controllers} == {capture_ns}
    assert {event.pose_convention for event in controllers} == {
        WRIST_PROXY_POSE_CONVENTION
    }

    window.close()
    assert source.stop_count == 1
    assert browser.close_count == 1


def test_browser_xr_events_are_filtered_from_iwer_input() -> None:
    source = _FakeIsaacSource()
    browser = _FakeWebRTCWindow()
    marker = ResetUserInputEvent(timestamp=uint64(0))
    browser.input_events = UserInputEvents(
        [
            XRControllerUserInputEvent(
                timestamp=uint64(0),
                handedness="left",
                controller_id="browser-left",
                position=(9.0, 9.0, 9.0),
                orientation=(0.0, 0.0, 0.0, 1.0),
            ),
            marker,
        ]
    )
    window = _window(source, browser)
    assert source.started.wait(timeout=1.0)
    window.open(SessionDesc())

    capture_ns = time.perf_counter_ns() + 1_000_000
    source.push(
        Observation(
            frame=_frame(capture_ns, 19),
            capture_ns=capture_ns,
            source="isaac-teleop",
        )
    )
    _wait(lambda: source.poll_count == 1)

    events = window.get_user_input_events().get_events()
    assert marker in events
    assert not any(isinstance(event, XRControllerUserInputEvent) for event in events)
    controllers = [event for event in events if isinstance(event, ControllerFrameEvent)]
    assert len(controllers) == 1
    assert controllers[0].frame_id == 19
    assert controllers[0].left is not None and controllers[0].right is not None
    assert controllers[0].left.position == (-0.2, 1.0, -0.5)

    window.close()


def test_close_before_worker_runs_does_not_restart_stopped_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worker_entered = threading.Event()
    release_worker = threading.Event()
    original_run_source = ControllerSourceWindow._run_source

    def delayed_run_source(window: ControllerSourceWindow) -> None:
        worker_entered.set()
        assert release_worker.wait(timeout=2.0)
        original_run_source(window)

    monkeypatch.setattr(
        ControllerSourceWindow,
        "_run_source",
        delayed_run_source,
    )
    source = _FakeIsaacSource()
    browser = _FakeWebRTCWindow()
    window = _window(source, browser)
    assert worker_entered.wait(timeout=1.0)

    close_failures: list[BaseException] = []

    def close_window() -> None:
        try:
            window.close()
        except BaseException as error:  # noqa: BLE001 - asserted below
            close_failures.append(error)

    close_thread = threading.Thread(target=close_window, name="close-iwer-window")
    close_thread.start()
    try:
        _wait(lambda: source.stop_count == 1)
    finally:
        release_worker.set()
    close_thread.join(timeout=2.0)

    assert not close_thread.is_alive()
    assert close_failures == []
    assert not source.started.is_set()
    assert source.stop_count == 1
    assert browser.close_count == 1


def test_close_during_start_releases_resources_created_after_initial_stop() -> None:
    class LateStartingSource(_FakeIsaacSource):
        def __init__(self) -> None:
            super().__init__()
            self.start_entered = threading.Event()
            self.release_start = threading.Event()
            self.active = False

        def start(self) -> None:
            self.start_entered.set()
            assert self.release_start.wait(timeout=2.0)
            self.active = True
            self.started.set()

        def stop(self) -> None:
            self.stop_count += 1
            self.active = False

    source = LateStartingSource()
    browser = _FakeWebRTCWindow()
    window = _window(source, browser)
    assert source.start_entered.wait(timeout=1.0)

    close_failures: list[BaseException] = []

    def close_window() -> None:
        try:
            window.close()
        except BaseException as error:  # noqa: BLE001 - asserted below
            close_failures.append(error)

    close_thread = threading.Thread(target=close_window, name="close-starting-iwer")
    close_thread.start()
    try:
        _wait(lambda: source.stop_count == 1)
        assert close_thread.is_alive()
    finally:
        source.release_start.set()
    close_thread.join(timeout=2.0)

    assert not close_thread.is_alive()
    assert close_failures == []
    assert source.stop_count == 2
    assert not source.active
    assert browser.close_count == 1


@pytest.mark.parametrize("oob_hub", [False, True])
def test_window_constructs_isolated_isaac_source(
    monkeypatch: pytest.MonkeyPatch,
    oob_hub: bool,
) -> None:
    source = _FakeIsaacSource()
    received: dict[str, object] = {}

    def build_source(**kwargs: object) -> _FakeIsaacSource:
        received.update(kwargs)
        return source

    monkeypatch.setattr(window_module, "IsaacTeleopSource", build_source)
    window = IsaacTeleopWebRTCWindow(
        host="0.0.0.0",
        port=8081,
        python_executable="teleop-python",
        bridge_port=4321,
        source_rate_hz=90.0,
        tracking_rate_hz=80.0,
        wrist_offset_m=0.04,
        startup_timeout_s=12.0,
        install_dir="cloudxr-state",
        env_config="cloudxr.env",
        device_profile="Quest3",
        accept_eula=True,
        verbose=True,
        oob_hub=oob_hub,
        _webrtc_window=cast(WebRTCClientWindow, _FakeWebRTCWindow()),
    )
    assert source.started.wait(timeout=1.0)
    try:
        assert received == {
            "python_executable": "teleop-python",
            "port": 4321,
            "rate_hz": 90.0,
            "tracking_rate_hz": 80.0,
            "wrist_offset_m": 0.04,
            "startup_timeout_s": 12.0,
            "wait_for_controllers": False,
            "install_dir": "cloudxr-state",
            "env_config": "cloudxr.env",
            "device_profile": "Quest3",
            "accept_eula": True,
            "oob_hub": oob_hub,
            "launch_runtime": True,
            "verbose": True,
        }
    finally:
        window.close()


def _parsed(*arguments: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0)
    ISAAC_TELEOP_WEBRTC_MODE.add_arguments(parser)
    return parser.parse_args(arguments)


def test_mode_is_registered_as_a_v2_client_window_plugin() -> None:
    installed = {
        entry.name: entry
        for entry in entry_points(group="flashdreams.client_windows_v2")
    }

    assert installed["isaac-teleop-webrtc"].load() is ISAAC_TELEOP_WEBRTC_MODE


def test_mode_defaults_are_virtual_web_client_without_quest_oob() -> None:
    parsed = _parsed()
    ISAAC_TELEOP_WEBRTC_MODE.check_arguments(parsed)

    assert parsed.isaac_teleop_python == str(
        Path.home() / ".local/share/flashdreams/isaac-teleop/bin/python"
    )
    assert parsed.isaac_teleop_startup_timeout_s is None
    assert parsed.isaac_teleop_source_rate_hz == pytest.approx(120.0)
    assert parsed.isaac_teleop_wrist_offset_m == pytest.approx(0.05)
    assert parsed.isaac_teleop_oob_hub is False


@pytest.mark.parametrize("oob_hub", [False, True])
def test_mode_passes_the_opt_in_hub_to_the_window(
    monkeypatch: pytest.MonkeyPatch, oob_hub: bool
) -> None:
    received: dict[str, object] = {}
    sentinel = object()

    def build_window(**kwargs: object) -> object:
        received.update(kwargs)
        return sentinel

    monkeypatch.setattr(window_module, "IsaacTeleopWebRTCWindow", build_window)
    parsed = _parsed(*(["--isaac-teleop-oob-hub"] if oob_hub else []))

    assert ISAAC_TELEOP_WEBRTC_MODE.create(parsed) is sentinel
    assert received["oob_hub"] is oob_hub


def test_mode_preserves_explicit_sidecar_interpreter_override() -> None:
    parsed = _parsed("--isaac-teleop-python", ".venv-isaac-teleop/bin/python")
    ISAAC_TELEOP_WEBRTC_MODE.check_arguments(parsed)

    assert parsed.isaac_teleop_python == ".venv-isaac-teleop/bin/python"


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (("--isaac-teleop-port", "65536"), "isaac-teleop-port"),
        (("--isaac-teleop-source-rate-hz", "0"), "source-rate-hz"),
        (("--isaac-teleop-tracking-hz", "nan"), "tracking-hz"),
        (("--isaac-teleop-wrist-offset-m", "-0.1"), "wrist-offset"),
        (("--isaac-teleop-startup-timeout-s", "0"), "startup-timeout-s"),
    ],
)
def test_mode_rejects_invalid_source_configuration(
    arguments: tuple[str, ...], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        ISAAC_TELEOP_WEBRTC_MODE.check_arguments(_parsed(*arguments))


@pytest.mark.parametrize(
    "argument",
    [
        "--isaac-teleop-setup-oob",
        "--isaac-teleop-require-physical-wake",
        "--isaac-teleop-usb-local",
        "--isaac-teleop-host-client",
        "--isaac-teleop-headless-web-client",
        "--isaac-teleop-no-launch-runtime",
    ],
)
def test_mode_rejects_non_mvp_transports(argument: str) -> None:
    with pytest.raises(SystemExit) as error:
        _parsed(argument)
    assert error.value.code == 2
