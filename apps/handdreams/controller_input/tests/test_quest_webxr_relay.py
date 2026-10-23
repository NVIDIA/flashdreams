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

"""Exercise laptop CloudXR ownership independently of hardware and GPU runtimes."""

from __future__ import annotations

import argparse
import http.client
import threading
import time
from contextlib import contextmanager
from types import SimpleNamespace
from urllib.parse import parse_qs, urlsplit

import pytest
from flashdreams_controller_input import isaac_teleop_source, quest_relay, quest_webxr
from flashdreams_controller_input.controller_types import Observation

pytestmark = pytest.mark.ci_cpu


def _args(**overrides):
    args = dict(
        server="dgx.example",
        port=8445,
        ca_cert="ca.pem",
        cloudxr_port=48322,
        isaac_teleop_python="sdk-python",
        isaac_teleop_accept_eula=True,
        quest_device="quest",
        require_physical_wake=True,
        accept_server_certificate=True,
        keep_browser_open_on_error=False,
        startup_timeout=120,
    )
    return argparse.Namespace(**(args | overrides))


def _wait(predicate):
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        result = predicate()
        if result:
            return result
        time.sleep(0.005)
    pytest.fail("worker did not reach expected state")


@pytest.fixture
def rig(monkeypatch):
    events, messages = [], []
    state = SimpleNamespace(
        woke=False,
        headset=None,
        source=None,
        page_entered=False,
        block_source=False,
        fail_stage=None,
        health_failed=False,
        emit_frames=True,
        hub_status=200,
        hub_config={"serverIP": "127.0.0.1", "port": 48322},
        hub_peers=[],
        hub_failures=0,
        hub_error=ConnectionRefusedError(),
        hub_headers=None,
        health_checks=0,
        observation=None,
        usb=None,
        block_usb=False,
        wake_options=[],
    )

    class Headset:
        def __init__(self, **options):
            assert options == dict(
                device_serial="quest",
                require_physical_wake=options["require_physical_wake"],
                adb_executable="/resolved/adb",
            )
            state.wake_options.append(options["require_physical_wake"])
            assert state.headset is None or state.headset.stopped
            self.stopped = False
            state.headset = self

        @property
        def wake_ready(self):
            return state.woke

        @property
        def waiting_for_wake(self):
            return "headset-start" in events and not state.woke and not self.stopped

        def start(self):
            events.append("headset-start")
            if state.fail_stage == "headset":
                raise RuntimeError("headset failed")

        def check_alive(self):
            assert not self.stopped

        def stop(self):
            if not self.stopped:
                events.append("headset-stop")
                self.stopped = True

    class Source:
        def __init__(self, **options):
            self.options = options
            self.stopped = threading.Event()
            state.source = self

        def start(self, *, wait_for_frame):
            assert wait_for_frame is False
            events.append("source-start")
            assert state.woke
            if state.fail_stage == "source":
                raise RuntimeError("source failed")
            if state.block_source:
                assert self.stopped.wait(3)
                raise RuntimeError("cancelled source startup")
            if state.emit_frames:
                self.options["on_message"](b'{"type":"hello","protocol":5}')
                self.options["on_message"](b'{"type":"xr_frame","t_client":123.456}')

        def check_alive(self):
            state.health_checks += 1
            if state.health_failed:
                raise RuntimeError("sidecar failed")

        def poll(self):
            events.append("source-poll")
            self.check_alive()
            return state.observation

        def stop(self):
            if not self.stopped.is_set():
                events.append("source-stop")
                self.stopped.set()

    @contextmanager
    def page(serial, guard, timeout, origin, adb_executable):
        assert adb_executable == "/resolved/adb"
        assert serial == "quest"
        assert origin == "https://127.0.0.1:48322"
        assert timeout == 120
        events.append("page-open")
        if state.fail_stage == "page":
            raise RuntimeError("page failed")
        state.page_entered = True
        try:
            yield object()
        finally:
            events.append("page-close")

    def run_page(cdp, guard, args, url, **callbacks):
        query = parse_qs(urlsplit(url).query)
        assert url.startswith("https://127.0.0.1:48322/client/?")
        assert query["serverIP"] == ["127.0.0.1"]
        assert query["port"] == ["48322"]
        assert query["turnServer"] == ["turn:127.0.0.1:45678?transport=tcp"]
        assert query["turnCredential"] == [state.usb.credential]
        assert query["iceRelayOnly"] == ["1"]
        assert "dgx.example" not in url
        while True:
            if state.fail_stage == "run-page":
                raise RuntimeError("Isaac Teleop reported a connection error")
            guard.check_alive()
            time.sleep(0.005)

    class Connection:
        def __init__(self, host, port, *, timeout, context):
            assert (host, port, timeout) == ("127.0.0.1", 48322, 1)
            assert not context.check_hostname

        def request(self, method, path, *, headers=None):
            assert (method, path) == ("GET", "/api/oob/v1/state")
            state.hub_headers = headers or {}
            events.append("hub-request")
            if state.hub_failures:
                state.hub_failures -= 1
                raise state.hub_error

        def getresponse(self):
            import json

            return SimpleNamespace(
                status=state.hub_status,
                read=lambda size: json.dumps(
                    {"config": state.hub_config, "headsets": state.hub_peers}
                ).encode(),
            )

        def close(self):
            events.append("hub-close")

    def find_adb():
        events.append("find-adb")
        return "/resolved/adb"

    def device(serial, adb_executable):
        assert adb_executable == "/resolved/adb"
        return "quest"

    def adb(executable, serial, *parts):
        assert (executable, serial, parts) == (
            "/resolved/adb",
            "quest",
            ("get-devpath",),
        )
        return "usb:1-1"

    class USB:
        turn_port = 45678
        credential = "per-run-secret"

        def __init__(self, selected_adb, port):
            assert port == 48322
            assert selected_adb("get-devpath") == "usb:1-1"
            self.stopped = False
            state.usb = self

        def start(self, check_alive):
            assert state.woke
            events.append("usb-start")
            if state.fail_stage == "usb":
                raise RuntimeError("USB transport failed")
            while state.block_usb:
                check_alive()
                time.sleep(0.005)

        def check_alive(self):
            assert not self.stopped
            if state.fail_stage == "turn-exit":
                raise RuntimeError("TURN process exited")

        def stop(self):
            if not self.stopped:
                events.append("usb-stop")
                self.stopped = True

    monkeypatch.delenv("QUEST_META_PIN", raising=False)
    monkeypatch.setattr(quest_webxr, "cached_web_client_dir", lambda: "/cached")
    monkeypatch.setattr(quest_webxr.shutil, "which", lambda _: "/bin/turnserver")
    monkeypatch.setattr(quest_webxr, "_adb", adb)
    monkeypatch.setattr(quest_webxr, "USBLocalTransport", USB)
    monkeypatch.setattr(quest_webxr, "ManagedHeadset", Headset)
    monkeypatch.setattr(quest_webxr, "_find_adb", find_adb)
    monkeypatch.setattr(quest_webxr, "_device", device)
    monkeypatch.setattr(quest_webxr, "_page", page)
    monkeypatch.setattr(quest_webxr, "_run_page", run_page)
    monkeypatch.setattr(isaac_teleop_source, "IsaacTeleopSource", Source)
    monkeypatch.setattr(quest_webxr.http.client, "HTTPSConnection", Connection)
    producer = quest_webxr.QuestWebXRProducer(_args())
    state.producers = [producer]
    yield producer, state, events, messages
    for owned_producer in state.producers:
        owned_producer.stop()


def _injected_producer(rig):
    _, state, events, messages = rig
    source = isaac_teleop_source.IsaacTeleopSource(on_message=messages.append)
    producer = quest_webxr.QuestWebXRProducer(_args(), source=source)
    state.producers.append(producer)
    return producer


def _shared_producer(
    rig, *, quest_device="quest", can_connect=lambda: True, require_physical_wake=True
):
    _, state, _, messages = rig
    source = isaac_teleop_source.IsaacTeleopSource(on_message=messages.append)
    state.hub_config = {"serverIP": "192.0.2.10", "port": 48322}
    producer = quest_webxr.QuestWebXRProducer(
        _args(quest_device=quest_device, require_physical_wake=require_physical_wake),
        source=source,
        shared_runtime=True,
        hub_host="192.0.2.10",
        can_connect=can_connect,
    )
    state.producers.append(producer)
    return producer


@pytest.mark.parametrize("missing", ["adb", "headset", "ambiguous"])
def test_optional_local_headset_absence_leaves_the_borrowed_source_alone(
    rig, monkeypatch, missing
):
    _, state, events, _ = rig
    producer = _shared_producer(rig, quest_device=None)
    if missing == "adb":

        def no_adb():
            raise FileNotFoundError("ADB not found")

        monkeypatch.setattr(quest_webxr, "_find_adb", no_adb)
        expected = "No ADB installed; browser input is available"
    else:
        expected = (
            "No authorized USB headset detected"
            if missing == "headset"
            else "Multiple headsets; select one with --quest-device"
        )
        monkeypatch.setattr(quest_webxr, "_discover_device", lambda _: (None, expected))
    producer.start()
    _wait(lambda: producer.status_message == expected)
    assert producer.status_message == expected
    assert producer._failure is None
    assert state.headset is state.usb is None
    producer.stop()
    assert not state.source.stopped.is_set()
    assert not any(event.startswith("source-") for event in events)


def test_shared_headset_uses_lan_hub_but_usb_url_without_owning_the_source(rig):
    _, state, events, messages = rig
    producer = _shared_producer(rig)
    source = state.source
    state.fail_stage = "source"
    producer.start()
    _wait(lambda: "headset-start" in events)
    assert producer.status_message == "Press the headset power button"
    with pytest.raises(RuntimeError, match="poll the shared Isaac source"):
        producer.poll()
    state.woke = True
    _wait(lambda: state.page_entered)
    assert producer._source is source
    assert producer._failure is None
    assert producer.status_message == "Opening immersive VR..."
    assert state.hub_config["serverIP"] == "192.0.2.10"
    assert not messages
    assert not any(event.startswith("source-") for event in events)
    producer.stop()
    producer.stop()
    assert not source.stopped.is_set()
    assert (
        events.count("page-close")
        == events.count("usb-stop")
        == events.count("headset-stop")
        == 1
    )
    assert producer._failure is None


@pytest.mark.parametrize("stage", ["before-start", "wake", "usb", "hub", "browser"])
def test_shared_helper_cancellation_never_stops_the_borrowed_runtime(rig, stage):
    _, state, events, _ = rig
    producer = _shared_producer(rig)
    if stage == "before-start":
        producer.stop()
        producer.start()
        assert not events
    else:
        state.block_usb = stage == "usb"
        state.hub_failures = 10000 if stage == "hub" else 0
        producer.start()
        _wait(lambda: "headset-start" in events)
        if stage != "wake":
            state.woke = True
            _wait(lambda: "usb-start" in events)
        if stage == "hub":
            _wait(lambda: "hub-request" in events)
        if stage == "browser":
            _wait(lambda: state.page_entered)
        producer.stop()
        assert not producer._thread.is_alive()
        assert state.headset.stopped
        assert state.usb is None or state.usb.stopped
    assert producer._failure is None
    assert not state.source.stopped.is_set()
    assert not any(event.startswith("source-") for event in events)


@pytest.mark.parametrize("stage", ["headset", "usb", "page", "turn-exit"])
def test_shared_headset_failure_cleans_only_its_owned_resources(rig, stage):
    _, state, events, _ = rig
    producer = _shared_producer(rig)
    state.woke = True
    if stage != "turn-exit":
        state.fail_stage = stage
    producer.start()
    if stage == "turn-exit":
        _wait(lambda: state.page_entered)
        state.fail_stage = stage
    _wait(lambda: producer._failure is not None)
    assert (
        producer.status_message
        == "Unavailable; check controller battery and press B to retry"
    )
    producer.stop()
    assert state.headset.stopped
    assert state.usb is None or state.usb.stopped
    assert not state.source.stopped.is_set()
    assert not any(event.startswith("source-") for event in events)


@pytest.mark.parametrize("stage", ["wake", "browser", "failure"])
def test_shared_headset_restart_cleans_up_then_requires_a_new_physical_wake(rig, stage):
    _, state, events, _ = rig
    producer = _shared_producer(rig, require_physical_wake=stage == "wake")
    producer.start()
    _wait(lambda: "headset-start" in events)
    if stage != "wake":
        state.woke = True
        _wait(lambda: state.page_entered)
    if stage == "failure":
        state.fail_stage = "turn-exit"
        _wait(lambda: producer._failure is not None)
        state.fail_stage = None
    old_headset, old_usb = state.headset, state.usb
    state.woke = state.page_entered = False
    producer.restart()
    _wait(lambda: events.count("headset-start") == 2)
    assert state.headset is not old_headset
    assert old_headset.stopped
    assert old_usb is None or old_usb.stopped
    assert events.index("headset-stop") < len(events) - 1
    assert state.wake_options == [stage == "wake", True]
    assert producer.status_message == "Press the headset power button"
    assert not producer.wake_ready
    assert not state.page_entered
    assert producer._failure is None
    state.woke = True
    _wait(lambda: state.page_entered)
    producer.stop()
    producer.restart()
    assert not producer._thread.is_alive()
    assert events.count("headset-start") == events.count("headset-stop") == 2
    assert not state.source.stopped.is_set()
    assert not any(event.startswith("source-") for event in events)


def test_shared_headset_can_be_discovered_on_explicit_retry(rig, monkeypatch):
    _, state, events, _ = rig
    producer = _shared_producer(rig, quest_device=None)
    monkeypatch.setattr(quest_webxr, "_discover_device", lambda _: (None, "No headset"))
    producer.start()
    _wait(lambda: producer.status_message == "No headset")
    monkeypatch.setattr(
        quest_webxr, "_discover_device", lambda _: ("quest", "Headset detected")
    )
    producer.restart()
    _wait(lambda: "headset-start" in events)
    assert producer.status_message == "Press the headset power button"
    producer.stop()
    assert not producer._thread.is_alive()
    assert not state.source.stopped.is_set()


def test_stop_during_restart_cleanup_cannot_launch_another_headset(rig, monkeypatch):
    _, state, events, _ = rig
    producer = _shared_producer(rig)
    state.woke = True
    producer.start()
    _wait(lambda: state.page_entered)
    cleanup_started, release_cleanup = threading.Event(), threading.Event()
    stop_headset = state.headset.stop

    def cleanup():
        cleanup_started.set()
        assert release_cleanup.wait(3)
        stop_headset()

    monkeypatch.setattr(state.headset, "stop", cleanup)
    producer.restart()
    try:
        assert cleanup_started.wait(3)
        assert producer.status_message == "Restarting local headset..."
        stopper = threading.Thread(target=producer.stop)
        stopper.start()
        _wait(lambda: producer._stopped.is_set())
    finally:
        release_cleanup.set()
    stopper.join(timeout=3)
    assert not stopper.is_alive()
    assert not producer._thread.is_alive()
    assert events.count("headset-start") == 1
    assert not state.source.stopped.is_set()


@pytest.mark.parametrize("owner_allows", [False, True])
@pytest.mark.parametrize("peers", [[], [{"streaming": False}], [{"streaming": True}]])
def test_shared_headset_connect_requires_no_owner_and_no_streaming_hub_peer(
    rig, owner_allows, peers
):
    _, state, events, _ = rig
    producer = _shared_producer(rig, can_connect=lambda: owner_allows)
    state.hub_peers = peers
    assert producer._connection_available() is (
        owner_allows and not any(peer["streaming"] for peer in peers)
    )
    assert ("hub-request" in events) is owner_allows
    assert not any(event.startswith("source-") for event in events)


@pytest.mark.parametrize(
    "error",
    [ConnectionRefusedError(), TimeoutError(), http.client.IncompleteRead(b"")],
)
@pytest.mark.parametrize("streaming", [False, True])
def test_preconnect_hub_outage_recovers_without_restarting_headset(
    rig, error, streaming
):
    _, state, events, _ = rig
    producer = _shared_producer(rig)
    producer._await_hub()
    state.hub_failures = 2
    state.hub_error = error
    state.hub_peers = [{"streaming": streaming}]

    assert producer._connection_available() is not streaming
    assert events.count("hub-request") == events.count("hub-close") == 4
    assert producer._failure is None
    assert not any(event.startswith(("headset-", "source-")) for event in events)


def test_preconnect_hub_retry_rechecks_controller_ownership(rig, monkeypatch):
    _, state, events, _ = rig
    owner_free = [True]
    producer = _shared_producer(rig, can_connect=lambda: owner_free[0])
    state.hub_failures = 1
    monkeypatch.setattr(
        producer._stopped, "wait", lambda _: owner_free.__setitem__(0, False)
    )

    assert not producer._connection_available()
    assert events.count("hub-request") == 2


def test_stop_while_hub_request_recovers_cannot_permit_connect(rig, monkeypatch):
    producer = _shared_producer(rig)
    snapshot = producer._hub_snapshot

    def stopped_snapshot():
        state = snapshot()
        producer.stop()
        return state

    monkeypatch.setattr(producer, "_hub_snapshot", stopped_snapshot)
    with pytest.raises(RuntimeError, match="WebXR input stopped"):
        producer._connection_available()


def test_preconnect_hub_retry_has_a_deadline(rig, monkeypatch):
    _, state, events, _ = rig
    producer = _shared_producer(rig)
    producer._args.startup_timeout = 0.25
    state.hub_failures = 10000
    clock = [0.0]
    monkeypatch.setattr(
        quest_webxr, "time", SimpleNamespace(monotonic=lambda: clock[0])
    )
    monkeypatch.setattr(
        producer._stopped, "wait", lambda delay: clock.__setitem__(0, clock[0] + delay)
    )

    with pytest.raises(TimeoutError, match="OOB hub did not become ready"):
        producer._connection_available()
    assert events.count("hub-request") == events.count("hub-close") == 3
    assert clock[0] == pytest.approx(0.3)


def test_stop_during_preconnect_hub_retry_cleans_up_without_stopping_shared_runtime(
    rig, monkeypatch
):
    _, state, events, _ = rig
    producer = _shared_producer(rig)
    state.woke = True

    def run_page(cdp, guard, args, url, **callbacks):
        state.hub_failures = 10000
        callbacks["can_connect"]()
        pytest.fail("an unknown hub state must never permit CONNECT")

    monkeypatch.setattr(quest_webxr, "_run_page", run_page)
    producer.start()
    _wait(lambda: events.count("hub-request") >= 2)
    producer.stop()

    assert state.page_entered
    assert "page-close" in events
    assert state.headset.stopped and state.usb.stopped
    assert not state.source.stopped.is_set()
    assert not producer._thread.is_alive()
    assert producer._failure is None


@pytest.mark.parametrize("invalid", ["status", "address", "peers"])
def test_preconnect_hub_semantic_errors_are_not_retried(rig, invalid):
    _, state, events, _ = rig
    producer = _shared_producer(rig)
    if invalid == "status":
        state.hub_status = 401
    elif invalid == "address":
        state.hub_config = {"serverIP": "wrong.example", "port": 48322}
    else:
        state.hub_peers = [None]

    with pytest.raises(RuntimeError, match="local (CloudXR|OOB hub)"):
        producer._connection_available()
    assert events.count("hub-request") == events.count("hub-close") == 1


def test_preconnect_hub_programming_error_is_not_retried(rig):
    _, state, events, _ = rig
    producer = _shared_producer(rig)
    state.hub_failures = 1
    state.hub_error = http.client.InvalidURL("invalid local URL")

    with pytest.raises(http.client.InvalidURL, match="invalid local URL"):
        producer._connection_available()
    assert events.count("hub-request") == events.count("hub-close") == 1


def test_usb_source_starts_after_wake_with_only_local_web_assets(rig):
    producer, state, events, messages = rig
    producer.start(messages.append, lambda: None)
    _wait(lambda: "headset-start" in events)
    assert state.source is state.usb is None
    state.woke = True
    _wait(lambda: state.page_entered)
    assert state.source.options["host_client"]
    assert state.source.options["offline_web_client"]
    assert state.source.options["cloudxr_host"] == "127.0.0.1"
    assert producer._args.usb_turn_port == state.usb.turn_port
    assert producer._args.usb_turn_credential == state.usb.credential
    assert events.index("usb-start") < events.index("source-start")
    producer.stop()
    producer.stop()
    assert events.index("page-close") < events.index("usb-stop")
    assert events.index("source-stop") < events.index("usb-stop")
    assert events.index("usb-stop") < events.index("headset-stop")
    assert events.count("usb-stop") == 1
    assert producer._failure is None


@pytest.mark.parametrize("missing", ["assets", "coturn", "pin"])
def test_usb_preflight_fails_before_headset_changes(rig, monkeypatch, missing):
    producer, state, events, messages = rig
    if missing == "assets":

        def no_assets():
            raise RuntimeError("Missing cached assets")

        monkeypatch.setattr(quest_webxr, "cached_web_client_dir", no_assets)
    elif missing == "coturn":
        monkeypatch.setattr(quest_webxr.shutil, "which", lambda _: None)
    else:
        monkeypatch.setenv("QUEST_META_PIN", "test-pin")
    producer.start(messages.append, lambda: None)
    _wait(lambda: producer._failure is not None)
    producer.stop()
    assert state.headset is state.usb is state.source is None
    assert "headset-start" not in events


def test_injected_source_starts_once_and_only_direct_poll_consumes_frames(
    rig, monkeypatch
):
    _, state, events, messages = rig
    producer = _injected_producer(rig)
    source = state.source
    state.observation = Observation(
        frame={"left": None, "right": None},
        capture_ns=123,
        source="isaac-teleop",
        generation=7,
    )

    def unexpected_source(**options):
        raise AssertionError("the injected source must own the only runtime")

    monkeypatch.setattr(isaac_teleop_source, "IsaacTeleopSource", unexpected_source)
    assert producer.poll() is None
    producer.start(wait_for_frame=False)
    _wait(lambda: "headset-start" in events)
    assert producer.poll() is None
    assert state.health_checks == 0
    assert "source-start" not in events
    assert "source-poll" not in events

    state.hub_failures = 10000
    state.woke = True
    _wait(lambda: "hub-request" in events)
    producer.start()
    producer.check_alive()
    assert producer._source is source
    assert producer._source_started
    assert events.count("source-start") == 1
    assert "source-poll" not in events
    assert not state.page_entered
    observation = producer.poll()
    assert observation == Observation(
        frame=state.observation.frame,
        capture_ns=123,
        source="quest-webxr",
        generation=7,
    )
    assert state.observation.source == "isaac-teleop"

    state.hub_failures = 0
    _wait(lambda: state.page_entered)
    checks = state.health_checks
    _wait(lambda: state.health_checks > checks)
    assert events.count("source-poll") == 1
    producer.stop()
    producer.stop()
    assert not producer._source_started
    assert events.count("source-stop") == events.count("headset-stop") == 1
    assert events.count("page-close") == 1


@pytest.mark.parametrize("accept_eula", [False, True])
def test_local_cloudxr_starts_only_after_wake_and_forwards_original_records(
    rig, accept_eula
):
    producer, state, events, messages = rig
    if not accept_eula:
        with pytest.raises(ValueError, match="accept-eula"):
            quest_webxr.QuestWebXRProducer(_args(isaac_teleop_accept_eula=False))
        assert events == []
        return
    producer._args.isaac_teleop_accept_eula = accept_eula
    producer.start(messages.append, lambda: events.append("disconnected"))
    _wait(lambda: "headset-start" in events)
    assert state.source is None
    assert not producer.wake_ready
    assert state.usb is None
    state.woke = True
    _wait(lambda: state.page_entered)
    assert state.source.options == dict(
        python_executable="sdk-python",
        accept_eula=accept_eula,
        oob_hub=True,
        wait_for_controllers=False,
        startup_timeout_s=120,
        cloudxr_host="127.0.0.1",
        cloudxr_port=48322,
        host_client=True,
        offline_web_client=True,
        on_message=messages.append,
        on_disconnect=state.source.options["on_disconnect"],
    )
    assert messages == [
        b'{"type":"hello","protocol":5}',
        b'{"type":"xr_frame","t_client":123.456}',
    ]
    producer.stop()
    producer.stop()
    assert not producer._thread.is_alive()
    assert events.count("source-stop") == events.count("headset-stop") == 1
    assert events.index("page-close") < events.index("headset-stop")
    assert events[-1] == "disconnected"
    assert producer._failure is None
    assert events.count("find-adb") == 1


@pytest.mark.parametrize("injected", [False, True])
@pytest.mark.parametrize(
    "stage", ["before-start", "wake", "usb", "source", "hub", "browser"]
)
def test_cancellation_unwinds_every_startup_stage(rig, stage, injected):
    producer, state, events, messages = rig
    if injected:
        producer = _injected_producer(rig)
    if stage == "before-start":
        producer.stop()
        producer.start(messages.append, lambda: None)
        assert events == (["source-stop"] if injected else [])
        return
    state.block_usb = stage == "usb"
    state.block_source = stage == "source"
    if stage == "hub":
        state.hub_failures = 10000
    producer.start(messages.append, lambda: None)
    _wait(lambda: "headset-start" in events)
    assert producer.poll() is None
    if stage != "wake":
        state.woke = True
        _wait(lambda: ("usb-start" if stage == "usb" else "source-start") in events)
    if stage == "source":
        assert producer.poll() is None
    if stage == "hub":
        _wait(lambda: "hub-request" in events)
    if stage == "browser":
        _wait(lambda: state.page_entered)
    producer.stop()
    assert not producer._thread.is_alive()
    assert state.headset.stopped
    assert state.usb is None or state.usb.stopped
    if state.source is not None:
        assert state.source.stopped.is_set()
    assert producer._failure is None


@pytest.mark.parametrize("injected", [False, True])
@pytest.mark.parametrize(
    "stage", ["headset", "usb", "source", "page", "health", "turn-exit"]
)
def test_failure_reaches_relay_and_cleans_owned_resources(rig, stage, injected):
    producer, state, events, messages = rig
    if injected:
        producer = _injected_producer(rig)
    state.woke = True
    if stage != "turn-exit":
        state.fail_stage = stage
    producer.start(messages.append, lambda: None)
    if stage == "health":
        _wait(lambda: state.page_entered)
        state.health_failed = True
    if stage == "turn-exit":
        _wait(lambda: state.page_entered)
        state.fail_stage = stage
    _wait(lambda: producer._failure is not None)
    with pytest.raises(RuntimeError, match="input failed"):
        producer.check_alive()
    producer.stop()
    assert state.headset.stopped
    assert state.usb is None or state.usb.stopped
    assert state.source is None or state.source.stopped.is_set()
    assert not producer._source_started
    assert "source-poll" not in events


@pytest.mark.parametrize("shared", [False, True])
@pytest.mark.parametrize("keep_open", [False, True])
@pytest.mark.parametrize("failure", ["run-page", "turn-exit", "health"])
def test_browser_failure_hold_defers_cleanup_until_stop(
    rig, shared, keep_open, failure
):
    producer, state, events, messages = rig
    if shared:
        producer = _shared_producer(rig)
    producer._args.keep_browser_open_on_error = keep_open
    state.woke = True

    def disconnect():
        messages.clear()
        events.append("disconnected")

    producer.start(messages.append, disconnect)
    _wait(lambda: state.page_entered)
    messages.append(b"queued controller frame")
    if failure == "health":
        state.health_failed = True
    else:
        state.fail_stage = failure
    _wait(lambda: producer._failure is not None)
    _wait(lambda: "disconnected" in events)
    assert not messages
    if keep_open:
        assert producer._thread.is_alive()
        assert not producer.wake_ready
        assert not state.headset.stopped
        assert not state.usb.stopped
        assert not state.source.stopped.is_set()
        assert "page-close" not in events
        assert producer.status_message == (
            "Connection failed; browser kept open; press B to retry"
            if shared
            else "Connection failed; browser kept open; Ctrl-C to close"
        )
        health_checks = state.health_checks
        producer.check_alive()
        assert state.health_checks == health_checks
        if not shared:
            assert producer.poll() is None
            assert "source-poll" not in events
    else:
        _wait(lambda: state.headset.stopped)
        assert "page-close" in events
        assert state.usb.stopped
        with pytest.raises(RuntimeError, match="input failed"):
            producer.check_alive()
    producer.stop()
    producer.stop()
    assert not producer._thread.is_alive()
    assert events.count("page-close") == 1
    assert events.index("page-close") < events.index("usb-stop")
    assert events.index("usb-stop") < events.index("headset-stop")
    assert events.count("headset-stop") == events.count("usb-stop") == 1
    assert state.source.stopped.is_set() is (not shared)
    assert events[-1] == "disconnected"


def test_browser_failure_hold_retry_cleans_up_before_new_physical_wake(rig):
    _, state, events, _ = rig
    producer = _shared_producer(rig, require_physical_wake=False)
    producer._args.keep_browser_open_on_error = True
    state.woke = True
    producer.start()
    _wait(lambda: state.page_entered)
    state.fail_stage = "run-page"
    _wait(lambda: producer._failure is not None)
    assert "page-close" not in events
    old_headset, old_usb = state.headset, state.usb
    state.woke = state.page_entered = False
    state.fail_stage = None
    producer.restart()
    _wait(lambda: events.count("headset-start") == 2)
    assert old_headset.stopped and old_usb.stopped
    assert events.index("page-close") < events.index("usb-stop")
    assert events.index("usb-stop") < events.index("headset-stop")
    assert events.index("headset-stop") < len(events) - 1
    assert state.wake_options == [False, True]
    assert producer._failure is None
    assert not producer.wake_ready
    assert producer.status_message == "Press the headset power button"
    assert not state.page_entered
    state.woke = True
    _wait(lambda: state.page_entered)
    assert producer.wake_ready
    producer.check_alive()
    producer.stop()
    assert events.count("page-close") == events.count("headset-stop") == 2
    assert not state.source.stopped.is_set()


@pytest.mark.parametrize("shared", [False, True])
@pytest.mark.parametrize("stage", ["headset", "usb", "page"])
def test_browser_hold_flag_does_not_preserve_pre_page_failures(rig, shared, stage):
    producer, state, events, messages = rig
    if shared:
        producer = _shared_producer(rig)
    producer._args.keep_browser_open_on_error = True
    state.woke = True
    state.fail_stage = stage
    producer.start(messages.append, lambda: None)
    _wait(lambda: producer._failure is not None)
    assert not state.page_entered
    assert state.headset.stopped
    assert state.usb is None or state.usb.stopped
    assert "browser kept open" not in producer.status_message
    with pytest.raises(RuntimeError, match="input failed"):
        producer.check_alive()
    producer.stop()
    assert not producer._thread.is_alive()
    if state.source is not None:
        assert state.source.stopped.is_set() is (not shared)


@pytest.mark.parametrize("shared", [False, True])
@pytest.mark.parametrize("stage", ["wake", "browser"])
def test_browser_hold_flag_never_intercepts_cancellation(rig, shared, stage):
    producer, state, events, messages = rig
    if shared:
        producer = _shared_producer(rig)
    producer._args.keep_browser_open_on_error = True
    producer.start(messages.append, lambda: None)
    _wait(lambda: "headset-start" in events)
    if stage == "browser":
        state.woke = True
        _wait(lambda: state.page_entered)
    producer.stop()
    assert not producer._thread.is_alive()
    assert producer._failure is None
    assert state.headset.stopped
    assert state.usb is None or state.usb.stopped
    if state.source is not None:
        assert state.source.stopped.is_set() is (not shared)


@pytest.mark.parametrize(
    "options, match",
    [
        ({"cloudxr_port": 0}, "port"),
        ({"cloudxr_port": 65536}, "port"),
        ({"startup_timeout": float("inf")}, "finite"),
        ({"startup_timeout": 0}, "finite"),
    ],
)
def test_invalid_webxr_configuration_fails_before_headset_start(options, match):
    with pytest.raises(ValueError, match=match):
        quest_webxr.QuestWebXRProducer(_args(**options))


@pytest.mark.parametrize("accept_certificate", [False, True])
@pytest.mark.parametrize("accept_eula", [False, True])
@pytest.mark.parametrize("keep_open", [False, True])
def test_cli_separates_dgx_transport_from_laptop_cloudxr(
    monkeypatch, accept_certificate, accept_eula, keep_open
):
    options = {}

    class Relay:
        def __init__(self, **kwargs):
            options.update(kwargs)

        def run(self):
            pass

        def stop(self):
            pass

    monkeypatch.setenv("FLASHDREAMS_CONTROLLER_TOKEN", "x" * 40)
    monkeypatch.setattr(quest_relay, "QuestRelay", Relay)
    monkeypatch.setattr(
        "sys.argv",
        [
            "flashdreams-quest-relay",
            "--server",
            "dgx.example",
            "--ca-cert",
            "ca.pem",
            "--cloudxr-port",
            "48323",
            *(["--isaac-teleop-accept-eula"] if accept_eula else []),
            "--require-physical-wake",
            *(["--accept-server-certificate"] if accept_certificate else []),
            *(["--keep-browser-open-on-error"] if keep_open else []),
            "--quest-device",
            "quest",
        ],
    )
    if not accept_eula:
        with pytest.raises(SystemExit) as error:
            quest_relay.main()
        assert error.value.code == 2
        assert options == {}
        return
    quest_relay.main()
    assert options["server"] == "dgx.example"
    assert options["port"] == 8445
    producer = options["webxr"]
    assert producer._args.cloudxr_port == 48323
    assert producer._args.require_physical_wake
    assert producer._args.accept_server_certificate is accept_certificate
    assert producer._args.isaac_teleop_accept_eula is accept_eula
    assert producer._args.keep_browser_open_on_error is keep_open


def test_browser_starts_after_hub_without_any_controller_sample(rig, monkeypatch):
    producer, state, events, messages = rig
    state.woke = True
    state.emit_frames = False
    state.hub_failures = 1
    monkeypatch.setenv("CONTROL_TOKEN", "local-hub-test-token")
    monkeypatch.setenv("FLASHDREAMS_CONTROLLER_TOKEN", "relay-test-token")
    monkeypatch.setenv("HTTPS_PROXY", "https://must-not-use-proxy.invalid")
    producer.start(messages.append, lambda: None)
    _wait(lambda: state.page_entered)
    assert not messages
    assert events.index("hub-request") < events.index("page-open")
    assert events.count("hub-close") == 2
    assert state.hub_headers == {}


@pytest.mark.parametrize(
    "status, config",
    [
        (401, None),
        (302, None),
        (404, None),
        (200, {"serverIP": "dgx.example", "port": 48322}),
        (200, {"serverIP": "127.0.0.1", "port": 9999}),
        (200, []),
    ],
)
def test_wrong_or_unauthorized_hub_never_opens_browser(rig, status, config):
    producer, state, events, messages = rig
    state.woke = True
    state.hub_status = status
    state.hub_config = config
    producer.start(messages.append, lambda: None)
    _wait(lambda: producer._failure is not None)
    producer.stop()
    assert not state.page_entered
    assert "hub-close" in events
    assert state.headset.stopped


def test_cancel_while_local_hub_is_not_ready(rig):
    producer, state, events, messages = rig
    state.woke = True
    state.hub_failures = 10000
    producer.start(messages.append, lambda: None)
    _wait(lambda: "hub-request" in events)
    producer.stop()
    assert not producer._thread.is_alive()
    assert not state.page_entered
    assert state.headset.stopped
    assert producer._failure is None
