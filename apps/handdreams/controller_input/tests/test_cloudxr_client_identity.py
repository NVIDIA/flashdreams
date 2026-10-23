# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU checks for connection-scoped IWER identity and hosted-client injection."""

from __future__ import annotations

import asyncio
import copy
import sys
import time
from contextlib import suppress
from types import ModuleType, SimpleNamespace

import pytest
from flashdreams_controller_input import cloudxr_client_identity as identity

pytestmark = pytest.mark.ci_cpu


def _peer(**overrides):
    return {
        "clientId": "browser-1",
        "streamingSince": 123,
        "connected": True,
        "streaming": True,
        "metricsByCadence": {identity._METRICS_CADENCE: {"metrics": {"iwer": 1}}},
        **overrides,
    }


def test_only_the_unique_connected_streaming_peer_supplies_identity():
    snapshot = {
        "headsets": [
            _peer(clientId="idle", streaming=False),
            _peer(clientId="gone", connected=False),
            _peer(),
        ]
    }
    assert identity._stream_identity(snapshot) == ("browser-1:123", True)
    snapshot["headsets"][-1]["streamingSince"] = 124
    assert identity._stream_identity(snapshot) == ("browser-1:124", True)


@pytest.mark.parametrize("value", [0, None, True, False, "1", "true", 2, [], {}])
def test_unknown_or_non_iwer_metric_never_enables_emulation(value):
    peer = _peer(
        metricsByCadence={identity._METRICS_CADENCE: {"metrics": {"iwer": value}}}
    )
    assert identity._stream_identity({"headsets": [peer]}) == ("browser-1:123", False)


@pytest.mark.parametrize("value", [1, 1.0])
def test_explicit_numeric_iwer_metric_enables_emulation(value):
    peer = _peer(
        metricsByCadence={identity._METRICS_CADENCE: {"metrics": {"iwer": value}}}
    )
    assert identity._stream_identity({"headsets": [peer]}) == ("browser-1:123", True)


@pytest.mark.parametrize(
    "snapshot",
    [
        {},
        {"headsets": []},
        {"headsets": [_peer(), _peer(clientId="second")]},
        {"headsets": [_peer(connected=False)]},
        {"headsets": [_peer(streaming=False)]},
        {"headsets": [_peer(clientId=None)]},
        {"headsets": [_peer(clientId="")]},
        {"headsets": [_peer(clientId="a" * 512)]},
        {"headsets": [_peer(streamingSince=None)]},
        {"headsets": [_peer(streamingSince=True)]},
        {"headsets": [_peer(streamingSince=-1)]},
        {"headsets": None},
        {"headsets": [None]},
        {"headsets": [_peer(), None]},
    ],
)
def test_unidentified_or_ambiguous_stream_defaults_to_full_setup(snapshot):
    assert identity._stream_identity(snapshot) == (None, False)


@pytest.mark.parametrize(
    "metrics",
    [
        None,
        [],
        {},
        {identity._METRICS_CADENCE: None},
        {identity._METRICS_CADENCE: {"metrics": None}},
        {identity._METRICS_CADENCE: {"metrics": []}},
    ],
)
def test_missing_or_malformed_metrics_do_not_identify_an_emulator(metrics):
    assert identity._stream_identity(
        {"headsets": [_peer(metricsByCadence=metrics)]}
    ) == ("browser-1:123", False)


def _install_sdk(monkeypatch, factory):
    sdk, cloudxr = ModuleType("isaacteleop"), ModuleType("isaacteleop.cloudxr")
    wss = SimpleNamespace(_make_http_handler=factory)
    setattr(cloudxr, "wss", wss)
    setattr(sdk, "cloudxr", cloudxr)
    monkeypatch.setitem(sys.modules, "isaacteleop", sdk)
    monkeypatch.setitem(sys.modules, "isaacteleop.cloudxr", cloudxr)
    return wss


class _Hub:
    def __init__(self):
        self.ws = object()
        self._headsets = {
            self.ws: SimpleNamespace(
                client_id="browser-1",
                streaming=True,
                streaming_since=0.123,
                metrics_by_cadence={
                    identity._METRICS_CADENCE: {"metrics": {"iwer": 1}}
                },
            )
        }
        self.snapshot_ready = asyncio.Event()
        self.snapshot_entered = asyncio.Event()
        self.snapshot_resume = asyncio.Event()
        self.pause_snapshot = False
        self.mutation_entered = asyncio.Event()
        self.mutation_resume = asyncio.Event()
        self.pause_mutation = False

    async def get_snapshot(self):
        snapshot = {
            "headsets": [
                _peer(
                    clientId=peer.client_id,
                    streaming=peer.streaming,
                    streamingSince=(
                        int(peer.streaming_since * 1000)
                        if peer.streaming_since is not None
                        else None
                    ),
                    metricsByCadence=copy.deepcopy(peer.metrics_by_cadence),
                )
                for peer in self._headsets.values()
            ]
        }
        if self.pause_snapshot:
            self.pause_snapshot = False
            self.snapshot_entered.set()
            await self.snapshot_resume.wait()
        self.snapshot_ready.set()
        return snapshot

    async def _wait_for_mutation(self):
        if self.pause_mutation:
            self.mutation_entered.set()
            await self.mutation_resume.wait()

    async def _handle_register(self, ws, client_id, _payload):
        await self._wait_for_mutation()
        self._headsets[ws] = SimpleNamespace(
            client_id=client_id,
            streaming=False,
            streaming_since=None,
            metrics_by_cadence={},
        )
        return True

    async def _dispatch_headset(self, ws, msg_type, payload):
        await self._wait_for_mutation()
        peer = self._headsets.get(ws)
        if peer is None:
            return
        if msg_type == "streamStatus":
            streaming = bool(payload.get("streaming", False))
            if streaming and not peer.streaming:
                peer.streaming_since = 0.124
            elif not streaming:
                peer.streaming_since = None
            peer.streaming = streaming
        elif msg_type == "clientMetrics":
            peer.metrics_by_cadence[payload["cadence"]] = {
                "metrics": payload["metrics"]
            }

    async def handle_connection(self, ws):
        self._headsets.pop(ws, None)


@pytest.mark.parametrize(
    ("path", "status", "static_dir", "response_present", "injected"),
    [
        ("/client", 200, "/cached", True, True),
        ("/client/?oobEnable=1", 200, "/cached", True, True),
        ("/client/index.html", 200, "/cached", True, True),
        ("/client/bundle.js", 200, "/cached", True, False),
        ("/oob/v1/ws", 200, "/cached", False, False),
        ("/api/oob/v1/state", 200, "/cached", True, False),
        ("/client/", 404, "/cached", True, False),
        ("/client/", 200, None, True, False),
    ],
)
def test_runtime_signal_is_injected_only_into_hosted_html(
    monkeypatch, path, status, static_dir, response_present, injected
):
    original = (
        b"<!DOCTYPE html><html><head><script src='bundle.js'></script></head></html>"
    )
    response = (
        SimpleNamespace(
            status_code=status,
            body=original,
            headers={"Content-Length": str(len(original))},
        )
        if response_present
        else None
    )
    calls = []

    def factory(host, port, **options):
        calls.append((host, port, options))

        async def handle(_connection, _request):
            return response

        return handle

    wss = _install_sdk(monkeypatch, factory)

    async def run():
        with identity.CloudXRClientIdentity(enabled=True):
            handler = wss._make_http_handler("127.0.0.1", 1234, static_dir=static_dir)
            assert await handler(None, SimpleNamespace(path=path)) is response
        assert wss._make_http_handler is factory

    asyncio.run(run())
    assert calls == [("127.0.0.1", 1234, {"hub": None, "static_dir": static_dir})]
    if response is not None:
        assert (identity._CLIENT_SCRIPT in response.body) is injected
        assert int(response.headers["Content-Length"]) == len(response.body)
        if injected:
            assert response.body.index(identity._CLIENT_SCRIPT) < response.body.index(
                b"<script src='bundle.js'>"
            )
        else:
            assert response.body == original


def test_delayed_snapshot_keeps_identity_and_exit_cancels_watcher(monkeypatch):
    def factory(*_args, **_kwargs):
        return None

    wss = _install_sdk(monkeypatch, factory)

    async def run():
        ready = asyncio.Event()

        hub = _Hub()
        hub.snapshot_ready = ready

        with identity.CloudXRClientIdentity(enabled=True) as tracker:
            assert tracker.sample() == (None, False)
            wss._make_http_handler("127.0.0.1", 1234, hub=hub)
            task = tracker._task
            assert task is not None
            await ready.wait()
            assert tracker.sample() == ("browser-1:123", True)
            hub.pause_snapshot = True
            await asyncio.wait_for(hub.snapshot_entered.wait(), 1)
            with monkeypatch.context() as elapsed:
                elapsed.setattr(time, "monotonic", lambda: 1e12)
                assert tracker.sample() == ("browser-1:123", True)
        with suppress(asyncio.CancelledError):
            await task
        assert task.cancelled()
        assert tracker.sample() == (None, False)
        assert wss._make_http_handler is factory

    asyncio.run(run())


def test_disabled_identity_does_not_require_the_sdk(monkeypatch):
    monkeypatch.setitem(sys.modules, "isaacteleop.cloudxr", None)
    with identity.CloudXRClientIdentity(enabled=False) as tracker:
        assert tracker.sample() == (None, False)


@pytest.mark.parametrize("initial", [(None, False), ("browser-1:123", True)])
def test_hub_failure_retries_without_replacing_last_confirmed_identity(caplog, initial):
    async def run():
        tracker = identity.CloudXRClientIdentity(enabled=True)
        tracker._identity = initial
        recovered = asyncio.Event()
        calls = 0

        async def get_snapshot():
            nonlocal calls
            calls += 1
            assert tracker.sample() == initial
            if calls <= 2:
                raise RuntimeError("temporary hub failure")
            recovered.set()
            return {"headsets": [_peer()]}

        task = asyncio.create_task(
            tracker._watch(SimpleNamespace(get_snapshot=get_snapshot))
        )
        tracker._task = task
        await asyncio.wait_for(recovered.wait(), 1)
        assert calls == 3
        assert not task.done()
        assert tracker.sample() == ("browser-1:123", True)
        assert caplog.text.count("retrying") == 1
        tracker.close()
        tracker.close()
        with suppress(asyncio.CancelledError):
            await task
        assert tracker.sample() == (None, False)

    asyncio.run(run())


@pytest.mark.parametrize("mutation", ["register", "streamStatus", "clientMetrics"])
def test_lifecycle_mutation_invalidates_identity_before_the_hub_await(
    monkeypatch, mutation
):
    wss = _install_sdk(monkeypatch, lambda *_args, **_kwargs: None)

    async def run():
        hub = _Hub()
        with identity.CloudXRClientIdentity(enabled=True) as tracker:
            wss._make_http_handler("127.0.0.1", 1234, hub=hub)
            await asyncio.wait_for(hub.snapshot_ready.wait(), 1)
            assert tracker.sample() == ("browser-1:123", True)
            hub.pause_mutation = True
            if mutation == "register":
                pending = asyncio.create_task(hub._handle_register(hub.ws, "new", {}))
            else:
                payload = (
                    {"streaming": False}
                    if mutation == "streamStatus"
                    else {"cadence": identity._METRICS_CADENCE, "metrics": {"iwer": 0}}
                )
                pending = asyncio.create_task(
                    hub._dispatch_headset(hub.ws, mutation, payload)
                )
            await asyncio.wait_for(hub.mutation_entered.wait(), 1)
            assert tracker.sample() == (None, False)
            hub.mutation_resume.set()
            await pending
            expected = {
                "register": (None, False),
                "streamStatus": (None, False),
                "clientMetrics": ("browser-1:123", False),
            }[mutation]
            assert tracker.sample() == expected
        with suppress(asyncio.CancelledError):
            assert tracker._task is not None
            await tracker._task

    asyncio.run(run())


def test_an_old_snapshot_cannot_restore_iwer_during_a_physical_handoff(monkeypatch):
    wss = _install_sdk(monkeypatch, lambda *_args, **_kwargs: None)

    async def run():
        hub = _Hub()
        with identity.CloudXRClientIdentity(enabled=True) as tracker:
            wss._make_http_handler("127.0.0.1", 1234, hub=hub)
            await asyncio.wait_for(hub.snapshot_ready.wait(), 1)
            hub.pause_snapshot = True
            old_snapshot = asyncio.create_task(tracker._refresh(hub))
            await asyncio.wait_for(hub.snapshot_entered.wait(), 1)
            hub.pause_mutation = True
            handoff = asyncio.create_task(
                hub._dispatch_headset(
                    hub.ws,
                    "clientMetrics",
                    {"cadence": identity._METRICS_CADENCE, "metrics": {"iwer": 0}},
                )
            )
            await asyncio.wait_for(hub.mutation_entered.wait(), 1)
            hub.snapshot_resume.set()
            await old_snapshot
            assert tracker.sample() == (None, False)
            hub.mutation_resume.set()
            await handoff
            assert tracker.sample() == ("browser-1:123", False)
        with suppress(asyncio.CancelledError):
            assert tracker._task is not None
            await tracker._task

    asyncio.run(run())


@pytest.mark.parametrize(
    ("kind", "payload"),
    [
        ("clientMetrics", {"cadence": "performance", "metrics": {"fps": 72}}),
        (
            "clientMetrics",
            {"cadence": identity._METRICS_CADENCE, "metrics": {"iwer": 1}},
        ),
        ("streamStatus", {"streaming": True}),
    ],
)
def test_same_connection_updates_do_not_interrupt_emulated_setup(
    monkeypatch, kind, payload
):
    wss = _install_sdk(monkeypatch, lambda *_args, **_kwargs: None)

    async def run():
        hub = _Hub()
        with identity.CloudXRClientIdentity(enabled=True) as tracker:
            wss._make_http_handler("127.0.0.1", 1234, hub=hub)
            await asyncio.wait_for(hub.snapshot_ready.wait(), 1)
            hub.pause_mutation = True
            pending = asyncio.create_task(hub._dispatch_headset(hub.ws, kind, payload))
            await asyncio.wait_for(hub.mutation_entered.wait(), 1)
            await tracker._refresh(hub)
            assert tracker.sample() == ("browser-1:123", True)
            hub.mutation_resume.set()
            await pending
            assert tracker.sample() == ("browser-1:123", True)
        with suppress(asyncio.CancelledError):
            assert tracker._task is not None
            await tracker._task

    asyncio.run(run())


def test_idle_peer_registration_and_disconnect_preserve_active_identity(monkeypatch):
    wss = _install_sdk(monkeypatch, lambda *_args, **_kwargs: None)

    async def run():
        hub = _Hub()
        idle_ws = object()
        with identity.CloudXRClientIdentity(enabled=True) as tracker:
            wss._make_http_handler("127.0.0.1", 1234, hub=hub)
            await asyncio.wait_for(hub.snapshot_ready.wait(), 1)
            hub.pause_mutation = True
            registration = asyncio.create_task(
                hub._handle_register(idle_ws, "idle", {})
            )
            await asyncio.wait_for(hub.mutation_entered.wait(), 1)
            await tracker._refresh(hub)
            assert tracker.sample() == ("browser-1:123", True)
            hub.mutation_resume.set()
            await registration
            assert tracker.sample() == ("browser-1:123", True)

            hub.pause_snapshot = True
            disconnection = asyncio.create_task(hub.handle_connection(idle_ws))
            await asyncio.wait_for(hub.snapshot_entered.wait(), 1)
            assert tracker.sample() == ("browser-1:123", True)
            hub.snapshot_resume.set()
            await disconnection
            assert tracker.sample() == ("browser-1:123", True)
        with suppress(asyncio.CancelledError):
            assert tracker._task is not None
            await tracker._task

    asyncio.run(run())


@pytest.mark.parametrize("boundary", ["disconnect", "stop-stream"])
def test_confirmed_handoff_during_poll_failure_does_not_inherit_iwer(
    monkeypatch, boundary
):
    wss = _install_sdk(monkeypatch, lambda *_args, **_kwargs: None)

    async def run():
        hub = _Hub()
        with identity.CloudXRClientIdentity(enabled=True) as tracker:
            wss._make_http_handler("127.0.0.1", 1234, hub=hub)
            await asyncio.wait_for(hub.snapshot_ready.wait(), 1)
            get_snapshot = hub.get_snapshot
            failed = asyncio.Event()

            async def unavailable():
                failed.set()
                raise RuntimeError("temporary polling outage")

            setattr(hub, "get_snapshot", unavailable)
            await asyncio.wait_for(failed.wait(), 1)
            assert tracker.sample() == ("browser-1:123", True)
            if boundary == "disconnect":
                await hub.handle_connection(hub.ws)
            else:
                await hub._dispatch_headset(
                    hub.ws, "streamStatus", {"streaming": False}
                )
            assert tracker.sample() == (None, False)

            physical_ws = object()
            await hub._handle_register(physical_ws, "physical", {})
            await hub._dispatch_headset(
                physical_ws, "streamStatus", {"streaming": True}
            )
            assert tracker.sample() == (None, False)
            setattr(hub, "get_snapshot", get_snapshot)
            await tracker._refresh(hub)
            assert tracker.sample() == ("physical:124", False)
            assert tracker._task is not None
            assert not tracker._task.done()
        with suppress(asyncio.CancelledError):
            assert tracker._task is not None
            await tracker._task

    asyncio.run(run())


@pytest.mark.parametrize(
    ("snapshot", "expected"),
    [
        ({"headsets": []}, (None, False)),
        ({"headsets": [_peer(), _peer(clientId="physical")]}, (None, False)),
        ({"headsets": [_peer(), None]}, (None, False)),
        ({"headsets": None}, (None, False)),
        ({"headsets": [_peer(metricsByCadence=None)]}, ("browser-1:123", False)),
        (
            {"headsets": [_peer(clientId="physical", metricsByCadence={})]},
            ("physical:123", False),
        ),
    ],
)
def test_successful_unknown_or_changed_snapshot_never_reuses_iwer(snapshot, expected):
    async def run():
        tracker = identity.CloudXRClientIdentity(enabled=True)
        tracker._identity = ("browser-1:123", True)

        async def get_snapshot():
            return snapshot

        await tracker._refresh(SimpleNamespace(get_snapshot=get_snapshot))
        assert tracker.sample() == expected
        tracker.close()

    asyncio.run(run())


def test_disconnection_clears_iwer_before_waiting_for_a_new_snapshot(monkeypatch):
    wss = _install_sdk(monkeypatch, lambda *_args, **_kwargs: None)

    async def run():
        hub = _Hub()
        with identity.CloudXRClientIdentity(enabled=True) as tracker:
            original = hub.handle_connection
            wss._make_http_handler("127.0.0.1", 1234, hub=hub)
            await asyncio.wait_for(hub.snapshot_ready.wait(), 1)
            hub.pause_snapshot = True
            pending = asyncio.create_task(hub.handle_connection(hub.ws))
            await asyncio.wait_for(hub.snapshot_entered.wait(), 1)
            assert tracker.sample() == (None, False)
            hub.snapshot_resume.set()
            await pending
            assert tracker.sample() == (None, False)
        with suppress(asyncio.CancelledError):
            assert tracker._task is not None
            await tracker._task
        assert hub.handle_connection == original

    asyncio.run(run())
