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

"""CPU contracts for the laptop-side Quest WebXR companion."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, cast
from urllib.parse import parse_qs, urlsplit

import pytest
from flashdreams_controller_input import quest_webxr as webxr

pytestmark = pytest.mark.ci_cpu

_ORIGIN = "https://dgx.example:48322"


class _Socket:
    def __init__(self, messages=()):
        self.sent = []
        self.messages = list(messages)

    def send(self, value):
        self.sent.append(json.loads(value))

    def recv(self, *, timeout):
        assert 0 < timeout <= 5
        if self.messages:
            return json.dumps(self.messages.pop(0))
        request = self.sent[-1]
        result = (
            {"targetId": "owned-tab"}
            if request["method"] == "Target.createTarget"
            else {}
        )
        return json.dumps({"id": request["id"], "result": result})


def _socket_event(cdp, method, request_id="owned-socket", **params):
    cdp._event(
        {"method": f"Network.{method}", "params": {"requestId": request_id, **params}}
    )


def _frame(cdp, direction, kind, payload, request_id="owned-socket"):
    _socket_event(
        cdp,
        f"webSocketFrame{direction}",
        request_id,
        response={"payloadData": json.dumps({"type": kind, "payload": payload})},
    )


def _websockets(monkeypatch, connect):
    module = ModuleType("websockets.sync.client")
    setattr(module, "connect", connect)
    monkeypatch.setitem(sys.modules, "websockets.sync.client", module)


@pytest.mark.parametrize(
    ("url", "action"),
    [
        (_ORIGIN + "/oob/v1/ws", "continue"),
        ("wss://dgx.example:48322/oob/v1/ws", "continue"),
        ("https://dgx.example:443/", "cancel"),
        ("http://dgx.example:48322/", "cancel"),
        ("https://dgx.example.evil:48322/", "cancel"),
        ("https://nvidia.github.io/", "cancel"),
    ],
)
def test_certificate_exceptions_only_continue_the_server_origin(url, action):
    socket = _Socket()
    webxr._CDP(socket, _ORIGIN)._event(
        {
            "method": "Security.certificateError",
            "params": {"eventId": 12, "requestURL": url},
        }
    )
    assert [(entry["method"], entry["params"]) for entry in socket.sent] == [
        ("Security.handleCertificateError", {"eventId": 12, "action": action})
    ]


def test_certificate_origin_normalizes_hostname_case_and_default_port():
    socket = _Socket()
    cdp = webxr._CDP(socket, "https://DGX.EXAMPLE:443")
    cdp._event(
        {
            "method": "Security.certificateError",
            "params": {"eventId": 1, "requestURL": "wss://dgx.example/oob/v1/ws"},
        }
    )
    assert socket.sent[-1]["params"]["action"] == "continue"
    assert webxr._https_origin("https://user:password@dgx.example/") == ""


@pytest.mark.parametrize(
    "url",
    [
        "wss://other.example:48322/oob/v1/ws",
        "wss://dgx.example:48322/other",
        "ws://dgx.example:48322/oob/v1/ws",
    ],
)
def test_other_websockets_cannot_report_streaming(url):
    cdp = webxr._CDP(_Socket(), _ORIGIN)
    _socket_event(cdp, "webSocketCreated", url=url)
    _frame(cdp, "Received", "hello", {"clientId": "other"})
    _frame(cdp, "Sent", "streamStatus", {"streaming": True})
    assert not (cdp.registered or cdp.streaming)


def test_stream_confirmation_requires_own_socket_hello_and_sent_status():
    cdp = webxr._CDP(_Socket(), _ORIGIN)
    _socket_event(cdp, "webSocketCreated", url="wss://dgx.example:48322/oob/v1/ws")
    _frame(cdp, "Sent", "hello", {"clientId": "client"})
    _frame(cdp, "Received", "streamStatus", {"streaming": True})
    _frame(cdp, "Received", "hello", {"clientId": "other"}, "unrelated")
    assert not (cdp.registered or cdp.streaming)
    _frame(cdp, "Sent", "streamStatus", {"streaming": True})
    assert cdp.streaming and not cdp.registered
    _frame(cdp, "Received", "hello", {"clientId": "client"})
    assert cdp.registered and cdp.streaming
    _frame(cdp, "Sent", "streamStatus", {"streaming": "true"})
    assert not cdp.streaming
    _frame(cdp, "Sent", "streamStatus", {"streaming": True})
    _socket_event(cdp, "webSocketClosed")
    assert not (cdp.registered or cdp.streaming)
    _socket_event(
        cdp, "webSocketCreated", "replacement", url="wss://dgx.example:48322/oob/v1/ws"
    )
    _frame(cdp, "Received", "hello", {"clientId": "old"})
    _frame(cdp, "Sent", "streamStatus", {"streaming": True})
    assert not (cdp.registered or cdp.streaming)


def test_cdp_handles_events_while_waiting_for_its_response():
    socket = _Socket(
        [
            {
                "method": "Security.certificateError",
                "params": {"eventId": 3, "requestURL": _ORIGIN},
            },
            {"id": 2, "result": {}},
            {"id": 1, "result": {"ready": True}},
        ]
    )
    assert webxr._CDP(socket, _ORIGIN).call("Runtime.evaluate") == {"ready": True}
    assert socket.sent[1]["method"] == "Security.handleCertificateError"


@pytest.mark.parametrize("failure", [None, "page-connect", "body", "cancel"])
@pytest.mark.parametrize("owns_forward", [True, False])
def test_page_restarts_browser_and_cleans_up_only_owned_resources(
    monkeypatch, failure, owns_forward
):
    adb_calls, connections = [], []
    browser = _Socket()

    def adb(executable, serial, *args):
        assert executable == "/resolved/adb"
        assert serial == "quest"
        adb_calls.append(args)
        if args == ("shell", "cat", "/proc/net/unix"):
            return "@chrome_devtools_remote"
        if args == ("forward", "--no-rebind", "tcp:0", webxr._SOCKET):
            return "45678"
        if args == ("forward", "--list"):
            serial = "quest" if owns_forward else "someone-else"
            return f"{serial} tcp:45678 {webxr._SOCKET}\nquest tcp:9222 unrelated"
        return ""

    @contextmanager
    def connect(url, **kwargs):
        connections.append(url)
        if "/page/" in url and failure == "page-connect":
            raise RuntimeError("page connection failed")
        yield browser if "/browser/" in url else _Socket()

    monkeypatch.setattr(webxr, "_adb", adb)
    monkeypatch.setattr(
        webxr,
        "_json",
        lambda *_: {
            "webSocketDebuggerUrl": "ws://device:9222/devtools/browser/session"
        },
    )
    _websockets(monkeypatch, connect)
    headset = SimpleNamespace(check_alive=lambda: None)

    def visit():
        with webxr._page(
            "quest", cast(Any, headset), 1, _ORIGIN, "/resolved/adb"
        ) as cdp:
            assert cdp.server_origin == _ORIGIN
            if failure == "body":
                raise RuntimeError("navigation failed")
            if failure == "cancel":
                raise KeyboardInterrupt

    for _ in range(2):
        if failure:
            with pytest.raises(
                KeyboardInterrupt if failure == "cancel" else RuntimeError
            ):
                visit()
        else:
            visit()
    assert [entry["method"] for entry in browser.sent] == [
        "Target.createTarget",
        "Target.closeTarget",
    ] * 2
    assert browser.sent[0]["params"] == {"url": "about:blank"}
    assert browser.sent[1]["params"] == {"targetId": "owned-tab"}
    assert all(url.startswith("ws://127.0.0.1:45678/") for url in connections)
    assert (("forward", "--remove", "tcp:45678") in adb_calls) is owns_forward
    assert ("forward", "--no-rebind", "tcp:0", webxr._SOCKET) in adb_calls
    assert [args for args in adb_calls if args[:2] == ("shell", "am")] == [
        ("shell", "am", "force-stop", "com.oculus.browser"),
        ("shell", "am", "start", "-n", "com.oculus.browser/.OculusLauncherActivity"),
    ] * 2
    assert not any("--remove-all" in args or "clear" in args for args in adb_calls)


@pytest.mark.parametrize(
    "failure", ["cancel-before-stop", "force-stop", "cancel-after-stop"]
)
def test_page_does_not_launch_after_cancellation_or_browser_stop_failure(
    monkeypatch, failure
):
    adb_calls = []

    def check_alive():
        if failure == "cancel-before-stop" or (
            failure == "cancel-after-stop" and adb_calls
        ):
            raise RuntimeError("cancelled")

    def adb(executable, serial, *args):
        assert (executable, serial) == ("/resolved/adb", "quest")
        adb_calls.append(args)
        if failure == "force-stop":
            raise RuntimeError("browser stop failed")
        return ""

    monkeypatch.setattr(webxr, "_adb", adb)
    _websockets(monkeypatch, lambda *_, **__: pytest.fail("unexpected CDP connection"))
    with pytest.raises(RuntimeError, match="cancelled|browser stop failed"):
        with webxr._page(
            "quest",
            cast(Any, SimpleNamespace(check_alive=check_alive)),
            1,
            _ORIGIN,
            "/resolved/adb",
        ):
            pytest.fail("unexpected page")
    assert adb_calls == (
        []
        if failure == "cancel-before-stop"
        else [("shell", "am", "force-stop", "com.oculus.browser")]
    )


@pytest.mark.parametrize("accept_certificate", [False, True])
def test_button_text_does_not_confirm_streaming_and_override_is_reverted(
    monkeypatch, capsys, accept_certificate
):
    calls = []
    clock = iter(range(20))
    monkeypatch.setattr(webxr.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(webxr.time, "sleep", lambda _: None)

    def call(method, **params):
        calls.append((method, params))
        return {
            "result": {
                "value": {"ready": False, "connected": True, "button": "DISCONNECT"}
            }
        }

    args = SimpleNamespace(
        accept_server_certificate=accept_certificate, startup_timeout=2
    )
    cdp = SimpleNamespace(call=call, registered=False, streaming=False)
    with pytest.raises(TimeoutError, match="did not start streaming"):
        webxr._run_page(
            cast(Any, cdp),
            cast(Any, SimpleNamespace(check_alive=lambda: None)),
            cast(Any, args),
            "https://client/",
        )
    assert "streaming confirmed" not in capsys.readouterr().out
    security = [
        (method, params) for method, params in calls if method.startswith("Security.")
    ]
    assert security == (
        [
            ("Security.enable", {}),
            ("Security.setOverrideCertificateErrors", {"override": True}),
            ("Security.setOverrideCertificateErrors", {"override": False}),
        ]
        if accept_certificate
        else []
    )


@pytest.mark.parametrize(
    ("banner", "detail"),
    [
        (
            "CloudXR session stopped: Could not establish the signaling WebSocket "
            "connection. https://user:password@example.test/?turnCredential=secret "
            "(Error code: 0xC0F22202)",
            " (CloudXR signaling WebSocket connection failed); error code 0xC0F22202",
        ),
        ("Unknown failure: turnCredential=secret\n\x1b[31m", ""),
        ("Error code: 0x123456789abcdef-secret", ""),
        ("x" * 4096 + "Error code: 0xC0F22202", ""),
    ],
)
def test_page_errors_report_only_safe_summary_and_restore_certificate_override(
    banner, detail
):
    calls = []

    def call(method, **params):
        calls.append((method, params))
        return {"result": {"value": {"error": banner}}}

    with pytest.raises(RuntimeError) as error:
        webxr._run_page(
            cast(Any, SimpleNamespace(call=call)),
            cast(Any, SimpleNamespace(check_alive=lambda: None)),
            cast(
                Any, SimpleNamespace(accept_server_certificate=True, startup_timeout=10)
            ),
            "https://client/?turnCredential=secret",
        )
    assert str(error.value) == (
        f"Isaac Teleop reported a connection error{detail}; "
        "inspect its error banner in the headset"
    )
    assert calls[-1] == ("Security.setOverrideCertificateErrors", {"override": False})
    assert ".slice(0, 4096)" in webxr._PAGE_STATE


def test_disconnect_after_confirmed_stream_is_reported(monkeypatch, capsys):
    cdp = SimpleNamespace(registered=True, streaming=True)
    cdp.call = lambda *_, **__: {"result": {"value": {}}}
    monkeypatch.setattr(webxr.time, "sleep", lambda _: setattr(cdp, "streaming", False))
    args = SimpleNamespace(accept_server_certificate=False, startup_timeout=10)
    with pytest.raises(RuntimeError, match="disconnected"):
        webxr._run_page(
            cast(Any, cdp),
            cast(Any, SimpleNamespace(check_alive=lambda: None)),
            cast(Any, args),
            "https://client/",
        )
    assert "streaming confirmed" in capsys.readouterr().out


@pytest.mark.parametrize("release_owner", [False, True])
def test_browser_waits_for_owner_without_timing_out_and_remains_cancellable(
    monkeypatch, release_owner
):
    clock = [0.0]
    calls, statuses = [], []
    cdp = SimpleNamespace(registered=False, streaming=False)
    monkeypatch.setattr(webxr.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(
        webxr.time, "sleep", lambda delay: clock.__setitem__(0, clock[0] + delay)
    )

    def call(method, **params):
        calls.append((clock[0], method, params))
        if method == "Input.dispatchMouseEvent" and params["type"] == "mouseReleased":
            cdp.registered = cdp.streaming = True
        return {"result": {"value": {"ready": True, "x": 20, "y": 30}}}

    def check_alive():
        if "Streaming over USB" in statuses or not release_owner and clock[0] >= 6:
            raise KeyboardInterrupt
        assert clock[0] < 12, "browser startup must complete after ownership release"

    cdp.call = call
    with pytest.raises(KeyboardInterrupt):
        webxr._run_page(
            cast(Any, cdp),
            cast(Any, SimpleNamespace(check_alive=check_alive)),
            cast(
                Any, SimpleNamespace(accept_server_certificate=True, startup_timeout=4)
            ),
            "https://127.0.0.1:48322/client/",
            can_connect=lambda: release_owner and clock[0] >= 6,
            on_status=statuses.append,
        )
    assert "Waiting for the active client to disconnect" in statuses
    clicks = [
        (when, params)
        for when, method, params in calls
        if method == "Input.dispatchMouseEvent"
    ]
    if release_owner:
        assert len(clicks) == 2
        assert all(when >= 8 for when, _ in clicks)
        assert "Starting immersive VR; check the headset for Allow" in statuses
        assert statuses[-1] == "Streaming over USB"
    else:
        assert not clicks
        assert not cdp.streaming
    assert calls[-1][1:] == (
        "Security.setOverrideCertificateErrors",
        {"override": False},
    )


@pytest.mark.parametrize("first_available", range(6))
@pytest.mark.parametrize("unavailable", ["missing", "non-executable", "directory"])
def test_adb_discovery_preserves_launcher_precedence(
    monkeypatch, tmp_path, first_available, unavailable
):
    home = tmp_path / "home"
    candidates = [
        tmp_path / "explicit adb",
        tmp_path / "bin" / "adb",
        tmp_path / "SDK root" / "platform-tools" / "adb",
        tmp_path / "SDK home" / "platform-tools" / "adb",
        home / "Android" / "Sdk" / "platform-tools" / "adb",
    ]
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("QUEST_ADB", "explicit adb")
    monkeypatch.setenv("PATH", str(candidates[1].parent))
    monkeypatch.setenv("ANDROID_SDK_ROOT", str(candidates[2].parent.parent))
    monkeypatch.setenv("ANDROID_HOME", str(candidates[3].parent.parent))
    monkeypatch.setattr(webxr.Path, "home", lambda: home)
    for index, candidate in enumerate(candidates):
        candidate.parent.mkdir(parents=True, exist_ok=True)
        if index < first_available and unavailable == "missing":
            continue
        if index < first_available and unavailable == "directory":
            candidate.mkdir()
            continue
        candidate.write_text("not a real ADB executable\n")
        candidate.chmod(0o700 if index >= first_available else 0o600)

    if first_available == len(candidates):
        with pytest.raises(FileNotFoundError, match="ADB not found.*QUEST_ADB"):
            webxr._find_adb()
    else:
        assert webxr._find_adb() == str(candidates[first_available])


def test_sdk_root_only_adb_runs_device_checks_without_path_lookup(
    monkeypatch, tmp_path
):
    executable = tmp_path / "SDK root" / "platform-tools" / "adb"
    executable.parent.mkdir(parents=True)
    executable.write_text("not a real ADB executable\n")
    executable.chmod(0o700)
    monkeypatch.delenv("QUEST_ADB", raising=False)
    monkeypatch.setenv("PATH", str(tmp_path / "empty-bin"))
    monkeypatch.setenv("ANDROID_SDK_ROOT", str(executable.parent.parent))
    monkeypatch.delenv("ANDROID_HOME", raising=False)
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        assert command[0] == str(executable)
        output = {
            ("devices",): "List of devices attached\nquest\tdevice\n",
            ("-s", "quest", "get-state"): "device\n",
            (
                "-s",
                "quest",
                "shell",
                "pm path com.oculus.browser",
            ): "package:/browser\n",
        }[tuple(command[1:])]
        return subprocess.CompletedProcess(command, 0, output)

    monkeypatch.setattr(webxr.subprocess, "run", run)
    resolved = webxr._find_adb()
    assert webxr._device(None, resolved) == "quest"
    assert len(calls) == 3


@pytest.mark.parametrize("quest_count", [0, 1, 2])
def test_optional_device_discovery_only_selects_one_authorized_usb_quest(
    monkeypatch, quest_count
):
    calls = []
    devices = {
        "locked": ("unauthorized", "usb:1-1", True),
        "offline": ("offline", "usb:1-2", True),
        "network-quest": ("device", "tcp:192.0.2.10:5555", True),
        "phone": ("device", "usb:1-3", False),
        **{
            f"quest-{index}": ("device", f"usb:2-{index}", True)
            for index in range(quest_count)
        },
    }

    def adb(executable, serial, *args):
        assert executable == "/resolved/adb"
        calls.append((serial, args))
        if args == ("devices",):
            assert serial is None
            return "List of devices attached\n" + "\n".join(
                f"{name}\t{state}" for name, (state, _, _) in devices.items()
            )
        state, path, browser = devices[serial]
        assert state == "device"
        if args == ("get-devpath",):
            return path
        assert path.startswith("usb:")
        assert args == ("shell", "pm", "path", "com.oculus.browser")
        return "package:/browser" if browser else ""

    monkeypatch.setattr(webxr, "_adb", adb)
    serial, message = webxr._discover_device("/resolved/adb")
    assert serial == ("quest-0" if quest_count == 1 else None)
    assert message == (
        "No authorized USB headset detected"
        if quest_count == 0
        else "Detected"
        if quest_count == 1
        else "Multiple headsets; select one with --quest-device"
    )
    assert not any(serial in ("locked", "offline") for serial, _ in calls)


@pytest.mark.parametrize("failed_probe", ["path", "browser"])
@pytest.mark.parametrize("timed_out", [False, True])
def test_optional_device_discovery_skips_failed_candidates(
    monkeypatch, failed_probe, timed_out
):
    def adb(executable, serial, *args):
        if args == ("devices",):
            return "List of devices attached\nunplugged\tdevice\nquest\tdevice\n"
        failing_args = (
            ("get-devpath",)
            if failed_probe == "path"
            else ("shell", "pm", "path", "com.oculus.browser")
        )
        if serial == "unplugged" and args == failing_args:
            if timed_out:
                raise subprocess.TimeoutExpired("adb", 5)
            raise RuntimeError("ADB device disappeared")
        return "usb:1-1" if args == ("get-devpath",) else "package:/browser"

    monkeypatch.setattr(webxr, "_adb", adb)
    assert webxr._discover_device("/resolved/adb") == ("quest", "Detected")


def test_adb_shell_arguments_are_remote_quoted(monkeypatch):
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, " package:/browser \n")

    monkeypatch.setattr(webxr.subprocess, "run", run)
    url = "https://example/path?token=a&next='; echo injected"
    assert (
        webxr._adb("/tmp/Android SDK/adb", "quest", "shell", "am", "start", "-d", url)
        == "package:/browser"
    )
    command, kwargs = calls[0]
    assert command[:4] == ["/tmp/Android SDK/adb", "-s", "quest", "shell"]
    assert shlex.split(command[4]) == ["am", "start", "-d", url]
    assert len(command) == 5 and not kwargs.get("shell")
    assert kwargs["timeout"] == 5


@pytest.mark.parametrize(
    "overrides",
    [
        {"cloudxr_port": 0},
        {"cloudxr_port": 65536},
        {"startup_timeout": float("nan")},
        {"startup_timeout": float("inf")},
        {"startup_timeout": 0},
    ],
)
def test_invalid_urls_fail_before_headset_actions(overrides):
    options = dict(
        cloudxr_port=48322,
        startup_timeout=120,
    )
    with pytest.raises(ValueError):
        webxr._urls(cast(Any, SimpleNamespace(**(options | overrides))))


def test_usb_url_hosts_assets_and_signaling_on_one_origin_with_tcp_turn(monkeypatch):
    token = "a&host=other/?# token"
    monkeypatch.setenv("CONTROL_TOKEN", token)
    monkeypatch.setenv("FLASHDREAMS_CONTROLLER_TOKEN", "relay-test-token")
    origin, url = webxr._urls(
        cast(
            Any,
            SimpleNamespace(
                cloudxr_port=48323,
                startup_timeout=120,
                usb_turn_port=45678,
                usb_turn_credential="per-run-secret",
            ),
        )
    )
    assert origin == "https://127.0.0.1:48323"
    assert url.startswith(origin + "/client/?")
    assert parse_qs(urlsplit(url).query) == {
        "oobEnable": ["1"],
        "headless": ["true"],
        "panelHiddenAtStart": ["true"],
        "serverIP": ["127.0.0.1"],
        "port": ["48323"],
        "turnServer": ["turn:127.0.0.1:45678?transport=tcp"],
        "turnUsername": ["cloudxr"],
        "turnCredential": ["per-run-secret"],
        "iceRelayOnly": ["1"],
    }


@pytest.mark.parametrize(
    "arguments",
    [
        ["--usb-local"],
        ["--cloudxr-host", "192.168.1.2"],
        ["--web-client-base", "https://example.com/"],
    ],
)
def test_physical_quest_cli_rejects_retired_transport_options(arguments):
    parser = argparse.ArgumentParser()
    webxr.add_relay_arguments(parser)
    with pytest.raises(SystemExit) as error:
        parser.parse_args(arguments)
    assert error.value.code == 2


def test_relay_defaults_to_isolated_python_without_implicit_acceptance():
    parser = argparse.ArgumentParser()
    webxr.add_relay_arguments(parser)
    args = parser.parse_args([])
    assert args.isaac_teleop_python == str(
        Path.home() / ".local/share/flashdreams/isaac-teleop/bin/python"
    )
    assert args.isaac_teleop_accept_eula is False
    assert "".join(webxr.CLOUDXR_EULA_HELP.split()) in "".join(
        parser.format_help().split()
    )
