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

"""CPU checks for offline Quest USB transport and resource ownership."""

from __future__ import annotations

import subprocess
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
from flashdreams_controller_input import quest_usb_local as usb

pytestmark = pytest.mark.ci_cpu


@pytest.fixture
def rig(monkeypatch, tmp_path):
    events = []
    state = SimpleNamespace(
        device="usb:1-2",
        addresses="2: wlan0 inet 10.70.12.4/24 brd 10.70.12.255 scope global wlan0",
        rules={"tcp:2222": "tcp:2222"},
        fail_reverse=None,
        process=None,
        exit_status=None,
        stubborn=False,
        listening=True,
        log=b"",
        launch_error=None,
        config=None,
        directory=None,
        now=0.0,
    )

    def adb(*args):
        events.append(args)
        if args == ("get-devpath",):
            return state.device
        if args == ("shell", "ip", "-o", "-4", "addr", "show", "up"):
            return state.addresses
        if args[:2] == ("reverse", "--no-rebind"):
            source, destination = args[2:]
            if source in state.rules or source == state.fail_reverse:
                raise RuntimeError("reverse endpoint is already occupied")
            state.rules[source] = destination
            return ""
        if args == ("reverse", "--list"):
            return "\n".join(
                f"UsbFfs {source} {destination}"
                for source, destination in state.rules.items()
            )
        if args[:2] == ("reverse", "--remove"):
            del state.rules[args[2]]
            return ""
        pytest.fail(f"unexpected ADB command: {args}")

    class Process:
        def __init__(self, command, **kwargs):
            events.append(("launch", command))
            assert command[:2] == ["/resolved/turnserver", "-c"]
            assert len(command) == 3
            assert kwargs["stdin"] == subprocess.DEVNULL
            assert kwargs["stderr"] == subprocess.STDOUT
            state.directory = Path(command[2]).parent
            state.config = Path(command[2]).read_text()
            assert state.directory.stat().st_mode & 0o777 == 0o700
            assert Path(command[2]).stat().st_mode & 0o777 == 0o600
            kwargs["stdout"].write(state.log)
            kwargs["stdout"].flush()
            state.process = self
            if state.launch_error:
                raise state.launch_error

        def poll(self):
            return state.exit_status

        def terminate(self):
            events.append(("terminate",))
            if not state.stubborn:
                state.exit_status = 0

        def kill(self):
            events.append(("kill",))
            state.exit_status = -9

        def wait(self, *, timeout):
            events.append(("wait", timeout))
            if state.exit_status is None:
                raise subprocess.TimeoutExpired("turnserver", timeout)
            return state.exit_status

    class Reservation:
        def bind(self, address):
            assert address == ("127.0.0.1", 0)

        def getsockname(self):
            return ("127.0.0.1", 42000)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    @contextmanager
    def connect(address, *, timeout):
        assert address == ("127.0.0.1", 42000)
        assert timeout <= 0.1
        events.append(("connect",))
        if not state.listening:
            raise ConnectionRefusedError
        yield object()

    def sleep(seconds):
        state.now += seconds

    monkeypatch.setattr(usb.shutil, "which", lambda executable: "/resolved/turnserver")
    monkeypatch.setattr(usb.subprocess, "Popen", Process)
    monkeypatch.setattr(usb.socket, "socket", lambda *args: Reservation())
    monkeypatch.setattr(usb.socket, "create_connection", connect)
    monkeypatch.setattr(usb.time, "monotonic", lambda: state.now)
    monkeypatch.setattr(usb.time, "sleep", sleep)
    monkeypatch.setattr(usb.tempfile, "tempdir", str(tmp_path))
    transport = usb.USBLocalTransport(adb, proxy_port=48322)
    yield transport, state, events
    transport.stop()


def test_offline_start_uses_private_config_and_cleans_only_owned_resources(rig):
    transport, state, events = rig
    transport.start(lambda: None)
    assert transport.turn_port == 42000
    assert len(transport.credential) >= 32
    config = state.config.splitlines()
    for option in (
        "listening-port=42000",
        "listening-ip=127.0.0.1",
        "relay-ip=127.0.0.1",
        "external-ip=127.0.0.1",
        "lt-cred-mech",
        f"user=cloudxr:{transport.credential}",
        "realm=cloudxr",
        "fingerprint",
        "allow-loopback-peers",
        "no-cli",
        "no-tls",
        "no-dtls",
        "no-udp",
        "no-tcp-relay",
        "simple-log",
        "log-file=stdout",
        f"pidfile={state.directory / 'turnserver.pid'}",
    ):
        assert option in config
    assert "no-tcp" not in config
    assert "no-udp-relay" not in config
    assert not any("relay-threads" in option for option in config)
    assert state.rules == {
        f"tcp:{port}": f"tcp:{port}" for port in (2222, 48322, 49100, 42000)
    }
    assert events.index(("connect",)) < events.index(
        ("reverse", "--no-rebind", "tcp:48322", "tcp:48322")
    )
    transport.start(lambda: None)
    transport.stop()
    transport.stop()
    assert state.rules == {"tcp:2222": "tcp:2222"}
    assert events.count(("terminate",)) == 1
    assert not state.directory.exists()
    assert not any(
        "--remove-all" in event or "kill-server" in event for event in events
    )
    with pytest.raises(RuntimeError, match="stopped"):
        transport.start(lambda: None)


@pytest.mark.parametrize("device", ["192.168.1.5:5555", "local", "unknown", ""])
def test_network_adb_is_rejected_before_launch(rig, device):
    transport, state, events = rig
    state.device = device
    with pytest.raises(RuntimeError, match="USB-connected"):
        transport.start(lambda: None)
    assert state.process is None
    assert state.rules == {"tcp:2222": "tcp:2222"}


@pytest.mark.parametrize(
    "addresses",
    [
        "",
        "1: lo inet 127.0.0.1/8 scope host lo",
        "1: lo inet 127.2.3.4/8 scope host lo",
        "2: wlan0 inet invalid scope global wlan0",
        "2: wlan0 inet 0.0.0.0/24 scope global wlan0",
    ],
)
def test_headset_needs_nonloopback_ipv4_before_launch(rig, addresses):
    transport, state, events = rig
    state.addresses = addresses
    with pytest.raises(RuntimeError, match="IPv4 address on the Quest"):
        transport.start(lambda: None)
    assert state.process is None


def test_missing_coturn_fails_without_mutation(rig, monkeypatch):
    transport, state, events = rig
    monkeypatch.setattr(usb.shutil, "which", lambda name: None)
    with pytest.raises(RuntimeError, match="install turnserver"):
        transport.start(lambda: None)
    assert state.process is None
    assert state.rules == {"tcp:2222": "tcp:2222"}


@pytest.mark.parametrize("occupied", [48322, 49100, 42000])
def test_existing_reverse_is_preserved_and_partial_setup_is_unwound(rig, occupied):
    transport, state, events = rig
    endpoint = f"tcp:{occupied}"
    state.rules[endpoint] = "tcp:59999"
    with pytest.raises(RuntimeError, match="occupied"):
        transport.start(lambda: None)
    assert state.rules == {"tcp:2222": "tcp:2222", endpoint: "tcp:59999"}
    assert ("reverse", "--remove", endpoint) not in events
    assert events.count(("terminate",)) == 1
    assert not state.directory.exists()


def test_replaced_rule_is_preserved_and_stubborn_owned_process_is_killed(rig):
    transport, state, events = rig
    transport.start(lambda: None)
    state.rules["tcp:48322"] = "tcp:60000"
    state.stubborn = True
    transport.stop()
    assert state.rules == {"tcp:2222": "tcp:2222", "tcp:48322": "tcp:60000"}
    assert ("reverse", "--remove", "tcp:48322") not in events
    assert events.count(("kill",)) == 1
    assert events.count(("wait", 3)) == 2


@pytest.mark.parametrize("stage", ["before-start", "listener", "reverses"])
def test_cancellation_unwinds_owned_resources(rig, stage):
    transport, state, events = rig
    state.listening = stage != "listener"

    def cancelled():
        if (
            stage == "before-start"
            or (stage == "listener" and state.process is not None)
            or (stage == "reverses" and "tcp:48322" in state.rules)
        ):
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        transport.start(cancelled)
    assert state.rules == {"tcp:2222": "tcp:2222"}
    if state.process is not None:
        assert state.exit_status == 0
        assert not state.directory.exists()


@pytest.mark.parametrize("failure", ["exit", "timeout", "launch"])
def test_start_failure_reports_cause_and_redacted_log_beginning_and_tail(rig, failure):
    transport, state, events = rig
    state.log = (
        f"FIRST: unknown option user=cloudxr:{transport.credential}\n".encode()
        + b"help text\n" * 1500
        + f"LAST: process ended {transport.credential}\n".encode()
    )
    if failure == "exit":
        state.exit_status = 1
        match = "exited with status 1"
    elif failure == "timeout":
        state.listening = False
        match = "did not become ready"
    else:
        state.launch_error = OSError("executable cannot start")
        match = "executable cannot start"
    with pytest.raises((RuntimeError, OSError), match=match) as error:
        transport.start(lambda: None)
    message = str(error.value)
    assert transport.credential not in message
    assert "FIRST: unknown option" in message
    assert "LAST: process ended" in message
    assert "user=cloudxr:<redacted>" in message
    assert len(message) < 10000
    assert not state.directory.exists()


def test_log_truncation_cannot_expose_a_partial_turn_credential(rig):
    transport, state, events = rig
    state.exit_status = 1
    credential = transport.credential.encode()
    state.log = b"x" * 4080 + credential + b"x" * 10000 + credential + b"x" * 4080
    with pytest.raises(RuntimeError) as error:
        transport.start(lambda: None)
    assert transport.credential[:16] not in str(error.value)
    assert transport.credential[-16:] not in str(error.value)


def test_running_turn_failure_is_observable(rig):
    transport, state, events = rig
    transport.start(lambda: None)
    state.exit_status = 2
    with pytest.raises(RuntimeError, match="exited with status 2"):
        transport.check_alive()


def test_cleanup_warning_preserves_unreadable_rules_and_stops_process(rig, capsys):
    transport, state, events = rig
    transport.start(lambda: None)
    original = transport._adb

    def failed_list(*args):
        if args == ("reverse", "--list"):
            raise RuntimeError("device disappeared")
        return original(*args)

    transport._adb = failed_list
    transport.stop()
    assert len(state.rules) == 4
    assert "could not remove owned reverse" in capsys.readouterr().out
    assert events.count(("terminate",)) == 1


@pytest.mark.parametrize("missing", ["index.html", "bundle.js", "bundle.emulator.js"])
@pytest.mark.parametrize("empty", [False, True])
def test_cached_client_requires_complete_nonempty_assets(
    tmp_path, monkeypatch, missing, empty
):
    monkeypatch.setenv("TELEOP_WEB_CLIENT_STATIC_DIR", str(tmp_path))
    for name in ("index.html", "bundle.js", "bundle.emulator.js"):
        if name != missing:
            (tmp_path / name).write_text("cached asset")
        elif empty:
            (tmp_path / name).touch()
    with pytest.raises(RuntimeError, match="Preinstall") as error:
        usb.cached_web_client_dir()
    assert missing in str(error.value)
    assert str(tmp_path) in str(error.value)


@pytest.mark.parametrize("override", [None, "", "  ", "~/explicit-client"])
def test_cached_client_resolves_default_or_expanded_override(
    tmp_path, monkeypatch, override
):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr(
        usb.os.path, "expanduser", lambda value: value.replace("~", str(tmp_path), 1)
    )
    if override is None:
        monkeypatch.delenv("TELEOP_WEB_CLIENT_STATIC_DIR", raising=False)
        directory = tmp_path / ".cloudxr" / "static-client"
    else:
        monkeypatch.setenv("TELEOP_WEB_CLIENT_STATIC_DIR", override)
        directory = (
            tmp_path / "explicit-client"
            if override.strip()
            else tmp_path / ".cloudxr" / "static-client"
        )
    directory.mkdir(parents=True)
    for name in ("index.html", "bundle.js", "bundle.emulator.js"):
        (directory / name).write_text("cached asset")
    assert usb.cached_web_client_dir() == directory
