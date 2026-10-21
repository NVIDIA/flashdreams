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

"""Shared server input wiring without CloudXR, a model, or a headset."""

import argparse
import os
import threading
from importlib.metadata import EntryPoint
from pathlib import Path
from types import SimpleNamespace

import pytest
import tomllib
from flashdreams_controller_input import (
    credentials,
    quest_usb_local,
    quest_webxr,
    remote_quest,
)
from flashdreams_controller_input import quest_remote_v2_window as window
from flashdreams_controller_input.frames import SOURCE_CLIENT_CAPTURE_NS_KEY
from flashdreams_controller_input.isaac_teleop_v2_window import ISAAC_TELEOP_WEBRTC_MODE

from flashdreams.runtime_v2 import client_window_factory

pytestmark = pytest.mark.ci_cpu


@pytest.fixture(autouse=True)
def isolated_headset_setup(monkeypatch, tmp_path):
    monkeypatch.delenv("TELEOP_STREAM_SERVER_IP", raising=False)
    monkeypatch.setattr(
        credentials,
        "server_credentials",
        lambda _: pytest.fail("Unexpected credential generation"),
    )
    monkeypatch.setattr(quest_usb_local, "cached_web_client_dir", lambda: tmp_path)
    monkeypatch.setattr(
        quest_webxr,
        "QuestWebXRProducer",
        lambda *_, **__: SimpleNamespace(
            status_message="Detecting headset", restart=lambda: None
        ),
    )


@pytest.fixture(
    params=[window.QUEST_REMOTE_WEBRTC_MODE, window.QUEST_REMOTE_NATIVE_WINDOW_MODE],
    ids=["webrtc", "native-window"],
)
def mode(request, monkeypatch):
    selected = request.param

    def output_mode(name, **_):
        assert name == selected.output_mode
        return SimpleNamespace(create=lambda _: object())

    monkeypatch.setattr(window, "client_window_mode", output_mode)
    return selected


def _arguments(tmp_path, *options):
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8081)
    window.QUEST_REMOTE_WEBRTC_MODE.add_arguments(parser)
    ISAAC_TELEOP_WEBRTC_MODE.add_arguments(parser)
    cert, key = tmp_path / "server.crt", tmp_path / "server.key"
    cert.touch()
    key.touch()
    return parser.parse_args(
        [
            "--isaac-teleop-accept-eula",
            "--controller-host",
            "10.111.69.99",
            "--controller-cert",
            str(cert),
            "--controller-key",
            str(key),
            *options,
        ]
    )


@pytest.mark.parametrize(
    "options,enabled",
    [([], True), (["--local-cloudxr"], True), (["--no-local-cloudxr"], False)],
)
def test_local_cloudxr_is_enabled_unless_explicitly_disabled(
    tmp_path, options, enabled
):
    assert _arguments(tmp_path, *options).local_cloudxr is enabled


def test_relay_only_window_starts_remote_input_without_local_runtime(
    tmp_path, monkeypatch, mode
):
    args = _arguments(tmp_path, "--no-local-cloudxr")
    args.isaac_teleop_accept_eula = False
    args.isaac_teleop_python = ""
    args.isaac_teleop_install_dir = ""
    args.isaac_teleop_source_rate_hz = -1.0
    args.quest_device = ""
    args.quest_cloudxr_port = 0
    monkeypatch.setenv("FLASHDREAMS_CONTROLLER_TOKEN", "x" * 48)
    monkeypatch.setenv("TELEOP_STREAM_SERVER_IP", "0.0.0.0")

    def unexpected_local_setup(*_, **__):
        pytest.fail("relay-only input must not initialize local CloudXR or Quest")

    monkeypatch.setattr(window, "IsaacTeleopSource", unexpected_local_setup)
    monkeypatch.setattr(quest_webxr, "QuestWebXRProducer", unexpected_local_setup)
    monkeypatch.setattr(quest_webxr, "_validate_options", unexpected_local_setup)
    monkeypatch.setattr(
        quest_usb_local, "cached_web_client_dir", unexpected_local_setup
    )
    monkeypatch.setattr(window, "prepare_web_client", unexpected_local_setup)
    started, stopped, output_closed = (
        threading.Event(),
        threading.Event(),
        threading.Event(),
    )
    remote = SimpleNamespace(start=started.set, poll=lambda: None, stop=stopped.set)
    received = {}

    def remote_source(**kwargs):
        received.update(kwargs)
        return remote

    output = SimpleNamespace(close=output_closed.set)

    def output_mode(name, **_):
        assert name == mode.output_mode
        return SimpleNamespace(create=lambda _: output)

    monkeypatch.setattr(remote_quest, "RemoteQuestSource", remote_source)
    monkeypatch.setattr(window, "client_window_mode", output_mode)
    wrapped = mode.create(args)
    try:
        assert started.wait(timeout=1.0)
        assert wrapped._source is remote
        assert wrapped.output_window is output
        assert received["certfile"] == args.controller_cert
        assert received["keyfile"] == args.controller_key
        assert received["token"] == "x" * 48
        assert received["wrist_offset_m"] == args.isaac_teleop_wrist_offset_m
    finally:
        wrapped.close()
    assert stopped.is_set()
    assert output_closed.is_set()
    assert not wrapped._source_thread.is_alive()


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("controller_host", "", "controller-host.*non-empty"),
        ("controller_host", "0.0.0.0", "controller-host.*concrete IPv4"),
        ("controller_port", 0, "controller-port"),
        ("controller_port", 65536, "controller-port"),
        ("controller_cert", Path("/missing/controller.crt"), "existing file"),
        ("controller_key", Path("/missing/controller.key"), "existing file"),
        ("isaac_teleop_wrist_offset_m", -0.1, "wrist-offset-m"),
        ("isaac_teleop_wrist_offset_m", float("inf"), "wrist-offset-m"),
        ("isaac_teleop_wrist_offset_m", float("nan"), "wrist-offset-m"),
    ],
)
def test_relay_only_window_validates_remote_options_before_source_creation(
    tmp_path, monkeypatch, mode, field, value, match
):
    args = _arguments(tmp_path, "--no-local-cloudxr")
    args.isaac_teleop_accept_eula = False
    setattr(args, field, value)
    monkeypatch.setenv("FLASHDREAMS_CONTROLLER_TOKEN", "x" * 48)
    monkeypatch.setattr(
        remote_quest,
        "RemoteQuestSource",
        lambda **_: pytest.fail("must validate before creating a source"),
    )
    with pytest.raises(ValueError, match=match):
        mode.create(args)


@pytest.mark.parametrize("token", ["", "x" * 31, "x" * 257])
def test_relay_only_window_rejects_invalid_authentication_token(
    tmp_path, monkeypatch, mode, token
):
    args = _arguments(tmp_path, "--no-local-cloudxr")
    args.isaac_teleop_accept_eula = False
    monkeypatch.setenv("FLASHDREAMS_CONTROLLER_TOKEN", token)
    with pytest.raises(ValueError, match="FLASHDREAMS_CONTROLLER_TOKEN"):
        mode.check_arguments(args)


@pytest.mark.parametrize("display", ["webrtc", "native-window"])
def test_registered_modes_select_the_builtin_display(tmp_path, monkeypatch, display):
    project = Path(__file__).resolve().parents[1] / "pyproject.toml"
    config = tomllib.loads(project.read_text())
    group = client_window_factory.CLIENT_WINDOW_ENTRY_POINT_GROUP
    entries = [
        EntryPoint(name=name, value=value, group=group)
        for name, value in config["project"]["entry-points"][group].items()
    ]
    monkeypatch.setattr(client_window_factory, "entry_points", lambda **_: entries)
    parser = argparse.ArgumentParser()
    client_window_factory.add_client_window_arguments(parser)
    cert, key = tmp_path / "server.crt", tmp_path / "server.key"
    cert.touch()
    key.touch()
    args = parser.parse_args(
        [
            "--mode",
            f"quest-remote-{display}",
            "--window-title",
            "HandDreams",
            "--controller-cert",
            str(cert),
            "--controller-key",
            str(key),
            "--isaac-teleop-accept-eula",
        ]
    )
    monkeypatch.setenv("FLASHDREAMS_CONTROLLER_TOKEN", "x" * 48)
    output = SimpleNamespace(server=SimpleNamespace(url="http://127.0.0.1:8081"))
    created = []

    for name in ("webrtc", "native-window"):
        builtin = client_window_factory.client_window_mode(
            name, modes=args._client_window_modes
        )

        def create(parsed, selected=name):
            created.append(selected)
            assert parsed.window_title == "HandDreams"
            return output

        monkeypatch.setattr(builtin, "create", create)
    monkeypatch.setattr(remote_quest, "RemoteQuestSource", lambda **_: object())
    monkeypatch.setattr(window, "IsaacTeleopSource", lambda **_: object())
    monkeypatch.setattr(
        window, "ControllerSourceWindow", lambda **kwargs: SimpleNamespace(**kwargs)
    )
    wrapped = client_window_factory.create_client_window(args)
    assert isinstance(wrapped, SimpleNamespace)
    assert wrapped.output_window is output
    assert created == [display]
    mode = client_window_factory.client_window_mode(
        args.mode, modes=args._client_window_modes
    )
    message = mode.starting(wrapped)
    assert message == (
        "Open http://127.0.0.1:8081 in a browser." if display == "webrtc" else None
    )


@pytest.mark.parametrize("quest_device", [None, "quest-serial"])
def test_shared_window_requires_explicit_eula_and_transport_credentials(
    tmp_path, monkeypatch, quest_device
):
    args = _arguments(tmp_path)
    args.quest_device = quest_device
    monkeypatch.setenv("FLASHDREAMS_CONTROLLER_TOKEN", "x" * 48)
    args.isaac_teleop_accept_eula = False
    with pytest.raises(ValueError, match="accept-eula"):
        window.QUEST_REMOTE_WEBRTC_MODE.check_arguments(args)
    args.isaac_teleop_accept_eula = True
    window.QUEST_REMOTE_WEBRTC_MODE.check_arguments(args)
    monkeypatch.delenv("FLASHDREAMS_CONTROLLER_TOKEN")
    with pytest.raises(ValueError, match="TOKEN"):
        window.QUEST_REMOTE_WEBRTC_MODE.check_arguments(args)


@pytest.mark.parametrize("host", ["127.0.0.1", "10.111.69.99", "dgx.example"])
@pytest.mark.parametrize("local_cloudxr", [True, False])
def test_default_credentials_are_prepared_only_when_creating_a_window(
    tmp_path, monkeypatch, mode, host, local_cloudxr
):
    args = _arguments(tmp_path)
    args.local_cloudxr = local_cloudxr
    args.controller_cert = args.controller_key = None
    args.controller_host = host
    monkeypatch.delenv("FLASHDREAMS_CONTROLLER_TOKEN", raising=False)
    address = "10.111.69.99" if host == "dgx.example" else host
    monkeypatch.setattr(credentials.socket, "gethostbyname", lambda _: address)
    calls = []

    def prepare(selected):
        calls.append(selected)
        return tmp_path

    monkeypatch.setattr(credentials, "server_credentials", prepare)
    monkeypatch.setattr(credentials, "read_token", lambda _: "x" * 48)
    monkeypatch.setattr(remote_quest, "RemoteQuestSource", lambda **kw: kw)
    monkeypatch.setattr(window, "IsaacTeleopSource", lambda **_: object())
    monkeypatch.setattr(window, "ControllerSourceWindow", lambda **kw: kw["source"])
    mode.check_arguments(args)
    assert calls == []

    source = mode.create(args)
    assert calls == [host]
    remote = source._remote if local_cloudxr else source
    assert remote["certfile"] == tmp_path / "server.crt"
    assert remote["keyfile"] == tmp_path / "server.key"
    assert remote["token"] == "x" * 48
    assert args.controller_cert is args.controller_key is None
    assert "FLASHDREAMS_CONTROLLER_TOKEN" not in os.environ


@pytest.mark.parametrize(
    "cert,key,token",
    [
        (True, False, False),
        (False, True, False),
        (False, False, True),
        (True, True, False),
        (True, False, True),
        (False, True, True),
    ],
)
@pytest.mark.parametrize("local_cloudxr", [True, False])
def test_partial_credentials_fail_before_managed_bootstrap(
    tmp_path, monkeypatch, mode, cert, key, token, local_cloudxr
):
    args = _arguments(tmp_path)
    args.local_cloudxr = local_cloudxr
    if not cert:
        args.controller_cert = None
    if not key:
        args.controller_key = None
    if token:
        monkeypatch.setenv("FLASHDREAMS_CONTROLLER_TOKEN", "x" * 48)
    else:
        monkeypatch.delenv("FLASHDREAMS_CONTROLLER_TOKEN", raising=False)
    with pytest.raises(ValueError, match="together, or omit all three"):
        mode.create(args)


def test_controller_help_is_side_effect_free(capsys):
    parser = argparse.ArgumentParser()
    window.QUEST_REMOTE_WEBRTC_MODE.add_arguments(parser)
    ISAAC_TELEOP_WEBRTC_MODE.add_arguments(parser)
    with pytest.raises(SystemExit) as error:
        parser.parse_args(["--help"])
    assert error.value.code == 0
    assert "private managed server identity" in " ".join(
        capsys.readouterr().out.split()
    )


@pytest.mark.parametrize("accept_eula", [False, True])
@pytest.mark.parametrize("address", ["10.111.69.99", "127.0.0.1"])
def test_shared_window_reuses_both_sources_and_existing_timing(
    tmp_path, monkeypatch, accept_eula, address, mode
):
    args = _arguments(tmp_path)
    args.controller_host = address
    args.isaac_teleop_accept_eula = accept_eula
    args.isaac_teleop_source_rate_hz = 110.0
    args.isaac_teleop_tracking_hz = 80.0
    args.isaac_teleop_wrist_offset_m = 0.04
    args.isaac_teleop_python = " /test/python "
    args.isaac_teleop_install_dir = " /test/cloudxr "
    args.isaac_teleop_device_profile = " Quest3 "
    monkeypatch.setenv("FLASHDREAMS_CONTROLLER_TOKEN", "x" * 48)
    received = {}
    remote, local = object(), object()

    def remote_source(**kwargs):
        received["remote"] = kwargs
        return remote

    def local_source(**kwargs):
        received["local"] = kwargs
        return local

    def create_window(**kwargs):
        received["window"] = kwargs
        return kwargs["source"]

    monkeypatch.setattr(remote_quest, "RemoteQuestSource", remote_source)
    monkeypatch.setattr(window, "IsaacTeleopSource", local_source)
    monkeypatch.setattr(window, "ControllerSourceWindow", create_window)
    preparation = []
    monkeypatch.setattr(
        window, "prepare_web_client", lambda **kwargs: preparation.append(kwargs)
    )
    if not accept_eula:
        with pytest.raises(ValueError, match="accept-eula"):
            mode.create(args)
        assert received == {}
        assert preparation == []
        return
    source = mode.create(args)
    assert preparation == [{"python": " /test/python "}]
    assert source is received["window"]["source"]
    assert received["window"]["client_capture_keys"] == (
        SOURCE_CLIENT_CAPTURE_NS_KEY,
        "client_capture_ns",
    )
    assert received["remote"]["host"] == address
    assert received["remote"]["wrist_offset_m"] == 0.04
    assert received["local"]["wrist_offset_m"] == 0.04
    assert received["local"]["accept_eula"] is accept_eula
    assert received["local"]["wait_for_controllers"] is False
    assert received["local"]["rate_hz"] == 110.0
    assert received["local"]["tracking_rate_hz"] == 80.0
    assert received["local"]["python_executable"] == "/test/python"
    assert received["local"]["install_dir"] == "/test/cloudxr"
    assert received["local"]["device_profile"] == "Quest3"
    assert received["local"]["startup_timeout_s"] is None
    assert received["local"]["cloudxr_host"] == address
    assert received["local"]["cloudxr_proxy_bind"] == address
    assert received["local"]["host_client"] is True
    assert received["local"]["offline_web_client"] is True
    assert received["local"]["oob_hub"] is True
    assert received["window"]["status_provider"] == source.connection_status


def test_client_preparation_failure_stops_before_sources_start(
    tmp_path, monkeypatch, mode
):
    args = _arguments(tmp_path)
    monkeypatch.setenv("FLASHDREAMS_CONTROLLER_TOKEN", "x" * 48)

    def preparation_failed(**_):
        raise ValueError("Could not prepare the local Isaac Teleop Web Client")

    def unexpected_source(**_):
        pytest.fail("No controller source should be created without client assets")

    monkeypatch.setattr(window, "prepare_web_client", preparation_failed)
    monkeypatch.setattr(remote_quest, "RemoteQuestSource", unexpected_source)
    monkeypatch.setattr(window, "IsaacTeleopSource", unexpected_source)
    with pytest.raises(ValueError, match="Could not prepare"):
        mode.create(args)


@pytest.mark.parametrize("physical_wake", [True, False])
@pytest.mark.parametrize("accept_certificate", [True, False])
@pytest.mark.parametrize("accept_eula", [False, True])
def test_attached_quest_wraps_the_single_configured_local_source(
    tmp_path, monkeypatch, physical_wake, accept_certificate, accept_eula, mode
):
    from flashdreams_controller_input import quest_webxr

    args = _arguments(tmp_path)
    args.isaac_teleop_accept_eula = accept_eula
    args.quest_device = "quest-serial"
    args.quest_cloudxr_port = 48323
    args.quest_require_physical_wake = physical_wake
    args.quest_accept_server_certificate = accept_certificate
    monkeypatch.setenv("FLASHDREAMS_CONTROLLER_TOKEN", "x" * 48)
    local, remote = object(), object()
    managed = SimpleNamespace(restart=lambda: None)
    calls = []

    def local_source(**kwargs):
        calls.append(("source", kwargs))
        return local

    def producer(options, **kwargs):
        calls.append(("managed", options, kwargs))
        return managed

    monkeypatch.setattr(remote_quest, "RemoteQuestSource", lambda **_: remote)
    monkeypatch.setattr(window, "IsaacTeleopSource", local_source)
    monkeypatch.setattr(quest_webxr, "QuestWebXRProducer", producer)
    monkeypatch.setattr(
        window, "ControllerSourceWindow", lambda **kwargs: kwargs["source"]
    )
    if not accept_eula:
        with pytest.raises(ValueError, match="accept-eula"):
            mode.create(args)
        assert calls == []
        return
    unified = mode.create(args)
    assert len(calls) == 2
    assert calls[0][0] == "source"
    assert calls[0][1]["oob_hub"] is True
    assert calls[0][1]["cloudxr_host"] == "10.111.69.99"
    assert calls[0][1]["cloudxr_proxy_bind"] == "10.111.69.99"
    assert calls[0][1]["host_client"] is True
    assert calls[0][1]["offline_web_client"] is True
    assert calls[0][1]["accept_eula"] is accept_eula
    assert calls[0][1]["cloudxr_port"] == 48323
    _, options, producer_args = calls[1]
    assert producer_args["source"] is local
    assert producer_args["shared_runtime"] is True
    assert producer_args["hub_host"] == "10.111.69.99"
    assert producer_args["can_connect"]()
    unified._owner = 1
    assert not producer_args["can_connect"]()
    assert options.quest_device == "quest-serial"
    assert options.require_physical_wake is physical_wake
    assert options.accept_server_certificate is accept_certificate
    assert options.isaac_teleop_accept_eula is accept_eula
    assert options.cloudxr_port == 48323
    assert (
        not {"cloudxr_host", "usb_local", "web_client_base", "server", "port"}
        & vars(options).keys()
    )
    assert unified._local is local and unified._remote is remote
    assert unified._headset is managed


@pytest.mark.parametrize(
    "options", [["--quest-usb-local"], ["--quest-cloudxr-host", "10.111.69.99"]]
)
def test_attached_quest_rejects_obsolete_transport_flags(options, capsys):
    parser = argparse.ArgumentParser()
    window.QUEST_REMOTE_WEBRTC_MODE.add_arguments(parser)
    with pytest.raises(SystemExit) as error:
        parser.parse_args(["--quest-device", "quest-serial", *options])
    assert error.value.code == 2
    assert "unrecognized arguments" in capsys.readouterr().err


@pytest.mark.parametrize("keep_open", [False, True])
def test_managed_browser_failure_hold_is_opt_in(keep_open):
    parser = argparse.ArgumentParser()
    window.QUEST_REMOTE_WEBRTC_MODE.add_arguments(parser)
    ISAAC_TELEOP_WEBRTC_MODE.add_arguments(parser)
    args = parser.parse_args(
        ["--quest-keep-browser-open-on-error"] if keep_open else []
    )
    assert (
        window.QUEST_REMOTE_WEBRTC_MODE._quest_options(args).keep_browser_open_on_error
        is keep_open
    )


@pytest.mark.parametrize(
    "options,match",
    [
        ({"quest_device": ""}, "non-empty"),
        ({"quest_device": "   "}, "non-empty"),
        ({"quest_device": "quest", "quest_cloudxr_port": 0}, "port"),
        ({"quest_device": "quest", "quest_cloudxr_port": 65536}, "port"),
    ],
)
def test_invalid_attached_quest_options_fail_before_source_creation(
    tmp_path, monkeypatch, options, match
):
    args = _arguments(tmp_path)
    monkeypatch.setenv("FLASHDREAMS_CONTROLLER_TOKEN", "x" * 48)
    monkeypatch.setattr(
        window, "IsaacTeleopSource", lambda **_: pytest.fail("must validate first")
    )
    for key, value in options.items():
        setattr(args, key, value)
    with pytest.raises(ValueError, match=match):
        window.QUEST_REMOTE_WEBRTC_MODE.create(args)


@pytest.mark.parametrize(
    "options",
    [
        {"quest_accept_server_certificate": True},
        {"quest_require_physical_wake": False},
        {"quest_cloudxr_port": 48323},
    ],
)
def test_autodiscovery_accepts_headset_options_without_an_explicit_device(
    tmp_path, monkeypatch, options, mode
):
    args = _arguments(tmp_path)
    monkeypatch.setenv("FLASHDREAMS_CONTROLLER_TOKEN", "x" * 48)
    for key, value in options.items():
        setattr(args, key, value)
    mode.check_arguments(args)
    assert mode._quest_options(args).quest_device is None


@pytest.mark.parametrize("advertised", [None, "0.0.0.0", "10.111.69.99"])
@pytest.mark.parametrize("bind_host", ["0.0.0.0", "0", "wildcard.example"])
def test_wildcard_bind_is_rejected_even_with_a_reachable_advertised_host(
    tmp_path, monkeypatch, advertised, bind_host, mode
):
    args = _arguments(tmp_path)
    args.controller_host = bind_host
    monkeypatch.setenv("FLASHDREAMS_CONTROLLER_TOKEN", "x" * 48)
    if advertised is not None:
        monkeypatch.setenv("TELEOP_STREAM_SERVER_IP", advertised)
    monkeypatch.setattr(credentials.socket, "gethostbyname", lambda _: "0.0.0.0")
    monkeypatch.setattr(
        remote_quest,
        "RemoteQuestSource",
        lambda **_: pytest.fail("must reject before source construction"),
    )
    with pytest.raises(ValueError, match="--controller-host.*concrete IPv4"):
        mode.create(args)


def test_wildcard_advertised_host_is_rejected(tmp_path, monkeypatch, mode):
    args = _arguments(tmp_path)
    monkeypatch.setenv("FLASHDREAMS_CONTROLLER_TOKEN", "x" * 48)
    monkeypatch.setenv("TELEOP_STREAM_SERVER_IP", "0.0.0.0")
    with pytest.raises(ValueError, match="TELEOP_STREAM_SERVER_IP.*reachable"):
        mode.create(args)


@pytest.mark.parametrize(
    "bind_host,override,resolved,advertised",
    [
        ("10.111.69.99", "dgx.example", "10.111.69.99", "dgx.example"),
        ("dgx.example", None, "10.111.69.99", "dgx.example"),
    ],
)
def test_cloudxr_bind_address_is_separate_from_advertised_address(
    tmp_path,
    monkeypatch,
    mode,
    bind_host,
    override,
    resolved,
    advertised,
):
    args = _arguments(tmp_path)
    args.controller_host = bind_host
    monkeypatch.setenv("FLASHDREAMS_CONTROLLER_TOKEN", "x" * 48)
    if override is not None:
        monkeypatch.setenv("TELEOP_STREAM_SERVER_IP", override)
    received = {}

    def resolve(host):
        assert host == bind_host
        return resolved

    def local_source(**kwargs):
        received.update(kwargs)
        return SimpleNamespace(ready_for_clients=True)

    monkeypatch.setattr(credentials.socket, "gethostbyname", resolve)
    monkeypatch.setattr(remote_quest, "RemoteQuestSource", lambda **_: object())
    monkeypatch.setattr(window, "IsaacTeleopSource", local_source)
    monkeypatch.setattr(
        window, "ControllerSourceWindow", lambda **kwargs: kwargs["source"]
    )
    source = mode.create(args)
    assert received["cloudxr_host"] == advertised
    assert received["cloudxr_proxy_bind"] == resolved
    status = source.connection_status()
    expected_client = f"https://{advertised}:48322/client/"
    assert status.web_client_url.startswith(expected_client + "?")
    assert f"serverIP={advertised}&" in status.web_client_url
    assert status.certificate_url == f"https://{advertised}:48322/"
