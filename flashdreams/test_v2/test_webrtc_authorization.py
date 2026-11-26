# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU authorization and lifecycle checks for V2 browser admission."""

# ruff: noqa: E402 - optional serving dependencies must follow importorskip.

import argparse
import asyncio
import json
import os
import shutil
import subprocess
import time
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, Mock

import pytest

pytest.importorskip("aiohttp")
pytest.importorskip("aiortc")

from aiohttp import ClientSession, web
from aiohttp.test_utils import make_mocked_request
from aiortc.mediastreams import MediaStreamError
from av import VideoFrame

from flashdreams.runtime_v2.client_window_factory import (
    add_client_window_arguments,
    client_window_mode,
)
from flashdreams.runtime_v2.selected_file import (
    FileSelectionRequest,
    SelectedFile,
    SelectedFilesStatus,
)
from flashdreams.runtime_v2.serving import webrtc_server
from flashdreams.runtime_v2.serving.security import BrowserSessionAuth
from flashdreams.runtime_v2.session_desc import PresentationMode, SessionDesc
from flashdreams.runtime_v2.user_input_event import (
    InputOwnerChangedUserInputEvent,
    KeyboardInputState,
    KeyboardUserInputEvent,
    QueryStringUserInputEvent,
)
from flashdreams.runtime_v2.video_tensor import VideoTensorLayout
from flashdreams.runtime_v2.webrtc_client_window import WebRTCClientWindow

pytestmark = pytest.mark.ci_cpu

TOKEN = "A" * 43
NEW_TOKEN = "B" * 43
ORIGIN = "https://viewer.example"


def _clean_configuration(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FLASHDREAMS_WEBRTC_TOKEN_FILE", raising=False)
    monkeypatch.delenv("FLASHDREAMS_WEBRTC_ALLOWED_ORIGINS", raising=False)


def _request(
    path: str = "/api/webrtc/offer",
    *,
    token: str | None = TOKEN,
    host: str = "viewer.example",
    origin: str | None = ORIGIN,
    peer: str = "127.0.0.1",
    headers: dict[str, str] | None = None,
    data: bytes = b"abc",
) -> Any:
    fields = {"Host": host, **(headers or {})}
    if origin is not None:
        fields["Origin"] = origin
    if token is not None:
        fields["Authorization"] = f"Bearer {token}"
    request: Any = make_mocked_request(
        "POST",
        path,
        headers=fields,
        transport=Mock(get_extra_info=Mock(return_value=(peer, 54321))),
        payload=Mock(readexactly=AsyncMock(return_value=data)),
    )
    request.json = AsyncMock(return_value={"sdp": "offer-sdp", "type": "offer"})
    return request


@pytest.mark.parametrize("host", ["0.0.0.0", "::", "192.0.2.1", "gpu.example"])
def test_nonlocal_startup_requires_explicit_bearer_and_origin(
    host: str, monkeypatch
) -> None:
    _clean_configuration(monkeypatch)
    with pytest.raises(ValueError, match="TOKEN_FILE"):
        BrowserSessionAuth(host)
    with pytest.raises(ValueError, match="ALLOWED_ORIGINS"):
        BrowserSessionAuth(host, auth_token=TOKEN)


@pytest.mark.parametrize("host", ["0.0.0.0", "127.0.0.1"])
def test_authenticated_direct_and_ingress_never_have_a_local_bypass(
    host: str, monkeypatch
) -> None:
    _clean_configuration(monkeypatch)
    policy = BrowserSessionAuth(host, auth_token=TOKEN, allowed_origins=(ORIGIN,))
    for token in (None, NEW_TOKEN):
        with pytest.raises(web.HTTPUnauthorized):
            policy.authorize(_request(token=token), 8080, protected=True)
    assert policy.authorize(_request(), 8080, protected=True) == 0
    # Ingress headers grant no authority and cannot rewrite the approved Host.
    with pytest.raises(web.HTTPForbidden):
        policy.authorize(
            _request(
                host="attacker.example", headers={"X-Forwarded-Host": "viewer.example"}
            ),
            8080,
            protected=True,
        )


@pytest.mark.parametrize(
    "fields",
    [
        {"origin": "https://attacker.example"},
        {"origin": "null"},
        {"host": "attacker.example"},
        {"host": "viewer.example:444"},
        {"headers": {"Sec-Fetch-Site": "cross-site"}},
    ],
)
def test_authenticated_browser_origin_and_host_are_allowlisted(fields: dict) -> None:
    policy = BrowserSessionAuth("0.0.0.0", auth_token=TOKEN, allowed_origins=(ORIGIN,))
    with pytest.raises(web.HTTPForbidden):
        policy.authorize(_request(**fields), 8080, protected=True)


def test_configured_origin_uses_browser_default_port_and_hostname_normalization() -> (
    None
):
    policy = BrowserSessionAuth(
        "0.0.0.0", auth_token=TOKEN, allowed_origins=("https://VIEWER.example:443",)
    )
    assert policy.browser_origin == ORIGIN
    assert policy.authorize(_request(), 8080, protected=True) == 0


@pytest.mark.parametrize(
    "origin",
    [
        "null",
        "https://host/path",
        "https://u:p@host",
        "https://host?token=secret",
        "https://host#secret",
        "*",
        "https://host:bad",
        "http://viewer.example",
    ],
)
def test_origin_configuration_is_exact(origin: str) -> None:
    with pytest.raises(ValueError):
        BrowserSessionAuth("0.0.0.0", auth_token=TOKEN, allowed_origins=(origin,))


@pytest.mark.parametrize("origin", [None, "http://localhost:8080"])
def test_direct_loopback_stays_simple(origin: str | None, monkeypatch) -> None:
    _clean_configuration(monkeypatch)
    policy = BrowserSessionAuth("127.0.0.1")
    assert (
        policy.authorize(
            _request(host="localhost:8080", origin=origin, token=None),
            8080,
            protected=True,
        )
        == 0
    )
    policy.rotate(None)
    with pytest.raises(web.HTTPUnauthorized):
        policy.authorize(
            _request(host="localhost:8080", origin=origin, token=None),
            8080,
            protected=True,
        )


@pytest.mark.parametrize(
    "fields",
    [
        {"origin": "https://attacker.example"},
        {"origin": "null"},
        {"host": "rebind.attacker.example:8080", "origin": None},
        {"host": "localhost:9999", "origin": None},
        {"peer": "192.0.2.1"},
        {"headers": {"Forwarded": "for=127.0.0.1"}},
        {"headers": {"X-Forwarded-For": "127.0.0.1"}},
        {"headers": {"X-Forwarded-Host": "localhost:8080"}},
        {"headers": {"X-Real-IP": "127.0.0.1"}},
        {"headers": {"Sec-Fetch-Site": "cross-site"}},
    ],
)
def test_local_bypass_rejects_cross_origin_rebinding_and_forwarded_requests(
    fields: dict, monkeypatch
) -> None:
    _clean_configuration(monkeypatch)
    policy = BrowserSessionAuth("127.0.0.1")
    options: dict[str, Any] = {
        "host": "localhost:8080",
        "origin": "http://localhost:8080",
        "token": None,
    }
    options.update(fields)
    with pytest.raises(web.HTTPForbidden):
        policy.authorize(_request(**options), 8080, protected=True)


def test_token_file_is_private_and_never_printed(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _clean_configuration(monkeypatch)
    path = tmp_path / "owner-token"
    path.write_text(TOKEN + "\n")
    path.chmod(0o600)
    monkeypatch.setenv("FLASHDREAMS_WEBRTC_TOKEN_FILE", str(path))
    policy = BrowserSessionAuth("127.0.0.1")
    with pytest.raises(web.HTTPUnauthorized):
        policy.authorize(
            _request(host="localhost:8080", origin=None, token=None),
            8080,
            protected=True,
        )
    path.chmod(0o644)
    with pytest.raises(ValueError) as error:
        BrowserSessionAuth("127.0.0.1")
    assert TOKEN not in str(error.value)
    # Unopenable files are usage errors, not tracebacks.
    link = tmp_path / "link"
    link.symlink_to(path)
    for target, reason in ((tmp_path / "missing", "ENOENT"), (link, "ELOOP")):
        monkeypatch.setenv("FLASHDREAMS_WEBRTC_TOKEN_FILE", str(target))
        with pytest.raises(ValueError, match=reason):
            BrowserSessionAuth("127.0.0.1")


def test_non_utf8_authorization_is_unauthorized_not_a_server_error() -> None:
    policy = BrowserSessionAuth(
        "127.0.0.1", auth_token=TOKEN, allowed_origins=(ORIGIN,)
    )
    # aiohttp decodes undecodable header bytes as surrogate escapes.
    request: Any = SimpleNamespace(
        headers={"Host": "viewer.example", "Authorization": "Bearer \udcff"}
    )
    with pytest.raises(web.HTTPUnauthorized):
        policy.authorize(request, 8080, protected=True)


def _server(monkeypatch: pytest.MonkeyPatch, **options) -> Any:
    """Return a socketless server whose collaborators tests replace with doubles."""
    _clean_configuration(monkeypatch)
    # Exercise normal initialization without sockets, models, ICE, or GPU work.
    monkeypatch.setattr(
        webrtc_server.WebRTCServer, "_run_server", lambda self: self._started.set()
    )
    settings: dict[str, Any] = {"auth_token": TOKEN, "allowed_origins": (ORIGIN,)}
    instance = webrtc_server.WebRTCServer(port=8080, **(settings | options))
    instance._offer_lock = asyncio.Lock()
    instance._session_desc = SessionDesc(
        output_layout=VideoTensorLayout.tchw,
        presentation_mode=PresentationMode.ON_DEMAND,
        frames_per_second_for_ui=30,
        frames_per_second_for_step=30,
        video_width=16,
        video_height=16,
    )
    instance._event_origin_ns = time.monotonic_ns()
    instance.register_input_callback(Mock())
    instance.register_file_selector_callback(Mock())
    return instance


def test_interrupted_first_launch_leaves_no_empty_credential(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from flashdreams.runtime_v2.serving import security

    monkeypatch.delenv("FLASHDREAMS_WEBRTC_TOKEN_FILE", raising=False)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))

    def crash(_size: int) -> str:
        raise OSError("disk full")

    with monkeypatch.context() as patched:
        patched.setattr(security.secrets, "token_urlsafe", crash)
        with pytest.raises(ValueError, match="could not be created"):
            security.load_pairing_token()
    assert list((tmp_path / "flashdreams").iterdir()) == []
    # The next launch creates a usable credential instead of failing forever.
    token = security.load_pairing_token()
    assert (tmp_path / "flashdreams" / "webrtc-token").read_text().strip() == token


def test_default_credential_works_without_posix_file_apis(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Windows has no O_NOFOLLOW, O_NONBLOCK or geteuid and reports mode 666."""
    from flashdreams.runtime_v2.serving import security

    monkeypatch.delenv("FLASHDREAMS_WEBRTC_TOKEN_FILE", raising=False)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))

    class WindowsOS:
        name = "nt"

        def __getattr__(self, attribute: str) -> object:
            if attribute in ("O_NOFOLLOW", "O_NONBLOCK", "geteuid"):
                raise AttributeError(attribute)
            return getattr(os, attribute)

    monkeypatch.setattr(security, "os", WindowsOS())
    token = security.load_pairing_token()
    (tmp_path / "flashdreams" / "webrtc-token").chmod(0o666)
    assert security.load_pairing_token() == token


@pytest.mark.asyncio
async def test_default_launch_pairs_the_browser_so_local_clients_cannot_displace_it(
    tmp_path, monkeypatch
) -> None:
    _clean_configuration(monkeypatch)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.setattr(
        webrtc_server.WebRTCServer, "_run_server", lambda self: self._started.set()
    )
    peers = _peers(monkeypatch)
    parser = argparse.ArgumentParser()
    add_client_window_arguments(parser)
    mode = client_window_mode("webrtc")
    credential = tmp_path / "flashdreams" / "webrtc-token"
    local: dict[str, Any] = {"host": "127.0.0.1:8080", "origin": None}

    def launch(*options: str) -> webrtc_server.WebRTCServer:
        args = parser.parse_args(["--mode", "webrtc", "--port", "8080", *options])
        mode.check_arguments(args)
        server = cast(WebRTCClientWindow, mode.create(args)).server
        server._offer_lock = asyncio.Lock()
        server._session_desc = SessionDesc(
            output_layout=VideoTensorLayout.tchw,
            presentation_mode=PresentationMode.ON_DEMAND,
            frames_per_second_for_ui=30,
            frames_per_second_for_step=30,
            video_width=16,
            video_height=16,
        )
        server._event_origin_ns = time.monotonic_ns()
        return server

    # Every rank validates the launch; none of that reads or creates the secret.
    mode.check_arguments(parser.parse_args(["--mode", "webrtc"]))
    assert not credential.exists()
    server = launch()
    assert credential.stat().st_mode & 0o777 == 0o600
    assert credential.parent.stat().st_mode & 0o777 == 0o700
    token = credential.read_text().strip()
    await server._offer(_request(token=token, **local))
    paired = server._peer_connection
    assert paired is peers[0]
    with pytest.raises(web.HTTPUnauthorized):
        await server._offer(_request(token=None, **local))
    assert server._peer_connection is paired
    paired.close.assert_not_awaited()
    # A restart reuses the credential, so a paired tab survives it.
    assert launch().pairing_url == f"http://127.0.0.1:8080/#pair={token}"
    credential.chmod(0)
    mode.check_arguments(parser.parse_args(["--mode", "webrtc"]))
    credential.chmod(0o600)
    # Tokenless loopback stays an explicit choice for a single-user machine.
    tokenless = launch("--no-login")
    await tokenless._offer(_request(token=None, **local))
    assert tokenless._peer_connection is peers[-1]
    for refused in (["--host", "0.0.0.0"], []):
        if not refused:
            monkeypatch.setenv("FLASHDREAMS_WEBRTC_ALLOWED_ORIGINS", ORIGIN)
        args = parser.parse_args(["--mode", "webrtc", "--no-login", *refused])
        with pytest.raises(ValueError, match="--no-login"):
            mode.check_arguments(args)


def _peers(monkeypatch: pytest.MonkeyPatch, state: str = "connected") -> list[Mock]:
    created: list[Mock] = []

    def create() -> Mock:
        peer = Mock(connectionState=state)
        peer.handlers = {}
        peer.on = lambda name: lambda callback: peer.handlers.setdefault(name, callback)
        peer.setRemoteDescription = AsyncMock()
        peer.createAnswer = AsyncMock(
            return_value=SimpleNamespace(sdp="answer", type="answer")
        )
        peer.setLocalDescription = AsyncMock()
        peer.localDescription = SimpleNamespace(sdp="answer", type="answer")
        peer.close = AsyncMock()
        created.append(peer)
        return peer

    monkeypatch.setattr(webrtc_server, "RTCPeerConnection", create)
    return created


@pytest.mark.asyncio
async def test_owner_refresh_and_retired_callbacks_frames_input_and_uploads(
    monkeypatch,
) -> None:
    server, peers = _server(monkeypatch), _peers(monkeypatch)
    viewer = Mock()
    server.register_viewer_callback(viewer)
    await server._offer(_request())
    first, first_track = server._peer_connection, server._video_track
    channel = Mock(label="controls", readyState="open")
    callbacks: dict[str, Callable] = {}
    channel.on = lambda name: lambda callback: callbacks.setdefault(name, callback)
    first.handlers["datachannel"](channel)
    callbacks["message"](json.dumps({"type": "focus", "focused": True}))
    assert server._input_callback.call_count == 1
    server._pending_file_requests["pick"] = webrtc_server._PendingFileSelection(
        selector_id="1",
        request=FileSelectionRequest(request_id="1", initial_path="/tmp"),
    )
    await server._offer(_request())
    replacement, replacement_track = server._peer_connection, server._video_track
    assert replacement is peers[1] and replacement is not first
    assert first_track.enqueue(VideoFrame(width=16, height=16)) is False
    with pytest.raises(MediaStreamError):
        await first_track.recv()
    assert server._pending_file_requests == {}
    assert (
        server._file_selector_callback.call_args.args[1]
        is SelectedFilesStatus.UNAVAILABLE
    )
    for callback in (
        callbacks["open"],
        lambda: callbacks["message"]('{"type":"reset"}'),
    ):
        callback()
    first.connectionState = "failed"
    await first.handlers["connectionstatechange"]()
    await callbacks["close"]()
    assert server._peer_connection is replacement
    assert server._video_track is replacement_track
    assert server._input_callback.call_count == 1
    states = [call.args[0] for call in viewer.call_args_list]
    assert len(states) == 4 and states[0] is None and states[2] is None
    assert states[1] is not None
    assert states[3] is states[1]  # The same bearer holder refreshed.


@pytest.mark.asyncio
async def test_delayed_retired_callbacks_cannot_clear_replacement_window_input(
    monkeypatch,
) -> None:
    server, peers = _server(monkeypatch), _peers(monkeypatch)
    window = WebRTCClientWindow()
    server._input_callback = window.server._input_callback
    window.server = server
    server.register_viewer_callback(window._viewer_changed)
    await server._offer(_request())
    first = peers[0]
    channel = Mock(label="controls", readyState="open")
    callbacks = {}
    channel.on = lambda name: lambda callback: callbacks.setdefault(name, callback)
    first.handlers["datachannel"](channel)
    callbacks["message"]('{"type":"keyboard","key":"old","pressed":true}')
    await server._offer(_request())
    server._buffer_browser_message('{"type":"keyboard","key":"new","pressed":true}')
    first.connectionState = "disconnected"
    await first.handlers["connectionstatechange"]()
    await callbacks["close"]()
    callbacks["message"]('{"type":"keyboard","key":"late","pressed":true}')
    events = window.get_user_input_events().get_events()
    assert [type(event) for event in events] == [
        InputOwnerChangedUserInputEvent,
        KeyboardUserInputEvent,
    ]
    assert cast(KeyboardUserInputEvent, events[1]).key == "new"
    assert server._peer_connection is peers[1]


@pytest.mark.asyncio
async def test_unauthorized_offer_does_not_read_body_or_replace_owner(
    monkeypatch,
) -> None:
    server, peers = _server(monkeypatch), _peers(monkeypatch)
    await server._offer(_request())
    active, track = server._peer_connection, server._video_track
    request = _request(token=None)
    with pytest.raises(web.HTTPUnauthorized):
        await server._offer(request)
    request.json.assert_not_awaited()
    assert len(peers) == 1 and server._peer_connection is active
    assert server._video_track is track
    active.close.assert_not_awaited()


@pytest.mark.asyncio
async def test_revocation_disconnects_owner_and_old_credentials_cannot_reconnect(
    monkeypatch,
) -> None:
    server = _server(monkeypatch)
    _peers(monkeypatch)
    await server._offer(_request())
    active, track = server._peer_connection, server._video_track
    await server._rotate_auth_token(None)
    assert server._peer_connection is None
    assert not server._media_connected.is_set()
    assert not track.enqueue(VideoFrame(width=16, height=16))
    active.close.assert_awaited_once()
    for token in (None, TOKEN):
        with pytest.raises(web.HTTPUnauthorized):
            await server._offer(_request(token=token))
    await server._rotate_auth_token(NEW_TOKEN)
    with pytest.raises(web.HTTPUnauthorized):
        await server._offer(_request())
    assert (await server._offer(_request(token=NEW_TOKEN))).status == 200


@pytest.mark.asyncio
async def test_revocation_finishes_teardown_when_a_viewer_callback_raises(
    monkeypatch,
) -> None:
    server = _server(monkeypatch)
    _peers(monkeypatch)
    await server._offer(_request())
    active, track = server._peer_connection, server._video_track
    server._browser_focus = (1, True)
    viewer = Mock()
    server.register_viewer_callback(viewer)
    viewer.side_effect = OSError("cleanup failed")
    with pytest.raises(OSError, match="cleanup failed"):
        await server._rotate_auth_token(None)
    assert server._peer_connection is None and server._video_track is None
    active.close.assert_awaited_once()
    assert not track.enqueue(VideoFrame(width=16, height=16))
    # A retired viewer's focus cannot be replayed into a reopened session.
    assert server.browser_focused is None


@pytest.mark.asyncio
async def test_viewer_identity_follows_the_bearer_not_the_peer(monkeypatch) -> None:
    server = _server(monkeypatch)
    _peers(monkeypatch)
    viewers: list[object | None] = []
    server.register_viewer_callback(viewers.append)
    assert viewers == [None]
    with pytest.raises(RuntimeError, match="viewer callback is already registered"):
        server.register_viewer_callback(viewers.append)
    await server._offer(_request())
    await server._offer(_request())  # The paired browser reloads.
    assert len(viewers) == 4 and viewers[2] is None
    assert viewers[1] is not None and viewers[3] is viewers[1]
    rotated = "B" * 43
    await server._rotate_auth_token(rotated)
    await server._offer(_request(token=rotated))
    assert viewers[-1] not in (None, viewers[1])
    server.close()
    assert viewers[-1] is None


@pytest.mark.asyncio
async def test_each_no_login_peer_gets_a_fresh_viewer_identity(monkeypatch) -> None:
    server = _server(monkeypatch, auth_token=None, allowed_origins=())
    _peers(monkeypatch)
    viewers: list[object | None] = []
    server.register_viewer_callback(viewers.append)
    local: dict[str, Any] = {"token": None, "host": "127.0.0.1:8080", "origin": None}
    await server._offer(_request(**local))
    await server._offer(_request(**local))
    assert len(viewers) == 4 and viewers[2] is None
    assert viewers[1] is not None and viewers[3] not in (None, viewers[1])
    server.close()


def test_new_viewer_retains_unread_handoff_and_its_current_query(monkeypatch) -> None:
    _clean_configuration(monkeypatch)
    monkeypatch.setattr(
        webrtc_server.WebRTCServer, "_run_server", lambda self: self._started.set()
    )
    window = WebRTCClientWindow()
    try:
        window.open(
            SessionDesc(
                output_layout=VideoTensorLayout.tchw, video_width=16, video_height=16
            )
        )
        window.put_user_input_event(
            KeyboardUserInputEvent(
                timestamp=window.server.event_timestamp_us(),
                key="p",
                state=KeyboardInputState.PRESSED,
            )
        )
        window._viewer_changed(None)
        window.put_user_input_event(
            QueryStringUserInputEvent(
                timestamp=window.server.event_timestamp_us(), query_string="scene=new"
            )
        )
        viewer = object()
        window._viewer_changed(viewer)
        events = window.get_user_input_events().get_events()
        assert [type(event) for event in events] == [
            InputOwnerChangedUserInputEvent,
            QueryStringUserInputEvent,
        ]
        assert cast(QueryStringUserInputEvent, events[1]).query_string == "scene=new"
        assert window._viewer is viewer
    finally:
        window.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "stage", ["body", "lock", "remote", "answer", "local", "retire"]
)
async def test_revocation_at_every_await_cannot_publish_a_negotiating_peer(
    stage, monkeypatch
) -> None:
    server, peers = _server(monkeypatch), _peers(monkeypatch)
    await server._offer(_request())
    original = server._peer_connection
    request = _request()
    entered, resume = asyncio.Event(), asyncio.Event()

    async def pause(*args):
        entered.set()
        await resume.wait()
        if stage == "body":
            return {"sdp": "offer", "type": "offer"}
        return SimpleNamespace(sdp="answer", type="answer")

    if stage == "body":
        request.json.side_effect = pause
    elif stage == "lock":
        await server._offer_lock.acquire()
    elif stage == "retire":
        original.close.side_effect = pause
    else:
        factory = webrtc_server.RTCPeerConnection

        def create():
            peer = factory()
            attribute = {
                "remote": "setRemoteDescription",
                "answer": "createAnswer",
                "local": "setLocalDescription",
            }[stage]
            getattr(peer, attribute).side_effect = pause
            return peer

        monkeypatch.setattr(webrtc_server, "RTCPeerConnection", create)
    negotiating = asyncio.create_task(server._offer(request))
    if stage == "lock":
        await asyncio.sleep(0)
        await server._rotate_auth_token(NEW_TOKEN)
        server._offer_lock.release()
    else:
        await asyncio.wait_for(entered.wait(), timeout=2)
        await server._rotate_auth_token(NEW_TOKEN)
        resume.set()
    with pytest.raises(web.HTTPUnauthorized):
        await negotiating
    assert server._peer_connection is None
    for peer in peers:
        peer.close.assert_awaited_once()
        assert peer.addTrack.call_args.args[0]._closed


@pytest.mark.asyncio
async def test_upload_authorizes_before_read_or_token_consumption_and_fences_rotation(
    monkeypatch,
) -> None:
    server = _server(monkeypatch)
    _peers(monkeypatch)
    await server._offer(_request())
    policy = webrtc_server._PendingFileSelection(
        selector_id="1",
        request=FileSelectionRequest(request_id="1", initial_path="/tmp"),
    )
    server._pending_file_requests["pick"] = policy
    path = "/api/files?id=pick&name=a.bin&index=0&count=1"
    request = _request(path, token=None, headers={"Content-Length": "3"})
    with pytest.raises(web.HTTPUnauthorized):
        await server._upload_selected_file(request)
    request.content.readexactly.assert_not_awaited()
    assert server._pending_file_requests["pick"] is policy

    async def read_and_revoke(size):
        # Simulate a read that completes despite cancellation or retirement.
        server._auth.rotate(NEW_TOKEN)
        return b"abc"

    request = _request(path, headers={"Content-Length": "3"})
    request.content.readexactly.side_effect = read_and_revoke
    with pytest.raises(web.HTTPUnauthorized):
        await server._upload_selected_file(request)
    server._file_selector_callback.assert_not_called()


@pytest.mark.asyncio
async def test_upload_fences_same_owner_peer_replacement(monkeypatch) -> None:
    server = _server(monkeypatch)
    _peers(monkeypatch)
    await server._offer(_request())
    server._pending_file_requests["pick"] = webrtc_server._PendingFileSelection(
        selector_id="1",
        request=FileSelectionRequest(request_id="1", initial_path="/tmp"),
    )
    request = _request(
        "/api/files?id=pick&name=a.bin&index=0&count=1", headers={"Content-Length": "3"}
    )

    async def read_and_replace(size):
        server._uploading_file_requests["pick"].reader = None
        await server._offer(_request())
        return b"abc"

    request.content.readexactly.side_effect = read_and_replace
    with pytest.raises(web.HTTPConflict):
        await server._upload_selected_file(request)
    assert all(
        call.args[1] is SelectedFilesStatus.UNAVAILABLE
        for call in server._file_selector_callback.call_args_list
    )


def test_window_discards_retired_input_before_notifying_consumer(monkeypatch) -> None:
    _clean_configuration(monkeypatch)
    monkeypatch.setattr(
        webrtc_server.WebRTCServer, "_run_server", lambda self: self._started.set()
    )
    window = WebRTCClientWindow()
    callback = Mock()
    window.register_viewer_callback(callback)
    callback.reset_mock()
    window._input_events.append(Mock())
    window._file_selector_reports.append(
        (1, SelectedFilesStatus.OK, (SelectedFile(name="old.bin", data=b"old"),))
    )
    window._viewer_changed(None)
    assert len(window._input_events) == 1
    assert isinstance(window._input_events[0], InputOwnerChangedUserInputEvent)
    assert list(window._file_selector_reports) == [
        (1, SelectedFilesStatus.UNAVAILABLE, ())
    ]
    callback.assert_called_once_with(None)


@pytest.mark.asyncio
@pytest.mark.parametrize("state", ["connecting", "disconnected"])
async def test_retirement_before_connect_still_invalidates_consumers(
    state,
    monkeypatch,
) -> None:
    server = _server(monkeypatch)
    _peers(monkeypatch, state=state)
    viewer = Mock()
    server.register_viewer_callback(viewer)
    viewer.assert_called_once_with(None)
    viewer.reset_mock()
    await server._offer(_request())
    peer = server._peer_connection
    viewer.assert_not_called()
    channel = Mock(label="controls", readyState="open")
    callbacks = {}
    channel.on = lambda name: lambda callback: callbacks.setdefault(name, callback)
    peer.handlers["datachannel"](channel)
    callbacks["message"]('{"type":"reset"}')
    server._input_callback.assert_not_called()
    viewer.reset_mock()
    await server._release_peer_connection(peer)
    viewer.assert_called_once_with(None)


def test_browser_pairing_paste_link_refresh_and_forget() -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is required for the browser pairing self-check.")
    script = Path(webrtc_server.__file__).parent / "web" / "app.js"
    check = r"""
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const source = fs.readFileSync(process.argv[1], 'utf8').split('const peer =')[0];
const storage = new Map();
const calls = [];
const replaced = [];
const elements = new Map();
const document = {getElementById(id) {
  if (!elements.has(id)) elements.set(id, {
    value: '', hidden: true, handlers: {},
    addEventListener(name, callback) { this.handlers[name] = callback; },
    showModal() { this.open = true; }, focus() {},
  });
  return elements.get(id);
}};
const makeContext = (href = 'https://viewer.example/', statuses = []) => {
  const context = vm.createContext({
    document, Headers, URL, window: {location: new URL(href)},
    console, setTimeout, showStatus() {},
    history: {replaceState: (_state, _title, url) => replaced.push(url)},
    sessionStorage: {
      getItem: key => storage.get(key),
      setItem: (key, value) => storage.set(key, value),
      removeItem: key => storage.delete(key),
    },
    fetch: async (url, options) => {
      calls.push({url, options});
      const status = statuses.shift() ?? 200;
      return {ok: status === 200, status, json: async () => ({open: true})};
    },
  });
  vm.runInContext(source, context);
  return context;
};
(async () => {
  let context = makeContext();
  const pending = vm.runInContext('requestPairing()', context);
  const input = elements.get('pairing-token');
  input.value = 'C'.repeat(43);
  elements.get('pairing-form').handlers.submit();
  await pending;
  assert.equal(input.value, '');
  assert.equal(elements.get('forget-pairing').hidden, false);
  // Refresh uses tab storage; the URL contains no credential.
  context = makeContext();
  await vm.runInContext("apiFetch('/api/webrtc/offer', {method: 'POST'})", context);
  const {url, options} = calls[0];
  assert.equal(url, '/api/webrtc/offer');
  assert.equal(options.headers.get('Authorization'), 'Bearer ' + 'C'.repeat(43));
  assert.equal(options.credentials, 'omit');
  assert.equal(options.redirect, 'error');
  assert.equal(options.referrerPolicy, 'no-referrer');
  vm.runInContext("rememberToken('')", context);
  assert.equal(storage.size, 0);
  assert.equal(elements.get('forget-pairing').hidden, true);
  const before = calls.length;
  await assert.rejects(
    vm.runInContext("apiFetch('https://attacker.example/api/files')", context),
    /same origin/,
  );
  context = makeContext('http://viewer.example/');
  await assert.rejects(vm.runInContext("apiFetch('/api/webrtc/offer')", context), /HTTPS/);
  assert.equal(calls.length, before);
  context = makeContext('http://127.0.0.1:8089/');
  await vm.runInContext("apiFetch('/api/webrtc/offer')", context);
  assert.equal(calls.length, before + 1);
  // A refused page policy ends the wait instead of polling forever.
  context = makeContext('http://viewer.example/');
  await assert.rejects(vm.runInContext('waitForServer()', context), /HTTPS/);
  assert.equal(calls.length, before + 1);
  // Client and server agree that all of 127.0.0.0/8 is loopback.
  context = makeContext('http://127.0.0.2:8089/');
  assert.equal((await vm.runInContext('waitForServer()', context)).open, true);
  // A refused token is explained, not silently re-prompted.
  storage.set('flashdreams-webrtc-owner', 'D'.repeat(43));
  context = makeContext('https://viewer.example/', [401]);
  const waiting = vm.runInContext('waitForServer()', context);
  await new Promise(resolve => setImmediate(resolve));
  const notice = elements.get('pairing-error');
  assert.match(notice.textContent, /not accepted/);
  assert.equal(notice.hidden, false);
  elements.get('pairing-token').value = 'E'.repeat(43);
  elements.get('pairing-form').handlers.submit();
  assert.equal((await waiting).open, true);
  assert.equal(calls.at(-1).options.headers.get('Authorization'), 'Bearer ' + 'E'.repeat(43));
  // Pages opened without a link leave the address bar alone.
  assert.equal(replaced.length, 0);
  // The terminal's one-click link pairs the tab and leaves no credential in the
  // address bar; the fragment itself is never part of a request URL.
  storage.clear();
  context = makeContext('http://127.0.0.1:8089/?view=1#pair=' + 'F'.repeat(43));
  assert.equal(storage.get('flashdreams-webrtc-owner'), 'F'.repeat(43));
  assert.deepEqual(replaced, ['/?view=1']);
  assert.equal(elements.get('forget-pairing').hidden, false);
  await vm.runInContext("apiFetch('/healthz')", context);
  assert.equal(calls.at(-1).url, '/healthz');
  assert.equal(calls.at(-1).options.headers.get('Authorization'), 'Bearer ' + 'F'.repeat(43));
})().catch(error => { console.error(error); process.exitCode = 1; });
"""
    subprocess.run([node, "-e", check, str(script)], check=True, timeout=10)


@pytest.mark.asyncio
async def test_real_http_routes_preserve_owner_on_rejection_and_allow_authenticated_refresh(
    monkeypatch,
) -> None:
    server, peers = _server(monkeypatch), _peers(monkeypatch)
    # Exercise production route/middleware wiring with real HTTP, mocked peers.
    server._port = 0
    await server._start_server()
    endpoint = f"http://127.0.0.1:{server.port}"
    headers = {"Host": "viewer.example", "Origin": ORIGIN}
    try:
        async with ClientSession(base_url=endpoint) as client:
            response = await client.get("/", headers=headers)
            assert response.status == 200
            assert response.headers["Referrer-Policy"] == "no-referrer"
            assert response.headers["Cache-Control"] == "no-store"
            authorized = {**headers, "Authorization": f"Bearer {TOKEN}"}
            response = await client.post(
                "/api/webrtc/offer",
                headers=authorized,
                json={"sdp": "offer", "type": "offer"},
            )
            assert response.status == 200
            active = server._peer_connection
            server._pending_file_requests["pick"] = webrtc_server._PendingFileSelection(
                selector_id="1",
                request=FileSelectionRequest(request_id="1", initial_path="/tmp"),
            )
            for path in (
                "/api/webrtc/offer",
                "/api/files?id=pick&name=a.bin&index=0&count=1",
            ):
                response = await client.post(path, headers=headers, data=b"abc")
                assert response.status == 401
                assert server._peer_connection is active
            assert "pick" in server._pending_file_requests
            response = await client.post(
                "/api/files?id=pick&name=a.bin&index=0&count=1",
                headers=authorized,
                data=b"abc",
            )
            assert response.status == 204
            assert (
                server._file_selector_callback.call_args.args[1]
                is SelectedFilesStatus.OK
            )
            assert server._file_selector_callback.call_args.args[2][0].data == b"abc"
            assert (await client.get("/healthz", headers=headers)).status == 401
            assert (await client.get("/healthz", headers=authorized)).status == 200
            for bad in (
                {"Origin": "https://attacker.example"},
                {"Host": "rebind.attacker.example"},
            ):
                response = await client.post(
                    "/api/webrtc/offer",
                    headers={**authorized, **bad},
                    json={"sdp": "offer", "type": "offer"},
                )
                assert response.status == 403
                assert server._peer_connection is active
            response = await client.post(
                "/api/webrtc/offer",
                headers=authorized,
                json={"sdp": "offer", "type": "offer"},
            )
            assert response.status == 200 and server._peer_connection is peers[1]
    finally:
        await server._shutdown()


@pytest.mark.asyncio
async def test_real_http_direct_local_browser_and_cross_origin_rejection(
    monkeypatch,
) -> None:
    server, peers = _server(monkeypatch), _peers(monkeypatch)
    server._auth = BrowserSessionAuth("127.0.0.1")
    server._port = 0
    await server._start_server()
    endpoint = f"http://127.0.0.1:{server.port}"
    headers = {"Origin": endpoint}
    try:
        async with ClientSession(base_url=endpoint) as client:
            response = await client.post(
                "/api/webrtc/offer",
                headers=headers,
                json={"sdp": "offer", "type": "offer"},
            )
            assert response.status == 200
            active = server._peer_connection
            for bad in (
                {"Origin": "https://attacker.example"},
                {
                    "Host": f"rebind.attacker.example:{server.port}",
                    "Origin": "https://attacker.example",
                },
                {"X-Forwarded-For": "127.0.0.1"},
            ):
                response = await client.post(
                    "/api/webrtc/offer",
                    headers={**headers, **bad},
                    json={"sdp": "offer", "type": "offer"},
                )
                assert response.status == 403
                assert server._peer_connection is active
            assert len(peers) == 1
    finally:
        await server._shutdown()
