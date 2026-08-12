# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from flashdreams.serving.webrtc import warmup as warmup_module
from flashdreams.serving.webrtc.messages import make_error_payload
from flashdreams.serving.webrtc.warmup import run_loopback_warmup_session

pytestmark = pytest.mark.ci_cpu


@pytest.mark.asyncio
async def test_loopback_warmup_fails_on_server_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    channel = _FakeLoopbackChannel(
        incoming_on_send=(make_error_payload("shared driver failed"),)
    )
    _install_fake_loopback_peer(monkeypatch, channel=channel)

    with pytest.raises(RuntimeError, match="shared driver failed"):
        await run_loopback_warmup_session(
            num_chunks=1,
            warmup_timeout_s=1.0,
            create_answer=_fake_create_answer,
            action_payloads=(_step_action(),),
        )


@pytest.mark.asyncio
async def test_loopback_warmup_fails_on_early_channel_close(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    channel = _FakeLoopbackChannel(close_on_send=True)
    _install_fake_loopback_peer(monkeypatch, channel=channel)

    with pytest.raises(RuntimeError, match=r"0/1 chunk"):
        await run_loopback_warmup_session(
            num_chunks=1,
            warmup_timeout_s=1.0,
            create_answer=_fake_create_answer,
            action_payloads=(_step_action(),),
        )


async def _fake_create_answer(*, offer_sdp: str, offer_type: str) -> dict[str, str]:
    del offer_sdp, offer_type
    return {"sdp": "answer-sdp", "type": "answer"}


def _step_action() -> dict[str, object]:
    return {"type": "action", "action": {"event": "step"}}


def _install_fake_loopback_peer(
    monkeypatch: pytest.MonkeyPatch,
    *,
    channel: "_FakeLoopbackChannel",
) -> None:
    monkeypatch.setattr(
        warmup_module,
        "RTCPeerConnection",
        lambda _configuration: _FakeLoopbackPeer(channel=channel),
    )

    async def wait_for_ice_gathering_complete(*args: Any, **kwargs: Any) -> None:
        del args, kwargs

    monkeypatch.setattr(
        warmup_module,
        "wait_for_ice_gathering_complete",
        wait_for_ice_gathering_complete,
    )


class _FakeLoopbackPeer:
    iceGatheringState = "complete"

    def __init__(self, *, channel: "_FakeLoopbackChannel") -> None:
        self.localDescription: Any | None = None
        self._channel = channel

    def createDataChannel(self, *args: Any, **kwargs: Any) -> "_FakeLoopbackChannel":
        del args, kwargs
        return self._channel

    def addTransceiver(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs

    def on(self, event_name: str) -> Any:
        del event_name

        def decorator(callback: Any) -> Any:
            return callback

        return decorator

    async def createOffer(self) -> Any:
        return SimpleNamespace(sdp="offer-sdp", type="offer")

    async def setLocalDescription(self, description: Any) -> None:
        self.localDescription = description

    async def setRemoteDescription(self, description: Any) -> None:
        del description
        self._channel.open()

    async def close(self) -> None:
        self._channel.close()


class _FakeLoopbackChannel:
    readyState = "open"

    def __init__(
        self,
        *,
        incoming_on_send: tuple[dict[str, str], ...] = (),
        close_on_send: bool = False,
    ) -> None:
        self._incoming_on_send = list(incoming_on_send)
        self._close_on_send = close_on_send
        self._handlers: dict[str, Any] = {}

    def on(self, event_name: str) -> Any:
        def decorator(callback: Any) -> Any:
            self._handlers[event_name] = callback
            return callback

        return decorator

    def send(self, message: str) -> None:
        del message
        if self._incoming_on_send:
            self._handlers["message"](
                warmup_module.json.dumps(self._incoming_on_send.pop(0))
            )
        if self._close_on_send:
            self.close()

    def open(self) -> None:
        self._handlers["open"]()

    def close(self) -> None:
        self.readyState = "closed"
        close_handler = self._handlers.get("close")
        if close_handler is not None:
            close_handler()
