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

"""CPU tests for the MJPEG presenter's ``--stream-token`` gate."""

from __future__ import annotations

import urllib.error
import urllib.request
from collections.abc import Iterator

import numpy as np
import pytest
from crazy_robotaxi.streaming_presenter import MJPEGStreamingPresenter
from omnidreams_game_engine.config import RasterConfig
from omnidreams_game_engine.input.keyboard import KeyboardState

TOKEN = "sekrit-demo-token"


def _make_presenter(token: str | None) -> MJPEGStreamingPresenter:
    return MJPEGStreamingPresenter(
        raster=RasterConfig(),
        keyboard=KeyboardState(),
        bind_host="127.0.0.1",
        bind_port=0,
        stream_token=token,
    )


def _base_url(presenter: MJPEGStreamingPresenter) -> str:
    host, port = (
        presenter._server.server_address[0],
        presenter._server.server_address[1],
    )
    return f"http://{host}:{port}"


def _get_status(
    url: str, *, headers: dict[str, str] | None = None
) -> tuple[int, bytes]:
    request = urllib.request.Request(url, headers=headers or {})
    try:
        with urllib.request.urlopen(request, timeout=5.0) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as error:
        return error.code, error.read()


@pytest.fixture
def gated() -> Iterator[tuple[MJPEGStreamingPresenter, str]]:
    presenter = _make_presenter(TOKEN)
    try:
        yield presenter, _base_url(presenter)
    finally:
        presenter.close()


@pytest.fixture
def open_presenter() -> Iterator[tuple[MJPEGStreamingPresenter, str]]:
    presenter = _make_presenter(None)
    try:
        yield presenter, _base_url(presenter)
    finally:
        presenter.close()


def test_rejects_requests_without_token(
    gated: tuple[MJPEGStreamingPresenter, str],
) -> None:
    _, base = gated
    for path in ("/", "/state", "/control?key=w&down=1", "/stream", "/scenes"):
        status, _ = _get_status(base + path)
        assert status == 403, path


def test_rejects_requests_with_wrong_token(
    gated: tuple[MJPEGStreamingPresenter, str],
) -> None:
    _, base = gated
    status, _ = _get_status(base + "/state?token=not-the-token")
    assert status == 403
    status, _ = _get_status(
        base + "/state", headers={"X-Stream-Token": "not-the-token"}
    )
    assert status == 403


def test_accepts_valid_token_as_query_param(
    gated: tuple[MJPEGStreamingPresenter, str],
) -> None:
    _, base = gated
    status, body = _get_status(f"{base}/?token={TOKEN}")
    assert status == 200
    assert b"<html" in body
    status, _ = _get_status(f"{base}/state?token={TOKEN}")
    assert status == 200
    status, _ = _get_status(f"{base}/control?key=w&down=1&token={TOKEN}")
    assert status == 204


def test_accepts_valid_token_as_header(
    gated: tuple[MJPEGStreamingPresenter, str],
) -> None:
    _, base = gated
    status, _ = _get_status(base + "/state", headers={"X-Stream-Token": TOKEN})
    assert status == 200


def test_control_with_token_reaches_the_keyboard(
    gated: tuple[MJPEGStreamingPresenter, str],
) -> None:
    presenter, base = gated
    status, _ = _get_status(f"{base}/control?key=3&down=1&token={TOKEN}")
    assert status == 204
    assert presenter._keyboard.view_mode == "physx"


def test_stream_with_token_serves_multipart_frames(
    gated: tuple[MJPEGStreamingPresenter, str],
) -> None:
    presenter, base = gated
    presenter._publish(np.zeros((8, 8, 3), dtype=np.uint8))
    request = urllib.request.Request(f"{base}/stream?token={TOKEN}")
    with urllib.request.urlopen(request, timeout=5.0) as response:
        assert response.status == 200
        assert "multipart/x-mixed-replace" in response.headers["Content-Type"]
        first_bytes = response.read(24)
    assert b"--interactive_drive" in first_bytes


def test_served_page_embeds_token_plumbing(
    gated: tuple[MJPEGStreamingPresenter, str],
) -> None:
    _, base = gated
    status, body = _get_status(f"{base}/?token={TOKEN}")
    assert status == 200
    page = body.decode("utf-8")
    # The page reads the token from its own URL and appends it to every
    # request it makes, including both MJPEG streams.
    assert "URLSearchParams(window.location.search).get('token')" in page
    assert "withToken('/stream')" in page
    assert "withToken('/bev_stream')" in page
    assert "withToken('/state')" in page
    # No raw, un-tokenized fetches remain in the page.
    assert "fetch('/" not in page


def test_absent_token_flag_keeps_open_behavior(
    open_presenter: tuple[MJPEGStreamingPresenter, str],
) -> None:
    _, base = open_presenter
    status, body = _get_status(base + "/")
    assert status == 200
    assert b"<html" in body
    status, _ = _get_status(base + "/state")
    assert status == 200


def test_empty_token_flag_keeps_open_behavior() -> None:
    presenter = _make_presenter("   ")
    try:
        status, _ = _get_status(_base_url(presenter) + "/state")
        assert status == 200
    finally:
        presenter.close()
