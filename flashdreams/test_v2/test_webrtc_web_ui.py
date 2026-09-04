# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for the application-served browser UI on the v2 WebRTC server."""

# ruff: noqa: E402 - optional WebRTC imports must follow importorskip.

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
import pytest_asyncio

pytestmark = pytest.mark.ci_cpu

pytest.importorskip("aiohttp")
pytest.importorskip("aiortc")

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from flashdreams.api_v2.web_ui import IWebUiProvider
from flashdreams.runtime_v2.serving.webrtc_server import WebRTCServer


class StubWebUi:
    """Smallest application that satisfies :class:`IWebUiProvider`."""

    def __init__(self, root: Path) -> None:
        self._root = root
        self.applied: list[dict[str, Any]] = []
        self.frame: tuple[bytes, str] | None = (b"\xff\xd8jpeg", "image/jpeg")

    def web_root(self) -> Path:
        return self._root

    def initial_scene(self) -> Mapping[str, Any]:
        return {"prompt": "a scene", "event_catalog": [{"event_id": "storm"}]}

    def first_frame(self) -> tuple[bytes, str] | None:
        return self.frame

    def apply_session_input(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        if payload.get("prompt") == "":
            raise ValueError("Prompt must not be empty.")
        self.applied.append(dict(payload))
        return {"prompt": payload.get("prompt", "unchanged")}


@pytest.fixture
def web_root(tmp_path: Path) -> Path:
    """Return a web root holding one of each thing a page loads."""
    root = tmp_path / "web"
    (root / "assets").mkdir(parents=True)
    (root / "index.html").write_text("<html>page</html>", encoding="utf-8")
    (root / "adapter.js").write_text("// adapter", encoding="utf-8")
    (root / "assets" / "logo.svg").write_text("<svg/>", encoding="utf-8")
    (tmp_path / "outside.txt").write_text("SECRET", encoding="utf-8")
    return root


@pytest.fixture
def server() -> WebRTCServer:
    """Return a server with only the state its HTTP handlers read.

    ``__init__`` starts a thread and binds a socket, neither of which these
    tests need; the handlers under test are the ones the running server
    registers.
    """
    instance = WebRTCServer.__new__(WebRTCServer)
    instance._web_ui = None
    instance._closed = False
    instance._client_connected = False
    instance._hide_cursor = False
    instance._lock_cursor_to_window = False
    instance._session_desc = None
    return instance


@pytest_asyncio.fixture
async def client(server: WebRTCServer) -> TestClient:
    """Return a client for the same routes ``_start_server`` registers."""
    app = web.Application()
    app.router.add_get("/", server._serve_browser)
    app.router.add_get("/app.js", server._serve_browser_script)
    app.router.add_get("/healthz", server._health)
    app.router.add_get("/request_session", server._serve_web_ui_page)
    app.router.add_get("/api/session/initial_scene", server._initial_scene)
    app.router.add_get("/api/session/first_frame", server._first_frame)
    app.router.add_post("/api/session/input", server._session_input)
    app.router.add_get("/{web_asset:.*}", server._serve_web_ui_asset)
    async with TestClient(TestServer(app)) as test_client:
        yield test_client


def test_stub_satisfies_the_protocol(web_root: Path) -> None:
    assert isinstance(StubWebUi(web_root), IWebUiProvider)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "path",
    ["/request_session", "/adapter.js", "/api/session/initial_scene", "/api/session/first_frame"],
)
async def test_routes_are_absent_without_an_application(
    client: TestClient, path: str
) -> None:
    """An application without a UI must see no behaviour change."""
    assert (await client.get(path)).status == 404


@pytest.mark.asyncio
async def test_builtin_routes_survive_a_web_ui(
    client: TestClient, server: WebRTCServer, web_root: Path
) -> None:
    """The catch-all must not shadow the runtime's own viewer."""
    server.serve_web_ui(StubWebUi(web_root))

    assert (await client.get("/")).status == 200
    assert (await client.get("/app.js")).status == 200
    assert (await client.get("/healthz")).status == 200


@pytest.mark.asyncio
async def test_page_and_assets_are_served(
    client: TestClient, server: WebRTCServer, web_root: Path
) -> None:
    server.serve_web_ui(StubWebUi(web_root))

    page = await client.get("/request_session")
    assert page.status == 200
    assert await page.text() == "<html>page</html>"
    assert (await (await client.get("/adapter.js")).text()) == "// adapter"
    assert (await client.get("/assets/logo.svg")).status == 200
    assert (await client.get("/missing.js")).status == 404


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "attack",
    ["/../outside.txt", "/..%2Foutside.txt", "/assets/../../outside.txt"],
)
async def test_paths_cannot_escape_the_web_root(
    client: TestClient, server: WebRTCServer, web_root: Path, attack: str
) -> None:
    server.serve_web_ui(StubWebUi(web_root))

    assert (await client.get(attack)).status == 404


@pytest.mark.asyncio
async def test_initial_scene_is_returned_verbatim(
    client: TestClient, server: WebRTCServer, web_root: Path
) -> None:
    """Serving must not interpret a scene: what the application says, ships."""
    web_ui = StubWebUi(web_root)
    server.serve_web_ui(web_ui)

    response = await client.get("/api/session/initial_scene")

    assert response.status == 200
    assert await response.json() == dict(web_ui.initial_scene())


@pytest.mark.asyncio
async def test_first_frame_is_served_with_its_content_type(
    client: TestClient, server: WebRTCServer, web_root: Path
) -> None:
    server.serve_web_ui(StubWebUi(web_root))

    response = await client.get("/api/session/first_frame")

    assert response.status == 200
    assert response.headers["Content-Type"] == "image/jpeg"
    assert await response.read() == b"\xff\xd8jpeg"


@pytest.mark.asyncio
async def test_absent_first_frame_is_not_an_error(
    client: TestClient, server: WebRTCServer, web_root: Path
) -> None:
    web_ui = StubWebUi(web_root)
    web_ui.frame = None
    server.serve_web_ui(web_ui)

    assert (await client.get("/api/session/first_frame")).status == 404


@pytest.mark.asyncio
async def test_json_input_reaches_the_application(
    client: TestClient, server: WebRTCServer, web_root: Path
) -> None:
    web_ui = StubWebUi(web_root)
    server.serve_web_ui(web_ui)

    response = await client.post("/api/session/input", json={"prompt": "new scene"})

    assert response.status == 200
    assert (await response.json())["prompt"] == "new scene"
    assert web_ui.applied[-1] == {"prompt": "new scene"}


@pytest.mark.asyncio
async def test_form_input_reaches_the_application(
    client: TestClient, server: WebRTCServer, web_root: Path
) -> None:
    web_ui = StubWebUi(web_root)
    server.serve_web_ui(web_ui)

    response = await client.post("/api/session/input", data={"prompt": "formy"})

    assert response.status == 200
    assert web_ui.applied[-1] == {"prompt": "formy"}


@pytest.mark.asyncio
async def test_multipart_upload_arrives_as_bytes(
    client: TestClient, server: WebRTCServer, web_root: Path
) -> None:
    """A page posts a first frame beside text fields; both must survive."""
    web_ui = StubWebUi(web_root)
    server.serve_web_ui(web_ui)

    response = await client.post(
        "/api/session/input",
        data={
            "prompt": "with image",
            "text_events": json.dumps([{"event_id": "storm", "prompt": "wind"}]),
            "image": b"\x89PNGfake",
        },
    )

    assert response.status == 200
    applied = web_ui.applied[-1]
    assert applied["image"] == b"\x89PNGfake"
    assert isinstance(applied["text_events"], str)
    assert applied["image_content_type"].startswith("application/")


@pytest.mark.asyncio
async def test_application_rejection_is_a_bad_request(
    client: TestClient, server: WebRTCServer, web_root: Path
) -> None:
    """A ValueError is the application's way of saying 400, not 500."""
    server.serve_web_ui(StubWebUi(web_root))

    assert (await client.post("/api/session/input", json={"prompt": ""})).status == 400


@pytest.mark.asyncio
async def test_malformed_json_is_a_bad_request(
    client: TestClient, server: WebRTCServer, web_root: Path
) -> None:
    server.serve_web_ui(StubWebUi(web_root))

    response = await client.post(
        "/api/session/input",
        data="{not json",
        headers={"Content-Type": "application/json"},
    )

    assert response.status == 400


@pytest.mark.asyncio
async def test_non_object_json_is_a_bad_request(
    client: TestClient, server: WebRTCServer, web_root: Path
) -> None:
    server.serve_web_ui(StubWebUi(web_root))

    response = await client.post("/api/session/input", json=["not", "an", "object"])

    assert response.status == 400


def test_a_closed_server_refuses_a_web_ui(
    server: WebRTCServer, web_root: Path
) -> None:
    server._closed = True

    with pytest.raises(RuntimeError, match="closed"):
        server.serve_web_ui(StubWebUi(web_root))
