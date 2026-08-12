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

from __future__ import annotations

import argparse
import json
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import as_file, files
from os import PathLike
from pathlib import Path
from socket import socket
from socketserver import BaseServer
from urllib.parse import urlsplit

WEB_DIR_RESOURCE = files("flashdreams.serving.webrtc").joinpath("web")


class MockUIRequestHandler(SimpleHTTPRequestHandler):
    """Serve the static viewer without preloading a model runtime."""

    def __init__(
        self,
        request: socket | tuple[bytes, socket],
        client_address: tuple[str, int],
        server: BaseServer,
        *,
        directory: str | PathLike[str] | None = None,
        model_web_dir: Path | None = None,
    ) -> None:
        self.model_web_dir = model_web_dir
        super().__init__(
            request,
            client_address,
            server,
            directory=directory,
        )

    def _rewrite_path(self) -> bool:
        path = urlsplit(self.path).path
        if path == "/":
            self.send_response(302)
            self.send_header("Location", "/request_session?mock=1")
            self.end_headers()
            return True
        if path == "/request_session":
            self.path = "/request_session.html"
        elif path.startswith("/static/"):
            self.path = "/" + path.removeprefix("/static/")
        return False

    def do_GET(self) -> None:
        if self._serve_ui_config():
            return
        if self._serve_model_asset(head_only=False):
            return
        if self._rewrite_path():
            return
        super().do_GET()

    def do_HEAD(self) -> None:
        if self._serve_ui_config():
            return
        if self._serve_model_asset(head_only=True):
            return
        if self._rewrite_path():
            return
        super().do_HEAD()

    def _serve_ui_config(self) -> bool:
        if urlsplit(self.path).path != "/api/ui/config":
            return False
        ui_config: dict[str, str | None] = {"adapter_module": None}
        if self.model_web_dir is not None:
            if (self.model_web_dir / "adapter.js").is_file():
                ui_config["adapter_module"] = "/model-static/adapter.js?v=model-ui-v2"
            if (self.model_web_dir / "adapter.css").is_file():
                ui_config["model_stylesheet"] = (
                    "/model-static/adapter.css?v=model-ui-v2"
                )
        payload = json.dumps(ui_config).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(payload)
        return True

    def _serve_model_asset(self, *, head_only: bool) -> bool:
        path = urlsplit(self.path).path
        if not path.startswith("/model-static/") or self.model_web_dir is None:
            return False
        relative = Path(path.removeprefix("/model-static/"))
        if relative.is_absolute() or ".." in relative.parts:
            self.send_error(404)
            return True
        original_directory = self.directory
        original_path = self.path
        try:
            self.directory = str(self.model_web_dir)
            self.path = "/" + relative.as_posix()
            if head_only:
                super().do_HEAD()
            else:
                super().do_GET()
        finally:
            self.directory = original_directory
            self.path = original_path
        return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve the shared WebRTC mock UI.")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument(
        "--model-web-dir",
        type=Path,
        default=None,
        help="Optional integration web directory containing adapter.js.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with as_file(WEB_DIR_RESOURCE) as web_dir:
        handler = partial(
            MockUIRequestHandler,
            directory=str(web_dir),
            model_web_dir=args.model_web_dir,
        )
        server = ThreadingHTTPServer((args.host, args.port), handler)
        print(
            f"Serving shared mock UI at http://{args.host}:{args.port}/request_session?mock=1"
        )
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\nStopping mock UI server.")
        finally:
            server.server_close()


if __name__ == "__main__":
    main()
