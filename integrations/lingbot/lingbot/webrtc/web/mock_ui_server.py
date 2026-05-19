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
from pathlib import Path
from urllib.parse import urlsplit

WEB_DIR = Path(__file__).resolve().parent
REPO_ROOT = WEB_DIR.parents[4]
EXAMPLE_DATA_DIR = REPO_ROOT / "assets/example_data/lingbot_world"
FIRST_FRAME_PATH = EXAMPLE_DATA_DIR / "image.jpg"
PROMPT_PATH = EXAMPLE_DATA_DIR / "prompt.txt"


class MockUIRequestHandler(SimpleHTTPRequestHandler):
    """Serve the static viewer without preloading the Lingbot runtime."""

    def _send_json(self, payload: dict[str, object], *, head_only: bool) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if not head_only:
            self.wfile.write(body)

    def _send_first_frame(self, *, head_only: bool) -> None:
        if not FIRST_FRAME_PATH.exists():
            self.send_error(404, f"Missing first frame: {FIRST_FRAME_PATH}")
            return

        self.send_response(200)
        self.send_header("Content-Type", "image/jpeg")
        self.send_header("Content-Length", str(FIRST_FRAME_PATH.stat().st_size))
        self.end_headers()
        if not head_only:
            with FIRST_FRAME_PATH.open("rb") as handle:
                self.copyfile(handle, self.wfile)

    def _handle_api(self, *, head_only: bool) -> bool:
        path = urlsplit(self.path).path
        if path == "/api/session/initial_scene":
            prompt = PROMPT_PATH.read_text(encoding="utf-8").splitlines()[0].strip()
            self._send_json(
                {
                    "first_frame_url": "/api/session/first_frame",
                    "prompt": prompt,
                    "model": "lingbot-world-fast-flash",
                    "resolution": {"width": 832, "height": 464},
                },
                head_only=head_only,
            )
            return True
        if path == "/api/session/first_frame":
            self._send_first_frame(head_only=head_only)
            return True
        return False

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
        if self._handle_api(head_only=False):
            return
        if self._rewrite_path():
            return
        super().do_GET()

    def do_HEAD(self) -> None:
        if self._handle_api(head_only=True):
            return
        if self._rewrite_path():
            return
        super().do_HEAD()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve the Lingbot mock UI.")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8090)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    handler = partial(MockUIRequestHandler, directory=str(WEB_DIR))
    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(f"Serving mock UI at http://{args.host}:{args.port}/request_session?mock=1")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping mock UI server.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
