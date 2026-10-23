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

"""CPU checks for clickable client-window regions."""

import shutil
import subprocess
from dataclasses import FrozenInstanceError
from pathlib import Path
from unittest.mock import Mock

import pytest

from flashdreams.api_v2.client_window import ClientWindowLink, IClientWindow
from flashdreams.api_v2.loop import UILoopRequests
from flashdreams.runtime_v2.blit_model_output_to_screen_loop import (
    BlitModelOutputToScreenLoop,
)
from flashdreams.runtime_v2.session_runner import _apply_window_requests

pytestmark = pytest.mark.ci_cpu


def test_link_requests_replace_and_clear_regions() -> None:
    link = ClientWindowLink("Isaac Teleop", "https://localhost:48322/", (0, 0, 1, 1))
    with pytest.raises(FrozenInstanceError):
        link.url = "https://elsewhere.test/"  # ty: ignore[invalid-assignment]
    requests = UILoopRequests()
    loop = BlitModelOutputToScreenLoop[object]()
    loop._ui_loop_requests = requests
    window = Mock(spec=IClientWindow)
    loop.request_links((link,))
    _apply_window_requests(window, requests)
    window.request_links.assert_called_once_with((link,))
    loop.request_links(())
    _apply_window_requests(window, requests)
    window.request_links.assert_called_with(())
    for invalid in ([link], ("link",), None):
        with pytest.raises(TypeError):
            loop.request_links(invalid)  # ty: ignore[invalid-argument-type]
    with pytest.raises(ValueError):
        loop.request_links((link,) * 17)


@pytest.mark.parametrize(
    "url",
    [
        "javascript:alert(1)",
        "file:///tmp/file",
        "//example.test/",
        "https:///path",
        "https://user:pass@example.test/",
        "https://@example.test/",
        "https://example.test/\npath",
        "https://example.test/\x7fpath",
        "https://example.test/\x85path",
        " https://example.test/",
        "https://example.test\\@evil.test/",
        "https://bad host/",
        "https://example.test:99999/",
        "https://example.test/" + "x" * 4096,
    ],
)
def test_link_rejects_invalid_destination(url: str) -> None:
    with pytest.raises(ValueError):
        ClientWindowLink("Open", url, (0, 0, 1, 1))


@pytest.mark.parametrize(
    "bounds",
    [
        (0, 0, 0, 1),
        (0, 1, 1, 0),
        (-1, 0, 1, 1),
        (0, 0, 2, 1),
        (0, 0, float("nan"), 1),
        (0, 0, float("inf"), 1),
    ],
)
def test_link_rejects_invalid_rectangle(bounds) -> None:
    with pytest.raises(ValueError):
        ClientWindowLink("Open", "https://example.test/", bounds)


@pytest.mark.parametrize(
    "bounds", [[0, 0, 1, 1], (0, 1), (False, 0, 1, 1), ("0", 0, 1, 1)]
)
def test_link_requires_immutable_numeric_rectangle(bounds) -> None:
    with pytest.raises(TypeError):
        ClientWindowLink("Open", "https://example.test/", bounds)


@pytest.mark.parametrize("label", ["", " ", "x" * 257])
def test_link_rejects_invalid_label(label: str) -> None:
    with pytest.raises(ValueError):
        ClientWindowLink(label, "http://127.0.0.1/", (0, 0, 1, 1))


def test_browser_link_commands() -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required for the browser link check.")
    subprocess.run(
        [node, str(Path(__file__).with_name("client_window_links_check.cjs"))],
        check=True,
        timeout=15,
    )
