# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The page's control protocol, which has to match what the runtime accepts.

The page was written for the v1 server, whose data channel carried key
presses, text events, heartbeats and disconnects as one message family. The v2
channel takes device events only and rejects the rest, so ``toRuntimeEvent``
translates between them. Getting that mapping wrong is not a visible failure:
an unknown type is answered with an error the page logs and ignores, and
sending the runtime's ``close`` ends the whole run, which reads as the server
exiting on its own.

The source assertions run everywhere. When node is available the translation
is executed as well, which is the stronger check.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.ci_cpu

_PAGE = Path(__file__).resolve().parents[1] / "apps" / "cam2v" / "web" / "request_session.js"

_EXPECTED = [
    ({"type": "action", "action": {"event": "keydown", "key": "w"}},
     {"type": "keyboard", "key": "w", "pressed": True}),
    ({"type": "action", "action": {"event": "keyup", "key": "w"}},
     {"type": "keyboard", "key": "w", "pressed": False}),
    # The v2 runtime generates continuously, so there is nothing to ask for.
    ({"type": "action", "action": {"event": "step"}}, None),
    # It holds the connection open without a keepalive.
    ({"type": "heartbeat", "t": 1}, None),
    # The one that matters: "close" would end the run, so a tab refresh must
    # not be translated into one.
    ({"type": "disconnect"}, None),
    ({"type": "reset"}, {"type": "reset"}),
]


def test_the_page_exists_where_the_server_serves_it() -> None:
    assert _PAGE.is_file(), f"{_PAGE} is served at /request_session.js"


def test_the_page_never_sends_a_close_event() -> None:
    """A close event ends the application run, not just one peer connection."""
    source = _PAGE.read_text(encoding="utf-8")

    sends = [
        line.strip()
        for line in source.splitlines()
        if "controlChannel.send" in line or 'type: "close"' in line
    ]

    assert not [line for line in sends if '"close"' in line], (
        "the page sends the runtime's close event, which stops the session "
        f"with no replacement and exits the process: {sends}"
    )


def test_disconnect_is_dropped_rather_than_translated() -> None:
    source = _PAGE.read_text(encoding="utf-8")

    assert 'if (type === "disconnect") {' in source
    disconnect_branch = source.split('if (type === "disconnect") {', 1)[1][:400]
    assert "return null" in disconnect_branch


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
@pytest.mark.parametrize(("payload", "expected"), _EXPECTED)
def test_translation(payload: dict, expected: dict | None) -> None:
    """Execute the page's own translation against each message it can send."""
    source = _PAGE.read_text(encoding="utf-8")
    start = source.index("function toRuntimeEvent")
    end = source.index("// Text events are session state")
    script = (
        source[start:end]
        + "\nconsole.log(JSON.stringify("
        + f"toRuntimeEvent({json.dumps(payload)})"
        + " ?? null))"
    )

    result = subprocess.run(
        [shutil.which("node") or "node", "-e", script],
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )

    assert json.loads(result.stdout.strip()) == expected
