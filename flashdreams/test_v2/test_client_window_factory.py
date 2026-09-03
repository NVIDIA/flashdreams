# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for the v2 client-window factory.

A mode owns the arguments it takes and what it says about where the run went,
so what is covered here is each mode answering for itself.
"""

import argparse
from pathlib import Path

import pytest
from flashdreams.runtime_v2.cli import _parser, _validate_artifact_paths
from flashdreams.runtime_v2.client_window_factory import (
    add_client_window_arguments,
    client_window_mode,
    create_client_window,
)
from flashdreams.runtime_v2.mp4_client_window import Mp4ClientWindow
from flashdreams.runtime_v2.native_window_client_window import (
    NativeWindowClientWindow,
)

pytestmark = pytest.mark.ci_cpu


def _parsed(arguments: list[str]) -> argparse.Namespace:
    """Parse arguments the way a command offering every mode would."""
    parser = argparse.ArgumentParser()
    add_client_window_arguments(parser)
    return parser.parse_args(arguments)


def test_a_run_goes_to_a_file_unless_it_says_otherwise(tmp_path: Path) -> None:
    parsed = _parsed(["--output-path", str(tmp_path / "clip.mp4")])

    window = create_client_window(parsed)

    assert parsed.mode == "mp4"
    assert isinstance(window, Mp4ClientWindow)


def test_a_file_run_with_nowhere_to_write_says_so() -> None:
    with pytest.raises(ValueError, match="--output-path is required"):
        create_client_window(_parsed([]))


def test_the_file_is_named_once_there_is_something_in_it(tmp_path: Path) -> None:
    """A run says nothing before the file is written, and the path after."""
    mode = client_window_mode("mp4")
    window = mode.create(_parsed(["--output-path", str(tmp_path / "clip.mp4")]))

    assert mode.starting(window) is None
    assert mode.finished(window) == str(tmp_path / "clip.mp4")


def test_an_unsupported_mode_is_refused() -> None:
    with pytest.raises(ValueError, match="Unsupported"):
        create_client_window(argparse.Namespace(mode="local"))


def test_a_native_window_mode_is_lazy_and_keeps_its_title() -> None:
    window = create_client_window(
        _parsed(["--mode", "native-window", "--window-title", "World model"])
    )

    assert isinstance(window, NativeWindowClientWindow)
    assert window.title == "World model"


@pytest.mark.parametrize(
    ("first_flag", "second_flag"),
    [
        ("--profile-path", "--stats-path"),
        ("--profile-path", "--output-path"),
        ("--stats-path", "--output-path"),
    ],
)
def test_output_artifacts_require_distinct_paths(
    tmp_path: Path,
    first_flag: str,
    second_flag: str,
) -> None:
    shared = tmp_path / "run-artifact"
    arguments = [
        "demo",
        "--output-path",
        str(tmp_path / "output.mp4"),
        first_flag,
        str(shared),
        second_flag,
        str(shared),
    ]
    parsed = _parser().parse_args(arguments)

    with pytest.raises(ValueError, match="must use different paths"):
        _validate_artifact_paths(parsed)


def test_output_artifacts_accept_distinct_paths(tmp_path: Path) -> None:
    parsed = _parser().parse_args(
        [
            "demo",
            "--profile-path",
            str(tmp_path / "profile.jsonl"),
            "--stats-path",
            str(tmp_path / "stats.json"),
            "--output-path",
            str(tmp_path / "output.mp4"),
        ]
    )

    _validate_artifact_paths(parsed)


class TestWebRTC:
    """Modes that serve a client, which need the serving stack installed."""

    @pytest.fixture(autouse=True)
    def _serving_installed(self) -> None:
        pytest.importorskip("aiohttp")
        pytest.importorskip("aiortc")

    def test_a_browser_run_is_told_where_to_connect(self) -> None:
        """The port can be one the operating system chose, so nobody can guess it."""
        from flashdreams.runtime_v2.webrtc_client_window import WebRTCClientWindow

        mode = client_window_mode("webrtc")
        window = mode.create(_parsed(["--mode", "webrtc"]))
        try:
            assert isinstance(window, WebRTCClientWindow)
            assert window.metrics_snapshot()["webrtc_sender_queue_capacity_count"] == 2
            assert mode.starting(window) == f"Open {window.server.url} in a browser."
        finally:
            window.close()
