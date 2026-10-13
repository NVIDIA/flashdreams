# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for the v2 client-window factory.

A mode owns the arguments it takes and what it says about where the run went,
so what is covered here is each mode answering for itself.
"""

import argparse
import asyncio
from pathlib import Path

import pytest

from flashdreams.runtime_v2 import client_window_factory
from flashdreams.runtime_v2.cli import _parser
from flashdreams.runtime_v2.client_window_factory import (
    CLIENT_WINDOW_ENTRY_POINT_GROUP,
    ClientWindowMode,
    add_client_window_arguments,
    client_window_mode,
    create_client_window,
)
from flashdreams.runtime_v2.mp4_client_window import Mp4ClientWindow
from flashdreams.runtime_v2.native_window_client_window import (
    NativeWindowClientWindow,
)
from flashdreams.runtime_v2.null_client_window import NullClientWindow

pytestmark = pytest.mark.ci_cpu


def _parsed(arguments: list[str]) -> argparse.Namespace:
    """Parse arguments the way a command offering every mode would."""
    parser = argparse.ArgumentParser()
    add_client_window_arguments(parser)
    return parser.parse_args(arguments)


class _EntryPoint:
    """Minimal importlib entry point for registry tests."""

    def __init__(
        self,
        name: str,
        loaded: object,
        *,
        value: str = "plugin:mode",
        load_error: BaseException | None = None,
    ) -> None:
        self.name = name
        self.value = value
        self._loaded = loaded
        self._load_error = load_error

    def load(self) -> object:
        if self._load_error is not None:
            raise self._load_error
        return self._loaded


class _PluginMode(ClientWindowMode):
    """Test mode with one plugin-owned CLI argument."""

    def __init__(self, name: str = "plugin-window") -> None:
        self.name = name

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        self._output_action = parser.add_argument("--plugin-output", type=Path)

    def create(self, parsed_args: argparse.Namespace) -> Mp4ClientWindow:
        return Mp4ClientWindow(getattr(parsed_args, self._output_action.dest))


def _installed_modes(
    monkeypatch: pytest.MonkeyPatch, *entry_point: _EntryPoint
) -> None:
    def installed(*, group: str) -> tuple[_EntryPoint, ...]:
        assert group == CLIENT_WINDOW_ENTRY_POINT_GROUP
        return entry_point

    monkeypatch.setattr(client_window_factory, "entry_points", installed)


def test_an_omitted_mode_is_left_for_the_application() -> None:
    assert _parsed([]).mode is None


def test_a_file_run_with_nowhere_to_write_says_so() -> None:
    with pytest.raises(ValueError, match="--output-path is required"):
        create_client_window(_parsed(["--mode", "mp4"]))


def test_the_file_is_named_once_there_is_something_in_it(tmp_path: Path) -> None:
    """A run says nothing before the file is written, and the path after."""
    mode = client_window_mode("mp4")
    window = mode.create(
        _parsed(["--mode", "mp4", "--output-path", str(tmp_path / "clip.mp4")])
    )

    assert mode.starting(window) is None
    assert mode.finished(window) == str(tmp_path / "clip.mp4")


def test_an_unsupported_mode_is_refused() -> None:
    with pytest.raises(ValueError, match="Unsupported"):
        create_client_window(argparse.Namespace(mode="local"))


def test_internal_null_mode_discards_preload_output() -> None:
    assert isinstance(
        client_window_mode("null").create(argparse.Namespace()), NullClientWindow
    )


def test_a_native_window_mode_is_lazy_and_keeps_its_title() -> None:
    window = create_client_window(
        _parsed(["--mode", "native-window", "--window-title", "World model"])
    )

    assert isinstance(window, NativeWindowClientWindow)
    assert window.title == "World model"


def test_an_installed_mode_owns_its_flashdreams_run_v2_arguments(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _installed_modes(
        monkeypatch,
        _EntryPoint("plugin-window", _PluginMode),
    )
    output = tmp_path / "plugin.mp4"

    parsed = _parser().parse_args(
        [
            "test-application",
            "--mode",
            "plugin-window",
            "--plugin-output",
            str(output),
        ]
    )
    window = create_client_window(parsed)

    assert parsed.plugin_output == output
    assert isinstance(window, Mp4ClientWindow)
    assert window.path == output


def test_factory_modes_are_preserved_per_parser(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    constructed: list[_PluginMode] = []

    def factory() -> _PluginMode:
        mode = _PluginMode()
        constructed.append(mode)
        return mode

    _installed_modes(monkeypatch, _EntryPoint("plugin-window", factory))
    first = _parsed(
        ["--mode", "plugin-window", "--plugin-output", str(tmp_path / "first.mp4")]
    )
    second = _parsed(
        ["--mode", "plugin-window", "--plugin-output", str(tmp_path / "second.mp4")]
    )

    first_window = create_client_window(first)
    second_window = create_client_window(second)

    assert isinstance(first_window, Mp4ClientWindow)
    assert isinstance(second_window, Mp4ClientWindow)
    assert first_window.path == tmp_path / "first.mp4"
    assert second_window.path == tmp_path / "second.mp4"
    assert len(constructed) == 2
    assert constructed[0] is not constructed[1]


def test_unavailable_plugin_does_not_block_direct_builtin_lookup(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    _installed_modes(
        monkeypatch,
        _EntryPoint(
            "broken-window", None, load_error=ModuleNotFoundError("missing_sdk")
        ),
    )

    assert client_window_mode("mp4").name == "mp4"
    assert "broken-window" in caplog.text
    assert "plugin:mode" in caplog.text
    assert "missing_sdk" in caplog.text


def test_empty_mode_snapshot_does_not_rediscover(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        client_window_factory,
        "client_window_modes",
        lambda: pytest.fail("Modes were rediscovered"),
    )

    with pytest.raises(ValueError, match="Unsupported"):
        client_window_mode("mp4", modes=())


@pytest.mark.parametrize("during_load", [False, True])
@pytest.mark.parametrize(
    "error", [KeyboardInterrupt(), SystemExit(3), asyncio.CancelledError()]
)
def test_plugin_cancellation_propagates(
    monkeypatch: pytest.MonkeyPatch, during_load: bool, error: BaseException
) -> None:
    def factory() -> ClientWindowMode:
        raise error

    _installed_modes(
        monkeypatch,
        _EntryPoint(
            "plugin-window", factory, load_error=error if during_load else None
        ),
    )

    with pytest.raises(type(error)) as raised:
        _parsed([])

    assert raised.value is error


def test_an_installed_mode_must_have_the_entry_point_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _installed_modes(
        monkeypatch,
        _EntryPoint("plugin-window", _PluginMode("different-name")),
    )

    with pytest.raises(ValueError, match="names must match"):
        _parsed([])


def test_an_installed_mode_must_implement_the_mode_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _installed_modes(monkeypatch, _EntryPoint("plugin-window", lambda: object()))

    with pytest.raises(TypeError, match="expected a ClientWindowMode"):
        _parsed([])


@pytest.mark.parametrize(
    "entry_points",
    [
        (_EntryPoint("mp4", _PluginMode("mp4")),),
        (
            _EntryPoint("plugin-window", _PluginMode()),
            _EntryPoint("plugin-window", _PluginMode(), value="other:mode"),
        ),
        (
            _EntryPoint("plugin-window", None, load_error=ImportError("missing_sdk")),
            _EntryPoint("plugin-window", _PluginMode(), value="other:mode"),
        ),
    ],
    ids=["built-in", "plugin", "unavailable-plugin"],
)
def test_duplicate_mode_names_are_rejected(
    monkeypatch: pytest.MonkeyPatch, entry_points: tuple[_EntryPoint, ...]
) -> None:
    _installed_modes(monkeypatch, *entry_points)

    with pytest.raises(ValueError, match="already registered"):
        _parsed([])


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
