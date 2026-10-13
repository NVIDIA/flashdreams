# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Create v2 client windows from runtime arguments.

A mode is one way to watch a run: an MP4 file, a browser, whatever comes after
them. Each mode owns its arguments and user-facing messages. Integrations add
modes through the ``flashdreams.client_windows_v2`` entry-point group.
"""

import argparse
import logging
from abc import ABC, abstractmethod
from importlib.metadata import EntryPoint, entry_points
from pathlib import Path
from typing import TYPE_CHECKING, cast

from flashdreams.api_v2.client_window import IClientWindow
from flashdreams.runtime_v2.mp4_client_window import Mp4ClientWindow
from flashdreams.runtime_v2.null_client_window import NullClientWindow

if TYPE_CHECKING:
    from flashdreams.runtime_v2.webrtc_client_window import WebRTCClientWindow

_LOGGER = logging.getLogger(__name__)


class ClientWindowMode(ABC):
    """One way to present a run."""

    name: str
    """What ``--mode`` calls this."""

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        """Add the arguments this mode takes and no other does."""

    def check_arguments(self, parsed_args: argparse.Namespace) -> None:
        """Report a usage error before anything is built.

        A command calls this while it can still print its usage, rather than
        after loading a model to find out the run had nowhere to go.

        Raises:
            ValueError: The arguments do not describe a window this can create.
        """
        del parsed_args

    @abstractmethod
    def create(self, parsed_args: argparse.Namespace) -> IClientWindow:
        """Create the window.

        Raises:
            ValueError: Whatever :meth:`check_arguments` reports.
        """

    def starting(self, client_window: IClientWindow) -> str | None:
        """Return what to tell the user before the run, such as where to watch."""
        del client_window
        return None

    def finished(self, client_window: IClientWindow) -> str | None:
        """Return what to tell the user after a run that generated everything."""
        del client_window
        return None


class _NullMode(ClientWindowMode):
    """Discard output from an internal preload run."""

    name = "null"

    def create(self, parsed_args: argparse.Namespace) -> IClientWindow:
        del parsed_args
        return NullClientWindow()


CLIENT_WINDOW_ENTRY_POINT_GROUP = "flashdreams.client_windows_v2"
"""Entry-point group whose values expose client-window modes or factories."""


class _Mp4Mode(ClientWindowMode):
    """Write the run to a file, with nobody watching it happen."""

    name = "mp4"

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--output-path", type=Path, help="MP4 file to write. Required for mp4."
        )

    def check_arguments(self, parsed_args: argparse.Namespace) -> None:
        if parsed_args.output_path is None:
            raise ValueError("--output-path is required when writing an MP4.")

    def create(self, parsed_args: argparse.Namespace) -> IClientWindow:
        self.check_arguments(parsed_args)
        return Mp4ClientWindow(parsed_args.output_path)

    def finished(self, client_window: IClientWindow) -> str | None:
        """Return the file, now that there is something in it to watch."""
        return str(cast(Mp4ClientWindow, client_window).path)


class _WebRTCMode(ClientWindowMode):
    """Stream the run to a browser."""

    name = "webrtc"

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--host", default="127.0.0.1", help="Interface to serve on."
        )
        parser.add_argument("--port", type=int, default=0, help="Port to serve on.")

    def create(self, parsed_args: argparse.Namespace) -> IClientWindow:
        # Imported here so a run writing a file needs none of the serving stack.
        from flashdreams.runtime_v2.webrtc_client_window import WebRTCClientWindow

        return WebRTCClientWindow(host=parsed_args.host, port=parsed_args.port)

    def starting(self, client_window: IClientWindow) -> str | None:
        """Return where to connect, which nobody can guess when the port is free."""
        server = cast("WebRTCClientWindow", client_window).server
        return f"Open {server.url} in a browser."


class _NativeWindowMode(ClientWindowMode):
    """Present the run in a GPU-backed local OS window."""

    name = "native-window"

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--window-title",
            default="FlashDreams",
            help="Native window title. Default: %(default)s.",
        )

    def create(self, parsed_args: argparse.Namespace) -> IClientWindow:
        # Imported here so file and browser runs need none of the SlangPy stack.
        from flashdreams.runtime_v2.native_window_client_window import (
            NativeWindowClientWindow,
        )

        return NativeWindowClientWindow(title=parsed_args.window_title)


_BUILTIN_MODES: tuple[ClientWindowMode, ...] = (
    _Mp4Mode(),
    _WebRTCMode(),
    _NativeWindowMode(),
    _NullMode(),
)
"""Built-in modes a run can be presented through."""


def client_window_modes() -> tuple[ClientWindowMode, ...]:
    """Return built-in and installed client-window modes in a stable order.

    Plugin entry points may expose a :class:`ClientWindowMode` directly or a
    zero-argument callable returning one. Their entry-point names are the
    public ``--mode`` values and therefore must match the returned mode names.
    Import and factory failures are logged and skipped; invalid registrations
    still fail validation.

    Raises:
        TypeError: A plugin does not expose a client-window mode.
        ValueError: A mode name is invalid, mismatched, or already registered.
    """
    modes = list(_BUILTIN_MODES)
    origins = {mode.name: "FlashDreams" for mode in modes}
    discovered = sorted(
        entry_points(group=CLIENT_WINDOW_ENTRY_POINT_GROUP),
        key=lambda item: (item.name, item.value),
    )
    for entry_point in discovered:
        existing = origins.get(entry_point.name)
        if existing is not None:
            raise ValueError(
                f"Client-window mode {entry_point.name!r} from "
                f"{entry_point.value!r} is already registered by {existing}."
            )
        origins[entry_point.name] = entry_point.value
        mode = _mode_from_entry_point(entry_point)
        if mode is None:
            continue
        if not isinstance(mode.name, str) or not mode.name.strip():
            raise ValueError(
                f"Client-window mode from {entry_point.value!r} must have a "
                "non-empty string name."
            )
        if mode.name != entry_point.name:
            raise ValueError(
                f"Client-window entry point {entry_point.name!r} loaded mode "
                f"{mode.name!r}; the names must match."
            )
        modes.append(mode)
    return tuple(modes)


def _mode_from_entry_point(entry_point: EntryPoint) -> ClientWindowMode | None:
    """Load and validate one installed client-window mode."""
    try:
        value = entry_point.load()
        mode = (
            value()
            if callable(value) and not isinstance(value, ClientWindowMode)
            else value
        )
    except Exception as error:
        _LOGGER.warning(
            "Skipping unavailable client-window mode %r from %r: %s: %s",
            entry_point.name,
            entry_point.value,
            type(error).__name__,
            error,
        )
        return None
    if not isinstance(mode, ClientWindowMode):
        raise TypeError(
            f"Client-window entry point {entry_point.value!r} returned "
            f"{type(mode).__name__}; expected a ClientWindowMode."
        )
    return mode


def add_client_window_arguments(parser: argparse.ArgumentParser) -> None:
    """Add ``--mode`` and the arguments each mode takes.

    Every mode's arguments are added, since which mode was asked for is not
    known until they are parsed. A mode reads its own and no others.
    The parsed namespace retains these mode instances for window creation.
    An omitted ``--mode`` is left as ``None`` for the caller to resolve.

    Args:
        parser: Parser receiving the shared and mode-specific arguments.
    """
    modes = client_window_modes()
    parser.set_defaults(_client_window_modes=modes)
    parser.add_argument(
        "--mode",
        choices=tuple(mode.name for mode in modes),
        help="Where the run goes. Default: the application's preference, otherwise mp4.",
    )
    parser.add_argument(
        "--stats-path",
        type=Path,
        default=None,
        help=(
            "JSON file to record model-step measurements in, for a benchmark "
            "to read. This also enables synchronized per-stage pipeline "
            "profiling. If a run replaces its session, the file contains the "
            "final session."
        ),
    )
    for mode in modes:
        mode.add_arguments(parser)


def client_window_mode(
    name: str, *, modes: tuple[ClientWindowMode, ...] | None = None
) -> ClientWindowMode:
    """Return the mode of that name.

    Args:
        name: Public client-window mode name.
        modes: Modes already discovered for this invocation; ``None`` performs
            fresh discovery.

    Raises:
        ValueError: Nothing here presents a run that way.
    """
    for mode in client_window_modes() if modes is None else modes:
        if mode.name == name:
            return mode
    raise ValueError(f"Unsupported client-window mode: {name!r}.")


def create_client_window(parsed_args: argparse.Namespace) -> IClientWindow:
    """Create the client window selected by the presentation mode.

    Args:
        parsed_args: Runtime arguments. Mode-specific fields are read only by
            the selected mode.

    Returns:
        Client window for the selected mode.

    Raises:
        ValueError: ``mode`` is unsupported, or its arguments are incomplete.
    """
    return client_window_mode(
        parsed_args.mode, modes=getattr(parsed_args, "_client_window_modes", None)
    ).create(parsed_args)
