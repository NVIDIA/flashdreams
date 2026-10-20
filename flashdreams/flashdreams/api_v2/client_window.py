# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Client window abstract interface."""

import math
from abc import ABC
from dataclasses import dataclass
from urllib.parse import urlsplit

from .input_source import InputSource
from .output_sink import OutputSink


@dataclass(frozen=True, slots=True)
class ClientWindowLink:
    """Clickable HTTP link over a normalized rectangle of the output content."""

    label: str
    """Accessible link label, containing one to 256 characters."""

    url: str
    """Absolute HTTP(S) destination without user information or control characters."""

    bounds: tuple[float, float, float, float]
    """Normalized ``(left, top, right, bottom)`` bounds within the output content."""

    def __post_init__(self) -> None:
        """Validate immutable bounds and a safe browser destination.

        Raises:
            TypeError: A field has an unsupported type.
            ValueError: A label, destination, or rectangle is invalid.
        """
        if not isinstance(self.label, str) or not isinstance(self.url, str):
            raise TypeError("Link label and URL must be strings.")
        if not self.label.strip() or len(self.label) > 256:
            raise ValueError("Link label must contain one to 256 characters.")
        if (
            not self.url
            or len(self.url) > 4096
            or any(ord(char) < 32 or 127 <= ord(char) <= 159 for char in self.url)
            or self.url != self.url.strip()
            or "\\" in self.url
        ):
            raise ValueError(
                "Link URL contains invalid characters or exceeds 4096 characters."
            )
        parsed = urlsplit(self.url)
        if (
            parsed.scheme not in ("http", "https")
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or any(char.isspace() for char in parsed.netloc)
        ):
            raise ValueError(
                "Link URL must be absolute HTTP(S) without user information."
            )
        # Accessing port also rejects malformed or out-of-range ports.
        parsed.port
        if (
            not isinstance(self.bounds, tuple)
            or len(self.bounds) != 4
            or any(
                isinstance(value, bool) or not isinstance(value, (int, float))
                for value in self.bounds
            )
        ):
            raise TypeError("Link bounds must be a tuple of four numbers.")
        if not all(0 <= value <= 1 and math.isfinite(value) for value in self.bounds):
            raise ValueError("Link bounds must be finite and within [0, 1].")
        left, top, right, bottom = self.bounds
        if left >= right or top >= bottom:
            raise ValueError("Link bounds must have positive width and height.")


class IClientWindow(InputSource, OutputSink, ABC):
    """Handle application input and output for one client window.

    The runtime opens the window with a session's description, then reads input
    and writes results until the run ends. A window stays open across a session
    reset/replacement. A window may be opened/closed by the runtime during a session
    reset/replacement.

    A window does not describe the output shape. The session does, and the window
    is given that description in :meth:`OutputSink.open`.

    One thread makes every call on a window, so an implementation needs no
    locking except when its backend delivers input from another thread.

    Created by the runtime, never by an application.
    """

    # Optional to implement
    def request_links(self, links: tuple[ClientWindowLink, ...]) -> None:
        """Replace clickable output regions; an empty tuple clears all links."""
        pass

    # Optional to implement
    def request_hide_cursor(self, hide_cursor: bool) -> None:
        """Show or hide the cursor for this client window."""
        pass

    # Optional to implement
    def request_lock_cursor_to_window(self, lock_cursor_to_window: bool) -> None:
        """Release or capture pointer motion for this client window."""
        pass

    # Optional to implement
    def request_new_window_size(self, new_window_size: tuple[int, int]) -> None:
        """Resize this client window without replacing it.

        Backends whose output dimensions cannot change may leave this unimplemented.

        Args:
            new_window_size: Requested ``(width, height)`` in pixels.
        """
        pass
