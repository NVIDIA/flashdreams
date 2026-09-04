# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Optional application hook for serving an application-owned browser UI.

The v2 WebRTC server ships one minimal viewer for every application. An
application that wants its own richer page -- a scene picker, an event panel,
a heads-up display -- implements :class:`IWebUiProvider` and gets its files
served alongside that viewer, plus three endpoints the page can call.

The serving layer stays generic: it copies :meth:`IWebUiProvider.initial_scene`
into a JSON response and hands :meth:`IWebUiProvider.apply_session_input` the
decoded request body without inspecting either. What a "scene" is, and which
inputs a page may change, belong entirely to the application.

Applications that do not implement this protocol are unaffected: the server
registers none of these routes and serves its built-in viewer as before.
"""

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class IWebUiProvider(Protocol):
    """An application that serves its own browser UI."""

    def web_root(self) -> Path:
        """Return the directory holding the application's web assets.

        Its ``index.html`` is served at ``/request_session``; the remaining
        files are served by name. Paths are resolved inside this directory
        only, so a request cannot escape it.
        """
        ...

    def initial_scene(self) -> Mapping[str, Any]:
        """Return what the page needs to render before any frame arrives.

        Returned verbatim as JSON from ``GET /api/session/initial_scene`` and
        again from ``POST /api/session/input``, so a page can render the
        result of its own change without a second request.
        """
        ...

    def first_frame(self) -> tuple[bytes, str] | None:
        """Return the session's first frame as ``(data, content_type)``.

        ``None`` when the session has no first frame to show yet, which the
        server reports as ``404`` rather than an error.
        """
        ...

    def apply_session_input(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        """Apply one page-submitted change and return the resulting scene.

        Args:
            payload: Decoded request body. Multipart uploads arrive with file
                parts as ``bytes`` and every other part as ``str``.

        Returns:
            The scene as :meth:`initial_scene` would report it afterwards.

        Raises:
            ValueError: The payload is not a change this application accepts.
                The server reports it as ``400`` with the message.
        """
        ...


__all__ = ["IWebUiProvider"]
