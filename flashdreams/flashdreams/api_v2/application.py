# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Application abstract interface."""

from abc import ABC, abstractmethod
from collections.abc import Sequence

from flashdreams.runtime_v2.session_desc import SessionDesc

from .session import ISession


class IApplication(ABC):
    """One application, for as long as the process runs.

    Parses its own arguments and holds whatever its sessions share, such as a
    checkpoint or a compiled pipeline. It outlives every session it creates, so
    that shared state is loaded once here and released in :meth:`close`.

    An application module implements this and :class:`ISession`. The runtime
    creates everything else and passes it in.
    """

    @abstractmethod
    def init(self, commandline_args: Sequence[str]) -> None:
        """Parse application arguments and validate startup state."""
        ...

    def default_session_desc(self) -> SessionDesc | None:
        """Return this initialized application's default session description.

        The application owns its model's output requirements and defaults, such
        as its layout, preferred dimensions, and frame rate. The runtime asks
        after :meth:`init`, then applies the caller's explicit requests to this
        default before calling :meth:`create_session`. That method accepts the
        resolved description or rejects it; it does not silently change the
        stream the runtime already asked the client window to prepare for.

        Returns:
            The default session description, or ``None`` when the application
            has no output requirements and accepts the runtime's defaults.
        """
        return None

    @abstractmethod
    def create_session(self, session_desc: SessionDesc) -> ISession:
        """Create one isolated, uninitialized session for ``session_desc``.

        Args:
            session_desc: Session the runtime is asking for.

        Returns:
            A session that produces ``session_desc``.

        Raises:
            ValueError: The application cannot honour ``session_desc``.
        """
        ...

    def close(self) -> None:
        """Release whatever the application holds.

        Not abstract, and does nothing by default, so an application with nothing
        to release does not implement it.
        """
        return
