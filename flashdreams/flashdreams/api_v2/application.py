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

    def session_desc(self) -> SessionDesc | None:
        """Return the description of a session this application would generate.

        A caller has to describe a session before there is one to describe, and
        only the application knows what its model was trained for. Asked before
        :meth:`init`, so describing a session costs nothing.

        Returns:
            The session to create when nobody asks for another, or ``None``,
            the default, from an application that generates whatever it is
            asked for. Its caller describes the session instead.
        """
        return None

    @abstractmethod
    def create_session(self, session_desc: SessionDesc) -> ISession:
        """Create one isolated, uninitialized session for ``session_desc``.

        Args:
            session_desc: Session the runtime is asking for.

        Returns:
            A session for ``session_desc``, resolved to what this application can
            actually produce.

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
