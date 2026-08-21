# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Application session abstract interface."""

from abc import ABC, abstractmethod

from flashdreams.runtime_v2.session_desc import SessionDesc
from flashdreams.runtime_v2.step_result import StepResult
from flashdreams.runtime_v2.user_input_events import UserInputEvents


class ISession(ABC):
    """One run of an application, and the state that run builds up.

    Created by :meth:`IApplication.create_session`. Holds the KV cache, game
    state and anything else that must not carry into another run; anything shared
    between runs belongs to the application. A session runs no loop of its own:
    the runtime calls :meth:`step` once per step and decides when to stop.

    Note:
        Call order is :meth:`init`, then :meth:`step` per step from index zero.
        :meth:`reset` can happen mid-run, after which the index starts again from
        zero. :meth:`close` ends it.
    """

    @abstractmethod
    def init(self) -> None:
        """Load the model and anything else this run needs.

        Must not do client I/O, since this can run before a client connects.
        """
        ...

    @property
    @abstractmethod
    def session_desc(self) -> SessionDesc:
        """Return the description used to create this session.

        The runtime reads it before :meth:`init` runs, since it opens the client
        window with it.
        """
        ...

    @abstractmethod
    def step(self, step_index: int, events: UserInputEvents) -> StepResult:
        """Produce one result for ``step_index``.

        Args:
            step_index: Zero-based index of the step to produce.
            events: User input events collected since the previous step.

        Returns:
            Result carrying ``step_index``.
        """
        ...

    def step_ui(self, events: UserInputEvents) -> None:
        """React to input at the UI rate, faster than :meth:`step`.

        This exists so UI work is not held up by generation. ``run_session``
        calls it on its I/O thread every tick, while :meth:`step` may still be
        running, so a session implementing both must guard what they share. The
        same events reach :meth:`step` in its next batch, so a session can
        respond here and generate from them later.

        It cannot produce output yet, so today it can only update state. The
        default does nothing, and a session with nothing to do at this rate
        leaves it alone.

        Args:
            events: User input events collected since the previous tick.
        """
        return

    def is_finished(self) -> bool:
        """Report whether this session has generated everything it has to.

        ``run_session`` asks before every step and ends the run when the answer
        is yes. A session that knows its own length says so here, rather than
        the caller counting steps on its behalf: a fixed rollout knows how many
        blocks it has, where a caller only knows what it was told.

        The default never finishes, which is what an interactive session wants:
        a run like that ends when its client goes away.

        A reset is applied before this is asked, so a session starting over is
        asked about the run it is starting, not the one it just finished.

        Returns:
            Whether the run should end rather than take another step.
        """
        return False

    def reset(self) -> None:
        """Reset per-generation state so the session can run again.

        ``run_session`` calls this when a window reports a reset event, and then
        steps from index zero again. A session that cannot start over should say
        so rather than half-reset.

        The next :meth:`step` still receives the batch the reset arrived in,
        including the events before it, so a held key stays held across the
        restart. Ignore the older events here if this session must not inherit
        them.

        Raises:
            NotImplementedError: The session does not support reuse.
        """
        raise NotImplementedError(f"{type(self).__name__} does not support reset.")

    def close(self) -> None:
        """Release whatever this run holds.

        Runs even when :meth:`init` raised, so an implementation releases what it
        managed to acquire and tolerates being called on a session that never
        finished starting. Not abstract, and does nothing by default, so a session
        with nothing to release does not implement it.
        """
        return
