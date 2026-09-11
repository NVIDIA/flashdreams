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

"""Rank-zero decisions for starting, resetting, and stopping model steps."""

from __future__ import annotations

from datetime import timedelta
from enum import IntEnum

import torch
import torch.distributed as dist
from loguru import logger

from flashdreams.core.distributed.parallel import ParallelContext

AGREEMENT_TIMEOUT = timedelta(days=1)
"""How long a rank waits to hear what rank 0 decided.

Measured in days rather than the default minutes because the wait is for a
person. Between two chunks of a served stream every rank but rank 0 is parked in
this broadcast, and a session with nobody looking at it is idle for as long as
nobody is looking at it. The timeout is a deadlock guard, not a latency budget.
"""


class StepAction(IntEnum):
    """What rank 0 decided the mesh does next.

    Ordered by how disruptive it is, which is only a convention here -- rank 0
    is the one deciding, so nothing combines two of these.
    """

    CONTINUE = 0
    """Take another step."""

    RESET = 1
    """Start the clip over, because a client asked to. Rank 0 has already reset
    by the time it says this: the runtime calls ``reset`` on the generation
    change, and the other ranks never see one because a client's events reach
    rank 0's event buffer alone."""

    STOP = 2
    """Generate no more. The clip ran out, a client closed the window, a control
    source ran dry, or the step raised."""


class StepAgreement:
    """One decision per model step, taken by rank 0 and heard by every rank.

    **The invariant is a count.** Every rank calls :meth:`agree` exactly once per
    iteration of the model loop, and rank 0 calls :meth:`announce_stop` for the
    iteration it leaves on rather than finishes. A broadcast pairs by call order
    and carries no sequence number to check that with -- deliberately, since both
    sides step any counter on every call and it would always agree -- so the
    invariant is held by keeping the call sites few and named. They are
    :meth:`~flashdreams.api_v2.loop.IModelLoop.is_finished`, which the
    runtime calls once per iteration on every rank before anything else in it,
    and :meth:`~flashdreams.api_v2.loop.IModelLoop.close`, which runs once
    on every path out of the loop. :meth:`announce_stop` is the one call that may
    have nobody to pair with, and says there why that is safe.

    One thing this cannot rescue: a rank that fails *inside* a step leaves its
    peers blocked in a NCCL collective rather than in this one, where no message
    over a separate group can reach them. That is NCCL's watchdog to report, and
    it does. What is fixed here is the ordinary end of a run, which had no
    watchdog at all.

    Not thread-safe, and does not need to be: every call comes from the model
    thread, and the one exit that does not -- a session shut down before its
    model thread started -- has no model thread to race with.
    """

    def __init__(
        self,
        ctx: ParallelContext,
        *,
        timeout: timedelta = AGREEMENT_TIMEOUT,
    ) -> None:
        """
        Args:
            ctx: the mesh this rank sits on. A single-rank context makes every
                method here a no-op.
            timeout: how long a rank waits to hear rank 0's decision.
        """
        self._ctx = ctx
        self._timeout = timeout
        self._stopped = False
        # Every rank must create the group during session initialization, before
        # an early close can send rank zero and workers down different paths.
        self._group: dist.ProcessGroup | None = (
            dist.new_group(backend="gloo", timeout=self._timeout)
            if self.distributed
            else None
        )

    @property
    def distributed(self) -> bool:
        """Whether there is anyone to agree with.

        False for one rank, and false for a mesh nothing initialized -- which is
        what a hand-built multi-rank context in a test is, and what makes one
        testable without spawning processes.
        """
        return (
            self._ctx.world_size > 1 and dist.is_available() and dist.is_initialized()
        )

    @property
    def stopped(self) -> bool:
        """Whether a stop has been agreed, so nothing more will be."""
        return self._stopped

    def agree(self, wanted: StepAction) -> StepAction:
        """Return what the mesh does next, which is what rank 0 wants.

        Args:
            wanted: what this rank would do. Read on rank 0 and ignored
                everywhere else, so a rank that is not presenting may pass
                whatever it likes -- :data:`StepAction.CONTINUE` is what the
                model loop passes, since a follower has no opinion.

        Returns:
            Rank 0's action, on every rank.
        """
        action = self._broadcast(wanted) if self.distributed else wanted
        self._stopped = self._stopped or action is StepAction.STOP
        return action

    def announce_stop(self) -> None:
        """Say the run is over, for an exit that :meth:`agree` never reached.

        Rank 0 leaves the model loop from more places than it decides from: the
        runtime checks its shutdown event both before and after the call that
        :meth:`agree` rides on, and a step that raises leaves from inside the
        step. Every one of those paths ends in the loop's ``close``, so that is
        where the ranks still waiting get told.

        Does nothing once a stop has been agreed, and nothing at all on a rank
        that is not rank 0 -- it is not the one being listened to, and a second
        broadcast from it would be one nobody is expecting.

        **This is the one message that may go unheard, so it is the one that does
        not raise.** ``run_session(steps=N)`` bounds every rank alike, so they can
        all leave the loop having agreed nothing and this arrives at a group the
        other ranks have already left. Gloo reports that as a closed connection
        rather than by waiting -- which is the behaviour worth having, since the
        alternative would hold a finished run open for the group's whole timeout
        -- but it must not turn a clean run into a failed one, and the only rank
        it could have helped has already gone.
        """
        if self._stopped or not self._ctx.is_main:
            return
        # Set before the send, so a send that fails is not one to retry.
        self._stopped = True
        if not self.distributed:
            return
        try:
            self._broadcast(StepAction.STOP)
        except RuntimeError as error:
            logger.info(
                "No rank was waiting to be told the run had "
                "ended, so the stop went unsent ({}). Expected when every rank "
                "left on the same bound; a symptom if one of them is stuck.",
                error,
            )

    def _broadcast(self, wanted: StepAction) -> StepAction:
        """Send rank 0's action, or receive it, over the control group."""
        payload = torch.tensor([int(wanted)], dtype=torch.int64)
        dist.broadcast(payload, src=0, group=self._control_group())
        return StepAction(int(payload.item()))

    def _control_group(self) -> dist.ProcessGroup:
        """The CPU group decisions travel over, built in ``__init__``.

        Not created here on demand: see the note in ``__init__`` for the path
        where "on demand" means "on one rank only", and hangs it for a day.
        """
        assert self._group is not None, "a distributed agreement has a group"
        return self._group


__all__ = ["AGREEMENT_TIMEOUT", "StepAction", "StepAgreement"]
