# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Step admission and session-result synchronization across a process mesh."""

from __future__ import annotations

from datetime import timedelta
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist

from flashdreams.core.distributed.parallel import ParallelContext
from flashdreams.runtime_v2.user_input_events import UserInputEvents

if TYPE_CHECKING:
    from flashdreams.runtime_v2.session_desc import SessionDesc


AGREEMENT_TIMEOUT = timedelta(minutes=5)
"""Maximum wait for a rank at a runtime boundary; model collectives have their own timeout."""


class StepAgreement:
    """Coordinate model-step boundaries and the final session result.

    The model-thread checks cover cancellation, preparation, and input before
    committing every rank to a step. After every model thread stops, the calling
    threads poll rank zero's result once per UI tick. A failed rank bypasses
    result polling, and cleanup performs no collective.
    """

    def __init__(
        self, ctx: ParallelContext, *, timeout: timedelta = AGREEMENT_TIMEOUT
    ) -> None:
        if (
            not dist.is_initialized()
            or dist.get_world_size() != ctx.world_size
            or dist.get_rank() != ctx.rank
        ):
            raise ValueError(
                "the session mesh must match the initialized process world"
            )
        self._ctx = ctx
        # Construct on every rank before session initialization can fail locally.
        self._group = dist.new_group(backend="gloo", timeout=timeout)

    def ready(self, *, stopping: bool, failed: bool = False) -> bool:
        """Admit a step only when every rank is ready; report peer preparation failures."""
        status = torch.tensor([2 if failed else int(stopping)], dtype=torch.int64)
        dist.all_reduce(status, op=dist.ReduceOp.MAX, group=self._group)
        if status.item() == 2:
            raise RuntimeError("A model rank failed while preparing the next step.")
        return status.item() == 0

    def inputs(
        self, events: UserInputEvents, generation: int
    ) -> tuple[UserInputEvents, int]:
        """Broadcast rank zero's trusted, pickleable input batch and reset generation."""
        payload = [(events, generation) if self._ctx.is_main else None]
        dist.broadcast_object_list(payload, src=0, group=self._group)
        result = payload[0]
        if result is None:
            raise RuntimeError("Rank zero did not broadcast step inputs.")
        return result

    def session_result(
        self, next_session: SessionDesc | None, *, stopping: bool
    ) -> tuple[bool, SessionDesc | None]:
        """Broadcast rank zero's continue, terminal, or replacement result."""
        payload = [(stopping, next_session) if self._ctx.is_main else None]
        dist.broadcast_object_list(payload, src=0, group=self._group)
        result = payload[0]
        if result is None:
            raise RuntimeError("Rank zero did not broadcast the session result.")
        return result

    def close(self) -> None:
        """Release the local control group without a shutdown collective."""
        dist.destroy_process_group(self._group)
