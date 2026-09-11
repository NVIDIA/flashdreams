# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Step admission and input synchronization across a model process mesh."""

from datetime import timedelta

import torch
import torch.distributed as dist

from flashdreams.core.distributed.parallel import ParallelContext
from flashdreams.runtime_v2.user_input_events import UserInputEvents

AGREEMENT_TIMEOUT = timedelta(minutes=5)
"""Maximum wait for a rank at a runtime boundary; model collectives have their own timeout."""


class StepAgreement:
    """Two admission checks per step, on the model thread of every rank.

    The first checks cancellation before broadcasting input. The second checks
    preparation and cancellation before committing to model execution. Once
    admitted, every rank executes the step even if its UI requests a stop.
    Cleanup performs no collective: an exception must reach the process
    supervisor so it can terminate peers blocked inside model collectives.
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

    def close(self) -> None:
        """Release the local control group without a shutdown collective."""
        dist.destroy_process_group(self._group)
