# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Protocol for outputs consumed directly from model-step result batches."""

from abc import abstractmethod
from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from flashdreams.runtime_v2.session_desc import SessionDesc
from flashdreams.runtime_v2.step_result import StepResult


@runtime_checkable
class ModelOutputSink(Protocol):
    """Consume complete model-step batches while a session is running.

    Unlike a client window, a model-output sink sees results before UI
    composition and receives the generation that produced them. Generations
    increase when a session resets.
    """

    @abstractmethod
    def open(self, session_desc: SessionDesc) -> None:
        """Prepare to consume model results from ``session_desc``."""
        ...

    @abstractmethod
    def write(self, generation: int, results: Sequence[StepResult]) -> None:
        """Consume one complete model-step result batch."""
        ...

    @abstractmethod
    def close(self, *, commit: bool = True) -> None:
        """Finish the run and release resources.

        Args:
            commit: Whether model generation completed without an execution
                failure. Transactional sinks publish buffered outputs only
                when this is true. Sinks that stream outputs may ignore it.
        """
        ...


__all__ = ["ModelOutputSink"]
