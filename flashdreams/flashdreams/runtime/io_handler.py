# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Input/output boundary used by the application runner."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from flashdreams.runtime.application_runner import ApplicationRunner


@runtime_checkable
class IOHandler(Protocol):
    """Polymorphic user-input and frame-output behavior."""

    input_mode: str
    realtime: bool

    @classmethod
    def from_argv(cls, args: Sequence[str]) -> tuple[IOHandler, list[str]]:
        """Consume handler arguments and return remaining application arguments."""
        ...

    def run(self, runner: ApplicationRunner) -> object:
        """Attach this IO behavior to an application runner."""
        ...


__all__ = ["IOHandler"]
