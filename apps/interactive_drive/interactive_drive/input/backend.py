# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from dataclasses import dataclass
from typing import Protocol

from interactive_drive.types import DriverCommand


@dataclass(frozen=True)
class SampledInput:
    command: DriverCommand
    sample_time: float


class InputBackend(Protocol):
    def sample(self) -> SampledInput:
        """Sample control inputs and return command + sample timestamp."""
        ...
