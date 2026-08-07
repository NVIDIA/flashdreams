# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared time-domain value objects."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True, kw_only=True, slots=True)
class TimeWindow:
    """Half-open time window in seconds since session start."""

    start_s: float
    end_s: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.start_s) or not math.isfinite(self.end_s):
            raise ValueError("TimeWindow bounds must be finite seconds.")
        if self.start_s < 0 or self.end_s < 0:
            raise ValueError("TimeWindow bounds must be non-negative.")
        if self.end_s < self.start_s:
            raise ValueError("TimeWindow.end_s must be >= start_s.")

    def contains(self, timestamp_s: float) -> bool:
        """Return whether ``timestamp_s`` falls within this half-open window."""
        return self.start_s <= timestamp_s < self.end_s


__all__ = ["TimeWindow"]
