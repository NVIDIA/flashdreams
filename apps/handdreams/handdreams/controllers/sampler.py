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

"""Nearest-original controller observations on a source-clock action grid."""

from __future__ import annotations

from collections import deque

_HISTORY_SIZE = 16
"""Retain enough samples to bracket a 15 Hz grid from a 120 Hz source."""


class ControlWindowSampler:
    """Bounded controller history for incremental nearest-neighbour sampling.

    Args:
        rate_hz: Spacing of the output grid.
    """

    interval_ns: int
    """Integer source-clock spacing between consecutive control samples."""

    def __init__(self, *, rate_hz: float) -> None:
        if rate_hz <= 0.0:
            raise ValueError("control rate must be positive")
        self.interval_ns = int(1e9 / float(rate_hz))
        self._history: deque[dict] = deque(maxlen=_HISTORY_SIZE)

    def add(self, frame: dict) -> bool:
        """Record an observation, ignoring duplicates and out-of-order samples.

        Returns:
            Whether the observation advanced the history.
        """
        capture_ns = frame.get("client_capture_ns")
        if not isinstance(capture_ns, int):
            raise ValueError("observation requires an integer client_capture_ns")
        if self._history:
            newest = int(self._history[-1]["client_capture_ns"])
            if capture_ns <= newest:
                return False
        self._history.append(frame)
        return True

    def sample_at(self, capture_ns: int) -> dict | None:
        """Select the original observation nearest a source-clock grid point.

        Return ``None`` outside the retained history. Timestamps and button/pose
        values remain untouched; equidistant samples select the newer one.
        """
        if not self._history:
            return None
        oldest = int(self._history[0]["client_capture_ns"])
        newest = int(self._history[-1]["client_capture_ns"])
        if not oldest <= capture_ns <= newest:
            return None
        return min(
            reversed(self._history),
            key=lambda frame: abs(int(frame["client_capture_ns"]) - capture_ns),
        )

    def clear(self) -> None:
        self._history.clear()

    def __len__(self) -> int:
        return len(self._history)
