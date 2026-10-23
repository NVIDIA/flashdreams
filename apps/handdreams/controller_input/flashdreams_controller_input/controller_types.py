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

"""Controller observations, source contracts, and transport constants."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

DEFAULT_PORT = 8444
"""Default local receiver port for the Isaac Teleop bridge."""

ISAAC_TELEOP_BRIDGE_PROTOCOL_VERSION = 5
"""Protocol spoken between the isolated bridge and host source."""

DEFAULT_ISAAC_TELEOP_DIRECTORY = Path.home() / ".local/share/flashdreams/isaac-teleop"
"""User-local SDK environment shared by setup and every launch path."""

SUPPORTED_ISAAC_TELEOP_VERSION = "1.4.142"
"""Upstream SDK base version whose controller contract the bridge implements."""

ISAAC_TELEOP_REQUIREMENT = (
    f"isaacteleop[cloudxr,retargeters-lite]=={SUPPORTED_ISAAC_TELEOP_VERSION}"
)
CLOUDXR_EULA_URL = (
    "https://github.com/NVIDIA/IsaacCapture/blob/main/deps/cloudxr/CLOUDXR_LICENSE"
)
CLOUDXR_EULA_HELP = (
    f"Accept the NVIDIA CloudXR EULA for {ISAAC_TELEOP_REQUIREMENT}: {CLOUDXR_EULA_URL}"
)


@dataclass(frozen=True, slots=True)
class Observation:
    """One controller sample tagged with the source that produced it.

    Attributes:
        frame: Controller state in the Quest wire schema, or ``None`` for an
            explicit disconnect. Return no observation when no sample is ready.
        capture_ns: Host capture time.
        source: Name of the producing source.
        generation: Monotonic tracking-origin generation. A change resets the
            V2 session before any poses from the new origin are delivered.
    """

    frame: dict | None
    capture_ns: int
    source: str
    generation: int = 0


@runtime_checkable
class InputSource(Protocol):
    """A source of controller observations.

    ``start`` may block for a long time; the window always calls it on a worker
    thread. ``poll`` must be cheap and non-blocking beyond its own pacing.
    """

    name: str

    def start(self) -> None:
        """Bring the source up. May block."""

    def stop(self) -> None:
        """Release everything the source owns. Must be idempotent."""

    def poll(self) -> Observation | None:
        """Return the newest observation, or ``None`` if none is available."""
