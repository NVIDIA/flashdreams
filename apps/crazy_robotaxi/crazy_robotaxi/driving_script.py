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

"""Frame-counted driving commands for repeatable game runs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from omnidreams_game_engine.types import DriverCommand
from ruamel.yaml import YAML


@dataclass(frozen=True, slots=True)
class DrivingScript:
    """Validated control segments indexed by simulation frame."""

    path: Path
    """Source path retained for capture provenance."""

    segments: tuple[tuple[int, DriverCommand], ...]
    """Durations and complete commands; each segment resets omitted controls."""

    @property
    def frame_count(self) -> int:
        """Return the number of frames requested by the script."""
        return sum(count for count, _ in self.segments)

    def batch(self, start: int, count: int) -> tuple[tuple[DriverCommand, ...], int]:
        """Return a model-sized batch and the number of scripted frames in it.

        Pad the last batch with neutral controls because models require complete
        chunks. The caller discards that tail from video and telemetry exports.
        Frame zero is the game's initial pose, matching ordinary gameplay.
        """
        if start < 0 or count <= 0 or start >= self.frame_count:
            raise ValueError("Driving batch must start inside the script")
        commands: list[DriverCommand] = []
        offset = 0
        for duration, command in self.segments:
            overlap = max(0, min(start + count, offset + duration) - max(start, offset))
            commands.extend([command] * overlap)
            offset += duration
            if offset >= start + count:
                break
        valid_count = len(commands)
        commands.extend([DriverCommand(manual_control=True)] * (count - valid_count))
        return tuple(commands), valid_count


def load_driving_script(path: Path) -> DrivingScript:
    """Load strict YAML control segments without expanding a whole run in memory."""
    path = path.expanduser().resolve()
    with path.open(encoding="utf-8") as source:
        document = YAML(typ="safe").load(source)
    if not isinstance(document, dict) or set(document) != {"steps"}:
        raise ValueError("Driving script must contain only a 'steps' list")
    steps = document["steps"]
    if not isinstance(steps, list) or not steps:
        raise ValueError("Driving script 'steps' must be a non-empty list")
    segments = []
    pedals = {"throttle": (0.0, 1.0), "brake": (0.0, 1.0), "steer": (-1.0, 1.0)}
    flags = {"handbrake", "reverse", "stop", "steer_is_direct"}
    for index, step in enumerate(steps):
        if not isinstance(step, dict) or set(step) - {"frames", *pedals, *flags}:
            raise ValueError(f"Driving step {index} contains unknown controls")
        frames = step.get("frames")
        if isinstance(frames, bool) or not isinstance(frames, int) or frames <= 0:
            raise ValueError(f"Driving step {index} requires positive integer frames")
        controls = cast(
            dict[str, Any],
            {key: value for key, value in step.items() if key != "frames"},
        )
        for name, (minimum, maximum) in pedals.items():
            value = controls.get(name, 0.0)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not minimum <= value <= maximum
            ):
                raise ValueError(
                    f"Driving step {index}: {name} must be in [{minimum}, {maximum}]"
                )
        for name in flags & controls.keys():
            if not isinstance(controls[name], bool):
                raise ValueError(f"Driving step {index}: {name} must be boolean")
        segments.append((frames, DriverCommand(**controls, manual_control=True)))
    return DrivingScript(path=path, segments=tuple(segments))
