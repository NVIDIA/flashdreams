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

"""Immutable game and map choices exchanged across the V2 loop boundary."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

GameMode = Literal["taxi", "race"]


@dataclass(frozen=True, slots=True)
class GameRaceCourseOption:
    """Lightweight authored race-course metadata displayed by the UI thread."""

    course_id: str
    """Stable course identifier scoped to its map."""

    spawn_id: str
    """Required authored spawn used to initialize the course."""

    preview_image_path: Path | None = None
    """Resolved course-spawn image, when authored."""


@dataclass(frozen=True, slots=True)
class GameMapOption:
    """Lightweight authored-map metadata displayed by the UI thread."""

    map_id: str
    """Stable identifier stored in scores and race times."""

    name: str
    """Human-readable map name shown in the selection screen."""

    path: Path
    """Resolved authored-map path loaded after selection."""

    race_courses: tuple[GameRaceCourseOption, ...] = ()
    """Ordered race courses available on this map."""

    preview_image_path: Path | None = None
    """Resolved map menu thumbnail after applying authored fallback rules."""

    @property
    def race_course_ids(self) -> tuple[str, ...]:
        """Return ordered stable course identifiers."""
        return tuple(course.course_id for course in self.race_courses)

    def race_course(self, course_id: str | None) -> GameRaceCourseOption | None:
        """Return the matching course menu metadata, when available."""
        return next(
            (course for course in self.race_courses if course.course_id == course_id),
            None,
        )


@dataclass(frozen=True, slots=True)
class GameSelection:
    """One complete menu choice queued for the model thread."""

    mode: GameMode
    """Rules mode chosen on the first selection screen."""

    map_option: GameMapOption
    """Map metadata chosen on the second selection screen."""

    race_course_id: str | None = None
    """Race course selected with the map; ``None`` in taxi mode."""
