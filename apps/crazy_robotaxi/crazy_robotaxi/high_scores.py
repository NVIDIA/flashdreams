# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Persistent taxi and race leaderboard storage."""

from __future__ import annotations

import csv
import os
import re
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from filelock import FileLock
from loguru import logger

from flashdreams.core.io.disk import default_flashdreams_cache_dir

_CSV_FIELDS = ("name", "score", "achieved_at_utc")
_RACE_CSV_FIELDS = (
    "map_id",
    "course_id",
    "name",
    "elapsed_time_us",
    "achieved_at_utc",
)

_PLAYER_NAME_RE = re.compile(r"[A-Za-z0-9 _-]{1,12}")


def format_race_time_us(elapsed_time_us: int) -> str:
    """Format a race duration as minutes, seconds, and milliseconds.

    Args:
        elapsed_time_us: Nonnegative duration in integer microseconds.

    Returns:
        Duration formatted as ``M:SS.XXX`` with unbounded minutes.
    """
    total_milliseconds = (max(0, elapsed_time_us) + 500) // 1_000
    minutes, milliseconds_in_minute = divmod(total_milliseconds, 60_000)
    seconds, milliseconds = divmod(milliseconds_in_minute, 1_000)
    return f"{minutes}:{seconds:02d}.{milliseconds:03d}"


def default_high_scores_path() -> Path:
    """Return the default persistent taxi leaderboard path."""
    return default_flashdreams_cache_dir() / "crazy-robotaxi" / "highscores.csv"


def default_race_times_path() -> Path:
    """Return the default persistent race leaderboard path."""
    return default_flashdreams_cache_dir() / "crazy-robotaxi" / "race_times.csv"


def validate_player_name(name: str) -> str:
    """Normalize and validate a leaderboard player name.

    Args:
        name: Candidate player name.

    Returns:
        Name with surrounding whitespace removed.

    Raises:
        ValueError: The normalized name is empty, too long, or contains an
            unsupported character.
    """
    normalized = name.strip()
    if _PLAYER_NAME_RE.fullmatch(normalized) is None:
        raise ValueError(
            "Name must be 1-12 characters using letters, numbers, spaces, "
            "hyphens, or underscores."
        )
    return normalized


@dataclass(frozen=True)
class HighScoreEntry:
    """One persisted leaderboard result."""

    name: str
    """Player name shown on the leaderboard."""

    score: int
    """Final game score."""

    achieved_at_utc: str
    """UTC ISO-8601 timestamp used to order tied scores."""

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation of the entry."""
        return {
            "name": self.name,
            "score": self.score,
            "achieved_at_utc": self.achieved_at_utc,
        }


class HighScoreStore:
    """Read and atomically update a top-ten CSV leaderboard."""

    def __init__(self, path: Path, *, limit: int = 10) -> None:
        self._path = path
        self._limit = limit
        self._lock_path = path.with_suffix(f"{path.suffix}.lock")

    @property
    def path(self) -> Path:
        """Return the leaderboard CSV path."""
        return self._path

    def read(self) -> tuple[HighScoreEntry, ...]:
        """Return the sorted leaderboard while tolerating malformed rows."""
        if not self._path.exists():
            return ()
        try:
            with FileLock(self._lock_path):
                return self._read_unlocked()
        except OSError as exc:
            logger.warning(f"[taxi] could not lock high scores at {self._path}: {exc}")
            return self._read_unlocked()

    def qualifying_rank(self, score: int) -> int | None:
        """Return the prospective rank for ``score``, or ``None`` if excluded."""
        if score <= 0:
            return None
        entries = self.read()
        if len(entries) >= self._limit and score <= entries[-1].score:
            return None
        return 1 + sum(entry.score >= score for entry in entries)

    def record(
        self,
        name: str,
        score: int,
        *,
        achieved_at_utc: str | None = None,
    ) -> tuple[HighScoreEntry | None, tuple[HighScoreEntry, ...]]:
        """Insert a qualifying score and return it with the updated board.

        Args:
            name: Player name to validate and persist.
            score: Final game score.
            achieved_at_utc: Optional ISO-8601 timestamp for deterministic tests.

        Returns:
            Inserted entry, or ``None`` if a concurrent update displaced the
            score, together with the current top-ten leaderboard.
        """
        normalized_name = validate_player_name(name)
        if score <= 0:
            return None, self.read()
        timestamp = achieved_at_utc or datetime.now(timezone.utc).isoformat(
            timespec="seconds"
        )
        entry = HighScoreEntry(normalized_name, int(score), timestamp)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with FileLock(self._lock_path):
            entries = list(self._read_unlocked())
            inserted: HighScoreEntry | None = entry
            if len(entries) >= self._limit and score <= entries[-1].score:
                inserted = None
            else:
                entries.append(entry)
            board = self._sort(entries)
            self._write_unlocked(board)
            return inserted, board

    def _read_unlocked(self) -> tuple[HighScoreEntry, ...]:
        if not self._path.exists():
            return ()
        entries: list[HighScoreEntry] = []
        try:
            with self._path.open(newline="", encoding="utf-8") as csv_file:
                for row_number, row in enumerate(csv.DictReader(csv_file), start=2):
                    try:
                        name = validate_player_name(row.get("name", ""))
                        score = int(row.get("score", ""))
                        timestamp = row.get("achieved_at_utc", "")
                        datetime.fromisoformat(timestamp)
                    except (TypeError, ValueError):
                        logger.warning(
                            f"[taxi] ignoring malformed high-score row {row_number} "
                            f"in {self._path}"
                        )
                        continue
                    if score <= 0:
                        continue
                    entries.append(HighScoreEntry(name, score, timestamp))
        except (OSError, csv.Error) as exc:
            logger.warning(
                f"[taxi] could not read high scores from {self._path}: {exc}"
            )
            return ()
        return self._sort(entries)

    def _sort(self, entries: list[HighScoreEntry]) -> tuple[HighScoreEntry, ...]:
        return tuple(
            sorted(entries, key=lambda entry: (-entry.score, entry.achieved_at_utc))[
                : self._limit
            ]
        )

    def _write_unlocked(self, entries: tuple[HighScoreEntry, ...]) -> None:
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                newline="",
                encoding="utf-8",
                dir=self._path.parent,
                prefix=f".{self._path.name}.",
                suffix=".tmp",
                delete=False,
            ) as csv_file:
                temporary_path = Path(csv_file.name)
                writer = csv.DictWriter(csv_file, fieldnames=_CSV_FIELDS)
                writer.writeheader()
                for entry in entries:
                    writer.writerow(entry.as_dict())
                csv_file.flush()
                os.fsync(csv_file.fileno())
            os.replace(temporary_path, self._path)
        finally:
            if temporary_path is not None and temporary_path.exists():
                temporary_path.unlink()


@dataclass(frozen=True)
class RaceTimeEntry:
    """One map- and course-specific race result."""

    map_id: str
    """Stable ID of the map on which the result was achieved."""

    course_id: str
    """Course ID scoped to ``map_id``."""

    name: str
    """Player name shown on the leaderboard."""

    elapsed_time_us: int
    """Total race time in integer microseconds."""

    achieved_at_utc: str
    """UTC ISO-8601 timestamp used to order tied times."""

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation of the entry."""
        return {
            "map_id": self.map_id,
            "course_id": self.course_id,
            "name": self.name,
            "elapsed_time_us": self.elapsed_time_us,
            "elapsed_time_s": self.elapsed_time_us / 1_000_000.0,
            "elapsed_time": format_race_time_us(self.elapsed_time_us),
            "achieved_at_utc": self.achieved_at_utc,
        }


class RaceTimeStore:
    """Atomically maintain a top-ten race board for every map/course pair."""

    def __init__(self, path: Path, *, limit: int = 10) -> None:
        self._path = path
        self._limit = limit
        self._lock_path = path.with_suffix(f"{path.suffix}.lock")

    @property
    def path(self) -> Path:
        """Return the shared race-times CSV path."""
        return self._path

    def read(self, map_id: str, course_id: str) -> tuple[RaceTimeEntry, ...]:
        """Return the board for one map/course pair."""
        entries = self._read_locked()
        return self._board(entries, map_id, course_id)

    def qualifying_rank(
        self, map_id: str, course_id: str, elapsed_time_us: int
    ) -> int | None:
        """Return the prospective rank for a total time, if it qualifies."""
        if elapsed_time_us <= 0:
            return None
        board = self.read(map_id, course_id)
        if len(board) >= self._limit and elapsed_time_us >= board[-1].elapsed_time_us:
            return None
        return 1 + sum(entry.elapsed_time_us <= elapsed_time_us for entry in board)

    def record(
        self,
        map_id: str,
        course_id: str,
        name: str,
        elapsed_time_us: int,
        *,
        achieved_at_utc: str | None = None,
    ) -> tuple[RaceTimeEntry | None, tuple[RaceTimeEntry, ...]]:
        """Insert a qualifying total time and return the updated scoped board."""
        map_id = self._validate_scope_id("map_id", map_id)
        course_id = self._validate_scope_id("course_id", course_id)
        normalized_name = validate_player_name(name)
        if elapsed_time_us <= 0:
            return None, self.read(map_id, course_id)
        timestamp = achieved_at_utc or datetime.now(timezone.utc).isoformat(
            timespec="seconds"
        )
        entry = RaceTimeEntry(
            map_id, course_id, normalized_name, int(elapsed_time_us), timestamp
        )
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with FileLock(self._lock_path):
            entries = list(self._read_unlocked())
            board = self._board(entries, map_id, course_id)
            inserted: RaceTimeEntry | None = entry
            if (
                len(board) >= self._limit
                and elapsed_time_us >= board[-1].elapsed_time_us
            ):
                inserted = None
            else:
                entries.append(entry)
            entries = self._trim_all(entries)
            self._write_unlocked(entries)
            return inserted, self._board(entries, map_id, course_id)

    def _read_locked(self) -> tuple[RaceTimeEntry, ...]:
        if not self._path.exists():
            return ()
        try:
            with FileLock(self._lock_path):
                return self._read_unlocked()
        except OSError as exc:
            logger.warning(f"[race] could not lock times at {self._path}: {exc}")
            return self._read_unlocked()

    def _read_unlocked(self) -> tuple[RaceTimeEntry, ...]:
        if not self._path.exists():
            return ()
        entries: list[RaceTimeEntry] = []
        try:
            with self._path.open(newline="", encoding="utf-8") as csv_file:
                for row_number, row in enumerate(csv.DictReader(csv_file), start=2):
                    try:
                        map_id = self._validate_scope_id(
                            "map_id", row.get("map_id", "")
                        )
                        course_id = self._validate_scope_id(
                            "course_id", row.get("course_id", "")
                        )
                        name = validate_player_name(row.get("name", ""))
                        elapsed = int(row.get("elapsed_time_us", ""))
                        timestamp = row.get("achieved_at_utc", "")
                        datetime.fromisoformat(timestamp)
                        if elapsed <= 0:
                            raise ValueError
                    except (TypeError, ValueError):
                        logger.warning(
                            f"[race] ignoring malformed time row {row_number} "
                            f"in {self._path}"
                        )
                        continue
                    entries.append(
                        RaceTimeEntry(map_id, course_id, name, elapsed, timestamp)
                    )
        except (OSError, csv.Error) as exc:
            logger.warning(f"[race] could not read times from {self._path}: {exc}")
            return ()
        return tuple(entries)

    def _board(
        self,
        entries: tuple[RaceTimeEntry, ...] | list[RaceTimeEntry],
        map_id: str,
        course_id: str,
    ) -> tuple[RaceTimeEntry, ...]:
        scoped = (
            entry
            for entry in entries
            if entry.map_id == map_id and entry.course_id == course_id
        )
        return tuple(
            sorted(
                scoped, key=lambda entry: (entry.elapsed_time_us, entry.achieved_at_utc)
            )[: self._limit]
        )

    def _trim_all(self, entries: list[RaceTimeEntry]) -> tuple[RaceTimeEntry, ...]:
        scopes = sorted({(entry.map_id, entry.course_id) for entry in entries})
        return tuple(
            entry
            for map_id, course_id in scopes
            for entry in self._board(entries, map_id, course_id)
        )

    @staticmethod
    def _validate_scope_id(field: str, value: object) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field} must be a non-empty string")
        return value.strip()

    def _write_unlocked(self, entries: tuple[RaceTimeEntry, ...]) -> None:
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                newline="",
                encoding="utf-8",
                dir=self._path.parent,
                prefix=f".{self._path.name}.",
                suffix=".tmp",
                delete=False,
            ) as csv_file:
                temporary_path = Path(csv_file.name)
                writer = csv.DictWriter(csv_file, fieldnames=_RACE_CSV_FIELDS)
                writer.writeheader()
                for entry in entries:
                    row = entry.as_dict()
                    writer.writerow({field: row[field] for field in _RACE_CSV_FIELDS})
                csv_file.flush()
                os.fsync(csv_file.fileno())
            os.replace(temporary_path, self._path)
        finally:
            if temporary_path is not None and temporary_path.exists():
                temporary_path.unlink()
