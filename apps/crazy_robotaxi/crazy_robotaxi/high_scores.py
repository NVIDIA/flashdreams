# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Persistent taxi-game high-score storage."""

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

_CSV_FIELDS = ("name", "score", "achieved_at_utc")
_PLAYER_NAME_RE = re.compile(r"[A-Za-z0-9 _-]{1,12}")


def default_high_scores_path() -> Path:
    """Return the default persistent taxi leaderboard path."""
    from omnidreams.scenes import FLASHDREAMS_CACHE_DIR

    return FLASHDREAMS_CACHE_DIR / "crazy-robotaxi" / "highscores.csv"


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
