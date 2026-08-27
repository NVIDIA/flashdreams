# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""CPU tests for persistent taxi-game high scores."""

from pathlib import Path

import pytest
from crazy_robotaxi.high_scores import (
    HighScoreStore,
    format_race_time_us,
    validate_player_name,
)

pytestmark = pytest.mark.ci_cpu


@pytest.mark.parametrize(
    ("elapsed_time_us", "formatted"),
    [
        (0, "0:00.000"),
        (1_234_000, "0:01.234"),
        (61_234_000, "1:01.234"),
        (3_599_999_600, "60:00.000"),
    ],
)
def test_race_time_format_uses_minutes_seconds_and_milliseconds(
    elapsed_time_us: int,
    formatted: str,
) -> None:
    assert format_race_time_us(elapsed_time_us) == formatted


@pytest.mark.parametrize("name", ["Ada", "PLAYER 1", "A-B_C"])
def test_player_name_validation_accepts_supported_names(name: str) -> None:
    assert validate_player_name(f" {name} ") == name


@pytest.mark.parametrize("name", ["", "   ", "too-long-name!", "bad.name"])
def test_player_name_validation_rejects_invalid_names(name: str) -> None:
    with pytest.raises(ValueError, match="Name must be"):
        validate_player_name(name)


def test_store_orders_scores_and_uses_earlier_timestamp_for_ties(
    tmp_path: Path,
) -> None:
    store = HighScoreStore(tmp_path / "scores.csv")

    store.record("LATER", 900, achieved_at_utc="2026-08-10T12:00:01+00:00")
    store.record("HIGH", 1200, achieved_at_utc="2026-08-10T12:00:02+00:00")
    store.record("EARLIER", 900, achieved_at_utc="2026-08-10T12:00:00+00:00")

    assert [(entry.name, entry.score) for entry in store.read()] == [
        ("HIGH", 1200),
        ("EARLIER", 900),
        ("LATER", 900),
    ]


def test_store_retains_top_ten_and_requires_strictly_better_tenth_place(
    tmp_path: Path,
) -> None:
    store = HighScoreStore(tmp_path / "scores.csv")
    for index in range(10):
        store.record(
            f"P{index}",
            1000 - index * 10,
            achieved_at_utc=f"2026-08-10T12:00:{index:02d}+00:00",
        )

    assert store.qualifying_rank(910) is None
    assert store.qualifying_rank(911) == 10
    inserted, board = store.record(
        "NEW",
        911,
        achieved_at_utc="2026-08-10T13:00:00+00:00",
    )

    assert inserted is not None
    assert len(board) == 10
    assert board[-1].name == "NEW"


def test_store_excludes_zero_scores_from_qualification_and_persistence(
    tmp_path: Path,
) -> None:
    path = tmp_path / "scores.csv"
    store = HighScoreStore(path)

    assert store.qualifying_rank(0) is None
    inserted, board = store.record(
        "ZERO",
        0,
        achieved_at_utc="2026-08-10T12:00:00+00:00",
    )

    assert inserted is None
    assert board == ()
    assert path.exists() is False


def test_store_skips_malformed_rows_and_preserves_csv_escaping(tmp_path: Path) -> None:
    path = tmp_path / "scores.csv"
    path.write_text(
        "name,score,achieved_at_utc\n"
        '"PLAYER 1",800,2026-08-10T12:00:00+00:00\n'
        "ZERO,0,2026-08-10T12:00:01+00:00\n"
        "BAD,not-a-score,not-a-date\n",
        encoding="utf-8",
    )
    store = HighScoreStore(path)

    assert [(entry.name, entry.score) for entry in store.read()] == [("PLAYER 1", 800)]
    store.record("A-B_C", 900, achieved_at_utc="2026-08-10T13:00:00+00:00")

    assert [(entry.name, entry.score) for entry in store.read()] == [
        ("A-B_C", 900),
        ("PLAYER 1", 800),
    ]
    assert list(tmp_path.glob(".scores.csv.*.tmp")) == []
