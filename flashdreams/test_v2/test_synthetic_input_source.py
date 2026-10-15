# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for descriptor-backed synthetic input."""

import json
import re
from pathlib import Path

import pytest

from flashdreams.api_v2.user_input_event import UserInputEvent
from flashdreams.runtime_v2.synthetic_input_source import SyntheticInputSource
from flashdreams.runtime_v2.user_input_event import (
    CloseUserInputEvent,
    KeyboardInputState,
    KeyboardUserInputEvent,
    ResetUserInputEvent,
)

pytestmark = pytest.mark.ci_cpu


class _Clock:
    def __init__(self, now_ns: int) -> None:
        self.now_ns = now_ns

    def __call__(self) -> int:
        return self.now_ns


def _write_descriptor(tmp_path: Path, contents: object) -> Path:
    path = tmp_path / "input.json"
    path.write_text(json.dumps(contents), encoding="utf-8")
    return path


def _descriptor(events: list[object]) -> dict[str, object]:
    return {"version": 1, "events": events}


def test_events_release_on_their_ui_poll_once_with_emission_timestamps(
    tmp_path: Path,
) -> None:
    path = _write_descriptor(
        tmp_path,
        _descriptor(
            [
                {
                    "on": "ui_loop",
                    "at": 0,
                    "event": {"type": "keyboard", "key": "w", "state": "Pressed"},
                },
                {"on": "ui_loop", "at": 2, "event": {"type": "reset"}},
                {
                    "on": "ui_loop",
                    "at": 2,
                    "event": {"type": "keyboard", "key": "s", "state": "Released"},
                },
            ]
        ),
    )
    clock = _Clock(1_000_000)
    source = SyntheticInputSource(path, clock_ns=clock)

    source.start_session()
    clock.now_ns = 1_007_000
    first = source.get_user_input_events().get_events()
    clock.now_ns = 1_011_000
    second = source.get_user_input_events().get_events()
    clock.now_ns = 1_019_000
    third = source.get_user_input_events().get_events()
    fourth = source.get_user_input_events().get_events()

    assert len(first) == 1
    assert isinstance(first[0], KeyboardUserInputEvent)
    assert first[0].get_timestamp() == 7
    assert first[0].key == "w"
    assert first[0].state is KeyboardInputState.PRESSED
    assert second == []
    assert len(third) == 2
    assert isinstance(third[0], ResetUserInputEvent)
    assert third[0].get_timestamp() == 19
    assert isinstance(third[1], KeyboardUserInputEvent)
    assert third[1].get_timestamp() == 19
    assert third[1].key == "s"
    assert third[1].state is KeyboardInputState.RELEASED
    assert fourth == []


def test_replacement_session_rebases_timestamps_without_replaying_events(
    tmp_path: Path,
) -> None:
    clock = _Clock(1_000_000)
    source = SyntheticInputSource(
        _write_descriptor(
            tmp_path,
            _descriptor(
                [
                    {"on": "ui_loop", "at": 0, "event": {"type": "reset"}},
                    {"on": "ui_loop", "at": 1, "event": {"type": "close"}},
                ]
            ),
        ),
        clock_ns=clock,
    )

    source.start_session()
    clock.now_ns = 1_007_000
    first = source.get_user_input_events().get_events()
    source.end_session()
    clock.now_ns = 2_000_000
    source.start_session()
    clock.now_ns = 2_011_000
    second = source.get_user_input_events().get_events()

    assert len(first) == 1
    assert first[0].get_timestamp() == 7
    assert len(second) == 1
    assert isinstance(second[0], CloseUserInputEvent)
    assert second[0].get_timestamp() == 11


@pytest.mark.parametrize(
    "contents",
    [
        [],
        {"version": 2, "events": []},
        {"version": 1.0, "events": []},
        {"version": 1, "events": [], "extra": True},
        _descriptor(
            [
                {
                    "on": "time",
                    "at": 0,
                    "event": {"type": "reset"},
                }
            ]
        ),
        _descriptor(
            [
                {
                    "on": "ui_loop",
                    "at": 0,
                    "event": {"type": "reset", "timestamp": 0},
                }
            ]
        ),
        _descriptor(
            [
                {
                    "on": "ui_loop",
                    "at": -1,
                    "event": {"type": "reset"},
                }
            ]
        ),
    ],
)
def test_the_parser_rejects_unsupported_v1_forms(
    tmp_path: Path, contents: object
) -> None:
    with pytest.raises((ValueError, TypeError)):
        SyntheticInputSource(_write_descriptor(tmp_path, contents))


def test_documentation_covers_every_registered_event_type() -> None:
    documentation = (
        Path(__file__).parents[2]
        / "docs"
        / "source"
        / "documentation"
        / "synthetic_input.md"
    ).read_text(encoding="utf-8")
    documented_types = set(
        re.findall(r"^\| `([^`]+)` \|", documentation, flags=re.MULTILINE)
    )

    assert documented_types == set(UserInputEvent._type_name_owners)
