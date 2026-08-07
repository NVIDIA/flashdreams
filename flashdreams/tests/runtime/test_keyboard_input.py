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

"""Pydantic validation tests for builtin keyboard input."""

from typing import Any

import pytest
from flashdreams.runtime.builtin.user_input.keyboard import (
    KeyboardEvent,
    KeyboardKey,
    RawUserKeyboardEvent,
)
from pydantic import TypeAdapter, ValidationError

pytestmark = pytest.mark.ci_cpu

_RAW_KEYBOARD_EVENT_ADAPTER = TypeAdapter(RawUserKeyboardEvent)


## Keyboard event validation


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("keydown", KeyboardEvent.KEY_DOWN),
        ("keyup", KeyboardEvent.KEY_UP),
    ],
)
def test_keyboard_event_parses_enum_values(
    value: str,
    expected: KeyboardEvent,
) -> None:
    """Verify Pydantic parses wire values into keyboard event members."""
    event = _RAW_KEYBOARD_EVENT_ADAPTER.validate_python(
        {"timestamp": 1.0, "event": value, "key": "w"}
    )

    assert event["event"] is expected


@pytest.mark.parametrize("value", ["keypress", 1, None])
def test_keyboard_event_rejects_invalid_enum_values(value: Any) -> None:
    """Verify Pydantic rejects values outside the keyboard event enum."""
    with pytest.raises(ValidationError):
        _RAW_KEYBOARD_EVENT_ADAPTER.validate_python(
            {"timestamp": 1.0, "event": value, "key": "w"}
        )


## Keyboard key validation


@pytest.mark.parametrize("expected", list(KeyboardKey))
def test_keyboard_key_parses_enum_values(expected: KeyboardKey) -> None:
    """Verify Pydantic parses supported key strings into enum members."""
    event = _RAW_KEYBOARD_EVENT_ADAPTER.validate_python(
        {
            "timestamp": 1.0,
            "event": KeyboardEvent.KEY_DOWN,
            "key": expected.value,
        }
    )

    assert event["key"] is expected


@pytest.mark.parametrize("value", ["enter", "", 1, None])
def test_keyboard_key_rejects_invalid_enum_values(value: Any) -> None:
    """Verify Pydantic rejects values outside the keyboard key enum."""
    with pytest.raises(ValidationError):
        _RAW_KEYBOARD_EVENT_ADAPTER.validate_python(
            {"timestamp": 1.0, "event": "keydown", "key": value}
        )
