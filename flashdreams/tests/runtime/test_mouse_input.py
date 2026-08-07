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

"""Pydantic validation tests for builtin mouse input."""

from typing import Any

import pytest
from flashdreams.runtime.builtin.user_input.mouse import (
    MouseButton,
    MouseEvent,
    RawUserMouseEvent,
)
from pydantic import TypeAdapter, ValidationError

pytestmark = pytest.mark.ci_cpu

_RAW_MOUSE_EVENT_ADAPTER = TypeAdapter(RawUserMouseEvent)


## Mouse event validation


@pytest.mark.parametrize("expected", list(MouseEvent))
def test_mouse_event_parses_enum_values(expected: MouseEvent) -> None:
    """Verify Pydantic parses mouse edge strings into enum members."""
    event = _RAW_MOUSE_EVENT_ADAPTER.validate_python(
        {"timestamp": 1.0, "event": expected.value, "button": "left"}
    )

    assert event["event"] is expected


## Mouse button validation


@pytest.mark.parametrize("expected", list(MouseButton))
def test_mouse_button_parses_enum_values(expected: MouseButton) -> None:
    """Verify Pydantic parses mouse button strings into enum members."""
    event = _RAW_MOUSE_EVENT_ADAPTER.validate_python(
        {
            "timestamp": 1.0,
            "event": MouseEvent.BUTTON_DOWN,
            "button": expected.value,
        }
    )

    assert event["button"] is expected


@pytest.mark.parametrize("value", ["mousemove", "click", 1, None])
def test_mouse_event_rejects_invalid_enum_values(value: Any) -> None:
    """Verify Pydantic rejects values outside the mouse event enum."""
    with pytest.raises(ValidationError):
        _RAW_MOUSE_EVENT_ADAPTER.validate_python(
            {"timestamp": 1.0, "event": value, "button": "left"}
        )


@pytest.mark.parametrize("value", ["back", "", 1, None])
def test_mouse_button_rejects_invalid_enum_values(value: Any) -> None:
    """Verify Pydantic rejects values outside the mouse button enum."""
    with pytest.raises(ValidationError):
        _RAW_MOUSE_EVENT_ADAPTER.validate_python(
            {"timestamp": 1.0, "event": "mousedown", "button": value}
        )
