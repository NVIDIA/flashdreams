# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for composing a client window with an input source."""

from typing import cast

import pytest
from numpy import uint64

from flashdreams.api_v2.client_window import IClientWindow
from flashdreams.api_v2.input_source import InputSource, SessionInputSource
from flashdreams.runtime_v2.composite_client_window import CompositeClientWindow
from flashdreams.runtime_v2.session_desc import SessionDesc
from flashdreams.runtime_v2.step_result import StepResult
from flashdreams.runtime_v2.user_input_event import (
    FocusUserInputEvent,
    ResetUserInputEvent,
)
from flashdreams.runtime_v2.user_input_events import UserInputEvents

pytestmark = pytest.mark.ci_cpu


class _Window:
    def __init__(self) -> None:
        self.calls: list[object] = []

    def get_user_input_events(self) -> UserInputEvents:
        return UserInputEvents([FocusUserInputEvent(timestamp=uint64(0))])

    def open(self, session_desc: SessionDesc) -> None:
        self.calls.append(session_desc)

    def write(self, result: StepResult) -> None:
        self.calls.append(result)

    def request_new_window_size(self, new_window_size: tuple[int, int]) -> None:
        self.calls.append(new_window_size)

    def request_hide_cursor(self, hide_cursor: bool) -> None:
        self.calls.append(("hide", hide_cursor))

    def request_lock_cursor_to_window(self, lock_cursor_to_window: bool) -> None:
        self.calls.append(("lock", lock_cursor_to_window))

    def close(self) -> None:
        self.calls.append("close")


class _InputSource(InputSource):
    def get_user_input_events(self) -> UserInputEvents:
        return UserInputEvents([ResetUserInputEvent(timestamp=uint64(0))])


class _SessionInputSource(_InputSource, SessionInputSource):
    def __init__(self) -> None:
        self.calls: list[str] = []

    def open(self) -> None:
        self.calls.append("open")

    def close(self) -> None:
        self.calls.append("close")


def test_the_composite_preserves_wrapped_input_before_additional_input() -> None:
    window = CompositeClientWindow(cast(IClientWindow, _Window()), _InputSource())

    events = window.get_user_input_events().get_events()

    assert [type(event) for event in events] == [
        FocusUserInputEvent,
        ResetUserInputEvent,
    ]


def test_the_composite_delegates_the_client_window_lifecycle() -> None:
    client_window = _Window()
    window = CompositeClientWindow(cast(IClientWindow, client_window), _InputSource())
    session_desc = cast(SessionDesc, object())
    result = cast(StepResult, object())

    window.open(session_desc)
    window.write(result)
    window.request_hide_cursor(True)
    window.request_lock_cursor_to_window(True)
    window.request_new_window_size((320, 180))
    window.close()

    assert window.client_window is client_window
    assert client_window.calls == [
        session_desc,
        result,
        ("hide", True),
        ("lock", True),
        (320, 180),
        "close",
    ]


def test_the_composite_preserves_the_temporary_session_input_lifecycle() -> None:
    source = _SessionInputSource()
    window = CompositeClientWindow(cast(IClientWindow, _Window()), source)

    window.open(cast(SessionDesc, object()))
    window.close()

    assert source.calls == ["open", "close"]
