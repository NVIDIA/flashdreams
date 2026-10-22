# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for composing a primary window with secondary I/O."""

from typing import cast

import pytest
from numpy import uint64

from flashdreams.api_v2.client_window import ClientWindowLink, IClientWindow
from flashdreams.api_v2.input_source import InputSource
from flashdreams.runtime_v2.composite_client_window import (
    CompositeClientWindow,
    CompositeClientWindowError,
)
from flashdreams.runtime_v2.session_desc import SessionDesc
from flashdreams.runtime_v2.step_result import StepResult
from flashdreams.runtime_v2.user_input_event import (
    CloseUserInputEvent,
    FocusUserInputEvent,
    ResetUserInputEvent,
)
from flashdreams.runtime_v2.user_input_events import UserInputEvents

pytestmark = pytest.mark.ci_cpu


class _Window:
    def __init__(
        self,
        calls: list[object] | None = None,
        *,
        open_error: BaseException | None = None,
        close_error: BaseException | None = None,
    ) -> None:
        self.calls = [] if calls is None else calls
        self._open_error = open_error
        self._close_error = close_error

    def get_user_input_events(self) -> UserInputEvents:
        return UserInputEvents([FocusUserInputEvent(timestamp=uint64(0))])

    def open(self, session_desc: SessionDesc) -> None:
        self.calls.append(("primary", "open", session_desc))
        if self._open_error is not None:
            raise self._open_error

    def write(self, result: StepResult) -> None:
        self.calls.append(("primary", "write", result))

    def request_links(self, links: tuple[ClientWindowLink, ...]) -> None:
        self.calls.append(("links", links))

    def request_new_window_size(self, new_window_size: tuple[int, int]) -> None:
        self.calls.append(new_window_size)

    def request_hide_cursor(self, hide_cursor: bool) -> None:
        self.calls.append(("hide", hide_cursor))

    def request_lock_cursor_to_window(self, lock_cursor_to_window: bool) -> None:
        self.calls.append(("lock", lock_cursor_to_window))

    def close(self) -> None:
        self.calls.append(("primary", "close"))
        if self._close_error is not None:
            raise self._close_error


class _InputSource(InputSource):
    def __init__(self, event: ResetUserInputEvent | CloseUserInputEvent) -> None:
        self._event = event

    def get_user_input_events(self) -> UserInputEvents:
        return UserInputEvents([self._event])


class _OutputSink:
    def __init__(
        self,
        name: str,
        calls: list[object],
        *,
        open_error: BaseException | None = None,
        write_error: BaseException | None = None,
        close_error: BaseException | None = None,
    ) -> None:
        self._name = name
        self._calls = calls
        self._open_error = open_error
        self._write_error = write_error
        self._close_error = close_error

    def open(self, session_desc: SessionDesc) -> None:
        self._calls.append((self._name, "open", session_desc))
        if self._open_error is not None:
            raise self._open_error

    def write(self, result: StepResult) -> None:
        self._calls.append((self._name, "write", result))
        if self._write_error is not None:
            raise self._write_error

    def close(self) -> None:
        self._calls.append((self._name, "close"))
        if self._close_error is not None:
            raise self._close_error


def test_the_composite_preserves_primary_and_secondary_input_order() -> None:
    reset = ResetUserInputEvent(timestamp=uint64(0))
    close = CloseUserInputEvent(timestamp=uint64(0))
    sources = [_InputSource(reset), _InputSource(close)]
    window = CompositeClientWindow(
        cast(IClientWindow, _Window()), secondary_input_sources=sources
    )
    sources.clear()

    events = window.get_user_input_events().get_events()

    assert [type(event) for event in events] == [
        FocusUserInputEvent,
        ResetUserInputEvent,
        CloseUserInputEvent,
    ]
    assert len(window.secondary_input_sources) == 2


def test_requests_route_only_to_the_primary_window() -> None:
    primary = _Window()
    window = CompositeClientWindow(cast(IClientWindow, primary))
    links = (ClientWindowLink("Docs", "https://example.test", (0, 0, 1, 1)),)

    window.request_links(links)
    window.request_hide_cursor(True)
    window.request_lock_cursor_to_window(True)
    window.request_new_window_size((320, 180))

    assert primary.calls == [
        ("links", links),
        ("hide", True),
        ("lock", True),
        (320, 180),
    ]


def test_lifecycle_and_results_reach_primary_and_every_secondary_sink() -> None:
    calls: list[object] = []
    primary = _Window(calls)
    first = _OutputSink("first", calls)
    second = _OutputSink("second", calls)
    window = CompositeClientWindow(
        cast(IClientWindow, primary),
        secondary_output_sinks=(first, second),
        on_open=lambda: calls.append(("inputs", "open")),
        on_close=lambda: calls.append(("inputs", "close")),
        on_ui_step_completed=lambda: calls.append(("inputs", "ui_step")),
    )
    session_desc = cast(SessionDesc, object())
    result = cast(StepResult, object())

    window.open(session_desc)
    window.write(result)
    window.ui_step_completed()
    window.close()

    assert window.primary_window is primary
    assert window.secondary_output_sinks == (first, second)
    assert calls == [
        ("primary", "open", session_desc),
        ("first", "open", session_desc),
        ("second", "open", session_desc),
        ("inputs", "open"),
        ("primary", "write", result),
        ("first", "write", result),
        ("second", "write", result),
        ("inputs", "ui_step"),
        ("inputs", "close"),
        ("second", "close"),
        ("first", "close"),
        ("primary", "close"),
    ]


def test_open_failure_rolls_back_only_opened_components_in_reverse_order() -> None:
    calls: list[object] = []
    failure = RuntimeError("open failed")
    window = CompositeClientWindow(
        cast(IClientWindow, _Window(calls)),
        secondary_output_sinks=(
            _OutputSink("first", calls),
            _OutputSink("failing", calls, open_error=failure),
            _OutputSink("unreached", calls),
        ),
    )
    session_desc = cast(SessionDesc, object())

    with pytest.raises(RuntimeError, match="open failed") as raised:
        window.open(session_desc)

    assert raised.value is failure
    assert calls == [
        ("primary", "open", session_desc),
        ("first", "open", session_desc),
        ("failing", "open", session_desc),
        ("first", "close"),
        ("primary", "close"),
    ]


def test_replacement_session_reopens_every_component_in_order() -> None:
    calls: list[object] = []
    primary = _Window(calls)
    secondary = _OutputSink("secondary", calls)
    window = CompositeClientWindow(
        cast(IClientWindow, primary),
        secondary_output_sinks=(secondary,),
        on_open=lambda: calls.append(("inputs", "open")),
    )
    first = cast(SessionDesc, object())
    second = cast(SessionDesc, object())

    window.open(first)
    window.open(second)

    assert calls == [
        ("primary", "open", first),
        ("secondary", "open", first),
        ("inputs", "open"),
        ("primary", "open", second),
        ("secondary", "open", second),
        ("inputs", "open"),
    ]


def test_write_failure_fails_fast() -> None:
    calls: list[object] = []
    failure = RuntimeError("write failed")
    window = CompositeClientWindow(
        cast(IClientWindow, _Window(calls)),
        secondary_output_sinks=(
            _OutputSink("failing", calls, write_error=failure),
            _OutputSink("unreached", calls),
        ),
    )
    result = cast(StepResult, object())

    with pytest.raises(RuntimeError, match="write failed"):
        window.write(result)

    assert calls == [
        ("primary", "write", result),
        ("failing", "write", result),
    ]


def test_close_attempts_every_component_and_reports_every_failure() -> None:
    calls: list[object] = []
    input_error = RuntimeError("input close failed")
    sink_error = RuntimeError("sink close failed")
    primary_error = RuntimeError("primary close failed")
    window = CompositeClientWindow(
        cast(IClientWindow, _Window(calls, close_error=primary_error)),
        secondary_output_sinks=(
            _OutputSink("first", calls),
            _OutputSink("second", calls, close_error=sink_error),
        ),
        on_close=lambda: (_ for _ in ()).throw(input_error),
    )

    with pytest.raises(CompositeClientWindowError) as raised:
        window.close()

    assert raised.value.errors == (input_error, sink_error, primary_error)
    assert calls == [
        ("second", "close"),
        ("first", "close"),
        ("primary", "close"),
    ]
