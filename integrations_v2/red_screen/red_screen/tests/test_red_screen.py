# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU test for the red screen application's response to key input."""

import threading

import pytest
import torch
from numpy import uint64
from red_screen import create_app

from flashdreams.api_v2.client_window import IClientWindow
from flashdreams.api_v2.session import ISession
from flashdreams.runtime_v2.session_desc import SessionDesc
from flashdreams.runtime_v2.session_runner import run_session
from flashdreams.runtime_v2.step_result import StepResult
from flashdreams.runtime_v2.user_input_event import (
    KeyboardUserInputEventData,
    UserInputEvent,
)
from flashdreams.runtime_v2.user_input_events import UserInputEvents
from flashdreams.runtime_v2.video_tensor import VideoTensorLayout

pytestmark = pytest.mark.ci_cpu

_ACTIVATION_KEY = "r"
"""Matches the application's default activation key."""

_FRAME_SIZE = 2
"""Frame width and height; small enough to assert over every pixel."""


class ScriptedClientWindow(IClientWindow):
    """Report input once and record what is presented.

    The runner polls input on its own thread, so a window cannot line events up
    with individual steps. It reports its events on the first poll, which the
    runner makes before generation starts, and nothing after that. Behaviour that
    depends on which step an event lands on is covered by stepping a session
    directly instead.
    """

    def __init__(self, initial_events: UserInputEvents | None = None) -> None:
        """
        Args:
            initial_events: Events to report on the first poll.
        """
        self.session_desc: SessionDesc | None = None
        self.results: list[StepResult] = []
        self._pending = initial_events
        self._lock = threading.Lock()
        self._is_open = False

    def get_user_input_events(self) -> UserInputEvents:
        with self._lock:
            pending, self._pending = self._pending, None
        return pending or UserInputEvents([])

    def open(self, session_desc: SessionDesc) -> None:
        self.session_desc = session_desc
        self._is_open = True

    def write(self, result: StepResult) -> None:
        assert self._is_open
        self.results.append(result)

    def close(self) -> None:
        self._is_open = False


## Helpers


def _session_desc(layout: VideoTensorLayout = VideoTensorLayout.bcthw) -> SessionDesc:
    return SessionDesc(
        output_layout=layout,
        frames_per_second_for_ui=1,
        frames_per_second_for_step=1,
        video_width=_FRAME_SIZE,
        video_height=_FRAME_SIZE,
    )


def _key_event(*, pressed: bool, key: str = _ACTIVATION_KEY) -> UserInputEvents:
    return UserInputEvents(
        [
            UserInputEvent(
                timestamp=uint64(0),
                event_data=KeyboardUserInputEventData(key=key, pressed=pressed),
            )
        ]
    )


def _is_red(result: StepResult) -> bool:
    return bool(
        torch.all(result.output[:, 0] == 1.0) and torch.all(result.output[:, 1:] == 0.0)
    )


def _is_black(result: StepResult) -> bool:
    return bool(torch.all(result.output == 0.0))


def _run(
    initial_events: UserInputEvents | None = None, *, steps: int
) -> ScriptedClientWindow:
    app = create_app()
    app.init([])
    session = app.create_session(_session_desc())
    window = ScriptedClientWindow(initial_events)
    try:
        run_session(session, window, steps=steps)
    finally:
        app.close()
    return window


def _new_session() -> ISession:
    app = create_app()
    app.init([])
    session = app.create_session(_session_desc())
    session.init()
    return session


## Tests


def test_red_screen_holds_red_between_key_edges() -> None:
    # Key down at step 0 and up at step 2. The step in between carries no events,
    # so it exercises held state rather than a repeated key-down.
    session = _new_session()

    frames = [
        session.step(0, _key_event(pressed=True)),
        session.step(1, UserInputEvents([])),
        session.step(2, _key_event(pressed=False)),
        session.step(3, UserInputEvents([])),
    ]

    assert [_is_red(frame) for frame in frames] == [True, True, False, False]


def test_red_screen_ignores_other_keys() -> None:
    session = _new_session()

    assert _is_black(session.step(0, _key_event(pressed=True, key="q")))


def test_red_screen_starts_black_without_input() -> None:
    window = _run(steps=2)

    assert len(window.results) == 2
    assert all(_is_black(result) for result in window.results)


def test_red_screen_turns_red_for_a_key_the_window_already_holds() -> None:
    # The runner collects input before the first step, so a key already down when
    # the run starts applies from step 0.
    window = _run(_key_event(pressed=True), steps=3)

    assert len(window.results) == 3
    assert all(_is_red(result) for result in window.results)


def test_red_screen_frames_match_the_session_desc() -> None:
    window = _run(_key_event(pressed=True), steps=1)

    result = window.results[0]
    assert result.output.shape == (1, 3, 1, _FRAME_SIZE, _FRAME_SIZE)
    assert result.output.dtype is torch.float32
    assert result.frame_count == 1
    assert result.output_layout is VideoTensorLayout.bcthw


def test_session_desc_available_before_any_client_window() -> None:
    # WebRTC precondition: the runtime must be able to describe the output before a
    # client connects, so this has to hold with no window in existence.
    app = create_app()
    app.init([])
    session = app.create_session(_session_desc())
    session.init()

    assert session.session_desc == _session_desc()


def test_create_session_rejects_unsupported_layout() -> None:
    app = create_app()
    app.init([])

    with pytest.raises(ValueError, match="bcthw"):
        app.create_session(_session_desc(layout=VideoTensorLayout.tchw))


def test_create_session_before_init_raises() -> None:
    app = create_app()

    with pytest.raises(RuntimeError, match="init"):
        app.create_session(_session_desc())


def test_reset_releases_the_held_key() -> None:
    app = create_app()
    app.init([])
    session = app.create_session(_session_desc())
    session.init()
    assert _is_red(session.step(0, _key_event(pressed=True)))

    session.reset()

    assert _is_black(session.step(0, UserInputEvents([])))
