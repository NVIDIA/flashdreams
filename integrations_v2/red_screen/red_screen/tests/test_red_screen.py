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
from flashdreams.runtime_v2.session_desc import (
    BackpressureMode,
    PresentationMode,
    SessionDesc,
)
from flashdreams.runtime_v2.session_runner import run_session
from flashdreams.runtime_v2.step_result import StepResult
from flashdreams.runtime_v2.user_input_event import (
    KeyboardInputState,
    KeyboardUserInputEvent,
)
from flashdreams.runtime_v2.user_input_events import UserInputEvents
from flashdreams.runtime_v2.video_tensor import VideoTensorLayout

pytestmark = pytest.mark.ci_cpu

_ACTIVATION_KEY = "r"
"""Matches the application's default activation key."""

_FRAME_SIZE = 2
"""Frame width and height; small enough to assert over every pixel."""


class ScriptedClientWindow(IClientWindow):
    """Report one batch of input per poll and record what is presented.

    The runner polls on its own thread, so a window cannot aim an event at a
    chosen step. It can still deliver input part way through a run: the first poll
    happens before generation starts, and a run that blocks on a full queue cannot
    start another step until a later poll has happened.
    """

    def __init__(self, batches: list[UserInputEvents] | None = None) -> None:
        """
        Args:
            batches: Events to report, one entry per poll. Polls past the end of
                the script report nothing.
        """
        self.session_desc: SessionDesc | None = None
        self.results: list[StepResult] = []
        self._batches = list(batches or [])
        self._lock = threading.Lock()
        self._is_open = False

    def get_user_input_events(self) -> UserInputEvents:
        with self._lock:
            if self._batches:
                return self._batches.pop(0)
        return UserInputEvents([])

    def open(self, session_desc: SessionDesc) -> None:
        self.session_desc = session_desc
        self._is_open = True

    def write(self, result: StepResult) -> None:
        assert self._is_open
        self.results.append(result)

    def close(self) -> None:
        self._is_open = False


## Helpers


def _session_desc(
    layout: VideoTensorLayout = VideoTensorLayout.bcthw,
    *,
    frames_per_second_for_ui: int = 1,
) -> SessionDesc:
    return SessionDesc(
        output_layout=layout,
        backpressure_mode=BackpressureMode.BLOCK,
        presentation_mode=PresentationMode.ON_DEMAND,
        frames_per_second_for_ui=frames_per_second_for_ui,
        frames_per_second_for_step=1,
        video_width=_FRAME_SIZE,
        video_height=_FRAME_SIZE,
    )


def _key_event(*, pressed: bool, key: str = _ACTIVATION_KEY) -> UserInputEvents:
    return UserInputEvents(
        [
            KeyboardUserInputEvent(
                timestamp=uint64(0),
                key=key,
                state=(
                    KeyboardInputState.PRESSED
                    if pressed
                    else KeyboardInputState.RELEASED
                ),
            )
        ]
    )


def _is_red(result: StepResult) -> bool:
    # Frames carry [-1, 1], so full red is 1.0 and the other channels are -1.0.
    output = result.read_output()
    return bool(torch.all(output[:, 0] == 1.0) and torch.all(output[:, 1:] == -1.0))


def _is_black(result: StepResult) -> bool:
    return bool(torch.all(result.read_output() == -1.0))


def _step(session: ISession, step_index: int, events: UserInputEvents) -> StepResult:
    results = session.model_loop.step(step_index, events)
    assert isinstance(results, list)
    result = results[0]
    assert isinstance(result, StepResult)
    return result


def _run(
    initial_events: UserInputEvents | None = None, *, steps: int
) -> ScriptedClientWindow:
    app = create_app()
    app.init([])
    session = app.create_session(_session_desc())
    window = ScriptedClientWindow([initial_events] if initial_events else None)
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
        _step(session, 0, _key_event(pressed=True)),
        _step(session, 1, UserInputEvents([])),
        _step(session, 2, _key_event(pressed=False)),
        _step(session, 3, UserInputEvents([])),
    ]

    assert [_is_red(frame) for frame in frames] == [True, True, False, False]


def test_red_screen_ignores_other_keys() -> None:
    session = _new_session()

    assert _is_black(_step(session, 0, _key_event(pressed=True, key="q")))


def test_red_screen_uses_last_event_to_adjust_color_intensity() -> None:
    session = _new_session()

    increased = _step(session, 0, _key_event(pressed=True, key="w"))
    last_event_decreases = _step(
        session,
        1,
        UserInputEvents(
            [
                KeyboardUserInputEvent(
                    timestamp=uint64(0),
                    key="w",
                    state=KeyboardInputState.PRESSED,
                ),
                KeyboardUserInputEvent(
                    timestamp=uint64(1),
                    key="s",
                    state=KeyboardInputState.PRESSED,
                ),
            ]
        ),
    )

    increased_output = increased.read_output()
    assert torch.allclose(
        increased_output[:, 0],
        torch.full_like(increased_output[:, 0], -0.8),
    )
    assert _is_black(last_event_decreases)


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


def test_red_screen_turns_red_for_a_key_pressed_during_the_run() -> None:
    # End to end through the runner: the key is not held at the start, so only
    # input delivered while the run is going can turn the screen red.
    app = create_app()
    app.init([])
    session = app.create_session(_session_desc(frames_per_second_for_ui=100))
    window = ScriptedClientWindow([UserInputEvents([]), _key_event(pressed=True)])

    # The single-slot pending chunk queue means a step cannot start until a tick
    # has presented the previous one. Every tick polls input before it presents,
    # which makes the key reach a step deterministically.
    try:
        run_session(session, window, steps=3)
    finally:
        app.close()

    assert len(window.results) == 3
    assert _is_black(window.results[0])
    assert _is_red(window.results[-1])


def test_red_screen_frames_match_the_session_desc() -> None:
    window = _run(_key_event(pressed=True), steps=1)

    result = window.results[0]
    output = result.read_output()
    assert output.shape == (1, 3, 1, _FRAME_SIZE, _FRAME_SIZE)
    assert output.dtype is torch.float32
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
    assert _is_red(_step(session, 0, _key_event(pressed=True)))

    session.model_loop.reset()

    assert _is_black(_step(session, 0, UserInputEvents([])))
