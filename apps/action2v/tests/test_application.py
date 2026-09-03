# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for the shared action-to-video application."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest
import tomli as tomllib
import torch
from action2v import (
    Action2VModelLoop,
    Action2VSession,
    Action2VStep,
    ActionEventAccumulator,
    ActionSnapshot,
)
from action2v.dummy import create_app as create_dummy_app
from numpy import uint64

from flashdreams.api_v2.client_window import IClientWindow
from flashdreams.runtime_v2.session_desc import SessionDesc
from flashdreams.runtime_v2.session_runner import run_session
from flashdreams.runtime_v2.step_result import StepResult
from flashdreams.runtime_v2.user_input_event import (
    FocusUserInputEvent,
    KeyboardInputState,
    KeyboardUserInputEvent,
    MouseUserInputEvent,
)
from flashdreams.runtime_v2.user_input_events import UserInputEvents
from flashdreams.runtime_v2.video_tensor import VideoTensorLayout

pytestmark = pytest.mark.ci_cpu


def _events(*events):
    return UserInputEvents(list(events))


def test_event_accumulator_separates_held_and_transient_input() -> None:
    """Keep held controls while consuming motion and wheel deltas once."""
    accumulator = ActionEventAccumulator()
    first = accumulator.consume(
        _events(
            KeyboardUserInputEvent(
                timestamp=uint64(0), key="w", state=KeyboardInputState.PRESSED
            ),
            MouseUserInputEvent(timestamp=uint64(1), action="move", x=0.2, y=0.3),
            MouseUserInputEvent(timestamp=uint64(2), action="move", x=0.5, y=0.1),
            MouseUserInputEvent(
                timestamp=uint64(3), action="button", button=0, pressed=True
            ),
            MouseUserInputEvent(timestamp=uint64(4), action="wheel", wheel_y=1.5),
        )
    )
    held = accumulator.consume(UserInputEvents([]))
    cleared = accumulator.consume(
        _events(FocusUserInputEvent(timestamp=uint64(5), focused=False))
    )

    assert first.keys == frozenset({"W"})
    assert first.mouse_buttons == frozenset({0})
    assert first.mouse_dx == pytest.approx(0.3)
    assert first.mouse_dy == pytest.approx(-0.2)
    assert first.wheel_y == 1.5
    assert held == ActionSnapshot(keys=frozenset({"W"}), mouse_buttons=frozenset({0}))
    assert cleared == ActionSnapshot()


class _ModelSession:
    """Small model-session stand-in recording actions and lifecycle calls."""

    def __init__(self) -> None:
        self.seed_frames = torch.zeros((4, 3, 2, 4))
        self.actions: list[Any] = []
        self.resets = 0
        self.closed = False

    def step(self, step_index: int, action: Any) -> Action2VStep:
        self.actions.append(action)
        return Action2VStep(
            frames=torch.full((4, 3, 2, 4), step_index / 10),
            metrics={"model_step": step_index},
        )

    def reset(self) -> None:
        self.resets += 1

    def close(self) -> None:
        self.closed = True


class _CursorWindow(IClientWindow):
    """Record cursor requests and reject them outside the open lifecycle."""

    def __init__(self) -> None:
        self.opened = False
        self.cursor_requests: list[tuple[str, bool]] = []

    def open(self, session_desc: SessionDesc) -> None:
        del session_desc
        self.opened = True

    def request_hide_cursor(self, hide_cursor: bool) -> None:
        assert self.opened
        self.cursor_requests.append(("hide", hide_cursor))

    def request_lock_cursor_to_window(self, lock_cursor_to_window: bool) -> None:
        super().request_lock_cursor_to_window(lock_cursor_to_window)
        assert self.opened
        self.cursor_requests.append(("lock", lock_cursor_to_window))

    def get_user_input_events(self) -> UserInputEvents:
        return UserInputEvents([])

    def write(self, result: StepResult) -> None:
        del result

    def close(self) -> None:
        self.opened = False


def test_action2v_requests_cursor_options_only_after_window_opens() -> None:
    session = Action2VSession(
        model_session_factory=_ModelSession,
        session_desc=SessionDesc(video_width=4, video_height=2),
        action_mapper=lambda snapshot: snapshot,
    )
    window = _CursorWindow()

    run_session(session, window, steps=1)

    assert window.cursor_requests == [("hide", True), ("lock", True)]


def test_shared_session_emits_seed_then_live_actions_and_resets() -> None:
    """Drive seed, live input, and reset through the shared loop."""
    model_session = _ModelSession()
    session = Action2VSession(
        model_session_factory=lambda: model_session,
        session_desc=SessionDesc(
            output_layout=VideoTensorLayout.tchw,
            video_width=4,
            video_height=2,
        ),
        action_mapper=lambda snapshot: snapshot,
    )
    session.init()
    loop = cast(Action2VModelLoop, session.model_loop)

    seed = loop.step(0, UserInputEvents([]))[0]
    pressed = _events(
        KeyboardUserInputEvent(
            timestamp=uint64(0), key="w", state=KeyboardInputState.PRESSED
        )
    )
    first = loop.step(1, pressed)[0]
    second = loop.step(2, UserInputEvents([]))[0]

    assert [seed.frame_count, first.frame_count, second.frame_count] == [4, 4, 4]
    assert model_session.actions == [
        ActionSnapshot(keys=frozenset({"W"})),
        ActionSnapshot(keys=frozenset({"W"})),
    ]
    assert first.metrics == {
        "model_step": 1,
        "autoregressive_index": 1,
        "generated_frames": 4,
    }
    assert not loop.is_finished()
    loop.reset()
    assert model_session.resets == 1
    assert not loop.is_finished()
    loop.close()
    assert model_session.closed


def test_dummy_application_runs_without_a_model() -> None:
    """Exercise application discovery inputs and one CPU generation step."""
    first_frame = Path(__file__).parents[1] / "action2v" / "assets" / "dummy_frame.ppm"
    app = create_dummy_app()
    app.init(["--first-frame", str(first_frame), "--seed", "3"])
    session_desc = app.session_desc()
    assert session_desc is not None
    session = app.create_session(session_desc)
    session.init()
    loop = cast(Action2VModelLoop, session.model_loop)

    seed = loop.step(0, UserInputEvents([]))[0]
    generated = loop.step(1, UserInputEvents([]))[0]

    assert (
        seed.read_output().shape
        == generated.read_output().shape
        == (
            4,
            3,
            180,
            320,
        )
    )
    assert generated.metrics["dummy_step"] == 1
    assert not loop.is_finished()


def test_help_documents_the_live_application_options(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Expose every shared application option under its public name."""
    with pytest.raises(SystemExit) as exit_info:
        create_dummy_app().init(["--help"])

    assert exit_info.value.code == 0
    help_text = capsys.readouterr().out
    for option in (
        "--first-frame",
        "--seed",
        "--device",
        "--profile",
        "--mouse-sensitivity",
    ):
        assert option in help_text
    for option in (
        "--seed-image",
        "--example-data",
        "--actions",
        "--actions-file",
        "--controls-file",
    ):
        assert option not in help_text

        with pytest.raises(SystemExit) as invalid_option:
            create_dummy_app().init([option, "input.json"])
        assert invalid_option.value.code == 2


def test_shared_package_registers_the_dummy_application() -> None:
    """Keep the CPU demo discoverable through ``flashdreams-run-v2``."""
    package_root = Path(__file__).parents[1]
    with (package_root / "pyproject.toml").open("rb") as stream:
        manifest = tomllib.load(stream)
    entry_points = manifest["project"]["entry-points"]["flashdreams.applications_v2"]
    assert entry_points == {"action2v-dummy": "action2v.dummy:create_app"}
