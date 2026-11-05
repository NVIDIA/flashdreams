# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for the shared action-to-video application."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, cast

import pytest
import tomli as tomllib
import torch
from action2v import (
    Action2VModelLoop,
    Action2VSession,
    ActionEventAccumulator,
    ActionSnapshot,
)
from action2v.dummy import create_app as create_dummy_app
from numpy import uint64

from flashdreams.api_v2.client_window import IClientWindow
from flashdreams.api_v2.user_input_event import UserInputEvent
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


def test_event_accumulator_ignores_input_until_focus_returns() -> None:
    """Keep UI hover and clicks out of model actions after capture is released."""
    accumulator = ActionEventAccumulator()
    accumulator.consume(
        _events(FocusUserInputEvent(timestamp=uint64(0), focused=False))
    )
    ignored = accumulator.consume(
        _events(
            KeyboardUserInputEvent(
                timestamp=uint64(1), key="w", state=KeyboardInputState.PRESSED
            ),
            MouseUserInputEvent(timestamp=uint64(2), action="move", x=0.1, y=0.1),
            MouseUserInputEvent(timestamp=uint64(3), action="move", x=0.9, y=0.9),
            MouseUserInputEvent(
                timestamp=uint64(4), action="button", button=0, pressed=True
            ),
            MouseUserInputEvent(timestamp=uint64(5), action="wheel", wheel_y=1),
        )
    )
    assert ignored == ActionSnapshot()

    resumed = accumulator.consume(
        _events(
            FocusUserInputEvent(timestamp=uint64(6), focused=True),
            MouseUserInputEvent(timestamp=uint64(7), action="move", x=0.4, y=0.4),
            MouseUserInputEvent(timestamp=uint64(8), action="move", x=0.5, y=0.6),
            KeyboardUserInputEvent(
                timestamp=uint64(9), key="w", state=KeyboardInputState.PRESSED
            ),
        )
    )
    assert resumed.keys == frozenset({"W"})
    assert resumed.mouse_dx == pytest.approx(0.1)
    assert resumed.mouse_dy == pytest.approx(0.2)
    assert resumed.mouse_buttons == frozenset()


@pytest.mark.parametrize("pointer_first", [False, True])
@pytest.mark.parametrize("button_held", [False, True])
def test_recapture_preserves_held_controls_across_channel_delivery_orders(
    pointer_first: bool, button_held: bool
) -> None:
    """Keep the recapture press when either channel outruns the other's blur."""
    pytest.importorskip("aiortc")
    from flashdreams.runtime_v2.webrtc_client_window import WebRTCClientWindow

    window = WebRTCClientWindow()
    inputs = ActionEventAccumulator()

    def packet(generation: int, owns_focus: bool, **payload: Any) -> str:
        return json.dumps(
            {
                **payload,
                "focus_context": {"generation": generation, "focused": owns_focus},
            }
        )

    press = {
        "type": "mouse",
        "action": "button",
        "x": 0.5,
        "y": 0.5,
        "button": 0,
        "pressed": True,
    }
    try:
        window.open(
            SessionDesc(
                output_layout=VideoTensorLayout.tchw, video_width=4, video_height=4
            )
        )
        window.server._buffer_browser_message(
            packet(1, True, type="focus", focused=True)
        )
        window.server._buffer_browser_message(
            packet(1, True, type="keyboard", key="w", pressed=True)
        )
        if button_held:
            window.server._buffer_browser_message(packet(1, True, **press))
        inputs.consume(window.get_user_input_events())
        controls = [
            packet(2, False, type="keyboard", key="w", pressed=False),
            packet(2, False, type="focus", focused=False),
            packet(3, True, type="focus", focused=True),
            packet(3, True, type="keyboard", key="d", pressed=True),
        ]
        pointer = [packet(1, True, type="mouse", action="move", x=0.99, y=0.99)]
        if button_held:
            pointer.append(
                packet(2, False, **{**press, "pressed": False, "cancelled": True})
            )
        pointer.append(packet(3, True, **press))
        for message in pointer + controls if pointer_first else controls + pointer:
            window.server._buffer_browser_message(message)
        snapshot = inputs.consume(window.get_user_input_events())
        assert snapshot.keys == frozenset({"D"})
        assert snapshot.mouse_buttons == frozenset({0})
        assert snapshot.mouse_dx == snapshot.mouse_dy == 0
    finally:
        window.close()


class _DiffusionModel:
    dtype = torch.float32

    def __init__(self) -> None:
        self.rng = torch.Generator().manual_seed(3)


class _Pipeline:
    """Small standard-pipeline stand-in recording actions and lifecycle calls."""

    device = torch.device("cpu")

    def __init__(self) -> None:
        self.diffusion_model = _DiffusionModel()
        self.actions: list[Any] = []
        self.initialized_caches: list[dict[str, int]] = []

    def initialize_cache(self, *, seed_pixels: torch.Tensor) -> dict[str, int]:
        assert seed_pixels.shape == (1, 4, 3, 2, 4)
        cache = {"autoregressive_index": 0}
        self.initialized_caches.append(cache)
        return cache

    def generate(
        self,
        *,
        autoregressive_index: int,
        cache: dict[str, int],
        input: Any,
    ) -> torch.Tensor:
        cache["autoregressive_index"] = autoregressive_index
        self.actions.append(input)
        return torch.full((1, 4, 3, 2, 4), autoregressive_index / 10)

    def finalize(
        self,
        *,
        autoregressive_index: int,
        cache: dict[str, int],
    ) -> dict[str, int]:
        assert cache["autoregressive_index"] == autoregressive_index
        return {"model_step": autoregressive_index}


def _session(pipeline: _Pipeline) -> Action2VSession:
    return Action2VSession(
        pipeline=pipeline,
        pipeline_lock=threading.Lock(),
        session_desc=SessionDesc(
            output_layout=VideoTensorLayout.tchw,
            video_width=4,
            video_height=2,
        ),
        seed_frames=torch.zeros((4, 3, 2, 4)),
        seed=3,
        action_mapper=lambda snapshot: snapshot,
        total_blocks=10,
    )


class _CursorWindow(IClientWindow):
    """Record cursor requests and reject them outside the open lifecycle."""

    def __init__(self) -> None:
        self.opened = False
        self.cursor_requests: list[tuple[str, bool]] = []

    def put_user_input_event(self, event: UserInputEvent) -> None:
        del event

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


class _KeyWindow(_CursorWindow):
    """Report one keyboard event while recording the normal window lifecycle."""

    def __init__(self, key: str) -> None:
        super().__init__()
        self._key = key

    def get_user_input_events(self) -> UserInputEvents:
        key = self._key
        if not key:
            return UserInputEvents([])
        self._key = ""
        return _events(
            KeyboardUserInputEvent(
                timestamp=uint64(0), key=key, state=KeyboardInputState.PRESSED
            )
        )


def test_action2v_requests_cursor_options_only_after_window_opens() -> None:
    session = _session(_Pipeline())
    window = _CursorWindow()

    run_session(session, window, steps=1)

    assert window.cursor_requests == [("hide", True), ("lock", True)]


def test_t_key_requests_a_new_session_by_default() -> None:
    """Route the default binding through the UI loop's new-session request."""
    session = _session(_Pipeline())
    window = _KeyWindow("t")

    next_session_desc = run_session(session, window)

    assert next_session_desc is session.session_desc
    window.close()


def test_shared_session_emits_seed_then_live_actions_and_resets() -> None:
    """Drive seed, live input, and reset through the shared loop."""
    pipeline = _Pipeline()
    session = _session(pipeline)
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
    assert pipeline.actions == [
        ActionSnapshot(keys=frozenset({"W"})),
        ActionSnapshot(keys=frozenset({"W"})),
    ]
    assert not loop.is_finished()
    loop.reset()
    assert len(pipeline.initialized_caches) == 2
    assert not loop.is_finished()
    loop.close()
    assert loop.state.cache is None


def test_dummy_application_runs_without_a_model() -> None:
    """Exercise application discovery inputs and one CPU generation step."""
    first_frame = Path(__file__).parents[1] / "action2v" / "assets" / "dummy_frame.ppm"
    app = create_dummy_app()
    app.init(
        [
            "--image-path",
            str(first_frame),
            "--seed",
            "3",
            "--total-blocks",
            "1",
            "--no-ui",
        ]
    )
    assert app.pipeline_config.diffusion_model.seed == 3
    session_desc = app.session_desc()
    assert session_desc is not None
    session = app.create_session(session_desc)
    session.init()
    with pytest.raises(RuntimeError, match="has not registered a UI loop"):
        session.ui_loop
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
    assert loop.is_finished()


def test_help_documents_the_live_application_options(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Expose every shared application option under its public name."""
    with pytest.raises(SystemExit) as exit_info:
        create_dummy_app().init(["--help"])

    assert exit_info.value.code == 0
    help_text = capsys.readouterr().out
    for option in (
        "--image-path",
        "--example-data",
        "--device",
        "--total-blocks",
        "--ui",
        "--seed",
        "--mouse-sensitivity",
        "--reset-key",
    ):
        assert option in help_text
    for option in (
        "--first-frame",
        "--profile",
        "--seed-image",
        "--actions",
        "--actions-file",
        "--controls-file",
    ):
        assert option not in help_text

        with pytest.raises(SystemExit) as invalid_option:
            create_dummy_app().init([option, "input.json"])
        assert invalid_option.value.code == 2


@pytest.mark.parametrize("reset_key", ["", "1", "tt", "é"])
def test_reset_key_rejects_non_ascii_letters(reset_key: str) -> None:
    """Reject reset bindings outside the documented single-letter range."""
    with pytest.raises(SystemExit) as exit_info:
        create_dummy_app().init(["--reset-key", reset_key])

    assert exit_info.value.code == 2


def test_reset_key_cli_option_rebinds_the_session_key() -> None:
    """Pass a normalized reset binding from the application into its UI loop."""
    first_frame = Path(__file__).parents[1] / "action2v" / "assets" / "dummy_frame.ppm"
    app = create_dummy_app()
    app.init(["--image-path", str(first_frame), "--reset-key", "r"])
    session = app.create_session(app.session_desc())
    window = _KeyWindow("R")

    next_session_desc = run_session(session, window)

    assert next_session_desc is session.session_desc
    window.close()
    app.close()


def test_shared_package_registers_the_dummy_application() -> None:
    """Keep the CPU demo discoverable through ``flashdreams-run-v2``."""
    package_root = Path(__file__).parents[1]
    with (package_root / "pyproject.toml").open("rb") as stream:
        manifest = tomllib.load(stream)
    entry_points = manifest["project"]["entry-points"]["flashdreams.applications_v2"]
    assert entry_points == {"action2v-dummy": "action2v.dummy:create_app"}
