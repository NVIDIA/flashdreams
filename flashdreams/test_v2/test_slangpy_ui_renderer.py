# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for the v2 SlangPy UI renderer."""

from enum import Enum
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from numpy import uint64

from flashdreams.runtime_v2 import slangpy_ui_renderer as renderer_module
from flashdreams.runtime_v2.slangpy_ui_renderer import (
    _route_input_events,
    _SlangPyUI,
    _SlangPyUIRenderer,
)
from flashdreams.runtime_v2.user_input_event import (
    FocusUserInputEvent,
    InputOwnerChangedUserInputEvent,
    KeyboardInputState,
    KeyboardUserInputEvent,
    MouseUserInputEvent,
)
from flashdreams.runtime_v2.user_input_events import UserInputEvents

pytestmark = pytest.mark.ci_cpu


def test_space_key_routes_a_text_input_codepoint() -> None:
    ui_context = Mock()
    slangpy = SimpleNamespace(
        KeyboardEvent=lambda: SimpleNamespace(),
        KeyboardEventType=SimpleNamespace(
            key_press="key_press",
            key_release="key_release",
            input="input",
        ),
        KeyCode=SimpleNamespace(space="space"),
        KeyModifierFlags=SimpleNamespace(none="none"),
    )
    events = UserInputEvents(
        [
            KeyboardUserInputEvent(
                timestamp=uint64(0),
                key=" ",
                state=KeyboardInputState.PRESSED,
            )
        ]
    )

    _route_input_events(
        events,
        ui_context=ui_context,
        slangpy=slangpy,
        width=1,
        height=1,
    )

    key_event, text_event = [
        call.args[0] for call in ui_context.handle_keyboard_event.call_args_list
    ]
    assert (key_event.type, key_event.key) == ("key_press", "space")
    assert (text_event.type, text_event.codepoint) == ("input", ord(" "))


def test_shifted_character_preserves_key_and_text_input() -> None:
    ui_context = Mock()
    slangpy = SimpleNamespace(
        KeyboardEvent=lambda: SimpleNamespace(),
        KeyboardEventType=SimpleNamespace(
            key_press="key_press",
            key_release="key_release",
            input="input",
        ),
        KeyCode=SimpleNamespace(a="a"),
        KeyModifierFlags=SimpleNamespace(none="none"),
    )
    events = UserInputEvents(
        [
            KeyboardUserInputEvent(
                timestamp=uint64(0),
                key="A",
                state=KeyboardInputState.PRESSED,
            )
        ]
    )

    _route_input_events(
        events,
        ui_context=ui_context,
        slangpy=slangpy,
        width=1,
        height=1,
    )

    key_event, text_event = [
        call.args[0] for call in ui_context.handle_keyboard_event.call_args_list
    ]
    assert (key_event.type, key_event.key) == ("key_press", "a")
    assert (text_event.type, text_event.codepoint) == ("input", ord("A"))


def test_mouse_input_is_routed_through_slangpy_ui_context() -> None:
    ui_context = Mock()
    slangpy = SimpleNamespace(
        KeyModifierFlags=SimpleNamespace(none="none"),
        MouseButton=SimpleNamespace(left="left", middle="middle", right="right"),
        MouseEvent=lambda: SimpleNamespace(),
        MouseEventType=SimpleNamespace(
            button_down="button_down",
            button_up="button_up",
            move="move",
            scroll="scroll",
        ),
    )
    events = UserInputEvents(
        [
            MouseUserInputEvent(
                timestamp=uint64(0),
                action="button",
                x=0.25,
                y=0.75,
                button=0,
                pressed=True,
            )
        ]
    )

    _route_input_events(
        events,
        ui_context=ui_context,
        slangpy=slangpy,
        width=400,
        height=200,
    )

    routed = ui_context.handle_mouse_event.call_args.args[0]
    assert routed.type == "button_down"
    assert routed.button == "left"
    assert routed.pos == (100.0, 150.0)


@pytest.mark.parametrize(
    "focus_type", [FocusUserInputEvent, InputOwnerChangedUserInputEvent]
)
def test_slangpy_focus_loss_releases_inputs_before_new_owner_edges(focus_type) -> None:
    key_codes = Enum("key_codes", ["a", "space", "unknown"])
    held_keys = {key_codes.a}
    held_buttons = {"left", "middle", "right"}
    mouse_moves = []
    slangpy = SimpleNamespace(
        KeyboardEvent=lambda: SimpleNamespace(),
        KeyboardEventType=SimpleNamespace(
            key_press="key_press", key_release="key_release", input="input"
        ),
        KeyCode=key_codes,
        KeyModifierFlags=SimpleNamespace(none="none"),
        MouseButton=SimpleNamespace(left="left", middle="middle", right="right"),
        MouseEvent=lambda: SimpleNamespace(),
        MouseEventType=SimpleNamespace(
            button_down="button_down", button_up="button_up", move="move"
        ),
    )

    def keyboard(event) -> None:
        if event.type == "key_press":
            held_keys.add(event.key)
        elif event.type == "key_release":
            held_keys.discard(event.key)

    def mouse(event) -> None:
        if event.type == "move":
            mouse_moves.append(event.pos)
        elif event.type == "button_up":
            assert mouse_moves[-1] == (-1.0, -1.0)
            held_buttons.discard(event.button)
        elif event.type == "button_down":
            held_buttons.add(event.button)

    _route_input_events(
        UserInputEvents(
            [
                focus_type(timestamp=uint64(1), focused=False),
                KeyboardUserInputEvent(
                    timestamp=uint64(2), key=" ", state=KeyboardInputState.PRESSED
                ),
                MouseUserInputEvent(
                    timestamp=uint64(3), action="button", button=2, pressed=True
                ),
            ]
        ),
        ui_context=SimpleNamespace(
            handle_keyboard_event=keyboard, handle_mouse_event=mouse
        ),
        slangpy=slangpy,
        width=400,
        height=200,
    )

    assert held_keys == {key_codes.space}
    assert held_buttons == {"right"}


def test_owner_change_replaces_native_context_and_preserves_widgets(
    monkeypatch,
) -> None:
    class Screen:
        def __init__(self):
            self.children = []
            self.visible = True
            self.enabled = True

        def remove_all_children(self):
            for child in self.children:
                child.parent = None
            self.children.clear()

        def add_child(self, child):
            child.parent = self
            self.children.append(child)

    old_screen = Screen()
    old_screen.enabled = False
    widget = SimpleNamespace(
        position=(10, 20), size=(400, 200), value="retained", callback=Mock()
    )
    old_screen.add_child(widget)
    old_context = SimpleNamespace(screen=old_screen, pending=["old input"])
    new_context = SimpleNamespace(
        screen=Screen(), pending=[], begin_frame=Mock(), end_frame=Mock()
    )
    device = Mock()
    stream = Mock()

    def create_context(actual_device):
        assert actual_device is device
        stream.synchronize.assert_called_once()
        device.wait_for_idle.assert_called_once()
        return new_context

    slangpy = SimpleNamespace(
        ui=SimpleNamespace(Context=create_context, Window=SimpleNamespace)
    )
    renderer = _SlangPyUIRenderer(width=2, height=2, slangpy_module=slangpy)
    renderer._device = device
    renderer._ui_context = old_context
    renderer._ui = _SlangPyUI(slangpy.ui, old_screen)
    renderer._target = object()
    renderer._rgba_buffer = object()
    renderer._rgba_tensor = torch.zeros((2, 2, 4), dtype=torch.uint8)
    monkeypatch.setattr(renderer_module, "_current_cuda_stream", lambda: 0)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: stream)

    def route(events, *, ui_context, **kwargs):
        assert ui_context is new_context
        assert not ui_context.pending
        ui_context.pending.extend(events.get_events()[1:])

    monkeypatch.setattr(renderer_module, "_route_input_events", route)
    new_key = KeyboardUserInputEvent(
        timestamp=uint64(2), key="w", state=KeyboardInputState.PRESSED
    )
    renderer.render(
        5,
        UserInputEvents(
            [InputOwnerChangedUserInputEvent(timestamp=uint64(1)), new_key]
        ),
        Mock(),
    )

    assert renderer._ui_context is new_context
    assert renderer._ui.screen is new_context.screen
    assert new_context.pending == [new_key]
    assert old_context.pending == ["old input"]
    assert not old_screen.children
    assert new_context.screen.children == [widget]
    assert new_context.screen.children[0] is widget
    assert widget.parent is new_context.screen
    assert (widget.position, widget.size, widget.value) == (
        (10, 20),
        (400, 200),
        "retained",
    )
    assert not new_context.screen.enabled
    widget.callback.assert_not_called()
