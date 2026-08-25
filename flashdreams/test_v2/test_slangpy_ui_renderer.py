# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for the v2 SlangPy UI renderer."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from flashdreams.runtime_v2.slangpy_ui_renderer import _route_input_events
from flashdreams.runtime_v2.user_input_event import (
    MouseUserInputEvent,
)
from flashdreams.runtime_v2.user_input_events import UserInputEvents
from numpy import uint64

pytestmark = pytest.mark.ci_cpu


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
