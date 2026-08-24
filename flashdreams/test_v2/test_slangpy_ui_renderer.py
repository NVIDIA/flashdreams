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

"""CPU tests for the v2 SlangPy UI renderer."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from numpy import uint64

from flashdreams.runtime_v2._slangpy_ui_renderer import _route_input_events
from flashdreams.runtime_v2.user_input_event import (
    MouseUserInputEventData,
    UserInputEvent,
)
from flashdreams.runtime_v2.user_input_events import UserInputEvents

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
            UserInputEvent(
                timestamp=uint64(0),
                event_data=MouseUserInputEventData(
                    action="button",
                    x=0.25,
                    y=0.75,
                    button=0,
                    pressed=True,
                ),
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
