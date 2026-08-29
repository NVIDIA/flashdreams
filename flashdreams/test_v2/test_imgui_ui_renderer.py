# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for the V2 Dear ImGui renderer and loop contracts."""

import queue
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from numpy import uint64

from flashdreams.runtime_v2.imgui_ui_loop import ImGuiUILoop
from flashdreams.runtime_v2.imgui_ui_renderer import (
    _ImGui,
    _rgba_pixels,
    _route_imgui_input_events,
)
from flashdreams.runtime_v2.presentation_manager import PresentationManager
from flashdreams.runtime_v2.step_result import StepResult
from flashdreams.runtime_v2.user_input_event import (
    KeyboardInputState,
    KeyboardUserInputEvent,
    MouseUserInputEvent,
)
from flashdreams.runtime_v2.user_input_events import UserInputEvents
from flashdreams.runtime_v2.video_tensor import VideoTensorLayout

pytestmark = pytest.mark.ci_cpu


def _slangpy_events() -> SimpleNamespace:
    return SimpleNamespace(
        KeyboardEvent=lambda: SimpleNamespace(),
        KeyboardEventType=SimpleNamespace(
            key_press="key_press",
            key_release="key_release",
            input="input",
        ),
        KeyCode=SimpleNamespace(space="space"),
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


def test_imgui_input_uses_concrete_pr512_events() -> None:
    bridge = Mock()
    events = UserInputEvents(
        [
            KeyboardUserInputEvent(
                timestamp=uint64(0),
                key=" ",
                state=KeyboardInputState.PRESSED,
            ),
            MouseUserInputEvent(
                timestamp=uint64(1),
                action="button",
                x=0.25,
                y=0.75,
                button=0,
                pressed=True,
            ),
        ]
    )

    _route_imgui_input_events(
        events,
        slangpy=_slangpy_events(),
        bridge=bridge,
        width=400,
        height=200,
    )

    key_event, text_event = [
        call.args[0] for call in bridge.handle_keyboard_event.call_args_list
    ]
    assert (key_event.type, key_event.key) == ("key_press", "space")
    assert (text_event.type, text_event.codepoint) == ("input", ord(" "))
    mouse_event = bridge.handle_mouse_event.call_args.args[0]
    assert mouse_event.type == "button_down"
    assert mouse_event.button == "left"
    assert mouse_event.pos == (100.0, 150.0)


def test_rgba_pixels_accepts_rgb_and_rejects_channel_first() -> None:
    pixels = _rgba_pixels(np.full((3, 4, 3), 12, dtype=np.uint8))

    assert pixels.shape == (3, 4, 4)
    assert pixels.flags.c_contiguous
    assert np.all(pixels[..., :3] == 12)
    assert np.all(pixels[..., 3] == 255)

    with pytest.raises(ValueError, match="HWC RGB/RGBA"):
        _rgba_pixels(torch.zeros(3, 4, 5))


def test_imgui_pixel_images_reuse_textures_with_the_same_shape() -> None:
    texture = Mock()
    device = Mock()
    device.create_texture.return_value = texture
    imgui = SimpleNamespace(
        ImVec2=lambda x, y: (x, y),
        image=Mock(return_value="drawn"),
    )
    bridge = SimpleNamespace(texture_ref=lambda value: ("texture", value))
    slangpy = SimpleNamespace(
        Format=SimpleNamespace(rgba8_unorm_srgb="rgba8"),
        TextureUsage=SimpleNamespace(shader_resource="shader_resource"),
    )
    ui = _ImGui(device, slangpy, imgui, bridge)
    pixels = np.zeros((8, 12, 3), dtype=np.uint8)

    assert ui.image("bev", pixels, size=(120.0, 80.0)) == "drawn"
    ui.image("bev", pixels, size=(120.0, 80.0))

    device.create_texture.assert_called_once()
    assert texture.copy_from_numpy.call_count == 2
    imgui.image.assert_called_with(("texture", texture), (120.0, 80.0))


class _Renderer:
    def __init__(self) -> None:
        self.reset_count = 0
        self.closed = False

    def render(self, step_index, events, step_ui):
        step_ui(SimpleNamespace(), step_index, events)
        return torch.zeros(4, 3, 4)

    def reset(self) -> None:
        self.reset_count += 1

    def close(self) -> None:
        self.closed = True


class _Loop(ImGuiUILoop[None]):
    def step_ui(self, imgui, step_index, events):
        del imgui, step_index, events
        frames = self.presented_model_frames()
        return frames[0] if frames else None


def test_imgui_loop_composites_over_the_presented_model_frame() -> None:
    video = torch.full((1, 3, 3, 4), -0.5)
    presentation = PresentationManager()
    presentation.publish(
        0,
        [StepResult(0, video, 1, VideoTensorLayout.tchw)],
    )
    presentation.advance(0)
    renderer = _Renderer()
    loop = _Loop(renderer=renderer)
    loop.register_session_loop_objects(
        state=None,
        frequency=60,
        shutdown_event=threading.Event(),
        failure_queue=queue.Queue(),
    )
    loop.register_session_ui_loop_objects(
        output_layout=VideoTensorLayout.tchw,
        presentation_manager=presentation,
    )

    result = loop.step(0, UserInputEvents([]))

    output = result.read_output()
    assert output.shape == (1, 3, 3, 4)
    assert torch.all(output == -0.5)
    loop.reset()
    loop.close()
    assert renderer.reset_count == 1
    assert renderer.closed
