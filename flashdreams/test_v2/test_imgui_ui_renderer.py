# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for the V2 Dear ImGui renderer and loop contracts."""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from numpy import uint64

from flashdreams.runtime_v2.imgui_ui_loop import ImGuiUILoop
from flashdreams.runtime_v2.imgui_ui_renderer import (
    _ImGui,
    _ImGuiUIRenderer,
    _rgba_pixels,
    _route_imgui_input_events,
)
from flashdreams.runtime_v2.session_desc import SessionDesc
from flashdreams.runtime_v2.user_input_event import (
    KeyboardInputState,
    KeyboardUserInputEvent,
    MouseUserInputEvent,
    ViewportSizeUserInputEvent,
)
from flashdreams.runtime_v2.user_input_events import UserInputEvents

pytestmark = pytest.mark.ci_cpu


def test_imgui_renderer_uses_an_explicit_cuda_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    renderer = _ImGuiUIRenderer(width=4, height=3, cuda_device="cuda:1")
    contexts: list[tuple[str, torch.device]] = []

    class _DeviceContext:
        def __init__(self, device: torch.device) -> None:
            self.device = torch.device(device)

        def __enter__(self) -> None:
            contexts.append(("enter", self.device))

        def __exit__(self, *_args: object) -> None:
            contexts.append(("exit", self.device))

    expected = torch.zeros(4, 3, 4)
    monkeypatch.setattr(torch.cuda, "device", _DeviceContext)
    monkeypatch.setattr(renderer, "_render", Mock(return_value=expected))

    assert renderer._cuda_device == torch.device("cuda:1")
    assert renderer.render(0, UserInputEvents([]), Mock()) is expected
    assert contexts == [
        ("enter", torch.device("cuda:1")),
        ("exit", torch.device("cuda:1")),
    ]
    with pytest.raises(ValueError, match="requires a CUDA device"):
        _ImGuiUIRenderer(width=4, height=3, cuda_device="cpu")


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
    ui.image("bev", pixels, size=(120.0, 80.0), uv0=(0.1, 0.0), uv1=(0.9, 1.0))
    imgui.image.assert_called_with(
        ("texture", texture), (120.0, 80.0), uv0=(0.1, 0.0), uv1=(0.9, 1.0)
    )


def test_browser_viewport_resizes_ui_and_presentation_without_changing_model_shape():
    renderer = Mock(width=4, height=3)
    renderer.resize.side_effect = lambda w, h: (
        setattr(renderer, "width", w),
        setattr(renderer, "height", h),
    )
    renderer.render.side_effect = lambda *_: torch.zeros(
        4, renderer.height, renderer.width
    )
    loop = _Loop(renderer=renderer)
    manager = Mock()
    manager.composite.side_effect = (
        lambda background, overlay: overlay[:3] if background is None else background
    )
    desc = SessionDesc(video_width=4, video_height=3)
    loop.register_session_ui_loop_objects(
        session_desc=desc, presentation_manager=manager
    )
    events = UserInputEvents(
        [ViewportSizeUserInputEvent(timestamp=uint64(0), width=8, height=6)]
    )
    result = loop.step(0, events)[0]
    assert result._output.shape[-2:] == (6, 8)
    assert loop.flush_ui_loop_requests().new_window_size == (8, 6)
    assert (desc.video_width, desc.video_height) == (4, 3)
    loop.step(1, events)
    renderer.resize.assert_called_once_with(8, 6)
    assert loop.flush_ui_loop_requests() is None


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
