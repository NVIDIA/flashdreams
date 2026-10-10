# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU checks for multiview placement, driving controls and stream identity."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from interactive_drive.multiview import (
    AuxiliaryView,
    CameraView,
    MultiviewUILoop,
    MultiviewUIState,
)

from flashdreams.runtime_v2.user_input_events import UserInputEvents

pytestmark = pytest.mark.ci_cpu


class RecordingUI:
    Cond_ = SimpleNamespace(always=1)
    WindowFlags_ = SimpleNamespace(
        no_decoration=1, no_move=2, no_saved_settings=4, no_inputs=8
    )
    ImVec2 = staticmethod(lambda x, y: SimpleNamespace(x=x, y=y))

    def __init__(self):
        self.images = {}
        self.labels = []
        self.panels = 0
        self.sliders = []
        self.clicked = set()
        self.image_sizes = {}
        self.tooltips = []
        self.rectangles = {}

    def __getattr__(self, name):
        if name in {
            "set_next_window_pos",
            "set_next_window_size",
            "separator",
            "same_line",
            "progress_bar",
        }:
            return lambda *args, **kwargs: None
        raise AttributeError(name)

    def set_next_window_pos(self, position, *args):
        self.next_position = position

    def set_next_window_size(self, size, *args):
        self.next_size = size

    def begin(self, title, **kwargs):
        self.panels += 1
        self.rectangles[title] = (
            self.next_position.x,
            self.next_position.y,
            self.next_size.x,
            self.next_size.y,
        )

    def end(self):
        self.panels -= 1

    def text(self, text):
        self.labels.append(text)

    text_wrapped = text

    def combo(self, label, index, options):
        return False, index

    def slider_int(self, label, value, minimum, maximum):
        self.sliders.append(label)
        return False, value

    def button(self, label):
        return label in self.clicked

    def get_text_line_height_with_spacing(self):
        return 17.0

    def is_item_hovered(self):
        return True

    def set_tooltip(self, text):
        self.tooltips.append(text)

    def get_content_region_avail(self):
        return self.ImVec2(300, 180)

    def image(self, key, pixels, *, size):
        assert size[0] > 0 and size[1] > 0
        self.images[key] = pixels
        self.image_sizes[key] = size


@pytest.mark.parametrize("count", range(1, 12))
@pytest.mark.parametrize("selection", [{}, {"visible_count": 1}])
def test_grid_keeps_camera_and_lidar_identity_focus_and_aspect(count, selection):
    cameras = tuple(
        CameraView(str(i), f"Camera {i}", caption=f"Caption {i}") for i in range(count)
    )
    state = MultiviewUIState(
        model_loop=Mock(),
        title="Sensors",
        prompt="",
        scene_options=(),
        cameras=cameras,
        auxiliary_views=(AuxiliaryView("lidar", "LiDAR", "Generated"),),
        **selection,
    )
    loop = MultiviewUILoop(renderer=Mock())
    loop.state = state
    loop.get_ui_loop_size = lambda: (1920, 1080)
    frames = [torch.full((3, 18, 32), i, dtype=torch.uint8) for i in range(count + 1)]
    loop.presented_model_frame = lambda i: frames[i]
    ui = RecordingUI()
    loop.step_ui(ui, 0, UserInputEvents([]))
    assert ui.panels == 0
    shown = selection.get("visible_count", count)
    assert state.visible_count == shown
    assert list(ui.images) == [f"stream-{i}" for i in (*range(shown), count)]
    rects = [r for name, r in ui.rectangles.items() if name.startswith("Stream##")]
    for i, a in enumerate(rects):
        for b in rects[i + 1 :]:
            assert (
                a[0] + a[2] <= b[0]
                or b[0] + b[2] <= a[0]
                or a[1] + a[3] <= b[1]
                or b[1] + b[3] <= a[1]
            )
    for i in (*range(shown), count):
        assert torch.equal(ui.images[f"stream-{i}"], frames[i].permute(1, 2, 0))
        w, h = ui.image_sizes[f"stream-{i}"]
        assert w / h == pytest.approx(32 / 18)
    state.visible_count = 1
    reduced = RecordingUI()
    loop.step_ui(reduced, 1, UserInputEvents([]))
    assert list(reduced.images) == ["stream-0", f"stream-{count}"]
    assert torch.equal(
        reduced.images[f"stream-{count}"], frames[count].permute(1, 2, 0)
    )
    state.focus_channel = count
    ui = RecordingUI()
    loop.step_ui(ui, 1, UserInputEvents([]))
    assert list(ui.images) == [f"stream-{count}"]
    ui.clicked.add("Return to grid")
    loop.step_ui(ui, 2, UserInputEvents([]))
    assert state.focus_channel == -1


@pytest.mark.parametrize("names", [("front",), ("lidar", "lidar")])
def test_duplicate_sensor_names_keep_distinct_panels_and_textures(names):
    state = MultiviewUIState(
        model_loop=Mock(),
        title="Sensors",
        prompt="",
        scene_options=(),
        cameras=(CameraView("front", "Front camera"),),
        auxiliary_views=tuple(AuxiliaryView(name, name) for name in names),
        visible_count=1,
    )
    loop = MultiviewUILoop(renderer=Mock())
    loop.state = state
    loop.get_ui_loop_size = lambda: (1920, 1080)
    frames = [
        torch.full((3, 18, 32), i, dtype=torch.uint8) for i in range(1 + len(names))
    ]
    loop.presented_model_frame = lambda i: frames[i]
    ui = RecordingUI()
    loop.step_ui(ui, 0, UserInputEvents([]))
    assert ui.panels == 0
    assert len([name for name in ui.rectangles if name.startswith("Stream##")]) == len(
        frames
    )
    assert len(ui.images) == len(frames)
    for channel, frame in enumerate(frames):
        assert torch.equal(ui.images[f"stream-{channel}"], frame.permute(1, 2, 0))
        state.focus_channel = channel
        focused = RecordingUI()
        loop.step_ui(focused, 1, UserInputEvents([]))
        assert list(focused.images) == [f"stream-{channel}"]
        assert torch.equal(focused.images[f"stream-{channel}"], frame.permute(1, 2, 0))
