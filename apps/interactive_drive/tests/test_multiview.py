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
    sensor_layout,
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
        self.image_positions = {}
        self.cursor_x, self.cursor_y = 11.0, 23.0
        self.tooltips = []
        self.rectangles = {}

    def __getattr__(self, name):
        if name in {
            "set_next_window_pos",
            "set_next_window_size",
            "separator",
            "same_line",
            "progress_bar",
            "set_next_item_width",
        }:
            return lambda *args, **kwargs: None
        raise AttributeError(name)

    def set_next_window_pos(self, position, *args):
        self.next_position = position

    def set_next_window_size(self, size, *args):
        self.next_size = size

    def begin(self, title, **kwargs):
        self.cursor_x, self.cursor_y = 11.0, 23.0
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

    def get_cursor_pos_x(self):
        return self.cursor_x

    def get_cursor_pos_y(self):
        return self.cursor_y

    def set_cursor_pos_x(self, x):
        self.cursor_x = x

    def set_cursor_pos_y(self, y):
        self.cursor_y = y

    def selectable(self, label, selected):
        return label in self.clicked, selected

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
        self.image_positions[key] = (self.cursor_x, self.cursor_y)


@pytest.mark.parametrize("count", range(1, 12))
@pytest.mark.parametrize("selection", [{}, {"visible_count": 1}])
@pytest.mark.parametrize("bearings", [False, True])
def test_layout_keeps_camera_and_lidar_identity_focus_and_aspect(
    count, selection, bearings
):
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
        bearing_layout=bearings,
        **selection,
    )
    loop = MultiviewUILoop(renderer=Mock())
    loop.state = state
    loop.get_ui_loop_size = Mock(return_value=(1920, 1080))
    frames = [torch.full((3, 18, 32), i, dtype=torch.uint8) for i in range(count + 1)]
    loop.presented_model_frame = Mock(side_effect=lambda i: frames[i])
    ui = RecordingUI()
    loop.step_ui(ui, 0, UserInputEvents([]))
    assert ui.panels == 0
    assert ui.sliders == []
    shown = selection.get("visible_count", count)
    assert state.visible_count == shown
    assert list(ui.images) == [f"stream-{i}" for i in (*range(shown), count)]
    if not bearings or shown == 1:
        # Expected row-major grid dimensions for two through twelve sensors.
        columns, rows = (
            (2, 1),
            (2, 2),
            (2, 2),
            (3, 2),
            (3, 2),
            (3, 3),
            (3, 3),
            (3, 3),
            (4, 3),
            (4, 3),
            (4, 3),
        )[shown - 1]
        for cell, channel in enumerate((*range(shown), count)):
            row, column = divmod(cell, columns)
            assert ui.rectangles[f"Stream##{channel}"] == pytest.approx(
                (
                    column * 1600 / columns + 4,
                    row * 1080 / rows + 4,
                    1600 / columns - 8,
                    1080 / rows - 8,
                )
            )
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
    for focus in (0, count):
        state.focus_channel = focus
        ui = RecordingUI()
        loop.step_ui(ui, 1, UserInputEvents([]))
        assert list(ui.images) == [f"stream-{focus}"]
        ui.clicked.add("Return to overview")
        loop.step_ui(ui, 2, UserInputEvents([]))
        assert state.focus_channel == -1
        for channel in (0, count):
            assert (
                ui.rectangles[f"Stream##{channel}"]
                == reduced.rectangles[f"Stream##{channel}"]
            )


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
    loop.get_ui_loop_size = Mock(return_value=(1920, 1080))
    frames = [
        torch.full((3, 18, 32), i, dtype=torch.uint8) for i in range(1 + len(names))
    ]
    loop.presented_model_frame = Mock(side_effect=lambda i: frames[i])
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


@pytest.mark.parametrize("count", range(1, 12))
@pytest.mark.parametrize("lidar", [0, 1])
@pytest.mark.parametrize("mirror", [False, True])
def test_bearing_layout_bounds_identity_and_sides(count, lidar, mirror):
    rig = [
        ("front_wide", 0),
        ("cross_right", -90),
        ("rear_right", -135),
        ("rear_tele", 180),
        ("rear_left", 135),
        ("cross_left", 90),
        ("front_tele", 0),
        ("front_fisheye", 0),
        ("left_fisheye", 90),
        ("right_fisheye", -90),
        ("rear_fisheye", 180),
    ]
    cameras = tuple(
        CameraView(name, name, -yaw if mirror else yaw) for name, yaw in rig[:count]
    )
    layout = sensor_layout(cameras, lidar)
    assert set(layout) == set(range(count + lidar))
    for channel, (x, y, w, h) in layout.items():
        assert -1e-9 <= x < x + w <= 1 + 1e-9
        assert 0 <= y < y + h <= 1 + 1e-9
        if channel < count:
            yaw = cameras[channel].yaw_degrees
            if 0 < yaw < 180:
                assert x + w / 2 < 0.5
            elif -180 < yaw < 0:
                assert x + w / 2 > 0.5
            if abs(yaw) == 180:
                assert y > layout[0][1]
        for other, (ox, oy, ow, oh) in layout.items():
            if other != channel:
                assert (
                    x + w <= ox + 1e-9
                    or ox + ow <= x + 1e-9
                    or y + h <= oy + 1e-9
                    or oy + oh <= y + 1e-9
                )
    reordered = tuple(reversed(cameras))
    other = sensor_layout(reordered, lidar)
    for i in range(count):
        assert layout[i] == other[count - i - 1]


def test_title_focus_and_absent_auxiliary_preserve_stream_identity():
    state = MultiviewUIState(
        model_loop=Mock(),
        title="Sensors",
        prompt="",
        scene_options=(),
        cameras=(CameraView("front", "Front"), CameraView("rear", "Rear", 180)),
        auxiliary_views=(AuxiliaryView("lidar", "LiDAR"),),
    )
    loop = MultiviewUILoop(renderer=Mock())
    loop.state = state
    loop.get_ui_loop_size = Mock(return_value=(1280, 720))
    loop.presented_model_frame = Mock(
        side_effect=lambda i: torch.full((3, 9, 16), i, dtype=torch.uint8)
    )
    ui = RecordingUI()
    ui.clicked.add("Rear  [+]##focus-1")
    loop.step_ui(ui, 0, UserInputEvents([]))
    assert state.focus_channel == 1
    focused = RecordingUI()
    loop.step_ui(focused, 1, UserInputEvents([]))
    assert list(focused.images) == ["stream-1"]
    state.focus_channel = -1
    state.auxiliary_views = ()
    hidden = RecordingUI()
    loop.step_ui(hidden, 2, UserInputEvents([]))
    assert set(hidden.images) == {"stream-0", "stream-1"}


@pytest.mark.parametrize("yaw", [float("nan"), float("inf")])
def test_layout_rejects_invalid_bearings(yaw):
    with pytest.raises(ValueError, match="finite"):
        sensor_layout((CameraView("bad", "Bad", yaw),))


def test_configured_lidar_waits_in_place_until_its_stream_arrives():
    state = MultiviewUIState(
        model_loop=Mock(),
        title="Sensors",
        prompt="",
        scene_options=(),
        cameras=(CameraView("front", "Front"),),
        auxiliary_views=(AuxiliaryView("lidar", "LiDAR"),),
    )
    loop = MultiviewUILoop(renderer=Mock())
    loop.state = state
    loop.get_ui_loop_size = Mock(return_value=(1280, 720))
    loop.presented_model_frame = Mock(
        side_effect=lambda i: None if i else torch.zeros(3, 9, 16, dtype=torch.uint8)
    )
    waiting = RecordingUI()
    loop.step_ui(waiting, 0, UserInputEvents([]))
    assert "Stream##1" in waiting.rectangles
    assert "Waiting for sensor stream..." in waiting.labels
    loop.presented_model_frame = Mock(
        side_effect=lambda i: torch.full((3, 9, 16), i, dtype=torch.uint8)
    )
    arrived = RecordingUI()
    loop.step_ui(arrived, 1, UserInputEvents([]))
    assert arrived.rectangles["Stream##1"] == waiting.rectangles["Stream##1"]
    assert "stream-1" in arrived.images


@pytest.mark.parametrize(
    "yaws, expected",
    [
        ((0, 180), ((0, 0, 1, 0.5), (0, 0.5, 1, 0.5))),
        ((180, 0), ((0, 0.5, 1, 0.5), (0, 0, 1, 0.5))),
        ((-90, 90), ((0.5, 0, 0.5, 1), (0, 0, 0.5, 1))),
        ((90, -90), ((0, 0, 0.5, 1), (0.5, 0, 0.5, 1))),
    ],
)
@pytest.mark.parametrize("reduced", [False, True])
def test_two_camera_bearings_and_reduced_visible_count(yaws, expected, reduced):
    cameras = tuple(CameraView(str(i), str(i), yaw) for i, yaw in enumerate(yaws))
    assert sensor_layout(cameras) == dict(enumerate(expected))
    state = MultiviewUIState(
        model_loop=Mock(),
        title="Sensors",
        prompt="",
        scene_options=(),
        cameras=cameras + ((CameraView("extra", "Extra", 45),) if reduced else ()),
        visible_count=2,
    )
    loop = MultiviewUILoop(renderer=Mock())
    loop.state = state
    loop.get_ui_loop_size = Mock(return_value=(1920, 1080))
    loop.presented_model_frame = Mock(
        side_effect=lambda i: torch.full((3, 9, 16), i, dtype=torch.uint8)
    )
    ui = RecordingUI()
    loop.step_ui(ui, 0, UserInputEvents([]))
    assert list(ui.images) == ["stream-0", "stream-1"]
    for i, (x, y, w, h) in enumerate(expected):
        assert ui.rectangles[f"Stream##{i}"] == pytest.approx(
            (x * 1600 + 4, y * 1080 + 4, w * 1600 - 8, h * 1080 - 8)
        )


@pytest.mark.parametrize(
    "shape, focused, size, position",
    [
        ((3, 50, 200), False, (300, 75), (11, 75.5)),
        ((3, 50, 200), True, (300, 75), (11, 67)),
        ((3, 200, 50), False, (45, 180), (138.5, 23)),
        ((3, 200, 50), True, (40.75, 163), (140.625, 23)),
    ],
)
def test_images_center_in_available_space_with_focused_caption(
    shape, focused, size, position
):
    state = MultiviewUIState(
        model_loop=Mock(),
        title="Sensors",
        prompt="",
        scene_options=(),
        cameras=(CameraView("front", "Front", caption="Camera caption"),),
        focus_channel=0 if focused else -1,
    )
    loop = MultiviewUILoop(renderer=Mock())
    loop.state = state
    loop.get_ui_loop_size = Mock(return_value=(1280, 720))
    loop.presented_model_frame = Mock(
        return_value=torch.zeros(shape, dtype=torch.uint8)
    )
    ui = RecordingUI()
    loop.step_ui(ui, 0, UserInputEvents([]))
    assert ui.image_sizes["stream-0"] == pytest.approx(size)
    assert ui.image_positions["stream-0"] == pytest.approx(position)
    assert ("Camera caption" in ui.labels) == focused
