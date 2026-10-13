# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections import deque
from contextlib import nullcontext
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import interactive_drive.app as app_module
import interactive_drive.backends.world_model as world_model_module
import interactive_drive.core as core_module
import numpy as np
import pytest
import torch
from interactive_drive import (
    DEFAULT_SCENE_FILENAME,
    DEFAULT_SCENE_REPO_ID,
    DriveTelemetry,
    InteractiveDriveApplication,
    InteractiveDriveApplicationDefaults,
    InteractiveDriveConfig,
    InteractiveDriveModelLoop,
    InteractiveDriveModelState,
    InteractiveDriveSession,
    InteractiveDriveUILoop,
    download_default_scene,
)
from interactive_drive.app import _hud_scale
from interactive_drive.backends.world_model import WorldModelRenderBackend
from interactive_drive.config import AppConfig, PresentationLayout, RasterConfig
from interactive_drive.input.keyboard import command_from_snapshot
from interactive_drive.scene_fixture import build_synthetic_scene_usdz
from interactive_drive.scene_loader import load_scene_bundle, reseed_scene_bundle
from interactive_drive.types import ControlSnapshot, PresentedFrame

from flashdreams.core.distributed.parallel import ParallelContext
from flashdreams.infra.acceleration.frame_prefetch import LazyCudaFrame
from flashdreams.infra.postprocess import (
    VideoPostprocessChainConfig,
    VideoPostProcessorConfig,
    VideoSpec,
)
from flashdreams.runtime_v2.session_desc import PresentationMode
from flashdreams.runtime_v2.user_input_event import (
    GamepadUserInputEvent,
    KeyboardInputState,
    KeyboardUserInputEvent,
)
from flashdreams.runtime_v2.user_input_events import UserInputEvents

pytestmark = pytest.mark.ci_cpu


@dataclass(kw_only=True)
class _FakePostProcessorConfig(VideoPostProcessorConfig):
    scale: int = 2

    def output_spec(self, input_spec: VideoSpec) -> VideoSpec:
        return replace(
            input_spec,
            width=input_spec.width * self.scale,
            height=input_spec.height * self.scale,
        )


class _FakeUI:
    Cond_ = SimpleNamespace(once="once")

    def __init__(self) -> None:
        self.text_lines: list[str] = []
        self.images: list[str] = []
        self.progress: list[float] = []
        self.checkbox_labels: list[str] = []
        self.style = SimpleNamespace(font_scale_main=1.0)
        self.window_sizes: list[tuple[float, float]] = []
        self.windows: list[str] = []

    @staticmethod
    def ImVec2(x: float, y: float) -> tuple[float, float]:
        return (x, y)

    def get_style(self) -> SimpleNamespace:
        return self.style

    def set_next_window_pos(self, position: Any, condition: Any) -> None:
        del position, condition

    def set_next_window_size(self, size: Any, condition: Any) -> None:
        del condition
        self.window_sizes.append(size)

    def begin(self, title: str) -> None:
        self.windows.append(title)

    def end(self) -> None:
        pass

    def text(self, value: str) -> None:
        self.text_lines.append(value)

    def combo(self, label: str, index: int, options: list[str]) -> tuple[bool, int]:
        del label, options
        return False, index

    def checkbox(self, label: str, value: bool) -> tuple[bool, bool]:
        self.checkbox_labels.append(label)
        return False, value

    def button(self, label: str) -> bool:
        del label
        return False

    def same_line(self) -> None:
        pass

    def progress_bar(self, fraction: float, size: Any) -> None:
        del size
        self.progress.append(fraction)

    def image(
        self,
        key: str,
        pixels: Any,
        *,
        size: tuple[float, float],
    ) -> None:
        del pixels, size
        self.images.append(key)

    def separator(self) -> None:
        pass


def test_gamepad_left_stick_only_steers() -> None:
    event = GamepadUserInputEvent(
        timestamp=np.uint64(0),
        axes=(-0.5, -1.0),
    )

    command = core_module._gamepad_command(event)

    assert command is not None
    assert command.steer == 0.5
    assert command.throttle == 0.0
    assert command.brake == 0.0


def test_s_reverses_and_space_matches_controller_brake_input() -> None:
    reverse = command_from_snapshot(ControlSnapshot(pressed={"s"}))
    brake = command_from_snapshot(ControlSnapshot(pressed={" "}))
    drive_input = core_module.DriveInputState()
    drive_input.apply(
        UserInputEvents(
            [
                KeyboardUserInputEvent(
                    timestamp=np.uint64(0),
                    key=" ",
                    state=KeyboardInputState.PRESSED,
                )
            ]
        )
    )
    runtime_brake = drive_input.command()
    left_trigger = core_module._gamepad_command(
        GamepadUserInputEvent(
            timestamp=np.uint64(0),
            axes=(0.0, 0.0),
            buttons=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0),
        )
    )

    assert reverse.throttle == 1.0
    assert reverse.brake == 0.0
    assert reverse.reverse
    assert left_trigger is not None
    assert brake.throttle == left_trigger.throttle
    assert brake.brake == left_trigger.brake
    assert brake.manual_control == left_trigger.manual_control
    assert runtime_brake.brake == left_trigger.brake
    assert not brake.stop
    assert not brake.reverse


def _loop_pressing(
    scene: Path, key: str, *, reverse: bool = True, physics: bool = True
) -> core_module.InteractiveDriveModelLoop:
    app = InteractiveDriveApplication(
        defaults=InteractiveDriveApplicationDefaults(reverse=reverse, physics=physics)
    )
    app.init(["--scene", str(scene)])
    session = app.create_session(app.session_desc())
    session.init()
    model_loop = session.model_loop
    assert isinstance(model_loop, InteractiveDriveModelLoop)
    model_loop._apply_events(
        UserInputEvents(
            [
                KeyboardUserInputEvent(
                    timestamp=np.uint64(0),
                    key=key,
                    state=KeyboardInputState.PRESSED,
                )
            ]
        )
    )
    return model_loop


def _loop_pressing_s(
    scene: Path, *, reverse: bool
) -> core_module.InteractiveDriveModelLoop:
    return _loop_pressing(scene, "s", reverse=reverse)


def test_asking_to_reverse_brakes_for_a_model_that_has_only_driven_forwards(
    tmp_path: Path,
) -> None:
    """Braking and not nothing, and the throttle has to go with the gear.

    S is the keyboard's reverse and it arrives with full throttle, so dropping
    the gear alone would drive the car forwards on the key pressed to go back.
    """
    scene = tmp_path / "local.usdz"
    scene.touch()

    command = _loop_pressing_s(scene, reverse=False)._command()

    assert not command.reverse
    assert command.throttle == 0.0
    assert command.brake == 1.0


def test_a_drive_with_no_block_limit_never_finishes(tmp_path: Path) -> None:
    """What an endless interactive run rests on, and the parser calls 0 unbounded."""
    scene = tmp_path / "local.usdz"
    scene.touch()

    def loop_after(blocks: str, generated: int) -> InteractiveDriveModelLoop:
        app = InteractiveDriveApplication()
        app.init(["--scene", str(scene), "--total-blocks", blocks])
        session = app.create_session(app.session_desc())
        session.init()
        model_loop = session.model_loop
        assert isinstance(model_loop, InteractiveDriveModelLoop)
        model_loop.state.blocks_generated = generated
        return model_loop

    assert not loop_after("0", 10_000).is_finished()
    assert loop_after("3", 3).is_finished()
    assert not loop_after("3", 2).is_finished()


def test_a_model_that_can_back_up_still_gets_the_gear(tmp_path: Path) -> None:
    scene = tmp_path / "local.usdz"
    scene.touch()

    command = _loop_pressing_s(scene, reverse=True)._command()

    assert command.reverse
    assert command.throttle == 1.0


def test_a_drive_without_colliders_keeps_the_collider_view_out_of_reach(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The overlay is drawn from the physics world, so with none there is none.

    Three ways in and they all have to agree, since presenting the stream is
    what raises and by then the driver has already asked for it.
    """
    scene = tmp_path / "local.usdz"
    scene.touch()
    app = InteractiveDriveApplication(
        defaults=InteractiveDriveApplicationDefaults(physics=False)
    )
    app.init(["--scene", str(scene)])
    session = app.create_session(app.session_desc())
    session.init()
    ui_loop = session.ui_loop
    assert isinstance(ui_loop, InteractiveDriveUILoop)
    monkeypatch.setattr(app_module, "invoke_async", lambda loop, operation: None)

    assert ui_loop.state.view_modes == ("rgb", "hdmap")
    ui_loop._toggle_view()
    assert ui_loop.state.view_mode == "hdmap"
    ui_loop._toggle_view()
    assert ui_loop.state.view_mode == "rgb"

    assert _loop_pressing(scene, "3", physics=False).state.view_mode == "rgb"
    with pytest.raises(SystemExit):
        InteractiveDriveApplication(
            defaults=InteractiveDriveApplicationDefaults(physics=False)
        ).init(["--scene", str(scene), "--view", "physx"])


def test_a_drive_that_simulates_colliders_can_still_look_at_them(
    tmp_path: Path,
) -> None:
    scene = tmp_path / "local.usdz"
    scene.touch()

    assert _loop_pressing(scene, "3", physics=True).state.view_mode == "physx"


def test_control_sprites_load_at_hud_resolution() -> None:
    wheel = app_module._load_control_sprite("steering_wheel", (138, 138))
    pedal = app_module._load_control_sprite("throttle_pressed", (64, 138))

    assert wheel.shape == (138, 138, 4)
    assert pedal.shape == (138, 64, 4)


def test_model_step_publishes_bev_channel_and_complete_elapsed_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    vehicle = SimpleNamespace(speed_mps=0.0, steer_rad=0.0)
    trajectory = SimpleNamespace(
        boundary_state_after_chunk=vehicle,
        timestamps_us=np.array([100], dtype=np.int64),
    )
    bev_frames = [
        np.full((2, 3, 3), fill_value=value, dtype=np.uint8) for value in (17, 29)
    ]
    chunk = SimpleNamespace(
        frames=[SimpleNamespace(bev_host_uint8=bev_frame) for bev_frame in bev_frames]
    )
    backend = SimpleNamespace(
        initial_chunk_frames=2,
        chunk_frames=2,
        render_first_chunk=lambda _: chunk,
        finalize=lambda: {},
    )
    config = SimpleNamespace(
        total_blocks=2,
        app=SimpleNamespace(
            chunk=SimpleNamespace(frame_interval_us=33_333),
            vehicle=object(),
            postprocess=VideoPostprocessChainConfig(),
            postprocess_device="cuda:0",
            world_model_device="cuda:0",
            presentation=PresentationLayout(1, 1, 0),
        ),
    )
    physics_world: Any = object()
    state = InteractiveDriveModelState(
        backend_factory=lambda _: backend,
        config=config,
        desc=SimpleNamespace(output_layout="tchw", video_width=1, video_height=1),
        scene_loader=lambda *args: object(),
        scene=object(),
        vehicle=vehicle,
        backend=backend,
        physics_world=physics_world,
        view_mode="physx",
    )
    elapsed: list[float] = []
    trajectory_calls: list[dict[str, Any]] = []
    clock = iter((10.0, 10.123))
    monkeypatch.setattr(core_module.time, "perf_counter", lambda: next(clock))

    def sample_trajectory(**kwargs: Any) -> Any:
        trajectory_calls.append(kwargs)
        return trajectory

    monkeypatch.setattr(core_module, "sample_chunk_trajectory", sample_trajectory)
    monkeypatch.setattr(
        core_module,
        "_frame_chunk_tensor",
        lambda frame_chunk, view_mode, **_: torch.zeros((2, 3, 1, 1)),
    )
    monkeypatch.setattr(core_module, "_telemetry_status", lambda *args: "ready")
    monkeypatch.setattr(
        InteractiveDriveModelState,
        "_publish_drive_telemetry",
        lambda self, chunk, model_loop_ms: elapsed.append(model_loop_ms),
    )
    loop = InteractiveDriveModelLoop()
    loop.state = state

    results = loop.step(0, UserInputEvents([]))

    assert len(results) == 2
    assert results[1].frame_count == 2
    output = results[1].read_output()
    assert tuple(output.shape) == (2, 3, 2, 3)
    assert output[:, 0, 0, 0].tolist() == [17, 29]
    assert elapsed == [pytest.approx(123.0)]
    assert trajectory_calls[0]["physics_world"] is physics_world
    assert trajectory_calls[0]["capture_physics_debug"] is True


def test_a_worker_ranks_step_generates_but_presents_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """It must still generate, and must not pay to present.

    Every rank has to arrive at the collectives inside a chunk or the ones that
    did will wait forever, so a worker runs the same step. What it must not do
    is tile, resize and copy out a grid nobody will see, and returning no
    result is how the runtime reads a step that presented nothing.
    """
    vehicle = SimpleNamespace(speed_mps=0.0, steer_rad=0.0)
    trajectory = SimpleNamespace(
        boundary_state_after_chunk=vehicle,
        timestamps_us=np.array([100], dtype=np.int64),
    )
    rendered: list[str] = []
    backend = SimpleNamespace(
        initial_chunk_frames=2,
        chunk_frames=2,
        render_first_chunk=lambda _: rendered.append("generated")
        or SimpleNamespace(frames=[]),
        finalize=lambda: {},
    )
    config = SimpleNamespace(
        total_blocks=2,
        app=SimpleNamespace(
            chunk=SimpleNamespace(frame_interval_us=33_333),
            vehicle=object(),
            postprocess=VideoPostprocessChainConfig(),
            postprocess_device="cuda:0",
            world_model_device="cuda:0",
            presentation=PresentationLayout(1, 1, 0),
        ),
    )
    state = InteractiveDriveModelState(
        backend_factory=lambda _: backend,
        config=config,
        desc=SimpleNamespace(output_layout="tchw", video_width=1, video_height=1),
        scene_loader=lambda *args: object(),
        scene=object(),
        vehicle=vehicle,
        backend=backend,
        parallel_context=replace(ParallelContext.single("cpu"), cp_rank=1, cp_size=2),
    )
    monkeypatch.setattr(core_module, "sample_chunk_trajectory", lambda **_: trajectory)

    def refuse(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("a rank nobody is watching presented a frame")

    monkeypatch.setattr(core_module, "_frame_chunk_tensor", refuse)
    loop = InteractiveDriveModelLoop()
    loop.state = state

    results = loop.step(0, UserInputEvents([]))

    assert results == []
    assert rendered == ["generated"]
    assert state.blocks_generated == 1


def test_world_model_preserves_frame_order_across_buffered_startup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = object.__new__(WorldModelRenderBackend)
    backend._pending_raster_frames = deque()
    backend._first_transition_frame = None
    backend._cache = None
    backend._pending_finalization_index = None
    backend._view_names = ("camera_front_wide_120fov",)

    def raster_frames(start: int, count: int) -> tuple[PresentedFrame, ...]:
        return tuple(
            PresentedFrame(
                timestamp_us=index,
                view_rgb_host_uint8=(np.zeros((1, 1, 3), dtype=np.uint8),),
                depth_host_f32=None,
            )
            for index in range(start, start + count)
        )

    assert (
        backend._merge_frames(raster_frames(0, 5), (), annotate_first_transition=True)
        == ()
    )

    first_models = [object() for _ in range(5)]
    first = backend._merge_frames(raster_frames(5, 8), [first_models])
    assert [frame.timestamp_us for frame in first] == list(range(5))
    assert [frame.model_rgb_host_uint8 for frame in first] == first_models
    assert first[-1].status_message == "Optimizing world model..."

    steady_models = [object() for _ in range(8)]
    steady = backend._merge_frames(raster_frames(13, 8), [steady_models])
    assert [frame.timestamp_us for frame in steady] == list(range(5, 13))
    assert [frame.model_rgb_host_uint8 for frame in steady] == steady_models

    tail_models = [object() for _ in range(8)]
    tail_tensor = torch.zeros((1, 1, 8, 3, 1, 1))
    backend._postprocess_stream = SimpleNamespace(finish=lambda: tail_tensor)
    monkeypatch.setattr(
        "interactive_drive.backends.world_model.lazy_rgb_frames_from_video_tensor",
        lambda _output, **_: tail_models,
    )
    tail = backend.finish()
    assert [frame.timestamp_us for frame in tail] == list(range(13, 21))
    assert [frame.model_rgb_host_uint8 for frame in tail] == tail_models
    assert not backend._pending_raster_frames


def _rig_raster_frame(timestamp_us: int, views: int) -> PresentedFrame:
    return PresentedFrame(
        timestamp_us=timestamp_us,
        view_rgb_host_uint8=tuple(
            np.full((1, 1, 3), view, dtype=np.uint8) for view in range(views)
        ),
        depth_host_f32=None,
    )


def test_scene_loader_fills_a_rig_of_one_from_a_synthetic_scene(
    tmp_path: Path,
) -> None:
    scene_path = build_synthetic_scene_usdz(tmp_path / "synthetic.usdz")
    camera_names = ("camera_front_wide_120fov",)

    bundle = load_scene_bundle(
        scene_path, camera_names, "default", None, RasterConfig()
    )

    assert [camera.logical_name for camera in bundle.cameras] == list(camera_names)
    assert len(bundle.initial_rgbs) == 1
    reseeded = reseed_scene_bundle(
        bundle, scene_path, camera_names, "default", None, RasterConfig()
    )
    assert len(reseeded.cameras) == len(reseeded.initial_rgbs) == 1


def test_scene_loader_refuses_a_scene_with_no_cameras(tmp_path: Path) -> None:
    scene_path = build_synthetic_scene_usdz(tmp_path / "synthetic.usdz")

    with pytest.raises(ValueError, match="at least one camera"):
        load_scene_bundle(scene_path, (), "default", None, RasterConfig())


def test_a_frame_presents_its_first_camera_and_keeps_the_rest() -> None:
    frame = _rig_raster_frame(0, views=4)
    assert frame.views == 4
    assert frame.rgb_host_uint8[0, 0, 0] == 0
    assert [view[0, 0, 0] for view in frame.view_rgb_host_uint8] == [0, 1, 2, 3]
    assert frame.model_rgb_host_uint8 is None


def test_conditioning_transposes_a_chunk_into_one_column_per_camera() -> None:
    frames = tuple(_rig_raster_frame(index, views=3) for index in range(5))
    views = world_model_module._condition_views(frames)
    assert len(views) == 3
    assert all(len(column) == 5 for column in views)
    assert [column[0][0, 0, 0] for column in views] == [0, 1, 2]


def test_conditioning_refuses_a_chunk_whose_rig_changes_mid_way() -> None:
    frames = (_rig_raster_frame(0, views=3), _rig_raster_frame(1, views=2))
    with pytest.raises(ValueError, match="same cameras in every frame"):
        world_model_module._condition_views(frames)


def test_merging_pairs_every_camera_with_its_own_model_output() -> None:
    backend = object.__new__(WorldModelRenderBackend)
    backend._pending_raster_frames = deque()
    backend._first_transition_frame = None

    backend._view_names = ("front", "left", "right")

    model_views = [
        [f"view{view}-frame{index}" for index in range(2)] for view in range(3)
    ]
    merged = backend._merge_frames(
        tuple(_rig_raster_frame(index, views=3) for index in range(2)),
        model_views,
    )

    assert len(merged) == 2
    assert merged[0].view_model_rgb_host_uint8 == (
        "view0-frame0",
        "view1-frame0",
        "view2-frame0",
    )
    assert merged[1].view_model_rgb_host_uint8 == (
        "view0-frame1",
        "view1-frame1",
        "view2-frame1",
    )
    assert merged[0].model_rgb_host_uint8 == "view0-frame0"
    assert [frame.views for frame in merged] == [3, 3]


def test_merging_refuses_views_that_disagree_on_length() -> None:
    backend = object.__new__(WorldModelRenderBackend)
    backend._pending_raster_frames = deque()
    backend._first_transition_frame = None

    backend._view_names = ("front", "left")

    with pytest.raises(ValueError, match="unequal frame counts"):
        backend._merge_frames(
            (_rig_raster_frame(0, views=2),),
            [["one"], ["one", "two"]],
        )


def test_merging_refuses_a_model_that_skips_a_camera() -> None:
    backend = object.__new__(WorldModelRenderBackend)
    backend._pending_raster_frames = deque()
    backend._first_transition_frame = None
    backend._view_names = ("front", "left", "right")

    with pytest.raises(ValueError, match="does not cover the scene's rig"):
        backend._merge_frames(
            (_rig_raster_frame(0, views=3),),
            [["front-frame0"], ["left-frame0"]],
        )


def test_conditioning_tensor_carries_the_rig_on_the_view_axis() -> None:
    backend = object.__new__(WorldModelRenderBackend)
    backend._pipeline = cast(Any, SimpleNamespace(device=torch.device("cpu")))

    frames = tuple(_rig_raster_frame(index, views=3) for index in range(4))
    tensor = backend._condition_tensor(world_model_module._condition_views(frames))

    assert tuple(tensor.shape) == (1, 3, 4, 3, 1, 1)
    pixels = (tensor[0, :, 0, 0, 0, 0].float() + 1.0) * 127.5
    assert [round(float(pixel)) for pixel in pixels] == [0, 1, 2]


def test_every_camera_in_the_rig_is_given_the_scene_to_condition_on(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The pipeline reads ``text`` as ``[B, V]`` and sizes its caches from it.

    One sentence for a rig of four left the text a view wide while the images
    and the conditioning carried the whole rig, so the caches came out too
    small to generate into.
    """
    monkeypatch.setattr(torch.cuda, "device", lambda _device: nullcontext())
    asked: dict[str, Any] = {}
    backend = object.__new__(WorldModelRenderBackend)
    backend._pipeline = cast(
        Any,
        SimpleNamespace(
            device=torch.device("cpu"),
            initialize_cache=lambda **kwargs: asked.update(kwargs),
        ),
    )
    backend._view_names = ("front", "left", "right", "rear")

    backend._initialize_cache(
        [np.full((2, 3, 3), view, dtype=np.uint8) for view in range(4)],
        "A drive through somewhere.",
    )

    assert asked["text"] == [["A drive through somewhere."] * 4]
    assert asked["view_names"] == ["front", "left", "right", "rear"]
    assert tuple(asked["image"].shape) == (1, 4, 1, 3, 2, 3)


def test_initial_image_tensor_carries_one_frame_per_camera() -> None:
    backend = object.__new__(WorldModelRenderBackend)
    backend._pipeline = cast(Any, SimpleNamespace(device=torch.device("cpu")))

    tensor = backend._initial_rgb_tensor(
        [np.full((2, 3, 3), view, dtype=np.uint8) for view in range(4)]
    )

    assert tuple(tensor.shape) == (1, 4, 1, 3, 2, 3)


def _rig_model_frame(views: int, *, height: int = 1, width: int = 1) -> PresentedFrame:
    frame = _rig_raster_frame(0, views=views)
    frame.view_rgb_host_uint8 = tuple(
        np.full((height, width, 3), view, dtype=np.uint8) for view in range(views)
    )
    frame.view_model_rgb_host_uint8 = tuple(
        np.full((height, width, 3), 100 + view, dtype=np.uint8) for view in range(views)
    )
    return frame


def test_a_wide_rig_is_presented_as_a_grid_in_rig_order() -> None:
    chunk = cast(Any, SimpleNamespace(frames=[_rig_model_frame(4)]))

    output = core_module._frame_chunk_tensor(
        chunk, "rgb", layout=PresentationLayout(2, 2, None)
    )

    assert tuple(output.shape) == (1, 3, 2, 2)
    assert output[0, 0].tolist() == [[100, 101], [102, 103]]


def test_a_grid_blacks_out_the_cells_the_rig_does_not_fill() -> None:
    chunk = cast(Any, SimpleNamespace(frames=[_rig_model_frame(3)]))

    output = core_module._frame_chunk_tensor(
        chunk, "rgb", layout=PresentationLayout(2, 2, None)
    )

    assert output[0, 0].tolist() == [[100, 101], [102, 0]]


def test_presenting_one_camera_emits_it_alone_at_its_own_size() -> None:
    chunk = cast(Any, SimpleNamespace(frames=[_rig_model_frame(4)]))

    output = core_module._frame_chunk_tensor(
        chunk, "rgb", layout=PresentationLayout(1, 1, 2)
    )

    assert tuple(output.shape) == (1, 3, 1, 1)
    assert output[0, 0, 0, 0] == 102


def test_a_grid_shows_the_collider_overlay_on_the_camera_it_was_drawn_from() -> None:
    frame = _rig_model_frame(2, height=2, width=2)
    physx = np.zeros((2, 2, 3), dtype=np.uint8)
    physx[0, 0] = 33
    frame.physx_rgb_host_uint8 = physx
    chunk = cast(Any, SimpleNamespace(frames=[frame]))

    output = core_module._frame_chunk_tensor(
        chunk, "physx", layout=PresentationLayout(2, 1, None)
    )

    # The first camera's cell carries the overlay; the second shows its own
    # conditioning, which is what the overlay would have been drawn over.
    assert output[0, 0, 0, 0] == 33
    assert output[0, 0, 1, 0] == 0
    assert output[0, 0, 0, 2] == 1


def test_a_grid_refuses_cameras_that_do_not_fit() -> None:
    chunk = cast(Any, SimpleNamespace(frames=[_rig_model_frame(4)]))

    with pytest.raises(ValueError, match="do not fit a 2x1 grid"):
        core_module._frame_chunk_tensor(
            chunk, "rgb", layout=PresentationLayout(2, 1, None)
        )


def test_a_rig_lays_out_as_close_to_square_as_it_goes() -> None:
    def layout(*names: str) -> tuple[int, int, int | None]:
        config = AppConfig(scene_path=Path("scene.usdz"), camera_names=names)
        presentation = config.presentation
        return presentation.columns, presentation.rows, presentation.view_index

    assert layout("front") == (1, 1, None)
    assert layout("front", "left") == (2, 1, None)
    assert layout("front", "left", "right") == (2, 2, None)
    assert layout("front", "left", "right", "rear") == (2, 2, None)


def test_presenting_one_camera_resolves_it_against_the_rig() -> None:
    config = AppConfig(
        scene_path=Path("scene.usdz"),
        camera_names=("camera_front_wide_120fov", "camera_cross_left_120fov"),
        present_camera="camera:cross:left:120fov",
    )

    assert config.presentation == PresentationLayout(1, 1, 1)


def test_a_scene_refuses_a_presented_camera_outside_its_rig() -> None:
    with pytest.raises(ValueError, match="is not in the rig"):
        AppConfig(
            scene_path=Path("scene.usdz"),
            camera_names=("camera_front_wide_120fov",),
            present_camera="camera_cross_left_120fov",
        )


def test_a_camera_rig_keeps_its_order_and_refuses_repeats() -> None:
    assert core_module._camera_rig("front, left ,right") == ("front", "left", "right")
    with pytest.raises(ValueError, match="same camera twice"):
        core_module._camera_rig("camera_front_wide_120fov,camera:front:wide:120fov")


def test_drive_telemetry_publishes_frame_chunk_size(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    published: list[DriveTelemetry] = []
    ui_state = SimpleNamespace(set_drive_telemetry=published.append)
    monkeypatch.setattr(
        core_module,
        "invoke_async",
        lambda _loop, callback: callback(ui_state),
    )
    state = InteractiveDriveModelState(
        backend_factory=lambda _: None,
        config=SimpleNamespace(
            app=SimpleNamespace(scene_path=tmp_path / "scene.usdz", variant="default")
        ),
        desc=SimpleNamespace(),
        scene_loader=lambda *args: object(),
        vehicle=SimpleNamespace(speed_mps=0.0, steer_rad=0.0),
        ui_loop=object(),
    )
    chunk = SimpleNamespace(
        frames=[
            SimpleNamespace(bev_host_uint8=None),
            SimpleNamespace(bev_host_uint8=None),
            SimpleNamespace(bev_host_uint8=None),
        ]
    )

    state._publish_drive_telemetry(chunk, model_loop_ms=12.5)

    assert len(published) == 1
    assert published[0].frames_in_chunk == 3


def test_interactive_drive_uses_regular_application_contract() -> None:
    app = InteractiveDriveApplication()
    assert app.session_desc().video_width == 1280
    assert app.session_desc().video_height == 704

    closed: list[bool] = []
    backend = cast(Any, SimpleNamespace(close=lambda: closed.append(True)))
    loop = InteractiveDriveModelLoop()
    loop.state = cast(
        Any,
        SimpleNamespace(
            backend=backend,
            owns_backend=False,
            physics_world=None,
            scene=object(),
            control_group=None,
        ),
    )
    loop.close()
    assert closed == []
    app._backend = backend
    app.close()
    assert closed == [True]


def test_a_worker_rank_presents_nothing_and_builds_no_hud(tmp_path: Path) -> None:
    """One drive has one window however many ranks are generating it.

    A rank nobody is watching that built a HUD anyway would spend a chunk of
    every step tiling and resizing a picture it then drops, and would open an
    ImGui window on a machine with no display. The mesh is the only way to know
    which rank that is, so it reaches the session and the model state.
    """
    scene = tmp_path / "local.usdz"
    scene.touch()
    worker = replace(ParallelContext.single("cpu"), cp_rank=1, cp_size=2)
    app = InteractiveDriveApplication(
        defaults=InteractiveDriveApplicationDefaults(parallel_context=worker)
    )

    app.init(["--scene", str(scene)])
    session = app.create_session(app.session_desc())
    session.init()

    assert not worker.is_main
    assert app._config is not None
    assert app._config.no_ui is True
    assert session.parallel_context is worker
    assert session._registered_ui_loop is None
    assert session.model_loop.state.parallel_context is worker


def test_rank_zero_keeps_its_window(tmp_path: Path) -> None:
    """The mesh must not turn the drive itself headless.

    Forcing `no_ui` off the rank rather than off the flag is the whole point,
    and getting the comparison the wrong way round would present nothing at
    all under a launcher while every test on one card still passed.
    """
    scene = tmp_path / "local.usdz"
    scene.touch()
    main = replace(ParallelContext.single("cpu"), cp_rank=0, cp_size=2)
    app = InteractiveDriveApplication(
        defaults=InteractiveDriveApplicationDefaults(parallel_context=main)
    )

    app.init(["--scene", str(scene)])

    assert main.is_main
    assert app._config is not None
    assert app._config.no_ui is False


def test_a_drive_on_one_card_declares_no_mesh(tmp_path: Path) -> None:
    """Which is every integration that has not asked for one."""
    scene = tmp_path / "local.usdz"
    scene.touch()
    app = InteractiveDriveApplication()

    app.init(["--scene", str(scene)])
    session = app.create_session(app.session_desc())

    assert session.parallel_context is None


def test_interactive_drive_no_ui_skips_ui_and_bev(tmp_path: Path) -> None:
    scene = tmp_path / "local.usdz"
    scene.touch()
    app = InteractiveDriveApplication(
        defaults=InteractiveDriveApplicationDefaults(width=1168, height=640)
    )

    desc = app.session_desc()
    assert (desc.video_width, desc.video_height) == (1168, 640)

    app.init(["--scene", str(scene), "--no-ui"])

    assert app._config is not None
    assert app._config.no_ui is True
    assert app._config.app.raster.resolution_wh == (1168, 640)
    assert app._config.app.bev.enabled is False
    assert app._config.app.bev.show_ego_car is False

    session = app.create_session(desc)
    session.init()

    assert isinstance(session.model_loop, InteractiveDriveModelLoop)
    assert session._registered_ui_loop is None


def test_an_integration_can_supply_its_own_backend(tmp_path: Path) -> None:
    scene = tmp_path / "local.usdz"
    scene.touch()
    asked: list[AppConfig] = []
    backend = cast(Any, SimpleNamespace(close=lambda: None))

    def factory(config: AppConfig) -> Any:
        asked.append(config)
        return backend

    app = InteractiveDriveApplication(
        defaults=InteractiveDriveApplicationDefaults(backend_factory=factory)
    )
    app.init(["--scene", str(scene)])

    # Prepared during init, so a model loads while a scene is still being picked.
    assert len(asked) == 1
    assert asked[0].scene_path == scene
    session = app.create_session(app.session_desc())
    assert session._owns_backend is False


def test_an_integration_cannot_name_two_models() -> None:
    with pytest.raises(ValueError, match="not both"):
        InteractiveDriveApplicationDefaults(
            pipeline_config=cast(Any, object()),
            backend_factory=lambda _: cast(Any, object()),
        )


def test_the_session_reports_the_grid_size_for_a_wide_rig(tmp_path: Path) -> None:
    scene = tmp_path / "local.usdz"
    scene.touch()
    app = InteractiveDriveApplication()

    app.init(["--scene", str(scene), "--camera", "front,left,right,rear"])

    assert app._config is not None
    assert app._config.app.camera_names == ("front", "left", "right", "rear")
    desc = app.session_desc()
    assert (desc.video_width, desc.video_height) == (1280 * 2, 704 * 2)


def test_the_session_reports_one_camera_when_asked_to_present_one(
    tmp_path: Path,
) -> None:
    scene = tmp_path / "local.usdz"
    scene.touch()
    app = InteractiveDriveApplication()

    app.init(
        [
            "--scene",
            str(scene),
            "--camera",
            "front,left,right,rear",
            "--present-camera",
            "right",
        ]
    )

    desc = app.session_desc()
    assert (desc.video_width, desc.video_height) == (1280, 704)


def test_interactive_drive_resolves_default_scene_when_omitted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scene = tmp_path / "default.usdz"
    scene.touch()
    calls: list[None] = []

    def resolve_default_scene() -> Path:
        calls.append(None)
        return scene

    monkeypatch.setattr(
        core_module,
        "download_default_scene",
        resolve_default_scene,
    )
    app = InteractiveDriveApplication()
    app.init(["--total-blocks", "0"])

    assert calls == [None]
    assert app._config is not None
    assert app._config.app.scene_path == scene


def test_interactive_drive_exposes_game_mode(tmp_path: Path) -> None:
    scene = tmp_path / "local.usdz"
    scene.touch()
    app = InteractiveDriveApplication()

    app.init(["--scene", str(scene), "--game-mode"])

    assert app._config is not None
    assert app._config.app.game_mode is True
    assert app._config.app.vehicle.speed_limit_enabled is True
    assert app._config.app.vehicle.actor_collision_enabled is True
    assert app._config.app.vehicle.static_collision_enabled is True


def test_world_model_accepts_postprocess_preset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scene = tmp_path / "default.usdz"
    scene.touch()
    monkeypatch.setattr(
        core_module,
        "discover_postprocess_presets",
        lambda: {"example-preset": object()},
    )
    monkeypatch.setattr(
        "flashdreams.plugins.registry.resolve_postprocess_preset",
        lambda _: _FakePostProcessorConfig(),
    )
    app = InteractiveDriveApplication(
        defaults=InteractiveDriveApplicationDefaults(),
    )
    initial_desc = app.session_desc()

    app.init(
        [
            "--scene",
            str(scene),
            "--postprocess-preset",
            "example-preset",
            "--postprocess-device",
            "cuda:1",
        ],
    )

    assert app._config is not None
    assert isinstance(app._config, InteractiveDriveConfig)
    assert app._config.app.postprocess_preset == "example-preset"
    assert app._config.app.postprocess_device == "cuda:1"
    (processor,) = app._config.app.postprocess.processors
    assert isinstance(processor, _FakePostProcessorConfig)
    assert processor.device == "cuda:1"
    assert (app.session_desc().video_width, app.session_desc().video_height) == (
        2560,
        1408,
    )
    session = app.create_session(initial_desc)
    assert (session.session_desc.video_width, session.session_desc.video_height) == (
        2560,
        1408,
    )
    session.init()
    assert isinstance(session.ui_loop, InteractiveDriveUILoop)
    assert session.ui_loop.state.show_postprocess_toggle
    assert session.ui_loop.renderer._cuda_device == torch.device("cuda:1")


def test_runtime_controls_display_launch_configuration() -> None:
    state = app_module.InteractiveDriveUIState(
        model_loop=cast(Any, object()),
        title="Interactive Drive",
        prompt="Drive",
        scene_options=(),
        world_model_device="cuda:1",
        raster_device="cuda:1",
        postprocess_preset="swiftvr-2x",
        postprocess_device="cuda:0",
    )
    loop = InteractiveDriveUILoop(width=1280, height=704)
    loop.state = state
    ui = _FakeUI()

    loop._draw_runtime_config(ui)

    assert ui.text_lines == [
        "World model GPU  cuda:1",
        "Ludus raster GPU cuda:1",
        "Postprocessor   swiftvr-2x",
        "Postprocess GPU cuda:0",
    ]


def test_default_scene_uses_original_hugging_face_location(tmp_path: Path) -> None:
    scene = tmp_path / "default.usdz"
    scene.touch()
    calls: list[dict[str, str]] = []

    def fake_download(**kwargs: str) -> str:
        calls.append(kwargs)
        return str(scene)

    assert download_default_scene(fake_download) == scene
    assert calls == [
        {
            "repo_id": DEFAULT_SCENE_REPO_ID,
            "repo_type": "dataset",
            "filename": DEFAULT_SCENE_FILENAME,
        }
    ]


def test_standalone_application_downloads_default_scene(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scene = tmp_path / "default.usdz"
    scene.touch()
    calls: list[None] = []

    def fake_default_scene() -> Path:
        calls.append(None)
        return scene

    monkeypatch.setattr(
        core_module,
        "download_default_scene",
        fake_default_scene,
    )
    app = InteractiveDriveApplication()

    app.init(["--total-blocks", "0"])

    assert calls == [None]
    assert app._config is not None
    assert app._config.app.scene_path == scene


def test_interactive_drive_prefers_explicit_scene(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scene = tmp_path / "local.usdz"
    scene.touch()

    def unexpected_default_scene() -> Path:
        raise AssertionError("default scene should not be resolved")

    monkeypatch.setattr(
        core_module,
        "download_default_scene",
        unexpected_default_scene,
    )
    app = InteractiveDriveApplication()
    app.init(["--scene", str(scene)])

    assert app._config is not None
    assert app._config.app.scene_path == scene


def test_interactive_drive_owns_a_separate_session_and_ui_loop(
    tmp_path: Path,
) -> None:
    scene = tmp_path / "local.usdz"
    scene.touch()
    app = InteractiveDriveApplication()
    app.init(["--scene", str(scene)])

    session = app.create_session(
        replace(
            app.session_desc(),
            presentation_mode=PresentationMode.ON_DEMAND,
        )
    )
    assert isinstance(session, InteractiveDriveSession)
    assert session.session_desc.presentation_mode is PresentationMode.ON_DEMAND
    assert app._config is not None
    assert app._config.app.bev.enabled is True
    assert app._config.app.bev.show_ego_car is True

    session.init()
    assert isinstance(session.ui_loop, InteractiveDriveUILoop)


def test_interactive_drive_hud_draws_imgui_controls_and_images(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scene = tmp_path / "local.usdz"
    scene.touch()
    app = InteractiveDriveApplication()
    app.init(["--scene", str(scene)])
    session = app.create_session(app.session_desc())
    session.init()
    loop = session.ui_loop
    assert isinstance(loop, InteractiveDriveUILoop)
    model_loop = session.model_loop
    assert isinstance(model_loop, InteractiveDriveModelLoop)
    assert loop.state.drive_input is not model_loop.state.drive_input
    pixel = np.zeros((4, 4, 4), dtype=np.uint8)
    loop.state.sprites = {
        "steering_wheel": pixel,
        "throttle_pressed": pixel,
        "throttle_unpressed": pixel,
        "brake_pressed": pixel,
        "brake_unpressed": pixel,
    }
    loop.state.set_drive_telemetry(
        DriveTelemetry(
            speed_mps=12.0,
            reverse=False,
            blocks_generated=7,
            frames_in_chunk=13,
            scene_path=scene,
            variant="default",
            postprocess_enabled=True,
            model_loop_ms=123.45,
        )
    )
    presented_channels: list[int] = []

    def presented_frame(channel_index: int) -> torch.Tensor:
        presented_channels.append(channel_index)
        return torch.zeros((3, 8, 8), dtype=torch.uint8)

    monkeypatch.setattr(
        loop._presentation_manager,
        "presented_frame",
        presented_frame,
    )
    steering_events = UserInputEvents(
        [GamepadUserInputEvent(timestamp=np.uint64(0), axes=(-0.2, -1.0))]
    )

    ui = _FakeUI()
    loop.step_ui(ui, 0, steering_events)

    assert "Block 7" in ui.text_lines
    assert "Input  wheel/gamepad" in ui.text_lines
    assert "Speed   26.8 mph" in ui.text_lines
    assert "Gear   D" in ui.text_lines
    assert "Steer  +0.20" in ui.text_lines
    assert "frames_in_chunk: 13" in ui.text_lines
    assert "model_loop_ms: 123.5" in ui.text_lines
    assert ui.images == [
        "steering-wheel",
        "brake-pedal",
        "throttle-pedal",
        "bev-minimap",
    ]
    assert ui.progress == [0.0, 0.0]
    assert ui.checkbox_labels == []
    # Positive steering means left, so the HUD wheel rotates counterclockwise.
    assert loop.state.wheel_cache_angle == 36
    assert "Map" in ui.windows
    assert presented_channels == [1, 0]
    assert model_loop.state.drive_input.command().throttle == 0.0

    model_loop._apply_events(steering_events)

    assert model_loop.state.drive_input.command().throttle == 0.0
    assert model_loop.state.drive_input.command().steer == 0.2


def test_the_map_panel_is_left_out_for_a_backend_that_draws_no_overhead_view(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The panel reads the second presented frame, which is only a BEV for a rig
    this app's own rasterizer built. For anything else it is another camera of
    the drive, so an empty panel is the better of two wrong pictures and leaving
    it out is better still.
    """
    scene = tmp_path / "local.usdz"
    scene.touch()
    app = InteractiveDriveApplication(
        defaults=InteractiveDriveApplicationDefaults(bev=False)
    )
    app.init(["--scene", str(scene)])
    session = app.create_session(app.session_desc())
    session.init()
    loop = session.ui_loop
    assert isinstance(loop, InteractiveDriveUILoop)
    assert not loop.state.show_bev
    pixel = np.zeros((4, 4, 4), dtype=np.uint8)
    loop.state.sprites = {
        "steering_wheel": pixel,
        "throttle_pressed": pixel,
        "throttle_unpressed": pixel,
        "brake_pressed": pixel,
        "brake_unpressed": pixel,
    }
    presented_channels: list[int] = []

    def presented_frame(channel_index: int) -> torch.Tensor:
        presented_channels.append(channel_index)
        return torch.zeros((3, 8, 8), dtype=torch.uint8)

    monkeypatch.setattr(
        loop._presentation_manager,
        "presented_frame",
        presented_frame,
    )
    ui = _FakeUI()

    loop.step_ui(ui, 0, UserInputEvents([]))

    assert "Map" not in ui.windows
    assert "bev-minimap" not in ui.images
    assert "Waiting for BEV output." not in ui.text_lines
    assert presented_channels == [0]


def test_interactive_drive_view_button_cycles_all_three_views(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scene = tmp_path / "local.usdz"
    scene.touch()
    app = InteractiveDriveApplication()
    app.init(["--scene", str(scene)])
    session = app.create_session(app.session_desc())
    session.init()
    loop = session.ui_loop
    assert isinstance(loop, InteractiveDriveUILoop)
    selected: list[str] = []
    model_state = SimpleNamespace(set_view_mode=selected.append)
    monkeypatch.setattr(
        app_module,
        "invoke_async",
        lambda _loop, callback: callback(model_state),
    )

    for expected in ("hdmap", "physx", "rgb"):
        loop._toggle_view()
        assert loop.state.view_mode == expected

    assert selected == ["hdmap", "physx", "rgb"]


def test_number_keys_select_rgb_hdmap_and_physx_views() -> None:
    state = InteractiveDriveModelState(
        backend_factory=cast(Any, lambda _: None),
        config=cast(Any, SimpleNamespace(app=SimpleNamespace(physics=True))),
        desc=cast(Any, SimpleNamespace()),
        scene_loader=cast(Any, lambda *args: object()),
    )
    loop = InteractiveDriveModelLoop()
    loop.state = state

    for key, expected in (("1", "rgb"), ("2", "hdmap"), ("3", "physx")):
        loop._apply_events(
            UserInputEvents(
                [
                    KeyboardUserInputEvent(
                        timestamp=np.uint64(0),
                        key=key,
                        state=KeyboardInputState.PRESSED,
                    )
                ]
            )
        )
        assert state.view_mode == expected
        assert key not in state.drive_input.pressed_keys


def test_frame_view_selects_rgb_hdmap_and_physx_streams() -> None:
    hdmap = np.full((2, 3, 3), 11, dtype=np.uint8)
    rgb = np.full((2, 3, 3), 22, dtype=np.uint8)
    physx = np.zeros((2, 3, 3), dtype=np.uint8)
    physx[0, 0] = 33
    chunk: Any = SimpleNamespace(
        frames=[
            PresentedFrame(
                timestamp_us=0,
                view_rgb_host_uint8=(hdmap,),
                depth_host_f32=None,
                view_model_rgb_host_uint8=(rgb,),
                physx_rgb_host_uint8=physx,
            )
        ]
    )

    assert core_module._frame_chunk_tensor(chunk, "rgb")[0, 0, 0, 0] == 22
    assert core_module._frame_chunk_tensor(chunk, "hdmap")[0, 0, 0, 0] == 11
    physx_view = core_module._frame_chunk_tensor(chunk, "physx")
    assert physx_view[0, 0, 0, 0] == 33
    assert physx_view[0, 0, 0, 1] == 11
    assert core_module._frame_chunk_tensor(chunk, "rgb", output_size=(4, 6)).shape == (
        1,
        3,
        4,
        6,
    )


def test_frame_view_delegates_cuda_readiness_to_lazy_frame(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, object]] = []
    event = object()

    class _CudaTensor:
        is_cuda = True
        device = torch.device("cuda:1")
        ndim = 3
        shape = (2, 3, 3)

        def __getitem__(self, _index: object) -> "_CudaTensor":
            return self

        def permute(self, *_dims: int) -> torch.Tensor:
            return torch.full((3, 2, 3), 23, dtype=torch.uint8)

        def record_stream(self, stream: object) -> None:
            calls.append(("record", stream))

    class _CudaStream:
        device = torch.device("cuda:1")

        def wait_event(self, source_event: object) -> None:
            calls.append(("wait", source_event))

    tensor = _CudaTensor()
    stream = _CudaStream()
    frame = LazyCudaFrame(
        [tensor],
        0,
        source_event=event,
    )
    real_is_tensor = torch.is_tensor
    monkeypatch.setattr(
        torch,
        "is_tensor",
        lambda value: value is tensor or real_is_tensor(value),
    )
    monkeypatch.setattr(torch.cuda, "current_stream", lambda _device: stream)

    chunk = cast(
        Any,
        SimpleNamespace(
            frames=[
                PresentedFrame(
                    timestamp_us=0,
                    view_rgb_host_uint8=(np.zeros((2, 3, 3), dtype=np.uint8),),
                    depth_host_f32=None,
                    view_model_rgb_host_uint8=(frame,),
                )
            ]
        ),
    )

    output = core_module._frame_chunk_tensor(chunk, "rgb")

    assert tuple(output.shape) == (1, 3, 2, 3)
    assert output[0, 0, 0, 0] == 23
    assert calls == [("wait", event), ("record", stream)]


def test_interactive_drive_discovers_scenes_and_weather_variants(
    tmp_path: Path,
) -> None:
    first_uuid = "0d404ff7-2b66-498c-b047-1ed8cded60d4"
    second_uuid = "11111111-2222-3333-4444-555555555555"
    base = tmp_path / f"clipgt-{first_uuid}.usdz"
    rain = tmp_path / f"clipgt-{first_uuid}-rain.usdz"
    other = tmp_path / f"clipgt-{second_uuid}.usdz"
    for scene in (base, rain, other):
        scene.touch()
    app = InteractiveDriveApplication()

    app.init(["--scene", str(rain)])

    assert len(app._interactive_scene_options) == 2
    selected = next(
        option for option in app._interactive_scene_options if option.path == base
    )
    assert selected.variants == ("default", "rain")
    assert app._config is not None
    assert app._config.app.scene_path == base
    assert app._config.app.variant == "rain"


def test_the_hud_is_left_alone_at_the_size_it_was_drawn_for() -> None:
    assert _hud_scale(1280, 704) == 1.0
    # Never scaled up: the panel sizes are a layout, not a ratio.
    assert _hud_scale(1920, 1080) == 1.0


def test_the_hud_shrinks_for_a_model_that_generates_a_smaller_frame() -> None:
    """Otherwise a 832x480 model gets a 354px panel over 43% of its picture."""
    scale = _hud_scale(832, 480)

    assert 0.6 <= scale < 0.7
    assert 354 * scale < 832 * 0.3


def test_the_hud_stops_shrinking_before_its_text_is_unreadable() -> None:
    assert _hud_scale(320, 240) == 0.6


def test_a_grid_of_one_camera_is_that_camera() -> None:
    """Nothing to tile, and the drive most integrations run takes this path."""
    cell = torch.arange(24, dtype=torch.uint8).reshape(2, 3, 4)

    assert core_module._mosaic([cell], columns=1, rows=1) is cell


def test_closing_a_session_lets_go_of_the_group_its_buttons_agreed_over(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A replaced session would otherwise leave its gloo group behind."""
    scene = tmp_path / "local.usdz"
    scene.touch()
    app = InteractiveDriveApplication()
    app.init(["--scene", str(scene)])
    session = app.create_session(app.session_desc())
    session.init()
    model_loop = session.model_loop
    assert isinstance(model_loop, InteractiveDriveModelLoop)
    group: Any = object()
    model_loop.state.control_group = group
    released: list[Any] = []
    monkeypatch.setattr(core_module.dist, "destroy_process_group", released.append)

    model_loop.close()

    assert released == [group]
    assert model_loop.state.control_group is None


def test_an_ordinary_restart_asks_for_no_prompt_of_its_own(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Restarting and rewording are different requests.

    A restart carrying any prompt replaces whatever the scene conditions on
    itself, which is a prompt file in the archive for one backend and the
    sample's per-camera captions for another. The HUD used to send a stand-in
    sentence, so every restart quietly reworded the drive.
    """
    scene = tmp_path / "local.usdz"
    scene.touch()

    def restart_asks_for(*args: str) -> str | None:
        app = InteractiveDriveApplication()
        app.init(["--scene", str(scene), *args])
        session = app.create_session(app.session_desc())
        session.init()
        ui_loop = session.ui_loop
        model_loop = session.model_loop
        assert isinstance(ui_loop, InteractiveDriveUILoop)
        assert isinstance(model_loop, InteractiveDriveModelLoop)
        monkeypatch.setattr(
            app_module,
            "invoke_async",
            lambda _loop, callback: callback(model_loop.state),
        )

        ui_loop._restart()

        queued = model_loop.state.queued
        assert queued is not None, "the restart never reached the model loop"
        return queued.prompt

    assert restart_asks_for() is None
    # An override is the one thing a restart should carry, and it carries it
    # every time rather than only on the run that supplied it.
    assert restart_asks_for("--prompt", "Drive through snow.") == "Drive through snow."


def test_a_canvas_the_hud_fits_in_is_a_whole_multiple_of_the_drive() -> None:
    """Asked for rather than imposed, and only where the drive is small.

    The drive is resized into the canvas, so a factor that is not whole would
    stretch the picture, and 832x480 is not the HUD's aspect ratio.
    """
    assert core_module.hud_canvas(832, 480) == (1664, 960)
    assert _hud_scale(*core_module.hud_canvas(832, 480)) == 1.0
    # Already past it, so nothing to buy: four 832x480 views tile to this.
    assert core_module.hud_canvas(1664, 960) == (1664, 960)
    assert core_module.hud_canvas(1280, 704) == (1280, 704)


def test_a_drive_asking_for_a_full_size_hud_gets_a_session_to_draw_it_in(
    tmp_path: Path,
) -> None:
    scene = tmp_path / "local.usdz"
    scene.touch()

    def desc_of(full_size_hud: bool, *args: str) -> tuple[int, int]:
        app = InteractiveDriveApplication(
            defaults=InteractiveDriveApplicationDefaults(
                width=832, height=480, full_size_hud=full_size_hud
            )
        )
        app.init(["--scene", str(scene), *args])
        desc = app.session_desc()
        return desc.video_width, desc.video_height

    assert desc_of(True) == (1664, 960)
    # Off by default, so a variant chosen for speed keeps the pixels it chose.
    assert desc_of(False) == (832, 480)
    # Nothing draws a HUD here, so the canvas would be four times the output
    # pixels for a headless run, at a resolution it did not ask for.
    assert desc_of(True, "--no-ui") == (832, 480)


def test_the_hud_font_knob_is_one_imgui_actually_has() -> None:
    """The fake cannot catch a rename, and imgui 1.92 dropped the old one.

    ``io.font_global_scale`` went when per-font scaling arrived, and nothing
    raised until the HUD drew its first frame on a real context.
    """
    from imgui_bundle import imgui

    assert hasattr(imgui.Style, "font_scale_main")
