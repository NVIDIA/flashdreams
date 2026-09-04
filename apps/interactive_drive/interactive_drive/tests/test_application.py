# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import interactive_drive.app as app_module
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
from interactive_drive.input.keyboard import command_from_snapshot
from interactive_drive.types import ControlSnapshot

from flashdreams.runtime_v2.session_desc import BackpressureMode, PresentationMode
from flashdreams.runtime_v2.user_input_event import (
    GamepadUserInputEvent,
    KeyboardInputState,
    KeyboardUserInputEvent,
)
from flashdreams.runtime_v2.user_input_events import UserInputEvents

pytestmark = pytest.mark.ci_cpu


class _FakeUI:
    Cond_ = SimpleNamespace(once="once")

    def __init__(self) -> None:
        self.text_lines: list[str] = []
        self.images: list[str] = []
        self.progress: list[float] = []
        self.checkbox_labels: list[str] = []

    @staticmethod
    def ImVec2(x: float, y: float) -> tuple[float, float]:
        return (x, y)

    def set_next_window_pos(self, position: Any, condition: Any) -> None:
        del position, condition

    def set_next_window_size(self, size: Any, condition: Any) -> None:
        del size, condition

    def begin(self, title: str) -> None:
        del title

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
    )
    config = SimpleNamespace(
        total_blocks=2,
        app=SimpleNamespace(
            chunk=SimpleNamespace(frame_interval_us=33_333),
            vehicle=object(),
        ),
    )
    physics_world: Any = object()
    state = InteractiveDriveModelState(
        backend_factory=lambda _: backend,
        config=config,
        desc=SimpleNamespace(output_layout="tchw"),
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
        lambda frame_chunk, view_mode: torch.zeros((2, 3, 1, 1)),
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
    app = InteractiveDriveApplication(
        defaults=InteractiveDriveApplicationDefaults(
            pipeline_config=cast(Any, object()),
        ),
    )

    app.init(
        [
            "--scene",
            str(scene),
            "--postprocess-preset",
            "example-preset",
        ]
    )

    assert app._config is not None
    assert isinstance(app._config, InteractiveDriveConfig)
    assert app._config.app.postprocess.preset == "example-preset"
    session = app.create_session(app.session_desc())
    session.init()
    assert isinstance(session.ui_loop, InteractiveDriveUILoop)
    assert session.ui_loop.state.show_postprocess_toggle


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
    assert presented_channels == [1, 0]
    assert model_loop.state.drive_input.command().throttle == 0.0

    model_loop._apply_events(steering_events)

    assert model_loop.state.drive_input.command().throttle == 0.0
    assert model_loop.state.drive_input.command().steer == 0.2


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
        config=cast(Any, SimpleNamespace()),
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
            SimpleNamespace(
                rgb_host_uint8=hdmap,
                model_rgb_host_uint8=rgb,
                physx_rgb_host_uint8=physx,
            )
        ]
    )

    assert core_module._frame_chunk_tensor(chunk, "rgb")[0, 0, 0, 0] == 22
    assert core_module._frame_chunk_tensor(chunk, "hdmap")[0, 0, 0, 0] == 11
    physx_view = core_module._frame_chunk_tensor(chunk, "physx")
    assert physx_view[0, 0, 0, 0] == 33
    assert physx_view[0, 0, 0, 1] == 11


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
