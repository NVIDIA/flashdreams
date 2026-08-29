# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for Crazy Robotaxi's application boundary against FlashDreams V2."""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import patch

import numpy as np
import pytest
import torch
from crazy_robotaxi.application import (
    CrazyRobotaxiApplication,
    CrazyRobotaxiApplicationDefaults,
    _fit_bev_renderer_to_ui,
)
from crazy_robotaxi.dynamics import TaxiVehicleConfig
from crazy_robotaxi.game_selection import GameSelection
from crazy_robotaxi.physics import TaxiPhysicsWorld
from crazy_robotaxi.rules import TaxiGameSnapshot
from crazy_robotaxi.session import (
    CrazyRobotaxiModelLoop,
    CrazyRobotaxiSession,
    ModelState,
    _restart_requested,
    _taxi_driver_command,
)
from crazy_robotaxi.ui import CrazyRobotaxiImGuiUILoop
from omnidreams.apps.crazy_robotaxi.adapter import (
    OMNIDREAMS_CRAZY_ROBOTAXI_DEFAULTS,
    OMNIDREAMS_CRAZY_ROBOTAXI_FAST_PERF_DEFAULTS,
    OMNIDREAMS_CRAZY_ROBOTAXI_PERF_DEFAULTS,
)
from omnidreams.config import (
    OMNIDREAMS_FAST_PERF_PIPELINE_CONFIG,
    OMNIDREAMS_PERF_PIPELINE_CONFIG,
    OMNIDREAMS_PIPELINE_CONFIG,
)
from omnidreams_game_engine.config import BevConfig, RasterConfig
from omnidreams_game_engine.input import DriverInput
from omnidreams_game_engine.renderer_settings import RendererSettings
from omnidreams_game_engine.simulation.game_physics import GamePhysicsWorld
from omnidreams_game_engine.types import (
    CameraCalibration,
    DriverCommand,
    SceneDefinition,
)

from flashdreams.runtime_v2.native_window_client_window import (
    NativeWindowClientWindow,
)
from flashdreams.runtime_v2.session_desc import PresentationMode
from flashdreams.runtime_v2.step_result import StepResult
from flashdreams.runtime_v2.user_input_event import (
    GamepadUserInputEvent,
    KeyboardInputState,
    KeyboardUserInputEvent,
)
from flashdreams.runtime_v2.user_input_events import UserInputEvents
from flashdreams.runtime_v2.video_tensor import VideoTensorLayout

pytestmark = pytest.mark.ci_cpu

_DEMO_RACE_MAP = (
    Path(__file__).parents[1]
    / "crazy_robotaxi"
    / "maps"
    / "flashdreams_raceway.robotaxi.yaml"
)


def _application(
    *,
    defaults: CrazyRobotaxiApplicationDefaults = OMNIDREAMS_CRAZY_ROBOTAXI_DEFAULTS,
    **kwargs: Any,
) -> CrazyRobotaxiApplication:
    return CrazyRobotaxiApplication(defaults=defaults, **kwargs)


def _scene(*, width: int = 1280, height: int = 704) -> SceneDefinition:
    calibration = CameraCalibration(
        clipgt_name="front",
        logical_name="camera_front_wide_120fov",
        width=width,
        height=height,
        cx=width / 2.0,
        cy=height / 2.0,
        polynomial=np.zeros(6, dtype=np.float32),
        is_backward_polynomial=False,
        linear_cde=np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
        sensor_to_rig_flu=np.eye(4, dtype=np.float32),
    )
    return SceneDefinition(
        scene_path=Path("scene.arrow"),
        scene_id="test",
        metadata={},
        selected_camera=calibration,
        initial_rig_to_world=np.eye(4, dtype=np.float32),
        initial_timestamp_us=0,
        initial_yaw_rad=0.0,
        initial_speed_mps=0.0,
        initial_rgb=np.zeros((height, width, 3), dtype=np.uint8),
        prompt="taxi",
        line_layers=(),
        triangle_layers=(),
    )


def test_application_registers_model_and_imgui_ui_loops() -> None:
    pipeline = object()
    pipeline_requests: list[tuple[object, str]] = []
    app = _application(
        pipeline_factory=lambda config, device: (
            pipeline_requests.append((config, device)) or pipeline
        ),
        scene_factory=lambda request, raster: _scene(),
    )
    desc = app.session_desc()
    app.init(
        [
            "--device",
            "cpu",
            "--total-blocks",
            "2",
            "--profile-input-latency",
            "--show-fps",
        ]
    )

    session = app.create_session(desc)
    assert isinstance(session, CrazyRobotaxiSession)
    session.init()
    ui_loop, model_loop = session._take_loops()

    assert desc.output_layout is VideoTensorLayout.tchw
    assert desc.frames_per_second_for_ui == 60
    assert desc.frames_per_second_for_step == 30
    assert session.session_desc.presentation_mode is PresentationMode.CONTINUOUS
    assert isinstance(model_loop, CrazyRobotaxiModelLoop)
    assert isinstance(ui_loop, CrazyRobotaxiImGuiUILoop)
    assert session._presentation_manager._presentation_stream is None
    assert model_loop.state.pipeline is None
    assert pipeline_requests == []
    assert model_loop.state.scene is None
    assert model_loop.state.rollout is None
    assert not model_loop.state.game_selected
    assert model_loop.state.ui_loop is ui_loop
    assert ui_loop.state.model_loop is model_loop
    assert len(ui_loop.state.map_options) == 2
    assert ui_loop.state.map_options[0].path.name == "boulevard_district.robotaxi.yaml"
    assert ui_loop.state.profile_input_latency
    assert ui_loop.state.show_fps
    assert session._config.renderer.bev.width == 234
    assert session._config.renderer.bev.height == 234

    menu_results = model_loop.step(0, UserInputEvents([]))
    assert len(menu_results) == 1
    assert menu_results[0].frame_count == 1
    assert torch.all(menu_results[0].read_output() == -1.0)

    rollout_closed: list[bool] = []
    model_loop.state.scene = _scene()
    model_loop.state.rollout = cast(
        Any,
        SimpleNamespace(close=lambda: rollout_closed.append(True)),
    )
    model_loop.state.game_selected = True
    model_loop.state.return_to_map_menu()
    assert rollout_closed == [True]
    assert model_loop.state.scene is None
    assert not model_loop.state.game_selected
    model_loop.state.request_exit()
    assert model_loop.is_finished()


def test_complete_cli_game_selection_starts_without_menus(monkeypatch) -> None:
    monkeypatch.setattr(
        "crazy_robotaxi.session.WorldModelRollout", lambda **_: SimpleNamespace()
    )
    app = _application(
        pipeline_factory=lambda config, device: object(),
        scene_factory=lambda request, raster: _scene(),
    )
    app.init(
        [
            "--device",
            "cpu",
            "--prewarm-blocks",
            "0",
            "--game-mode",
            "race",
            "--map",
            str(_DEMO_RACE_MAP),
            "--race-course",
            "grand-prix",
        ]
    )
    assert app._config is not None
    assert app._config.cli_game_mode == "race"
    assert app._config.cli_map_path == _DEMO_RACE_MAP.resolve()
    assert app._config.cli_race_course_id == "grand-prix"

    session = app.create_session(app.session_desc())
    session.init()
    ui_loop, model_loop = session._take_loops()

    assert ui_loop.state._menu_stage == "loading"
    model_loop._run_message_batch()
    assert model_loop.state.game_selected
    assert model_loop.state.config.game_mode == "race"
    assert model_loop.state.config.race_course_id == "grand-prix"


def test_native_window_accepts_crazy_robotaxi_output_contract() -> None:
    """Keep the app's fixed output contract compatible with V2 native output."""

    class Presenter:
        should_close = False

        def __init__(self) -> None:
            self.frames: list[torch.Tensor] = []
            self.closed = False

        def set_input_callbacks(self, **callbacks: object) -> None:
            assert set(callbacks) == {
                "on_keyboard_event",
                "on_mouse_event",
                "on_gamepad_event",
                "on_gamepad_state",
            }

        def present_frame(self, frame: torch.Tensor) -> bool:
            self.frames.append(frame)
            return True

        def close(self) -> None:
            self.closed = True

    desc = _application().session_desc()
    presenter = Presenter()
    presenter_arguments: dict[str, object] = {}

    def create_presenter(**arguments: object) -> Presenter:
        presenter_arguments.update(arguments)
        return presenter

    window = NativeWindowClientWindow(
        title="Crazy Robotaxi",
        presenter_factory=cast(Any, create_presenter),
    )
    source = torch.zeros(
        (1, 3, desc.video_height, desc.video_width),
        dtype=torch.float32,
    )

    window.open(desc)
    window.write(
        StepResult(
            step_index=0,
            output=source,
            frame_count=1,
            output_layout=desc.output_layout,
        )
    )
    window.close()

    assert presenter_arguments == {
        "width": desc.video_width,
        "height": desc.video_height,
        "title": "Crazy Robotaxi",
    }
    assert len(presenter.frames) == 1
    assert presenter.frames[0].shape == (
        desc.video_height,
        desc.video_width,
        3,
    )
    assert presenter.frames[0].device == source.device
    assert presenter.frames[0].dtype is torch.uint8
    assert torch.all(presenter.frames[0] == 128)
    assert presenter.closed


def test_pressed_r_requests_a_v2_game_restart() -> None:
    pressed = KeyboardUserInputEvent(
        timestamp=np.uint64(1),
        key="R",
        state=KeyboardInputState.PRESSED,
    )
    released = KeyboardUserInputEvent(
        timestamp=np.uint64(2),
        key="r",
        state=KeyboardInputState.RELEASED,
    )

    assert _restart_requested(UserInputEvents([pressed]))
    assert not _restart_requested(UserInputEvents([released]))


def test_pressed_gamepad_start_requests_a_v2_game_restart() -> None:
    released = (False,) * 10
    pressed = (*released[:9], True)

    assert _restart_requested(
        UserInputEvents(
            [
                GamepadUserInputEvent(
                    timestamp=np.uint64(1), action="state", pressed=pressed
                )
            ]
        )
    )
    assert not _restart_requested(
        UserInputEvents(
            [
                GamepadUserInputEvent(
                    timestamp=np.uint64(2), action="state", pressed=released
                )
            ]
        )
    )


def test_pressed_r_can_discard_an_unsubmitted_score() -> None:
    class RestartRequested(Exception):
        pass

    snapshot = TaxiGameSnapshot(
        phase="seeking_pickup",
        target_xyz_m=(25.0, 0.0, 0.0),
        distance_m=25.0,
        relative_bearing_rad=0.0,
        target_radius_m=5.0,
        remaining_time_s=None,
        score=1200,
        global_remaining_time_s=0.0,
        session_state="awaiting_name",
        high_score_rank=1,
    )

    class ProbeState:
        game_selected = True
        config = SimpleNamespace(profile_input_latency=False)

        def __init__(self) -> None:
            self.driver_input = DriverInput()

        @staticmethod
        def ensure_rollout() -> object:
            return SimpleNamespace(engine=SimpleNamespace(current_game_frame=snapshot))

        @staticmethod
        def restart_game() -> None:
            raise RestartRequested

    pressed = KeyboardUserInputEvent(
        timestamp=np.uint64(1),
        key="R",
        state=KeyboardInputState.PRESSED,
    )
    loop = CrazyRobotaxiModelLoop()
    loop.state = cast(Any, ProbeState())

    with pytest.raises(RestartRequested):
        loop.step(0, UserInputEvents([pressed]))


def test_model_input_is_applied_before_rollout_work() -> None:
    class InputOrderVerified(Exception):
        pass

    class ProbeState:
        game_selected = True
        config = SimpleNamespace(profile_input_latency=False)

        def __init__(self) -> None:
            self.driver_input = DriverInput()

        def ensure_rollout(self) -> None:
            assert self.driver_input.command().throttle == 1.0
            raise InputOrderVerified

    pressed = KeyboardUserInputEvent(
        timestamp=np.uint64(1),
        key="w",
        state=KeyboardInputState.PRESSED,
    )
    loop = CrazyRobotaxiModelLoop()
    loop.state = cast(Any, ProbeState())

    with pytest.raises(InputOrderVerified):
        loop.step(0, UserInputEvents([pressed]))


def test_taxi_space_key_restores_handbrake_over_shared_input_mapping() -> None:
    driver_input = DriverInput()
    driver_input.apply(
        UserInputEvents(
            [
                KeyboardUserInputEvent(
                    timestamp=np.uint64(1),
                    key=" ",
                    state=KeyboardInputState.PRESSED,
                )
            ]
        ),
    )

    command = _taxi_driver_command(driver_input.command())

    assert command.handbrake
    assert not command.stop


def test_taxi_keyboard_restores_arcade_brake_reverse() -> None:
    driver_input = DriverInput(pressed_keys={"a", "s"})
    vehicle = TaxiVehicleConfig()

    command = _taxi_driver_command(driver_input.command())

    assert vehicle.steer_rate_rad_per_s == pytest.approx(3.5 * vehicle.max_steer_rad)
    assert vehicle.steer_return_rate_rad_per_s == pytest.approx(
        5.0 * vehicle.max_steer_rad
    )
    assert command.steer == 1.0
    assert not command.steer_is_direct
    assert command.manual_control
    assert command.throttle == 0.0
    assert command.brake == 1.0
    assert not command.reverse


def test_leaderboard_does_not_finish_the_v2_model_loop() -> None:
    snapshot = TaxiGameSnapshot(
        phase="seeking_pickup",
        target_xyz_m=(25.0, 0.0, 0.0),
        distance_m=25.0,
        relative_bearing_rad=0.0,
        target_radius_m=5.0,
        remaining_time_s=None,
        score=1200,
        global_remaining_time_s=0.0,
        session_state="leaderboard",
    )

    class UILoop:
        def __init__(self) -> None:
            self.operations = []

        def _invoke_async(self, operation) -> None:
            self.operations.append(operation)

    rollout = SimpleNamespace(
        engine=SimpleNamespace(current_game_frame=snapshot),
        close=lambda: None,
        reset=lambda: None,
    )
    ui_loop = UILoop()
    state = ModelState(
        pipeline_factory=lambda: object(),
        pipeline=object(),
        scene_factory=cast(Any, lambda request, raster: object()),
        scene=cast(Any, object()),
        config=cast(
            Any,
            SimpleNamespace(total_blocks=None, pipeline_profiling=False),
        ),
        session_desc=cast(
            Any,
            SimpleNamespace(
                frames_per_second_for_step=30,
                video_height=4,
                video_width=4,
            ),
        ),
        driver_input=DriverInput(),
        ui_loop=cast(Any, ui_loop),
        rollout=cast(Any, rollout),
        last_video=torch.zeros(1, 3, 4, 4),
        last_pose=np.eye(4, dtype=np.float32),
        prewarm_complete=True,
        game_selected=True,
    )
    loop = CrazyRobotaxiModelLoop()
    loop.state = state

    results = loop.step(0, UserInputEvents([]))

    assert len(results) == 1
    assert not state.finished
    assert not loop.is_finished()
    assert len(ui_loop.operations) == 1


@pytest.mark.parametrize(
    ("arguments", "expected"),
    [
        ([], False),
        (["--profile-pipeline"], True),
    ],
)
def test_pipeline_profiling_is_an_app_local_opt_in(
    arguments: list[str],
    expected: bool,
) -> None:
    configured = []
    app = _application(
        pipeline_factory=lambda config, device: configured.append(config) or object(),
        scene_factory=lambda request, raster: _scene(),
    )
    app.init(arguments)

    session = cast(CrazyRobotaxiSession, app.create_session(app.session_desc()))

    assert configured == []
    session._pipeline_factory()
    assert configured[0].enable_sync_and_profile is expected
    assert app._config is not None
    assert app._config.pipeline_profiling is expected
    assert OMNIDREAMS_PIPELINE_CONFIG.enable_sync_and_profile


def test_model_adapters_keep_their_packaged_pipeline_configs() -> None:
    assert (
        OMNIDREAMS_CRAZY_ROBOTAXI_DEFAULTS.pipeline_config is OMNIDREAMS_PIPELINE_CONFIG
    )
    assert (
        OMNIDREAMS_CRAZY_ROBOTAXI_PERF_DEFAULTS.pipeline_config
        is OMNIDREAMS_PERF_PIPELINE_CONFIG
    )
    assert (
        OMNIDREAMS_CRAZY_ROBOTAXI_FAST_PERF_DEFAULTS.pipeline_config
        is OMNIDREAMS_FAST_PERF_PIPELINE_CONFIG
    )


def test_fast_perf_combines_native_dit_and_native_vae_paths() -> None:
    pipeline: Any = OMNIDREAMS_FAST_PERF_PIPELINE_CONFIG
    perf_pipeline: Any = OMNIDREAMS_PERF_PIPELINE_CONFIG
    assert pipeline.name == "omnidreams-fast-perf"
    assert pipeline.diffusion_model.seed is None
    assert pipeline.decoder.use_compile is perf_pipeline.decoder.use_compile
    assert pipeline.decoder.use_cuda_graph is True
    assert pipeline.image_encoder.native_vae_acceleration == "required"
    assert pipeline.image_encoder.native_vae_backend == "fp8"
    assert pipeline.image_encoder.native_vae_fp8_auto_export is True
    assert pipeline.encoder.native_vae_acceleration == "required"
    assert pipeline.encoder.native_vae_backend == "fp8"
    assert pipeline.encoder.native_vae_fp8_auto_export is True
    assert pipeline.diffusion_model.transformer.native_dit_acceleration == "required"
    assert (
        pipeline.diffusion_model.transformer.native_dit_backend == "fp8_kvcache_cudnn"
    )
    assert pipeline.diffusion_model.transformer.native_dit_attention_backend == "cudnn"


@pytest.mark.parametrize("resolution_wh", [(1280, 704), (1168, 640)])
def test_adapter_dimensions_configure_renderer_geometry(
    resolution_wh: tuple[int, int], monkeypatch
) -> None:
    monkeypatch.setattr(
        "crazy_robotaxi.session.WorldModelRollout", lambda **_: SimpleNamespace()
    )
    configured: list[object] = []
    raster_sizes: list[tuple[int, int]] = []

    def load_test_scene(request: object, raster: RasterConfig) -> SceneDefinition:
        del request
        size = raster.resolution_wh
        raster_sizes.append(size)
        return _scene(width=size[0], height=size[1])

    app = _application(
        defaults=replace(
            OMNIDREAMS_CRAZY_ROBOTAXI_FAST_PERF_DEFAULTS,
            width=resolution_wh[0],
            height=resolution_wh[1],
        ),
        pipeline_factory=lambda config, device: configured.append(config) or object(),
        scene_factory=load_test_scene,
    )
    app.init(["--device", "cpu", "--prewarm-blocks", "0"])
    desc = replace(
        app.session_desc(),
        video_width=resolution_wh[0],
        video_height=resolution_wh[1],
    )

    session = app.create_session(desc)
    assert isinstance(session, CrazyRobotaxiSession)
    session.init()
    _, model_loop = session._take_loops()
    model_loop.state.select_game(
        GameSelection(mode="taxi", map_option=session._map_options[0])
    )

    assert configured == [app._pipeline_config]
    assert raster_sizes == [resolution_wh]
    assert session._config.renderer.raster.resolution_wh == resolution_wh
    expected_bev_size = min(resolution_wh[0] // 4, resolution_wh[1] // 3)
    assert session._config.renderer.bev.width == expected_bev_size
    assert session._config.renderer.bev.height == expected_bev_size
    assert model_loop.state.scene is not None
    assert model_loop.state.scene.initial_rgb.shape == (
        resolution_wh[1],
        resolution_wh[0],
        3,
    )


def test_fast_perf_honors_explicit_pipeline_overrides() -> None:
    app = _application(
        defaults=OMNIDREAMS_CRAZY_ROBOTAXI_FAST_PERF_DEFAULTS,
    )

    app.init(
        [
            "--seed",
            "7",
            "--no-compile",
            "--profile-pipeline",
        ]
    )

    pipeline = cast(Any, app._pipeline_config)
    transformer = pipeline.diffusion_model.transformer
    assert pipeline.diffusion_model.seed == 7
    assert transformer.compile_network is False
    assert transformer.native_dit_acceleration == "required"
    assert transformer.skip_finalize_kv_cache is True
    assert pipeline.diffusion_model.scheduler.denoising_timesteps == [1000, 100]
    assert pipeline.enable_sync_and_profile is True


def test_bev_render_fit_preserves_authored_aspect_ratio_and_smaller_sources() -> None:
    raster = RasterConfig()
    wide = RendererSettings(raster=raster, bev=BevConfig(width=800, height=400))
    small = RendererSettings(raster=raster, bev=BevConfig(width=120, height=80))

    fitted_wide = _fit_bev_renderer_to_ui(
        wide,
        video_width=1280,
        video_height=704,
    )
    fitted_small = _fit_bev_renderer_to_ui(
        small,
        video_width=1280,
        video_height=704,
    )

    assert (fitted_wide.bev.width, fitted_wide.bev.height) == (234, 117)
    assert fitted_small is small


@pytest.mark.parametrize(
    ("arguments", "expected"),
    [
        ([], False),
        (["--profile-input-latency"], True),
    ],
)
def test_input_latency_profiling_is_an_app_local_opt_in(
    arguments: list[str],
    expected: bool,
) -> None:
    app = _application()

    app.init(arguments)

    assert app._config is not None
    assert app._config.profile_input_latency is expected
    session = app.create_session(app.session_desc())
    assert (
        session.session_desc.metadata.get("trace_chunk_lifecycle") is True
    ) is expected
    trace_path = session.session_desc.metadata.get("trace_chunk_lifecycle_path")
    assert (trace_path is not None) is expected
    if trace_path is not None:
        assert Path(trace_path).name == "crazy-robotaxi-input-trace.log"


def test_input_latency_trace_accepts_an_explicit_path(tmp_path) -> None:
    trace_path = tmp_path / "robotaxi-input.log"
    app = _application()

    app.init(["--profile-input-latency", str(trace_path)])

    assert app._config is not None
    assert app._config.profile_input_latency
    assert app._config.input_trace_path == trace_path.resolve()
    session = app.create_session(app.session_desc())
    assert session.session_desc.metadata["trace_chunk_lifecycle_path"] == str(
        trace_path.resolve()
    )


@pytest.mark.parametrize(
    ("arguments", "expected"),
    [
        ([], False),
        (["--show-fps"], True),
        (["--show-fps", "--no-show-fps"], False),
    ],
)
def test_fps_counter_is_an_app_local_option(
    arguments: list[str],
    expected: bool,
) -> None:
    app = _application()

    app.init(arguments)

    assert app._config is not None
    assert app._config.show_fps is expected


@pytest.mark.parametrize("prewarm_blocks", [0, 4, 7])
def test_application_configures_prepresentation_warmup(prewarm_blocks: int) -> None:
    app = _application(
        pipeline_factory=lambda config, device: object(),
        scene_factory=lambda request, raster: _scene(),
    )
    app.init(["--prewarm-blocks", str(prewarm_blocks)])

    assert app._config is not None
    assert app._config.prewarm_blocks == prewarm_blocks


def test_application_rejects_negative_prewarm_blocks() -> None:
    app = _application()

    with pytest.raises(ValueError, match="must be non-negative"):
        app.init(["--prewarm-blocks", "-1"])


def test_model_state_prewarms_neutral_blocks_once_then_resets(monkeypatch) -> None:
    class FakeRollout:
        def __init__(self, **kwargs) -> None:
            del kwargs
            self.steps: list[tuple[int, tuple[DriverCommand, ...]]] = []
            self.reset_count = 0

        def frame_count(self, autoregressive_index: int) -> int:
            return autoregressive_index + 1

        def step(self, *, autoregressive_index: int, commands):
            self.steps.append((autoregressive_index, commands))
            return object()

        def reset(self) -> None:
            self.reset_count += 1

        def close(self) -> None:
            return

    monkeypatch.setattr("crazy_robotaxi.session.WorldModelRollout", FakeRollout)
    app = _application(
        pipeline_factory=lambda config, device: object(),
        scene_factory=lambda request, raster: _scene(),
    )
    app.init(["--device", "cpu", "--prewarm-blocks", "4"])
    session = app.create_session(app.session_desc())
    assert isinstance(session, CrazyRobotaxiSession)
    session.init()
    ui_loop, model_loop = session._take_loops()
    model_loop.state.select_game(
        GameSelection(mode="taxi", map_option=session._map_options[0])
    )

    rollout = model_loop.state.rollout
    assert rollout is not None
    ui_loop._run_message_batch()

    assert [index for index, _ in rollout.steps] == [0, 1, 2, 3]
    assert [len(commands) for _, commands in rollout.steps] == [1, 2, 3, 4]
    assert all(
        command == DriverCommand()
        for _, commands in rollout.steps
        for command in commands
    )
    assert rollout.reset_count == 1
    assert model_loop.state.blocks_generated == 0
    assert model_loop.state.prewarm_complete
    assert ui_loop.state._loading_status == "STARTING GAME"

    assert model_loop.state.ensure_rollout() is rollout
    assert len(rollout.steps) == 4
    ui_loop.state._name_input = "DRIVER 7"
    model_loop.state.restart_game()
    ui_loop._run_message_batch()
    assert rollout.reset_count == 2
    assert ui_loop.state._name_input == ""
    assert len(rollout.steps) == 4


def test_taxi_physics_uses_spatial_and_traffic_topology_refreshes_only() -> None:
    world = object.__new__(TaxiPhysicsWorld)
    world.graph = type("Graph", (), {"objects": ()})()
    world._physics_center_xy = np.zeros(2, dtype=np.float32)
    center = np.asarray([40.0, -4.0], dtype=np.float32)
    with patch.object(
        GamePhysicsWorld,
        "synchronize_window",
        return_value=True,
    ) as synchronize:
        changed = world.synchronize_window(center, timestamp_us=2_000_000)

    assert changed
    synchronize.assert_called_once_with(center, timestamp_us=None)


def test_taxi_physics_forwards_forced_controller_refresh() -> None:
    world = object.__new__(TaxiPhysicsWorld)
    world._has_external_actor_controllers = False
    center = np.asarray([0.0, 0.0], dtype=np.float32)
    with patch.object(
        GamePhysicsWorld,
        "synchronize_window",
        return_value=True,
    ) as synchronize:
        changed = world.synchronize_window(
            center,
            timestamp_us=2_000_000,
            force_controller_refresh=True,
        )

    assert changed
    synchronize.assert_called_once_with(
        center,
        2_000_000,
        force_controller_refresh=True,
    )


def test_application_rejects_geometry_the_model_does_not_produce() -> None:
    app = _application(
        pipeline_factory=lambda config, device: object(),
        scene_factory=lambda request, raster: _scene(),
    )
    app.init([])
    desc = app.session_desc()
    desc = type(desc)(
        output_layout=desc.output_layout,
        presentation_mode=desc.presentation_mode,
        frames_per_second_for_ui=desc.frames_per_second_for_ui,
        frames_per_second_for_step=desc.frames_per_second_for_step,
        video_width=640,
        video_height=desc.video_height,
    )

    with pytest.raises(ValueError, match="do not match renderer"):
        app.create_session(desc)


def test_application_rejects_mismatched_generation_rate() -> None:
    app = _application(
        pipeline_factory=lambda config, device: object(),
        scene_factory=lambda request, raster: _scene(),
    )
    app.init([])

    with pytest.raises(ValueError, match="30 frames per second"):
        app.create_session(replace(app.session_desc(), frames_per_second_for_step=60))


def test_application_forces_continuous_presentation_for_interactive_input() -> None:
    app = _application(
        pipeline_factory=lambda config, device: object(),
        scene_factory=lambda request, raster: _scene(),
    )
    app.init([])

    session = app.create_session(
        replace(
            app.session_desc(),
            presentation_mode=PresentationMode.ON_DEMAND,
        )
    )

    assert session.session_desc.presentation_mode is PresentationMode.CONTINUOUS
