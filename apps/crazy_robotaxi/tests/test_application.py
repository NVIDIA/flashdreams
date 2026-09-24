# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for Crazy Robotaxi's application boundary against FlashDreams V2."""

import queue
import threading
from dataclasses import dataclass, field, replace
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
    _configure_live_edit_pipeline,
    _fit_bev_renderer_to_ui,
)
from crazy_robotaxi.controls import BoundActionState, ControlsConfig
from crazy_robotaxi.dynamics import TaxiVehicleConfig
from crazy_robotaxi.game_selection import GameSelection
from crazy_robotaxi.headless_ui import CrazyRobotaxiHeadlessUILoop
from crazy_robotaxi.live_edit.config import (
    LiveEditCoinsConfig,
    LiveEditConfig,
    LiveEditObstacleConfig,
    LiveEditStyleConfig,
    LiveEditWeatherConfig,
)
from crazy_robotaxi.physics import TaxiPhysicsWorld
from crazy_robotaxi.rules import TaxiGameSnapshot
from crazy_robotaxi.session import (
    CrazyRobotaxiModelLoop,
    CrazyRobotaxiSession,
    ModelState,
    _taxi_driver_command,
)
from crazy_robotaxi.ui import CrazyRobotaxiImGuiUILoop, TaxiHudState
from omnidreams_game_engine.config import BevConfig, RasterConfig
from omnidreams_game_engine.input import DriverInput
from omnidreams_game_engine.renderer_settings import RendererSettings
from omnidreams_game_engine.simulation.game_physics import GamePhysicsWorld
from omnidreams_game_engine.types import (
    CameraCalibration,
    DriverCommand,
    SceneDefinition,
)
from torch import Tensor

from flashdreams.api_v2.loop import ModelInferenceState
from flashdreams.infra.diffusion.model import DiffusionModelConfig
from flashdreams.infra.diffusion.scheduler.base import SchedulerConfig
from flashdreams.infra.diffusion.transformer.base import TransformerConfig
from flashdreams.infra.encoder.base import EncoderConfig
from flashdreams.infra.pipeline import StreamInferencePipelineConfig
from flashdreams.runtime_v2.session_desc import PresentationMode
from flashdreams.runtime_v2.user_input_event import (
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


@dataclass(kw_only=True)
class _StubTransformerConfig(TransformerConfig):
    """Adds the fields CrazyRobotaxiApplication logs unconditionally.

    Placeholder values only; no test in this file inspects them (tests that
    care about real acceleration/backend settings live under
    integrations_v2/omnidreams/tests/, since apps/ must not import
    integrations_v2/).
    """

    native_dit_acceleration: str | None = None
    native_dit_backend: str | None = None
    native_dit_attention_backend: str | None = None
    skip_finalize_kv_cache: bool = False
    compile_network: bool = False


@dataclass(kw_only=True)
class _StubEncoderConfig(EncoderConfig):
    """Adds the fields CrazyRobotaxiApplication logs unconditionally."""

    native_vae_acceleration: str | None = None
    native_vae_backend: str | None = None


@dataclass(kw_only=True)
class _StubSchedulerConfig(SchedulerConfig):
    """Adds the field CrazyRobotaxiApplication logs unconditionally."""

    denoising_timesteps: list[int] = field(default_factory=list)


_STUB_PIPELINE_CONFIG = StreamInferencePipelineConfig(
    name="crazy-robotaxi-test-stub",
    diffusion_model=DiffusionModelConfig(
        transformer=_StubTransformerConfig(),
        scheduler=_StubSchedulerConfig(),
    ),
    encoder=_StubEncoderConfig(),
)
"""A pipeline config with no model behind it, built from base flashdreams
config classes only. CrazyRobotaxiApplication reads ``.name`` and logs
several transformer/encoder fields unconditionally, so app-level tests
need a real, structured pipeline config, not a real *model*."""

_STUB_DEFAULTS = CrazyRobotaxiApplicationDefaults(pipeline_config=_STUB_PIPELINE_CONFIG)


def _application(
    *,
    defaults: CrazyRobotaxiApplicationDefaults = _STUB_DEFAULTS,
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
    assert all(
        option.preview_image_path is not None for option in ui_loop.state.map_options
    )
    raceway = next(
        option
        for option in ui_loop.state.map_options
        if option.path.name == "flashdreams_raceway.robotaxi.yaml"
    )
    assert raceway.race_courses[0].spawn_id == "race_start"
    assert raceway.race_courses[0].preview_image_path is not None
    assert ui_loop.state.profile_input_latency
    assert ui_loop.state.show_fps
    assert ui_loop.state.gamepad_button_style == session._config.gamepad_button_style
    assert (
        ui_loop.state.show_live_edit_buttons is session._config.show_live_edit_buttons
    )
    assert (
        ui_loop.state.live_edit_mapping_location
        == session._config.live_edit_mapping_location
    )
    assert (
        ui_loop.state.native_dit_disabled_for_live_edit
        is session._config.native_dit_disabled_for_live_edit
    )
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
    assert app._config.initial_game_mode == "race"
    assert app._config.initial_map_path == _DEMO_RACE_MAP.resolve()
    assert app._config.initial_race_course_id == "grand-prix"

    session = app.create_session(app.session_desc())
    session.init()
    ui_loop, model_loop = session._take_loops()

    assert ui_loop.state._menu_stage == "loading"
    model_loop._run_message_batch()
    assert model_loop.state.game_selected
    assert model_loop.state.config.game_mode == "race"
    assert model_loop.state.config.race_course_id == "grand-prix"


def test_user_config_overrides_model_and_game_without_selecting_menus(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """\
model:
  device: cpu
  pipeline:
    diffusion_model:
      seed: 5678
game:
  gamepad_button_style: PlayStation
  taxi:
    seed: 1234
presentation:
  show_live_edit_buttons: false
  live_edit_mapping_location: control hints
  show_current_prompt: true
runtime:
  prewarm_blocks: 0
""",
        encoding="utf-8",
    )
    app = _application()

    app.init(["--config", str(config_path)])

    assert app._config is not None
    assert app._config.initial_game_mode is None
    assert app._config.initial_map_path is None
    assert app._config.initial_race_course_id is None
    assert app._config.model_preset_name == _STUB_PIPELINE_CONFIG.name
    assert app._config.device == "cpu"
    assert app._config.game.seed == 1234
    assert app._config.gamepad_button_style == "PlayStation"
    assert not app._config.show_live_edit_buttons
    assert app._config.live_edit_mapping_location == "control hints"
    assert app._config.show_current_prompt
    pipeline_config = app._pipeline_config
    assert pipeline_config is not None
    assert pipeline_config.diffusion_model.seed == 5678
    assert app.session_desc().video_width == 1280
    assert app.session_desc().video_height == 704


def test_explicit_cli_overrides_user_config_without_rewriting_it(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """\
model:
  device: cuda
presentation:
  show_fps: false
runtime:
  prewarm_blocks: 7
""",
        encoding="utf-8",
    )
    app = _application()

    app.init(
        [
            "--config",
            str(config_path),
            "--device",
            "cpu",
            "--show-fps",
            "--prewarm-blocks",
            "0",
        ]
    )

    assert app._config is not None
    assert app._config.device == "cpu"
    assert app._config.show_fps
    assert app._config.prewarm_blocks == 0
    document = app._config.settings_document
    assert document is not None
    assert document.settings.model.device == "cuda"
    assert not document.settings.presentation.show_fps
    assert document.settings.runtime.prewarm_blocks == 7
    assert document.cli_overrides[("model", "device")] == "cpu"
    assert document.cli_overrides[("presentation", "show_fps")] is True


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
            self.control_actions = BoundActionState(ControlsConfig())

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
            self.control_actions = BoundActionState(ControlsConfig())

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
            SimpleNamespace(
                total_blocks=None,
                pipeline_profiling=False,
                controls=ControlsConfig(),
            ),
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
        last_hdmap=torch.ones(1, 3, 4, 4),
        last_pose=np.eye(4, dtype=np.float32),
        prewarm_complete=True,
        game_selected=True,
    )
    loop = CrazyRobotaxiModelLoop()
    loop.state = state

    results = loop.step(0, UserInputEvents([]))

    assert len(results) == 2
    assert torch.all(results[1].read_output() == 1.0)
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
def test_diagnostics_flag_does_not_enable_pipeline_profiling(
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
    assert app._config is not None
    assert app._config.pipeline_profiling is expected


@pytest.mark.parametrize(
    ("live_edit", "expected_native_dit"),
    [
        (LiveEditConfig(style=LiveEditStyleConfig(enabled=True)), "disabled"),
        (LiveEditConfig(weather=LiveEditWeatherConfig(enabled=True)), "disabled"),
        (
            LiveEditConfig(
                obstacle=LiveEditObstacleConfig(enabled=True, guide_scale=1.0)
            ),
            "disabled",
        ),
        (LiveEditConfig(coins=LiveEditCoinsConfig(enabled=True)), "required"),
        (
            LiveEditConfig(
                obstacle=LiveEditObstacleConfig(enabled=True, guide_scale=0.0)
            ),
            "required",
        ),
    ],
)
def test_live_edit_disables_native_dit_only_when_required(
    live_edit: LiveEditConfig,
    expected_native_dit: str,
) -> None:
    transformer = replace(
        cast(
            _StubTransformerConfig,
            _STUB_PIPELINE_CONFIG.diffusion_model.transformer,
        ),
        native_dit_acceleration="required",
    )
    pipeline = replace(
        _STUB_PIPELINE_CONFIG,
        diffusion_model=replace(
            _STUB_PIPELINE_CONFIG.diffusion_model,
            transformer=transformer,
        ),
    )

    configured = _configure_live_edit_pipeline(pipeline, live_edit)

    assert (
        configured.diffusion_model.transformer.native_dit_acceleration
        == expected_native_dit
    )
    assert configured.encoder == pipeline.encoder


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
            _STUB_DEFAULTS,
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


def test_live_edit_item_cli_enables_required_abilities(tmp_path: Path) -> None:
    app = _application()
    arguments = [
        "--config",
        str(tmp_path / "config.yaml"),
        "--live-edit-items",
        "--live-edit-item-types",
        "rain,mystery",
        "--no-live-edit-style",
        "--no-live-edit-weather",
    ]

    with patch(
        "crazy_robotaxi.application.resolve_live_edit_assets",
        side_effect=lambda config: config,
    ):
        app.init(arguments)

    assert app._config is not None
    assert app._config.live_edit.items.enabled
    assert app._config.live_edit.style.enabled
    assert app._config.live_edit.weather.enabled


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
    tmp_path: Path,
    arguments: list[str],
    expected: bool,
) -> None:
    app = _application()

    app.init(["--config", str(tmp_path / "config.yaml"), *arguments])

    assert app._config is not None
    assert app._config.show_fps is expected


def test_controls_directory_is_a_separate_cli_only_config(tmp_path: Path) -> None:
    controls_dir = tmp_path / "controls"
    controls_dir.mkdir()
    (controls_dir / "keyboard.yaml").write_text(
        "schema_version: 1\nrestart: [p]\n",
        encoding="utf-8",
    )
    app = _application()

    app.init(
        [
            "--config",
            str(tmp_path / "config.yaml"),
            "--controls-dir",
            str(controls_dir),
        ]
    )

    assert app._config is not None
    assert app._config.controls.keyboard.restart[0] is not None
    assert app._config.controls.keyboard.restart[0].code == "p"
    assert (
        app._config.control_documents["keyboard"].path
        == (controls_dir / "keyboard.yaml").resolve()
    )


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

    with pytest.raises(ValueError) as exc_info:
        app.create_session(desc)
    assert str(exc_info.value) == (
        "Session/model dimensions (640, 704) do not match renderer raster dimensions "
        "(1280, 704). You may have changed Width or Height under Options > Renderer > "
        "Raster, or used --width or --height; those settings control model rendering. "
        "To resize only the displayed output, use Options > Presentation > Width and "
        "Height or --display-width and --display-height."
    )


def test_user_settings_resize_window_to_presentation_resolution(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "presentation:\n  width: 1920\n  height: 1080\n",
        encoding="utf-8",
    )
    app = _application()
    model_desc = app.session_desc()
    app.init(["--config", str(config_path), "--device", "cpu"])

    session = cast(CrazyRobotaxiSession, app.create_session(model_desc))
    session.init()
    ui_loop = session.ui_loop

    assert isinstance(ui_loop, CrazyRobotaxiImGuiUILoop)
    assert session.session_desc.video_width == 1280
    assert session.session_desc.video_height == 704
    assert session._config.renderer.raster.resolution_wh == (1280, 704)
    assert (ui_loop.state.width, ui_loop.state.height) == (1280, 704)
    assert ui_loop.state.presentation_size == (1920, 1080)
    renderer = cast(Any, ui_loop.renderer)
    assert (renderer.width, renderer.height) == (1280, 704)
    assert ui_loop.flush_ui_loop_requests() is None

    state = SimpleNamespace(
        width=1280,
        height=704,
        presentation_size=(1920, 1080),
        consume_input_events=lambda _events: None,
        draw=lambda *_args, **_kwargs: None,
    )

    def resize(width: int, height: int) -> None:
        state.width = width
        state.height = height

    state.resize = resize
    ui_loop.state = cast(Any, state)
    assert ui_loop.step_ui(None, 0, UserInputEvents([])) is None
    request = ui_loop.flush_ui_loop_requests()

    assert (renderer.width, renderer.height) == (1280, 704)
    assert request is not None
    assert request.new_window_size == (1920, 1080)

    assert ui_loop.step_ui(None, 1, UserInputEvents([])) is None
    assert (ui_loop.state.width, ui_loop.state.height) == (1920, 1080)
    assert (renderer.width, renderer.height) == (1920, 1080)
    assert ui_loop.flush_ui_loop_requests() is None

    state.presentation_size = (1600, 900)
    assert ui_loop.step_ui(None, 2, UserInputEvents([])) is None
    request = ui_loop.flush_ui_loop_requests()
    assert request is not None
    assert request.new_window_size == (1600, 900)
    assert (renderer.width, renderer.height) == (1920, 1080)

    assert ui_loop.step_ui(None, 3, UserInputEvents([])) is None
    assert (ui_loop.state.width, ui_loop.state.height) == (1600, 900)
    assert (renderer.width, renderer.height) == (1600, 900)
    assert ui_loop.flush_ui_loop_requests() is None


def test_display_cli_overrides_saved_presentation_resolution(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "presentation:\n  width: 1920\n  height: 1080\n",
        encoding="utf-8",
    )
    app = _application()
    model_desc = app.session_desc()
    app.init(
        [
            "--config",
            str(config_path),
            "--display-width",
            "2560",
            "--display-height",
            "1440",
            "--device",
            "cpu",
        ]
    )

    session = cast(CrazyRobotaxiSession, app.create_session(model_desc))
    session.init()

    assert session._config.presentation_resolution_wh == (2560, 1440)


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


def test_no_ui_registers_headless_loop_and_keeps_cli_selection() -> None:
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
            "--no-ui",
            "--total-blocks",
            "4",
            "--game-mode",
            "race",
            "--race-course",
            "grand-prix",
            "--map",
            str(_DEMO_RACE_MAP),
        ]
    )
    assert app._config is not None
    assert app._config.no_ui

    session = app.create_session(app.session_desc())
    session.init()
    ui_loop, model_loop = session._take_loops()

    assert isinstance(ui_loop, CrazyRobotaxiHeadlessUILoop)
    assert isinstance(model_loop, CrazyRobotaxiModelLoop)
    assert model_loop.state.ui_loop is ui_loop
    assert ui_loop.state.initial_game_mode == "race"
    assert ui_loop.state.initial_map_path == _DEMO_RACE_MAP.resolve()
    assert ui_loop.state.initial_race_course_id == "grand-prix"


def test_no_ui_in_race_mode_requires_a_course() -> None:
    app = _application(
        pipeline_factory=lambda config, device: object(),
        scene_factory=lambda request, raster: _scene(),
    )
    app.init(
        [
            "--device",
            "cpu",
            "--no-ui",
            "--total-blocks",
            "4",
            "--game-mode",
            "race",
            "--map",
            str(_DEMO_RACE_MAP),
        ]
    )

    session = app.create_session(app.session_desc())
    with pytest.raises(ValueError, match="requires --race-course"):
        session.init()


def test_no_ui_requires_explicit_mode_map_and_block_count() -> None:
    app = _application(
        pipeline_factory=lambda config, device: object(),
        scene_factory=lambda request, raster: _scene(),
    )
    app.init(["--device", "cpu", "--no-ui"])

    session = app.create_session(app.session_desc())
    with pytest.raises(ValueError, match="--no-ui requires"):
        session.init()


def test_headless_loop_presents_video_channel_and_finishes() -> None:
    frame = torch.zeros((3, 4, 6))
    presented: list[tuple[Tensor, ...]] = [(frame, torch.ones((3, 4, 6)))]
    manager = SimpleNamespace(
        presented_frames=lambda: presented[0],
        presented_frame_count=1,
        composite=lambda background, layer: layer,
        has_pending_frames=lambda: False,
    )
    resets: list[str] = []
    state = SimpleNamespace(reset=lambda: resets.append("hud"))
    loop = CrazyRobotaxiHeadlessUILoop()
    loop.register_session_loop_objects(
        state=cast(TaxiHudState, state),
        frequency=0,
        shutdown_event=threading.Event(),
        failure_queue=queue.Queue(),
    )
    loop.register_session_ui_loop_objects(
        session_desc=_application().session_desc(),
        presentation_manager=cast(Any, manager),
    )
    loop._set_model_loop(
        cast(Any, SimpleNamespace(inference_state=ModelInferenceState.FINISHED))
    )

    result = loop.step(0, UserInputEvents([]))

    assert result is not None
    output = result.read_output()
    assert output.shape == (1, 3, 4, 6)
    assert torch.equal(output[0], frame)
    assert loop.is_finished()
    loop.reset()
    assert resets == ["hud"]
    presented[0] = ()
    assert loop.step(1, UserInputEvents([])) is None
