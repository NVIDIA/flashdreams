# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for Crazy Robotaxi's application boundary against FlashDreams V2."""

import queue
import re
import threading
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from crazy_robotaxi.application import (
    CrazyRobotaxiApplication,
    CrazyRobotaxiApplicationDefaults,
    _configure_live_edit_pipeline,
    _fit_bev_renderer_to_ui,
    _parser,
)
from crazy_robotaxi.controls import BoundActionState, ControlsConfig
from crazy_robotaxi.dynamics import TaxiVehicleConfig
from crazy_robotaxi.free_roam import FreeRoamSnapshot
from crazy_robotaxi.game_selection import GameSelection
from crazy_robotaxi.headless_ui import CrazyRobotaxiHeadlessUILoop
from crazy_robotaxi.live_edit.config import (
    LiveEditCoinsConfig,
    LiveEditConfig,
    LiveEditMapContextConfig,
    LiveEditObstacleConfig,
    LiveEditStyleConfig,
    LiveEditWeatherConfig,
    StyleSkin,
    WeatherPreset,
)
from crazy_robotaxi.physics import TaxiPhysicsWorld
from crazy_robotaxi.race import RaceGameSnapshot
from crazy_robotaxi.rules import TaxiGameSnapshot
from crazy_robotaxi.session import (
    CrazyRobotaxiModelLoop,
    CrazyRobotaxiSession,
    ModelState,
    _pace_realtime_chunk,
    _taxi_driver_command,
)
from crazy_robotaxi.settings import (
    SETTING_CLI_FLAGS,
    SettingsDocument,
    setting_value,
)
from crazy_robotaxi.ui import CrazyRobotaxiImGuiUILoop, TaxiHudState
from ludus_renderer.physx import PhysXStepTimings, _CompactPhysicsStep
from omnidreams_game_engine.config import BevConfig, RasterConfig
from omnidreams_game_engine.engine import GameEngine
from omnidreams_game_engine.model import WorldModelRollout
from omnidreams_game_engine.simulation.ego_vehicle_kinematics import (
    EgoVehicleKinematics,
)
from omnidreams_game_engine.game_map.types import GameMapTrafficVehicle
from omnidreams_game_engine.game_map.vicinity import GameMapVicinity
from omnidreams_game_engine.input import DriverInput
from omnidreams_game_engine.renderer_settings import RendererSettings
from omnidreams_game_engine.simulation.game_physics import (
    GamePhysicsWorld,
    _body_state_from_vehicle,
)
from omnidreams_game_engine.simulation.map_traffic import MapTrafficController
from omnidreams_game_engine.types import (
    CameraCalibration,
    ConditionBatch,
    DriverCommand,
    PhysicsDebugFrame,
    SceneDefinition,
    VehicleState,
)
from torch import Tensor

from flashdreams.api_v2.loop import ModelInferenceState
from flashdreams.infra.diffusion.model import DiffusionModelConfig
from flashdreams.infra.diffusion.scheduler.base import SchedulerConfig
from flashdreams.infra.diffusion.transformer.base import TransformerConfig
from flashdreams.infra.encoder.base import EncoderConfig
from flashdreams.infra.pipeline import StreamInferencePipelineConfig
from flashdreams.runtime_v2 import cli as runtime_cli
from flashdreams.runtime_v2.session_desc import PresentationMode, SessionDesc
from flashdreams.runtime_v2.user_input_event import (
    KeyboardInputState,
    KeyboardUserInputEvent,
)
from flashdreams.runtime_v2.user_input_events import UserInputEvents
from flashdreams.runtime_v2.video_tensor import VideoTensorLayout

pytestmark = pytest.mark.ci_cpu

_RACE_MAP = Path(__file__).parent / "maps" / "race_course.robotaxi.yaml"

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

_STUB_DEFAULTS = CrazyRobotaxiApplicationDefaults(
    pipeline_config=_STUB_PIPELINE_CONFIG, realtime_gameplay=False
)


def _application(
    *,
    defaults: CrazyRobotaxiApplicationDefaults = _STUB_DEFAULTS,
    **kwargs: Any,
) -> CrazyRobotaxiApplication:
    kwargs.setdefault("pipeline_factory", lambda config, device: object())
    kwargs.setdefault("scene_factory", lambda request, raster: _scene())
    kwargs.setdefault("native_preparer", lambda: None)
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


@pytest.mark.parametrize("inline", (False, True))
@pytest.mark.parametrize("mode", (None, "null", "mp4", "webrtc", "native-window"))
def test_cli_exports_options_without_starting_the_game(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    inline: bool,
    mode: str | None,
) -> None:
    def unexpected(*args: object, **kwargs: object) -> None:
        pytest.fail("Documentation export attempted game preparation")

    app = _application(
        pipeline_factory=unexpected,
        scene_factory=unexpected,
        native_preparer=unexpected,
    )
    mode_factory = runtime_cli.client_window_mode

    def cpu_mode(name: str, **kwargs: Any) -> Any:
        mode = mode_factory(name, **kwargs)
        monkeypatch.setattr(
            mode,
            "create",
            lambda parsed: SimpleNamespace(close=lambda: None),
        )
        monkeypatch.setattr(mode, "starting", lambda window: None)
        return mode

    monkeypatch.setattr(runtime_cli, "create_application", lambda slug: app)
    monkeypatch.setattr(runtime_cli, "client_window_mode", cpu_mode)
    monkeypatch.setattr(
        "crazy_robotaxi.application.load_controls_documents", unexpected
    )
    config_path = tmp_path / "invalid-config.yaml"
    config_path.write_text("invalid: [", encoding="utf-8")
    output = tmp_path / "options.md"
    flag = "--export-options-docs"
    export_args = [f"{flag}={output}"] if inline else [flag, str(output)]
    mode_args = [] if mode is None else ["--mode", mode]

    with pytest.raises(SystemExit) as exc:
        runtime_cli.entrypoint(
            [
                "test-app",
                *mode_args,
                "--output-path",
                str(tmp_path / "unused.mp4"),
                "--total-model-steps",
                "1",
                "--",
                *export_args,
                "--config",
                str(config_path),
                "--no-ui",
            ]
        )

    assert exc.value.code == 0
    reference = output.read_text(encoding="utf-8")
    assert reference.startswith("# Crazy Robotaxi options reference\n")
    assert "### GAME → TAXI" in reference
    assert "| **Seed:** | `game.taxi.seed` | Crazy Robotaxi |" in reference
    assert "--game-seed, --seed" in reference
    assert "## Application arguments\n\n```text\n" in reference
    assert "--export-options-docs PATH" in reference
    assert "race mode also requires --race-course" in " ".join(reference.split())
    assert "| YAML key |" in reference
    assert config_path.read_text(encoding="utf-8") == "invalid: ["
    assert app._config is None


def test_options_export_reports_a_write_error(tmp_path: Path, capsys) -> None:
    with pytest.raises(SystemExit) as exc:
        _application().init(["--export-options-docs", str(tmp_path)])

    assert exc.value.code == 2
    assert "cannot export options documentation" in capsys.readouterr().err


def test_tooltip_cli_flags_change_their_settings(tmp_path: Path) -> None:
    parser = _parser(_STUB_DEFAULTS)
    actions = {
        flag: action for action in parser._actions for flag in action.option_strings
    }
    paths_by_flag: dict[str, list[tuple[str, ...]]] = {}
    for key, help_text in SETTING_CLI_FLAGS.items():
        for flag in re.findall(r"--[a-z][a-z0-9-]*", help_text):
            assert flag in actions
            paths_by_flag.setdefault(flag, []).append(tuple(key.split(".")))

    document = SettingsDocument.load(
        tmp_path / "config.yaml",
        pipeline_config=_STUB_PIPELINE_CONFIG,
        width=1280,
        height=704,
    )
    base = replace(
        document.settings,
        presentation=replace(document.settings.presentation, width=1280, height=704),
    )
    app = _application()
    for flag, paths in paths_by_flag.items():
        action = actions[flag]
        before = base
        argv = [flag]
        current = setting_value(before, paths[0])
        if action.nargs == 0:
            enabled = not flag.startswith("--no-")
            for path in paths:
                before = document.update(before, path, not enabled)
        elif action.dest in ("live_edit_skin_first", "live_edit_weather_first"):
            assert isinstance(current, tuple)
            selected = current[-1]
            assert isinstance(selected, (StyleSkin, WeatherPreset))
            argv.append(selected.name)
        elif action.dest == "live_edit_item_types":
            argv.append("nitro")
        elif action.choices:
            argv.append(next(choice for choice in action.choices if choice != current))
        elif action.type is Path:
            argv.append(str(tmp_path / "override"))
        elif action.type is float:
            assert isinstance(current, (int, float))
            argv.append(str(current + 0.1))
        elif action.type is int:
            assert current is None or isinstance(current, int)
            argv.append(str(current + 1 if current is not None else 2))
        else:
            argv.append("cpu")

        after, _overrides = app._apply_cli_settings(before, parser.parse_args(argv))

        for path in paths:
            assert setting_value(after, path) != setting_value(before, path), (
                flag,
                path,
            )
    assert not document.path.exists()


@pytest.mark.parametrize("mode", (None, "null", "mp4", "webrtc", "native-window"))
def test_application_registers_model_and_imgui_ui_loops(mode: str | None) -> None:
    pipeline_closed: list[bool] = []
    pipeline = SimpleNamespace(close=lambda: pipeline_closed.append(True))
    pipeline_requests: list[tuple[object, str]] = []
    app = _application(
        pipeline_factory=lambda config, device: (
            pipeline_requests.append((config, device)) or pipeline
        ),
        scene_factory=lambda request, raster: _scene(),
    )
    desc = app.session_desc()
    if mode is not None:
        assert app.resolve_client_window_mode(mode) == mode
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
    assert ui_loop.state.allow_exit is (mode != "webrtc")
    assert session._presentation_manager._presentation_stream is None
    assert model_loop.state.pipeline is None
    assert app._pipeline is not None
    assert pipeline_requests == [(app._pipeline_config, "cpu")]
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
    model_loop.state.pipeline = pipeline
    model_loop.close()
    assert pipeline_closed == []
    app.close()
    assert pipeline_closed == [True]


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


def test_free_roam_cli_selection_starts_without_a_race_course(monkeypatch) -> None:
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
            "free-roam",
            "--map",
            str(_DEMO_RACE_MAP),
        ]
    )

    session = app.create_session(app.session_desc())
    session.init()
    ui_loop, model_loop = session._take_loops()
    model_loop._run_message_batch()

    assert ui_loop.state._menu_stage == "loading"
    assert model_loop.state.game_selected
    assert model_loop.state.config.game_mode == "free-roam"
    assert model_loop.state.config.race_course_id is None


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
        session_desc = SessionDesc()

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
        session_desc = SessionDesc()

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


@pytest.mark.parametrize(
    ("frame_count", "model_step_wall_ms", "expected_duration_ms", "expected_wait_s"),
    [
        (8, 100.0, 800.0 / 3.0, 1.0 / 6.0),
        (4, 100.0, 400.0 / 3.0, 1.0 / 30.0),
        (8, 300.0, 800.0 / 3.0, None),
    ],
)
def test_pace_realtime_chunk_waits_for_remaining_frame_budget(
    frame_count: int,
    model_step_wall_ms: float,
    expected_duration_ms: float,
    expected_wait_s: float | None,
) -> None:
    class RecordingEvent:
        def __init__(self) -> None:
            self.waits: list[float] = []

        def wait(self, timeout: float) -> None:
            self.waits.append(timeout)

    event = RecordingEvent()

    chunk_duration_ms, realtime_margin_ms = _pace_realtime_chunk(
        cast(threading.Event, event),
        frame_count=frame_count,
        frames_per_second=30,
        model_step_wall_ms=model_step_wall_ms,
    )

    assert chunk_duration_ms == pytest.approx(expected_duration_ms)
    assert realtime_margin_ms == pytest.approx(
        expected_duration_ms - model_step_wall_ms
    )
    assert event.waits == (
        [] if expected_wait_s is None else [pytest.approx(expected_wait_s)]
    )


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
                metadata={},
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

    assert app._pipeline is not None
    assert configured == [app._pipeline_config]
    session._pipeline_factory()
    assert configured == [app._pipeline_config]
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
        (
            LiveEditConfig(dynamic_prompts=LiveEditMapContextConfig(enabled=True)),
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
    assert len(raster_sizes) == len(app._scenes)
    assert set(raster_sizes) == {resolution_wh}
    assert session._config.renderer.raster.resolution_wh == resolution_wh
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


@pytest.mark.parametrize("prewarm_blocks", [0, 4])
@pytest.mark.parametrize("mode", ["free-roam", "race", "taxi"])
def test_realtime_prewarm_preserves_frozen_world_cache_and_restart(
    monkeypatch, tmp_path: Path, prewarm_blocks: int, mode: str
) -> None:
    from crazy_robotaxi.free_roam import FreeRoamGameRules
    from crazy_robotaxi.high_scores import RaceTimeStore
    from crazy_robotaxi.race import RaceController, RaceGameRules
    from crazy_robotaxi.rules import TaxiGameConfig, TaxiGameController, TaxiGameRules
    from omnidreams_game_engine.game_map import load_game_map

    engines = []
    trajectories = []
    generated = []

    class Pipeline:
        device = "cpu"

        def initialize_cache(self, **kwargs):
            return []

        def get_num_output_frames(self, index):
            return 5 if index == 0 else 8

        def generate(self, *, autoregressive_index, cache, input):
            assert autoregressive_index == len(cache)
            cache.append(autoregressive_index)
            generated.append((cache, autoregressive_index))
            return input

        def finalize(self, **kwargs):
            return {}

    class Renderer:
        def load_scene(self, scene):
            pass

        def render(self, trajectory):
            trajectories.append(trajectory)
            return ConditionBatch(
                torch.zeros(1, 1, len(trajectory.vehicle_states), 3, 4, 4)
            )

        def close(self):
            pass

    def build_engine():
        rules = FreeRoamGameRules()
        if mode == "race":
            game_map = load_game_map(_RACE_MAP)
            rules = RaceGameRules(
                RaceController(
                    game_map,
                    game_map.race_courses[0],
                    VehicleState(4, 2, 1, 0, 0, 0),
                    RaceTimeStore(tmp_path / "race-times.csv"),
                )
            )
        elif mode == "taxi":
            rules = TaxiGameRules(
                TaxiGameController(
                    scene_id="warmup-test",
                    reference_route_world=np.array(
                        [[0, 0, 0], [1000, 0, 0]], dtype=np.float32
                    ),
                    initial_state=VehicleState(4, 2, 1, 0, 0, 0),
                    config=TaxiGameConfig(
                        seed=42, high_scores_path=tmp_path / "scores.csv"
                    ),
                )
            )
        monkeypatch.setattr(
            rules, "advance_frames", MagicMock(wraps=rules.advance_frames)
        )
        simulation = EgoVehicleKinematics(
            initial_state=VehicleState(4, 2, 1, 0, 0, 0),
            vehicle_config=TaxiVehicleConfig(),
            ground_snapper=None,
            initial_timestamp_us=123_000,
            include_initial_state_in_first_chunk=True,
        )
        simulation._integrate_fn = MagicMock(wraps=simulation._integrate_fn)
        engine = GameEngine(
            simulation=simulation,
            rules=rules,
            condition_renderer=Renderer(),
            frame_interval_s=1 / 30,
        )
        monkeypatch.setattr(
            engine,
            "live_edit",
            SimpleNamespace(
                style=SimpleNamespace(
                    attach_v2=lambda *args: None,
                    update_map_context=MagicMock(),
                    after_v2_chunk=lambda: None,
                    active_prompt="",
                ),
                hud_statuses=lambda: None,
            ),
            raising=False,
        )
        engines.append(engine)
        return engine

    rollout = WorldModelRollout(
        pipeline=Pipeline(),
        scene=_scene(width=4, height=4),
        engine_factory=build_engine,
    )
    app = _application()
    app.init(["--device", "cpu", "--prewarm-blocks", str(prewarm_blocks)])
    session = app.create_session(app.session_desc())
    session.init()
    ui_loop, loop = session._take_loops()
    state = loop.state
    state.config = replace(state.config, realtime_gameplay=True, total_blocks=1)
    state.session_desc = replace(state.session_desc, video_width=4, video_height=4)
    state.rollout = rollout
    state.game_selected = True
    # The model-side engine may render recorded observations, but no gameplay
    # process is needed to exercise cache lifetime and AR numbering on CPU.
    monkeypatch.setattr(ModelState, "ensure_rollout", lambda self: self.rollout)
    for epoch in range(2):
        initial_game_frame = rollout.engine.current_game_frame
        state._prewarm_rollout()
        cache = rollout.cache
        assert cache == list(range(prewarm_blocks))
        assert state.blocks_generated == 0
        assert state.prewarmed_blocks == prewarm_blocks
        assert state.prewarm_complete
        assert len(engines) == epoch + 1
        engine = engines[-1]
        engine.rules.advance_frames.assert_not_called()
        engine.simulation._integrate_fn.assert_not_called()
        if prewarm_blocks:
            engine.live_edit.style.update_map_context.assert_called_once_with(
                engine.simulation.current_state
            )
        else:
            engine.live_edit.style.update_map_context.assert_not_called()
        assert engine.current_game_frame == initial_game_frame
        if mode == "race":
            assert engine.current_game_frame.session_state == "awaiting_start"
            assert engine.current_game_frame.elapsed_time_us == 0
        for trajectory in trajectories:
            assert set(trajectory.timestamps_us) == {123_000}
            assert all(
                v == VehicleState(4, 2, 1, 0, 0, 0) for v in trajectory.vehicle_states
            )
            np.testing.assert_array_equal(
                trajectory.rig_poses_world,
                np.repeat(
                    trajectory.rig_poses_world[:1],
                    len(trajectory.vehicle_states),
                    axis=0,
                ),
            )
            if mode == "taxi":
                assert isinstance(initial_game_frame, TaxiGameSnapshot)
                passengers = trajectory.dynamic_actors
                assert passengers
                expected = (
                    initial_game_frame.pickup_passengers_xyz_m
                    or initial_game_frame.pickup_targets_xyz_m
                )
                assert len(passengers) == len(expected)
                ordered = sorted(
                    passengers, key=lambda a: tuple(a.translations_world[0])
                )
                for actor, point in zip(ordered, sorted(expected), strict=True):
                    assert actor.entity_id.startswith("taxi-passenger-")
                    assert actor.timestamps_us.tolist() == [123_000]
                    np.testing.assert_allclose(
                        actor.translations_world[0, :2], point[:2]
                    )
        trajectories.clear()
        results = loop.step(0, UserInputEvents([]))
        assert generated[-1][0] is cache
        assert generated[-1][1] == prewarm_blocks
        assert state.blocks_generated == 1
        assert state.finished
        assert results[0].frame_count == (8 if prewarm_blocks else 5)
        trajectories.clear()
        state.restart_game()
        assert rollout.cache is not cache
        assert state.prewarmed_blocks == 0
        assert not state.prewarm_complete
        assert state.prewarm_wall_ms == 0
    state.close()


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


def test_traffic_window_refresh_is_not_repeated_by_next_physics_step() -> None:
    native = MagicMock()
    vehicle = TaxiVehicleConfig()
    with patch(
        "omnidreams_game_engine.simulation.game_physics.GameplayPhysXWorld",
        return_value=native,
    ):
        world = TaxiPhysicsWorld(_scene(), vehicle)
    traffic = GameMapTrafficVehicle(
        vehicle_id="test-car",
        node_ids=("a", "b"),
        end_behavior="wrap",
        vehicle_type="car",
        dimensions_lwh_m=(4.5, 1.8, 1.5),
        speed_mps=None,
        start_distance_m=10.0,
        centerline_world=np.asarray([[0, 0, 0], [100, 0, 0], [0, 0, 0]]),
        speed_limits_mps=np.full(3, 10.0),
        route_element_ids=("road", "road"),
    )
    controller = MapTrafficController((traffic,), vehicle)
    world._map_traffic = controller
    world._actor_controllers = (controller,)
    resolver = MagicMock()
    world._vicinity_resolver = resolver
    state = VehicleState(0, 0, 0, 0, 0, 0)
    native.step_compact.return_value = _CompactPhysicsStep(
        ego=_body_state_from_vehicle(state, world._ego_model.half_extents_m[2]),
        impact=False,
        struck_object_ids=frozenset(),
        actor_samples=(),
        track_samples=(),
        timings=PhysXStepTimings(0, 0, 0, 0, 0, 0, 0),
    )
    for visible in (frozenset({"road"}), frozenset()):
        resolver.resolve.return_value = GameMapVicinity("road", visible, visible)
        native.synchronize.reset_mock()
        assert world.synchronize_window(np.zeros(2), timestamp_us=0)
        expected = {obj.object_id: obj for obj in controller.active_objects}
        assert world._active_objects_by_id == expected
        GamePhysicsWorld.step(world, state, 0, 1 / 30)
        native.synchronize.assert_called_once()


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


@pytest.mark.parametrize("no_ui", (False, True))
@pytest.mark.parametrize("game_mode", ("race", "free-roam"))
def test_cli_selection_starts_game_with_or_without_ui(
    monkeypatch: pytest.MonkeyPatch, no_ui: bool, game_mode: str
) -> None:
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
    state = SimpleNamespace(
        reset=lambda: resets.append("hud"), consume_input_events=lambda events: None
    )
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

    assert result
    output = result[0].read_output()
    assert output.shape == (1, 3, 4, 6)
    assert torch.equal(output[0], frame)
    assert loop.is_finished()
    loop.reset()
    assert resets == ["hud"]
    presented[0] = ()
    assert loop.step(1, UserInputEvents([])) == []


@pytest.mark.parametrize("game_mode", ["taxi", "race", "free-roam"])
@pytest.mark.parametrize("source", ["file", "socket"])
def test_scripted_game_runs_through_runtime_with_all_views_and_aligned_telemetry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    game_mode: str,
    source: str,
) -> None:
    import json
    import socket
    import time

    from crazy_robotaxi.debug import RobotaxiDebugWindow
    from crazy_robotaxi.settings import SettingsDocument
    from omnidreams_game_engine.types import VehicleState
    from PIL import Image

    from flashdreams.runtime_v2.application_runner import ApplicationRunner

    script = tmp_path / "drive.yaml"
    commands = [
        {
            "id": "press",
            "timestamp_us": 100_000,
            "type": "keyboard",
            "key": "w",
            "pressed": True,
        },
        {
            "id": "release",
            "timestamp_us": 400_000,
            "type": "keyboard",
            "key": "w",
            "pressed": False,
        },
        {"id": "end", "timestamp_us": 1_000_000, "type": "close"},
    ]
    script.write_text(json.dumps({"events": commands}))
    (tmp_path / "settings.yaml").write_text(
        "game:\n  taxi:\n    vehicle: {max_speed_mps: 17}\n"
        "    rules: {global_time_s: 180}\n"
        "live_edit:\n  items: {nitro_duration_s: 2.5}\n"
    )
    snapshot: TaxiGameSnapshot | RaceGameSnapshot | FreeRoamSnapshot = TaxiGameSnapshot(
        phase="seeking_pickup",
        target_xyz_m=(25, 0, 0),
        distance_m=25,
        relative_bearing_rad=0,
        target_radius_m=5,
        remaining_time_s=None,
        score=0,
    )
    if game_mode == "free-roam":
        snapshot = FreeRoamSnapshot()
    elif game_mode == "race":
        snapshot = RaceGameSnapshot(
            map_id="test",
            course_id="test-loop",
            session_state="racing",
            target_kind="checkpoint",
            target_element_id="finish",
            target_xyz_m=(25, 0, 0),
            gate_start_xyz_m=(25, -5, 0),
            gate_end_xyz_m=(25, 5, 0),
            checkpoint_markers=False,
            distance_m=25,
            relative_bearing_rad=0,
            checkpoint_index=0,
            checkpoint_count=1,
            completed_laps=0,
            lap_count=0,
            elapsed_time_us=0,
            best_time_us=None,
        )
    applied = []
    closed = []
    debug_enabled = []

    def engine_step(commands):
        start = len(applied)
        applied.extend(commands)
        count = len(commands)
        vehicles = tuple(VehicleState(start + i, 0, 0, 0, i, 0) for i in range(count))
        hdmap = torch.zeros(1, 1, count, 3, 4, 8)
        for index in range(count):
            hdmap[:, :, index] = (start + index) / 10 - 1
        return SimpleNamespace(
            metrics={},
            game_frames=tuple(snapshot for i in range(count)),
            trajectory=SimpleNamespace(
                timestamps_us=np.arange(start, start + count) * 33333,
                rig_poses_world=np.repeat(
                    np.eye(4, dtype=np.float32)[None], count, axis=0
                ),
                vehicle_states=vehicles,
                physx_timings=None,
                physics_debug_frames=tuple(
                    PhysicsDebugFrame(
                        ego_position_m=np.asarray(
                            [v.x_m, v.y_m, 0.8], dtype=np.float32
                        ),
                        ego_orientation_xyzw=np.asarray([0, 0, 0, 1], dtype=np.float32),
                        ego_dimensions_lwh=np.asarray([4.8, 2, 1.6], dtype=np.float32),
                        actor_positions_m=np.empty((0, 3), dtype=np.float32),
                        actor_orientations_xyzw=np.empty((0, 4), dtype=np.float32),
                        actor_dimensions_lwh=np.empty((0, 3), dtype=np.float32),
                        barrier_segments_xy_m=np.empty((0, 2, 2), dtype=np.float32),
                        barrier_thicknesses_m=np.empty(0, dtype=np.float32),
                        barrier_heights_m=np.empty(0, dtype=np.float32),
                        vehicle_state=v,
                        actor_collision=False,
                        static_barrier_collision=False,
                        physics_started_ns=time.monotonic_ns(),
                        physics_completed_ns=time.monotonic_ns(),
                    )
                    for v in vehicles
                ),
            ),
            condition=SimpleNamespace(hdmap_bvtchw=hdmap, bev_tchw=None),
        )

    engine = SimpleNamespace(
        current_game_frame=snapshot,
        step=engine_step,
        is_running=True,
        simulation=SimpleNamespace(set_physx_debug_enabled=debug_enabled.append),
        close=lambda: closed.append(True),
    )
    monkeypatch.setattr(
        "crazy_robotaxi.session.build_taxi_engine", lambda **kwargs: engine
    )
    original_apply = DriverInput.apply
    read_keys = []

    def apply(driver, events):
        read_keys.extend(
            event
            for event in events.get_events()
            if isinstance(event, KeyboardUserInputEvent)
        )
        return original_apply(driver, events)

    monkeypatch.setattr(DriverInput, "apply", apply)
    encoded = {}

    class Encoder:
        def __init__(self, path, **kwargs):
            self.path = path
            encoded[path.stem] = []

        def write(self, frames):
            encoded[self.path.stem].extend(frames.copy())

        def close(self):
            pass

    monkeypatch.setattr("crazy_robotaxi.debug_capture.Mp4Encoder", Encoder)

    def generate(**kwargs):
        # Deliberately slower than a chunk: inputs must still arrive while busy.
        time.sleep(0.18)
        return -kwargs["input"]

    pipeline = SimpleNamespace(
        device="cpu",
        initialize_cache=lambda **kwargs: None,
        get_num_output_frames=lambda index: 4,
        generate=generate,
        finalize=lambda **kwargs: None,
    )
    defaults = replace(_STUB_DEFAULTS, width=8, height=4)
    app = _application(
        defaults=defaults,
        pipeline_factory=lambda config, device: pipeline,
        scene_factory=lambda request, raster: _scene(width=8, height=4),
    )
    app.resolve_client_window_mode("robotaxi-debug")

    class Window(RobotaxiDebugWindow):
        def open(self, desc):
            super().open(desc)
            if source == "socket":
                with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as client:
                    for command in commands:
                        client.sendto(
                            json.dumps(command).encode(),
                            str(self.directory / "control.sock"),
                        )

    window = Window(
        tmp_path / "capture",
        snapshot_every=2,
        cpu_sample_ms=10,
        generation_delays_ms=(120, 0),
    )
    ApplicationRunner(app, window).run(
        app.session_desc(),
        [
            "--device",
            "cpu",
            "--game-time-s",
            "140",
            "--game-mode",
            game_mode,
            "--map",
            str(_RACE_MAP),
            *(["--race-course", "test-loop"] if game_mode == "race" else []),
            *(["--drive-script", str(script)] if source == "file" else []),
            "--no-ui",
            "--prewarm-blocks",
            "0",
            "--config",
            str(tmp_path / "settings.yaml"),
        ],
        timeout_seconds=10,
    )
    records = [
        json.loads(line)
        for line in (window.directory / "telemetry.jsonl").read_text().splitlines()
    ]
    assert len(records) == len(applied)
    frame_ids = [record["frame"] for record in records]
    assert all(b > a for a, b in zip(frame_ids, frame_ids[1:]))
    assert [record["physics_vehicle"]["x_m"] for record in records] == frame_ids
    assert [record["simulation_timestamp_us"] for record in records] == [
        i * 33333 for i in frame_ids
    ]
    assert any(command.throttle > 0 for command in applied)
    assert applied[-1].throttle == 0
    assert [int(event.timestamp) for event in read_keys] == [100_000, 400_000]
    assert debug_enabled and all(debug_enabled) and closed == [True]
    assert (
        len(encoded["generated"])
        == len(encoded["hdmap"])
        == len(encoded["physics"])
        == len(records)
    )
    for index, frame_id in enumerate(frame_ids):
        assert int(encoded["hdmap"][index][0, 0, 0]) == pytest.approx(
            np.clip(frame_id * 12.75, 0, 255), abs=1
        )
    with Image.open(
        window.directory / "frames/generated" / f"{len(records) - 1:06d}.png"
    ) as image:
        assert image.size == (8, 4)
    acknowledgments = [
        json.loads(line)
        for line in (window.directory / "input_events.jsonl").read_text().splitlines()
    ]
    assert [
        event["id"] for event in acknowledgments if event["status"] == "consumed"
    ] == ["press", "release"]
    assert not [event for event in acknowledgments if event["status"] == "rejected"]
    steps = [
        json.loads(line)
        for line in (window.directory / "model_steps.jsonl").read_text().splitlines()
    ]
    assert steps and all("model_step_cpu_ms" in step["metrics"] for step in steps)
    assert all(step["metrics"]["generate_wall_ms"] >= 150 for step in steps)
    assert all(
        step["metrics"]["generate_cpu_ms"] < step["metrics"]["generate_wall_ms"]
        for step in steps
    )
    samples = [
        json.loads(line)
        for line in (window.directory / "cpu_samples.jsonl").read_text().splitlines()
    ]
    assert any(
        any(frame["function"] == "generate" for frame in sample["stack"])
        for sample in samples
    )
    manifest = json.loads((window.directory / "manifest.json").read_text())
    assert manifest["finalized"] and manifest["frames_written"] == len(records)
    assert manifest["model_preset"] == _STUB_PIPELINE_CONFIG.name
    report = json.loads((window.directory / "timing_summary.json").read_text())
    assert report["generation_adherence"] == "not_evaluated"
    assert report["generation_delays"]
    assert report["generation_delays"][0]["requested_ms"] == 120
    progress = report["physics_progress"][0]
    assert progress["simulation_to_wall_ratio"] < 0.9
    assert progress["physics_gap_ms"]["max"] >= 150
    by_id = {row["id"]: row for row in report["inputs"]}
    press = by_id["press"]
    assert press["physics_association"] == "observed"
    assert press["dispatched_us"] < press["consumed_us"] <= press["physics_started_us"]
    assert press["scheduled_to_physics_started_ms"] > 100
    assert press["generation_returned_us"] >= press["physics_completed_us"]
    assert press["first_presentation_us"] >= press["generation_returned_us"]
    captured_settings = window.directory / "inputs/settings.yaml"
    assert json.loads(captured_settings.read_text()) == manifest["settings"]
    restored = SettingsDocument.load(
        captured_settings,
        pipeline_config=_STUB_PIPELINE_CONFIG,
        width=8,
        height=4,
    ).settings
    assert restored.game.taxi.vehicle.max_speed_mps == 17
    assert restored.game.taxi.rules.global_time_s == 140
    assert restored.live_edit.items.nitro_duration_s == 2.5
    assert restored.model.device == "cpu"
    assert restored.runtime.prewarm_blocks == 0
    assert (
        window.directory / "inputs" / _RACE_MAP.name
    ).read_text() == _RACE_MAP.read_text()
    from crazy_robotaxi.driving_script import load_driving_script

    if source == "file":
        captured_script = load_driving_script(window.directory / "inputs/drive.yaml")
        assert captured_script.events == load_driving_script(script).events


@pytest.mark.parametrize("no_ui", [False, True])
def test_live_debug_respects_ui_selection_and_runtime_input(
    tmp_path: Path, no_ui: bool
) -> None:
    from crazy_robotaxi.debug import (
        DEBUG_METADATA_KEY,
        RobotaxiLiveDebugUILoop,
        RobotaxiDebugUILoop,
    )

    app = _application()
    app.resolve_client_window_mode("robotaxi-debug")
    app.init(
        [
            "--device",
            "cpu",
            "--game-mode",
            "free-roam",
            "--map",
            str(_RACE_MAP),
            "--config",
            str(tmp_path / "settings.yaml"),
            "--prewarm-blocks",
            "2",
            "--total-blocks",
            "2",
            "--debug-view",
            "paired" if no_ui else "gameplay",
        ]
    )
    session = app.create_session(app.session_desc())
    session.init()
    ui_loop, model_loop = session._take_loops()
    assert isinstance(
        ui_loop, RobotaxiDebugUILoop if no_ui else RobotaxiLiveDebugUILoop
    )
    assert isinstance(model_loop, CrazyRobotaxiModelLoop)
    assert model_loop.state.config.driving_script is None
    assert model_loop.state.config.prewarm_blocks == 2
    assert session.session_desc.presentation_mode is PresentationMode.CONTINUOUS
    metadata = session.session_desc.metadata[DEBUG_METADATA_KEY]
    assert metadata["input_mode"] == "runtime" and metadata["paired_views"] == no_ui


@pytest.mark.parametrize(
    "mode, extra, error",
    [
        ("robotaxi-debug", ["--ui"], "conflicts"),
        ("native", [], "requires --mode"),
    ],
)
def test_debug_view_rejects_conflicting_or_non_debug_use(tmp_path, mode, extra, error):
    app = _application()
    app.resolve_client_window_mode(mode)
    with pytest.raises(ValueError, match=error):
        app.init(
            [
                "--config",
                str(tmp_path / "settings.yaml"),
                "--debug-view",
                "paired",
                *extra,
            ]
        )


@pytest.mark.parametrize("missing", ["--map", "--game-mode", "--race-course"])
def test_scripted_headless_startup_requires_complete_cli_selection(
    tmp_path: Path,
    missing: str,
) -> None:
    script = tmp_path / "drive.yaml"
    script.write_text("events: [{id: end, timestamp_us: 0, type: close}]")
    arguments = [
        "--device",
        "cpu",
        "--drive-script",
        str(script),
        "--game-mode",
        "race",
        "--map",
        str(_RACE_MAP),
        "--race-course",
        "test-loop",
    ]
    index = arguments.index(missing)
    del arguments[index : index + 2]
    with pytest.raises(ValueError, match="--drive-script.*requires"):
        _application().init(arguments)


@pytest.mark.parametrize("game_mode", ["taxi", "free-roam", "race"])
def test_engine_only_debug_skips_model_loading_and_labels_capture(tmp_path, game_mode):
    from crazy_robotaxi.debug import DEBUG_METADATA_KEY
    from crazy_robotaxi.debug_engine import SemanticDebugPipeline

    prepared = []

    def unexpected_model(*_args):
        pytest.fail("Engine-only debug attempted to construct a world model")

    app = _application(
        pipeline_factory=unexpected_model,
        native_preparer=lambda: prepared.append(True),
    )
    app.resolve_client_window_mode("robotaxi-debug")
    app.init(
        [
            "--config",
            str(tmp_path / "settings.yaml"),
            "--map",
            str(_RACE_MAP),
            "--game-mode",
            game_mode,
            "--device",
            "cpu",
            "--debug-engine-only",
            *(["--race-course", "test-loop"] if game_mode == "race" else []),
        ]
    )
    pipeline = app._pipeline
    assert isinstance(pipeline, SemanticDebugPipeline)
    assert prepared == [True]
    session = app.create_session(app.session_desc())
    metadata = session.session_desc.metadata[DEBUG_METADATA_KEY]
    assert metadata["engine_only"] is True
    assert metadata["world_model_loaded"] is False
    assert metadata["video_source"] == "semantic"
    assert metadata["input_mode"] == "runtime"
    assert metadata["game_mode"] == game_mode
    app.close()
    with pytest.raises(RuntimeError, match="closed"):
        pipeline.initialize_cache()


def test_engine_only_debug_requires_headless_debug_mode(tmp_path):
    def unexpected(*_args):
        pytest.fail("Invalid engine-only mode attempted preparation")

    app = _application(
        pipeline_factory=unexpected,
        scene_factory=unexpected,
        native_preparer=unexpected,
    )
    app.resolve_client_window_mode("native")
    with pytest.raises(ValueError, match="requires --mode robotaxi-debug"):
        app.init(["--config", str(tmp_path / "settings.yaml"), "--debug-engine-only"])


def test_engine_only_rejects_live_edit_before_resolving_model_assets(
    tmp_path, monkeypatch
):
    def unexpected(*_args):
        pytest.fail("Engine-only debug attempted model-asset preparation")

    monkeypatch.setattr(
        "crazy_robotaxi.application.resolve_live_edit_assets", unexpected
    )
    app = _application(pipeline_factory=unexpected)
    app.resolve_client_window_mode("robotaxi-debug")
    with pytest.raises(ValueError, match="live-edit abilities disabled"):
        app.init(
            [
                "--config",
                str(tmp_path / "settings.yaml"),
                "--debug-engine-only",
                "--live-edit-style",
            ]
        )


@pytest.mark.parametrize("game_mode", ["taxi", "free-roam", "race"])
@pytest.mark.parametrize("engine_only", [False, True])
@pytest.mark.parametrize("disable_traffic", [False, True])
def test_debug_traffic_selection_reaches_scene_requests_and_capture(
    tmp_path, game_mode, engine_only, disable_traffic
):
    from crazy_robotaxi.debug import DEBUG_METADATA_KEY

    requests = []
    app = _application(
        scene_factory=lambda request, raster: requests.append(request) or _scene()
    )
    app.resolve_client_window_mode("robotaxi-debug")
    app.init(
        [
            "--config",
            str(tmp_path / "settings.yaml"),
            "--map",
            str(_RACE_MAP),
            "--game-mode",
            game_mode,
            "--device",
            "cpu",
            *(["--debug-engine-only"] if engine_only else []),
            *(["--debug-no-traffic"] if disable_traffic else []),
            *(["--race-course", "test-loop"] if game_mode == "race" else []),
        ]
    )
    assert requests
    assert all(request.traffic_enabled is not disable_traffic for request in requests)
    session = app.create_session(app.session_desc())
    metadata = session.session_desc.metadata[DEBUG_METADATA_KEY]
    assert metadata["traffic_enabled"] is not disable_traffic
    assert metadata["engine_only"] is engine_only
    app.close()


def test_no_traffic_flag_rejects_native_mode_before_preparation(tmp_path):
    def unexpected(*_args):
        pytest.fail("Invalid traffic debug mode attempted preparation")

    app = _application(
        pipeline_factory=unexpected,
        scene_factory=unexpected,
        native_preparer=unexpected,
    )
    app.resolve_client_window_mode("native")
    with pytest.raises(
        ValueError, match="--debug-no-traffic requires --mode robotaxi-debug"
    ):
        app.init(["--config", str(tmp_path / "settings.yaml"), "--debug-no-traffic"])

