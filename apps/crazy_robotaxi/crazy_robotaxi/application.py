# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Process-lifetime Crazy Robotaxi application composition."""

from __future__ import annotations

import argparse
import logging
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from functools import partial
from pathlib import Path
from typing import Any, Literal

from omnidreams_game_engine.camera_defaults import DEFAULT_FRONT_CAMERA_LOGICAL_NAME
from omnidreams_game_engine.cli_args import (
    ExplicitArgTrackingArgumentParser,
    arg_was_explicit,
)
from omnidreams_game_engine.config import BevConfig, RasterConfig
from omnidreams_game_engine.game_map import GAME_MAP_SUFFIX, load_game_map_header
from omnidreams_game_engine.renderer_settings import RendererSettings
from omnidreams_game_engine.scene import SceneRequest, load_scene
from omnidreams_game_engine.types import SceneDefinition

from crazy_robotaxi.controls import (
    ControlDevice,
    ControlsConfig,
    ControlsDocument,
    GamepadButtonStyle,
    controls_config,
    default_controls_dir,
    load_controls_documents,
)
from crazy_robotaxi.game_selection import GameMapOption, GameMode, GameRaceCourseOption
from crazy_robotaxi.high_scores import default_high_scores_path, default_race_times_path
from crazy_robotaxi.live_edit.config import (
    LiveEditConfig,
    add_live_edit_args,
    live_edit_config_from_args,
    resolve_live_edit_assets,
)
from crazy_robotaxi.rules import TaxiGameConfig
from crazy_robotaxi.session import CrazyRobotaxiSession
from crazy_robotaxi.settings import (
    CrazyRobotaxiUserSettings,
    LiveEditMappingLocation,
    SettingsDocument,
    default_config_path,
)
from crazy_robotaxi.ui import bev_display_extent
from flashdreams.api_v2.application import IApplication
from flashdreams.api_v2.session import ISession
from flashdreams.infra.config import derive_config
from flashdreams.runtime_v2.session_desc import PresentationMode, SessionDesc
from flashdreams.runtime_v2.video_tensor import VideoTensorLayout

_ROOT = Path(__file__).resolve().parent
_DEFAULT_MAP = _ROOT / "maps" / "boulevard_district.robotaxi.yaml"
_VIDEO_FPS = 30
"""Generated-video cadence required by the model."""

_UI_FPS = 60
"""Input polling and HUD cadence used by Interactive Drive."""

_DEFAULT_INPUT_TRACE_PATH = (
    Path(tempfile.gettempdir()) / "crazy-robotaxi-input-trace.log"
)
"""Default line-oriented input trace written by the profiling flag."""

_LOGGER = logging.getLogger(__name__)

_DEFAULT_PREWARM_BLOCKS = 8
"""Blocks covering chunk2 cache filling and the first steady-state AR shape."""


@dataclass(frozen=True, slots=True)
class CrazyRobotaxiApplicationDefaults:
    """Defaults supplied by a world-model integration."""

    title: str = "Crazy Robotaxi"
    slug: str = "crazy-robotaxi"
    width: int = 1280
    height: int = 704
    pipeline_config: Any | None = None


@dataclass(frozen=True, slots=True)
class ApplicationConfig:
    """Validated options shared by sessions created by one application."""

    scene_request: SceneRequest
    renderer: RendererSettings
    game: TaxiGameConfig
    device: str
    total_blocks: int | None
    model_preset_name: str
    pipeline_profiling: bool
    prewarm_blocks: int
    """Hidden neutral blocks generated before the first presented game frame."""

    profile_input_latency: bool
    """Whether the UI displays and logs input-to-model-frame diagnostics."""

    input_trace_path: Path | None
    """Lifecycle trace destination when input profiling is enabled."""

    show_fps: bool
    """Whether the HUD displays the measured generated-video frame rate."""

    hud_enabled: bool = True
    """Whether gameplay HUD overlays are visible."""

    show_control_hints: bool = True
    """Whether gameplay control hints start visible."""

    show_live_edit_buttons: bool = True
    """Whether live-edit actions appear as clickable HUD buttons."""

    live_edit_mapping_location: LiveEditMappingLocation = "buttons"
    """Where active live-edit mappings appear in the gameplay HUD."""

    controls: ControlsConfig = ControlsConfig()
    """Process-start gameplay bindings."""

    gamepad_button_style: GamepadButtonStyle = "Xbox"
    """Gamepad button names displayed to the player."""

    control_documents: dict[ControlDevice, ControlsDocument] = field(
        default_factory=dict
    )
    """Per-device YAML documents edited by the Controls screens."""

    settings_document: SettingsDocument | None = None
    """User-authored YAML document edited by the Options screen."""

    initial_game_mode: GameMode | None = None
    """Configured game mode that skips the mode menu, if any."""

    initial_map_path: Path | None = None
    """Configured map that skips the map menu, if any."""

    initial_race_course_id: str | None = None
    """Configured race course that skips the course menu, if any."""

    game_mode: Literal["taxi", "race"] = "taxi"
    """Rules mode selected for every session created by the application."""

    race_course_id: str | None = None
    """Requested race course, or ``None`` for the map's first course."""

    race_times_path: Path | None = None
    """Persistent map- and course-scoped race leaderboard."""

    live_edit: LiveEditConfig = LiveEditConfig()
    """Flag-gated prompt, style, weather, pickup, nitro, and obstacle abilities."""

    native_dit_disabled_for_live_edit: bool = False
    """Whether live editing forced native DiT acceleration off."""

    visual_flare_enabled: bool = False
    """Whether collision feedback may darken the presented game frame."""


PipelineFactory = Callable[[Any, str], Any]
SceneFactory = Callable[[SceneRequest, Any], SceneDefinition]
_TRACE_METADATA_KEY = "trace_chunk_lifecycle"
_TRACE_PATH_METADATA_KEY = "trace_chunk_lifecycle_path"


class CrazyRobotaxiApplication(IApplication):
    """Configure isolated V2 game sessions with model-owned defaults."""

    def __init__(
        self,
        *,
        pipeline_factory: PipelineFactory | None = None,
        defaults: CrazyRobotaxiApplicationDefaults | None = None,
        scene_factory: SceneFactory | None = None,
    ) -> None:
        self._application_defaults = defaults or CrazyRobotaxiApplicationDefaults()
        self._defaults = RendererSettings(
            raster=RasterConfig(
                width=self._application_defaults.width,
                height=self._application_defaults.height,
            ),
            bev=BevConfig(),
        )
        self._pipeline_factory = pipeline_factory or _build_pipeline
        self._scene_factory = scene_factory or load_scene
        self._pipeline_config = self._application_defaults.pipeline_config
        self._config: ApplicationConfig | None = None
        self._map_options: tuple[GameMapOption, ...] = ()

    def session_desc(self) -> SessionDesc:
        """Declare the trained single-view output contract without loading."""
        raster = (
            self._defaults.raster
            if self._config is None
            else self._config.renderer.raster
        )
        return SessionDesc(
            output_layout=VideoTensorLayout.tchw,
            frames_per_second_for_ui=_UI_FPS,
            frames_per_second_for_step=_VIDEO_FPS,
            video_width=raster.width,
            video_height=raster.height,
        )

    def init(self, commandline_args: Sequence[str]) -> None:
        """Parse application options without starting another runtime."""
        default_pipeline = self._pipeline_config
        if default_pipeline is None:
            raise RuntimeError("A world-model integration must provide pipeline_config")
        args = _parser(self._application_defaults).parse_args(list(commandline_args))
        control_documents = load_controls_documents(
            args.controls_dir or default_controls_dir()
        )
        controls = controls_config(control_documents)
        config_path = args.config or default_config_path()
        settings_document = SettingsDocument.load(
            config_path,
            pipeline_config=default_pipeline,
            width=self._application_defaults.width,
            height=self._application_defaults.height,
        )
        settings, cli_overrides = self._apply_cli_settings(
            settings_document.settings,
            args,
        )
        settings_document.cli_overrides = cli_overrides
        initial_game_mode: GameMode | None = (
            args.game_mode if arg_was_explicit(args, "game_mode") else None
        )
        initial_map_path: Path | None = (
            args.map if arg_was_explicit(args, "map") else None
        )
        initial_race_course_id: str | None = (
            args.race_course if arg_was_explicit(args, "race_course") else None
        )
        if (
            settings.runtime.total_blocks is not None
            and settings.runtime.total_blocks <= 0
        ):
            raise ValueError("--total-blocks must be positive")
        if settings.runtime.prewarm_blocks < 0:
            raise ValueError("--prewarm-blocks must be non-negative")
        if settings.game.taxi.rules.global_time_s <= 0:
            raise ValueError("--game-time-s must be positive")
        if initial_game_mode != "race" and initial_race_course_id is not None:
            raise ValueError("--race-course requires --game-mode race")
        map_path: Path = args.map
        if initial_game_mode == "race":
            header = load_game_map_header(map_path.expanduser())
            if not header.race_course_ids:
                raise ValueError(f"Map {header.map_id!r} defines no race courses")
            if (
                initial_race_course_id is not None
                and initial_race_course_id not in header.race_course_ids
            ):
                available = ", ".join(header.race_course_ids)
                raise ValueError(
                    f"Unknown race course {initial_race_course_id!r}; available: {available}"
                )
        renderer = RendererSettings(
            raster=settings.renderer.raster,
            bev=settings.renderer.bev,
        )
        game = settings.game.taxi.game_config(
            default_high_scores_path=default_high_scores_path()
        )
        pipeline_config = settings.model.pipeline
        model_preset_name = pipeline_config.name
        configured_pipeline = _configure_live_edit_pipeline(
            pipeline_config,
            settings.live_edit,
        )
        native_dit_disabled_for_live_edit = configured_pipeline is not pipeline_config
        pipeline_config = configured_pipeline
        live_edit = resolve_live_edit_assets(settings.live_edit)
        self._pipeline_config = pipeline_config
        self._config = ApplicationConfig(
            scene_request=SceneRequest(
                map_path=map_path.expanduser(),
                camera_name=DEFAULT_FRONT_CAMERA_LOGICAL_NAME,
                use_prompt_context=settings.live_edit.map_context.enabled,
                force_recompile=bool(args.force_map_recompile),
            ),
            renderer=renderer,
            game=game,
            device=settings.model.device,
            total_blocks=settings.runtime.total_blocks,
            model_preset_name=model_preset_name,
            pipeline_profiling=settings.diagnostics.profile_pipeline,
            prewarm_blocks=settings.runtime.prewarm_blocks,
            profile_input_latency=settings.diagnostics.profile_input_latency,
            input_trace_path=(
                _DEFAULT_INPUT_TRACE_PATH
                if settings.diagnostics.profile_input_latency
                and settings.diagnostics.input_trace_path is None
                else settings.diagnostics.input_trace_path
            ),
            show_fps=settings.presentation.show_fps,
            hud_enabled=settings.presentation.hud_enabled,
            show_control_hints=settings.presentation.show_control_hints,
            show_live_edit_buttons=settings.presentation.show_live_edit_buttons,
            live_edit_mapping_location=(
                settings.presentation.live_edit_mapping_location
            ),
            controls=controls,
            gamepad_button_style=settings.game.gamepad_button_style,
            control_documents=control_documents,
            settings_document=settings_document,
            initial_game_mode=initial_game_mode,
            initial_map_path=(
                None if initial_map_path is None else map_path.expanduser().resolve()
            ),
            initial_race_course_id=initial_race_course_id,
            game_mode=initial_game_mode or "taxi",
            race_course_id=initial_race_course_id,
            race_times_path=(
                default_race_times_path()
                if settings.game.race.times_path is None
                else settings.game.race.times_path.expanduser()
            ),
            live_edit=live_edit,
            native_dit_disabled_for_live_edit=native_dit_disabled_for_live_edit,
            visual_flare_enabled=settings.game.effects.visual_flare,
        )
        self._map_options = _discover_game_maps(map_path)

    def _apply_cli_settings(
        self,
        base: CrazyRobotaxiUserSettings,
        args: argparse.Namespace,
    ) -> tuple[CrazyRobotaxiUserSettings, dict[tuple[str, ...], object]]:
        """Apply explicit CLI values over the user-authored settings tree."""
        settings = base
        overrides: dict[tuple[str, ...], object] = {}

        def explicit(name: str, path: tuple[str, ...], value: object) -> bool:
            if not arg_was_explicit(args, name):
                return False
            overrides[path] = value
            return True

        game = settings.game
        taxi = game.taxi
        rules = taxi.rules
        if explicit(
            "game_time_s", ("game", "taxi", "rules", "global_time_s"), args.game_time_s
        ):
            rules = replace(rules, global_time_s=args.game_time_s)
        if explicit("game_seed", ("game", "taxi", "seed"), args.game_seed):
            taxi = replace(taxi, seed=args.game_seed)
        if explicit("seed", ("game", "taxi", "seed"), args.seed):
            taxi = replace(taxi, seed=args.seed)
        if explicit(
            "high_scores", ("game", "taxi", "high_scores_path"), args.high_scores
        ):
            taxi = replace(taxi, high_scores_path=args.high_scores)
        race = game.race
        if explicit("race_times", ("game", "race", "times_path"), args.race_times):
            race = replace(race, times_path=args.race_times)
        effects = game.effects
        if explicit(
            "visual_flare", ("game", "effects", "visual_flare"), args.visual_flare
        ):
            effects = replace(effects, visual_flare=bool(args.visual_flare))
        game = replace(
            game, taxi=replace(taxi, rules=rules), race=race, effects=effects
        )
        model = settings.model
        if explicit("device", ("model", "device"), args.device):
            model = replace(model, device=args.device)
        pipeline = model.pipeline
        if explicit(
            "compile",
            ("model", "pipeline", "diffusion_model", "transformer", "compile_network"),
            args.compile,
        ):
            pipeline = derive_config(
                pipeline,
                diffusion_model={
                    "transformer": {"compile_network": bool(args.compile)}
                },
            )
        model_seed = (
            args.model_seed if arg_was_explicit(args, "model_seed") else args.seed
        )
        model_seed_explicit = arg_was_explicit(args, "model_seed") or arg_was_explicit(
            args, "seed"
        )
        if model_seed_explicit:
            overrides[("model", "pipeline", "diffusion_model", "seed")] = model_seed
            pipeline = derive_config(pipeline, diffusion_model={"seed": model_seed})
        diagnostics = settings.diagnostics
        if explicit(
            "profile_pipeline",
            ("diagnostics", "profile_pipeline"),
            args.profile_pipeline,
        ):
            diagnostics = replace(
                diagnostics, profile_pipeline=bool(args.profile_pipeline)
            )
        pipeline = derive_config(
            pipeline,
            enable_sync_and_profile=bool(diagnostics.profile_pipeline),
        )
        trace_path = args.profile_input_latency
        if arg_was_explicit(args, "profile_input_latency"):
            overrides[("diagnostics", "profile_input_latency")] = trace_path is not None
            overrides[("diagnostics", "input_trace_path")] = trace_path
            diagnostics = replace(
                diagnostics,
                profile_input_latency=trace_path is not None,
                input_trace_path=(
                    None if trace_path is None else trace_path.expanduser().resolve()
                ),
            )
        presentation = settings.presentation
        if explicit("show_fps", ("presentation", "show_fps"), args.show_fps):
            presentation = replace(presentation, show_fps=bool(args.show_fps))
        runtime = settings.runtime
        for name, field_name in (
            ("total_blocks", "total_blocks"),
            ("prewarm_blocks", "prewarm_blocks"),
        ):
            value = getattr(args, name)
            if explicit(name, ("runtime", field_name), value):
                runtime = replace(runtime, **{field_name: value})
        args._live_edit_settings = settings.live_edit
        live_edit = live_edit_config_from_args(args)
        if any(
            destination.startswith("live_edit_")
            for destination in getattr(args, "_explicit_arg_dests", ())
        ):
            overrides[("live_edit",)] = live_edit
        renderer = settings.renderer
        raster = renderer.raster
        for name in ("width", "height"):
            value = getattr(args, name)
            if explicit(name, ("renderer", "raster", name), value):
                raster = replace(raster, **{name: value})
        settings = replace(
            settings,
            game=game,
            model=replace(model, pipeline=pipeline),
            renderer=replace(renderer, raster=raster),
            presentation=presentation,
            live_edit=live_edit,
            runtime=runtime,
            diagnostics=diagnostics,
        )
        return settings, overrides

    def create_session(self, session_desc: SessionDesc) -> ISession:
        """Create one session after validating its fixed model geometry."""
        config = self._config
        if config is None:
            raise RuntimeError("init() must run before create_session()")
        pipeline_config = self._pipeline_config
        if pipeline_config is None:
            raise RuntimeError("init() must select a pipeline before create_session()")
        if session_desc.output_layout is not VideoTensorLayout.tchw:
            raise ValueError("Crazy Robotaxi produces tchw output")
        if session_desc.frames_per_second_for_step != _VIDEO_FPS:
            raise ValueError("Crazy Robotaxi generates video at 30 frames per second")
        actual = session_desc.video_width, session_desc.video_height
        config = replace(
            config,
            renderer=_fit_bev_renderer_to_ui(
                config.renderer,
                video_width=actual[0],
                video_height=actual[1],
            ),
        )
        expected = config.renderer.raster.resolution_wh
        if actual != expected:
            raise ValueError(
                f"Session dimensions {actual} do not match renderer {expected}"
            )
        transformer = pipeline_config.diffusion_model.transformer
        scheduler = pipeline_config.diffusion_model.scheduler
        encoder = pipeline_config.encoder
        bev = config.renderer.bev
        bev_resolution = f"{bev.width}x{bev.height}" if bev.enabled else "disabled"
        _LOGGER.info(
            "Crazy Robotaxi model preset=%s resolution=%sx%s native_dit=%s "
            "native_backend=%s attention_backend=%s native_vae=%s "
            "native_vae_backend=%s skip_finalize=%s "
            "denoising_timesteps=%s bev=%s",
            config.model_preset_name,
            actual[0],
            actual[1],
            transformer.native_dit_acceleration,
            transformer.native_dit_backend,
            transformer.native_dit_attention_backend,
            encoder.native_vae_acceleration,
            encoder.native_vae_backend,
            transformer.skip_finalize_kv_cache,
            list(scheduler.denoising_timesteps),
            bev_resolution,
        )
        return CrazyRobotaxiSession(
            pipeline_factory=partial(
                self._pipeline_factory,
                pipeline_config,
                config.device,
            ),
            scene_factory=self._scene_factory,
            map_options=self._map_options,
            config=config,
            session_desc=replace(
                session_desc,
                presentation_mode=PresentationMode.CONTINUOUS,
                metadata={
                    **session_desc.metadata,
                    **(
                        {
                            _TRACE_METADATA_KEY: True,
                            _TRACE_PATH_METADATA_KEY: str(config.input_trace_path),
                        }
                        if config.input_trace_path is not None
                        else {}
                    ),
                },
            ),
        )

    def close(self) -> None:
        """Release application configuration state."""
        self._config = None
        self._map_options = ()


def _build_pipeline(config: Any, device: str) -> Any:
    return config.setup().to(device).eval()


def _configure_live_edit_pipeline(config: Any, live_edit: LiveEditConfig) -> Any:
    """Disable native DiT when live-edit hooks need the Python transformer."""
    transformer = config.diffusion_model.transformer
    native_mode = getattr(transformer, "native_dit_acceleration", "disabled")
    if not live_edit.requires_python_dit or native_mode in {"disabled", None, False}:
        return config
    _LOGGER.warning("Disabling native DiT acceleration for live-edit features")
    return derive_config(
        config,
        diffusion_model={"transformer": {"native_dit_acceleration": "disabled"}},
    )


def _discover_game_maps(selected_path: Path) -> tuple[GameMapOption, ...]:
    """Read menu metadata for bundled maps and maps beside the CLI selection."""
    selected = selected_path.expanduser().resolve()
    paths = {selected}
    for directory in (_DEFAULT_MAP.parent, selected.parent):
        if directory.is_dir():
            paths.update(
                path.resolve() for path in directory.glob(f"*{GAME_MAP_SUFFIX}")
            )

    options: list[GameMapOption] = []
    for path in paths:
        header = load_game_map_header(path)
        options.append(
            GameMapOption(
                map_id=header.map_id,
                name=header.name,
                path=header.source_path,
                race_courses=tuple(
                    GameRaceCourseOption(
                        course_id=course.course_id,
                        spawn_id=course.spawn_id,
                        preview_image_path=course.spawn_image_path,
                    )
                    for course in header.race_courses
                ),
                preview_image_path=header.menu_thumbnail_path,
            )
        )
    return tuple(
        sorted(options, key=lambda item: (item.path != selected, item.name.casefold()))
    )


def _fit_bev_renderer_to_ui(
    renderer: RendererSettings,
    *,
    video_width: int,
    video_height: int,
) -> RendererSettings:
    """Avoid rasterizing a HUD-only BEV above its presented pixel extent."""
    bev = renderer.bev
    if not bev.enabled:
        return renderer
    maximum_width, maximum_height = bev_display_extent(video_width, video_height)
    scale = min(
        1.0,
        maximum_width / bev.width,
        maximum_height / bev.height,
    )
    if scale >= 1.0:
        return renderer
    fitted = replace(
        bev,
        width=max(1, round(bev.width * scale)),
        height=max(1, round(bev.height * scale)),
    )
    return replace(renderer, bev=fitted)


def _parser(
    defaults: CrazyRobotaxiApplicationDefaults,
) -> argparse.ArgumentParser:
    parser = ExplicitArgTrackingArgumentParser(
        prog=f"flashdreams-run-v2 {defaults.slug} --",
        description="Drive Crazy Robotaxi on an authored semantic map.",
    )
    parser.add_argument("--config", type=Path)
    parser.add_argument("--controls-dir", type=Path)
    parser.add_argument("--map", type=Path, default=_DEFAULT_MAP)
    parser.add_argument("--width", type=int, default=defaults.width)
    parser.add_argument("--height", type=int, default=defaults.height)
    parser.add_argument("--force-map-recompile", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--total-blocks", type=int)
    parser.add_argument("--game-time-s", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--game-seed", type=int)
    parser.add_argument("--model-seed", type=int)
    parser.add_argument("--high-scores", type=Path)
    parser.add_argument("--game-mode", choices=("taxi", "race"), default="taxi")
    parser.add_argument(
        "--visual-flare",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--race-course")
    parser.add_argument("--race-times", type=Path)
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction)
    parser.add_argument(
        "--profile-pipeline",
        action="store_true",
        help="synchronize each chunk and emit diagnostic GPU stage timings",
    )
    parser.add_argument(
        "--prewarm-blocks",
        type=int,
        default=_DEFAULT_PREWARM_BLOCKS,
        help=(
            "generate hidden neutral blocks before presentation to compile and "
            "autotune AR shapes (default: 8; 0 disables)"
        ),
    )
    parser.add_argument(
        "--profile-input-latency",
        nargs="?",
        type=Path,
        const=_DEFAULT_INPUT_TRACE_PATH,
        metavar="TRACE_PATH",
        help=(
            "show input diagnostics and write the chunk lifecycle trace "
            f"(default path: {_DEFAULT_INPUT_TRACE_PATH})"
        ),
    )
    parser.add_argument(
        "--show-fps",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="show the measured generated-video frame rate in the HUD",
    )
    add_live_edit_args(parser)
    return parser
