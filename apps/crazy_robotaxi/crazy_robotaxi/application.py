# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Process-lifetime Crazy Robotaxi application composition."""

from __future__ import annotations

import argparse
import logging
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path
from typing import Any, Literal

from omnidreams_game_engine.cli_args import (
    ExplicitArgTrackingArgumentParser,
    arg_was_explicit,
)
from omnidreams_game_engine.config import BevConfig, RasterConfig
from omnidreams_game_engine.engine_settings import (
    EngineSettings,
    MapLaunchSettings,
    RenderingSettings,
    WorldModelLaunchSettings,
)
from omnidreams_game_engine.game_map import GAME_MAP_SUFFIX, load_game_map_header
from omnidreams_game_engine.renderer_settings import RendererSettings
from omnidreams_game_engine.scene import SceneRequest, load_scene
from omnidreams_game_engine.types import SceneDefinition

from crazy_robotaxi.config import CrazyRobotaxiSettings
from crazy_robotaxi.game_selection import GameMapOption, GameMode
from crazy_robotaxi.high_scores import default_high_scores_path, default_race_times_path
from crazy_robotaxi.live_edit.config import (
    LiveEditConfig,
    add_live_edit_args,
    live_edit_config_from_args,
    resolve_live_edit_assets,
)
from crazy_robotaxi.rules import TaxiGameConfig
from crazy_robotaxi.session import CrazyRobotaxiSession
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

    cli_game_mode: GameMode | None = None
    """Game mode supplied explicitly on the command line, if any."""

    cli_map_path: Path | None = None
    """Map supplied explicitly on the command line, if any."""

    cli_race_course_id: str | None = None
    """Race course supplied explicitly on the command line, if any."""

    game_mode: Literal["taxi", "race"] = "taxi"
    """Rules mode selected for every session created by the application."""

    race_course_id: str | None = None
    """Requested race course, or ``None`` for the map's first course."""

    race_times_path: Path | None = None
    """Persistent map- and course-scoped race leaderboard."""

    live_edit: LiveEditConfig = LiveEditConfig()
    """Flag-gated style, weather, pickup, nitro, and obstacle abilities."""

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
        pipeline_config = self._pipeline_config
        if pipeline_config is None:
            raise RuntimeError("A world-model integration must provide pipeline_config")
        args = _parser(self._application_defaults).parse_args(list(commandline_args))
        input_trace_path = args.profile_input_latency
        args.profile_input_latency = input_trace_path is not None
        engine_settings = self._resolve_engine_settings(args)
        game_settings = self._resolve_game_settings(args)
        if (
            engine_settings.runtime.total_blocks is not None
            and engine_settings.runtime.total_blocks <= 0
        ):
            raise ValueError("--total-blocks must be positive")
        if args.game_time_s is not None and args.game_time_s <= 0.0:
            raise ValueError("--game-time-s must be positive")
        if engine_settings.runtime.prewarm_blocks < 0:
            raise ValueError("--prewarm-blocks must be non-negative")
        if game_settings.mode != "race" and (
            arg_was_explicit(args, "race_course")
            or arg_was_explicit(args, "race_times")
        ):
            raise ValueError("--race-course and --race-times require --game-mode race")
        map_path = engine_settings.map.path
        if map_path is None:
            raise ValueError("A map path is required (set engine.map.path or --map)")
        if game_settings.mode == "race":
            header = load_game_map_header(map_path.expanduser())
            if not header.race_course_ids:
                raise ValueError(f"Map {header.map_id!r} defines no race courses")
            if (
                game_settings.race.course is not None
                and game_settings.race.course not in header.race_course_ids
            ):
                available = ", ".join(header.race_course_ids)
                raise ValueError(
                    f"Unknown race course {game_settings.race.course!r}; available: {available}"
                )
        renderer = RendererSettings(
            raster=engine_settings.rendering.raster,
            bev=engine_settings.rendering.bev,
        )
        game = game_settings.game
        game = replace(
            game,
            global_time_s=(
                game.global_time_s if args.game_time_s is None else args.game_time_s
            ),
            high_scores_path=(
                default_high_scores_path()
                if game_settings.taxi.high_scores_path is None
                else game_settings.taxi.high_scores_path.expanduser()
            ),
        )
        model_preset_name = pipeline_config.name
        if engine_settings.world_model.compile is not None:
            pipeline_config = derive_config(
                pipeline_config,
                diffusion_model={
                    "transformer": {
                        "compile_network": bool(engine_settings.world_model.compile)
                    }
                },
            )
        if game_settings.taxi.seed is not None:
            pipeline_config = derive_config(
                pipeline_config,
                diffusion_model={"seed": int(game_settings.taxi.seed)},
            )
        pipeline_config = derive_config(
            pipeline_config,
            enable_sync_and_profile=bool(engine_settings.world_model.profile_pipeline),
        )
        game_settings = replace(
            game_settings, live_edit=resolve_live_edit_assets(game_settings.live_edit)
        )
        self._pipeline_config = pipeline_config
        self._config = ApplicationConfig(
            scene_request=SceneRequest(
                map_path=map_path.expanduser(),
                camera_name=engine_settings.map.camera,
                variant=engine_settings.map.variant,
                prompt=engine_settings.map.prompt,
                force_recompile=engine_settings.map.force_recompile,
            ),
            renderer=renderer,
            game=game,
            device=engine_settings.world_model.device,
            total_blocks=engine_settings.runtime.total_blocks,
            model_preset_name=model_preset_name,
            pipeline_profiling=bool(engine_settings.world_model.profile_pipeline),
            prewarm_blocks=engine_settings.runtime.prewarm_blocks,
            profile_input_latency=engine_settings.runtime.profile_input_latency,
            input_trace_path=(
                None
                if input_trace_path is None
                else input_trace_path.expanduser().resolve()
            ),
            show_fps=engine_settings.presentation.show_fps,
            cli_game_mode=(
                game_settings.mode if arg_was_explicit(args, "game_mode") else None
            ),
            cli_map_path=(
                map_path.expanduser().resolve()
                if arg_was_explicit(args, "map")
                else None
            ),
            cli_race_course_id=(
                game_settings.race.course
                if arg_was_explicit(args, "race_course")
                else None
            ),
            game_mode=game_settings.mode,
            race_course_id=game_settings.race.course,
            race_times_path=(
                default_race_times_path()
                if game_settings.race.times_path is None
                else game_settings.race.times_path.expanduser()
            ),
            live_edit=game_settings.live_edit,
            visual_flare_enabled=game_settings.effects.visual_flare_enabled,
        )
        self._map_options = _discover_game_maps(
            map_path,
            requested_variant=engine_settings.map.variant,
        )

    def _resolve_engine_settings(self, args: argparse.Namespace) -> EngineSettings:
        settings = EngineSettings(
            map=MapLaunchSettings(path=_DEFAULT_MAP),
            world_model=WorldModelLaunchSettings(),
            rendering=RenderingSettings(
                raster=self._defaults.raster,
                bev=self._defaults.bev,
            ),
        )
        return replace(
            settings,
            map=replace(
                settings.map,
                path=args.map,
                camera=args.camera,
                variant=args.variant,
                prompt=args.prompt,
                force_recompile=args.force_map_recompile,
            ),
            rendering=replace(
                settings.rendering,
                raster=replace(
                    settings.rendering.raster, width=args.width, height=args.height
                ),
            ),
            world_model=replace(
                settings.world_model,
                device=args.device,
                compile=args.compile,
                profile_pipeline=args.profile_pipeline,
            ),
            presentation=replace(
                settings.presentation,
                show_fps=bool(args.show_fps),
            ),
            runtime=replace(
                settings.runtime,
                total_blocks=args.total_blocks,
                prewarm_blocks=args.prewarm_blocks,
                profile_input_latency=args.profile_input_latency,
            ),
        )

    def _resolve_game_settings(self, args: argparse.Namespace) -> CrazyRobotaxiSettings:
        settings = CrazyRobotaxiSettings()
        game = settings.game
        taxi = settings.taxi
        race = settings.race
        if arg_was_explicit(args, "game_mode"):
            settings = replace(settings, mode=args.game_mode)
        if arg_was_explicit(args, "visual_flare"):
            settings = replace(
                settings,
                effects=replace(
                    settings.effects,
                    visual_flare_enabled=bool(args.visual_flare),
                ),
            )
        if arg_was_explicit(args, "seed"):
            taxi = replace(taxi, seed=args.seed)
            game = replace(game, seed=args.seed)
        if arg_was_explicit(args, "high_scores"):
            taxi = replace(taxi, high_scores_path=args.high_scores)
        if arg_was_explicit(args, "race_course"):
            race = replace(race, course=args.race_course)
        if arg_was_explicit(args, "race_times"):
            race = replace(race, times_path=args.race_times)
        settings = replace(settings, game=game, taxi=taxi, race=race)
        args._live_edit_settings = settings.live_edit
        return replace(settings, live_edit=live_edit_config_from_args(args))

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


def _discover_game_maps(
    selected_path: Path,
    *,
    requested_variant: str,
) -> tuple[GameMapOption, ...]:
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
        variants = tuple(item.name for item in header.variants)
        preferred = requested_variant if path == selected else "default"
        variant = (
            preferred
            if preferred in variants
            else ("default" if "default" in variants else variants[0])
        )
        options.append(
            GameMapOption(
                map_id=header.map_id,
                name=header.name,
                path=header.source_path,
                variant=variant,
                race_course_ids=header.race_course_ids,
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
    parser.add_argument("--map", type=Path, default=_DEFAULT_MAP)
    parser.add_argument("--width", type=int, default=defaults.width)
    parser.add_argument("--height", type=int, default=defaults.height)
    parser.add_argument("--camera", default="camera_front_wide_120fov")
    parser.add_argument("--variant", default="default")
    parser.add_argument("--prompt")
    parser.add_argument("--force-map-recompile", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--total-blocks", type=int)
    parser.add_argument("--game-time-s", type=float)
    parser.add_argument("--seed", type=int)
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
