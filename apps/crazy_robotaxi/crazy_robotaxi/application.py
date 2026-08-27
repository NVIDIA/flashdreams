# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Process-lifetime Crazy Robotaxi application composition."""

from __future__ import annotations

import argparse
import copy
import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal

from omnidreams.config import (
    RUNNER_SV_2STEPS_CHUNK2_LOC6_LIGHTVAE_LIGHTTAE,
    RUNNER_SV_2STEPS_CHUNK2_LOC6_LIGHTVAE_LIGHTTAE_NATIVE_PERF,
    RUNNER_SV_2STEPS_CHUNK2_LOC6_LIGHTVAE_LIGHTTAE_PERF,
)
from omnidreams_game_engine.cli_args import (
    ExplicitArgTrackingArgumentParser,
    arg_was_explicit,
)
from omnidreams_game_engine.config import BevConfig, DriverInputConfig, RasterConfig
from omnidreams_game_engine.engine_settings import (
    EngineSettings,
    MapLaunchSettings,
    RenderingSettings,
    WorldModelLaunchSettings,
    load_engine_settings,
)
from omnidreams_game_engine.game_map import load_game_map_header
from omnidreams_game_engine.renderer_settings import (
    RendererSettings,
    load_renderer_settings,
)
from omnidreams_game_engine.scene import SceneRequest, load_scene
from omnidreams_game_engine.types import SceneDefinition

from crazy_robotaxi.config import CrazyRobotaxiSettings, load_game_settings
from crazy_robotaxi.high_scores import default_high_scores_path, default_race_times_path
from crazy_robotaxi.live_edit.config import (
    LiveEditConfig,
    add_live_edit_args,
    live_edit_config_from_args,
)
from crazy_robotaxi.rules import TaxiGameConfig
from crazy_robotaxi.session import CrazyRobotaxiSession
from crazy_robotaxi.ui import bev_display_extent
from flashdreams.api_v2.application import IApplication
from flashdreams.api_v2.session import ISession
from flashdreams.infra.config import derive_config
from flashdreams.runtime_v2.session_desc import SessionDesc
from flashdreams.runtime_v2.video_tensor import VideoTensorLayout

_ROOT = Path(__file__).resolve().parent
_DEFAULT_MAP = _ROOT / "maps" / "boulevard_district.robotaxi.yaml"
_VIDEO_FPS = 30
"""Generation and UI cadence; each UI tick advances one generated frame."""

_LOGGER = logging.getLogger(__name__)

_DEFAULT_PREWARM_BLOCKS = 4
"""Blocks covering chunk2 cache filling and the first steady-state AR shape."""

_ORIGINAL_PERF_PIPELINE = derive_config(
    RUNNER_SV_2STEPS_CHUNK2_LOC6_LIGHTVAE_LIGHTTAE_PERF.pipeline,
    name="crazy-robotaxi-original-perf",
    diffusion_model={
        "seed": None,
        "transformer": {
            "compile_network": True,
            "skip_finalize_kv_cache": True,
            "native_dit_acceleration": "required",
            "native_dit_backend": "fp8_kvcache_cudnn",
            "native_dit_attention_backend": "cudnn",
        },
        "scheduler": {
            "denoising_timesteps": [1000, 100],
            "num_inference_steps": 2,
        },
    },
)

_FAST_PERF_PIPELINE = derive_config(
    RUNNER_SV_2STEPS_CHUNK2_LOC6_LIGHTVAE_LIGHTTAE_NATIVE_PERF.pipeline,
    name="crazy-robotaxi-fast-perf",
    diffusion_model=copy.deepcopy(_ORIGINAL_PERF_PIPELINE.diffusion_model),
)
"""Candidate maximum-performance path: native FP8 LightVAE and native FP8 DiT."""


@dataclass(frozen=True, slots=True)
class _ModelPreset:
    """App-owned pipeline selection and renderer geometry policy."""

    pipeline: Any
    renderer_follows_session: bool = False


_MODEL_PRESETS = {
    "standard": _ModelPreset(RUNNER_SV_2STEPS_CHUNK2_LOC6_LIGHTVAE_LIGHTTAE.pipeline),
    "perf": _ModelPreset(RUNNER_SV_2STEPS_CHUNK2_LOC6_LIGHTVAE_LIGHTTAE_PERF.pipeline),
    "native-perf": _ModelPreset(
        RUNNER_SV_2STEPS_CHUNK2_LOC6_LIGHTVAE_LIGHTTAE_NATIVE_PERF.pipeline
    ),
    "original-perf": _ModelPreset(
        _ORIGINAL_PERF_PIPELINE,
        renderer_follows_session=True,
    ),
    "fast-perf": _ModelPreset(
        _FAST_PERF_PIPELINE,
        renderer_follows_session=True,
    ),
}


@dataclass(frozen=True, slots=True)
class ApplicationConfig:
    """Validated options shared by sessions created by one application."""

    scene_request: SceneRequest
    renderer: RendererSettings
    game: TaxiGameConfig
    driver_input: DriverInputConfig
    device: str
    total_blocks: int | None
    model_preset_name: str
    renderer_follows_session: bool
    pipeline_profiling: bool
    prewarm_blocks: int
    """Hidden neutral blocks generated before the first presented game frame."""

    profile_input_latency: bool
    """Whether the UI displays and logs input-to-model-frame diagnostics."""

    show_fps: bool
    """Whether the HUD displays the measured generated-video frame rate."""

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


class CrazyRobotaxiApplication(IApplication):
    """Load one model and create isolated V2 game sessions."""

    def __init__(
        self,
        *,
        pipeline_factory: PipelineFactory | None = None,
        scene_factory: SceneFactory | None = None,
    ) -> None:
        self._defaults = RendererSettings(raster=RasterConfig(), bev=BevConfig())
        self._pipeline_factory = pipeline_factory or _build_pipeline
        self._scene_factory = scene_factory or load_scene
        self._pipeline_config: Any = _MODEL_PRESETS["standard"].pipeline
        self._pipeline: Any | None = None
        self._config: ApplicationConfig | None = None

    def session_desc(self) -> SessionDesc:
        """Declare the trained single-view output contract without loading."""
        raster = (
            self._defaults.raster
            if self._config is None
            else self._config.renderer.raster
        )
        return SessionDesc(
            output_layout=VideoTensorLayout.tchw,
            frames_per_second_for_ui=_VIDEO_FPS,
            frames_per_second_for_step=_VIDEO_FPS,
            video_width=raster.width,
            video_height=raster.height,
        )

    def init(self, commandline_args: Sequence[str]) -> None:
        """Parse application options without starting another runtime."""
        args = _parser().parse_args(list(commandline_args))
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
        model_preset_name = engine_settings.world_model.model_preset
        if model_preset_name not in _MODEL_PRESETS:
            available = ", ".join(_MODEL_PRESETS)
            raise ValueError(
                f"Unknown model preset {model_preset_name!r}; available: {available}"
            )
        model_preset = _MODEL_PRESETS[model_preset_name]
        pipeline_config = model_preset.pipeline
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
            driver_input=game_settings.driver_input,
            device=engine_settings.world_model.device,
            total_blocks=engine_settings.runtime.total_blocks,
            model_preset_name=model_preset_name,
            renderer_follows_session=model_preset.renderer_follows_session,
            pipeline_profiling=bool(engine_settings.world_model.profile_pipeline),
            prewarm_blocks=engine_settings.runtime.prewarm_blocks,
            profile_input_latency=engine_settings.runtime.profile_input_latency,
            show_fps=engine_settings.presentation.show_fps,
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

    def _resolve_engine_settings(self, args: argparse.Namespace) -> EngineSettings:
        settings = EngineSettings(
            map=MapLaunchSettings(path=_DEFAULT_MAP),
            world_model=WorldModelLaunchSettings(),
            rendering=RenderingSettings(
                raster=self._defaults.raster,
                bev=self._defaults.bev,
            ),
        )
        if args.engine_config is not None:
            settings = load_engine_settings(args.engine_config, base=settings)
        if args.renderer_config is not None:
            legacy = load_renderer_settings(args.renderer_config)
            settings = replace(
                settings,
                rendering=RenderingSettings(raster=legacy.raster, bev=legacy.bev),
            )
        map_settings = settings.map
        for destination, field_name in (
            ("map", "path"),
            ("camera", "camera"),
            ("variant", "variant"),
            ("prompt", "prompt"),
            ("force_map_recompile", "force_recompile"),
        ):
            if arg_was_explicit(args, destination):
                map_settings = replace(
                    map_settings, **{field_name: getattr(args, destination)}
                )
        world_model = settings.world_model
        for destination in ("model_preset", "device", "compile", "profile_pipeline"):
            if arg_was_explicit(args, destination):
                world_model = replace(
                    world_model, **{destination: getattr(args, destination)}
                )
        runtime = settings.runtime
        for destination in ("total_blocks", "prewarm_blocks", "profile_input_latency"):
            if arg_was_explicit(args, destination):
                runtime = replace(runtime, **{destination: getattr(args, destination)})
        if runtime.profile_world_model:
            world_model = replace(world_model, profile_pipeline=True)
        presentation = settings.presentation
        if arg_was_explicit(args, "show_fps"):
            presentation = replace(presentation, show_fps=bool(args.show_fps))
        return replace(
            settings,
            map=map_settings,
            world_model=world_model,
            presentation=presentation,
            runtime=runtime,
        )

    def _resolve_game_settings(self, args: argparse.Namespace) -> CrazyRobotaxiSettings:
        settings = CrazyRobotaxiSettings(
            driver_input=DriverInputConfig(
                steering_scale=1.0,
                steering_rate_per_s=3.5,
                steering_return_rate_per_s=5.0,
            )
        )
        if args.game_config is not None:
            settings = load_game_settings(args.game_config, base=settings)
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
        if session_desc.output_layout is not VideoTensorLayout.tchw:
            raise ValueError("Crazy Robotaxi produces tchw output")
        if (
            session_desc.frames_per_second_for_ui != _VIDEO_FPS
            or session_desc.frames_per_second_for_step != _VIDEO_FPS
        ):
            raise ValueError(
                "Crazy Robotaxi generates and presents video at 30 frames per second"
            )
        actual = session_desc.video_width, session_desc.video_height
        if config.renderer_follows_session:
            config = replace(
                config,
                renderer=replace(
                    config.renderer,
                    raster=replace(
                        config.renderer.raster,
                        width=actual[0],
                        height=actual[1],
                    ),
                ),
            )
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
        transformer = self._pipeline_config.diffusion_model.transformer
        scheduler = self._pipeline_config.diffusion_model.scheduler
        encoder = self._pipeline_config.encoder
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
        if self._pipeline is None:
            self._pipeline = self._pipeline_factory(
                self._pipeline_config,
                config.device,
            )
        scene = self._scene_factory(config.scene_request, config.renderer.raster)
        return CrazyRobotaxiSession(
            pipeline=self._pipeline,
            scene=scene,
            config=config,
            session_desc=session_desc,
        )

    def close(self) -> None:
        """Release process-lifetime model resources."""
        pipeline = self._pipeline
        self._pipeline = None
        self._config = None
        close = getattr(pipeline, "close", None)
        if callable(close):
            close()


def create_app() -> IApplication:
    """Return a fresh V2 application for entry-point discovery."""
    return CrazyRobotaxiApplication()


def _build_pipeline(config: Any, device: str) -> Any:
    return config.setup().to(device).eval()


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


def _parser() -> argparse.ArgumentParser:
    parser = ExplicitArgTrackingArgumentParser(
        prog="flashdreams-run-v2 crazy-robotaxi --",
        description="Drive Crazy Robotaxi on an authored semantic map.",
    )
    parser.add_argument("--engine-config", type=Path)
    parser.add_argument("--map", type=Path, default=_DEFAULT_MAP)
    parser.add_argument("--game-config", type=Path)
    parser.add_argument("--renderer-config", type=Path)
    parser.add_argument("--camera", default="camera_front_wide_120fov")
    parser.add_argument("--variant", default="default")
    parser.add_argument("--prompt")
    parser.add_argument("--force-map-recompile", action="store_true")
    parser.add_argument(
        "--model-preset", choices=tuple(_MODEL_PRESETS), default="standard"
    )
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
            "autotune AR shapes (default: 4; 0 disables)"
        ),
    )
    parser.add_argument(
        "--profile-input-latency",
        action="store_true",
        help="show and log UI-input-to-model-frame latency diagnostics",
    )
    parser.add_argument(
        "--show-fps",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="show the measured generated-video frame rate in the HUD",
    )
    add_live_edit_args(parser)
    return parser
