# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Process-lifetime Crazy Robotaxi application composition."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from omnidreams.config import (
    RUNNER_SV_2STEPS_CHUNK2_LOC6_LIGHTVAE_LIGHTTAE,
    RUNNER_SV_2STEPS_CHUNK2_LOC6_LIGHTVAE_LIGHTTAE_NATIVE_PERF,
    RUNNER_SV_2STEPS_CHUNK2_LOC6_LIGHTVAE_LIGHTTAE_PERF,
)
from omnidreams_game_engine.config import DriverInputConfig
from omnidreams_game_engine.renderer_settings import (
    RendererSettings,
    load_renderer_settings,
)
from omnidreams_game_engine.scene import SceneRequest, load_scene
from omnidreams_game_engine.types import SceneDefinition

from crazy_robotaxi.config import load_game_settings
from crazy_robotaxi.high_scores import default_high_scores_path
from crazy_robotaxi.rules import TaxiGameConfig
from crazy_robotaxi.session import CrazyRobotaxiSession
from flashdreams.api_v2.application import IApplication
from flashdreams.api_v2.session import ISession
from flashdreams.infra.config import derive_config
from flashdreams.runtime_v2.session_desc import SessionDesc
from flashdreams.runtime_v2.video_tensor import VideoTensorLayout

_ROOT = Path(__file__).resolve().parent
_DEFAULT_MAP = _ROOT / "maps" / "boulevard_district.robotaxi.yaml"
_DEFAULT_GAME = _ROOT / "configs" / "default_game.yaml"
_DEFAULT_RENDERER = _ROOT / "configs" / "default_renderer.yaml"
_MODEL_PRESETS = {
    "standard": RUNNER_SV_2STEPS_CHUNK2_LOC6_LIGHTVAE_LIGHTTAE.pipeline,
    "perf": RUNNER_SV_2STEPS_CHUNK2_LOC6_LIGHTVAE_LIGHTTAE_PERF.pipeline,
    "native-perf": RUNNER_SV_2STEPS_CHUNK2_LOC6_LIGHTVAE_LIGHTTAE_NATIVE_PERF.pipeline,
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
        self._defaults = load_renderer_settings(_DEFAULT_RENDERER)
        self._pipeline_factory = pipeline_factory or _build_pipeline
        self._scene_factory = scene_factory or load_scene
        self._pipeline_config: Any = _MODEL_PRESETS["standard"]
        self._pipeline: Any | None = None
        self._config: ApplicationConfig | None = None

    def session_desc(self) -> SessionDesc:
        """Declare the trained single-view output contract without loading."""
        raster = self._defaults.raster
        return SessionDesc(
            output_layout=VideoTensorLayout.tchw,
            frames_per_second_for_ui=60,
            frames_per_second_for_step=30,
            video_width=raster.width,
            video_height=raster.height,
        )

    def init(self, commandline_args: Sequence[str]) -> None:
        """Parse application options without starting another runtime."""
        args = _parser().parse_args(list(commandline_args))
        if args.total_blocks is not None and args.total_blocks <= 0:
            raise ValueError("--total-blocks must be positive")
        if args.game_time_s is not None and args.game_time_s <= 0.0:
            raise ValueError("--game-time-s must be positive")
        renderer = load_renderer_settings(args.renderer_config)
        settings = load_game_settings(args.game_config)
        game = settings.game
        game = replace(
            game,
            seed=args.seed,
            global_time_s=(
                game.global_time_s if args.game_time_s is None else args.game_time_s
            ),
            high_scores_path=(
                default_high_scores_path()
                if args.high_scores is None
                else args.high_scores.expanduser()
            ),
        )
        pipeline_config = _MODEL_PRESETS[args.model_preset]
        if args.compile is not None:
            pipeline_config = derive_config(
                pipeline_config,
                diffusion_model={
                    "transformer": {"compile_network": bool(args.compile)}
                },
            )
        if args.seed is not None:
            pipeline_config = derive_config(
                pipeline_config,
                diffusion_model={"seed": int(args.seed)},
            )
        self._pipeline_config = pipeline_config
        self._config = ApplicationConfig(
            scene_request=SceneRequest(
                map_path=args.map.expanduser(),
                camera_name=args.camera,
                variant=args.variant,
                prompt=args.prompt,
                force_recompile=args.force_map_recompile,
            ),
            renderer=renderer,
            game=game,
            driver_input=settings.driver_input,
            device=args.device,
            total_blocks=args.total_blocks,
        )

    def create_session(self, session_desc: SessionDesc) -> ISession:
        """Create one session after validating its fixed model geometry."""
        config = self._config
        if config is None:
            raise RuntimeError("init() must run before create_session()")
        if session_desc.output_layout is not VideoTensorLayout.tchw:
            raise ValueError("Crazy Robotaxi produces tchw output")
        if session_desc.frames_per_second_for_step != 30:
            raise ValueError("Crazy Robotaxi produces video at 30 frames per second")
        expected = config.renderer.raster.resolution_wh
        actual = session_desc.video_width, session_desc.video_height
        if actual != expected:
            raise ValueError(
                f"Session dimensions {actual} do not match renderer {expected}"
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


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="flashdreams-run-v2 crazy-robotaxi --",
        description="Drive Crazy Robotaxi on an authored semantic map.",
    )
    parser.add_argument("--map", type=Path, default=_DEFAULT_MAP)
    parser.add_argument("--game-config", type=Path, default=_DEFAULT_GAME)
    parser.add_argument("--renderer-config", type=Path, default=_DEFAULT_RENDERER)
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
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction)
    return parser
