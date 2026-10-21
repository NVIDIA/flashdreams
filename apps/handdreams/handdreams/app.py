# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""First-class HandDreams application on the FlashDreams V2 API."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from flashdreams_controller_input.isaac_teleop_source import DEFAULT_ISAAC_TELEOP_PYTHON

from flashdreams.api_v2.application import IApplication
from flashdreams.api_v2.session import ISession
from flashdreams.infra.postprocess import (
    VideoPostprocessChainConfig,
    VideoSpec,
    create_video_postprocess_stream,
)
from flashdreams.plugins.registry import discover_postprocess_presets
from flashdreams.runtime_v2.session_desc import (
    BackpressureMode,
    PresentationMode,
    SessionDesc,
)
from flashdreams.runtime_v2.video_tensor import VideoTensorLayout

from .config import (
    HandDreamsApplicationDefaults,
    HandDreamsLaunchConfig,
    HanddreamsScenario,
)
from .scenes import ScenePreview
from .session import HandDreamsSession
from .teleop import IsaacTeleopControllerPipeline, TeleopControllerPipeline
from .ui import scene_strip_height

_DEFAULT_UI_FPS = 36
_DEFAULT_STEP_FPS = 36
_MAX_SCENES = 5
"""Maximum catalog size; prepared scene caches stay resident until shutdown."""

ControllerPipelineFactory = Callable[..., TeleopControllerPipeline]


class HandDreamsV2Application(IApplication):
    """Own the loaded inference pipeline and controller calibration service."""

    def __init__(
        self,
        *,
        defaults: HandDreamsApplicationDefaults,
        pipeline_factory: ControllerPipelineFactory = IsaacTeleopControllerPipeline,
    ) -> None:
        self.defaults = defaults
        self._pipeline_factory = pipeline_factory
        self._config: HandDreamsLaunchConfig | None = None
        self._inference_pipeline: Any | None = None
        self._pipeline: TeleopControllerPipeline | None = None
        self._scene_catalog: tuple[str, ...] = ()
        self._scene_previews: tuple[ScenePreview, ...] = ()
        self._prepared_scenes: dict[str | None, Any] = {}
        """Keep each scene's conditioning identical across warmed-cache reuse."""
        self._closed = False

    @property
    def launch_config(self) -> HandDreamsLaunchConfig:
        if self._config is None:
            raise RuntimeError(
                "HandDreamsV2Application.init() must run before create_session()."
            )
        return self._config

    @property
    def scene_catalog(self) -> tuple[str, ...]:
        """Names accepted by SessionDesc.metadata sample selection."""
        return self._scene_catalog

    def default_client_window_mode(self) -> str:
        """Use the standard WebRTC display by default."""
        return "webrtc"

    def resolve_client_window_mode(self, mode: str) -> str:
        """Attach shared Isaac controller input to interactive displays."""
        if mode in {"webrtc", "native-window"}:
            return f"quest-remote-{mode}"
        return mode

    def init(self, commandline_args: Sequence[str]) -> None:
        """Parse arguments and prepare the shared controller and model services."""
        if self._closed:
            raise RuntimeError("cannot initialize a closed HandDreams application")
        if self._config is not None:
            raise RuntimeError("HandDreams application is already initialized")
        parser = argparse.ArgumentParser(
            prog="flashdreams-run-v2 handdreams --",
            description="Run hand video from Isaac Teleop or Quest controllers.",
        )
        parser.add_argument(
            "--sample",
            help="Initial scene name; defaults to the first discovered scene.",
        )
        scene_source = parser.add_mutually_exclusive_group()
        scene_source.add_argument(
            "--scene-path",
            type=Path,
            help="Scene directory containing first_frame.jpg and scene.json.",
        )
        scene_source.add_argument(
            "--sample-dir",
            type=Path,
            help="Root directory containing scene directories.",
        )
        parser.add_argument(
            "--prewarm-all-scenes",
            action="store_true",
            help="Compatibility option; every available scene is prepared during startup.",
        )
        parser.add_argument("--total-steps", type=int, default=2_147_483_647)
        parser.add_argument("--duration-seconds", type=float)
        parser.add_argument(
            "--control-fps",
            type=int,
            help="Controller sampling and model FPS; defaults to each scene's FPS.",
        )
        parser.add_argument(
            "--postprocess-preset",
            choices=("", *sorted(discover_postprocess_presets())),
            default="",
            help="Optional generated-video post-processing preset.",
        )
        parser.add_argument(
            "--postprocess-device",
            default="cuda:0",
            help="Device used by the selected postprocessor.",
        )
        parser.add_argument("--synchronize-metrics", action="store_true")
        self.defaults.add_model_arguments(parser)
        _add_teleop_arguments(parser)
        args = parser.parse_args(list(commandline_args))

        scenario = HanddreamsScenario(
            sample=args.sample,
            scene_path=args.scene_path,
            sample_dir=args.sample_dir,
        )
        self._scene_catalog = self.defaults.discover_scene_catalog(scenario)
        if not self._scene_catalog:
            raise ValueError("HandDreams requires at least one scene")
        # ponytail: five resident warm scenes; larger catalogs need bounded caching.
        if len(self._scene_catalog) > _MAX_SCENES:
            raise ValueError(
                f"HandDreams discovered {len(self._scene_catalog)} scenes; "
                f"the prepared-scene limit is {_MAX_SCENES}. "
                "Use --scene-path for one scene or --sample-dir with a smaller catalog."
            )
        if scenario.sample is None:
            scenario = replace(scenario, sample=self._scene_catalog[0])
        elif scenario.sample not in self._scene_catalog:
            raise ValueError(
                f"requested HandDreams sample is absent: {scenario.sample}"
            )
        self._scene_previews = self.defaults.scene_previews(
            scenario, self._scene_catalog
        )
        pipeline_config = self.defaults.resolve_pipeline_config(vars(args))
        self._config = HandDreamsLaunchConfig(
            scenario=scenario,
            pipeline_config=pipeline_config,
            postprocess=VideoPostprocessChainConfig.from_preset(
                args.postprocess_preset,
                device=args.postprocess_device,
            ),
            total_steps=args.total_steps,
            duration_seconds=args.duration_seconds,
            control_fps=args.control_fps,
            synchronize_metrics=args.synchronize_metrics,
            isaac_teleop_python=args.isaac_teleop_python,
            isaac_teleop_startup_timeout_s=args.isaac_teleop_startup_timeout_s,
            position_gain=args.position_gain,
            grip_threshold=args.grip_threshold,
            stale_ms=args.stale_ms,
            gripper_slew_per_action=args.gripper_slew_per_action,
            open_claw_threshold=args.open_claw_threshold,
            auto_start=args.auto_start,
        )
        config = self._config
        if self._pipeline is None:
            self._pipeline = self._pipeline_factory(
                python_executable=str(config.isaac_teleop_python),
                startup_timeout_s=config.isaac_teleop_startup_timeout_s,
            )
            self._pipeline.start()
        # Validate the calibration service before downloading model weights.
        if self._inference_pipeline is None:
            self._inference_pipeline = self.defaults.inference_pipeline_factory(
                config.pipeline_config
            )
        for name in dict.fromkeys((config.scenario.sample, *self._scene_catalog)):
            if name in self._prepared_scenes:
                continue
            print(f"[handdreams] preparing scene {name}", flush=True)
            scene = self.defaults.scene_factory(
                replace(self.launch_config.scenario, sample=name),
                self._inference_pipeline,
            )
            self.defaults.rollout_builder_factory(
                pipeline=self._inference_pipeline, scene=scene
            ).prepare_scene()
            self._prepared_scenes[name] = scene
            print(f"[handdreams] scene {name} ready for controllers", flush=True)

    def session_desc(self) -> SessionDesc:
        """Advertise fixed checkpoint geometry before arguments are parsed."""
        return SessionDesc(
            output_layout=VideoTensorLayout.tchw,
            backpressure_mode=BackpressureMode.BLOCK,
            presentation_mode=PresentationMode.CONTINUOUS,
            frames_per_second_for_ui=_DEFAULT_UI_FPS,
            frames_per_second_for_step=_DEFAULT_STEP_FPS,
            video_width=self.defaults.video_width,
            video_height=self.defaults.video_height,
        )

    def create_session(self, session_desc: SessionDesc) -> ISession:
        """Create per-session scene/rollout resources from app-owned services."""
        if self._closed:
            raise RuntimeError("cannot create a session from a closed application")
        config = self.launch_config
        input_spec = VideoSpec(
            width=self.defaults.video_width,
            height=self.defaults.video_height,
            fps=session_desc.frames_per_second_for_step,
        )
        output_spec = config.postprocess.output_spec(input_spec)
        _validate_session_desc(
            session_desc, input_spec=input_spec, output_spec=output_spec
        )
        if output_spec.channels != 3 or output_spec.fps != input_spec.fps:
            raise ValueError(
                "HandDreams post-processing must preserve RGB channels and frame rate"
            )
        requested = session_desc.metadata.get("sample", config.scenario.sample)
        if not isinstance(requested, str) or requested not in self._scene_catalog:
            raise ValueError(
                f"unknown HandDreams scene {requested!r}; choose from "
                + ", ".join(self._scene_catalog)
            )
        require_scene_selection = session_desc.metadata.get("select_scene") is True or (
            not config.auto_start and "sample" not in session_desc.metadata
        )
        session_desc = replace(
            session_desc,
            metadata={
                **session_desc.metadata,
                "sample": requested,
                "scene_catalog": self._scene_catalog,
            },
        )
        pipeline = self._pipeline
        inference_pipeline = self._inference_pipeline
        if inference_pipeline is None:
            raise RuntimeError(
                "HandDreamsV2Application.init() must prepare the inference pipeline."
            )
        if pipeline is None:
            raise RuntimeError(
                "HandDreamsV2Application.init() must prepare the controller pipeline."
            )
        if requested not in self._prepared_scenes:
            raise RuntimeError(
                f"HandDreams scene {requested!r} was not prepared during init()."
            )
        scene = self._prepared_scenes[requested]
        postprocess_stream = create_video_postprocess_stream(
            postprocess=config.postprocess,
            output_layout=session_desc.output_layout.value,
            fps=input_spec.fps,
            per_view=False,
            world_size=1,
        )
        if postprocess_stream is not None:
            postprocess_stream.prepare(input_spec)
        session_desc = replace(
            session_desc,
            video_width=output_spec.width,
            video_height=output_spec.height + scene_strip_height(output_spec.width),
        )
        return HandDreamsSession(
            builder=self.defaults.rollout_builder_factory(
                pipeline=inference_pipeline,
                scene=scene,
            ),
            config=config,
            session_desc=session_desc,
            postprocess_stream=postprocess_stream,
            pipeline=pipeline,
            scene_catalog=self._scene_catalog,
            scene_previews=self._scene_previews,
            stage_width=output_spec.width,
            stage_height=output_spec.height,
            require_scene_selection=require_scene_selection,
        )

    def close(self) -> None:
        """Release every owned resource, even when an earlier close fails."""
        if self._closed:
            return
        self._closed = True
        pipeline, self._pipeline = self._pipeline, None
        inference_pipeline, self._inference_pipeline = self._inference_pipeline, None
        self._config = None
        self._scene_catalog = ()
        self._scene_previews = ()
        self._prepared_scenes.clear()

        authoritative_errors: list[tuple[str, Exception]] = []
        if pipeline is not None:
            try:
                pipeline.close()
            except Exception as error:  # noqa: BLE001 - finish all cleanup first
                authoritative_errors.append(("Isaac Teleop pipeline", error))
        close = getattr(inference_pipeline, "close", None)
        if callable(close):
            try:
                close()
            except Exception as error:  # noqa: BLE001 - finish all cleanup first
                authoritative_errors.append(("HandDreams inference pipeline", error))

        if authoritative_errors:
            _, primary = authoritative_errors[0]
            for owner, error in authoritative_errors[1:]:
                add_note = getattr(primary, "add_note", None)
                if callable(add_note):
                    add_note(f"Additional {owner} cleanup failure: {error!r}")
            raise primary


def _add_teleop_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--isaac-teleop-python",
        type=Path,
        default=Path(DEFAULT_ISAAC_TELEOP_PYTHON),
        help=f"Python for the isolated calibration service (default: {DEFAULT_ISAAC_TELEOP_PYTHON}).",
    )
    parser.add_argument("--isaac-teleop-startup-timeout-s", type=float, default=300.0)
    parser.add_argument("--position-gain", type=float, default=1.2)
    parser.add_argument("--grip-threshold", type=float, default=0.0)
    parser.add_argument("--stale-ms", type=float, default=500.0)
    parser.add_argument("--gripper-slew-per-action", type=float, default=0.25)
    parser.add_argument("--open-claw-threshold", type=float, default=0.1)
    parser.add_argument(
        "--auto-start",
        action="store_true",
        help="Skip scene selection, trigger capture, and alignment; calibrate once both controllers are tracked and open.",
    )


def _validate_session_desc(
    desc: SessionDesc, *, input_spec: VideoSpec, output_spec: VideoSpec
) -> None:
    if desc.output_layout is not VideoTensorLayout.tchw:
        raise ValueError("HandDreams V2 produces only tchw output")
    size = (desc.video_width, desc.video_height)
    expected = {
        (input_spec.width, input_spec.height),
        (output_spec.width, output_spec.height),
        (
            input_spec.width,
            input_spec.height + scene_strip_height(input_spec.width),
        ),
        (
            output_spec.width,
            output_spec.height + scene_strip_height(output_spec.width),
        ),
    }
    if size not in expected:
        raise ValueError(
            "HandDreams V2 requires model or presentation dimensions "
            f"{sorted(expected)}, got {size}"
        )


__all__ = ["HandDreamsV2Application"]
