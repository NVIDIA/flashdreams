# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OmniDreams adapter for the shared demo API."""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any

from flashdreams.infra.postprocess import VideoPostprocessChainConfig
from flashdreams.infra.video_output import VideoStepResult
from flashdreams.runtime import (
    RGB_VIDEO,
    CanonicalInputSchema,
    IdentityInputMapping,
    InferenceConfig,
    InferenceInput,
    InferenceInputSchema,
    InferenceOutputSchema,
    InputCanonicalizer,
    InputField,
    UserInputSchema,
)
from flashdreams.runtime.demo import (
    DemoRoute,
    DemoSpec,
    LocalWindowOutputSpec,
    Mp4OutputSpec,
    PreparedSession,
    WebRTCOutputSpec,
)
from flashdreams.runtime.interfaces import InferenceRuntime
from omnidreams.config import OMNIDREAMS_CONFIGS, OMNIDREAMS_RUNNERS
from omnidreams.webrtc.session import (
    OmnidreamsInferenceRuntime,
    OmnidreamsRuntimeConfig,
)

from .driving import (
    DRIVING_INPUT_SCHEMA,
    OmnidreamsDriverCommandMapping,
    OmnidreamsDrivingRuntime,
    OmnidreamsDrivingScenario,
)
from .replay import (
    OmnidreamsReplayRuntime,
    OmnidreamsReplayRuntimeOptions,
    PipelineFactory,
)
from .spec import (
    DEFAULT_OMNIDREAMS_PRESET,
    OMNIDREAMS_MODEL_ID,
    resolve_replay_scenario,
    resolve_webrtc_scenario,
)
from .webrtc import (
    OmnidreamsDemoWebRTCSessionManager,
    create_omnidreams_webrtc_app,
    validate_postprocess_preset,
)

ReplayRuntimeFactory = Callable[..., InferenceRuntime]
WebRTCRuntimeFactory = Callable[..., Any]


@dataclass(slots=True)
class _DrivingRuntimeSetup:
    app_config: Any
    backend: Any
    runtime_created: bool = False


class OmnidreamsDemoAdapter:
    """Model-owned OmniDreams adapter consumed by shared demo launchers."""

    def __init__(
        self,
        *,
        replay_runtime_factory: ReplayRuntimeFactory = OmnidreamsReplayRuntime,
        webrtc_runtime_factory: WebRTCRuntimeFactory = OmnidreamsInferenceRuntime,
        pipeline_factory: PipelineFactory | None = None,
    ) -> None:
        self._replay_runtime_factory = replay_runtime_factory
        self._webrtc_runtime_factory = webrtc_runtime_factory
        self._pipeline_factory = pipeline_factory
        self._mapping = IdentityInputMapping()
        self._driving_mapping = OmnidreamsDriverCommandMapping()
        self._driving_runtime_setup: _DrivingRuntimeSetup | None = None
        self._driving_scene_cache: dict[
            tuple[str, str, str | None],
            OmnidreamsDrivingScenario,
        ] = {}
        self._driving_setup_lock = threading.Lock()

    @property
    def model_id(self) -> str:
        return OMNIDREAMS_MODEL_ID

    @property
    def inference_input_schema(self) -> InferenceInputSchema:
        return InferenceInputSchema(
            global_conditioning_fields=(
                InputField(
                    name="scenario",
                    input_modality="omnidreams/replay-scenario",
                    description="Resolved OmniDreams replay scenario.",
                ),
            )
        )

    @property
    def canonical_input_schema(self) -> CanonicalInputSchema | None:
        return None

    @property
    def inference_output_schema(self) -> InferenceOutputSchema:
        return InferenceOutputSchema(
            modality=RGB_VIDEO,
            python_type=VideoStepResult,
            layouts=frozenset({"bvtchw"}),
        )

    def default_input_mapping(self) -> IdentityInputMapping:
        return self._mapping

    def supported_routes(self) -> tuple[DemoRoute, ...]:
        return (
            DemoRoute(input_mode="replay", output_mode="mp4"),
            DemoRoute(input_mode="keyboard-driving", output_mode="webrtc"),
            DemoRoute(input_mode="keyboard-driving", output_mode="local-window"),
        )

    def list_sessions(self, spec: DemoSpec) -> tuple[DemoSpec, ...]:
        if not isinstance(spec.output, LocalWindowOutputSpec):
            return (spec,)
        from omnidreams.demo.local_window import OmnidreamsLocalWindowScenario
        from omnidreams.interactive_drive.demo import _discover_scene_options

        scenario = spec.scenario or OmnidreamsLocalWindowScenario()
        if not isinstance(scenario, OmnidreamsLocalWindowScenario):
            raise TypeError(
                "OmniDreams local-window scenario must be "
                "OmnidreamsLocalWindowScenario."
            )
        if scenario.scene_dir is None or scenario.scene is None:
            return (spec,)
        options = _discover_scene_options(scenario.scene_dir, scenario.scene)
        sessions: list[DemoSpec] = []
        for option in options:
            for variant in option.variants:
                scene_path = option.variant_paths.get(variant, option.path)
                metadata = dict(spec.metadata)
                metadata.update(
                    scene_label=option.label,
                    variant_label=variant,
                )
                sessions.append(
                    replace(
                        spec,
                        scenario=replace(
                            scenario,
                            scene=scene_path,
                            variant=variant,
                        ),
                        metadata=metadata,
                    )
                )
        return tuple(sessions) or (spec,)

    def create_local_window_app(self, *, spec: DemoSpec) -> Any:
        """Build the selected standard or compatibility local-window app."""
        from omnidreams.demo.local_window import build_omnidreams_local_window_app

        return build_omnidreams_local_window_app(spec=spec, adapter=self)

    def prepare_session(self, spec: DemoSpec) -> PreparedSession:
        """Prepare model inputs for one demo session."""
        if spec.input_mode == "keyboard-driving" and isinstance(
            spec.output, LocalWindowOutputSpec
        ):
            scenario = self._prepare_driving_scenario(spec)
            return PreparedSession(
                initial_inputs=InferenceInput(
                    global_conditioning={"driving_scenario": scenario},
                ),
                inference_input_schema=DRIVING_INPUT_SCHEMA,
                source_schema=UserInputSchema(
                    description="live interactive-drive input"
                ),
                canonicalizer=InputCanonicalizer(),
                mapping=self._driving_mapping,
                metadata={
                    "model_id": self.model_id,
                    "preset_id": self._preset_id(spec.config),
                    "scene_label": scenario.scene.scene_path.stem,
                },
            )
        if spec.input_mode != "replay":
            raise ValueError(
                "OmniDreams interactive session preparation is not implemented; "
                f"input_mode={spec.input_mode!r} runs under its own app loop. "
                "Use create_local_window_app or create_webrtc_runtime."
            )
        if not isinstance(spec.output, Mp4OutputSpec):
            raise TypeError("OmniDreams replay demo currently requires MP4 output.")
        scenario = resolve_replay_scenario(
            spec.scenario,
            default_prompt=self._default_replay_prompt(spec.config),
        )
        return PreparedSession(
            initial_inputs=InferenceInput(
                global_conditioning={"scenario": scenario},
            ),
            source_schema=UserInputSchema(description="fixed OmniDreams replay input"),
            canonicalizer=InputCanonicalizer(),
            mapping=self._mapping,
            metadata={
                "model_id": self.model_id,
                "preset_id": self._preset_id(spec.config),
                "num_views": len(scenario.camera_names),
            },
        )

    def validate_config(self, config: InferenceConfig) -> None:
        if config.model_id != self.model_id:
            raise ValueError(
                f"OmniDreams adapter requires model_id={self.model_id!r}, "
                f"got {config.model_id!r}."
            )
        self._pipeline_config(config)

    def create_runtime(self, config: InferenceConfig) -> InferenceRuntime:
        self.validate_config(config)
        return self._replay_runtime_factory(
            config=config,
            options=OmnidreamsReplayRuntimeOptions(
                pipeline_config=self._pipeline_config(config),
                pipeline_factory=self._pipeline_factory,
            ),
        )

    def create_demo_runtime(self, spec: DemoSpec) -> InferenceRuntime:
        if spec.input_mode == "keyboard-driving" and isinstance(
            spec.output, LocalWindowOutputSpec
        ):
            setup = self._prepare_driving_runtime_setup(spec)
            with self._driving_setup_lock:
                if setup.runtime_created:
                    raise RuntimeError(
                        "Driving runtime was already created for this DemoSpec."
                    )
                setup.runtime_created = True
            config = spec.config
            if config is None:
                raise RuntimeError("DemoSpec.config was not initialized.")
            return OmnidreamsDrivingRuntime(
                config=config,
                app_config=setup.app_config,
                backend=setup.backend,
            )
        config = spec.config
        if config is None:
            raise RuntimeError("DemoSpec.config was not initialized.")
        return self.create_runtime(config)

    def create_webrtc_runtime(self, spec: DemoSpec) -> Any:
        runtime_config = self.create_webrtc_runtime_config(spec=spec, runtime=None)
        return self._webrtc_runtime_factory(config=runtime_config)

    def create_webrtc_runtime_config(
        self,
        *,
        spec: DemoSpec,
        runtime: Any,
    ) -> OmnidreamsRuntimeConfig:
        runtime_config = getattr(runtime, "config", None)
        if isinstance(runtime_config, OmnidreamsRuntimeConfig):
            return runtime_config
        if spec.input_mode != "keyboard-driving":
            raise ValueError(
                "OmniDreams WebRTC requires input_mode='keyboard-driving', "
                f"got {spec.input_mode!r}."
            )
        if not isinstance(spec.output, WebRTCOutputSpec):
            raise TypeError("OmniDreams WebRTC requires WebRTC output.")
        config = spec.config
        if config is None:
            raise RuntimeError("DemoSpec.config was not initialized.")
        self.validate_config(config)
        scenario = resolve_webrtc_scenario(spec.scenario)
        validate_postprocess_preset(scenario.postprocess_preset)

        preset_id = self._preset_id(config)
        pipeline_config = self._pipeline_config(config)
        seed = _option(config, "seed", 42)
        device = config.device or str(_option(config, "device", "cuda:0"))
        runtime_config = OmnidreamsRuntimeConfig(
            pipeline_config_name=preset_id,
            pipeline_config=pipeline_config,
            scene_dir=scenario.scene_dir,
            scene_uuid=scenario.scene_uuid,
            scene_variant=scenario.scene_variant,
            seed=None if seed is None else int(seed),
            device=device,
            video_height=spec.output.video_height,
            video_width=spec.output.video_width,
            fps=spec.output.fps,
            camera_name=scenario.camera_name,
            warmup_chunks=spec.output.warmup_chunks,
            warmup_timeout_s=spec.output.warmup_timeout_s,
            debug_serve_hdmaps=scenario.debug_serve_hdmaps,
            postprocess=VideoPostprocessChainConfig(preset=scenario.postprocess_preset),
            encoder_backend="default" if scenario.prefer_sw_encoder else "auto",
        )
        return _apply_webrtc_runtime_options(runtime_config, config.runtime_options)

    def create_webrtc_session_manager(
        self,
        *,
        spec: DemoSpec,
        runtime: Any,
        runtime_config: OmnidreamsRuntimeConfig,
        fps: int,
        client_liveness_timeout_s: float,
    ) -> OmnidreamsDemoWebRTCSessionManager:
        del spec
        return OmnidreamsDemoWebRTCSessionManager(
            runtime=runtime,
            runtime_config=runtime_config,
            fps=fps,
            client_liveness_timeout_s=client_liveness_timeout_s,
        )

    def create_webrtc_app(
        self,
        *,
        spec: DemoSpec,
        session_manager: Any,
        request_session_url: str,
    ) -> Any:
        return create_omnidreams_webrtc_app(
            spec=spec,
            session_manager=session_manager,
            request_session_url=request_session_url,
        )

    def _prepare_driving_runtime_setup(
        self,
        spec: DemoSpec,
    ) -> _DrivingRuntimeSetup:
        from omnidreams.demo.local_window import build_interactive_drive_app
        from omnidreams.interactive_drive.cli import prepare_config_and_backend
        from omnidreams.interactive_drive.demo import build_parser

        with self._driving_setup_lock:
            if self._driving_runtime_setup is not None:
                return self._driving_runtime_setup
            argv = build_interactive_drive_app(spec).argv
            args = build_parser().parse_args(list(argv))
            app_config, backend = prepare_config_and_backend(args)
            setup = _DrivingRuntimeSetup(app_config=app_config, backend=backend)
            self._driving_runtime_setup = setup
            return setup

    def _prepare_driving_scenario(
        self,
        spec: DemoSpec,
    ) -> OmnidreamsDrivingScenario:
        from omnidreams.demo.local_window import build_interactive_drive_app
        from omnidreams.interactive_drive.cli import prepare_app_config
        from omnidreams.interactive_drive.demo import build_parser
        from omnidreams.interactive_drive.scene_loader import load_scene_bundle
        from omnidreams.interactive_drive.simulation.ego_vehicle_kinematics import (
            build_ground_snapper,
            build_map_bounds,
        )

        runtime_setup = self._prepare_driving_runtime_setup(spec)
        argv = build_interactive_drive_app(spec).argv
        args = build_parser().parse_args(list(argv))
        app_config = prepare_app_config(args)
        if _driving_runtime_signature(app_config) != _driving_runtime_signature(
            runtime_setup.app_config
        ):
            raise ValueError(
                "Driving scene requires a different model/runtime configuration; "
                "create a new InteractiveDriveApplication."
            )
        cache_key = (
            str(app_config.scene_path),
            app_config.variant,
            app_config.prompt_override,
        )
        with self._driving_setup_lock:
            cached = self._driving_scene_cache.get(cache_key)
        if cached is not None:
            return cached
        scene = load_scene_bundle(
            scene_path=app_config.scene_path,
            camera_name=app_config.camera_name,
            variant=app_config.variant,
            prompt_override=app_config.prompt_override,
            raster=app_config.raster,
        )
        scenario = OmnidreamsDrivingScenario(
            app_config=app_config,
            scene=scene,
            map_bounds=build_map_bounds(scene),
            ground_snapper=build_ground_snapper(scene),
        )
        with self._driving_setup_lock:
            return self._driving_scene_cache.setdefault(cache_key, scenario)

    def _preset_id(self, config: InferenceConfig | None) -> str:
        return (
            DEFAULT_OMNIDREAMS_PRESET
            if config is None or config.preset_id is None
            else config.preset_id
        )

    def _pipeline_config(self, config: InferenceConfig) -> Any:
        custom = config.runtime_options.get("pipeline_config")
        if custom is not None:
            return custom
        preset_id = self._preset_id(config)
        try:
            return OMNIDREAMS_CONFIGS[preset_id]
        except KeyError as exc:
            supported = ", ".join(sorted(OMNIDREAMS_CONFIGS))
            raise ValueError(
                f"Unsupported OmniDreams preset_id={preset_id!r}. "
                f"Supported presets: {supported}."
            ) from exc

    def _default_replay_prompt(self, config: InferenceConfig | None) -> str:
        runner = OMNIDREAMS_RUNNERS.get(self._preset_id(config))
        return "" if runner is None else str(getattr(runner, "prompt", ""))


def _option(config: InferenceConfig, name: str, default: Any) -> Any:
    return config.runtime_options.get(name, default)


def _driving_runtime_signature(config: Any) -> tuple[Any, ...]:
    return (
        config.backend,
        config.manifest_path,
        config.chunk,
        config.raster,
        config.vehicle,
        config.world_model_profile,
        config.world_model_offload_text_encoder,
        config.postprocess,
        config.bev,
    )


def _apply_webrtc_runtime_options(
    runtime_config: OmnidreamsRuntimeConfig,
    options: Any,
) -> OmnidreamsRuntimeConfig:
    if not isinstance(options, dict):
        options = dict(options)
    overrides: dict[str, Any] = {}
    for name in (
        "move_speed_per_s",
        "rotate_speed_rad_per_s",
        "encoder_bitrate_bps",
        "encoder_gop",
    ):
        if name in options:
            overrides[name] = options[name]
    return replace(runtime_config, **overrides) if overrides else runtime_config


__all__ = [
    "OmnidreamsDemoAdapter",
    "ReplayRuntimeFactory",
    "WebRTCRuntimeFactory",
]
