# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OmniDreams adapter for the shared demo API."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from typing import Any

from omnidreams.config import OMNIDREAMS_CONFIGS, OMNIDREAMS_RUNNERS
from omnidreams.webrtc.session import (
    OmnidreamsInferenceRuntime,
    OmnidreamsRuntimeConfig,
)

from flashdreams.infra.postprocess import VideoPostprocessChainConfig
from flashdreams.runtime import (
    CanonicalInputSchema,
    IdentityInputMapping,
    InferenceConfig,
    InferenceInput,
    InferenceInputSchema,
    InputCanonicalizer,
    InputField,
    UserInputSchema,
)
from flashdreams.runtime.demo import (
    DemoSpec,
    Mp4OutputSpec,
    PreparedScenario,
    WebRTCOutputSpec,
)
from flashdreams.runtime.interfaces import InferenceRuntime

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

    def default_input_mapping(self) -> IdentityInputMapping:
        return self._mapping

    def supported_input_modes(self) -> tuple[str, ...]:
        return ("replay", "keyboard-driving")

    def supported_output_modes(self) -> tuple[str, ...]:
        return ("mp4", "webrtc")

    def prepare_scenario(self, spec: DemoSpec) -> PreparedScenario:
        if spec.input_mode != "replay":
            raise ValueError(
                "OmniDreams prepare_scenario currently supports only "
                f"input_mode='replay', got {spec.input_mode!r}."
            )
        if not isinstance(spec.output, Mp4OutputSpec):
            raise ValueError("OmniDreams replay demo currently requires MP4 output.")
        scenario = resolve_replay_scenario(
            spec.scenario,
            default_prompt=self._default_replay_prompt(spec.config),
        )
        return PreparedScenario(
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
            raise ValueError("OmniDreams WebRTC requires WebRTC output.")
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
