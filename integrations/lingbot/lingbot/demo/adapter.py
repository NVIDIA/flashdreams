# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lingbot adapter for the shared demo API."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from typing import Any

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
from lingbot.config import PIPELINE_CONFIGS, RUNNER_CONFIGS
from lingbot.runner import (
    EXAMPLE_DATA_DIR_LOCAL,
    example_data_dirname,
)
from lingbot.webrtc.session import (
    LingbotInferenceRuntime,
    LingbotRuntimeConfig,
)

from .replay import (
    LingbotReplayRuntime,
    LingbotReplayRuntimeOptions,
    PipelineFactory,
)
from .spec import (
    DEFAULT_LINGBOT_PRESET,
    LINGBOT_MODEL_ID,
    example_asset_urls,
    resolve_replay_scenario,
    resolve_webrtc_scenario,
)
from .webrtc import (
    LingbotDemoWebRTCSessionManager,
    create_lingbot_webrtc_app,
)

ReplayRuntimeFactory = Callable[..., InferenceRuntime]
WebRTCRuntimeFactory = Callable[..., Any]


class LingbotDemoAdapter:
    """Model-owned Lingbot adapter consumed by shared demo launchers."""

    def __init__(
        self,
        *,
        replay_runtime_factory: ReplayRuntimeFactory = LingbotReplayRuntime,
        webrtc_runtime_factory: WebRTCRuntimeFactory = LingbotInferenceRuntime,
        pipeline_factory: PipelineFactory | None = None,
    ) -> None:
        self._replay_runtime_factory = replay_runtime_factory
        self._webrtc_runtime_factory = webrtc_runtime_factory
        self._pipeline_factory = pipeline_factory
        self._mapping = IdentityInputMapping()

    @property
    def model_id(self) -> str:
        return LINGBOT_MODEL_ID

    @property
    def inference_input_schema(self) -> InferenceInputSchema:
        return InferenceInputSchema(
            global_conditioning_fields=(
                InputField(
                    name="scenario",
                    input_modality="lingbot/replay-scenario",
                    description="Resolved Lingbot replay scenario.",
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
                "Lingbot prepare_scenario currently supports only "
                f"input_mode='replay', got {spec.input_mode!r}."
            )
        if not isinstance(spec.output, Mp4OutputSpec):
            raise ValueError("Lingbot replay demo currently requires MP4 output.")
        scenario = resolve_replay_scenario(
            spec.scenario,
            default_prompt=self._default_replay_prompt(spec.config),
        )
        return PreparedScenario(
            initial_inputs=InferenceInput(
                global_conditioning={"scenario": scenario},
            ),
            source_schema=UserInputSchema(description="fixed Lingbot replay input"),
            canonicalizer=InputCanonicalizer(),
            mapping=self._mapping,
            metadata={
                "model_id": self.model_id,
                "preset_id": self._preset_id(spec.config),
            },
        )

    def validate_config(self, config: InferenceConfig) -> None:
        if config.model_id != self.model_id:
            raise ValueError(
                f"Lingbot adapter requires model_id={self.model_id!r}, "
                f"got {config.model_id!r}."
            )
        self._pipeline_config(config)

    def create_runtime(self, config: InferenceConfig) -> InferenceRuntime:
        self.validate_config(config)
        return self._replay_runtime_factory(
            config=config,
            options=LingbotReplayRuntimeOptions(
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
    ) -> LingbotRuntimeConfig:
        runtime_config = getattr(runtime, "config", None)
        if isinstance(runtime_config, LingbotRuntimeConfig):
            return runtime_config
        if spec.input_mode != "keyboard-driving":
            raise ValueError(
                "Lingbot WebRTC requires input_mode='keyboard-driving', "
                f"got {spec.input_mode!r}."
            )
        if not isinstance(spec.output, WebRTCOutputSpec):
            raise ValueError("Lingbot WebRTC requires WebRTC output.")
        config = spec.config
        if config is None:
            raise RuntimeError("DemoSpec.config was not initialized.")
        self.validate_config(config)
        scenario = resolve_webrtc_scenario(spec.scenario)

        preset_id = self._preset_id(config)
        pipeline_config = self._pipeline_config(config)
        seed = int(_option(config, "seed", 42))
        example_idx = int(_option(config, "example_idx", scenario.example_idx))
        example_dirname = example_data_dirname(example_idx)
        example_dir = EXAMPLE_DATA_DIR_LOCAL / example_dirname
        if (
            example_idx == 0
            and not example_dir.exists()
            and (EXAMPLE_DATA_DIR_LOCAL / "image.jpg").exists()
        ):
            example_dir = EXAMPLE_DATA_DIR_LOCAL
        urls = example_asset_urls(example_idx)
        compile_network = (
            bool(config.compile)
            if config.compile is not None
            else bool(_option(config, "compile_network", True))
        )
        runtime_config = LingbotRuntimeConfig(
            config_name=preset_id,
            pipeline_config=pipeline_config,
            compile_network=compile_network,
            seed=seed,
            context_parallel_size=int(_option(config, "context_parallel_size", 1)),
            device=config.device or str(_option(config, "device", "cuda:0")),
            video_height=spec.output.video_height,
            video_width=spec.output.video_width,
            fps=spec.output.fps,
            warmup_chunks=spec.output.warmup_chunks,
            warmup_timeout_s=spec.output.warmup_timeout_s,
            encoder_backend="default" if scenario.prefer_sw_encoder else "auto",
            example_data_dir=example_dir,
            default_image_url=urls["image"],
            default_intrinsics_url=urls["intrinsics"],
            default_poses_url=urls["poses"],
        )
        return _apply_webrtc_runtime_options(runtime_config, config.runtime_options)

    def create_webrtc_session_manager(
        self,
        *,
        spec: DemoSpec,
        runtime: Any,
        runtime_config: LingbotRuntimeConfig,
        fps: int,
        client_liveness_timeout_s: float,
    ) -> LingbotDemoWebRTCSessionManager:
        del spec
        return LingbotDemoWebRTCSessionManager(
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
        return create_lingbot_webrtc_app(
            spec=spec,
            session_manager=session_manager,
            request_session_url=request_session_url,
        )

    def _preset_id(self, config: InferenceConfig | None) -> str:
        return (
            DEFAULT_LINGBOT_PRESET
            if config is None or config.preset_id is None
            else config.preset_id
        )

    def _pipeline_config(self, config: InferenceConfig) -> Any:
        custom = config.runtime_options.get("pipeline_config")
        if custom is not None:
            return custom
        preset_id = self._preset_id(config)
        try:
            return PIPELINE_CONFIGS[preset_id]
        except KeyError as exc:
            supported = ", ".join(sorted(PIPELINE_CONFIGS))
            raise ValueError(
                f"Unsupported Lingbot preset_id={preset_id!r}. "
                f"Supported presets: {supported}."
            ) from exc

    def _default_replay_prompt(self, config: InferenceConfig | None) -> str:
        runner = RUNNER_CONFIGS.get(self._preset_id(config))
        return "" if runner is None else str(getattr(runner, "prompt", ""))


def _option(config: InferenceConfig, name: str, default: Any) -> Any:
    return config.runtime_options.get(name, default)


def _apply_webrtc_runtime_options(
    runtime_config: LingbotRuntimeConfig,
    options: Any,
) -> LingbotRuntimeConfig:
    if not isinstance(options, dict):
        options = dict(options)
    overrides: dict[str, Any] = {}
    for name in (
        "world_scale",
        "default_intrinsics",
        "default_prompt",
        "default_image_url",
        "default_intrinsics_url",
        "default_poses_url",
        "encoder_bitrate_bps",
        "encoder_gop",
        "text_events",
    ):
        if name in options:
            overrides[name] = options[name]
    return replace(runtime_config, **overrides) if overrides else runtime_config


__all__ = [
    "LingbotDemoAdapter",
    "ReplayRuntimeFactory",
    "WebRTCRuntimeFactory",
]
