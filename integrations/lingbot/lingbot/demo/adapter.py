# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lingbot adapter for the shared demo API."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from flashdreams.runtime import (
    InferenceConfig,
    InputCanonicalizer,
    UserInputSchema,
)
from flashdreams.runtime.demo import (
    DemoSpec,
    Mp4OutputSpec,
    PreparedScenario,
    WebRTCOutputSpec,
)
from flashdreams.runtime.interfaces import InferenceRuntime
from lingbot.runtime import (
    LingbotModelAdapter,
    LingbotReplayRuntime,
    PipelineFactory,
    build_lingbot_webrtc_runtime_config,
    inference_input_from_replay_inputs,
)
from lingbot.webrtc.session import (
    LingbotInferenceRuntime,
    LingbotRuntimeConfig,
)

from .spec import resolve_replay_inputs, resolve_webrtc_scenario
from .webrtc import (
    LingbotDemoWebRTCSessionManager,
    create_lingbot_webrtc_app,
)

ReplayRuntimeFactory = Callable[..., InferenceRuntime]
WebRTCRuntimeFactory = Callable[..., Any]


class LingbotDemoAdapter(LingbotModelAdapter):
    """Model-owned Lingbot adapter consumed by shared demo launchers."""

    def __init__(
        self,
        *,
        replay_runtime_factory: ReplayRuntimeFactory = LingbotReplayRuntime,
        webrtc_runtime_factory: WebRTCRuntimeFactory = LingbotInferenceRuntime,
        pipeline_factory: PipelineFactory | None = None,
    ) -> None:
        super().__init__(
            runtime_factory=replay_runtime_factory,
            pipeline_factory=pipeline_factory,
        )
        self._webrtc_runtime_factory = webrtc_runtime_factory

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

        replay_inputs = resolve_replay_inputs(
            spec.scenario,
            default_prompt=self.default_replay_prompt(spec.config),
        )
        return PreparedScenario(
            initial_inputs=inference_input_from_replay_inputs(replay_inputs),
            source_schema=UserInputSchema(description="fixed Lingbot replay input"),
            canonicalizer=InputCanonicalizer(),
            mapping=self.default_input_mapping(),
            metadata={
                "model_id": self.model_id,
                "preset_id": self.preset_id(spec.config),
            },
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

        compile_network = (
            bool(config.compile)
            if config.compile is not None
            else bool(_option(config, "compile_network", True))
        )
        return build_lingbot_webrtc_runtime_config(
            preset_id=self.preset_id(config),
            pipeline_config=self.pipeline_config(config),
            seed=int(_option(config, "seed", 42)),
            compile_network=compile_network,
            context_parallel_size=int(_option(config, "context_parallel_size", 1)),
            device=config.device or str(_option(config, "device", "cuda:0")),
            video_height=spec.output.video_height,
            video_width=spec.output.video_width,
            fps=spec.output.fps,
            warmup_chunks=spec.output.warmup_chunks,
            warmup_timeout_s=spec.output.warmup_timeout_s,
            example_idx=int(_option(config, "example_idx", scenario.example_idx)),
            prefer_sw_encoder=scenario.prefer_sw_encoder,
            runtime_options=config.runtime_options,
        )

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


def _option(config: InferenceConfig, name: str, default: Any) -> Any:
    return config.runtime_options.get(name, default)


__all__ = [
    "LingbotDemoAdapter",
    "ReplayRuntimeFactory",
    "WebRTCRuntimeFactory",
]
