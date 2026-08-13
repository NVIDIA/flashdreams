# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Host-owned WebRTC presentation for application runtimes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from flashdreams.runtime import InferenceConfig, InferenceInput
from flashdreams.runtime.demo import (
    DemoSpec,
    PreparedScenario,
    PreparedStep,
    ProviderCapabilities,
    RuntimeHost,
    UserInputWindow,
    WebRTCAppResources,
    WebRTCOutputSpec,
)
from flashdreams.runtime.types import StepRequirements
from flashdreams.serving.webrtc.demo import serve_webrtc_demo
from flashdreams.serving.webrtc.manager import BaseWebRTCSessionManager

from .contracts import AppRuntime


@dataclass(frozen=True, slots=True)
class WebRTCOptions:
    """Presentation settings owned by ``flashdreams-app``."""

    host: str
    port: int
    warmup_chunks: int
    warmup_timeout_s: float
    client_liveness_timeout_s: float
    device: str
    encoder_backend: str
    encoder_bitrate_bps: int
    encoder_gop: int


@dataclass(frozen=True, slots=True)
class _WebRTCRuntimeConfig:
    video_width: int
    video_height: int
    warmup_chunks: int
    warmup_timeout_s: float
    device: str
    encoder_backend: str
    encoder_bitrate_bps: int
    encoder_gop: int


class _InputProvider:
    capabilities = ProviderCapabilities(
        supports_realtime_clock=True,
        deterministic_given_inputs=True,
    )

    def __init__(self, runtime: AppRuntime) -> None:
        self._runtime = runtime

    def prepare_initial_input(self) -> InferenceInput:
        return self._runtime.initial_input

    def prepare_step(
        self, *, request: StepRequirements, user_window: UserInputWindow
    ) -> PreparedStep:
        del user_window
        return PreparedStep(inference_input=self._runtime.prepare_step_input(request))

    def reset(self, inputs: InferenceInput | None = None) -> None:
        del inputs

    def close(self) -> None:
        pass


def serve_webrtc(
    *, runtime: AppRuntime, options: WebRTCOptions, world_rank: int
) -> object:
    """Serve an application runtime through the shared WebRTC stack."""
    metadata = runtime.metadata
    output = WebRTCOutputSpec(
        host=options.host,
        port=options.port,
        fps=int(metadata.fps),
        video_width=metadata.video_width,
        video_height=metadata.video_height,
        warmup_chunks=options.warmup_chunks,
        warmup_timeout_s=options.warmup_timeout_s,
        client_liveness_timeout_s=options.client_liveness_timeout_s,
        preload_name=metadata.model_id,
    )
    spec = DemoSpec(
        model_id=metadata.model_id,
        input_mode="webrtc",
        output=output,
        config=InferenceConfig(model_id=metadata.model_id, device=options.device),
    )
    scenario = PreparedScenario(initial_inputs=runtime.initial_input)

    def create_model_input_provider(
        spec: DemoSpec,
        scenario: PreparedScenario,
    ) -> _InputProvider:
        del spec, scenario
        return _InputProvider(runtime)

    manager = BaseWebRTCSessionManager(
        runtime=runtime,
        runtime_config=_WebRTCRuntimeConfig(
            video_width=metadata.video_width,
            video_height=metadata.video_height,
            warmup_chunks=options.warmup_chunks,
            warmup_timeout_s=options.warmup_timeout_s,
            device=options.device,
            encoder_backend=options.encoder_backend,
            encoder_bitrate_bps=options.encoder_bitrate_bps,
            encoder_gop=options.encoder_gop,
        ),
        fps=int(metadata.fps),
        identity=metadata.model_id,
        shared_host=RuntimeHost(runtime),
        shared_spec=spec,
        shared_scenario=scenario,
        shared_model_input_provider_factory=create_model_input_provider,
        client_liveness_timeout_s=options.client_liveness_timeout_s,
    )
    return serve_webrtc_demo(
        output=output,
        model_id=metadata.model_id,
        session_manager=manager,
        app_resources=WebRTCAppResources(preload_name=metadata.model_id),
        world_rank=world_rank,
    )
