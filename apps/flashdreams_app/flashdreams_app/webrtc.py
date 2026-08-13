# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Host-owned WebRTC presentation for application runtimes."""

from __future__ import annotations

from dataclasses import dataclass

from flashdreams.runtime import InferenceConfig, InferenceInput, InferenceRuntime
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

from .contracts import AppConfig


# TODO: Move this contract into shared FlashDreams and expand it when the
# serving API needs caller-configurable transport and encoder tuning.
@dataclass(frozen=True, slots=True)
class WebRTCOptions:
    """Minimal WebRTC bind settings for the application prototype."""

    host: str
    """Server bind address."""

    port: int
    """Server bind port."""


class _InputProvider:
    capabilities = ProviderCapabilities(
        supports_realtime_clock=True,
        deterministic_given_inputs=True,
    )

    def __init__(self, initial_input: InferenceInput) -> None:
        self._initial_input = initial_input

    def prepare_initial_input(self) -> InferenceInput:
        return self._initial_input

    def prepare_step(
        self, *, request: StepRequirements, user_window: UserInputWindow
    ) -> PreparedStep:
        del request, user_window
        return PreparedStep(inference_input=InferenceInput())

    def reset(self, inputs: InferenceInput | None = None) -> None:
        del inputs

    def close(self) -> None:
        pass


def serve_webrtc(
    *,
    runtime: InferenceRuntime,
    config: AppConfig,
    initial_input: InferenceInput,
    options: WebRTCOptions,
    device: str,
    world_rank: int,
) -> object:
    """Serve an application runtime through the shared WebRTC stack.

    Args:
        runtime: Initialized application runtime.
        config: Model identity and video presentation configuration.
        initial_input: Global conditioning used to start live sessions.
        options: WebRTC bind settings.
        device: Device used by the runtime.
        world_rank: Distributed rank responsible for presentation.

    Returns:
        Serving backend result.
    """
    output = WebRTCOutputSpec(
        host=options.host,
        port=options.port,
        fps=int(config.fps),
        video_width=config.video_width,
        video_height=config.video_height,
        preload_name=config.model_id,
    )
    spec = DemoSpec(
        model_id=config.model_id,
        input_mode="webrtc",
        output=output,
        config=InferenceConfig(model_id=config.model_id, device=device),
    )
    scenario = PreparedScenario(initial_inputs=initial_input)

    def create_model_input_provider(
        spec: DemoSpec,
        scenario: PreparedScenario,
    ) -> _InputProvider:
        del spec, scenario
        return _InputProvider(initial_input)

    manager = BaseWebRTCSessionManager(
        runtime=runtime,
        runtime_config=output,
        fps=int(config.fps),
        identity=config.model_id,
        shared_host=RuntimeHost(runtime),
        shared_spec=spec,
        shared_scenario=scenario,
        shared_model_input_provider_factory=create_model_input_provider,
        client_liveness_timeout_s=output.client_liveness_timeout_s,
    )
    return serve_webrtc_demo(
        output=output,
        model_id=config.model_id,
        session_manager=manager,
        app_resources=WebRTCAppResources(preload_name=config.model_id),
        world_rank=world_rank,
    )
