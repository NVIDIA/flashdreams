# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""WebRTC I/O mode for application runtimes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

from flashdreams.runtime import (
    InferenceConfig,
    InferenceInput,
    InferenceRuntime,
    OutputArtifact,
)
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
from flashdreams.serving.webrtc.runtime import WebRTCRuntimeConfig

from .contracts import DriveSession, Runtime


@dataclass(frozen=True, slots=True)
class WebRTCMode:
    """Serve application sessions over WebRTC."""

    host: str
    """Server bind address."""

    port: int
    """Server bind port."""

    device: str
    """Device used by the application runtime."""

    world_rank: int
    """Distributed rank responsible for presentation."""

    name: str = "webrtc"
    """Stable mode name."""

    def run(
        self,
        runtime: Runtime,
        drive_session: DriveSession,
    ) -> tuple[OutputArtifact, ...]:
        """Serve live sessions until the WebRTC server exits."""
        del drive_session
        serve_webrtc(
            runtime=runtime,
            host=self.host,
            port=self.port,
            device=self.device,
            world_rank=self.world_rank,
        )
        return ()


class _InputProvider:
    """Provide empty transport inputs to application-owned sessions."""

    capabilities = ProviderCapabilities(
        supports_realtime_clock=True,
        deterministic_given_inputs=True,
    )

    def prepare_initial_input(self) -> InferenceInput:
        """Let the application runtime apply its session defaults."""
        return InferenceInput()

    def prepare_step(
        self, *, request: StepRequirements, user_window: UserInputWindow
    ) -> PreparedStep:
        """Return one empty step input for a non-interactive T2V session."""
        del request, user_window
        return PreparedStep(inference_input=InferenceInput())

    def reset(self, inputs: InferenceInput | None = None) -> None:
        """Discard reset inputs because resets create fresh app sessions."""
        del inputs

    def close(self) -> None:
        """Release provider resources."""


def serve_webrtc(
    *,
    runtime: Runtime,
    host: str,
    port: int,
    device: str,
    world_rank: int,
) -> object:
    """Serve an initialized application runtime through shared WebRTC.

    Args:
        runtime: Initialized application runtime.
        host: Server bind address.
        port: Server bind port.
        device: Device used by the runtime.
        world_rank: Distributed rank responsible for presentation.

    Returns:
        Serving backend result.
    """
    config = runtime.config
    output = WebRTCOutputSpec(
        host=host,
        port=port,
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
    scenario = PreparedScenario(initial_inputs=InferenceInput())

    def create_model_input_provider(
        spec: DemoSpec,
        scenario: PreparedScenario,
    ) -> _InputProvider:
        del spec, scenario
        return _InputProvider()

    inference_runtime = cast(InferenceRuntime, runtime)
    manager = BaseWebRTCSessionManager(
        runtime=inference_runtime,
        runtime_config=cast(WebRTCRuntimeConfig, cast(object, output)),
        fps=int(config.fps),
        identity=config.model_id,
        shared_host=RuntimeHost(inference_runtime),
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


__all__ = ["WebRTCMode", "serve_webrtc"]
