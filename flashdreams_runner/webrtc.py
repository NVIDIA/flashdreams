# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""WebRTC I/O mode for application runtimes."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol, cast

from flashdreams.runtime import (
    InferenceConfig,
    InferenceInput,
    InferenceRuntime,
    OutputArtifact,
)
from flashdreams.runtime.demo import (
    DemoSpec,
    ModelInputProvider,
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


@dataclass(slots=True)
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

    _customization: WebRTCCustomization | None = field(
        default=None,
        init=False,
        repr=False,
    )

    def customize(self, customization: WebRTCCustomization) -> None:
        """Install application-owned WebRTC behavior before serving starts."""
        if self._customization is not None:
            raise RuntimeError("WebRTC mode customization is already installed.")
        self._customization = customization

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
            customization=self._customization,
        )
        return ()


class _InputProvider:
    """Provide prepared session input and empty per-step transport input."""

    capabilities = ProviderCapabilities(
        supports_realtime_clock=True,
        deterministic_given_inputs=True,
    )

    def __init__(self, initial_input: InferenceInput) -> None:
        self._initial_input = initial_input

    def prepare_initial_input(self) -> InferenceInput:
        """Return the application-provided input for a new session."""
        return self._initial_input

    def prepare_step(
        self, *, request: StepRequirements, user_window: UserInputWindow
    ) -> PreparedStep:
        """Return one empty step input for a non-interactive T2V session."""
        del request, user_window
        return PreparedStep(inference_input=InferenceInput())

    def reset(self, inputs: InferenceInput | None = None) -> None:
        """Replace the initial input when the shared driver requests a reset."""
        if inputs is not None:
            self._initial_input = inputs

    def close(self) -> None:
        """Release provider resources."""


ModelInputProviderFactory = Callable[
    [DemoSpec, PreparedScenario],
    ModelInputProvider,
]
"""Factory used by shared WebRTC drivers to prepare application input."""


class WebRTCCustomization(Protocol):
    """Application-owned WebRTC UI and session-manager extension point.

    Applications only implement this interface when the generic WebRTC viewer
    is insufficient. The runner continues to own the server and transport.
    """

    def prepare_initial_input(self) -> InferenceInput:
        """Return the initial input used for the first browser generation."""
        ...

    def create_session_manager(
        self,
        *,
        runtime: Runtime,
        output: WebRTCOutputSpec,
        spec: DemoSpec,
        scenario: PreparedScenario,
        input_provider_factory: ModelInputProviderFactory,
    ) -> BaseWebRTCSessionManager[Any, Any]:
        """Create the transport manager used by the customized application."""
        ...

    def create_app_resources(
        self,
        *,
        session_manager: BaseWebRTCSessionManager[Any, Any],
    ) -> WebRTCAppResources:
        """Return packaged browser assets and optional HTTP routes."""
        ...


def serve_webrtc(
    *,
    runtime: Runtime,
    host: str,
    port: int,
    device: str,
    world_rank: int,
    customization: WebRTCCustomization | None = None,
) -> object:
    """Serve an initialized application runtime through shared WebRTC.

    Args:
        runtime: Initialized application runtime.
        host: Server bind address.
        port: Server bind port.
        device: Device used by the runtime.
        world_rank: Distributed rank responsible for presentation.
        customization: Optional application-owned UI and manager behavior.

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
    initial_input = (
        InferenceInput()
        if customization is None
        else customization.prepare_initial_input()
    )
    scenario = PreparedScenario(initial_inputs=initial_input)

    def create_model_input_provider(
        spec: DemoSpec,
        scenario: PreparedScenario,
    ) -> _InputProvider:
        del spec
        return _InputProvider(scenario.initial_inputs)

    inference_runtime = cast(InferenceRuntime, runtime)
    if customization is None:
        manager: BaseWebRTCSessionManager[Any, Any] = BaseWebRTCSessionManager(
            runtime=inference_runtime,
            runtime_config=cast(WebRTCRuntimeConfig, cast(object, output)),
            fps=int(config.fps),
            identity=config.model_id,
            shared_host=RuntimeHost(inference_runtime),
            shared_spec=spec,
            shared_scenario=scenario,
            shared_model_input_provider_factory=create_model_input_provider,
            client_liveness_timeout_s=output.client_liveness_timeout_s,
            runtime_ready=True,
        )
        app_resources = WebRTCAppResources(preload_name=config.model_id)
    else:
        manager = customization.create_session_manager(
            runtime=runtime,
            output=output,
            spec=spec,
            scenario=scenario,
            input_provider_factory=create_model_input_provider,
        )
        app_resources = customization.create_app_resources(
            session_manager=manager,
        )
    return serve_webrtc_demo(
        output=output,
        model_id=config.model_id,
        session_manager=manager,
        app_resources=app_resources,
        world_rank=world_rank,
    )


__all__ = [
    "ModelInputProviderFactory",
    "WebRTCCustomization",
    "WebRTCMode",
    "serve_webrtc",
]
