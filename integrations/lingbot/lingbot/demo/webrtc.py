# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lingbot WebRTC hooks for the shared demo API."""

from __future__ import annotations

from collections.abc import Callable
from importlib.resources import files
from typing import Any

from flashdreams.runtime import InferenceConfig
from flashdreams.runtime.demo import DemoSpec, WebRTCAppResources, WebRTCOutputSpec
from flashdreams.serving.webrtc.manager import BaseWebRTCSessionManager
from lingbot.runtime import (
    LingbotModelAdapter,
    build_lingbot_webrtc_runtime_config,
)
from lingbot.webrtc.server import configure_lingbot_webrtc_app
from lingbot.webrtc.session import (
    LingbotInferenceRuntime,
    LingbotRuntimeConfig,
    create_lingbot_webrtc_session_manager,
)

from .spec import resolve_webrtc_scenario

WebRTCRuntimeFactory = Callable[..., Any]


class LingbotWebRTCIntegration:
    """Bind Lingbot model behavior to the shared WebRTC transport."""

    def __init__(
        self,
        *,
        runtime_factory: WebRTCRuntimeFactory = LingbotInferenceRuntime,
        model_adapter: LingbotModelAdapter | None = None,
    ) -> None:
        self._runtime_factory = runtime_factory
        self._model_adapter = model_adapter or LingbotModelAdapter()

    def supported_input_modes(self) -> tuple[str, ...]:
        return ("keyboard-driving",)

    def create_runtime(self, spec: DemoSpec) -> Any:
        return self._runtime_factory(config=self.create_runtime_config(spec))

    def create_runtime_config(self, spec: DemoSpec) -> LingbotRuntimeConfig:
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
        self._model_adapter.validate_config(config)
        scenario = resolve_webrtc_scenario(spec.scenario)
        compile_network = (
            bool(config.compile)
            if config.compile is not None
            else bool(_option(config, "compile_network", True))
        )
        return build_lingbot_webrtc_runtime_config(
            preset_id=self._model_adapter.preset_id(config),
            pipeline_config=self._model_adapter.pipeline_config(config),
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

    def create_session_manager(
        self,
        *,
        spec: DemoSpec,
        runtime: Any,
    ) -> BaseWebRTCSessionManager[LingbotInferenceRuntime, LingbotRuntimeConfig]:
        if not isinstance(spec.output, WebRTCOutputSpec):
            raise ValueError("Lingbot WebRTC requires WebRTC output.")
        runtime_config = getattr(runtime, "config", None)
        if not isinstance(runtime_config, LingbotRuntimeConfig):
            raise TypeError("Lingbot WebRTC runtime must expose LingbotRuntimeConfig.")
        return create_lingbot_webrtc_session_manager(
            runtime=runtime,
            runtime_config=runtime_config,
            fps=spec.output.fps,
            client_liveness_timeout_s=spec.output.client_liveness_timeout_s,
        )

    def app_resources(self, spec: DemoSpec) -> WebRTCAppResources:
        """Return Lingbot assets and routes for the shared WebRTC app."""
        del spec
        return WebRTCAppResources(
            model_web_resource=files("lingbot.webrtc").joinpath("web"),
            preload_name="Lingbot",
            configure_app=configure_lingbot_webrtc_app,
        )


def _option(config: InferenceConfig, name: str, default: Any) -> Any:
    return config.runtime_options.get(name, default)


__all__ = [
    "LingbotWebRTCIntegration",
    "WebRTCRuntimeFactory",
]
