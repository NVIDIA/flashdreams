# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lingbot WebRTC hooks for the shared demo API."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from importlib.resources import files
from typing import Any

from flashdreams.runtime import InferenceConfig
from flashdreams.runtime.demo import DemoSpec, WebRTCAppResources, WebRTCOutputSpec
from flashdreams.serving.webrtc.bootstrap import run_webrtc_server
from flashdreams.serving.webrtc.demo import (
    CreateWebRTCApp,
    RunWebRTCServer,
    serve_webrtc_demo,
)
from flashdreams.serving.webrtc.manager import BaseWebRTCSessionManager
from flashdreams.serving.webrtc.server import create_webrtc_app
from lingbot.demo import LingbotDemoAdapter
from lingbot.runtime import (
    FIELD_FPS,
    FIELD_PIXEL_HEIGHT,
    FIELD_PIXEL_WIDTH,
    FIELD_PROMPT,
    FIELD_TOTAL_BLOCKS,
    LingbotModelAdapter,
    build_lingbot_webrtc_runtime_config,
)
from lingbot.webrtc.server import configure_lingbot_webrtc_app
from lingbot.webrtc.session import (
    LingbotInferenceRuntime,
    LingbotRuntimeConfig,
)

from .spec import resolve_webrtc_scenario

WebRTCRuntimeFactory = Callable[..., Any]


def serve_lingbot_webrtc_demo(
    *,
    spec: DemoSpec,
    world_rank: int = 0,
    runtime_factory: WebRTCRuntimeFactory = LingbotInferenceRuntime,
    model_adapter: LingbotModelAdapter | None = None,
    create_app_fn: CreateWebRTCApp = create_webrtc_app,
    server_runner: RunWebRTCServer = run_webrtc_server,
) -> object:
    """Create Lingbot's runtime and serve it through the shared WebRTC transport."""
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
    model_adapter = model_adapter or LingbotModelAdapter()
    model_adapter.validate_config(config)
    scenario = resolve_webrtc_scenario(spec.scenario)
    compile_network = (
        bool(config.compile)
        if config.compile is not None
        else bool(_option(config, "compile_network", True))
    )
    runtime_config = build_lingbot_webrtc_runtime_config(
        preset_id=model_adapter.preset_id(config),
        pipeline_config=model_adapter.pipeline_config(config),
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
    runtime = runtime_factory(config=runtime_config)
    demo_adapter = LingbotDemoAdapter()
    shared_spec = _shared_webrtc_spec(
        spec,
        runtime_config=runtime_config,
        example_idx=scenario.example_idx,
    )
    prepared = demo_adapter.prepare_scenario(shared_spec)
    manager = BaseWebRTCSessionManager(
        runtime=runtime,
        runtime_config=runtime_config,
        fps=spec.output.fps,
        identity=runtime_config.config_name,
        busy_message="A Lingbot session is already active.",
        warmup_label="Lingbot WebRTC",
        client_liveness_timeout_s=spec.output.client_liveness_timeout_s,
        shared_adapter=demo_adapter,
        shared_spec=shared_spec,
        shared_spec_factory=lambda session_input: _shared_webrtc_spec(
            spec,
            runtime_config=runtime_config,
            example_idx=scenario.example_idx,
            session_input=session_input,
        ),
        shared_scenario=prepared,
    )
    return serve_webrtc_demo(
        output=spec.output,
        model_id=spec.model_id,
        session_manager=manager,
        app_resources=WebRTCAppResources(
            model_web_resource=files("lingbot.webrtc").joinpath("web"),
            preload_name="Lingbot",
            configure_app=configure_lingbot_webrtc_app,
        ),
        world_rank=world_rank,
        create_app_fn=create_app_fn,
        server_runner=server_runner,
    )


def _option(config: InferenceConfig, name: str, default: Any) -> Any:
    return config.runtime_options.get(name, default)


def _shared_webrtc_spec(
    spec: DemoSpec,
    *,
    runtime_config: LingbotRuntimeConfig,
    example_idx: int,
    session_input: Any = None,
) -> DemoSpec:
    config = spec.config
    if config is None:
        raise RuntimeError("DemoSpec.config was not initialized.")
    runtime_options = dict(config.runtime_options)
    runtime_options.update(
        {
            "default_prompt": runtime_config.default_prompt,
            "pipeline_config": runtime_config.pipeline_config,
            "seed": runtime_config.seed,
        }
    )
    scenario: dict[str, Any] = {
        "camera_source": "events",
        "example_data": True,
        "example_idx": example_idx,
        "text_events": runtime_config.text_events,
        FIELD_TOTAL_BLOCKS: int(_option(config, "total_blocks", 1_000_000)),
        FIELD_PIXEL_HEIGHT: runtime_config.video_height,
        FIELD_PIXEL_WIDTH: runtime_config.video_width,
        FIELD_FPS: runtime_config.fps,
    }
    # Browser-provided first-frame payloads stay runtime/session-owned because
    # they can be bytes or remote payloads. The provider only needs the active
    # prompt/catalog plus example-data calibration for live camera mapping.
    prompt = getattr(session_input, "prompt", None)
    if prompt:
        scenario[FIELD_PROMPT] = str(prompt)
    text_events = getattr(session_input, "text_events", None)
    if text_events is not None:
        scenario["text_events"] = text_events
    return replace(
        spec,
        scenario=scenario,
        config=replace(
            config,
            preset_id=runtime_config.config_name,
            device=runtime_config.device,
            seed=runtime_config.seed,
            runtime_options=runtime_options,
        ),
    )


__all__ = [
    "WebRTCRuntimeFactory",
    "serve_lingbot_webrtc_demo",
]
