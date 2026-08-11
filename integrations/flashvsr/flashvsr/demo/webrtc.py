# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native FlashVSR runtime hooks for the shared WebRTC server."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from importlib.resources import files
from typing import Any, Literal

from loguru import logger

from flashdreams.runtime.demo import (
    DemoSpec,
    PreparedScenario,
    RuntimeHost,
    WebRTCAppResources,
    WebRTCOutputSpec,
)
from flashdreams.serving.webrtc.bootstrap import run_webrtc_server
from flashdreams.serving.webrtc.demo import (
    CreateWebRTCApp,
    RunWebRTCServer,
    serve_webrtc_demo,
)
from flashdreams.serving.webrtc.manager import BaseWebRTCSessionManager
from flashdreams.serving.webrtc.server import create_webrtc_app
from flashvsr.runtime import FlashVSRInferenceRuntime

from .adapter import FlashVSRDemoAdapter
from .providers import PREPARED_VIDEO_METADATA_KEY
from .server import (
    FlashVSRUploadController,
    FlashVSRWebRTCSessionInput,
    configure_flashvsr_webrtc_app,
)
from .spec import PreparedFlashVSRVideo, resolve_video_scenario

RuntimeFactory = Callable[..., Any]


@dataclass(frozen=True, kw_only=True, slots=True)
class FlashVSRWebRTCRuntimeConfig:
    """Transport facts consumed by the shared WebRTC session manager."""

    pipeline_config_name: str
    device: str
    video_height: int
    video_width: int
    fps: int
    warmup_chunks: int
    warmup_timeout_s: float
    encoder_backend: Literal["auto", "default", "nvenc"] = "auto"
    encoder_bitrate_bps: int = 6_000_000
    encoder_gop: int = 30


def serve_flashvsr_webrtc_demo(
    *,
    spec: DemoSpec,
    world_rank: int = 0,
    runtime_factory: RuntimeFactory = FlashVSRInferenceRuntime,
    create_app_fn: CreateWebRTCApp = create_webrtc_app,
    server_runner: RunWebRTCServer = run_webrtc_server,
) -> object:
    """Serve uploaded or server-side videos through the native runtime."""
    if spec.input_mode != "replay":
        raise ValueError("FlashVSR WebRTC requires input_mode='replay'.")
    if not isinstance(spec.output, WebRTCOutputSpec):
        raise ValueError("FlashVSR WebRTC requires WebRTC output.")
    if spec.config is None:
        raise RuntimeError("DemoSpec.config was not initialized.")

    adapter = FlashVSRDemoAdapter(runtime_factory=runtime_factory)
    scenario = resolve_video_scenario(spec.scenario)
    default_scenario: PreparedScenario | None = None
    default_video: PreparedFlashVSRVideo | None = None
    if scenario.input_path is not None:
        default_scenario = adapter.prepare_scenario(spec)
        candidate = default_scenario.metadata.get(PREPARED_VIDEO_METADATA_KEY)
        if not isinstance(candidate, PreparedFlashVSRVideo):
            raise TypeError("Prepared FlashVSR scenario is missing its decoded video.")
        default_video = candidate

    output = spec.output
    if default_scenario is not None:
        output = replace(
            output,
            video_height=int(default_scenario.metadata["target_height"]),
            video_width=int(default_scenario.metadata["target_width"]),
        )
    runtime_options = dict(spec.config.runtime_options)
    runtime_options.update(
        {
            "fps": output.fps,
            "chunk_size": scenario.chunk_size,
        }
    )
    config = replace(spec.config, runtime_options=runtime_options)
    shared_spec = replace(
        spec,
        scenario=scenario,
        output=output,
        config=config,
    )
    runtime = adapter.create_runtime(config)
    host = RuntimeHost(runtime)
    preset_id = adapter.preset_id(config)
    startup_warmup_chunks = output.warmup_chunks if default_scenario is not None else 0
    if default_scenario is None and output.warmup_chunks > 0:
        logger.info(
            "No startup FlashVSR input; deferring model construction until upload."
        )
    runtime_config = FlashVSRWebRTCRuntimeConfig(
        pipeline_config_name=preset_id,
        device=config.device or "cuda:0",
        video_height=output.video_height,
        video_width=output.video_width,
        fps=output.fps,
        warmup_chunks=startup_warmup_chunks,
        warmup_timeout_s=output.warmup_timeout_s,
        # Browser uploads can change resolution between sessions. aiortc's
        # software path accepts that; NVENC is bound to its startup dimensions.
        encoder_backend="default",
        encoder_gop=output.fps,
    )
    manager = BaseWebRTCSessionManager(
        runtime=runtime,
        runtime_config=runtime_config,
        fps=output.fps,
        identity=preset_id,
        busy_message="A FlashVSR session is already active.",
        warmup_label="FlashVSR WebRTC",
        fatal_generation_errors=True,
        client_liveness_timeout_s=output.client_liveness_timeout_s,
        shared_host=host,
        shared_adapter=adapter,
        shared_spec=shared_spec,
        shared_spec_factory=lambda session_input: _uploaded_session_spec(
            shared_spec,
            session_input=session_input,
        ),
        shared_scenario=default_scenario,
    )
    upload_controller = FlashVSRUploadController(
        manager=manager,
        adapter=adapter,
        spec=shared_spec,
        default_video=default_video,
    )
    return serve_webrtc_demo(
        output=output,
        model_id=spec.model_id,
        session_manager=manager,
        app_resources=WebRTCAppResources(
            model_web_resource=files("flashvsr.demo").joinpath("web"),
            preload_name="FlashVSR",
            configure_app=lambda app: configure_flashvsr_webrtc_app(
                app,
                controller=upload_controller,
            ),
        ),
        world_rank=world_rank,
        create_app_fn=create_app_fn,
        server_runner=server_runner,
    )


def _uploaded_session_spec(
    spec: DemoSpec,
    *,
    session_input: Any,
) -> DemoSpec:
    if not isinstance(session_input, FlashVSRWebRTCSessionInput):
        raise TypeError("FlashVSR WebRTC requires a decoded video upload.")
    if not isinstance(spec.output, WebRTCOutputSpec):
        raise ValueError("FlashVSR WebRTC requires WebRTC output.")
    if spec.config is None:
        raise RuntimeError("DemoSpec.config was not initialized.")
    prepared = session_input.prepared_video
    output = replace(
        spec.output,
        video_height=prepared.target_height,
        video_width=prepared.target_width,
    )
    runtime_options = dict(spec.config.runtime_options)
    runtime_options.update(
        {
            "fps": output.fps,
            "chunk_size": prepared.scenario.chunk_size,
        }
    )
    return replace(
        spec,
        scenario=prepared,
        output=output,
        config=replace(spec.config, runtime_options=runtime_options),
    )


__all__ = [
    "FlashVSRWebRTCRuntimeConfig",
    "RuntimeFactory",
    "serve_flashvsr_webrtc_demo",
]
