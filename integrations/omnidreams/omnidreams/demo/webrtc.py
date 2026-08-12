# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""OmniDreams browser hooks for the shared WebRTC demo runtime."""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import replace
from typing import Any

from loguru import logger
from omnidreams.config import OMNIDREAMS_CONFIGS

from flashdreams.runtime import InferenceConfig
from flashdreams.runtime.demo import (
    DemoSpec,
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

from .adapter import OmnidreamsDemoAdapter, RuntimeFactory
from .controls import WSAD_SUPPORTED_KEYS
from .spec import (
    DEFAULT_OMNIDREAMS_PRESET,
    OMNIDREAMS_MODEL_ID,
    resolve_webrtc_scenario,
)
from .webrtc_config import OmnidreamsWebRTCModelRuntimeConfig

WebRTCRuntimeFactory = Callable[..., Any]
SharedRuntimeFactory = RuntimeFactory


def serve_omnidreams_webrtc_demo(
    *,
    spec: DemoSpec,
    world_rank: int = 0,
    runtime_factory: WebRTCRuntimeFactory | None = None,
    shared_runtime_factory: SharedRuntimeFactory | None = None,
    create_app_fn: CreateWebRTCApp = create_webrtc_app,
    server_runner: RunWebRTCServer = run_webrtc_server,
) -> object:
    """Create OmniDreams' runtime and serve it through shared WebRTC transport."""
    if runtime_factory is not None and shared_runtime_factory is not None:
        raise ValueError(
            "Specify either legacy runtime_factory or shared_runtime_factory, not both."
        )
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
    if config.model_id != OMNIDREAMS_MODEL_ID:
        raise ValueError(
            f"OmniDreams WebRTC requires model_id={OMNIDREAMS_MODEL_ID!r}, "
            f"got {config.model_id!r}."
        )

    scenario = resolve_webrtc_scenario(spec.scenario)
    runtime_config = _webrtc_runtime_config(
        output=spec.output,
        config=config,
        scenario=scenario,
    )
    if _should_use_legacy_webrtc_path(
        scenario=scenario,
        runtime_factory=runtime_factory,
    ):
        from .webrtc_legacy import (  # noqa: PLC0415
            OmnidreamsWebRTCModelRuntime,
            _serve_legacy_omnidreams_webrtc_demo,
        )

        return _serve_legacy_omnidreams_webrtc_demo(
            spec=spec,
            output=spec.output,
            runtime_config=runtime_config,
            runtime_factory=runtime_factory or OmnidreamsWebRTCModelRuntime,
            world_rank=world_rank,
            create_app_fn=create_app_fn,
            server_runner=server_runner,
        )

    return _serve_shared_omnidreams_webrtc_demo(
        spec=_shared_webrtc_spec(spec, runtime_config=runtime_config),
        output=spec.output,
        runtime_config=runtime_config,
        shared_runtime_factory=shared_runtime_factory,
        world_rank=world_rank,
        create_app_fn=create_app_fn,
        server_runner=server_runner,
    )


def _webrtc_runtime_config(
    *,
    output: WebRTCOutputSpec,
    config: InferenceConfig,
    scenario: Any,
) -> OmnidreamsWebRTCModelRuntimeConfig:
    preset_id = _preset_id(config)
    seed = _option(config, "seed", 42)
    runtime_config = OmnidreamsWebRTCModelRuntimeConfig(
        pipeline_config_name=preset_id,
        pipeline_config=_pipeline_config(config),
        scene_dir=scenario.scene_dir,
        scene_uuid=scenario.scene_uuid,
        scene_variant=scenario.scene_variant,
        seed=None if seed is None else int(seed),
        device=config.device or str(_option(config, "device", "cuda:0")),
        video_height=output.video_height,
        video_width=output.video_width,
        fps=output.fps,
        camera_name=scenario.camera_name,
        warmup_chunks=output.warmup_chunks,
        warmup_timeout_s=output.warmup_timeout_s,
        debug_serve_hdmaps=scenario.debug_serve_hdmaps,
        encoder_backend="default" if scenario.prefer_sw_encoder else "auto",
    )
    return _apply_runtime_options(runtime_config, config.runtime_options)


def _should_use_legacy_webrtc_path(
    *,
    scenario: Any,
    runtime_factory: WebRTCRuntimeFactory | None,
) -> bool:
    if runtime_factory is not None:
        return True
    if bool(getattr(scenario, "debug_serve_hdmaps", False)):
        logger.info(
            "Using the legacy OmniDreams WebRTC path because debug HDMap "
            "streaming is still implemented by the compatibility facade."
        )
        return True
    if _distributed_world_size() > 1:
        logger.info(
            "Using the legacy OmniDreams WebRTC path for multi-rank serving; "
            "shared RuntimeHost distributed fan-out is not yet complete."
        )
        return True
    return False


def _serve_shared_omnidreams_webrtc_demo(
    *,
    spec: DemoSpec,
    output: WebRTCOutputSpec,
    runtime_config: OmnidreamsWebRTCModelRuntimeConfig,
    shared_runtime_factory: SharedRuntimeFactory | None,
    world_rank: int,
    create_app_fn: CreateWebRTCApp,
    server_runner: RunWebRTCServer,
) -> object:
    adapter = OmnidreamsDemoAdapter(runtime_factory=shared_runtime_factory)
    prepared = adapter.prepare_scenario(spec)
    config = spec.config
    if config is None:
        raise RuntimeError("DemoSpec.config was not initialized.")
    runtime = adapter.create_runtime(config)
    host = RuntimeHost(runtime)
    manager = BaseWebRTCSessionManager(
        runtime=runtime,
        runtime_config=runtime_config,
        fps=runtime_config.fps,
        identity=runtime_config.pipeline_config_name,
        busy_message="An OmniDreams session is already active.",
        warmup_label="OmniDreams WebRTC",
        supported_control_keys=WSAD_SUPPORTED_KEYS,
        fatal_generation_errors=True,
        client_liveness_timeout_s=output.client_liveness_timeout_s,
        shared_host=host,
        shared_adapter=adapter,
        shared_spec=spec,
        shared_scenario=prepared,
    )
    from importlib.resources import files

    return serve_webrtc_demo(
        output=output,
        model_id=spec.model_id,
        session_manager=manager,
        app_resources=WebRTCAppResources(
            model_web_resource=files("omnidreams.demo").joinpath("web"),
            preload_name="OmniDreams",
        ),
        world_rank=world_rank,
        create_app_fn=create_app_fn,
        server_runner=server_runner,
    )


def _shared_webrtc_spec(
    spec: DemoSpec,
    *,
    runtime_config: OmnidreamsWebRTCModelRuntimeConfig,
) -> DemoSpec:
    config = spec.config
    if config is None:
        raise RuntimeError("DemoSpec.config was not initialized.")
    runtime_options = dict(config.runtime_options)
    runtime_options.update(
        {
            "pipeline_config": runtime_config.pipeline_config,
            "seed": runtime_config.seed,
            "move_speed_per_s": runtime_config.move_speed_per_s,
            "rotate_speed_rad_per_s": runtime_config.rotate_speed_rad_per_s,
            "release_oneshot_encoders_after_cache_init": False,
        }
    )
    return replace(
        spec,
        config=replace(
            config,
            preset_id=runtime_config.pipeline_config_name,
            device=runtime_config.device,
            seed=runtime_config.seed,
            runtime_options=runtime_options,
        ),
    )


def _preset_id(config: InferenceConfig | None) -> str:
    return (
        DEFAULT_OMNIDREAMS_PRESET
        if config is None or config.preset_id is None
        else config.preset_id
    )


def _pipeline_config(config: InferenceConfig) -> Any:
    custom = config.runtime_options.get("pipeline_config")
    if custom is not None:
        return custom
    preset_id = _preset_id(config)
    try:
        return OMNIDREAMS_CONFIGS[preset_id]
    except KeyError as exc:
        supported = ", ".join(sorted(OMNIDREAMS_CONFIGS))
        raise ValueError(
            f"Unsupported OmniDreams preset_id={preset_id!r}. "
            f"Supported presets: {supported}."
        ) from exc


def _option(config: InferenceConfig, name: str, default: Any) -> Any:
    return config.runtime_options.get(name, default)


def _distributed_world_size() -> int:
    try:
        return int(os.environ.get("WORLD_SIZE", "1"))
    except ValueError:
        return 1


def _apply_runtime_options(
    runtime_config: OmnidreamsWebRTCModelRuntimeConfig,
    options: Any,
) -> OmnidreamsWebRTCModelRuntimeConfig:
    if not isinstance(options, dict):
        options = dict(options)
    overrides = {
        name: options[name]
        for name in (
            "move_speed_per_s",
            "rotate_speed_rad_per_s",
            "encoder_bitrate_bps",
            "encoder_gop",
        )
        if name in options
    }
    return replace(runtime_config, **overrides) if overrides else runtime_config


__all__ = [
    "OmnidreamsWebRTCModelRuntimeConfig",
    "SharedRuntimeFactory",
    "WebRTCRuntimeFactory",
    "serve_omnidreams_webrtc_demo",
]
