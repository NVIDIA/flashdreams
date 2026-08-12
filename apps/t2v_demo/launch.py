# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Launch capability that routes ``flashdreams-run t2v`` to the T2V app."""

from __future__ import annotations

from importlib.resources import files
from pathlib import Path
from typing import Literal, cast

from flashdreams.infra.runner import RunnerConfig
from flashdreams.runtime import InferenceConfig
from flashdreams.runtime.demo import (
    DemoSpec,
    Mp4OutputSpec,
    NullOutputSpec,
    WebRTCAppResources,
    WebRTCOutputSpec,
)
from flashdreams.runtime.demo.bootstrap import (
    configure_logging,
    initialize_cuda_distributed,
)
from flashdreams.runtime.demo.host import RuntimeHost
from flashdreams.runtime.demo.replay import run_replay_demo
from flashdreams.serving.launch import CallbackLaunchCapability, LaunchOptions
from flashdreams.serving.webrtc.demo import serve_webrtc_demo
from flashdreams.serving.webrtc.runtime import WebRTCSessionConfig

from .app import T2VWebRTCSessionManager, _configure_app
from .backends import resolve_backend
from .runtime import (
    FIELD_FPS,
    FIELD_PIXEL_HEIGHT,
    FIELD_PIXEL_WIDTH,
    FIELD_PROMPT,
    FIELD_TOTAL_BLOCKS,
    make_adapter,
)


def launch_t2v(
    config: RunnerConfig,
    mode: Literal["mp4", "null", "webrtc"],
    options: LaunchOptions,
) -> object:
    """Delegate parsed ``flashdreams-run`` arguments to the selected demo mode."""
    from .runner import T2VDemoRunnerConfig

    config = cast(T2VDemoRunnerConfig, config)
    configure_logging()
    adapter = make_adapter(config.backend)
    preset = resolve_backend(config.backend).resolve_runner(config.preset_id)
    scenario = {
        name: options.scenario.get(name, getattr(config, name) or default)
        for name, default in (
            (FIELD_PROMPT, preset.prompt),
            (FIELD_TOTAL_BLOCKS, preset.total_blocks),
            (FIELD_PIXEL_HEIGHT, preset.pixel_height),
            (FIELD_PIXEL_WIDTH, preset.pixel_width),
            (FIELD_FPS, preset.fps),
        )
    }
    preset_id = config.preset_id or adapter.backend.default_preset_name
    if mode in {"mp4", "null"}:
        output = (
            NullOutputSpec()
            if mode == "null"
            else Mp4OutputSpec(
                path=Path(
                    str(
                        options.output.get(
                            "path", options.output.get("output", config.output)
                        )
                    )
                ),
                fps=int(options.output.get("fps", scenario[FIELD_FPS])),
                output_layout="tchw",
            )
        )
        result = run_replay_demo(
            spec=DemoSpec(
                model_id=adapter.model_id,
                preset_id=preset_id,
                input_mode="replay",
                scenario=scenario,
                output=output,
                config=InferenceConfig(
                    model_id=adapter.model_id,
                    preset_id=preset_id,
                    device=config.device,
                    compile=config.compile,
                    runtime_options={"backend": adapter.backend.key},
                ),
            ),
            adapter=adapter,
        )
        if result.status != "completed":
            raise RuntimeError(
                result.reason or str(result.error) or "T2V replay failed."
            )
        return result

    context = initialize_cuda_distributed(default_device=config.device)
    output = WebRTCOutputSpec(
        host=str(options.host or options.output.get("host", "0.0.0.0")),
        port=int(options.port or options.output.get("port", 8080)),
        fps=int(options.output.get("fps", scenario[FIELD_FPS])),
        video_width=int(options.output.get("video_width", scenario[FIELD_PIXEL_WIDTH])),
        video_height=int(
            options.output.get("video_height", scenario[FIELD_PIXEL_HEIGHT])
        ),
        warmup_chunks=int(options.output.get("warmup_chunks", 0)),
        warmup_timeout_s=float(options.output.get("warmup_timeout_s", 600.0)),
        client_liveness_timeout_s=float(
            options.output.get("client_liveness_timeout_s", 30.0)
        ),
        preload_name="FlashDreams T2V",
    )
    spec = DemoSpec(
        model_id=adapter.model_id,
        preset_id=preset_id,
        input_mode="webrtc",
        scenario=scenario,
        output=output,
        config=InferenceConfig(
            model_id=adapter.model_id,
            preset_id=preset_id,
            device=str(context.device),
            compile=config.compile,
            runtime_options={"backend": adapter.backend.key},
        ),
    )
    prepared = adapter.prepare_scenario(spec)
    runtime = adapter.create_runtime(cast(InferenceConfig, spec.config))
    manager = T2VWebRTCSessionManager(
        runtime=runtime,
        runtime_config=WebRTCSessionConfig.from_output(output),
        fps=output.fps,
        identity=adapter.model_id,
        supported_control_keys=frozenset({"g"}),
        shared_host=RuntimeHost(runtime),
        shared_adapter=adapter,
        shared_spec=spec,
        shared_scenario=prepared,
        client_liveness_timeout_s=output.client_liveness_timeout_s,
        keep_connection_after_completed=True,
    )
    return serve_webrtc_demo(
        output=output,
        model_id=adapter.model_id,
        session_manager=manager,
        app_resources=WebRTCAppResources(
            model_web_resource=files("t2v_demo").joinpath("web"),
            configure_app=lambda app: _configure_app(
                app, manager=manager, backend=config.backend
            ),
            preload_name="FlashDreams T2V",
        ),
        world_rank=context.world_rank,
    )


LAUNCH_CAPABILITY = CallbackLaunchCapability(
    label="T2V",
    modes=("mp4", "null", "webrtc"),
    launch=launch_t2v,
)

__all__ = ["LAUNCH_CAPABILITY", "launch_t2v"]
