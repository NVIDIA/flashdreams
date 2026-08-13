# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Launch modes for the triangle application."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from pathlib import Path

from flashdreams.infra.runner import RunnerConfig
from flashdreams.runtime import InferenceConfig
from flashdreams.runtime.demo import (
    DemoSpec,
    Mp4OutputSpec,
    NativeWindowOutputSpec,
    NullOutputSpec,
    RuntimeHost,
    WebRTCAppResources,
    WebRTCOutputSpec,
)
from flashdreams.runtime.demo.replay import run_replay_demo
from flashdreams.serving.launch import LaunchMode, LaunchOptions, ResolvedLaunch
from flashdreams.serving.native_window import run_native_window_demo
from flashdreams.serving.webrtc.demo import serve_webrtc_demo
from flashdreams.serving.webrtc.manager import BaseWebRTCSessionManager
from triangle_app import (
    TRIANGLE_OUTPUT_MODES,
    TriangleApp,
    TriangleOutputMode,
    TriangleScenario,
    resolve_triangle_model,
)

from .runner import TriangleAppRunnerConfig


@dataclass(frozen=True, slots=True)
class TriangleWebRTCConfig:
    video_width: int
    video_height: int
    warmup_chunks: int = 0
    warmup_timeout_s: float = 30.0


class TriangleAppLaunchCapability:
    def supported_modes(
        self,
        config: RunnerConfig,
        options: LaunchOptions,
    ) -> tuple[LaunchMode, ...]:
        del config, options
        return TRIANGLE_OUTPUT_MODES

    def resolve(
        self,
        config: RunnerConfig,
        *,
        mode: LaunchMode,
        options: LaunchOptions,
    ) -> ResolvedLaunch | None:
        typed_mode = _mode(mode)
        if typed_mode is None:
            return None
        typed_config = _config(config)
        _validate_options(typed_mode, options)
        return ResolvedLaunch(
            mode=typed_mode,
            label=f"Triangle app {typed_mode}",
            summary={
                "application": typed_config.runner_name,
                "model": typed_config.model,
                "mode": typed_mode,
                "frames": typed_config.total_frames,
            },
            launch=partial(
                launch_triangle_app,
                typed_config,
                mode=typed_mode,
                options=options,
            ),
        )


def launch_triangle_app(
    config: TriangleAppRunnerConfig,
    *,
    mode: TriangleOutputMode = "local-window",
    options: LaunchOptions | None = None,
) -> object:
    options = options or LaunchOptions()
    adapter = resolve_triangle_model(config.model)
    spec = _spec(config, adapter=adapter, mode=mode, options=options)
    if mode == "mp4" or mode == "null":
        result = run_replay_demo(spec=spec, adapter=adapter)
        if result.status != "completed":
            reason = result.reason or str(result.error) or result.status
            raise RuntimeError(f"Triangle replay failed: {reason}")
        return result
    if mode == "local-window":
        return run_native_window_demo(spec=spec, adapter=adapter)
    return _serve_webrtc(spec=spec, adapter=adapter)


def _spec(
    config: TriangleAppRunnerConfig,
    *,
    adapter: TriangleApp,
    mode: TriangleOutputMode,
    options: LaunchOptions,
) -> DemoSpec:
    scenario = TriangleScenario(
        width=config.width,
        height=config.height,
        fps=config.fps,
        total_frames=config.total_frames,
    )
    return DemoSpec(
        model_id=adapter.model_id,
        input_mode="replay" if mode in {"mp4", "null"} else "keyboard-driving",
        output=_output(config, mode=mode, options=options),
        scenario=scenario,
        config=InferenceConfig(model_id=adapter.model_id, device=config.device),
    )


def _output(
    config: TriangleAppRunnerConfig,
    *,
    mode: TriangleOutputMode,
    options: LaunchOptions,
) -> Mp4OutputSpec | NativeWindowOutputSpec | NullOutputSpec | WebRTCOutputSpec:
    if mode == "null":
        return NullOutputSpec()
    if mode == "mp4":
        path = options.output.get("path", options.output.get("output", config.output))
        return Mp4OutputSpec(path=Path(str(path)), fps=config.fps, output_layout="tchw")
    if mode == "local-window":
        return NativeWindowOutputSpec(
            fps=config.fps,
            video_width=config.width,
            video_height=config.height,
            title=config.title,
            max_queued_chunks=config.max_queued_chunks,
            close_timeout_s=config.close_timeout_s,
        )
    return WebRTCOutputSpec(
        host=str(options.host or "0.0.0.0"),
        port=options.port or 8080,
        fps=config.fps,
        video_width=config.width,
        video_height=config.height,
    )


def _serve_webrtc(*, spec: DemoSpec, adapter: TriangleApp) -> object:
    output = spec.output
    if not isinstance(output, WebRTCOutputSpec):
        raise TypeError("WebRTC launch requires WebRTCOutputSpec.")
    if spec.config is None:
        raise RuntimeError("DemoSpec.config was not initialized.")
    scenario = adapter.prepare_scenario(spec)
    runtime = adapter.create_runtime(spec.config)
    host = RuntimeHost(runtime)
    manager = BaseWebRTCSessionManager(
        runtime=runtime,
        runtime_config=TriangleWebRTCConfig(
            video_width=output.video_width,
            video_height=output.video_height,
        ),
        fps=output.fps,
        identity=adapter.model_id,
        client_liveness_timeout_s=output.client_liveness_timeout_s,
        supported_control_keys=frozenset({"r", "g", "b", "space"}),
        shared_host=host,
        shared_adapter=adapter,
        shared_spec=spec,
        shared_scenario=scenario,
    )
    return serve_webrtc_demo(
        output=output,
        model_id=adapter.model_id,
        session_manager=manager,
        app_resources=WebRTCAppResources(preload_name="Triangle Model"),
        world_rank=0,
    )


def _validate_options(mode: TriangleOutputMode, options: LaunchOptions) -> None:
    if options.scenario:
        raise ValueError(
            "Triangle model uses typed runner arguments, not scenario overrides."
        )
    allowed_output = {"path", "output"} if mode == "mp4" else set()
    unknown = sorted(set(options.output) - allowed_output)
    if unknown:
        raise ValueError(f"Unsupported output fields: {', '.join(unknown)}.")


def _config(config: RunnerConfig) -> TriangleAppRunnerConfig:
    if not isinstance(config, TriangleAppRunnerConfig):
        raise TypeError("Triangle app launch requires its runner config.")
    return config


def _mode(mode: LaunchMode) -> TriangleOutputMode | None:
    match mode:
        case "mp4" | "null" | "webrtc" | "local-window":
            return mode
        case _:
            return None


LAUNCH_CAPABILITY = TriangleAppLaunchCapability()

__all__ = [
    "LAUNCH_CAPABILITY",
    "TriangleAppLaunchCapability",
    "launch_triangle_app",
]
