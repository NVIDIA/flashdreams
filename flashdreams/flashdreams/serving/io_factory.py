# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""IO factories for package applications."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

from flashdreams.runtime.demo import (
    DemoSpec,
    Mp4OutputSpec,
    NativeWindowOutputSpec,
    NullOutputSpec,
    OutputSpec,
    RuntimeHost,
    WebRTCAppResources,
    WebRTCOutputSpec,
)
from flashdreams.runtime.demo.application import (
    ApplicationMode,
    FlashDreamsApplication,
)
from flashdreams.runtime.demo.replay import run_replay_demo
from flashdreams.serving.native_window import run_native_window_demo
from flashdreams.serving.webrtc.demo import serve_webrtc_demo
from flashdreams.serving.webrtc.manager import BaseWebRTCSessionManager


@dataclass(frozen=True, slots=True)
class IOOptions:
    output_path: Path | None = None
    host: str | None = None
    port: int | None = None


@runtime_checkable
class IOFactory(Protocol):
    mode: ApplicationMode
    input_mode: str

    def create_output(
        self,
        application: FlashDreamsApplication,
        options: IOOptions,
    ) -> OutputSpec: ...

    def run(
        self,
        application: FlashDreamsApplication,
        spec: DemoSpec,
    ) -> object: ...


@dataclass(frozen=True, slots=True)
class ReplayIOFactory:
    mode: ApplicationMode
    input_mode: str = "replay"

    def create_output(
        self,
        application: FlashDreamsApplication,
        options: IOOptions,
    ) -> Mp4OutputSpec | NullOutputSpec:
        _reject_network_options(options)
        if self.mode == "null":
            if options.output_path is not None:
                raise ValueError("Null output does not accept --output.")
            return NullOutputSpec()
        return Mp4OutputSpec(
            path=options.output_path
            or Path("outputs") / f"{application.application_name}.mp4",
            fps=application.fps,
            output_layout=application.output_layout,
        )

    def run(
        self,
        application: FlashDreamsApplication,
        spec: DemoSpec,
    ) -> object:
        result = run_replay_demo(spec=spec, adapter=application)
        if result.status != "completed":
            reason = result.reason or str(result.error) or result.status
            raise RuntimeError(f"Application replay failed: {reason}")
        return result


class NativeWindowIOFactory:
    mode: ApplicationMode = "local-window"
    input_mode = "keyboard-driving"

    def create_output(
        self,
        application: FlashDreamsApplication,
        options: IOOptions,
    ) -> NativeWindowOutputSpec:
        _reject_all_options(options)
        return NativeWindowOutputSpec(
            fps=application.fps,
            video_width=application.video_width,
            video_height=application.video_height,
            title=application.title or application.application_name,
        )

    def run(
        self,
        application: FlashDreamsApplication,
        spec: DemoSpec,
    ) -> object:
        return run_native_window_demo(spec=spec, adapter=application)


@dataclass(frozen=True, slots=True)
class _WebRTCConfig:
    video_width: int
    video_height: int
    warmup_chunks: int = 0
    warmup_timeout_s: float = 30.0


class WebRTCIOFactory:
    mode: ApplicationMode = "webrtc"
    input_mode = "keyboard-driving"

    def create_output(
        self,
        application: FlashDreamsApplication,
        options: IOOptions,
    ) -> WebRTCOutputSpec:
        if options.output_path is not None:
            raise ValueError("WebRTC output does not accept --output.")
        return WebRTCOutputSpec(
            host=options.host or "0.0.0.0",
            port=options.port or 8080,
            fps=application.fps,
            video_width=application.video_width,
            video_height=application.video_height,
        )

    def run(
        self,
        application: FlashDreamsApplication,
        spec: DemoSpec,
    ) -> object:
        output = spec.output
        if not isinstance(output, WebRTCOutputSpec):
            raise TypeError("WebRTC factory requires WebRTCOutputSpec.")
        scenario = application.prepare_scenario(spec)
        runtime = application.create_runtime(application.config)
        host = RuntimeHost(runtime)
        manager = BaseWebRTCSessionManager(
            runtime=runtime,
            runtime_config=_WebRTCConfig(
                video_width=output.video_width,
                video_height=output.video_height,
            ),
            fps=output.fps,
            identity=application.application_name,
            client_liveness_timeout_s=output.client_liveness_timeout_s,
            supported_control_keys=application.supported_control_keys,
            shared_host=host,
            shared_adapter=application,
            shared_spec=spec,
            shared_scenario=scenario,
        )
        return serve_webrtc_demo(
            output=output,
            model_id=application.model_id,
            session_manager=manager,
            app_resources=WebRTCAppResources(preload_name=application.application_name),
            world_rank=0,
        )


def io_factories() -> dict[ApplicationMode, IOFactory]:
    factories: tuple[IOFactory, ...] = (
        ReplayIOFactory("mp4"),
        ReplayIOFactory("null"),
        WebRTCIOFactory(),
        NativeWindowIOFactory(),
    )
    return {factory.mode: factory for factory in factories}


def _reject_network_options(options: IOOptions) -> None:
    if options.host is not None or options.port is not None:
        raise ValueError("Replay output does not accept --host or --port.")


def _reject_all_options(options: IOOptions) -> None:
    if options != IOOptions():
        raise ValueError("Local-window output does not accept IO overrides.")


__all__ = [
    "IOFactory",
    "IOOptions",
    "NativeWindowIOFactory",
    "ReplayIOFactory",
    "WebRTCIOFactory",
    "io_factories",
]
