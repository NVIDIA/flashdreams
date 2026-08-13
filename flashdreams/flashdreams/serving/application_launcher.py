# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Application package discovery and execution."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import cache
from importlib.metadata import entry_points
from pathlib import Path
from typing import cast

from flashdreams.runtime.demo import (
    DemoSpec,
    Mp4OutputSpec,
    NativeWindowOutputSpec,
    NullOutputSpec,
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

APPLICATION_ENTRY_POINT_GROUP = "flashdreams.applications"
ApplicationFactory = Callable[[Sequence[str]], FlashDreamsApplication]


@cache
def application_factories() -> dict[str, ApplicationFactory]:
    factories: dict[str, ApplicationFactory] = {}
    for entry_point in entry_points(group=APPLICATION_ENTRY_POINT_GROUP):
        factory = entry_point.load()
        if not callable(factory):
            raise TypeError(
                f"Application entry point {entry_point.name!r} is not callable."
            )
        if entry_point.name in factories:
            raise ValueError(f"Duplicate application {entry_point.name!r}.")
        factories[entry_point.name] = factory
    return factories


def run_application_from_argv(argv: Sequence[str]) -> bool:
    if not argv:
        return False
    factory = application_factories().get(argv[0])
    if factory is None:
        return False
    mode, output_path, host, port, app_args = _parse_invocation(argv[1:])
    application = factory(app_args)
    if not isinstance(application, FlashDreamsApplication):
        raise TypeError(f"Application factory {argv[0]!r} returned an invalid object.")
    selected_mode = mode or application.default_mode
    _run_application(
        application,
        mode=selected_mode,
        output_path=output_path,
        host=host,
        port=port,
    )
    return True


@dataclass(frozen=True, slots=True)
class _WebRTCConfig:
    video_width: int
    video_height: int
    warmup_chunks: int = 0
    warmup_timeout_s: float = 30.0


def _run_application(
    application: FlashDreamsApplication,
    *,
    mode: ApplicationMode,
    output_path: Path | None,
    host: str | None,
    port: int | None,
) -> object:
    if mode not in application.supported_output_modes():
        raise ValueError(f"Application does not support output mode {mode!r}.")
    input_mode = "replay" if mode in {"mp4", "null"} else "keyboard-driving"
    if input_mode not in application.supported_input_modes():
        raise ValueError(f"Application does not support input mode {input_mode!r}.")
    output = _output(
        application,
        mode=mode,
        output_path=output_path,
        host=host,
        port=port,
    )
    spec = DemoSpec(
        model_id=application.model_id,
        input_mode=input_mode,
        output=output,
        scenario=application.scenario,
        config=application.config,
    )
    if mode == "mp4" or mode == "null":
        result = run_replay_demo(spec=spec, adapter=application)
        if result.status != "completed":
            reason = result.reason or str(result.error) or result.status
            raise RuntimeError(f"Application replay failed: {reason}")
        return result
    if mode == "local-window":
        return run_native_window_demo(spec=spec, adapter=application)
    if not isinstance(output, WebRTCOutputSpec):
        raise TypeError("WebRTC mode resolved a non-WebRTC output.")
    return _serve_webrtc(application, spec=spec, output=output)


def _output(
    application: FlashDreamsApplication,
    *,
    mode: ApplicationMode,
    output_path: Path | None,
    host: str | None,
    port: int | None,
) -> Mp4OutputSpec | NativeWindowOutputSpec | NullOutputSpec | WebRTCOutputSpec:
    if mode == "null":
        return NullOutputSpec()
    if mode == "mp4":
        return Mp4OutputSpec(
            path=output_path or Path("outputs") / f"{application.application_name}.mp4",
            fps=application.fps,
            output_layout=application.output_layout,
        )
    if mode == "local-window":
        return NativeWindowOutputSpec(
            fps=application.fps,
            video_width=application.video_width,
            video_height=application.video_height,
            title=application.title or application.application_name,
        )
    return WebRTCOutputSpec(
        host=host or "0.0.0.0",
        port=port or 8080,
        fps=application.fps,
        video_width=application.video_width,
        video_height=application.video_height,
    )


def _serve_webrtc(
    application: FlashDreamsApplication,
    *,
    spec: DemoSpec,
    output: WebRTCOutputSpec,
) -> object:
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


def _parse_invocation(
    args: Sequence[str],
) -> tuple[ApplicationMode | None, Path | None, str | None, int | None, list[str]]:
    remaining = list(args)
    mode: ApplicationMode | None = None
    if remaining and remaining[0] in {"mp4", "null", "webrtc", "local-window"}:
        mode = cast(ApplicationMode, remaining.pop(0))
    output = _pop_option(remaining, "--output")
    host = _pop_option(remaining, "--host")
    port = _pop_option(remaining, "--port")
    return (
        mode,
        None if output is None else Path(output),
        host,
        None if port is None else int(port),
        remaining,
    )


def _pop_option(args: list[str], name: str) -> str | None:
    if name not in args:
        return None
    index = args.index(name)
    if index + 1 >= len(args):
        raise ValueError(f"{name} requires a value.")
    value = args[index + 1]
    del args[index : index + 2]
    return value


__all__ = [
    "APPLICATION_ENTRY_POINT_GROUP",
    "ApplicationFactory",
    "application_factories",
    "run_application_from_argv",
]
