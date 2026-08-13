# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Built-in IO handlers for package applications."""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import cache
from importlib.metadata import EntryPoint, entry_points
from pathlib import Path
from typing import Protocol, runtime_checkable

from flashdreams.infra.postprocess import VideoTensorLayout
from flashdreams.runtime.application import FlashDreamsApplication
from flashdreams.runtime.application_runner import ApplicationRunner
from flashdreams.runtime.demo import (
    Mp4OutputSpec,
    NativeWindowOutputSpec,
    NullOutputSpec,
    OutputSpec,
    RuntimeHost,
    WebRTCAppResources,
    WebRTCOutputSpec,
)
from flashdreams.runtime.demo.contracts import DemoAdapter
from flashdreams.runtime.demo.replay import run_replay_demo
from flashdreams.runtime.io_handler import IOHandler
from flashdreams.runtime.worker import ModelExecutionWorker
from flashdreams.serving.native_window import run_native_window_presentation
from flashdreams.serving.native_window.runner import NativePresenter

IO_HANDLER_ENTRY_POINT_GROUP = "flashdreams.io_handlers"


@runtime_checkable
class VideoApplication(Protocol):
    application_name: str
    fps: int
    video_width: int
    video_height: int
    output_layout: VideoTensorLayout


@runtime_checkable
class NativeWindowApplication(VideoApplication, Protocol):
    title: str | None

    @property
    def native_presenter_factory(self) -> Callable[..., NativePresenter] | None: ...

    @property
    def native_key_bindings(self) -> Mapping[str, Sequence[str]] | None: ...


@runtime_checkable
class WebRTCApplication(VideoApplication, Protocol):
    supported_control_keys: frozenset[str]

    @property
    def webrtc_app_resources(self) -> WebRTCAppResources: ...


@runtime_checkable
class DriverApplication(FlashDreamsApplication, DemoAdapter, Protocol):
    """Application capabilities required by the existing session drivers."""


class BaseIOHandler(ABC):
    """Base for built-in application IO handlers."""

    input_mode: str
    realtime: bool

    @classmethod
    @abstractmethod
    def from_argv(
        cls,
        args: Sequence[str],
    ) -> tuple[IOHandler, list[str]]:
        """Consume IO arguments and return the unconsumed app arguments."""

    @abstractmethod
    def create_output(self, application: FlashDreamsApplication) -> OutputSpec:
        """Build the transport-specific output description."""

    @abstractmethod
    def run(self, runner: ApplicationRunner) -> object:
        """Run this IO behavior through the application runner."""


class ReplayIOHandler(BaseIOHandler):
    input_mode = "replay"
    realtime = False

    def run(self, runner: ApplicationRunner) -> object:
        application = _require_driver_application(runner.application)
        result = run_replay_demo(
            spec=runner.create_driver_spec(self.create_output(application)),
            adapter=application,
        )
        if result.status != "completed":
            reason = result.reason or str(result.error) or result.status
            raise RuntimeError(f"Application replay failed: {reason}")
        return result


@dataclass(frozen=True, slots=True)
class Mp4IOHandler(ReplayIOHandler):
    output_path: Path | None = None

    @classmethod
    def from_argv(
        cls,
        args: Sequence[str],
    ) -> tuple[IOHandler, list[str]]:
        remaining = list(args)
        output = _pop_option(remaining, "--output")
        _reject_options(remaining, "MP4", "--host", "--port")
        return cls(output_path=None if output is None else Path(output)), remaining

    def create_output(self, application: FlashDreamsApplication) -> Mp4OutputSpec:
        video = _require_video_application(application)
        return Mp4OutputSpec(
            path=self.output_path
            or Path("outputs") / f"{application.application_name}.mp4",
            fps=video.fps,
            output_layout=video.output_layout,
        )


class NullIOHandler(ReplayIOHandler):
    @classmethod
    def from_argv(
        cls,
        args: Sequence[str],
    ) -> tuple[IOHandler, list[str]]:
        remaining = list(args)
        _reject_options(remaining, "Null", "--output", "--host", "--port")
        return cls(), remaining

    def create_output(self, application: FlashDreamsApplication) -> NullOutputSpec:
        del application
        return NullOutputSpec()


class NativeWindowIOHandler(BaseIOHandler):
    input_mode = "keyboard-driving"
    realtime = True

    @classmethod
    def from_argv(
        cls,
        args: Sequence[str],
    ) -> tuple[IOHandler, list[str]]:
        remaining = list(args)
        _reject_options(
            remaining,
            "Native-window",
            "--output",
            "--host",
            "--port",
        )
        return cls(), remaining

    def create_output(
        self,
        application: FlashDreamsApplication,
    ) -> NativeWindowOutputSpec:
        native = _require_native_application(application)
        return NativeWindowOutputSpec(
            fps=native.fps,
            video_width=native.video_width,
            video_height=native.video_height,
            title=native.title or native.application_name,
        )

    def run(self, runner: ApplicationRunner) -> object:
        application = _require_driver_application(runner.application)
        native = _require_native_application(application)
        spec = runner.create_driver_spec(self.create_output(application))
        if native.native_presenter_factory is None:
            return run_native_window_presentation(
                spec=spec,
                adapter=application,
                key_bindings=native.native_key_bindings,
            )
        return run_native_window_presentation(
            spec=spec,
            adapter=application,
            presenter_factory=native.native_presenter_factory,
            key_bindings=native.native_key_bindings,
        )


@dataclass(frozen=True, slots=True)
class _WebRTCConfig:
    video_width: int
    video_height: int
    warmup_chunks: int = 0
    warmup_timeout_s: float = 30.0


@dataclass(frozen=True, slots=True)
class WebRTCIOHandler(BaseIOHandler):
    host: str = "0.0.0.0"
    port: int = 8080
    input_mode = "keyboard-driving"
    realtime = True

    @classmethod
    def from_argv(
        cls,
        args: Sequence[str],
    ) -> tuple[IOHandler, list[str]]:
        remaining = list(args)
        _reject_options(remaining, "WebRTC", "--output")
        host = _pop_option(remaining, "--host")
        port = _pop_option(remaining, "--port")
        return cls(
            host="0.0.0.0" if host is None else host,
            port=8080 if port is None else int(port),
        ), remaining

    def create_output(self, application: FlashDreamsApplication) -> WebRTCOutputSpec:
        webrtc = _require_webrtc_application(application)
        return WebRTCOutputSpec(
            host=self.host,
            port=self.port,
            fps=webrtc.fps,
            video_width=webrtc.video_width,
            video_height=webrtc.video_height,
        )

    def run(self, runner: ApplicationRunner) -> object:
        from flashdreams.serving.webrtc.demo import serve_webrtc_demo
        from flashdreams.serving.webrtc.manager import BaseWebRTCSessionManager

        if int(os.environ.get("WORLD_SIZE", "1")) != 1:
            raise RuntimeError("Application WebRTC output requires one process.")
        application = _require_driver_application(runner.application)
        spec = runner.create_driver_spec(self.create_output(application))
        output = spec.output
        if not isinstance(output, WebRTCOutputSpec):
            raise TypeError("WebRTC IO handler requires WebRTCOutputSpec.")
        webrtc = _require_webrtc_application(application)
        scenario = application.prepare_scenario(spec)
        worker = ModelExecutionWorker(device=application.config.device)
        try:
            runtime = worker.call_blocking(
                application.create_runtime,
                application.config,
            )
        except Exception:
            worker.close_blocking()
            raise
        host = RuntimeHost(runtime, worker=worker)
        manager = BaseWebRTCSessionManager(
            runtime=runtime,
            runtime_config=_WebRTCConfig(
                video_width=output.video_width,
                video_height=output.video_height,
                warmup_chunks=output.warmup_chunks,
                warmup_timeout_s=output.warmup_timeout_s,
            ),
            fps=output.fps,
            identity=application.application_name,
            client_liveness_timeout_s=output.client_liveness_timeout_s,
            supported_control_keys=webrtc.supported_control_keys,
            shared_host=host,
            shared_adapter=application,
            shared_spec=spec,
            shared_scenario=scenario,
        )
        return serve_webrtc_demo(
            output=output,
            model_id=application.model_id,
            session_manager=manager,
            app_resources=webrtc.webrtc_app_resources,
            world_rank=0,
        )


@cache
def io_handler_entry_points() -> dict[str, EntryPoint]:
    handlers: dict[str, EntryPoint] = {}
    for entry_point in entry_points(group=IO_HANDLER_ENTRY_POINT_GROUP):
        if entry_point.name in handlers:
            raise ValueError(f"Duplicate IO handler {entry_point.name!r}.")
        handlers[entry_point.name] = entry_point
    return handlers


def load_io_handler(
    name: str,
    args: Sequence[str],
) -> tuple[IOHandler, list[str]]:
    try:
        entry_point = io_handler_entry_points()[name]
    except KeyError as exc:
        available = ", ".join(sorted(io_handler_entry_points()))
        raise ValueError(
            f"Unknown IO handler {name!r}. Available handlers: {available}."
        ) from exc
    handler_type = entry_point.load()
    from_argv = getattr(handler_type, "from_argv", None)
    if not isinstance(handler_type, type) or not callable(from_argv):
        raise TypeError(f"IO handler entry point {name!r} must load a handler class.")
    handler, remaining = handler_type.from_argv(args)
    if not isinstance(handler, IOHandler):
        raise TypeError(f"IO handler factory {name!r} returned an invalid instance.")
    return handler, remaining


def _pop_option(args: list[str], name: str) -> str | None:
    prefix = f"{name}="
    for index, value in enumerate(args):
        if value.startswith(prefix):
            del args[index]
            return value.removeprefix(prefix)
    if name not in args:
        return None
    index = args.index(name)
    if index + 1 >= len(args):
        raise ValueError(f"{name} requires a value.")
    value = args[index + 1]
    del args[index : index + 2]
    return value


def _reject_options(args: list[str], label: str, *names: str) -> None:
    for name in names:
        prefix = f"{name}="
        if name in args or any(value.startswith(prefix) for value in args):
            raise ValueError(f"{label} IO handler does not accept {name}.")


def _require_video_application(
    application: FlashDreamsApplication,
) -> VideoApplication:
    if not isinstance(application, VideoApplication):
        raise TypeError("IO handler requires a video application.")
    return application


def _require_driver_application(
    application: FlashDreamsApplication,
) -> DriverApplication:
    if not isinstance(application, DriverApplication):
        raise TypeError("Built-in IO handlers require session-driver capabilities.")
    return application


def _require_native_application(
    application: FlashDreamsApplication,
) -> NativeWindowApplication:
    if not isinstance(application, NativeWindowApplication):
        raise TypeError("Native-window IO requires native capabilities.")
    return application


def _require_webrtc_application(
    application: FlashDreamsApplication,
) -> WebRTCApplication:
    if not isinstance(application, WebRTCApplication):
        raise TypeError("WebRTC IO requires WebRTC capabilities.")
    return application


__all__ = [
    "IO_HANDLER_ENTRY_POINT_GROUP",
    "BaseIOHandler",
    "Mp4IOHandler",
    "NativeWindowIOHandler",
    "NullIOHandler",
    "ReplayIOHandler",
    "WebRTCIOHandler",
    "io_handler_entry_points",
    "load_io_handler",
]
