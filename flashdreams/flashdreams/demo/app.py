# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared command lifecycle for public demo applications."""

from __future__ import annotations

import argparse
import asyncio
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from typing import Any

import torch
import torch.distributed as dist

from flashdreams.core.distributed import init as distributed_init
from flashdreams.runtime.demo.bootstrap import (
    configure_logging,
    initialize_cuda_distributed,
)
from flashdreams.runtime.demo.host import RuntimeHost
from flashdreams.runtime.demo.outputs import build_output_sink
from flashdreams.runtime.demo.run_modes import RunResult
from flashdreams.runtime.demo.spec import (
    DemoAdapter,
    DemoSpec,
    WebRTCAppResources,
    WebRTCOutputSpec,
)

from .application import Application, DemoAdapterApplication, IOHandler
from .io import IOHandlerServer, create_replay_io_handler
from .runner import Runner


@dataclass(frozen=True, slots=True)
class _PublicWebRTCConfig:
    video_width: int
    video_height: int
    warmup_chunks: int
    warmup_timeout_s: float


class DemoApplication:
    """Base command application shared by model replay and WebRTC demos."""

    def __init__(
        self,
        *,
        parse_args: Callable[[list[str] | None], argparse.Namespace] | None = None,
        replay_spec: Callable[[argparse.Namespace], DemoSpec] | None = None,
        replay_adapter: Callable[[], DemoAdapter] | None = None,
        webrtc_io_handler: Callable[..., IOHandlerServer] | None = None,
    ) -> None:
        self._parse_args_fn = parse_args
        self._replay_spec_fn = replay_spec
        self._replay_adapter_fn = replay_adapter
        self._webrtc_io_handler_fn = webrtc_io_handler

    def main(self, argv: list[str] | None = None) -> None:
        """Parse arguments, select an IO bundle, and run the selected mode."""
        configure_logging()
        args = self.parse_args(argv)
        selection = self.create_io_handler(args)

        if isinstance(selection, IOHandlerServer):
            result = selection.serve(lambda handler: self._run_handler(args, handler))
        else:
            result = self._run_handler(args, selection)
        _raise_for_failed_result(result)

    def parse_args(self, argv: list[str] | None = None) -> argparse.Namespace:
        """Parse this model's command-line arguments."""
        parse_args = getattr(self, "_parse_args_fn", None)
        if parse_args is None:
            raise NotImplementedError("DemoApplication.parse_args is not configured.")
        return parse_args(argv)

    def replay_spec(self, args: argparse.Namespace) -> DemoSpec:
        """Build the model-specific replay specification."""
        replay_spec = getattr(self, "_replay_spec_fn", None)
        if replay_spec is None:
            raise NotImplementedError("DemoApplication.replay_spec is not configured.")
        return replay_spec(args)

    def replay_adapter(self) -> DemoAdapter:
        """Create the model-specific replay adapter."""
        replay_adapter = getattr(self, "_replay_adapter_fn", None)
        if replay_adapter is None:
            raise NotImplementedError(
                "DemoApplication.replay_adapter is not configured."
            )
        return replay_adapter()

    def application(self, args: argparse.Namespace) -> Application:
        """Create the application object consumed by the public runner."""
        return DemoAdapterApplication(
            adapter=self.replay_adapter(),
            spec=self.replay_spec(args),
        )

    def create_io_handler(
        self,
        args: argparse.Namespace,
    ) -> IOHandler | IOHandlerServer:
        """Select the IO factory for the parsed command."""
        command = str(getattr(args, "command", ""))
        if command == "replay":
            spec = self.replay_spec(args)
            return create_replay_io_handler(output_sink=build_output_sink(spec.output))
        if command == "webrtc":
            context = initialize_cuda_distributed(
                default_device=args.device,
                distributed_init_fn=distributed_init,
                configure_logging_fn=configure_logging,
                torch_module=torch,
                dist_module=dist,
            )
            return self.webrtc_io_handler(args, context=context)
        raise AssertionError(f"Unhandled command: {command}")

    def webrtc_io_handler(
        self,
        args: argparse.Namespace,
        *,
        context: Any,
    ) -> IOHandlerServer:
        """Create the WebRTC server-shaped IO factory for this model."""
        webrtc_io_handler = getattr(self, "_webrtc_io_handler_fn", None)
        if webrtc_io_handler is not None:
            return webrtc_io_handler(args, context=context)
        del args, context
        raise ValueError("This demo application does not support WebRTC.")

    def _run_handler(self, args: argparse.Namespace, handler: IOHandler) -> RunResult:
        return Runner(
            io_handler=handler,
            app=self.application(args),
        ).run()


def run_replay_application(*, spec: DemoSpec, adapter: DemoAdapter) -> RunResult:
    """Run a finite demo spec through the public replay IO factory and runner."""
    return run_application_replay(
        app=DemoAdapterApplication(adapter=adapter, spec=spec)
    )


def run_application_replay(
    *, app: Application, launch_args: Sequence[str] = ()
) -> RunResult:
    """Run a finite public application through the replay IO factory and runner."""
    output_sink = None
    if isinstance(app, DemoAdapterApplication):
        output_sink = build_output_sink(app.spec.output)
    return Runner(
        io_handler=create_replay_io_handler(output_sink=output_sink),
        app=app,
        launch_args=tuple(launch_args),
    ).run()


def run_application_webrtc(
    *, app: Application, launch_args: Sequence[str] = ()
) -> object:
    """Serve a public ``DemoAdapterApplication`` through shared WebRTC serving."""
    del launch_args
    if not isinstance(app, DemoAdapterApplication):
        raise ValueError(
            "Direct WebRTC application launch requires DemoAdapterApplication."
        )
    output = app.spec.output
    if not isinstance(output, WebRTCOutputSpec):
        raise ValueError("Direct WebRTC application launch requires WebRTCOutputSpec.")
    configure_logging()
    config = app.spec.config
    if config is None:
        raise ValueError("Direct WebRTC application launch requires DemoSpec.config.")
    context = initialize_cuda_distributed(
        default_device=config.device or "cuda",
        distributed_init_fn=distributed_init,
        configure_logging_fn=configure_logging,
        torch_module=torch,
        dist_module=dist,
    )
    web_config = replace(config, device=str(context.device))
    spec = replace(
        app.spec,
        input_mode="webrtc",
        config=web_config,
    )
    adapter = app.adapter
    scenario = adapter.prepare_scenario(spec)
    runtime = adapter.create_runtime(web_config)
    manager: Any | None = None
    try:
        manager = _create_application_webrtc_manager(
            runtime=runtime,
            output=output,
            spec=spec,
            scenario=scenario,
            adapter=adapter,
        )
        app_resources = _create_application_webrtc_resources(
            adapter=adapter,
            manager=manager,
            output=output,
            spec=spec,
        )

        from flashdreams.serving.webrtc.demo import serve_webrtc_demo

        return serve_webrtc_demo(
            output=output,
            model_id=spec.model_id,
            session_manager=manager,
            app_resources=app_resources,
            world_rank=context.world_rank,
        )
    except BaseException as exc:
        # Once the runtime exists, aiohttp shutdown is not guaranteed to run
        # until the server fully starts. Clean up through the same manager/host
        # ownership path that normal WebRTC shutdown uses, while preserving the
        # startup failure as the primary exception.
        _cleanup_application_webrtc_startup_failure(
            manager=manager,
            runtime=runtime,
            primary_error=exc,
        )
        raise


def create_demo_application(
    *,
    parse_args: Callable[[list[str] | None], argparse.Namespace],
    replay_spec: Callable[[argparse.Namespace], DemoSpec],
    replay_adapter: Callable[[], DemoAdapter],
    webrtc_io_handler: Callable[..., IOHandlerServer] | None = None,
) -> DemoApplication:
    """Create a command app from functions instead of a pass-through subclass."""
    return DemoApplication(
        parse_args=parse_args,
        replay_spec=replay_spec,
        replay_adapter=replay_adapter,
        webrtc_io_handler=webrtc_io_handler,
    )


def _create_application_webrtc_manager(
    *,
    runtime: Any,
    output: WebRTCOutputSpec,
    spec: DemoSpec,
    scenario: Any,
    adapter: DemoAdapter,
) -> Any:
    from flashdreams.serving.webrtc.manager import BaseWebRTCSessionManager

    return BaseWebRTCSessionManager(
        runtime=runtime,
        runtime_config=_PublicWebRTCConfig(
            video_width=output.video_width,
            video_height=output.video_height,
            warmup_chunks=output.warmup_chunks,
            warmup_timeout_s=output.warmup_timeout_s,
        ),
        fps=output.fps,
        identity=str(getattr(adapter, "model_id", spec.model_id)),
        warmup_label=output.preload_name
        or _metadata_str(spec, "webrtc_preload_name", default="WebRTC"),
        supported_control_keys=_metadata_string_set(
            spec, "webrtc_supported_control_keys"
        ),
        client_liveness_timeout_s=output.client_liveness_timeout_s,
        shared_host=RuntimeHost(runtime),
        shared_adapter=adapter,
        shared_spec=spec,
        shared_scenario=scenario,
        keep_connection_after_completed=bool(
            spec.metadata.get("webrtc_keep_connection_after_completed", False)
        ),
    )


def _cleanup_application_webrtc_startup_failure(
    *,
    manager: Any | None,
    runtime: Any,
    primary_error: BaseException,
) -> None:
    try:
        if manager is None:
            runtime.close()
            return
        asyncio.run(manager.shutdown())
    except BaseException as cleanup_error:
        add_note = getattr(primary_error, "add_note", None)
        if callable(add_note):
            add_note(f"Additional WebRTC startup cleanup error: {cleanup_error!r}")


def _create_application_webrtc_resources(
    *,
    adapter: DemoAdapter,
    manager: Any,
    output: WebRTCOutputSpec,
    spec: DemoSpec,
) -> WebRTCAppResources:
    factory = getattr(adapter, "create_webrtc_app_resources", None)
    if callable(factory):
        resources = factory(manager=manager, output=output, spec=spec)
        if not isinstance(resources, WebRTCAppResources):
            raise TypeError(
                "Demo adapter create_webrtc_app_resources(...) must return "
                f"WebRTCAppResources, got {type(resources).__name__}."
            )
        return resources
    resources = spec.metadata.get("webrtc_app_resources")
    if isinstance(resources, WebRTCAppResources):
        return resources
    return WebRTCAppResources(preload_name=output.preload_name or spec.model_id)


def _metadata_str(spec: DemoSpec, name: str, *, default: str) -> str:
    value = spec.metadata.get(name, default)
    return str(value)


def _metadata_string_set(spec: DemoSpec, name: str) -> frozenset[str] | None:
    value = spec.metadata.get(name)
    if value is None:
        return None
    if isinstance(value, str):
        return frozenset({value})
    try:
        return frozenset(str(item) for item in value)
    except TypeError as exc:
        raise TypeError(f"DemoSpec.metadata[{name!r}] must be iterable.") from exc


def _raise_for_failed_result(result: RunResult) -> None:
    if result.status in {"completed", "skipped"}:
        return
    reason = result.reason or (str(result.error) if result.error is not None else None)
    if reason is None:
        reason = f"Demo ended with status {result.status!r}."
    print(reason, file=sys.stderr)
    raise SystemExit(1)


__all__ = [
    "DemoApplication",
    "create_demo_application",
    "run_application_webrtc",
    "run_application_replay",
    "run_replay_application",
]
