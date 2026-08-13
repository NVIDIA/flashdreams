# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared command lifecycle for public demo applications."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence
from typing import Any

import torch
import torch.distributed as dist

from flashdreams.core.distributed import init as distributed_init
from flashdreams.runtime.demo.bootstrap import (
    configure_logging,
    initialize_cuda_distributed,
)
from flashdreams.runtime.demo.outputs import build_output_sink
from flashdreams.runtime.demo.run_modes import RunResult
from flashdreams.runtime.demo.spec import DemoAdapter, DemoSpec

from .application import Application, DemoAdapterApplication, IOHandler
from .io import IOHandlerServer, create_replay_io_handler
from .runner import Runner


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
    return run_application_replay(app=DemoAdapterApplication(adapter=adapter, spec=spec))


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
    "run_application_replay",
    "run_replay_application",
]
