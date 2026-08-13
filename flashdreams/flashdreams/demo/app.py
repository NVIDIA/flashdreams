# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared command lifecycle for public demo applications."""

from __future__ import annotations

import argparse
import sys
from abc import ABC, abstractmethod
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


class DemoApplication(ABC):
    """Base command application shared by model replay and WebRTC demos."""

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

    @abstractmethod
    def parse_args(self, argv: list[str] | None = None) -> argparse.Namespace:
        """Parse this model's command-line arguments."""

    @abstractmethod
    def replay_spec(self, args: argparse.Namespace) -> DemoSpec:
        """Build the model-specific replay specification."""

    @abstractmethod
    def replay_adapter(self) -> DemoAdapter:
        """Create the model-specific replay adapter."""

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
        del args, context
        raise ValueError("This demo application does not support WebRTC.")

    def _run_handler(self, args: argparse.Namespace, handler: IOHandler) -> RunResult:
        return Runner(
            io_handler=handler,
            app=self.application(args),
        ).run()


def run_replay_application(*, spec: DemoSpec, adapter: DemoAdapter) -> RunResult:
    """Run a finite demo spec through the public replay IO factory and runner."""
    return Runner(
        io_handler=create_replay_io_handler(output_sink=build_output_sink(spec.output)),
        app=DemoAdapterApplication(adapter=adapter, spec=spec),
    ).run()


def _raise_for_failed_result(result: RunResult) -> None:
    if result.status in {"completed", "skipped"}:
        return
    reason = result.reason or (str(result.error) if result.error is not None else None)
    if reason is None:
        reason = f"Demo ended with status {result.status!r}."
    print(reason, file=sys.stderr)
    raise SystemExit(1)


__all__ = ["DemoApplication", "run_replay_application"]
