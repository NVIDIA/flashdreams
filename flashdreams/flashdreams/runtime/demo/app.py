# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared command lifecycle for model demo applications."""

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
from flashdreams.runtime.demo.replay import run_replay_demo
from flashdreams.runtime.demo.spec import DemoAdapter, DemoSpec


class DemoApplication(ABC):
    """Base command application shared by model replay and WebRTC demos."""

    def main(self, argv: list[str] | None = None) -> None:
        """Parse arguments and dispatch the selected demo mode."""
        configure_logging()
        args = self.parse_args(argv)
        if args.command == "replay":
            result = run_replay_demo(
                spec=self.replay_spec(args),
                adapter=self.replay_adapter(),
            )
            if result.status != "completed":
                reason = result.reason or (
                    str(result.error) if result.error is not None else None
                )
                if reason is None:
                    reason = f"Replay demo ended with status {result.status!r}."
                print(reason, file=sys.stderr)
                raise SystemExit(1)
            return
        if args.command == "webrtc":
            context = initialize_cuda_distributed(
                default_device=args.device,
                distributed_init_fn=distributed_init,
                configure_logging_fn=configure_logging,
                torch_module=torch,
                dist_module=dist,
            )
            self.prepare_webrtc(args, context=context)
            self.serve_webrtc(args, context=context)
            return
        raise AssertionError(f"Unhandled command: {args.command}")

    @abstractmethod
    def parse_args(self, argv: list[str] | None = None) -> argparse.Namespace:
        """Parse this model's command-line arguments."""

    @abstractmethod
    def replay_spec(self, args: argparse.Namespace) -> DemoSpec:
        """Build the model-specific replay specification."""

    @abstractmethod
    def replay_adapter(self) -> DemoAdapter:
        """Create the model-specific replay adapter."""

    def prepare_webrtc(self, args: argparse.Namespace, *, context: Any) -> None:
        """Perform optional model-specific setup before serving WebRTC."""
        del args, context

    @abstractmethod
    def serve_webrtc(self, args: argparse.Namespace, *, context: Any) -> None:
        """Build and serve the model-specific WebRTC demo."""


__all__ = ["DemoApplication"]
