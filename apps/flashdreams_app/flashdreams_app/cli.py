# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Command line host for independently installed FlashDreams app providers."""

from __future__ import annotations

import argparse
import importlib
from contextlib import ExitStack
from importlib import metadata
from pathlib import Path
from types import ModuleType
from typing import Sequence

import torch

from flashdreams.runtime.demo.bootstrap import (
    configure_logging,
    initialize_cuda_distributed,
)
from flashdreams.runtime.output import OutputArtifact

from .contracts import AppConfig, AppRuntime, PipelineAppSpec
from .outputs import FileOutput
from .runtime import PipelineAppRuntime
from .webrtc import WebRTCOptions, serve_webrtc


def build_parser() -> argparse.ArgumentParser:
    """Build the host parser without importing a provider package."""
    parser = argparse.ArgumentParser(prog="flashdreams-app")
    parser.add_argument(
        "provider", help="Installed provider distribution, e.g. t2v-app"
    )
    parser.add_argument("mode", choices=("mp4", "webrtc"))
    parser.add_argument("--output", type=Path, help="MP4 path (required for mp4)")
    parser.add_argument("--device", default="cuda", help="Runtime device")
    parser.add_argument(
        "--compile", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument(
        "--cuda-graph", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument("--host", default="0.0.0.0", help="WebRTC bind address")
    parser.add_argument("--port", type=int, default=8080, help="WebRTC bind port")
    parser.add_argument("--warmup-chunks", type=int, default=0)
    parser.add_argument("--warmup-timeout-s", type=float, default=600.0)
    parser.add_argument("--client-liveness-timeout-s", type=float, default=30.0)
    parser.add_argument(
        "--encoder-backend", choices=("auto", "default", "nvenc"), default="auto"
    )
    parser.add_argument("--encoder-bitrate-bps", type=int, default=6_000_000)
    parser.add_argument("--encoder-gop", type=int)
    return parser


def load_provider(distribution_name: str) -> ModuleType:
    """Load an installed provider after verifying its distribution is present."""
    try:
        distribution = metadata.distribution(distribution_name)
    except metadata.PackageNotFoundError as exc:
        raise ValueError(
            f"Provider distribution {distribution_name!r} is not installed."
        ) from exc

    package_names = metadata.packages_distributions()
    candidates = [
        name
        for name, distributions in package_names.items()
        if distribution.metadata["Name"] in distributions
    ]
    candidates.append(distribution_name.replace("-", "_"))
    for candidate in dict.fromkeys(candidates):
        try:
            return importlib.import_module(candidate)
        except ModuleNotFoundError as exc:
            if exc.name != candidate:
                raise
    raise ValueError(
        f"Provider {distribution_name!r} does not expose an importable Python module."
    )


def run(argv: Sequence[str] | None = None) -> tuple[OutputArtifact, ...]:
    """Dispatch one provider session to its selected presentation path.

    Args:
        argv: Command-line arguments; ``None`` reads the process arguments.

    Returns:
        Artifacts produced by the selected path. WebRTC returns an empty tuple.
    """
    probe = argparse.ArgumentParser(add_help=False)
    probe.add_argument("provider", nargs="?")
    probe.add_argument("mode", nargs="?")
    provider_args, _ = probe.parse_known_args(argv)
    parser = build_parser()
    if provider_args.provider is None:
        parser.parse_args(argv)
        return ()
    provider = load_provider(provider_args.provider)
    add_arguments = getattr(provider, "add_arguments", None)
    if callable(add_arguments):
        add_arguments(parser)
    args = parser.parse_args(argv)
    if args.mode == "mp4" and args.output is None:
        parser.error("--output is required for mp4 mode")

    environment = _initialize_environment(args.device)
    options = vars(args).copy()
    options["device"] = environment.device
    options["world_rank"] = environment.world_rank
    options["world_size"] = environment.world_size

    factory = getattr(provider, "create_app", None)
    if not callable(factory):
        raise TypeError(f"Provider {args.provider!r} must define create_app(config).")
    spec = factory(AppConfig(options=options))
    if not isinstance(spec, PipelineAppSpec):
        raise TypeError(
            f"Provider {args.provider!r} create_app() returned "
            f"{type(spec).__name__}, expected PipelineAppSpec."
        )
    runtime = PipelineAppRuntime(
        spec=spec,
        device=environment.device,
        compile=args.compile,
        cuda_graph=args.cuda_graph,
    )
    if args.mode == "webrtc":
        return _run_webrtc(runtime=runtime, args=args, environment=environment)
    if args.mode == "mp4":
        return _run_mp4(
            runtime=runtime,
            output_path=args.output,
            environment=environment,
        )
    raise AssertionError(f"Unsupported presentation mode: {args.mode!r}.")


def _run_webrtc(
    *,
    runtime: AppRuntime,
    args: argparse.Namespace,
    environment: "_Environment",
) -> tuple[OutputArtifact, ...]:
    """Run the WebRTC serving path and close its runtime.

    Args:
        runtime: Initialized application runtime.
        args: Parsed host and WebRTC arguments.
        environment: Initialized process and distributed environment.

    Returns:
        An empty tuple because WebRTC does not create file artifacts.
    """
    try:
        serve_webrtc(
            runtime=runtime,
            options=WebRTCOptions(
                host=args.host,
                port=args.port,
                warmup_chunks=args.warmup_chunks,
                warmup_timeout_s=args.warmup_timeout_s,
                client_liveness_timeout_s=args.client_liveness_timeout_s,
                device=environment.device,
                encoder_backend=args.encoder_backend,
                encoder_bitrate_bps=args.encoder_bitrate_bps,
                encoder_gop=args.encoder_gop or int(runtime.metadata.fps),
            ),
            world_rank=environment.world_rank,
        )
    finally:
        runtime.close()
    return ()


def _run_mp4(
    *,
    runtime: AppRuntime,
    output_path: Path,
    environment: "_Environment",
) -> tuple[OutputArtifact, ...]:
    """Run the finite MP4 generation path and close all owned resources.

    Args:
        runtime: Initialized application runtime.
        output_path: Destination for the generated MP4.
        environment: Initialized process and distributed environment.

    Returns:
        Artifacts emitted by the file output target.
    """

    with ExitStack() as resources:
        resources.callback(runtime.close)
        output = FileOutput(
            path=output_path,
            fps=runtime.metadata.fps,
            output_layout=runtime.metadata.output_layout,
            enabled=environment.world_rank == 0,
        )
        output_closed = False

        def close_output() -> None:
            if not output_closed:
                output.close()

        resources.callback(close_output)
        output.open()
        session = runtime.start_session(runtime.initial_input)
        resources.callback(session.close)
        while (request := session.next_step_request()) is not None:
            output.write(session.step(runtime.prepare_step_input(request)))
        artifacts = tuple(output.close())
        output_closed = True
        return artifacts


class _Environment:
    def __init__(self, *, device: str, world_rank: int, world_size: int) -> None:
        self.device = device
        self.world_rank = world_rank
        self.world_size = world_size


def _initialize_environment(device: str) -> _Environment:
    """Initialize logging, CUDA placement, and distributed state in the host."""
    if torch.device(device).type == "cuda":
        context = initialize_cuda_distributed(default_device=device)
        return _Environment(
            device=str(context.device),
            world_rank=context.world_rank,
            world_size=context.world_size,
        )
    configure_logging(world_rank=0)
    return _Environment(device=str(torch.device(device)), world_rank=0, world_size=1)


def main() -> None:
    """Console-script entry point."""
    artifacts = run()
    for artifact in artifacts:
        print(artifact.uri)
