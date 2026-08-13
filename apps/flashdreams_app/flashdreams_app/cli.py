# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Command line host for independently installed FlashDreams app providers."""

from __future__ import annotations

import argparse
import importlib
from contextlib import ExitStack
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Sequence

import torch

from flashdreams.runtime import InferenceInput, InferenceRuntime
from flashdreams.runtime.demo.bootstrap import (
    configure_logging,
    initialize_cuda_distributed,
)
from flashdreams.runtime.output import OutputArtifact

from .contracts import (
    AppConfig,
    AppProvider,
    AppRequest,
    AppSpec,
    Mp4RunSpec,
    WebRTCRunSpec,
)
from .outputs import FileOutput
from .runtime import PipelineAppRuntime
from .webrtc import WebRTCOptions, serve_webrtc


@dataclass(frozen=True, slots=True)
class _ProviderAndMode:
    """Top-level route parsed before loading an application provider."""

    provider: str
    """Installed provider distribution name."""

    mode: str
    """Selected presentation mode."""

    remaining_argv: tuple[str, ...]
    """Arguments delegated to the provider parser."""


def build_parser(provider: str, mode: str) -> argparse.ArgumentParser:
    """Build the selected mode's host-owned options parser.

    Args:
        provider: Provider name displayed in command usage.
        mode: Selected presentation mode.

    Returns:
        Parser ready for the provider to extend and invoke.
    """
    parser = argparse.ArgumentParser(prog=f"flashdreams-app {provider} {mode}")
    parser.add_argument("--device", default="cuda", help="Runtime device")
    if mode == "mp4":
        parser.add_argument("--output", type=Path, required=True, help="MP4 path")
    elif mode == "webrtc":
        parser.add_argument("--host", default="0.0.0.0", help="WebRTC bind address")
        parser.add_argument("--port", type=int, default=8080, help="WebRTC bind port")
    else:
        raise ValueError(f"Unsupported application mode: {mode!r}.")
    return parser


def load_provider(distribution_name: str) -> AppProvider:
    """Load an installed provider that satisfies the host contract."""
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
    incompatible: list[str] = []
    for candidate in dict.fromkeys(candidates):
        try:
            module = importlib.import_module(candidate)
        except ModuleNotFoundError as exc:
            if exc.name != candidate:
                raise
            continue
        if isinstance(module, AppProvider):
            return module
        incompatible.append(candidate)
    if incompatible:
        names = ", ".join(repr(name) for name in incompatible)
        raise TypeError(
            f"Provider distribution {distribution_name!r} exposes module(s) "
            f"{names}, but none satisfy AppProvider. Providers must define "
            "parse_options(parser, argv) and create_app_spec(request)."
        )
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
    route = _parse_provider_and_mode(argv)
    provider = load_provider(route.provider)
    options = provider.parse_options(
        build_parser(route.provider, route.mode),
        route.remaining_argv,
    )
    app_spec = _require_app_spec(
        provider.create_app_spec(AppRequest(mode=route.mode, options=options)),
        provider_name=route.provider,
        mode=route.mode,
    )
    args = argparse.Namespace(**options)
    environment = _initialize_environment(args.device)
    runtime: InferenceRuntime = PipelineAppRuntime(
        spec=app_spec.pipeline,
        config=app_spec.config,
        device=environment.device,
    )
    return _launch_mode(
        mode=route.mode,
        runtime=runtime,
        app_spec=app_spec,
        args=args,
        environment=environment,
    )


def _parse_provider_and_mode(
    argv: Sequence[str] | None,
) -> _ProviderAndMode:
    """Parse the provider and mode while preserving all remaining arguments."""
    parser = argparse.ArgumentParser(prog="flashdreams-app", add_help=False)
    parser.add_argument("provider", help="Installed provider distribution")
    parser.add_argument("mode", choices=("mp4", "webrtc"))
    args, remaining_argv = parser.parse_known_args(argv)
    return _ProviderAndMode(
        provider=args.provider,
        mode=args.mode,
        remaining_argv=tuple(remaining_argv),
    )


def _require_app_spec(
    value: object,
    *,
    provider_name: str,
    mode: str,
) -> AppSpec:
    """Validate a provider result before constructing its runtime."""
    if not isinstance(value, AppSpec):
        raise TypeError(
            f"Provider {provider_name!r} create_app_spec() returned "
            f"{type(value).__name__}, expected AppSpec."
        )
    if mode == "webrtc" and not isinstance(value.run, WebRTCRunSpec):
        raise TypeError("WebRTC mode requires WebRTCRunSpec from the provider.")
    if mode == "mp4" and not isinstance(value.run, Mp4RunSpec):
        raise TypeError("MP4 mode requires Mp4RunSpec from the provider.")
    return value


def _launch_mode(
    *,
    mode: str,
    runtime: InferenceRuntime,
    app_spec: AppSpec,
    args: argparse.Namespace,
    environment: "_Environment",
) -> tuple[OutputArtifact, ...]:
    """Launch the host-owned presentation path selected by the user."""
    if mode == "webrtc":
        assert isinstance(app_spec.run, WebRTCRunSpec)
        return _run_webrtc(
            runtime=runtime,
            config=app_spec.config,
            run_spec=app_spec.run,
            args=args,
            environment=environment,
        )
    if mode == "mp4":
        assert isinstance(app_spec.run, Mp4RunSpec)
        return _run_mp4(
            runtime=runtime,
            config=app_spec.config,
            run_spec=app_spec.run,
            output_path=args.output,
            environment=environment,
        )
    raise AssertionError(f"Unsupported presentation mode: {mode!r}.")


def _run_webrtc(
    *,
    runtime: InferenceRuntime,
    config: AppConfig,
    run_spec: WebRTCRunSpec,
    args: argparse.Namespace,
    environment: "_Environment",
) -> tuple[OutputArtifact, ...]:
    """Run the WebRTC serving path and close its runtime.

    Args:
        runtime: Initialized application runtime.
        config: Model identity and video presentation configuration.
        run_spec: Initial conditioning for live sessions.
        args: Parsed host and WebRTC arguments.
        environment: Initialized process and distributed environment.

    Returns:
        An empty tuple because WebRTC does not create file artifacts.
    """
    try:
        serve_webrtc(
            runtime=runtime,
            config=config,
            initial_input=run_spec.initial_input,
            options=WebRTCOptions(
                host=args.host,
                port=args.port,
            ),
            device=environment.device,
            world_rank=environment.world_rank,
        )
    finally:
        runtime.close()
    return ()


def _run_mp4(
    *,
    runtime: InferenceRuntime,
    config: AppConfig,
    run_spec: Mp4RunSpec,
    output_path: Path,
    environment: "_Environment",
) -> tuple[OutputArtifact, ...]:
    """Run the finite MP4 generation path and close all owned resources.

    Args:
        runtime: Initialized application runtime.
        config: Model identity and video presentation configuration.
        run_spec: Initial conditioning and finite generation length.
        output_path: Destination for the generated MP4.
        environment: Initialized process and distributed environment.

    Returns:
        Artifacts emitted by the file output target.
    """

    with ExitStack() as resources:
        resources.callback(runtime.close)
        output = FileOutput(
            path=output_path,
            fps=config.fps,
            output_layout=config.output_layout,
            enabled=environment.world_rank == 0,
        )
        output_closed = False

        def close_output() -> None:
            if not output_closed:
                output.close()

        resources.callback(close_output)
        output.open()
        session = runtime.start_session(run_spec.initial_input)
        resources.callback(session.close)
        for _ in range(run_spec.total_steps):
            request = session.next_step_request()
            if request is None:
                raise RuntimeError(
                    "Runtime session ended before the requested MP4 step count."
                )
            output.write(session.step(InferenceInput()))
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
    # File modes return persistent artifacts whose URI is the output location;
    # live modes such as WebRTC return no artifacts.
    for artifact in artifacts:
        print(artifact.uri)
