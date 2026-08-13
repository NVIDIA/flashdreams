# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Command-line shell for independently installed FlashDreams applications."""

from __future__ import annotations

import argparse
import importlib
from contextlib import ExitStack
from dataclasses import dataclass
from importlib import metadata
from typing import Sequence

import torch

from flashdreams.runtime import OutputArtifact, StepResult
from flashdreams.runtime.demo.bootstrap import (
    configure_logging,
    initialize_cuda_distributed,
)

from .contracts import (
    Application,
    ApplicationArguments,
    InputHandler,
    OutputHandler,
    Runtime,
    Session,
)
from .modes import MODE_NAMES, add_mode_arguments, create_io_handler


@dataclass(frozen=True, slots=True)
class _ApplicationAndMode:
    """Top-level route parsed before loading an application."""

    application: str
    """Installed application distribution name."""

    mode: str
    """Selected runner I/O mode."""

    remaining_argv: tuple[str, ...]
    """Arguments delegated to the selected mode and application."""


@dataclass(frozen=True, slots=True)
class _Environment:
    """Initialized device and distributed process information."""

    device: str
    """Resolved runtime device for this process."""

    world_rank: int
    """Global distributed rank."""

    world_size: int
    """Number of distributed processes."""


def build_parser(application: str, mode: str) -> argparse.ArgumentParser:
    """Build a parser containing runner and selected-mode arguments.

    Args:
        application: Application name displayed in command usage.
        mode: Selected I/O mode.

    Returns:
        Parser for the application factory to extend and invoke.
    """
    parser = argparse.ArgumentParser(prog=f"flashdreams-runner {application} {mode}")
    add_mode_arguments(parser, mode)
    return parser


def load_application(distribution_name: str) -> Application:
    """Load an installed module that satisfies the application ABI."""
    try:
        distribution = metadata.distribution(distribution_name)
    except metadata.PackageNotFoundError as exc:
        raise ValueError(
            f"Application distribution {distribution_name!r} is not installed."
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
        if isinstance(module, Application):
            return module
        incompatible.append(candidate)
    if incompatible:
        names = ", ".join(repr(name) for name in incompatible)
        raise TypeError(
            f"Application distribution {distribution_name!r} exposes module(s) "
            f"{names}, but none satisfy Application. An application must define "
            "create_runtime(arguments)."
        )
    raise ValueError(
        f"Application {distribution_name!r} does not expose an importable module."
    )


def run(argv: Sequence[str] | None = None) -> tuple[OutputArtifact, ...]:
    """Create an application runtime and dispatch it to one I/O mode.

    Args:
        argv: Command-line arguments; ``None`` reads the process arguments.

    Returns:
        Persistent artifacts produced by the selected mode.
    """
    route = _parse_application_and_mode(argv)
    application = load_application(route.application)
    arguments = ApplicationArguments(
        mode=route.mode,
        parser=build_parser(route.application, route.mode),
        argv=route.remaining_argv,
    )
    runtime = _require_runtime(
        application.create_runtime(arguments),
        application_name=route.application,
    )
    options = arguments.options
    environment = _initialize_environment(options.device)
    io_handler = create_io_handler(
        route.mode,
        options,
        device=environment.device,
        world_rank=environment.world_rank,
    )
    try:
        runtime.initialize(
            device=environment.device,
            io_handler=io_handler,
        )
        return io_handler.run(runtime, _drive_session)
    finally:
        runtime.destroy()


def _parse_application_and_mode(
    argv: Sequence[str] | None,
) -> _ApplicationAndMode:
    """Parse application and mode while preserving all remaining arguments."""
    parser = argparse.ArgumentParser(prog="flashdreams-runner", add_help=False)
    parser.add_argument("application", help="Installed application distribution")
    parser.add_argument("mode", choices=MODE_NAMES)
    args, remaining_argv = parser.parse_known_args(argv)
    return _ApplicationAndMode(
        application=args.application,
        mode=args.mode,
        remaining_argv=tuple(remaining_argv),
    )


def _require_runtime(value: object, *, application_name: str) -> Runtime:
    """Validate the application factory result before initialization."""
    if not isinstance(value, Runtime):
        raise TypeError(
            f"Application {application_name!r} create_runtime() returned "
            f"{type(value).__name__}, expected Runtime."
        )
    return value


def _drive_session(
    runtime: Runtime,
    input_handler: InputHandler,
    output_handler: OutputHandler,
) -> tuple[OutputArtifact, ...]:
    """Drive one application session through a pair of I/O handlers."""
    with ExitStack() as resources:
        input_handler.open()
        resources.callback(input_handler.close)

        output_handler.open(runtime.config)
        output_closed = False

        def close_output() -> None:
            if not output_closed:
                output_handler.close()

        resources.callback(close_output)
        session = runtime.create_session(input_handler.initial_input())
        if not isinstance(session, Session):
            raise TypeError(
                "Runtime.create_session() must return Session, got "
                f"{type(session).__name__}."
            )
        resources.callback(session.destroy)

        while (inputs := input_handler.read()) is not None:
            result = session.step(inputs)
            if not isinstance(result, StepResult):
                raise TypeError(
                    "Session.step() must return StepResult, got "
                    f"{type(result).__name__}."
                )
            output_handler.write(result)

        artifacts = tuple(output_handler.close())
        output_closed = True
        return artifacts


def _initialize_environment(device: str) -> _Environment:
    """Initialize logging, CUDA placement, and distributed process state."""
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
    """Run the console-script entry point."""
    artifacts = run()
    # Persistent modes return artifact URIs; live and headless modes return none.
    for artifact in artifacts:
        print(artifact.uri)


__all__ = ["build_parser", "load_application", "main", "run"]
