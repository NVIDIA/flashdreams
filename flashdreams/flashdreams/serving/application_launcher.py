# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Application package discovery and execution."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from functools import cache
from importlib.metadata import entry_points
from pathlib import Path
from typing import cast

from flashdreams.runtime.demo import DemoSpec
from flashdreams.runtime.demo.application import (
    ApplicationMode,
    FlashDreamsApplication,
)

from .io_factory import IOOptions, io_factories

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
    factory = io_factories()[mode]
    if factory.input_mode not in application.supported_input_modes():
        raise ValueError(
            f"Application does not support input mode {factory.input_mode!r}."
        )
    output = factory.create_output(
        application,
        IOOptions(output_path=output_path, host=host, port=port),
    )
    spec = DemoSpec(
        model_id=application.model_id,
        input_mode=factory.input_mode,
        output=output,
        scenario=application.scenario,
        config=application.config,
    )
    return factory.run(application, spec)


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
