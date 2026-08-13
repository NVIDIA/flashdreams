# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Application package discovery and execution."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from functools import cache
from importlib.metadata import EntryPoint, entry_points

from flashdreams.runtime.application import FlashDreamsApplication
from flashdreams.runtime.application_runner import ApplicationRunner

from .io_handlers import (
    io_handler_entry_points,
    load_io_handler,
)

APPLICATION_ENTRY_POINT_GROUP = "flashdreams.applications"


ApplicationFactory = Callable[[Sequence[str]], FlashDreamsApplication]


@cache
def application_entry_points() -> dict[str, EntryPoint]:
    applications: dict[str, EntryPoint] = {}
    for entry_point in entry_points(group=APPLICATION_ENTRY_POINT_GROUP):
        if entry_point.name in applications:
            raise ValueError(f"Duplicate application {entry_point.name!r}.")
        applications[entry_point.name] = entry_point
    return applications


def run_application_from_argv(argv: Sequence[str]) -> bool:
    if not argv:
        return False
    entry_point = application_entry_points().get(argv[0])
    if entry_point is None:
        return False
    factory = entry_point.load()
    if not callable(factory):
        raise TypeError(f"Application entry point {argv[0]!r} is not callable.")
    remaining = list(argv[1:])
    no_instantiate = _pop_flag(remaining, "--no-instantiate")
    io_handler_name = _pop_io_handler_name(remaining)
    if io_handler_name is not None:
        io_handler, app_args = load_io_handler(io_handler_name, remaining)
        application = factory(app_args)
    else:
        application = factory(remaining)
    if not isinstance(application, FlashDreamsApplication):
        raise TypeError(f"Application factory {argv[0]!r} returned an invalid object.")
    if io_handler_name is None:
        io_handler_name = application.default_io_handler
        io_handler, unused = load_io_handler(io_handler_name, ())
        if unused:
            raise RuntimeError("Default IO handler left unexpected arguments.")
    if no_instantiate:
        print(f"Resolved application: {application.application_name!r}")
        print(f"Selected IO handler: {io_handler_name}")
        return True
    ApplicationRunner(application=application, io_handler=io_handler).run()
    return True


def _pop_io_handler_name(args: list[str]) -> str | None:
    if not args or args[0] not in io_handler_entry_points():
        return None
    return args.pop(0)


def _pop_flag(args: list[str], name: str) -> bool:
    if name not in args:
        return False
    args.remove(name)
    return True


__all__ = [
    "APPLICATION_ENTRY_POINT_GROUP",
    "ApplicationFactory",
    "application_entry_points",
    "run_application_from_argv",
]
