# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The applications installed here, and finding the one a runner was asked for.

The registry is the entry points an install wrote down, rather than anything
this holds.
"""

import importlib
from importlib.metadata import EntryPoint, entry_points
from typing import Any

from flashdreams.api_v2.application import IApplication

APPLICATION_ENTRY_POINT_GROUP = "flashdreams.applications_v2"
"""Entry-point group whose values expose a zero-argument ``create_app`` factory.

Separate from the v1 ``flashdreams.applications`` group, which resolves to
``IFlashDreamsApplication`` and would refuse a v2 application registered there.
"""


def registered_application_slugs() -> tuple[str, ...]:
    """Return the installed application slugs, in a stable order."""
    return tuple(
        sorted(
            {item.name for item in entry_points(group=APPLICATION_ENTRY_POINT_GROUP)}
        )
    )


def create_application(slug: str) -> IApplication:
    """Return a new, uninitialized application for ``slug``.

    A registered entry point is preferred. Failing that the slug is read as a
    module name, so an integration that has not registered itself is still
    reachable by the name of the package it ships.

    Args:
        slug: Registered application name, such as ``t2v-self-forcing``, or an
            importable module exposing ``create_app``.

    Raises:
        ValueError: ``slug`` is empty.
        LookupError: Nothing installed matches ``slug``.
        TypeError: The factory returned something other than an
            :class:`IApplication`, or the module has no ``create_app``.
    """
    if not slug.strip():
        raise ValueError("An application slug is required.")

    for entry_point in entry_points(group=APPLICATION_ENTRY_POINT_GROUP):
        if entry_point.name == slug:
            return _from_entry_point(entry_point)

    module = _import_application_module(slug)
    factory = getattr(module, "create_app", None)
    if not callable(factory):
        raise TypeError(
            f"Application module {module.__name__!r} does not expose create_app()."
        )
    return _validated(factory(), origin=module.__name__)


def _from_entry_point(entry_point: EntryPoint) -> IApplication:
    """Build the application an entry point points at."""
    value = entry_point.load()
    return _validated(value() if callable(value) else value, origin=entry_point.value)


def _validated(value: Any, *, origin: str) -> IApplication:
    """Return ``value`` if it is an application, and say what it was if not.

    An integration still on the v1 contract lands here.
    """
    if not isinstance(value, IApplication):
        raise TypeError(
            f"Application factory {origin!r} returned {type(value).__name__}; "
            "expected an IApplication."
        )
    return value


def _import_application_module(slug: str) -> Any:
    """Import the module a slug names.

    Raises:
        LookupError: There is no such module, and no entry point matched either.
    """
    module_name = slug.replace("-", "_")
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        # A module that exists but imports something missing is a broken
        # install rather than an unknown slug.
        if exc.name != module_name:
            raise

    installed = ", ".join(registered_application_slugs())
    raise LookupError(
        f"No FlashDreams v2 application matches {slug!r}. "
        f"Installed applications: {installed or '(none)'}."
    )
