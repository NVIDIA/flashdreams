# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Discover :class:`RunnerConfig` plugins (entry-point + env-var)."""

from __future__ import annotations

import importlib
import os
import sys
import traceback
from typing import cast

from loguru import logger

from flashdreams.infra.runner import RunnerConfig

if sys.version_info < (3, 10):
    from importlib_metadata import entry_points  # type: ignore[import-not-found]
else:
    from importlib.metadata import entry_points

ENTRY_POINT_GROUP = "flashdreams.runner_configs"
"""Entry-point group external packages register :class:`RunnerConfig`
instances under (matches nerfstudio's ``nerfstudio.method_configs``
naming)."""

ENV_VAR = "FLASHDREAMS_RUNNER_CONFIGS"
"""Env-var backdoor for in-development runners that aren't installed yet.

Format: ``slug=module.path:attribute,slug2=other.module:attr``. The
attribute is loaded with ``getattr(import_module(module), attr)``; if
it is callable (and not already a :class:`RunnerConfig`) it is invoked
with no arguments to obtain the config. The ``slug=`` prefix is purely
for log readability -- the registry key always comes from
``cfg.runner_name``."""


def discover_runners() -> dict[str, RunnerConfig]:
    """Discover externally-registered runner configs.

    Looks at every entry point under :data:`ENTRY_POINT_GROUP` and at
    the ``slug=module:attr`` pairs in :data:`ENV_VAR`. Bad entries are
    logged and skipped -- the CLI must keep working even when a third-
    party plugin is broken.

    Each loaded value must be a :class:`RunnerConfig` (or a zero-arg
    factory returning one); the subcommand description is read off
    ``cfg.description``.

    Resolution order:

    1. Entry points sorted by ``ep.name`` so the winner of a collision
       is deterministic across installs.
    2. ``FLASHDREAMS_RUNNER_CONFIGS`` env-var entries, in declared
       order.

    On a ``runner_name`` collision the *first* seen config wins; the
    later one is logged and skipped, including the ``module:attr``
    origin of both configs so the plugin author can find and rename the
    duplicate.

    Returns:
        A dict keyed by ``cfg.runner_name``.
    """
    runners: dict[str, RunnerConfig] = {}
    # Tracks the ``module:attr`` (entry-point) or ``slug=module:attr``
    # (env-var) origin of each accepted runner so we can name *both*
    # sides of a collision in the warning message.
    origins: dict[str, str] = {}

    def _accept(cfg: RunnerConfig, origin: str) -> None:
        """Insert ``cfg`` unless its ``runner_name`` is already taken."""
        existing = origins.get(cfg.runner_name)
        if existing is not None:
            logger.warning(
                f"Skipping runner {cfg.runner_name!r} from {origin}: "
                f"slug already registered by {existing}. Rename one of "
                f"the two configs to disambiguate."
            )
            return
        runners[cfg.runner_name] = cfg
        origins[cfg.runner_name] = origin

    # Sort entry points by name so the "first one wins" rule above is
    # reproducible -- importlib.metadata gives no ordering guarantee.
    discovered = sorted(entry_points(group=ENTRY_POINT_GROUP), key=lambda ep: ep.name)
    for ep in discovered:
        origin = f"entry point {ep.name!r} -> {ep.value}"
        module_name = ep.value.split(":", 1)[0]
        top_level_module = module_name.split(".", 1)[0]
        try:
            value = ep.load()
        except ModuleNotFoundError as exc:
            # Common/expected on partial installs (e.g. `uv run --project ...`):
            # metadata can still expose runner entry points for integrations not
            # present in the active env. If the missing module is the entry point's
            # own package namespace, silently skip this plugin at debug level.
            missing_name = exc.name or ""
            if (
                missing_name == top_level_module
                or missing_name == module_name
                or missing_name.startswith(f"{top_level_module}.")
            ):
                logger.debug(
                    f"Skipping unavailable flashdreams runner {origin}: "
                    f"module {missing_name!r} is not installed in this environment."
                )
                continue
            logger.debug(
                f"Failed to load flashdreams runner {origin}:\n{traceback.format_exc()}"
            )
            continue
        except Exception:  # noqa: BLE001 - keep CLI alive on bad plugins
            logger.debug(
                f"Failed to load flashdreams runner {origin}:\n{traceback.format_exc()}"
            )
            continue
        if callable(value) and not isinstance(value, RunnerConfig):
            # Allow factories that return a config (matches nerfstudio's
            # env-var convention; equally useful at the entry point).
            try:
                value = value()
            except Exception:  # noqa: BLE001
                logger.warning(
                    f"Calling runner {origin} as a factory raised:"
                    f"\n{traceback.format_exc()}"
                )
                continue
        if not isinstance(value, RunnerConfig):
            logger.warning(
                f"Skipping runner {origin}: expected a RunnerConfig, "
                f"got {type(value).__name__}."
            )
            continue
        _accept(cast(RunnerConfig, value), origin)

    raw = os.environ.get(ENV_VAR)
    if raw:
        for definition in raw.split(","):
            definition = definition.strip()
            if not definition:
                continue
            origin = f"{ENV_VAR} entry {definition!r}"
            try:
                slug, path = definition.split("=", 1)
                module_name, attr = path.split(":", 1)
                logger.info(
                    f"Loading runner {slug!r} from {module_name}:{attr} ({ENV_VAR})"
                )
                attr_value = getattr(importlib.import_module(module_name), attr)
                if callable(attr_value) and not isinstance(attr_value, RunnerConfig):
                    attr_value = attr_value()
                if not isinstance(attr_value, RunnerConfig):
                    raise TypeError(
                        f"{module_name}:{attr} is not a RunnerConfig "
                        f"(got {type(attr_value).__name__})."
                    )
                _accept(attr_value, origin)
            except Exception:  # noqa: BLE001
                logger.warning(
                    f"Failed to load runner from {origin}:\n{traceback.format_exc()}"
                )

    return runners
