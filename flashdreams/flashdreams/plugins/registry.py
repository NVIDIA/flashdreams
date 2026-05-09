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

    Returns:
        A dict keyed by ``cfg.runner_name``.
    """
    runners: dict[str, RunnerConfig] = {}

    discovered = entry_points(group=ENTRY_POINT_GROUP)
    for ep in discovered:
        try:
            value = ep.load()
        except Exception:  # noqa: BLE001 - keep CLI alive on bad plugins
            logger.warning(
                f"Failed to load flashdreams runner entry point "
                f"{ep.name!r} from {ep.value!r}:\n{traceback.format_exc()}"
            )
            continue
        if callable(value) and not isinstance(value, RunnerConfig):
            # Allow factories that return a config (matches nerfstudio's
            # env-var convention; equally useful at the entry point).
            try:
                value = value()
            except Exception:  # noqa: BLE001
                logger.warning(
                    f"Calling runner entry point {ep.name!r} as a factory "
                    f"raised:\n{traceback.format_exc()}"
                )
                continue
        if not isinstance(value, RunnerConfig):
            logger.warning(
                f"Skipping runner entry point {ep.name!r}: expected a "
                f"RunnerConfig, got {type(value).__name__}."
            )
            continue
        cfg = cast(RunnerConfig, value)
        runners[cfg.runner_name] = cfg

    raw = os.environ.get(ENV_VAR)
    if raw:
        for definition in raw.split(","):
            definition = definition.strip()
            if not definition:
                continue
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
                runners[attr_value.runner_name] = attr_value
            except Exception:  # noqa: BLE001
                logger.warning(
                    f"Failed to load runner entry {definition!r} from "
                    f"{ENV_VAR}:\n{traceback.format_exc()}"
                )

    return runners
