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

"""Discover ``RunnerSpecification`` plugins (entry-point + env-var)."""

from __future__ import annotations

import importlib
import os
import sys
import traceback
from typing import cast

from loguru import logger

from flashdreams.infra.runner import RunnerConfig
from flashdreams.plugins.types import RunnerSpecification

if sys.version_info < (3, 10):
    # The 3.12 floor in pyproject.toml means we always take the modern
    # branch, but match nerfstudio's shape so a future Python downgrade
    # is one import swap.
    from importlib_metadata import entry_points  # type: ignore[import-not-found]
else:
    from importlib.metadata import entry_points

ENTRY_POINT_GROUP = "flashdreams.runner_configs"
"""Entry-point group external packages register ``RunnerSpecification``
instances under (matches nerfstudio's ``nerfstudio.method_configs``
naming)."""

ENV_VAR = "FLASHDREAMS_RUNNER_CONFIGS"
"""Env-var backdoor for in-development specs that aren't installed yet.

Format: ``slug=module.path:attribute,slug2=other.module:attr``. The
attribute is loaded with ``getattr(import_module(module), attr)``; if
it is callable it is invoked with no arguments to obtain the spec. The
``slug=`` prefix is purely for log readability -- the registry key
always comes from ``spec.config.runner_name``."""


def discover_runners() -> tuple[dict[str, RunnerConfig], dict[str, str]]:
    """Discover externally-registered runners.

    Looks at every entry point under :data:`ENTRY_POINT_GROUP` and at
    the ``slug=module:attr`` pairs in :data:`ENV_VAR`. Bad entries are
    logged and skipped -- the CLI must keep working even when a third-
    party plugin is broken.

    Returns:
        A pair ``(runners, descriptions)`` keyed by
        ``spec.config.runner_name``.
    """
    runners: dict[str, RunnerConfig] = {}
    descriptions: dict[str, str] = {}

    discovered = entry_points(group=ENTRY_POINT_GROUP)
    for ep in discovered:
        try:
            spec = ep.load()
        except Exception:  # noqa: BLE001 - keep CLI alive on bad plugins
            logger.warning(
                f"Failed to load flashdreams runner entry point "
                f"{ep.name!r} from {ep.value!r}:\n{traceback.format_exc()}"
            )
            continue
        if callable(spec) and not isinstance(spec, RunnerSpecification):
            # Allow factories that return a spec (matches nerfstudio's
            # env-var convention; equally useful at the entry point).
            try:
                spec = spec()
            except Exception:  # noqa: BLE001
                logger.warning(
                    f"Calling runner entry point {ep.name!r} as a factory "
                    f"raised:\n{traceback.format_exc()}"
                )
                continue
        if not isinstance(spec, RunnerSpecification):
            logger.warning(
                f"Skipping runner entry point {ep.name!r}: expected a "
                f"RunnerSpecification, got {type(spec).__name__}."
            )
            continue
        spec = cast(RunnerSpecification, spec)
        runners[spec.config.runner_name] = spec.config
        descriptions[spec.config.runner_name] = spec.description

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
                    f"Loading runner {slug!r} from {module_name}:{attr} "
                    f"({ENV_VAR})"
                )
                attr_value = getattr(importlib.import_module(module_name), attr)
                if callable(attr_value) and not isinstance(
                    attr_value, RunnerSpecification
                ):
                    attr_value = attr_value()
                if not isinstance(attr_value, RunnerSpecification):
                    raise TypeError(
                        f"{module_name}:{attr} is not a RunnerSpecification "
                        f"(got {type(attr_value).__name__})."
                    )
                runners[attr_value.config.runner_name] = attr_value.config
                descriptions[attr_value.config.runner_name] = attr_value.description
            except Exception:  # noqa: BLE001
                logger.warning(
                    f"Failed to load runner entry {definition!r} from "
                    f"{ENV_VAR}:\n{traceback.format_exc()}"
                )

    return runners, descriptions
