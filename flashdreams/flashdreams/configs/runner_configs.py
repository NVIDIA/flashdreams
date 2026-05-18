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

"""Aggregator for in-tree + plugin-discovered runner configs.

Each in-tree recipe's ``config.py`` (or ``config/<variant>.py``)
self-registers its slugs via
:func:`flashdreams.configs.registry.register_runner` at module-import
time, alongside the matching pipeline configs. This module just
imports those config modules for their side effects and exposes
:func:`all_runners`, which layers plugin discoveries on top of the
populated :data:`flashdreams.configs.registry._SUPPORTED_RUNNERS`
registry.

Recipes that haven't been wrapped into a runner stay reachable via
direct per-recipe imports
(``from flashdreams.recipes.<name>.config import <NAME>_CONFIGS``) for
serving / tests / programmatic use, but they do not appear here and
are not ``flashdreams-run`` subcommands. Runners are opt-in.

Adding a new in-tree runner:

1. Author ``recipes/<name>/runner.py`` with the :class:`Runner`
   subclass and its :class:`RunnerConfig` dataclass.
2. In the matching ``recipes/<name>/config.py``, define one
   ``RunnerConfig`` literal per shipped variant (each with a
   non-empty ``description``) alongside the pipeline-config literals,
   collect them in a ``<NAME>_RUNNERS`` dict, and loop
   :func:`~flashdreams.configs.registry.register_runner` over its
   items with ``source="builtin"``.
3. Add a one-line ``import flashdreams.recipes.<name>.config`` below
   so this module triggers the side effect at CLI startup. The smoke
   test in ``tests/test_recipe_configs.py`` enforces parity.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
from typing import Any

import tyro

# Each import below triggers the recipe's `register_runner(..., source="builtin")`
# calls against ``_SUPPORTED_RUNNERS``. Listed explicitly (no auto-walk
# of ``flashdreams.recipes``) so the in-tree inventory is one grep away.
import flashdreams.recipes.template.config  # noqa: F401, E402
from flashdreams.configs.registry import (
    _SUPPORTED_RUNNERS,
    register_runner,
    supported_runners,
)
from flashdreams.infra.runner import RunnerConfig
from flashdreams.plugins.registry import discover_runners


def _sort(
    runners: Mapping[str, RunnerConfig],
) -> OrderedDict[str, RunnerConfig]:
    """Sort the mapping by ``runner_name`` so subcommand listings are stable."""
    return OrderedDict(sorted(runners.items()))


def all_runners() -> OrderedDict[str, RunnerConfig]:
    """Return the runner registry covering builtin + plugin sources.

    Built-in runners always win over a same-slug plugin: the plugin
    layer goes through :func:`register_runner` with ``source="plugin"``,
    which logs and skips collisions. The result is sorted alphabetically
    by ``runner_name`` for stable subcommand listings.
    """
    runners = dict(_SUPPORTED_RUNNERS)
    for name, cfg in discover_runners().items():
        register_runner(name, cfg, source="plugin", target=runners)
    return _sort(runners)


def _annotated_base_runner_union():
    """Build the tyro subcommand union over every discovered runner.

    Built lazily so importing this module never pays the entry-point
    discovery cost (or its log noise) unless the CLI actually runs.

    The marker stack mirrors nerfstudio's ``ns-train``:

    * ``SuppressFixed`` -- hide the ``_target = (fixed)`` rows that
      every category-base config ships with, keeping the help text
      focused on user-overridable knobs.
    * ``FlagConversionOff`` -- don't auto-flip booleans into
      ``--no-foo`` flags inside nested configs.
    """
    runners = all_runners()
    descriptions = {k: cfg.description for k, cfg in runners.items()}
    # ``Any`` because ty rejects the runtime tyro union as a type-form
    # arg to the ``SuppressFixed`` / ``FlagConversionOff`` markers below.
    subcommand_union: Any = tyro.extras.subcommand_type_from_defaults(
        defaults=dict(runners),
        descriptions=descriptions,
        # Drop the ``runner:`` namespace prefix so users type
        # ``flashdreams-run template-offline``.
        prefix_names=False,
        sort_subcommands=True,
    )
    return tyro.conf.SuppressFixed[tyro.conf.FlagConversionOff[subcommand_union]]


# Re-export ``register_runner`` / ``supported_runners`` here too so
# downstream code can stick to a single ``flashdreams.configs.runner_configs``
# import path if desired.
__all__ = [
    "_annotated_base_runner_union",
    "all_runners",
    "register_runner",
    "supported_runners",
]
