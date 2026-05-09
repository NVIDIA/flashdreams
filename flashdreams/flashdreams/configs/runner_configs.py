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

"""Central registry of runner configs (in-tree + plugin-discovered).

Mirrors nerfstudio's ``configs/method_configs.py``:

* Each in-tree recipe ships a
  ``<NAME>_RUNNERS: dict[str, RunnerConfig]`` dict in
  ``recipes/<name>/runner.py`` whose values carry their CLI subcommand
  description on ``cfg.description``.
* This module merges them into ``BUILTIN_RUNNERS``.
* :func:`all_runners` then layers external :class:`RunnerConfig`
  discoveries on top so the ``flashdreams-run`` CLI sees a single
  sorted dict.

There is no central pipeline-config registry: a recipe that hasn't
been wrapped into a runner stays reachable via direct per-recipe
imports (``from flashdreams.recipes.<name>.config import <NAME>_CONFIGS``)
for serving / tests / programmatic use, but it does not appear here
and is not a ``flashdreams-run`` subcommand. That's the soft contract:
runners are opt-in.

Adding a new runner:

1. Author ``recipes/<name>/runner.py`` with one ``RunnerConfig``
   literal per shipped variant (each with a non-empty ``description``)
   and a ``<NAME>_RUNNERS`` dict.
2. Add a one-line import + spread into :data:`BUILTIN_RUNNERS`. The
   smoke test in ``tests/test_recipe_configs.py`` enforces parity.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
from typing import Any

import tyro

from flashdreams.infra.runner import RunnerConfig
from flashdreams.plugins.registry import discover_runners
from flashdreams.recipes.alpadreams.runner import ALPADREAMS_RUNNERS
from flashdreams.recipes.lingbot_world.runner import LINGBOT_WORLD_RUNNERS
from flashdreams.recipes.template.runner import TEMPLATE_RUNNERS
from flashdreams.recipes.wan.runner import WAN21_RUNNERS
from flashdreams.recipes.wan.runner_causal_wan21 import CAUSAL_WAN21_RUNNERS
from flashdreams.recipes.wan.runner_causal_wan22 import CAUSAL_WAN22_RUNNERS


def _merge(
    *dicts: Mapping[str, RunnerConfig],
) -> dict[str, RunnerConfig]:
    """Merge per-recipe runner dicts and reject duplicate ``runner_name`` keys.

    Duplicates would silently shadow a registered runner; rejecting
    them forces the offending recipe to pick a unique slug at
    definition time.
    """
    merged: dict[str, RunnerConfig] = {}
    for d in dicts:
        for key, cfg in d.items():
            if key in merged:
                raise ValueError(
                    f"Duplicate runner_name {key!r} in BUILTIN_RUNNERS: "
                    f"already registered as {type(merged[key]).__name__}, "
                    f"new entry is {type(cfg).__name__}."
                )
            merged[key] = cfg
    return merged


BUILTIN_RUNNERS: dict[str, RunnerConfig] = _merge(
    TEMPLATE_RUNNERS,
    WAN21_RUNNERS,
    CAUSAL_WAN21_RUNNERS,
    CAUSAL_WAN22_RUNNERS,
    ALPADREAMS_RUNNERS,
    LINGBOT_WORLD_RUNNERS,
)
"""Every shipped runner config, keyed by ``runner_name``."""


def merge_runners(
    runners: Mapping[str, RunnerConfig],
    new_runners: Mapping[str, RunnerConfig],
    overwrite: bool = True,
) -> OrderedDict[str, RunnerConfig]:
    """Merge ``new_runners`` into ``runners``.

    Mirrors :func:`nerfstudio.configs.method_configs.merge_methods`. The
    ``overwrite=False`` form is used by the layered :func:`all_runners`
    loader so a built-in is preferred over a same-slug plugin (and a
    plugin is preferred over the env-var fallback).
    """
    out: OrderedDict[str, RunnerConfig] = OrderedDict(runners)
    for k, v in new_runners.items():
        if overwrite or k not in out:
            out[k] = v
    return out


def _sort(
    runners: Mapping[str, RunnerConfig],
) -> OrderedDict[str, RunnerConfig]:
    """Sort the mapping by ``runner_name`` so subcommand listings are stable."""
    return OrderedDict(sorted(runners.items()))


def all_runners() -> OrderedDict[str, RunnerConfig]:
    """Return the runner registry covering builtin + plugin sources.

    Built-in runners always win over a same-named plugin (that's
    ``overwrite=False`` for the discovery layer): an external package
    cannot silently shadow a shipped slug.
    """
    discovered = discover_runners()
    merged = merge_runners(BUILTIN_RUNNERS, discovered, overwrite=False)
    return _sort(merged)


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
