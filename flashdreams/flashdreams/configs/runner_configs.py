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

Mirrors nerfstudio's ``configs/method_configs.py`` end-to-end:

* Each in-tree recipe ships a
  ``<NAME>_RUNNERS: dict[str, RunnerConfig]`` dict in
  ``recipes/<name>/runner.py``.
* This module merges them into ``BUILTIN_RUNNERS`` and pairs them with
  human-readable strings in ``BUILTIN_DESCRIPTIONS``.
* :func:`all_runners` then layers external
  :class:`~flashdreams.plugins.types.RunnerSpecification` discoveries on
  top so the ``flashdreams-run`` CLI sees a single sorted dict.

There is no central pipeline-config registry: a recipe that hasn't
been wrapped into a runner stays reachable via direct per-recipe
imports (``from flashdreams.recipes.<name>.config import <NAME>_CONFIGS``)
for serving / tests / programmatic use, but it does not appear here
and is not a ``flashdreams-run`` subcommand. That's the soft contract:
runners are opt-in.

Adding a new runner:

1. Author ``recipes/<name>/runner.py`` with one ``RunnerConfig``
   literal per shipped variant and a ``<NAME>_RUNNERS`` dict.
2. Add a one-line import + spread here, and one entry per slug in
   :data:`BUILTIN_DESCRIPTIONS` (the smoke test in
   ``tests/test_recipe_configs.py`` enforces parity).
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping

import tyro

from flashdreams.infra.runner import RunnerConfig
from flashdreams.plugins.registry import discover_runners
from flashdreams.recipes.template.runner import TEMPLATE_RUNNERS
from flashdreams.recipes.wan.runner import WAN21_RUNNERS


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
)
"""Every shipped runner config, keyed by ``runner_name``.

Currently covers the ``template`` and non-streaming ``wan21`` recipes;
the remaining recipes (``causal_wan21``, ``causal_wan22``,
``alpadreams``, ``lingbot_world``) still ship via direct per-recipe
imports of their ``<NAME>_CONFIGS`` dict and are not yet
``flashdreams-run``-able. Migrating them is the follow-up task -- author a
``recipes/<name>/runner.py`` and append the dict to ``_merge`` above."""

BUILTIN_DESCRIPTIONS: dict[str, str] = {
    # template -- reference recipe used by the developer skill.
    "template-offline": (
        "Reference template recipe: one-shot offline diffusion (synthetic inputs)."
    ),
    "template-autoregressive": (
        "Reference template recipe: streaming AR diffusion with sliding-window cache."
    ),
    "template-autoregressive-compiled": (
        "Reference template recipe: AR variant with torch.compile + CUDA graphs."
    ),
    # wan 2.1 -- baseline non-streaming Wan 2.1 demo.
    "wan21-t2v-1.3b-480p": (
        "Wan 2.1 T2V 1.3B at 480p (single AR step, prompt-only)."
    ),
    "wan21-i2v-14b-480p": (
        "Wan 2.1 I2V 14B at 480p (single AR step, prompt + first-frame)."
    ),
}
"""One-line description per built-in runner. Keys must equal
``BUILTIN_RUNNERS`` (asserted by
``tests/test_recipe_configs.py::test_builtin_descriptions_cover_runners``)."""


def merge_runners(
    runners: Mapping[str, RunnerConfig],
    descriptions: Mapping[str, str],
    new_runners: Mapping[str, RunnerConfig],
    new_descriptions: Mapping[str, str],
    overwrite: bool = True,
) -> tuple[OrderedDict[str, RunnerConfig], OrderedDict[str, str]]:
    """Merge ``new_runners`` into ``runners`` (and the same for descriptions).

    Mirrors :func:`nerfstudio.configs.method_configs.merge_methods`. The
    ``overwrite=False`` form is used by the layered :func:`all_runners`
    loader so a built-in is preferred over a same-slug plugin (and a
    plugin is preferred over the env-var fallback).
    """
    out_runners: OrderedDict[str, RunnerConfig] = OrderedDict(runners)
    out_descriptions: OrderedDict[str, str] = OrderedDict(descriptions)
    for k, v in new_runners.items():
        if overwrite or k not in out_runners:
            out_runners[k] = v
            out_descriptions[k] = new_descriptions.get(k, "")
    return out_runners, out_descriptions


def _sort(
    runners: Mapping[str, RunnerConfig],
    descriptions: Mapping[str, str],
) -> tuple[OrderedDict[str, RunnerConfig], OrderedDict[str, str]]:
    """Sort both mappings by ``runner_name`` so subcommand listings are stable."""
    return (
        OrderedDict(sorted(runners.items())),
        OrderedDict(sorted(descriptions.items())),
    )


def all_runners() -> tuple[
    OrderedDict[str, RunnerConfig], OrderedDict[str, str]
]:
    """Return ``(runners, descriptions)`` covering builtin + plugin sources.

    Built-in runners always win over a same-named plugin (that's
    ``overwrite=False`` for the discovery layer): an external package
    cannot silently shadow a shipped slug.
    """
    discovered_runners, discovered_descriptions = discover_runners()
    merged_runners, merged_descriptions = merge_runners(
        BUILTIN_RUNNERS,
        BUILTIN_DESCRIPTIONS,
        discovered_runners,
        discovered_descriptions,
        overwrite=False,
    )
    return _sort(merged_runners, merged_descriptions)


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
    runners, descriptions = all_runners()
    return tyro.conf.SuppressFixed[
        tyro.conf.FlagConversionOff[
            tyro.extras.subcommand_type_from_defaults(
                defaults=dict(runners),
                descriptions=dict(descriptions),
                # Drop the ``runner:`` namespace prefix so users type
                # ``flashdreams-run template-offline``, not
                # ``flashdreams-run runner:template-offline``.
                prefix_names=False,
                sort_subcommands=True,
            )
        ]
    ]
