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

"""Smoke tests for the central runner registry.

Cheap import-time checks that catch the common mis-registrations:
duplicate keys, dict key vs ``runner_name`` drift, recipes that
forgot to add their runners to the aggregator, and runner_name vs
pipeline.recipe_name drift (which would surface as confusing
``flashdreams-run <slug>`` failures).
"""

from __future__ import annotations

from flashdreams.configs.runner_configs import (
    BUILTIN_DESCRIPTIONS,
    BUILTIN_RUNNERS,
)
from flashdreams.recipes.template.runner import TEMPLATE_RUNNERS
from flashdreams.recipes.wan.runner import WAN21_RUNNERS


def test_builtin_runners_keys_match_runner_name() -> None:
    """Every registered runner's key must equal its ``runner_name``."""
    assert BUILTIN_RUNNERS, "BUILTIN_RUNNERS is empty -- aggregator broken?"
    mismatched = {
        key: cfg.runner_name
        for key, cfg in BUILTIN_RUNNERS.items()
        if cfg.runner_name != key
    }
    assert not mismatched, (
        f"BUILTIN_RUNNERS keys diverged from runner_name: {mismatched}"
    )


def test_builtin_runners_covers_every_runner_dict() -> None:
    """Each per-recipe ``<NAME>_RUNNERS`` dict must be merged in full.

    Catches the case where a new recipe added a ``<NAME>_RUNNERS``
    dict but forgot to wire it into the aggregator.
    """
    expected = {
        **TEMPLATE_RUNNERS,
        **WAN21_RUNNERS,
    }
    missing = set(expected) - set(BUILTIN_RUNNERS)
    assert not missing, f"BUILTIN_RUNNERS missing slugs: {sorted(missing)}"
    # And the converse: nothing extra slipped in via a later edit.
    extra = set(BUILTIN_RUNNERS) - set(expected)
    assert not extra, (
        f"BUILTIN_RUNNERS has slugs outside the per-recipe dicts: {sorted(extra)}"
    )


def test_builtin_runners_unique_runner_names() -> None:
    """No two registered runners share a ``runner_name``."""
    seen: dict[str, int] = {}
    for cfg in BUILTIN_RUNNERS.values():
        seen[cfg.runner_name] = seen.get(cfg.runner_name, 0) + 1
    duplicates = {name: count for name, count in seen.items() if count > 1}
    assert not duplicates, (
        f"duplicate runner_name in BUILTIN_RUNNERS: {duplicates}"
    )


def test_runner_name_mirrors_pipeline_recipe_name() -> None:
    """``runner_name`` must equal ``pipeline.recipe_name`` by convention.

    The CLI's contract is "``flashdreams-run <recipe_name>`` runs that recipe";
    a divergence here would silently rename one slug and break that
    contract. Per-runner literals are free to opt out, but the in-tree
    set must hold the line.
    """
    drifted = {
        key: (cfg.runner_name, cfg.pipeline.recipe_name)
        for key, cfg in BUILTIN_RUNNERS.items()
        if cfg.runner_name != cfg.pipeline.recipe_name
    }
    assert not drifted, (
        f"runner_name != pipeline.recipe_name (CLI contract): {drifted}"
    )


def test_builtin_descriptions_cover_runners() -> None:
    """Every shipped runner must have a one-line description, and vice versa.

    The CLI surfaces ``BUILTIN_DESCRIPTIONS`` next to every subcommand,
    so a missing entry shows up as an empty help line. A stale entry
    is a sign someone removed a runner and forgot to clean up the
    description.
    """
    missing = set(BUILTIN_RUNNERS) - set(BUILTIN_DESCRIPTIONS)
    assert not missing, (
        f"BUILTIN_DESCRIPTIONS missing entries for: {sorted(missing)}"
    )
    extra = set(BUILTIN_DESCRIPTIONS) - set(BUILTIN_RUNNERS)
    assert not extra, (
        f"BUILTIN_DESCRIPTIONS has stale entries for: {sorted(extra)}"
    )
    empty = [k for k, v in BUILTIN_DESCRIPTIONS.items() if not v.strip()]
    assert not empty, f"BUILTIN_DESCRIPTIONS has empty strings for: {empty}"
