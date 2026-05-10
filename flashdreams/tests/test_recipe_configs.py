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
forgot to add their runners to the aggregator, runner_name vs
pipeline.recipe_name drift (which would surface as confusing
``flashdreams-run <slug>`` failures), and missing CLI descriptions.
"""

from __future__ import annotations

# Importing ``runner_configs`` triggers each in-tree recipe's
# self-registration side effects. Without this import the registry
# would be empty when tests run in isolation.
import flashdreams.configs.runner_configs  # noqa: F401
from flashdreams.configs.registry import supported_runners
from flashdreams.recipes.alpadreams.config import ALPADREAMS_RUNNERS
from flashdreams.recipes.lingbot_world.config import LINGBOT_WORLD_RUNNERS
from flashdreams.recipes.template.config import TEMPLATE_RUNNERS
from flashdreams.recipes.wan.config.wan21 import WAN21_RUNNERS


def test_supported_runners_keys_match_runner_name() -> None:
    """Every registered runner's key must equal its ``runner_name``."""
    runners = supported_runners()
    assert runners, "supported_runners() is empty -- aggregator broken?"
    mismatched = {
        key: cfg.runner_name for key, cfg in runners.items() if cfg.runner_name != key
    }
    assert not mismatched, (
        f"supported_runners keys diverged from runner_name: {mismatched}"
    )


def test_supported_runners_covers_every_runner_dict() -> None:
    """Each per-recipe ``<NAME>_RUNNERS`` dict must be merged in full.

    Catches the case where a new recipe added a ``<NAME>_RUNNERS``
    dict but forgot to wire its ``runner.py`` into the aggregator.
    """
    runners = supported_runners()
    expected = {
        **TEMPLATE_RUNNERS,
        **WAN21_RUNNERS,
        **ALPADREAMS_RUNNERS,
        **LINGBOT_WORLD_RUNNERS,
    }
    missing = set(expected) - set(runners)
    assert not missing, f"supported_runners missing slugs: {sorted(missing)}"
    extra = set(runners) - set(expected)
    assert not extra, (
        f"supported_runners has slugs outside the per-recipe dicts: {sorted(extra)}"
    )


def test_supported_runners_unique_runner_names() -> None:
    """No two registered runners share a ``runner_name``."""
    seen: dict[str, int] = {}
    for cfg in supported_runners().values():
        seen[cfg.runner_name] = seen.get(cfg.runner_name, 0) + 1
    duplicates = {name: count for name, count in seen.items() if count > 1}
    assert not duplicates, f"duplicate runner_name in supported_runners: {duplicates}"


def test_runner_name_mirrors_pipeline_recipe_name() -> None:
    """``runner_name`` must equal ``pipeline.recipe_name`` by convention.

    The CLI's contract is "``flashdreams-run <recipe_name>`` runs that recipe";
    a divergence here would silently rename one slug and break that
    contract. Per-runner literals are free to opt out, but the in-tree
    set must hold the line.
    """
    drifted = {
        key: (cfg.runner_name, cfg.pipeline.recipe_name)
        for key, cfg in supported_runners().items()
        if cfg.runner_name != cfg.pipeline.recipe_name
    }
    assert not drifted, f"runner_name != pipeline.recipe_name (CLI contract): {drifted}"


def test_supported_runners_have_descriptions() -> None:
    """Every shipped runner must carry a non-empty ``cfg.description``.

    The CLI surfaces ``cfg.description`` next to every subcommand, so a
    missing entry shows up as an empty help line.
    """
    empty = [k for k, cfg in supported_runners().items() if not cfg.description.strip()]
    assert not empty, (
        f"supported_runners entries missing a non-empty description: {empty}"
    )
