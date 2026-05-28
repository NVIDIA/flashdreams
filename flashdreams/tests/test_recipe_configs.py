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
duplicate keys, dict key vs ``runner_name`` drift, integrations that
forgot to add their runners to the aggregator, runner_name vs
pipeline.name drift (which would surface as confusing
``flashdreams-run <slug>`` failures), and missing CLI descriptions.
"""

from __future__ import annotations

import pytest

# Importing ``runner_configs`` triggers each in-tree integration's
# self-registration side effects. Without this import the registry
# would be empty when tests run in isolation.
import flashdreams.configs.runner_configs  # noqa: F401
from flashdreams.configs.registry import supported_runners
from flashdreams.recipes.template.config import TEMPLATE_RUNNERS

pytestmark = pytest.mark.ci_cpu


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
    """Each per-integration ``<NAME>_RUNNERS`` dict must be merged in full.

    Catches the case where a new in-tree integration added a ``<NAME>_RUNNERS``
    dict but forgot to wire its ``runner.py`` into the aggregator.
    Out-of-tree plugin integrations are covered by their own smoke tests.
    """
    runners = supported_runners()
    expected = {
        **TEMPLATE_RUNNERS,
    }
    missing = set(expected) - set(runners)
    assert not missing, f"supported_runners missing slugs: {sorted(missing)}"
    extra = set(runners) - set(expected)
    assert not extra, (
        f"supported_runners has slugs outside the per-integration dicts: {sorted(extra)}"
    )


def test_supported_runners_unique_runner_names() -> None:
    """No two registered runners share a ``runner_name``."""
    seen: dict[str, int] = {}
    for cfg in supported_runners().values():
        seen[cfg.runner_name] = seen.get(cfg.runner_name, 0) + 1
    duplicates = {name: count for name, count in seen.items() if count > 1}
    assert not duplicates, f"duplicate runner_name in supported_runners: {duplicates}"


def test_runner_name_mirrors_pipeline_name() -> None:
    """``runner_name`` must equal ``pipeline.name`` by convention.

    The CLI's contract is "``flashdreams-run <name>`` runs that integration";
    a divergence here would silently rename one slug and break that
    contract. Per-runner literals are free to opt out, but the in-tree
    set must hold the line.
    """
    drifted = {
        key: (cfg.runner_name, cfg.pipeline.name)
        for key, cfg in supported_runners().items()
        if cfg.runner_name != cfg.pipeline.name
    }
    assert not drifted, f"runner_name != pipeline.name (CLI contract): {drifted}"


def test_supported_runners_have_descriptions() -> None:
    """Every shipped runner must carry a non-empty ``cfg.description``.

    The CLI surfaces ``cfg.description`` next to every subcommand, so a
    missing entry shows up as an empty help line.
    """
    empty = [k for k, cfg in supported_runners().items() if not cfg.description.strip()]
    assert not empty, (
        f"supported_runners entries missing a non-empty description: {empty}"
    )
