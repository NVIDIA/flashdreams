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

"""Cheap import-time checks for the ``omnidreams`` plugin.

The full numerics / GPU tests live alongside this file (they need GPU
+ checkpoints). These smoke tests just confirm the plugin is wired
correctly: importable, every ``runner_name`` mirrors its
``pipeline.name``, descriptions are non-empty, and the
entry-point declarations in ``pyproject.toml`` match the
``omnidreams.config`` ``RUNNER_*`` literals exactly.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import cast

import pytest
import tomli as tomllib
from omnidreams import config as config_mod
from omnidreams.config import OMNIDREAMS_RUNNERS

from flashdreams.infra.runner import RunnerConfig

pytestmark = pytest.mark.ci_cpu

ENTRY_POINT_GROUP = "flashdreams.runner_configs"


def test_runners_dict_is_non_empty() -> None:
    """Plugin must expose at least one runner."""
    assert OMNIDREAMS_RUNNERS, "OMNIDREAMS_RUNNERS is empty"


def test_runner_name_mirrors_pipeline_name() -> None:
    """``runner_name`` must equal ``pipeline.name`` per the CLI contract."""
    drifted = {
        slug: (cfg.runner_name, cfg.pipeline.name)
        for slug, cfg in OMNIDREAMS_RUNNERS.items()
        if cfg.runner_name != cfg.pipeline.name
    }
    assert not drifted, f"runner_name != pipeline.name: {drifted}"


def test_runners_have_descriptions() -> None:
    """Every shipped runner needs a non-empty CLI description."""
    empty = [
        slug for slug, cfg in OMNIDREAMS_RUNNERS.items() if not cfg.description.strip()
    ]
    assert not empty, f"runners missing description: {empty}"


def test_entry_points_match_module_literals() -> None:
    """The entry points in ``pyproject.toml`` must resolve to module attrs.

    Catches the common drift where someone adds a runner literal but
    forgets to wire it into the entry-point group (or vice versa);
    discovery would silently miss the new slug at the user's terminal.
    """
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    with pyproject.open("rb") as fh:
        meta = tomllib.load(fh)
    entries = meta["project"]["entry-points"][ENTRY_POINT_GROUP]
    declared_slugs = set(entries)
    module_slugs = set(OMNIDREAMS_RUNNERS)
    assert declared_slugs == module_slugs, (
        f"entry-point slugs ({sorted(declared_slugs)}) "
        f"!= module runners ({sorted(module_slugs)})"
    )

    for slug, target in entries.items():
        module_name, attr = target.split(":", 1)
        # Resolve the entry-point target the same way importlib.metadata
        # would, but skip the actual ``entry_points()`` call so the test
        # passes even when the plugin isn't pip-installed yet.
        assert module_name == "omnidreams.config", (
            f"unexpected module in entry point {slug!r}: {module_name}"
        )
        cfg = cast(RunnerConfig, getattr(config_mod, attr))
        assert cfg.runner_name == slug, (
            f"entry point {slug!r} -> {attr} resolves to "
            f"runner_name={cfg.runner_name!r}"
        )


@pytest.mark.skipif(
    sys.version_info < (3, 11),
    reason="entry-point discovery test relies on ``importlib.metadata`` 3.10+ shape",
)
def test_entry_points_discoverable_when_installed() -> None:
    """``importlib.metadata.entry_points`` finds the plugin's slugs.

    Requires the package to be installed (``uv sync`` from the repo
    root suffices since the plugin is a workspace member). Skipped
    automatically when running from a clean checkout. This is the
    integration check that mirrors what ``flashdreams-run``'s
    discovery layer actually does.
    """
    from importlib.metadata import entry_points

    eps = entry_points(group=ENTRY_POINT_GROUP)
    discovered = {ep.name for ep in eps if ep.value.startswith("omnidreams.")}
    if not discovered:
        pytest.skip("plugin not installed; run `uv sync` from the repo root first")
    assert discovered == set(OMNIDREAMS_RUNNERS), (
        f"discovered slugs ({sorted(discovered)}) != "
        f"plugin runners ({sorted(OMNIDREAMS_RUNNERS)})"
    )
