# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cheap import-time checks for the ltx_video plugin."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import cast

import pytest
import tomli as tomllib
from ltx_video import config as config_mod
from ltx_video.config import RUNNER_CONFIGS

from flashdreams.infra.runner import RunnerConfig

pytestmark = pytest.mark.ci_cpu

ENTRY_POINT_GROUP = "flashdreams.runner_configs"


def test_runners_dict_is_non_empty() -> None:
    assert RUNNER_CONFIGS, "RUNNER_CONFIGS is empty"


def test_three_runners_registered() -> None:
    assert set(RUNNER_CONFIGS) == {
        "ltx-video-t2v-2b",
        "ltx-video-t2v-2b-optimized",
        "ltx-video-t2v-2b-taehv",
    }


def test_runner_name_mirrors_pipeline_name() -> None:
    drifted = {
        slug: (cfg.runner_name, cfg.pipeline.name)
        for slug, cfg in RUNNER_CONFIGS.items()
        if cfg.runner_name != cfg.pipeline.name
    }
    assert not drifted, f"runner_name != pipeline.name: {drifted}"


def test_entry_points_match_module_literals() -> None:
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    with pyproject.open("rb") as fh:
        meta = tomllib.load(fh)
    entries = meta["project"]["entry-points"][ENTRY_POINT_GROUP]
    declared_slugs = set(entries)
    module_slugs = set(RUNNER_CONFIGS)
    assert declared_slugs == module_slugs

    for slug, target in entries.items():
        module_name, attr = target.split(":", 1)
        assert module_name == "ltx_video.config"
        cfg = cast(RunnerConfig, getattr(config_mod, attr))
        assert cfg.runner_name == slug


@pytest.mark.skipif(
    sys.version_info < (3, 11),
    reason="entry-point discovery test relies on importlib.metadata 3.10+ shape",
)
def test_entry_points_discoverable_when_installed() -> None:
    from importlib.metadata import entry_points

    eps = entry_points(group=ENTRY_POINT_GROUP)
    discovered = {ep.name for ep in eps if ep.value.startswith("ltx_video.")}
    if not discovered:
        pytest.skip("plugin not installed; run `uv sync --project integrations/ltx_video` first")
    assert discovered == set(RUNNER_CONFIGS)
