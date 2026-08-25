# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU-safe configuration and application binding checks for OmniDreams."""

from pathlib import Path

import pytest
import tomli as tomllib

from omnidreams.config import (
    OMNIDREAMS_APPLICATION_DEFAULTS,
    OMNIDREAMS_CONFIGS,
)

pytestmark = pytest.mark.ci_cpu


def test_pipeline_configs_are_keyed_by_name() -> None:
    """Keep every shipped model preset addressable by its canonical name."""
    assert OMNIDREAMS_CONFIGS
    assert OMNIDREAMS_CONFIGS == {
        config.name: config for config in OMNIDREAMS_CONFIGS.values()
    }


def test_hdmap2v_defaults_are_model_owned() -> None:
    """Keep the reusable HDMap2V app free of model preset selection logic."""
    assert OMNIDREAMS_APPLICATION_DEFAULTS.backend == "world_model"
    assert OMNIDREAMS_APPLICATION_DEFAULTS.slug == "hdmap2v"


def test_manifest_registers_model_owned_app_adapters() -> None:
    """Expose applications through nested adapters and no runner entry points."""
    path = Path(__file__).resolve().parents[1] / "pyproject.toml"
    manifest = tomllib.loads(path.read_text())
    entry_points = manifest["project"]["entry-points"]

    assert "flashdreams.runner_configs" not in entry_points
    assert entry_points["flashdreams.applications_v2"] == {
        "interactive-drive": "omnidreams.apps.interactive_drive.adapter:create_app",
        "hdmap2v": "omnidreams.apps.hdmap2v.adapter:create_app",
    }
