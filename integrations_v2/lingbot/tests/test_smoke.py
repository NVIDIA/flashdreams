# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU-safe architecture checks for the LingBot integration."""

from pathlib import Path

import pytest
import tomli as tomllib

from lingbot.config import (
    DEFAULT_LINGBOT_PRESET,
    LINGBOT_APPLICATION_DEFAULTS,
    PIPELINE_CONFIGS,
)

pytestmark = pytest.mark.ci_cpu


def test_pipeline_configs_are_keyed_by_name() -> None:
    """Keep configuration lookup deterministic for the application adapter."""
    assert PIPELINE_CONFIGS
    assert PIPELINE_CONFIGS == {config.name: config for config in PIPELINE_CONFIGS.values()}


def test_application_defaults_select_a_shipped_pipeline() -> None:
    """The app adapter must select its model through the root config module."""
    assert DEFAULT_LINGBOT_PRESET in PIPELINE_CONFIGS
    assert LINGBOT_APPLICATION_DEFAULTS.preset_id == DEFAULT_LINGBOT_PRESET


def test_pipeline_configs_reference_model_checkpoints() -> None:
    """Keep every public preset bound to a concrete LingBot checkpoint."""
    for config in PIPELINE_CONFIGS.values():
        transformer = config.diffusion_model.transformer
        assert transformer.checkpoint_path


def test_manifest_registers_only_the_v2_application_adapter() -> None:
    """Expose the shared demo through its model-owned adapter, not a runner."""
    path = Path(__file__).resolve().parents[1] / "pyproject.toml"
    manifest = tomllib.loads(path.read_text())
    entry_points = manifest["project"]["entry-points"]

    assert "flashdreams.runner_configs" not in entry_points
    assert entry_points["flashdreams.applications_v2"] == {
        "action2v-lingbot": "lingbot.apps.action2v.adapter:create_app"
    }
