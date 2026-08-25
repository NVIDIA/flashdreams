# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU-safe config checks for the HY-WorldPlay integration."""

from pathlib import Path

import pytest
import tomli as tomllib

from hy_worldplay.impl._action import HyWorldPlayWan21TransformerConfig
from hy_worldplay.config import PIPELINE_HY_WORLDPLAY_WAN_I2V_5B

pytestmark = pytest.mark.ci_cpu


def test_pipeline_uses_hy_worldplay_transformer() -> None:
    """Keep the shipped config bound to the HY model implementation."""
    pipeline = PIPELINE_HY_WORLDPLAY_WAN_I2V_5B
    transformer = pipeline.diffusion_model.transformer

    assert pipeline.name == "hy-worldplay-wan-i2v-5b"
    assert isinstance(transformer, HyWorldPlayWan21TransformerConfig)
    assert transformer.len_t == 4
    assert transformer.window_size_t == 4
    assert transformer.use_cuda_graph is False


def test_manifest_registers_only_the_v2_application_adapter() -> None:
    """Expose Action2V through the model binding without a legacy runner."""
    path = Path(__file__).resolve().parents[1] / "pyproject.toml"
    manifest = tomllib.loads(path.read_text())
    entry_points = manifest["project"]["entry-points"]

    assert "flashdreams.runner_configs" not in entry_points
    assert entry_points["flashdreams.applications_v2"] == {
        "action2v-hy-worldplay": "hy_worldplay.apps.action2v.adapter:create_app"
    }
