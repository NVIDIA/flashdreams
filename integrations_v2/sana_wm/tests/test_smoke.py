# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU-safe configuration checks for the SANA-WM integration."""

from pathlib import Path

import pytest
import tomli as tomllib

from sana_wm.impl.conditioning import (
    SanaWMConditioningEncoderConfig,
    SanaWMStreamingConditioningEncoderConfig,
)
from sana_wm.config import (
    PIPELINE_CONFIGS,
    PIPELINE_SANA_WM_BIDIRECTIONAL,
    PIPELINE_SANA_WM_STREAMING,
)
from sana_wm.impl.decoder import (
    SanaWMStreamingVideoDecoderConfig,
    SanaWMVideoDecoderConfig,
)
from sana_wm.impl.transformer import (
    SanaWMStreamingTransformerConfig,
    SanaWMTransformerConfig,
)

pytestmark = pytest.mark.ci_cpu


def test_pipeline_configs_are_keyed_by_name() -> None:
    """Expose only canonical model pipeline configs."""
    assert PIPELINE_CONFIGS == {
        PIPELINE_SANA_WM_BIDIRECTIONAL.name: PIPELINE_SANA_WM_BIDIRECTIONAL,
        PIPELINE_SANA_WM_STREAMING.name: PIPELINE_SANA_WM_STREAMING,
    }


def test_bidirectional_config_uses_sana_components() -> None:
    """Keep the bidirectional preset within model implementation boundaries."""
    config = PIPELINE_SANA_WM_BIDIRECTIONAL
    assert isinstance(config.encoder, SanaWMConditioningEncoderConfig)
    assert isinstance(config.diffusion_model.transformer, SanaWMTransformerConfig)
    assert isinstance(config.decoder, SanaWMVideoDecoderConfig)


def test_streaming_config_uses_sana_components() -> None:
    """Keep the streaming preset within model implementation boundaries."""
    config = PIPELINE_SANA_WM_STREAMING
    assert isinstance(config.encoder, SanaWMStreamingConditioningEncoderConfig)
    assert isinstance(config.diffusion_model.transformer, SanaWMStreamingTransformerConfig)
    assert isinstance(config.decoder, SanaWMStreamingVideoDecoderConfig)


def test_manifest_has_no_runner_entry_points() -> None:
    """Do not expose model execution through the retired runner API."""
    path = Path(__file__).resolve().parents[1] / "pyproject.toml"
    manifest = tomllib.loads(path.read_text())
    assert "entry-points" not in manifest["project"]
