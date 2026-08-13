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

"""CPU tests for YAML preset and pipeline-provider resolution."""

from __future__ import annotations

from pathlib import Path

import pytest
from t2v_app import application
from t2v_app.presets import load_preset_catalog

from flashdreams.core.checkpoint.remap import unwrap_generator_state_dict
from flashdreams.infra.diffusion.scheduler.fm import FlowMatchSchedulerConfig
from flashdreams.recipes.wan import Wan21TransformerConfig

pytestmark = pytest.mark.ci_cpu


def test_packaged_yaml_constructs_default_pipeline_config() -> None:
    catalog = load_preset_catalog()
    preset_id, preset = catalog.resolve(None)

    config = application._create_pipeline_config(preset_id, preset)

    assert preset_id == "causal-forcing-wan2.1-t2v-1.3b-chunkwise"
    assert config.name == preset_id
    transformer = config.diffusion_model.transformer
    scheduler = config.diffusion_model.scheduler
    assert isinstance(transformer, Wan21TransformerConfig)
    assert transformer.len_t == 3
    assert transformer.batch_shape == ()
    assert transformer.state_dict_transform is unwrap_generator_state_dict
    assert isinstance(scheduler, FlowMatchSchedulerConfig)
    assert scheduler.denoising_timesteps == [1000, 750, 500, 250]
    assert preset.runtime.pixel_width == 832


def test_catalog_rejects_incomplete_runtime_options(tmp_path: Path) -> None:
    catalog_path = tmp_path / "presets.yaml"
    catalog_path.write_text(
        """
schema_version: 1
default_preset_id: test
presets:
  test:
    provider: flashdreams.core.pipeline_presets:ObjectGraphPipelineProvider
    runtime:
      prompt: test
      total_blocks: 1
      pixel_height: 64
      pixel_width: 64
      fps: 16
    pipeline: {}
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="output_layout"):
        load_preset_catalog(catalog_path)


def test_catalog_allows_total_blocks_to_be_omitted(tmp_path: Path) -> None:
    catalog_path = tmp_path / "presets.yaml"
    catalog_path.write_text(
        """
schema_version: 1
default_preset_id: test
presets:
  test:
    provider: flashdreams.core.pipeline_presets:ObjectGraphPipelineProvider
    runtime:
      prompt: test
      pixel_height: 64
      pixel_width: 64
      fps: 16
      output_layout: tchw
    pipeline: {}
""".strip(),
        encoding="utf-8",
    )

    _, preset = load_preset_catalog(catalog_path).resolve(None)

    assert preset.runtime.total_blocks is None


def test_catalog_reports_yaml_presets_for_unknown_id() -> None:
    catalog = load_preset_catalog()

    with pytest.raises(ValueError, match="YAML presets"):
        catalog.resolve("not-a-preset")
