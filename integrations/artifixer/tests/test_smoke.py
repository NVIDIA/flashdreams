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

"""Cheap import-time checks for the ``artifixer`` plugin."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import cast

import pytest
import tomllib
from artifixer import config as config_mod
from artifixer.config import RUNNER_CONFIGS

from flashdreams.infra.runner import RunnerConfig

ENTRY_POINT_GROUP = "flashdreams.runner_configs"


def test_runners_dict_is_non_empty() -> None:
    assert RUNNER_CONFIGS, "RUNNER_CONFIGS is empty"


def test_runner_name_mirrors_pipeline_recipe_name() -> None:
    drifted = {
        slug: (cfg.runner_name, cfg.pipeline.recipe_name)
        for slug, cfg in RUNNER_CONFIGS.items()
        if cfg.runner_name != cfg.pipeline.recipe_name
    }
    assert not drifted, f"runner_name != pipeline.recipe_name: {drifted}"


def test_runners_have_descriptions() -> None:
    empty = [
        slug for slug, cfg in RUNNER_CONFIGS.items() if not cfg.description.strip()
    ]
    assert not empty, f"runners missing description: {empty}"


def test_entry_points_match_module_literals() -> None:
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    with pyproject.open("rb") as fh:
        meta = tomllib.load(fh)
    entries = meta["project"]["entry-points"][ENTRY_POINT_GROUP]
    declared_slugs = set(entries)
    module_slugs = set(RUNNER_CONFIGS)
    assert declared_slugs == module_slugs, (
        f"entry-point slugs ({sorted(declared_slugs)}) "
        f"!= module runners ({sorted(module_slugs)})"
    )

    for slug, target in entries.items():
        module_name, attr = target.split(":", 1)
        assert module_name == "artifixer.config", (
            f"unexpected module in entry point {slug!r}: {module_name}"
        )
        cfg = cast(RunnerConfig, getattr(config_mod, attr))
        assert cfg.runner_name == slug, (
            f"entry point {slug!r} -> {attr} resolves to "
            f"runner_name={cfg.runner_name!r}"
        )


@pytest.mark.skipif(
    sys.version_info < (3, 11),
    reason="entry-point discovery test relies on importlib.metadata 3.10+ shape",
)
def test_entry_points_discoverable_when_installed() -> None:
    from importlib.metadata import entry_points

    eps = entry_points(group=ENTRY_POINT_GROUP)
    discovered = {ep.name for ep in eps if ep.value.startswith("artifixer.")}
    if not discovered:
        pytest.skip("plugin not installed; run `uv sync` from the repo root first")
    assert discovered == set(RUNNER_CONFIGS), (
        f"discovered slugs ({sorted(discovered)}) != "
        f"plugin runners ({sorted(RUNNER_CONFIGS)})"
    )


def test_artifixer_hyperparams_match_dreamfix_stage3() -> None:
    """Sanity check: AR/scheduler knobs match the stage-3 DMD training config."""
    cfg = RUNNER_CONFIGS["artifixer-dmd-wan2.1-t2v-1.3b"]
    tcfg = cfg.pipeline.diffusion_model.transformer
    scfg = cfg.pipeline.diffusion_model.scheduler

    assert tcfg.len_t == 7, "ArtiFixer frames_per_block is 7 latent frames"
    assert tcfg.window_size_t == 21, "ArtiFixer local_attn_size is 21 latent frames"
    assert tcfg.sink_size_t == 7, "ArtiFixer sink_size is 7 latent frames"
    assert scfg.num_inference_steps == 4, "ArtiFixer DMD uses 4-step inference"
    assert scfg.shift == 5.0, "ArtiFixer FlowMatchScheduler uses shift=5"
    assert tcfg.guidance_scale == 1.0, "ArtiFixer KV-cache pipeline does not use CFG"


def test_artifixer_block_has_opacity_and_camera_mlps() -> None:
    """Phase 2.1: ArtifixerBlock instances carry opacity + camera MLPs."""
    import torch
    from artifixer.network import ArtifixerBlock, artifixer_embedding_dims

    opacity_dim, camera_dim = artifixer_embedding_dims((1, 2, 2))
    block = ArtifixerBlock(
        dim=1536,
        ffn_dim=8960,
        num_heads=12,
        opacity_embedding_dim=opacity_dim,
        camera_embedding_dim=camera_dim,
    )

    assert isinstance(block.opacity_embedding, torch.nn.Linear)
    assert isinstance(block.camera_embedding, torch.nn.Linear)
    assert block.opacity_embedding.weight.shape == (1536, opacity_dim)
    assert block.camera_embedding.weight.shape == (1536, camera_dim)

    # Zero-init contract: the wrapped block is a no-op extension of base
    # Wan behavior. Same invariant as dreamfix transformer.py L637-651.
    assert torch.all(block.opacity_embedding.weight == 0)
    assert torch.all(block.opacity_embedding.bias == 0)
    assert torch.all(block.camera_embedding.weight == 0)
    assert torch.all(block.camera_embedding.bias == 0)


def test_artifixer_recipe_uses_artifixer_network() -> None:
    """Phase 2.1: the shipped recipe wires up ArtifixerDiTNetwork."""
    from artifixer.network import ArtifixerDiTNetwork1pt3BConfig

    cfg = RUNNER_CONFIGS["artifixer-dmd-wan2.1-t2v-1.3b"]
    network_cfg = cfg.pipeline.diffusion_model.transformer.network
    assert isinstance(network_cfg, ArtifixerDiTNetwork1pt3BConfig), (
        f"recipe network is {type(network_cfg).__name__}, expected ArtifixerDiTNetwork1pt3BConfig"
    )


def test_zero_pad_state_dict_transform_fills_missing_keys() -> None:
    """Phase 2.1: the state_dict transform pads vanilla-Wan checkpoints."""
    import torch
    from artifixer.checkpoint import zero_pad_artifixer_keys
    from artifixer.network import artifixer_embedding_dims

    transform = zero_pad_artifixer_keys(
        num_layers=2, dim=1536, patch_size=(1, 2, 2), dtype=torch.bfloat16
    )
    padded = transform({"unrelated.weight": torch.zeros(8)})

    opacity_dim, camera_dim = artifixer_embedding_dims((1, 2, 2))
    for layer in range(2):
        prefix = f"blocks.{layer}."
        assert padded[prefix + "opacity_embedding.weight"].shape == (1536, opacity_dim)
        assert padded[prefix + "opacity_embedding.bias"].shape == (1536,)
        assert padded[prefix + "camera_embedding.weight"].shape == (1536, camera_dim)
        assert padded[prefix + "camera_embedding.bias"].shape == (1536,)
        assert padded[prefix + "opacity_embedding.weight"].dtype == torch.bfloat16
    # Untouched keys pass through.
    assert "unrelated.weight" in padded
