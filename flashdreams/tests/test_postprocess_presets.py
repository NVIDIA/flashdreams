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

"""CPU tests for postprocess preset discovery and chain resolution."""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import pytest
import torch
from flashvsr.impl.postprocess import FlashVSRPostProcessorConfig
from typing_extensions import Self

from flashdreams.infra.postprocess import (
    RTXVideoSuperResolutionPostProcessorConfig,
    VideoPostprocessChainConfig,
    VideoPostProcessorConfig,
)
from flashdreams.plugins.registry import (
    discover_postprocess_presets,
    resolve_postprocess_preset,
)

pytestmark = pytest.mark.ci_cpu


@dataclass(kw_only=True)
class _ExamplePostProcessorConfig(VideoPostProcessorConfig):
    _target: type[object] = field(default_factory=lambda: object)


@dataclass(kw_only=True)
class _DevicePostProcessorConfig(_ExamplePostProcessorConfig):
    device: str = "cuda:0"

    def with_device(self, device: str | torch.device) -> Self:
        return replace(self, device=str(torch.device(device)))


def test_discover_postprocess_presets_includes_flashvsr_entries() -> None:
    presets = discover_postprocess_presets()

    assert "flashvsr-v1.1-sparse-2.0" in presets
    assert "flashvsr-v1.1-sparse-1.5" in presets
    assert "flashvsr-v1.1-full-attn" in presets
    assert "rtx-super-resolution" in presets
    assert "rtx-super-resolution-ultra" in presets
    assert "rtx-deblur-ultra" in presets


def test_resolve_postprocess_preset_rejects_unknown_name() -> None:
    with pytest.raises(ValueError, match="Unknown postprocess preset"):
        resolve_postprocess_preset("not-a-real-preset")


def test_chain_from_preset_binds_a_copy_to_the_selected_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registered = _DevicePostProcessorConfig()
    monkeypatch.setattr(
        "flashdreams.plugins.registry.resolve_postprocess_preset",
        lambda name: registered,
    )

    chain = VideoPostprocessChainConfig.from_preset(
        "example-preset",
        device="cuda:1",
    )

    (configured,) = chain.processors
    assert isinstance(configured, _DevicePostProcessorConfig)
    assert configured is not registered
    assert configured.device == "cuda:1"
    assert registered.device == "cuda:0"
    assert chain.preset == ""


def test_chain_from_preset_rejects_a_processor_without_device_placement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "flashdreams.plugins.registry.resolve_postprocess_preset",
        lambda name: _ExamplePostProcessorConfig(),
    )

    with pytest.raises(ValueError, match="does not support explicit device"):
        VideoPostprocessChainConfig.from_preset(
            "example-preset",
            device="cuda:1",
        )


def test_chain_config_appends_preset_after_explicit_processors() -> None:
    explicit = _ExamplePostProcessorConfig()
    chain = VideoPostprocessChainConfig(
        processors=(explicit,),
        preset="flashvsr-v1.1-sparse-2.0",
    )

    resolved = chain.resolved_processors()

    assert resolved[0] is explicit
    preset = resolved[1]
    assert isinstance(preset, FlashVSRPostProcessorConfig)
    assert preset.sparse_ratio == 2.0


def test_chain_config_resolves_rtx_super_resolution_preset() -> None:
    chain = VideoPostprocessChainConfig(preset="rtx-super-resolution")

    preset = chain.resolved_processors()[0]

    assert isinstance(preset, RTXVideoSuperResolutionPostProcessorConfig)
    assert preset.scale == 2.0


@pytest.mark.parametrize(
    ("preset_name", "expected_quality", "expected_scale"),
    [
        ("rtx-super-resolution-ultra", "ULTRA", 2.0),
        ("rtx-deblur-ultra", "DEBLUR_ULTRA", 1.0),
    ],
)
def test_chain_config_resolves_additional_rtx_presets(
    preset_name: str,
    expected_quality: str,
    expected_scale: float,
) -> None:
    chain = VideoPostprocessChainConfig(preset=preset_name)

    preset = chain.resolved_processors()[0]

    assert isinstance(preset, RTXVideoSuperResolutionPostProcessorConfig)
    assert preset.quality == expected_quality
    assert preset.scale == expected_scale


def test_chain_config_is_enabled_for_preset_only() -> None:
    chain = VideoPostprocessChainConfig(preset="flashvsr-v1.1-sparse-2.0")

    assert chain.is_enabled()


def test_chain_config_requires_all_ranks_for_full_attn_preset() -> None:
    chain = VideoPostprocessChainConfig(preset="flashvsr-v1.1-full-attn")

    assert chain.requires_all_ranks(world_size=2)


def test_chain_config_rejects_sparse_preset_under_multi_gpu() -> None:
    chain = VideoPostprocessChainConfig(preset="flashvsr-v1.1-sparse-2.0")

    with pytest.raises(ValueError, match="does not support multi-GPU"):
        chain.validate_execution(world_size=2)


def test_chain_config_caches_preset_resolution() -> None:
    chain = VideoPostprocessChainConfig(preset="flashvsr-v1.1-sparse-2.0")

    assert chain.resolved_processors()[0] is chain.resolved_processors()[0]
