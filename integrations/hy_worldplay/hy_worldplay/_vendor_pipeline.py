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

"""Inert :class:`StreamInferencePipeline` stand-in for the vendor-wrapped runner."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch.nn as nn

from flashdreams.infra.diffusion.model import DiffusionModelConfig
from flashdreams.infra.diffusion.scheduler import SchedulerConfig
from flashdreams.infra.diffusion.transformer import TransformerConfig
from flashdreams.infra.pipeline import (
    StreamInferencePipeline,
    StreamInferencePipelineConfig,
)

__all__ = [
    "VENDOR_WRAPPER_RECIPE_NAME",
    "_NoopPipeline",
    "_NoopPipelineConfig",
]


## Sentinel recipe slug

VENDOR_WRAPPER_RECIPE_NAME = "hy-worldplay-vendor-wrapper-noop"
"""Sentinel ``recipe_name`` chosen never to collide with a real ``flashdreams.recipes.wan`` slug."""


## No-op configs

@dataclass(kw_only=True)
class _NoopDiffusionModelConfig(DiffusionModelConfig):
    """Stub :class:`DiffusionModelConfig` whose nested configs are never resolved."""

    transformer: TransformerConfig = field(default_factory=TransformerConfig)
    """Abstract-base placeholder; never ``setup()``-ed because :class:`_NoopPipeline` skips it."""

    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    """Abstract-base placeholder; never ``setup()``-ed because :class:`_NoopPipeline` skips it."""


@dataclass(kw_only=True)
class _NoopPipelineConfig(StreamInferencePipelineConfig):
    """Pipeline-shaped config that constructs an inert :class:`_NoopPipeline`."""

    _target: type = field(default_factory=lambda: _NoopPipeline)

    recipe_name: str = VENDOR_WRAPPER_RECIPE_NAME
    """Sentinel slug pinned to :data:`VENDOR_WRAPPER_RECIPE_NAME`."""

    diffusion_model: DiffusionModelConfig = field(
        default_factory=_NoopDiffusionModelConfig,
    )
    """Stub diffusion-model config; its ``transformer`` / ``scheduler`` are never resolved."""


## No-op pipeline

class _NoopPipeline(StreamInferencePipeline):
    """Inert :class:`nn.Module` standing in for a real :class:`StreamInferencePipeline`.

    The owning runner drives upstream's :class:`wan.generate.WanRunner`
    directly and never reads :attr:`encoder`, :attr:`decoder`, or
    :attr:`diffusion_model`; the base :class:`Runner` only needs
    ``pipeline.to(device).eval()`` to succeed.
    """

    def __init__(self, config: _NoopPipelineConfig) -> None:
        nn.Module.__init__(self)
        self.config = config
