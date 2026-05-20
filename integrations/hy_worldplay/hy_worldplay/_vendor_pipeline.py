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

"""No-op :class:`StreamInferencePipeline` stand-in for the phase-1 vendor wrapper.

The phase-1 ``HyWorldPlayWanI2VRunner`` delegates encode / diffuse /
decode to upstream's ``wan.generate.WanRunner`` and has no flashdreams
pipeline to drive, but :attr:`RunnerConfig.pipeline` is non-optional.
:class:`_NoopPipelineConfig` and :class:`_NoopPipeline` satisfy that
contract: the config redefines the parent's required
``diffusion_model`` field with a stub default factory, and the
pipeline replaces the parent ``__init__`` with a bare
:class:`nn.Module` init so ``config.diffusion_model.setup()`` never
runs.
"""

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
"""Sentinel ``recipe_name`` for the no-op pipeline; chosen to never
collide with any real ``flashdreams.recipes.wan`` slug."""


## No-op configs

@dataclass(kw_only=True)
class _NoopDiffusionModelConfig(DiffusionModelConfig):
    """Stub :class:`DiffusionModelConfig` whose nested configs are never resolved."""

    transformer: TransformerConfig = field(default_factory=TransformerConfig)
    """Abstract-base placeholder. :class:`_NoopPipeline` skips ``setup()``
    on its parent diffusion-model config, so this is never instantiated
    (the abstract :class:`Transformer` base would refuse anyway)."""

    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    """Abstract-base placeholder. Same reasoning as ``transformer``."""


@dataclass(kw_only=True)
class _NoopPipelineConfig(StreamInferencePipelineConfig):
    """Pipeline-shaped config that constructs an inert :class:`_NoopPipeline`."""

    _target: type = field(default_factory=lambda: _NoopPipeline)

    recipe_name: str = VENDOR_WRAPPER_RECIPE_NAME
    """Sentinel slug pinned to :data:`VENDOR_WRAPPER_RECIPE_NAME`."""

    diffusion_model: DiffusionModelConfig = field(
        default_factory=_NoopDiffusionModelConfig,
    )
    """Stub diffusion-model config. ``seed=None`` (the inherited default)
    short-circuits the per-rank seed-offset branch in
    :meth:`Runner.__init__`; ``transformer`` / ``scheduler`` are never
    ``setup()``-ed because :class:`_NoopPipeline` overrides ``__init__``."""


## No-op pipeline

class _NoopPipeline(StreamInferencePipeline):
    """Inert pipeline -- an :class:`nn.Module` with no encoder, decoder, or diffusion model.

    The owning runner drives upstream's :class:`wan.generate.WanRunner`
    directly and never reads :attr:`encoder`, :attr:`decoder`, or
    :attr:`diffusion_model`; the base :class:`Runner` only needs
    ``pipeline.to(device).eval()`` to succeed, which
    :class:`nn.Module` provides for free.
    """

    def __init__(self, config: _NoopPipelineConfig) -> None:
        nn.Module.__init__(self)
        self.config = config
