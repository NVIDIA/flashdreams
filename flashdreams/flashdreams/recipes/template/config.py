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

"""User-facing configs for the template recipe.

Hosts both the pre-built :class:`StreamInferencePipelineConfig`
literals (one per shipped variant) and the per-slug
:class:`TemplateRunnerConfig` literals that drive ``flashdreams-run``.
The runner-config literals self-register with
:mod:`flashdreams.configs.registry` at import time.
"""

from __future__ import annotations

from typing import cast

import torch

from flashdreams.configs.registry import register_runner
from flashdreams.infra.config import derive_config
from flashdreams.infra.diffusion.model import DiffusionModelConfig
from flashdreams.infra.diffusion.scheduler.fm import FlowMatchSchedulerConfig
from flashdreams.infra.pipeline import StreamInferencePipelineConfig
from flashdreams.recipes.template.decoder import TemplateDecoderConfig
from flashdreams.recipes.template.encoder import TemplateControlEncoderConfig
from flashdreams.recipes.template.runner import TemplateRunnerConfig
from flashdreams.recipes.template.transformer import TemplateTransformerConfig
from flashdreams.recipes.template.transformer.network import TemplateDiTConfig

TEMPLATE_OFFLINE = StreamInferencePipelineConfig(
    recipe_name="template-offline",
    encoder=TemplateControlEncoderConfig(
        control_channels=8,
        out_channels=4,
        dtype=torch.bfloat16,
    ),
    decoder=TemplateDecoderConfig(
        in_channels=4,
        out_channels=3,
        dtype=torch.bfloat16,
    ),
    diffusion_model=DiffusionModelConfig(
        seed=42,
        context_noise=0,
        transformer=TemplateTransformerConfig(
            network=TemplateDiTConfig(
                in_channels=4 * (2 * 2 * 2),
                context_channels=16,
                model_channels=128,
                num_heads=2,
            ),
            patch_size=(2, 2, 2),
            len_t=8,
            window_size_t=8,
            sink_size_t=0,
            guidance_scale=1.0,
            dtype=torch.bfloat16,
        ),
        scheduler=FlowMatchSchedulerConfig(
            num_inference_steps=2,
            denoising_timesteps=[1000, 500],
            warp_denoising_step=True,
            shift=5.0,
            num_train_timesteps=1000,
        ),
    ),
)
"""Offline (bidirectional, one-shot) reference rollout.

Single AR step over the full temporal window
(``window_size_t == len_t == 8``), CFG off, per-step control encoded
into the latent channel count, clean latent decoded to 3 channels.
``head_dim = 128 // 2 = 64`` so cuDNN flash-attention picks a stable
kernel; smaller head_dims (16/8) silently NaN. The network's
``in_channels`` is the post-patch width ``4 * (2 * 2 * 2) = 32``;
``patch_size = (2, 2, 2)`` must match
:attr:`TemplateTransformerConfig.patch_size`.
"""

TEMPLATE_AUTOREGRESSIVE = cast(
    StreamInferencePipelineConfig,
    derive_config(
        TEMPLATE_OFFLINE,
        recipe_name="template-autoregressive",
        diffusion_model=dict(
            transformer=dict(
                len_t=2,
                window_size_t=4,
            ),
            scheduler=dict(
                num_inference_steps=1,
                denoising_timesteps=[500],
            ),
        ),
    ),
)
"""Streaming AR variant: smaller per-chunk ``len_t`` (2) and a larger
``window_size_t`` (4 = 2 * len_t) so the KV cache fills over multiple
AR steps before rolling. CFG still off; patch ``guidance_scale > 1.0``
via :func:`derive_config` to enable it."""

TEMPLATE_AUTOREGRESSIVE_COMPILED = cast(
    StreamInferencePipelineConfig,
    derive_config(
        TEMPLATE_AUTOREGRESSIVE,
        recipe_name="template-autoregressive-compiled",
        diffusion_model=dict(
            transformer=dict(
                compile_network=True,
                use_cuda_graph=True,
            ),
        ),
    ),
)
"""Streaming AR with ``torch.compile`` + ``CUDAGraphWrapper`` enabled on
the DiT network. The fast deployment path: keep
``TEMPLATE_AUTOREGRESSIVE`` as the easy-to-debug default and reach for
this when measuring inference latency."""

TEMPLATE_CONFIGS: dict[str, StreamInferencePipelineConfig] = {
    cfg.recipe_name: cfg
    for cfg in (
        TEMPLATE_OFFLINE,
        TEMPLATE_AUTOREGRESSIVE,
        TEMPLATE_AUTOREGRESSIVE_COMPILED,
    )
}
"""All shipped template-recipe variants, keyed by ``recipe_name``."""


## Per-variant runner configs (slug == ``pipeline.recipe_name``).

TEMPLATE_OFFLINE_RUNNER = TemplateRunnerConfig(
    runner_name="template-offline",
    description=(
        "Reference template recipe: one-shot offline diffusion (synthetic inputs)."
    ),
    pipeline=TEMPLATE_OFFLINE,
    num_ar_steps=1,
)
"""Single-AR-step bidirectional rollout. Matches ``TEMPLATE_OFFLINE``'s
``window_size_t == len_t == 8`` (one chunk covers the full window)."""

TEMPLATE_AUTOREGRESSIVE_RUNNER = TemplateRunnerConfig(
    runner_name="template-autoregressive",
    description=(
        "Reference template recipe: streaming AR diffusion with sliding-window cache."
    ),
    pipeline=TEMPLATE_AUTOREGRESSIVE,
    # Two AR steps exercise both the KV-cache filling phase and the
    # first steady-state step (``window_size_t == 2 * len_t``).
    num_ar_steps=2,
)
"""Streaming AR rollout."""

TEMPLATE_AUTOREGRESSIVE_COMPILED_RUNNER = derive_config(
    TEMPLATE_AUTOREGRESSIVE_RUNNER,
    runner_name="template-autoregressive-compiled",
    description=(
        "Reference template recipe: AR variant with torch.compile + CUDA graphs."
    ),
    pipeline=TEMPLATE_AUTOREGRESSIVE_COMPILED,
)
"""Same I/O knobs as the AR base, pinned to the compiled + CUDA-graph pipeline."""


TEMPLATE_RUNNERS: dict[str, TemplateRunnerConfig] = {
    cfg.runner_name: cfg
    for cfg in (
        TEMPLATE_OFFLINE_RUNNER,
        TEMPLATE_AUTOREGRESSIVE_RUNNER,
        TEMPLATE_AUTOREGRESSIVE_COMPILED_RUNNER,
    )
}
"""All shipped template-recipe runners, keyed by ``runner_name``."""

for _name, _cfg in TEMPLATE_RUNNERS.items():
    register_runner(_name, _cfg, source="builtin")
