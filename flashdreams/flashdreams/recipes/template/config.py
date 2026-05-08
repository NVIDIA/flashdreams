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

"""Pipeline-config builders for the template recipe."""

from __future__ import annotations

from collections.abc import Callable
from math import prod
from typing import cast

import torch

from flashdreams.infra.config import derive_config
from flashdreams.infra.diffusion.model import DiffusionModelConfig
from flashdreams.infra.diffusion.scheduler.fm import FlowMatchSchedulerConfig
from flashdreams.infra.pipeline import StreamInferencePipelineConfig
from flashdreams.recipes.template.decoder import TemplateDecoderConfig
from flashdreams.recipes.template.encoder import TemplateControlEncoderConfig
from flashdreams.recipes.template.transformer import TemplateTransformerConfig
from flashdreams.recipes.template.transformer.network import TemplateDiTConfig

_DEFAULT_IN_CHANNELS = 4
"""Raw latent channel count for encoder / decoder. The network's
``in_channels`` is the post-patch width
``_DEFAULT_IN_CHANNELS * prod(_DEFAULT_PATCH_SIZE)``."""

_DEFAULT_CONTROL_CHANNELS = 8
_DEFAULT_OUT_CHANNELS = 3

_DEFAULT_PATCH_SIZE: tuple[int, int, int] = (2, 2, 2)
"""``(kt, kh, kw)`` 3D patch fold applied by the transformer before
the network sees the latent. Must agree with
:attr:`TemplateTransformerConfig.patch_size`."""

_DEFAULT_MODEL_CHANNELS = 128
"""``head_dim = model_channels // num_heads`` must be a size cuDNN's
flash-attention supports; 64 is safe, 16/8 silently NaN."""

_DEFAULT_NUM_HEADS = 2

_DEFAULT_DTYPE: torch.dtype = torch.bfloat16
"""Encoder / decoder dtype — kept in lock-step with
``TemplateTransformerConfig.dtype`` so ``input_proj`` doesn't see
mismatched control + latent dtypes."""

_DEFAULT_LEN_T_BIDIRECTIONAL = 8
_DEFAULT_LEN_T_STREAMING = 2
_DEFAULT_WINDOW_SIZE_T_STREAMING = 2 * _DEFAULT_LEN_T_STREAMING


def build_cfg_offline(
    *,
    seed: int = 42,
) -> StreamInferencePipelineConfig:
    """Build the offline (bidirectional, one-shot) template pipeline.

    Single AR step over the full temporal window
    (``window_size_t == len_t``), CFG off, per-step control encoded
    into the latent channel count, clean latent decoded to 3 channels.

    Args:
        seed: RNG seed for initial-noise draws.

    Returns:
        :class:`StreamInferencePipelineConfig` ready for ``.setup()``.
    """
    return StreamInferencePipelineConfig(
        encoder=TemplateControlEncoderConfig(
            control_channels=_DEFAULT_CONTROL_CHANNELS,
            out_channels=_DEFAULT_IN_CHANNELS,
            dtype=_DEFAULT_DTYPE,
        ),
        decoder=TemplateDecoderConfig(
            in_channels=_DEFAULT_IN_CHANNELS,
            out_channels=_DEFAULT_OUT_CHANNELS,
            dtype=_DEFAULT_DTYPE,
        ),
        diffusion_model=DiffusionModelConfig(
            seed=seed,
            context_noise=0,
            transformer=TemplateTransformerConfig(
                network=TemplateDiTConfig(
                    in_channels=_DEFAULT_IN_CHANNELS * prod(_DEFAULT_PATCH_SIZE),
                    context_channels=16,
                    model_channels=_DEFAULT_MODEL_CHANNELS,
                    num_heads=_DEFAULT_NUM_HEADS,
                ),
                patch_size=_DEFAULT_PATCH_SIZE,
                len_t=_DEFAULT_LEN_T_BIDIRECTIONAL,
                window_size_t=_DEFAULT_LEN_T_BIDIRECTIONAL,
                sink_size_t=0,
                guidance_scale=1.0,
                dtype=_DEFAULT_DTYPE,
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


def build_cfg_autoregressive(
    *,
    seed: int = 42,
) -> StreamInferencePipelineConfig:
    """Build the streaming AR template pipeline from :func:`build_cfg_offline`.

    Smaller per-chunk ``len_t`` and a larger ``window_size_t`` so the
    KV cache fills over multiple AR steps before rolling. CFG off;
    patch ``guidance_scale > 1.0`` on top via :func:`derive_config` to
    enable it.

    Args:
        seed: RNG seed for initial-noise draws.

    Returns:
        :class:`StreamInferencePipelineConfig` ready for ``.setup()``.
    """
    base = build_cfg_offline(seed=seed)
    # ``derive_config`` preserves the concrete subclass at runtime but
    # widens to :class:`InstantiateConfig` at the type level.
    return cast(
        StreamInferencePipelineConfig,
        derive_config(
            base,
            diffusion_model=dict(
                transformer=dict(
                    len_t=_DEFAULT_LEN_T_STREAMING,
                    window_size_t=_DEFAULT_WINDOW_SIZE_T_STREAMING,
                ),
                scheduler=dict(
                    num_inference_steps=1,
                    denoising_timesteps=[500],
                ),
            ),
        ),
    )


def with_compile_and_cuda_graph(
    base: StreamInferencePipelineConfig,
) -> StreamInferencePipelineConfig:
    """Return ``base`` with ``compile_network`` and ``use_cuda_graph`` flipped on."""
    return cast(
        StreamInferencePipelineConfig,
        derive_config(
            base,
            diffusion_model=dict(
                transformer=dict(
                    compile_network=True,
                    use_cuda_graph=True,
                ),
            ),
        ),
    )


TEMPLATE_CONFIG_BUILDERS: dict[str, Callable[..., StreamInferencePipelineConfig]] = {
    "offline": build_cfg_offline,
    "autoregressive": build_cfg_autoregressive,
}
