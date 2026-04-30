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

"""Pre-built :class:`WanInferencePipelineConfig` builders for streaming Wan 2.1.

Each builder maps a short name (``"self_forcing"``,
``"causal_forcing_chunkwise"``, ...) to a function that takes only the
runtime knobs the recipe layer must own (``cp_size``,
``compile_network``, ``seed``, ``i2v``) and returns a fully-constructed
:class:`WanInferencePipelineConfig` (the same pipeline class used by the
non-streaming :mod:`flashdreams.recipes.wan.config.wan21`).

Batch / video resolution / per-chunk temporal length are intentionally
*not* exposed at the recipe layer: they live on
:class:`Wan21TransformerConfig` and are hardcoded to canonical Wan 2.1
streaming defaults inside this module. Callers that want to deviate
should construct :class:`Wan21TransformerConfig` directly.
"""

from __future__ import annotations

from collections.abc import Callable

from torch import Tensor

from flashdreams.infra.diffusion.model import DiffusionModelConfig
from flashdreams.infra.diffusion.scheduler.fm import FlowMatchSchedulerConfig
from flashdreams.infra.encoder import EncoderConfig
from flashdreams.recipes.taehv import TeahvVAEDecoderConfig
from flashdreams.recipes.wan.autoencoder.i2v import I2VCtrlEncoderConfig
from flashdreams.recipes.wan.autoencoder.vae import (
    AVAILABLE_WAN_VAE_CHECKPOINT_PATHS,
    WanVAEDecoderConfig,
    WanVAEEncoderConfig,
)
from flashdreams.recipes.wan.pipeline import WanInferencePipelineConfig
from flashdreams.recipes.wan.transformer.impl.network import WanDiTNetwork1pt3BConfig
from flashdreams.recipes.wan.transformer.wan21 import Wan21TransformerConfig

AVAILABLE_CAUSAL_WAN21_CHECKPOINT_PATHS: dict[str, str | dict[str, str]] = {
    "self_forcing": "https://huggingface.co/gdhe17/Self-Forcing/blob/main/checkpoints/self_forcing_dmd.pt",
    "causal_forcing": {
        "chunkwise": "https://huggingface.co/zhuhz22/Causal-Forcing/blob/main/chunkwise/causal_forcing.pt",
        "framewise": "https://huggingface.co/zhuhz22/Causal-Forcing/blob/main/framewise/causal_forcing.pt",
    },
}


# ---------------------------------------------------------------------------
# Checkpoint remap
# ---------------------------------------------------------------------------


def _remap_self_or_causal_forcing_state_dict(
    state_dict: dict[str, Tensor],
) -> dict[str, Tensor]:
    """Remap the Self-Forcing / Causal-Forcing checkpoint layout to the bare
    ``WanDiTNetwork`` layout expected by :class:`Wan21Transformer`.

    These checkpoints wrap the network in extra modules, namely:

    - A top-level ``"generator_ema"`` (Self-Forcing) or ``"generator"``
      (Causal-Forcing) container.
    - A ``"model."`` or ``"net."`` outer prefix.
    - An ``"_fsdp_wrapped_module."`` inner prefix on the framewise variant.

    This function strips them (in that order) so the resulting state-dict
    matches the keys ``WanDiTNetwork.state_dict()`` exposes.
    """
    if "generator_ema" in state_dict:
        state_dict = state_dict["generator_ema"]  # type: ignore[assignment]  # ty:ignore[invalid-assignment]
    elif "generator" in state_dict:
        state_dict = state_dict["generator"]  # type: ignore[assignment]  # ty:ignore[invalid-assignment]

    out: dict[str, Tensor] = {}
    for k, v in state_dict.items():
        if k.startswith("model."):
            new_k = k[len("model.") :]
        elif k.startswith("net."):
            new_k = k[len("net.") :]
        else:
            new_k = k
        if new_k.startswith("_fsdp_wrapped_module."):
            new_k = new_k[len("_fsdp_wrapped_module.") :]
        out[new_k] = v
    return out


# ---------------------------------------------------------------------------
# Canonical Wan 2.1 streaming defaults
# ---------------------------------------------------------------------------

# Self-Forcing-style 4-step distillation schedule.
_DEFAULT_DENOISING_TIMESTEPS = [1000, 750, 500, 250]
_DEFAULT_NUM_TRAIN_TIMESTEPS = 1000

_DEFAULT_BATCH_SHAPE: tuple[int, ...] = (1,)
_DEFAULT_VIDEO_HEIGHT = 480
_DEFAULT_VIDEO_WIDTH = 832
_DEFAULT_LEN_T_LATENT = 3  # framewise variant overrides to 1.
_WAN_VAE_SPATIAL_COMPRESSION = 8


def _wan_vae_decoder_config() -> WanVAEDecoderConfig:
    """Wan VAE decoder config."""
    return WanVAEDecoderConfig()


def _taehv_vae_decoder_config() -> TeahvVAEDecoderConfig:
    """LightTAE (TAEHV) decoder config."""
    return TeahvVAEDecoderConfig()


def _scheduler_config(num_inference_steps: int = 4) -> FlowMatchSchedulerConfig:
    """Self-Forcing flow-match scheduler defaults."""
    timesteps = _DEFAULT_DENOISING_TIMESTEPS[:num_inference_steps]
    return FlowMatchSchedulerConfig(
        num_inference_steps=num_inference_steps,
        denoising_timesteps=timesteps,
        warp_denoising_step=True,
        shift=5.0,
        sigma_min=0.0,
        extra_one_step=True,
        num_train_timesteps=_DEFAULT_NUM_TRAIN_TIMESTEPS,
    )


def _transformer_config(
    *,
    checkpoint_path: str,
    cp_size: int,
    compile_network: bool,
    len_t_latent: int = _DEFAULT_LEN_T_LATENT,
    stamp_image_latent: bool = False,
) -> Wan21TransformerConfig:
    """Wan 1.3B transformer defaults for causal/streaming inference.

    Shape knobs (batch / video height / video width) are hardcoded to the
    canonical Wan 2.1 streaming defaults; only the runtime knobs the
    caller actually owns (CP size, torch.compile toggle) are exposed.

    I2V is handled purely by mask injection inside
    :meth:`Wan21Transformer.predict_flow` and
    :meth:`Wan21Transformer.postprocess_clean_latent`: the same patch
    embedding works for both T2V and I2V, so the network's ``in_dim`` and
    ``concat_padding_mask`` stay at their checkpoint-baked defaults.
    """
    return Wan21TransformerConfig(
        network=WanDiTNetwork1pt3BConfig(
            patch_embedding_type="conv3d",
        ),
        checkpoint_path=checkpoint_path,
        state_dict_transform=_remap_self_or_causal_forcing_state_dict,
        batch_shape=_DEFAULT_BATCH_SHAPE,
        height=_DEFAULT_VIDEO_HEIGHT // _WAN_VAE_SPATIAL_COMPRESSION,
        width=_DEFAULT_VIDEO_WIDTH // _WAN_VAE_SPATIAL_COMPRESSION,
        len_t=len_t_latent,
        cp_size=cp_size,
        # CFG off by default to match the legacy causal_wan2_1.
        guidance_scale=1.0,
        # Streaming defaults.
        window_size_t=21,
        sink_size_t=0,
        stamp_image_latent=stamp_image_latent,
        compile_network=compile_network,
    )


def _pipeline_encoder_config(*, i2v: bool) -> EncoderConfig | None:
    """Per-AR-step infra encoder slot.

    For I2V we wire an :class:`I2VCtrlEncoder` whose forward takes the
    per-AR-step pixel chunk built by
    :meth:`WanInferencePipeline._preprocess_i2v_input` and returns an
    :class:`I2VCtrl` (encoded latent + binary injection mask). The
    encoder's underlying Wan VAE is pinned to the same checkpoint as the
    decoder so the encoded latent matches the network's input
    distribution exactly. Pure T2V leaves the encoder slot ``None``.
    """
    if not i2v:
        return None
    return I2VCtrlEncoderConfig(
        encoder=WanVAEEncoderConfig(
            checkpoint_path=AVAILABLE_WAN_VAE_CHECKPOINT_PATHS["vae"],
        ),
    )


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def build_self_forcing(
    *,
    cp_size: int = 1,
    compile_network: bool = True,
    seed: int = 42,
    i2v: bool = False,
    enable_sync_and_profile: bool = False,
) -> WanInferencePipelineConfig:
    """Self-Forcing distilled checkpoint with the Wan VAE decoder.

    Set ``i2v=True`` to also attach a first-frame Wan VAE encoder for
    mask-injection image-to-video; callers must then pass ``image=...`` to
    :meth:`WanInferencePipeline.initialize_cache`.
    """
    return WanInferencePipelineConfig(
        enable_sync_and_profile=enable_sync_and_profile,
        encoder=_pipeline_encoder_config(i2v=i2v),
        decoder=_wan_vae_decoder_config(),
        diffusion_model=DiffusionModelConfig(
            seed=seed,
            transformer=_transformer_config(
                checkpoint_path=AVAILABLE_CAUSAL_WAN21_CHECKPOINT_PATHS["self_forcing"],  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]
                cp_size=cp_size,
                compile_network=compile_network,
            ),
            scheduler=_scheduler_config(num_inference_steps=4),
        ),
    )


def build_self_forcing_lighttae(
    *,
    cp_size: int = 1,
    compile_network: bool = True,
    seed: int = 42,
    i2v: bool = False,
    enable_sync_and_profile: bool = False,
) -> WanInferencePipelineConfig:
    """Self-Forcing distilled checkpoint with the LightTAE (TAEHV) decoder.

    See :func:`build_self_forcing` for the ``i2v`` flag semantics.
    """
    return WanInferencePipelineConfig(
        enable_sync_and_profile=enable_sync_and_profile,
        encoder=_pipeline_encoder_config(i2v=i2v),
        decoder=_taehv_vae_decoder_config(),
        diffusion_model=DiffusionModelConfig(
            seed=seed,
            transformer=_transformer_config(
                checkpoint_path=AVAILABLE_CAUSAL_WAN21_CHECKPOINT_PATHS["self_forcing"],  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]
                cp_size=cp_size,
                compile_network=compile_network,
            ),
            scheduler=_scheduler_config(num_inference_steps=4),
        ),
    )


def build_causal_forcing_chunkwise(
    *,
    cp_size: int = 1,
    compile_network: bool = True,
    seed: int = 42,
    i2v: bool = False,
    enable_sync_and_profile: bool = False,
) -> WanInferencePipelineConfig:
    """Causal-Forcing chunkwise checkpoint with the Wan VAE decoder.

    See :func:`build_self_forcing` for the ``i2v`` flag semantics.
    """
    return WanInferencePipelineConfig(
        enable_sync_and_profile=enable_sync_and_profile,
        encoder=_pipeline_encoder_config(i2v=i2v),
        decoder=_wan_vae_decoder_config(),
        diffusion_model=DiffusionModelConfig(
            seed=seed,
            transformer=_transformer_config(
                checkpoint_path=AVAILABLE_CAUSAL_WAN21_CHECKPOINT_PATHS[
                    "causal_forcing"
                ]["chunkwise"],  # type: ignore[index]  # ty:ignore[invalid-argument-type]
                cp_size=cp_size,
                compile_network=compile_network,
            ),
            scheduler=_scheduler_config(num_inference_steps=4),
        ),
    )


def build_causal_forcing_framewise(
    *,
    cp_size: int = 1,
    compile_network: bool = True,
    seed: int = 42,
    i2v: bool = False,
    enable_sync_and_profile: bool = False,
) -> WanInferencePipelineConfig:
    """Causal-Forcing framewise checkpoint with the Wan VAE decoder.

    See :func:`build_self_forcing` for the ``i2v`` flag semantics.
    """
    return WanInferencePipelineConfig(
        enable_sync_and_profile=enable_sync_and_profile,
        encoder=_pipeline_encoder_config(i2v=i2v),
        decoder=_wan_vae_decoder_config(),
        diffusion_model=DiffusionModelConfig(
            seed=seed,
            transformer=_transformer_config(
                checkpoint_path=AVAILABLE_CAUSAL_WAN21_CHECKPOINT_PATHS[
                    "causal_forcing"
                ]["framewise"],  # type: ignore[index]  # ty:ignore[invalid-argument-type]
                cp_size=cp_size,
                compile_network=compile_network,
                # framewise: one latent frame per chunk.
                len_t_latent=1,
                # I2V mode will replace the first latent frame with the image latent.
                stamp_image_latent=i2v,
            ),
            scheduler=_scheduler_config(num_inference_steps=4),
        ),
    )


CAUSAL_WAN21_CONFIG_BUILDERS: dict[str, Callable[..., WanInferencePipelineConfig]] = {
    "self_forcing": build_self_forcing,
    "self_forcing_lighttae": build_self_forcing_lighttae,
    "causal_forcing_chunkwise": build_causal_forcing_chunkwise,
    "causal_forcing_framewise": build_causal_forcing_framewise,
}
