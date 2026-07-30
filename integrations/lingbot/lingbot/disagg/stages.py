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

"""LingBot-specific encoder, DiT, and decoder service stages."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from flashdreams.infra.decoder import StreamingVideoDecoder
from flashdreams.infra.pipeline import DecoderStage, DiffusionStage, DiffusionStageCache
from flashdreams.infra.transfer import TensorBundle
from flashdreams.recipes.wan.autoencoder.i2v import I2VCtrl
from flashdreams.recipes.wan.autoencoder.vae import WanVAECache
from flashdreams.recipes.wan.transformer.constants import NEGATIVE_PROMPT
from lingbot.encoder.camctrl import (
    CamCtrlInput,
    I2VCamCtrlEmbeddings,
    I2VCamCtrlEncoder,
    I2VCamCtrlEncoderCache,
    I2VCamCtrlInput,
)
from lingbot.pipeline import LingbotWorldInferencePipelineConfig
from lingbot.transformer import (
    LingbotWorldTransformerCache,
    LingbotWorldTransformerConfig,
)


@dataclass(kw_only=True)
class LingbotConditioning:
    """One-shot encoder output transferred into the DiT worker."""

    height: int
    """Pre-patchify latent height."""

    width: int
    """Pre-patchify latent width."""

    text_embeddings: Tensor
    """Positive UMT5 prompt embeddings."""

    negative_text_embeddings: Tensor | None = None
    """Negative UMT5 embeddings when classifier-free guidance is enabled."""

    image_embeddings: Tensor | None = None
    """Optional CLIP first-frame embeddings."""


@dataclass(kw_only=True)
class LingbotEncoderStageCache:
    """Per-session state retained by the LingBot encoder worker."""

    encoder_cache: I2VCamCtrlEncoderCache
    """Streaming VAE and camera-control cache."""

    image: Tensor
    """First-frame pixels used to construct every I2V input chunk."""


def conditioning_to_bundle(conditioning: LingbotConditioning) -> TensorBundle:
    """Flatten one-shot conditioning into a transferable tensor bundle."""
    bundle = {"text_embeddings": conditioning.text_embeddings.contiguous()}
    if conditioning.negative_text_embeddings is not None:
        bundle["negative_text_embeddings"] = (
            conditioning.negative_text_embeddings.contiguous()
        )
    if conditioning.image_embeddings is not None:
        bundle["image_embeddings"] = conditioning.image_embeddings.contiguous()
    return bundle


def conditioning_from_bundle(
    bundle: TensorBundle,
    *,
    height: int,
    width: int,
) -> LingbotConditioning:
    """Reconstruct one-shot conditioning after a transfer."""
    return LingbotConditioning(
        height=height,
        width=width,
        text_embeddings=bundle["text_embeddings"],
        negative_text_embeddings=bundle.get("negative_text_embeddings"),
        image_embeddings=bundle.get("image_embeddings"),
    )


def encoder_output_to_bundle(output: I2VCamCtrlEmbeddings) -> TensorBundle:
    """Flatten per-step LingBot encoder output for transport."""
    assert not output._is_patchified, (
        "Encoder output must cross the service boundary before DiT patchification."
    )
    return {
        "i2v.latent": output.i2v.latent.contiguous(),
        "i2v.mask": output.i2v.mask.contiguous(),
        "plucker": output.plucker.contiguous(),
    }


def encoder_output_from_bundle(bundle: TensorBundle) -> I2VCamCtrlEmbeddings:
    """Reconstruct per-step LingBot encoder output after transport."""
    return I2VCamCtrlEmbeddings(
        i2v=I2VCtrl(
            latent=bundle["i2v.latent"],
            mask=bundle["i2v.mask"],
        ),
        plucker=bundle["plucker"],
    )


class LingbotEncoderStage(nn.Module):
    """Own LingBot's text, image, streaming VAE, and camera encoders."""

    def __init__(self, config: LingbotWorldInferencePipelineConfig) -> None:
        super().__init__()
        assert config.encoder is not None, "LingBot requires an I2V control encoder."
        encoder = config.encoder.setup()
        assert isinstance(encoder, I2VCamCtrlEncoder)
        self.encoder = encoder
        self.text_encoder = (
            config.text_encoder.setup() if config.text_encoder is not None else None
        )
        self.image_encoder = (
            config.image_encoder.setup() if config.image_encoder is not None else None
        )
        transformer_config = config.diffusion_model.transformer
        assert isinstance(transformer_config, LingbotWorldTransformerConfig)
        self.transformer_config = transformer_config

    @torch.no_grad()
    def initialize_cache(
        self,
        *,
        text: list[str],
        image: Tensor,
    ) -> tuple[LingbotEncoderStageCache, LingbotConditioning]:
        """Encode session-level context and initialize streaming encoder state."""
        assert self.text_encoder is not None, "LingBot text encoder is not configured."
        assert image.shape[-4] == 1, (
            f"image must contain exactly one frame, got shape {tuple(image.shape)}."
        )
        spatial_ratio = self.encoder.spatial_compression_ratio
        pixel_height, pixel_width = image.shape[-2:]
        assert pixel_height % spatial_ratio == 0
        assert pixel_width % spatial_ratio == 0

        text_embeddings = self.text_encoder(text)
        negative_text_embeddings = None
        if self.transformer_config.guidance_scale > 1.0:
            negative_text_embeddings = self.text_encoder([NEGATIVE_PROMPT] * len(text))
        image_embeddings = None
        if self.image_encoder is not None:
            image_embeddings = self.image_encoder(image.squeeze(-4))

        cache = LingbotEncoderStageCache(
            encoder_cache=self.encoder.initialize_autoregressive_cache(),
            image=image,
        )
        conditioning = LingbotConditioning(
            height=pixel_height // spatial_ratio,
            width=pixel_width // spatial_ratio,
            text_embeddings=text_embeddings,
            negative_text_embeddings=negative_text_embeddings,
            image_embeddings=image_embeddings,
        )
        return cache, conditioning

    def get_num_input_frames(self, autoregressive_index: int) -> int:
        """Return the pixel frames consumed by one encoder step."""
        return self.encoder.get_input_temporal_size(
            autoregressive_index,
            self.transformer_config.len_t,
        )

    def _preprocess_i2v_input(
        self,
        autoregressive_index: int,
        image: Tensor,
    ) -> Tensor:
        """Build the first-frame-plus-padding chunk expected by the Wan VAE."""
        expected_frames = self.get_num_input_frames(autoregressive_index)
        if autoregressive_index == 0:
            return F.pad(image, (0, 0, 0, 0, 0, 0, 0, expected_frames - 1))
        return torch.zeros(
            *image.shape[:-4],
            expected_frames,
            3,
            image.shape[-2],
            image.shape[-1],
            device=image.device,
            dtype=image.dtype,
        )

    @torch.no_grad()
    def encode(
        self,
        *,
        autoregressive_index: int,
        cache: LingbotEncoderStageCache,
        input: CamCtrlInput,
    ) -> I2VCamCtrlEmbeddings:
        """Encode one camera-control and I2V chunk."""
        i2v_input = self._preprocess_i2v_input(
            autoregressive_index,
            cache.image,
        )
        return self.encoder(
            input=I2VCamCtrlInput(i2v=i2v_input, camctrl=input),
            autoregressive_index=autoregressive_index,
            cache=cache.encoder_cache,
        )


class LingbotDiTStage(DiffusionStage[LingbotWorldTransformerCache]):
    """Own LingBot's scheduler, DiT weights, and evolving KV cache."""

    def __init__(self, config: LingbotWorldInferencePipelineConfig) -> None:
        super().__init__(config.diffusion_model)

    def initialize_cache(
        self,
        conditioning: LingbotConditioning,
    ) -> DiffusionStageCache[LingbotWorldTransformerCache]:
        """Build the resident DiT cache from encoder-stage conditioning."""
        return super().initialize_cache(
            height=conditioning.height,
            width=conditioning.width,
            text_embeddings=conditioning.text_embeddings,
            negative_text_embeddings=conditioning.negative_text_embeddings,
            image_embeddings=conditioning.image_embeddings,
        )


class LingbotDecoderStage(DecoderStage[WanVAECache]):
    """Own LingBot's streaming pixel decoder."""

    decoder: StreamingVideoDecoder[WanVAECache]

    def __init__(self, config: LingbotWorldInferencePipelineConfig) -> None:
        assert config.decoder is not None, "LingBot requires a video decoder."
        super().__init__(config.decoder)
        assert isinstance(self.decoder, StreamingVideoDecoder)

    def get_num_output_frames(self, autoregressive_index: int, len_t: int) -> int:
        """Return decoded pixel frames produced by one latent chunk."""
        return self.decoder.get_output_temporal_size(autoregressive_index, len_t)
