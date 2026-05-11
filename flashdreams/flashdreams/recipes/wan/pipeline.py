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

"""Unified Wan inference pipeline (Wan 2.1 / Wan 2.2, T2V and I2V)."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import Tensor

from flashdreams.infra.decoder import StreamingVideoDecoder
from flashdreams.infra.encoder import StreamingVideoEncoder
from flashdreams.infra.encoder.image.clip import (
    CLIPImageEncoder,
    CLIPImageEncoderConfig,
)
from flashdreams.infra.encoder.text.umt5 import (
    UMT5TextEncoder,
    UMT5TextEncoderConfig,
)
from flashdreams.infra.pipeline import (
    StreamInferencePipeline,
    StreamInferencePipelineCache,
    StreamInferencePipelineConfig,
)
from flashdreams.recipes.wan.autoencoder.i2v import I2VCtrlEncoderCache
from flashdreams.recipes.wan.autoencoder.vae import WanVAECache
from flashdreams.recipes.wan.transformer.constants import NEGATIVE_PROMPT
from flashdreams.recipes.wan.transformer.wan21 import (
    Wan21TransformerCache,
    Wan21TransformerConfig,
)
from flashdreams.recipes.wan.transformer.wan22 import (
    Wan22TransformerCache,
    Wan22TransformerConfig,
)


@dataclass(kw_only=True)
class WanInferencePipelineCache(
    StreamInferencePipelineCache[
        I2VCtrlEncoderCache,
        Wan21TransformerCache | Wan22TransformerCache,
        WanVAECache,
    ]
):
    """Per-rollout state for the Wan pipeline.

    Adds the I2V first-frame pixels on top of the inherited caches. Pixel-to-
    latent encoding happens per AR step inside the encoder, not here.
    """

    image: Tensor | None = None
    """First-frame pixels ``[*batch_shape, 1, 3, H, W]`` in ``[-1, 1]``;
    ``None`` for T2V."""


@dataclass(kw_only=True)
class WanInferencePipelineConfig(StreamInferencePipelineConfig):
    """Config for the Wan inference pipeline.

    T2V vs I2V is selected by the inherited ``encoder`` slot: ``None`` for
    T2V, an I2V control-encoder config for I2V.
    """

    _target: type["WanInferencePipeline"] = field(
        default_factory=lambda: WanInferencePipeline
    )

    text_encoder: UMT5TextEncoderConfig = field(default_factory=UMT5TextEncoderConfig)
    """UMT5 text encoder run once per rollout."""

    image_encoder: CLIPImageEncoderConfig | None = None
    """CLIP image encoder for I2V variants trained with
    ``cross_attn_enable_img=True`` (Wan 2.1 14B I2V). ``None`` skips CLIP
    cross-attention conditioning."""


class WanInferencePipeline(
    StreamInferencePipeline[
        I2VCtrlEncoderCache,
        Wan21TransformerCache | Wan22TransformerCache,
        WanVAECache,
    ]
):
    """Wan 2.1 / 2.2 inference pipeline, T2V and I2V.

    T2V and I2V share the same rollout loop; the difference is whether you
    pass an ``image`` to ``initialize_cache``. The pipeline config's
    ``encoder`` slot must agree (``None`` for T2V, an I2V config for I2V).

    Examples:

        pipeline: WanInferencePipeline = ...

        # T2V: latent (height, width) are required (the pipeline has no
        # image to derive them from). Convert from a target pixel size
        # via the decoder's ``spatial_compression_ratio``:
        #   height = pixel_h // pipeline.decoder.spatial_compression_ratio
        cache = pipeline.initialize_cache(
            text=["A cat surfing."], height=60, width=104
        )
        chunk = pipeline.generate(0, cache)
        pipeline.finalize(0, cache)
        chunk = pipeline.generate(1, cache)
        pipeline.finalize(1, cache)  # optional for the last rollout

        # I2V: pass ``image=first_frame``. ``height`` / ``width`` are
        # derived from the image and the decoder's spatial compression
        # ratio; passing them is optional (and cross-checked when set).
        i2v_cache = pipeline.initialize_cache(
            text=["A cat surfing."], image=first_frame
        )
    """

    text_encoder: UMT5TextEncoder
    image_encoder: CLIPImageEncoder | None

    def __init__(self, config: WanInferencePipelineConfig) -> None:
        super().__init__(config)
        self.text_encoder = config.text_encoder.setup()
        self.image_encoder = (
            config.image_encoder.setup() if config.image_encoder is not None else None
        )

    @property
    def _transformer_config(self) -> Wan21TransformerConfig | Wan22TransformerConfig:
        # Narrow the base transformer config to the Wan-specific union so
        # ``guidance_scale`` / ``len_t`` are visible to the type checker.
        cfg = self.diffusion_model.transformer.config
        assert isinstance(cfg, (Wan21TransformerConfig, Wan22TransformerConfig))
        return cfg

    @torch.no_grad()
    def initialize_cache(
        self,
        text: list[str],
        image: Tensor | None = None,
        *,
        height: int | None = None,
        width: int | None = None,
    ) -> WanInferencePipelineCache:
        """Initialize the per-rollout cache for a batch of prompts.

        Args:
            text: One prompt per batch element. Length must match the
                transformer's ``batch_shape``.
            image: First-frame pixels of shape ``[*batch_shape, 1, 3, H, W]``
                in ``[-1, 1]``. Required for I2V (``self.encoder`` is set),
                forbidden for T2V. ``H`` / ``W`` must equal
                ``height * decoder.spatial_compression_ratio`` and likewise
                for ``W``.
            height: Pre-patchify latent height (post-VAE). Optional for
                I2V — derived from ``image`` when omitted; required for T2V.
            width: Pre-patchify latent width (post-VAE). Same rules as
                ``height``.

        Returns:
            Cache to thread through ``generate`` / ``finalize``.
        """
        assert len(text) > 0, "text must be non-empty"
        n = len(text)

        text_embeddings = self.text_encoder(text)  # [B, L, D]

        guidance_scale = self._transformer_config.guidance_scale
        if guidance_scale > 1.0:
            negative_text_embeddings = self.text_encoder([NEGATIVE_PROMPT] * n)
        else:
            negative_text_embeddings = None

        # Encoder presence and image presence must agree. The image is *not*
        # VAE-encoded here: that happens per AR step inside the encoder so
        # the streaming Wan VAE's temporal cache advances correctly.
        if image is not None:
            assert self.encoder is not None, (
                "Image was provided but the pipeline has no I2V input "
                "encoder; configure encoder to a WanI2VCtrlEncoderConfig."
            )
            assert image.shape[-4] == 1, (
                f"image must have a single time step (T=1), got shape "
                f"{tuple(image.shape)}"
            )
        else:
            assert self.encoder is None, (
                "Image was not provided but the pipeline has an I2V input encoder."
            )

        # Derive (or cross-check) latent (height, width) from the image when
        # it is provided. The decoder owns the pixel<->latent ratio; the
        # encoder is assumed to share it (Wan VAE encoder/decoder do).
        if image is not None:
            assert isinstance(self.decoder, StreamingVideoDecoder), (
                f"I2V requires a StreamingVideoDecoder; "
                f"got {type(self.decoder).__name__}."
            )
            sp = self.decoder.spatial_compression_ratio
            pixel_h, pixel_w = image.shape[-2], image.shape[-1]
            assert pixel_h % sp == 0 and pixel_w % sp == 0, (
                f"image pixel size ({pixel_h}, {pixel_w}) must be divisible "
                f"by decoder.spatial_compression_ratio={sp}."
            )
            derived_h, derived_w = pixel_h // sp, pixel_w // sp
            if height is None:
                height = derived_h
            else:
                assert height == derived_h, (
                    f"height={height} does not match image latent height "
                    f"derived from pixels ({derived_h})."
                )
            if width is None:
                width = derived_w
            else:
                assert width == derived_w, (
                    f"width={width} does not match image latent width "
                    f"derived from pixels ({derived_w})."
                )
        assert height is not None and width is not None, (
            "T2V (image=None) requires explicit `height` and `width` latent dims."
        )

        image_embeddings: Tensor | None = None
        if self.image_encoder is not None:
            assert image is not None, (
                "image_encoder is configured but no image was provided."
            )
            # CLIP wants [..., C, H, W]; drop the T=1 axis.
            image_embeddings = self.image_encoder(image.squeeze(-4))

        parent = super().initialize_cache(
            transformer_context={
                "height": height,
                "width": width,
                "text_embeddings": text_embeddings,
                "negative_text_embeddings": negative_text_embeddings,
                "image_embeddings": image_embeddings,
            },
        )
        return WanInferencePipelineCache(
            transformer_cache=parent.transformer_cache,
            encoder_cache=parent.encoder_cache,
            decoder_cache=parent.decoder_cache,
            image=image,
        )

    def _preprocess_i2v_input(
        self,
        autoregressive_index: int,
        image: Tensor,
    ) -> Tensor:
        """Build the per-AR-step pixel chunk for the I2V encoder.

        Step 0 prepends the anchor frame and zero-pads along T so the VAE
        emits ``len_t`` latent frames with the encoded image at index 0.
        Later steps return all zeros so the streaming VAE flushes its
        temporal context; the encoder pairs that with an all-zero mask, so
        the latent contributes nothing to the network output.
        """
        H, W = image.shape[-2:]
        device = image.device
        dtype = image.dtype
        batch_shape = image.shape[:-4]

        expected_frames = self.get_num_input_frames(autoregressive_index)
        if autoregressive_index == 0:
            # F.pad pads from the last dim backward; this targets the T axis.
            num_pad = expected_frames - 1
            return F.pad(image, (0, 0, 0, 0, 0, 0, 0, num_pad))
        else:
            return torch.zeros(
                *batch_shape, expected_frames, 3, H, W, device=device, dtype=dtype
            )

    @torch.no_grad()
    def generate(
        self,
        autoregressive_index: int,
        cache: WanInferencePipelineCache,
    ) -> Tensor:
        """Generate one decoded video chunk.

        Args:
            autoregressive_index: AR step index, starting at 0.
            cache: Per-rollout cache from ``initialize_cache``.

        Returns:
            Decoded video of shape ``[*batch_shape, T, C, H, W]`` in ``[-1, 1]``.
        """
        input: Tensor | None = None
        if cache.image is not None:
            input = self._preprocess_i2v_input(autoregressive_index, cache.image)

        return super().generate(
            autoregressive_index=autoregressive_index,
            cache=cache,
            input=input,
        )

    def get_num_input_frames(self, autoregressive_index: int) -> int:
        """Number of input video frames the model expects at this AR step."""
        len_t = self._transformer_config.len_t
        assert isinstance(self.encoder, StreamingVideoEncoder), (
            f"get_num_input_frames requires a StreamingVideoEncoder; "
            f"got {type(self.encoder).__name__}."
        )
        return self.encoder.get_input_temporal_size(autoregressive_index, len_t)

    def get_num_output_frames(self, autoregressive_index: int) -> int:
        """Number of decoded video frames produced at this AR step."""
        len_t = self._transformer_config.len_t
        assert isinstance(self.decoder, StreamingVideoDecoder), (
            f"get_num_output_frames requires a StreamingVideoDecoder; "
            f"got {type(self.decoder).__name__}."
        )
        return self.decoder.get_output_temporal_size(autoregressive_index, len_t)
