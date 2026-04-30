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
from flashdreams.recipes.wan.constants import NEGATIVE_PROMPT
from flashdreams.recipes.wan.transformer.wan21 import Wan21TransformerCache
from flashdreams.recipes.wan.transformer.wan22 import Wan22TransformerCache


@dataclass(kw_only=True)
class WanInferencePipelineCache(
    StreamInferencePipelineCache[
        I2VCtrlEncoderCache | None,  # EncCacheT  # ty:ignore[invalid-type-arguments]
        Wan21TransformerCache | Wan22TransformerCache,  # TransformerCacheT
        WanVAECache,  # DecCacheT
    ]
):
    """Per-rollout state for :class:`WanInferencePipeline`.

    Extends the infra :class:`StreamInferencePipelineCache` with:

    - ``image``: the raw first-frame pixel tensor of shape
      ``[*batch_shape, 1, 3, H, W]`` in ``[-1, 1]`` (or ``None`` for T2V).
      Stashed verbatim — encoding happens inside the infra encoder
      (``I2VCtrlEncoder``) per AR step, fed by the per-AR-step
      pixel input that :meth:`WanInferencePipeline.generate` builds from
      this tensor.

    The streaming Wan VAE's per-rollout cache lives on ``encoder_cache``,
    the diffusion model's :class:`DiffusionModel.FinalState` lives on
    ``final_state``, and the *current* AR step's :class:`EventProfiler`
    lives on ``event_profiler`` — all inherited from the infra cache.
    """

    image: Tensor | None = None


@dataclass(kw_only=True)
class WanInferencePipelineConfig(StreamInferencePipelineConfig):
    """Hyperparameters for :class:`WanInferencePipeline`.

    Extends :class:`StreamInferencePipelineConfig` (inherits
    ``diffusion_model``, ``encoder``, ``decoder``) with the recipe-level
    text/image encoders. The infra encoder slot drives I2V vs T2V mode:

    - T2V: ``encoder = None``. :meth:`WanInferencePipeline.initialize_cache`
      rejects any ``image`` argument; :meth:`WanInferencePipeline.generate`
      forwards ``input=None`` to the inherited
      :meth:`StreamInferencePipeline.generate`.
    - I2V: ``encoder = I2VCtrlEncoderConfig(...)``. The recipe
      pipeline pads the user-provided first frame along T to one full
      latent chunk and hands the pixel chunk to the inherited infra
      ``generate``, which runs the encoder (Wan VAE) and forwards the
      resulting :class:`I2VCtrl` to the transformer as the ``input``
      argument.
    """

    _target: type["WanInferencePipeline"] = field(
        default_factory=lambda: WanInferencePipeline
    )

    text_encoder: UMT5TextEncoderConfig = field(default_factory=UMT5TextEncoderConfig)
    """Text encoder run once at the start of each rollout."""

    image_encoder: CLIPImageEncoderConfig | None = None
    """CLIP image encoder for I2V variants whose ``WanDiTNetwork`` was
    trained with ``cross_attn_enable_img=True`` (e.g. Wan 2.1 14B I2V).
    Run once at the start of each rollout to produce the
    ``image_embeddings`` that get baked into the transformer cache via
    cross-attention. Leave as ``None`` for T2V or for I2V variants that
    do not consume CLIP features."""


class WanInferencePipeline(
    StreamInferencePipeline[
        I2VCtrlEncoderCache | None,  # EncCacheT  # ty:ignore[invalid-type-arguments]
        Wan21TransformerCache | Wan22TransformerCache,  # TransformerCacheT
        WanVAECache,  # DecCacheT
    ]
):
    """Unified Wan inference pipeline (Wan 2.1 / Wan 2.2, T2V and I2V).

    Whether the rollout is T2V or I2V is determined by the inherited
    infra encoder slot on :attr:`WanInferencePipelineConfig`:
    ``encoder = None`` ⇒ T2V; ``encoder = I2VCtrlEncoderConfig(...)``
    ⇒ I2V. The two modes share the same one-shot AR rollout API.

    Usage (T2V)::

        config = build_wan21_t2v_1pt3b_480p(device=device)
        pipeline = config.setup().to(device=device)

        cache = pipeline.initialize_cache(text=["A cat..."])
        chunk = pipeline.generate(0, cache)
        pipeline.finalize(0, cache)  # optional for single-step use

    Usage (I2V — same loop; the recipe pipeline auto-builds the
    per-AR-step pixel input from the stashed first frame)::

        config = build_wan21_i2v_14b_480p(device=device)
        pipeline = config.setup().to(device=device)

        cache = pipeline.initialize_cache(text=["A cat..."], image=first_frame)
        chunk = pipeline.generate(0, cache)
        pipeline.finalize(0, cache)
    """

    text_encoder: UMT5TextEncoder
    image_encoder: CLIPImageEncoder | None

    def __init__(self, config: WanInferencePipelineConfig) -> None:
        super().__init__(config)
        self.text_encoder = config.text_encoder.setup()  # ty:ignore[invalid-assignment]
        self.image_encoder = (  # ty:ignore[invalid-assignment]
            config.image_encoder.setup() if config.image_encoder is not None else None
        )

    @torch.no_grad()
    def initialize_cache(  # type: ignore[override]
        self,
        text: list[str],
        image: Tensor | None = None,
    ) -> WanInferencePipelineCache:
        """Initialize the per-rollout cache for a batch of prompts.

        Args:
            text: Flat list of prompts (one per batch element). The batch
                size is inferred from ``len(text)`` and must match the
                underlying transformer config's ``batch_shape``.
            image: Optional first-frame pixel tensor of shape
                ``[*batch_shape, 1, 3, H, W]`` in ``[-1, 1]``, where ``H``
                / ``W`` must match the transformer config's
                ``height`` / ``width`` *times* the decoder's spatial
                compression ratio. Required iff this pipeline has an I2V
                input encoder wired (``self.encoder is not None``);
                rejected otherwise. Stashed verbatim on the returned
                cache — :meth:`generate` expands it to a per-AR-step
                pixel chunk that the infra encoder consumes. When
                :attr:`image_encoder` is configured, this same image is
                also fed (T-axis squeezed) to the CLIP image encoder
                once, and the resulting embeddings are baked into the
                transformer cache.
        """
        assert len(text) > 0, "text must be non-empty"
        n = len(text)

        text_embeddings = self.text_encoder(text)  # [B, L, D]

        guidance_scale = self.diffusion_model.transformer.config.guidance_scale  # ty:ignore[unresolved-attribute]
        if guidance_scale > 1.0:
            negative_text_embeddings = self.text_encoder([NEGATIVE_PROMPT] * n)
        else:
            negative_text_embeddings = None

        # Symmetric T2V/I2V validation: presence of an I2V encoder MUST
        # match the presence of an image. We do *not* VAE-encode the
        # image here — the infra encoder runs per AR step in
        # :meth:`generate` so the streaming Wan VAE's temporal cache
        # advances correctly.
        if image is not None:
            assert self.encoder is not None, (
                "Image was provided but the pipeline has no I2V input "
                "encoder; configure encoder to a "
                "I2VCtrlEncoderConfig for I2V."
            )
            assert image.shape[-4] == 1, (
                f"image must have a single time step (T=1), got shape "
                f"{tuple(image.shape)}"
            )
        else:
            assert self.encoder is None, (
                "Image was not provided but the pipeline has an I2V input encoder."
            )

        # CLIP image embeddings (only for I2V variants where the
        # underlying network has ``cross_attn_enable_img=True``).
        image_embeddings: Tensor | None = None
        if self.image_encoder is not None:
            assert image is not None, (
                "image_encoder is configured but no image was provided."
            )
            # Drop the T=1 axis: CLIP wants [..., C, H, W].
            image_embeddings = self.image_encoder(image.squeeze(-4))

        parent = super().initialize_cache(
            transformer_context={
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
        """Build the per-AR-step pixel chunk fed to the I2V input encoder.

        - AR step 0: ``cat(image, zeros)`` — one anchor pixel frame
          followed by ``(len_t - 1) * 4`` zero pixel frames so a fresh
          streaming Wan VAE produces ``len_t`` latent frames whose first
          frame is the encoded image.
        - AR steps ``> 0``: pure zeros so the streaming VAE flushes its
          temporal context. The encoder pairs this with an all-zero
          mask, so the resulting latent is ignored by the network's
          mask-injection arithmetic. Single-AR-step rollouts (the
          common Wan 2.1 non-streaming case) never hit this branch.
        """
        H, W = image.shape[-2:]
        device = image.device
        dtype = image.dtype
        batch_shape = image.shape[:-4]

        expected_frames = self.get_num_input_frames(autoregressive_index)
        if autoregressive_index == 0:
            # F.pad pads from the last dim backward; this tuple pads only
            # the T axis (the -4 dim) with ``num_pad`` zeros at the end.
            num_pad = expected_frames - 1
            return F.pad(image, (0, 0, 0, 0, 0, 0, 0, num_pad))
        else:
            return torch.zeros(
                *batch_shape, expected_frames, 3, H, W, device=device, dtype=dtype
            )

    @torch.no_grad()
    def generate(  # type: ignore[override]
        self,
        autoregressive_index: int,
        cache: WanInferencePipelineCache,
    ) -> Tensor:
        """Generate one decoded video chunk for AR step ``autoregressive_index``.

        For T2V the inherited infra :meth:`generate` is called with
        ``input=None``. For I2V the recipe builds a per-AR-step pixel
        chunk via :meth:`_preprocess_i2v_input` and hands it to the
        infra as ``input``; the infra runs the streaming Wan VAE
        encoder and pairs the resulting latent with a binary injection
        mask (:class:`I2VCtrl`) which it patchifies and forwards
        to the transformer as the ``input`` argument.

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
        """Number of input video frames accepted by the model."""
        len_t = self.diffusion_model.transformer.config.len_t  # ty:ignore[unresolved-attribute]
        temporal_compression_ratio = self.encoder.temporal_compression_ratio  # ty:ignore[unresolved-attribute]
        if autoregressive_index == 0:
            return 1 + (len_t - 1) * temporal_compression_ratio
        else:
            return len_t * temporal_compression_ratio

    def get_num_output_frames(self, autoregressive_index: int) -> int:
        """Number of output video frames produced by the model."""
        len_t = self.diffusion_model.transformer.config.len_t  # ty:ignore[unresolved-attribute]
        temporal_compression_ratio = self.decoder.temporal_compression_ratio  # ty:ignore[unresolved-attribute]
        if autoregressive_index == 0:
            return 1 + (len_t - 1) * temporal_compression_ratio
        else:
            return len_t * temporal_compression_ratio
