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

"""Streaming inference pipeline for Omnidreams (Cosmos DiT + HDMap + I2V)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypeAlias

import torch
from omnidreams.constants import NEGATIVE_PROMPT
from omnidreams.transformer import (
    CosmosTransformer,
    CosmosTransformerCache,
    CosmosTransformerConfig,
)
from omnidreams.vae_native import OmnidreamsWanVAEEncoderConfig
from torch import Tensor

from flashdreams.core.distributed.context_parallel import (
    cat_outputs_cp,
    split_inputs_cp,
    split_inputs_cp_object_list,
)
from flashdreams.infra.acceleration.encoder_lifecycle import (
    move_tensors_to_cpu,
    release_one_shot_encoder_references,
)
from flashdreams.infra.decoder import StreamingDecoderCache
from flashdreams.infra.encoder import StreamingEncoderCache
from flashdreams.infra.encoder.text.cosmos_reason1 import (
    CosmosReason1TextEncoder,
    CosmosReason1TextEncoderConfig,
)
from flashdreams.infra.pipeline import (
    StreamInferencePipeline,
    StreamInferencePipelineCache,
    StreamInferencePipelineConfig,
)
from flashdreams.recipes.taehv import TeahvVAEDecoder
from flashdreams.recipes.wan.autoencoder.vae import (
    WanVAEDecoder,
    WanVAEEncoder,
)

OmnidreamsPipelineCache: TypeAlias = StreamInferencePipelineCache[
    StreamingEncoderCache,  # StreamingEncoderCacheT
    CosmosTransformerCache,  # TransformerCacheT
    StreamingDecoderCache,  # StreamingDecoderCacheT
]


def _validate_pixel_resolution_alignment(
    pixel_height: int,
    pixel_width: int,
    *,
    spatial_compression_ratio: int,
    patch_spatial: int,
) -> None:
    required_alignment = int(spatial_compression_ratio) * int(patch_spatial)
    if pixel_height % required_alignment or pixel_width % required_alignment:
        raise ValueError(
            "image pixel size "
            f"({pixel_width}, {pixel_height}) must be divisible by "
            f"{required_alignment} so the VAE latent is divisible by "
            f"patch_spatial={patch_spatial}."
        )


@dataclass(kw_only=True)
class OmnidreamsPipelineConfig(StreamInferencePipelineConfig):
    """Config for the Omnidreams pipeline.

    The infra ``encoder`` slot must be set to the per-AR-step HDMap encoder.
    The transformer config must be a Cosmos transformer config.
    """

    _target: type["OmnidreamsPipeline"] = field(
        default_factory=lambda: OmnidreamsPipeline
    )

    text_encoder: CosmosReason1TextEncoderConfig | None = field(
        default_factory=CosmosReason1TextEncoderConfig
    )
    """Cosmos-Reason1 text encoder run once per rollout. Set to ``None``
    and use ``initialize_cache_from_embeddings`` with precomputed embeddings
    to skip loading it."""

    image_encoder: OmnidreamsWanVAEEncoderConfig | None = field(
        default_factory=OmnidreamsWanVAEEncoderConfig
    )
    """One-shot Wan VAE first-frame encoder. Pin its checkpoint to the
    VAE the network was trained against. ``None`` skips loading."""

    synthetic_text_max_length: int | None = None
    """Text token count for the interactive-drive synthetic latency path.

    ``None`` on every real run, where the text encoder produces the
    embeddings. The synthetic path drops the text encoder and instead builds
    zero text embeddings, so it records the production sequence length here
    (via ``derive_config``) to keep the embedding shape -- and therefore the
    measured cost -- identical to a real run."""


class OmnidreamsPipeline(
    StreamInferencePipeline[
        StreamingEncoderCache,
        CosmosTransformerCache,
        StreamingDecoderCache,
    ]
):
    """Omnidreams streaming inference pipeline (Cosmos DiT + HDMap + I2V mask).

    Examples:

        pipeline: OmnidreamsPipeline = ...

        cache = pipeline.initialize_cache(
            text=[["A driving scene..."]],
            image=first_frames,
            view_names=["camera_front_..."],
        )
        chunk = pipeline.generate(0, cache, hdmap=hdmap_chunk_0)
        pipeline.finalize(0, cache)
        chunk = pipeline.generate(1, cache, hdmap=hdmap_chunk_1)
        pipeline.finalize(1, cache)  # optional for the last rollout
    """

    text_encoder: CosmosReason1TextEncoder | None
    image_encoder: WanVAEEncoder | None

    def __init__(self, config: OmnidreamsPipelineConfig) -> None:
        super().__init__(config)
        self.text_encoder = (
            config.text_encoder.setup() if config.text_encoder is not None else None
        )
        self.image_encoder = (
            config.image_encoder.setup() if config.image_encoder is not None else None
        )

        assert self.encoder is not None, (
            "OmnidreamsPipeline requires a per-AR-step HDMap encoder; "
            "set StreamInferencePipelineConfig.encoder."
        )

        transformer = self.diffusion_model.transformer
        assert isinstance(transformer, CosmosTransformer), (
            "OmnidreamsPipeline requires a Cosmos transformer; "
            f"got {type(transformer).__name__}."
        )
        self._len_t_latent: int = transformer.config.len_t
        decoder = self.decoder
        assert isinstance(decoder, (WanVAEDecoder, TeahvVAEDecoder)), (
            "OmnidreamsPipeline requires a Wan or Taehv VAE decoder; "
            f"got {type(decoder).__name__}."
        )
        self._decoder_temporal_compression: int = type(
            decoder
        ).TEMPORAL_COMPRESSION_RATIO

        # Split views outside the transformer so the VAE doesn't duplicate work across CP ranks.
        self.V_group = transformer.cp_groups.V_group
        self.V_size = transformer.cp_groups.V_size
        transformer.cp_groups.V_group = None
        transformer.config.num_views //= self.V_size

    @property
    def device(self) -> torch.device:
        return self.diffusion_model.device

    @property
    def _use_negative_prompt(self) -> bool:
        cfg = self.diffusion_model.transformer.config
        assert isinstance(cfg, CosmosTransformerConfig)
        return cfg.requires_negative_text_embeddings

    @torch.no_grad()
    def initialize_cache(
        self,
        text: list[list[str]],
        image: Tensor,
        view_names: list[str] | None = None,
    ) -> OmnidreamsPipelineCache:
        """Initialize the per-rollout cache from raw prompts and images.

        Args:
            text: ``[B, V]`` nested list of prompts (one per view).
            image: First-frame pixels ``[B, V, 1, 3, H, W]`` in ``[-1, 1]``.
                ``H``/``W`` must equal latent ``height``/``width`` times the
                decoder's spatial compression ratio.
                CFG-enabled configs use the integration's training negative prompt
                automatically.
            view_names: View names (length ``V``); required when
                ``num_views > 1``.
        """
        assert self.text_encoder is not None and self.image_encoder is not None, (
            "initialize_cache(text=, image=) requires text_encoder and "
            "image_encoder to be loaded. They are None either because the "
            "configs were set to None at construction or because "
            "release_oneshot_encoders() has been called. If you have "
            "precomputed embeddings, use initialize_cache_from_embeddings()."
        )
        assert isinstance(text, list) and len(text) > 0 and isinstance(text[0], list), (
            f"text must be a [B, V] nested list of prompts, got {type(text)}"
        )
        self._validate_image_resolution(image)

        text_embeddings = torch.stack(
            [self.text_encoder(t) for t in text], dim=0
        )  # [B, V, L, D]
        image_embeddings = self.image_encoder(image)  # [B, V, 1, Cl, Hl, Wl]
        negative_text_embeddings: Tensor | None = None
        if self._use_negative_prompt:
            negative_text_embeddings = torch.stack(
                [
                    self.text_encoder([NEGATIVE_PROMPT for _ in prompt_row])
                    for prompt_row in text
                ],
                dim=0,
            )

        return self.initialize_cache_from_embeddings(
            text_embeddings=text_embeddings,
            image_embeddings=image_embeddings,
            negative_text_embeddings=negative_text_embeddings,
            view_names=view_names,
        )

    @torch.no_grad()
    def initialize_cache_from_embeddings(
        self,
        text_embeddings: Tensor,
        image_embeddings: Tensor,
        negative_text_embeddings: Tensor | None = None,
        view_names: list[str] | None = None,
    ) -> OmnidreamsPipelineCache:
        """Initialize the per-rollout cache from precomputed embeddings.

        Use this when the one-shot encoders aren't loaded (typically because
        embeddings were precomputed offline by ``precompute_embeddings`` to
        save VRAM at rollout time).

        Args:
            text_embeddings: ``[B, V, L, D]``. Moved to ``self.device``.
                Pass the full multi-view tensor; the CP split is applied here.
            image_embeddings: ``[B, V, 1, Cl, Hl, Wl]``. Same device / split
                contract as ``text_embeddings``. ``Hl`` / ``Wl`` define the
                per-rollout latent ``(height, width)`` forwarded to the
                transformer.
            negative_text_embeddings: Optional ``[B, V, L, D]`` embeddings for
                the integration's training negative prompt. Required when the
                transformer config requires negative text embeddings.
            view_names: View names (length ``V``); required when
                ``num_views > 1``.
        """
        text_embeddings = text_embeddings.to(device=self.device)
        image_embeddings = image_embeddings.to(device=self.device)
        if negative_text_embeddings is not None:
            negative_text_embeddings = negative_text_embeddings.to(device=self.device)

        # Image latent [..., Hl, Wl] = per-rollout latent spatial dims for the cache init.
        height = image_embeddings.shape[-2]
        width = image_embeddings.shape[-1]

        text_embeddings = split_inputs_cp(
            text_embeddings,
            seq_dim=1,
            cp_group=self.V_group,
        )
        image_embeddings = split_inputs_cp(
            image_embeddings,
            seq_dim=1,
            cp_group=self.V_group,
        )
        if negative_text_embeddings is not None:
            negative_text_embeddings = split_inputs_cp(
                negative_text_embeddings,
                seq_dim=1,
                cp_group=self.V_group,
            )
        if view_names is not None:
            view_names = split_inputs_cp_object_list(view_names, cp_group=self.V_group)

        transformer_context = {
            "height": height,
            "width": width,
            "text_embeddings": text_embeddings,
            "image_embeddings": image_embeddings,
            "view_names": view_names,
        }
        if negative_text_embeddings is not None:
            transformer_context["negative_text_embeddings"] = negative_text_embeddings

        return super().initialize_cache(
            transformer_context=transformer_context,
        )

    @torch.no_grad()
    def precompute_embeddings(
        self,
        text: list[list[str]],
        image: Tensor,
    ) -> dict[str, Tensor | None]:
        """Run only the one-shot encoders and return their outputs on CPU.

        Pair with ``initialize_cache_from_embeddings``: save the returned
        dict, then build a pipeline with the encoder configs set to ``None``
        and rebuild the cache from the loaded tensors. The returned tensors
        are not CP-split, so the same file works for any CP world size.

        Args:
            text: ``[B, V]`` nested list of prompts.
            image: ``[B, V, 1, 3, H, W]`` first-frame pixels in ``[-1, 1]``.

        Returns:
            ``{"text_embeddings": [B, V, L, D],
            "image_embeddings": [B, V, 1, Cl, Hl, Wl],
            "negative_text_embeddings": [B, V, L, D] | None}`` on CPU.
        """
        assert self.text_encoder is not None and self.image_encoder is not None, (
            "precompute_embeddings requires text_encoder and image_encoder "
            "to be loaded; build the pipeline with both configs non-None."
        )
        assert isinstance(text, list) and len(text) > 0 and isinstance(text[0], list), (
            f"text must be a [B, V] nested list of prompts, got {type(text)}"
        )
        self._validate_image_resolution(image)
        text_embeddings = torch.stack(
            [self.text_encoder(t) for t in text], dim=0
        )  # [B, V, L, D]
        image_embeddings = self.image_encoder(image)  # [B, V, 1, Cl, Hl, Wl]

        if self._use_negative_prompt:
            negative_text_embeddings = torch.stack(
                [
                    self.text_encoder([NEGATIVE_PROMPT for _ in prompt_row])
                    for prompt_row in text
                ],
                dim=0,
            )
        else:
            negative_text_embeddings = None

        return move_tensors_to_cpu(
            {
                "text_embeddings": text_embeddings,
                "image_embeddings": image_embeddings,
                "negative_text_embeddings": negative_text_embeddings,
            },
            torch_module=torch,
        )

    @torch.no_grad()
    def replace_text(
        self,
        cache: OmnidreamsPipelineCache,
        text: list[list[str]],
        *,
        guidance_scale: float = 1.0,
        guidance_chunks: int = 0,
        recache_last_chunk: bool = False,
    ) -> None:
        """Hot-swap the rollout's prompt between two AR steps.

        Encodes ``text`` with the resident text encoder and rebuilds the
        cross-attention text K/V in place; the self-attention history keeps
        the generated scene, so the video continues seamlessly under the new
        prompt. Call after ``finalize`` of one AR step and before
        ``generate`` of the next.

        Args:
            cache: Live per-rollout cache.
            text: ``[B, V]`` nested list of prompts, as in
                ``initialize_cache``.
            guidance_scale: Optional edit strength (``> 1.0`` pushes the
                flow along the new-minus-old text direction for the next
                ``guidance_chunks`` chunks at the cost of one extra network
                forward per denoising step).
            guidance_chunks: Number of upcoming chunks to guide.
            recache_last_chunk: Re-commit the previous chunk's KV history
                under the new prompt (one extra context forward), so the
                window the next chunk attends to is already "explained" by
                the new text. Helps the scene react faster after a swap.
        """
        assert self.text_encoder is not None, (
            "replace_text requires the text encoder to be loaded; use "
            "replace_text_from_embeddings with precomputed embeddings "
            "otherwise."
        )
        assert isinstance(text, list) and len(text) > 0 and isinstance(text[0], list), (
            f"text must be a [B, V] nested list of prompts, got {type(text)}"
        )
        text_embeddings = torch.stack(
            [self.text_encoder(t) for t in text], dim=0
        )  # [B, V, L, D]
        self.replace_text_from_embeddings(
            cache,
            text_embeddings,
            guidance_scale=guidance_scale,
            guidance_chunks=guidance_chunks,
            recache_last_chunk=recache_last_chunk,
        )

    @torch.no_grad()
    def replace_text_from_embeddings(
        self,
        cache: OmnidreamsPipelineCache,
        text_embeddings: Tensor,
        *,
        guidance_scale: float = 1.0,
        guidance_chunks: int = 0,
        recache_last_chunk: bool = False,
    ) -> None:
        """``replace_text`` for precomputed ``[B, V, L, D]`` embeddings."""
        transformer = self.diffusion_model.transformer
        assert isinstance(transformer, CosmosTransformer)
        text_embeddings = text_embeddings.to(device=self.device)
        text_embeddings = split_inputs_cp(
            text_embeddings, seq_dim=1, cp_group=self.V_group
        )
        transformer.replace_text_embeddings(
            cache.transformer_cache,
            text_embeddings,
            guidance_scale=guidance_scale,
            guidance_chunks=guidance_chunks,
        )
        if recache_last_chunk:
            self.recache_last_chunk(cache)

    _RECACHE_NOISE_SEED = 118_000
    """Base seed for the ReCache context-noise draw (offset by AR index)."""

    @torch.no_grad()
    def recache_last_chunk(self, cache: OmnidreamsPipelineCache) -> None:
        """Re-commit the previous chunk's KV history under the current text.

        Re-opens the just-finalized AR step (``BlockKVCache`` permits
        same-index rewrites: the window does not roll and the same physical
        slots are overwritten) and re-runs the context forward, so the
        cached history becomes consistent with a freshly swapped prompt.
        Requires the step's ``finalize`` to have completed; a no-op before
        the first ``generate``.

        The context-noise draw comes from a dedicated generator seeded by
        the AR index, not the model RNG — every noise rendition of the same
        clean latent is in-distribution for the context forward (each
        chunk's original commit already uses an independent draw), and
        keeping the model RNG untouched means the rollout's subsequent
        noise stream is identical with or without ReCache. Seedless
        configurations (``DiffusionModelConfig.seed is None``) fall back to
        the global RNG, matching their existing no-reproducibility
        contract.
        """
        final_state = cache.final_state
        if final_state is None:
            return
        diffusion_model = self.diffusion_model
        # Materialize the lazy model generator BEFORE snapshotting, so the
        # restore never resets the rollout's noise stream to its seed.
        seeded = diffusion_model.rng is not None
        saved_rng = diffusion_model._rng
        if seeded:
            diffusion_model._rng = torch.Generator(device=self.device).manual_seed(
                self._RECACHE_NOISE_SEED + final_state.autoregressive_index
            )
        try:
            final_state.cache.start(final_state.autoregressive_index)
            diffusion_model.finalize(final_state=final_state)
        finally:
            diffusion_model._rng = saved_rng

    def _validate_image_resolution(self, image: Tensor) -> None:
        transformer = self.diffusion_model.transformer
        assert isinstance(transformer, CosmosTransformer), (
            "OmnidreamsPipeline requires a Cosmos transformer; "
            f"got {type(transformer).__name__}."
        )
        decoder = self.decoder
        assert isinstance(decoder, (WanVAEDecoder, TeahvVAEDecoder)), (
            "OmnidreamsPipeline requires a Wan or Taehv VAE decoder; "
            f"got {type(decoder).__name__}."
        )
        _validate_pixel_resolution_alignment(
            int(image.shape[-2]),
            int(image.shape[-1]),
            spatial_compression_ratio=int(decoder.spatial_compression_ratio),
            patch_spatial=int(transformer.config.network.patch_spatial),
        )

    def release_oneshot_encoders(self) -> None:
        """Free the per-rollout text and first-frame image encoders.

        Cosmos-Reason1-7B alone is ~14 GB in bf16, so dropping it after
        ``initialize_cache`` reclaims significant VRAM for the AR rollout.
        Idempotent. Only safe for one-shot pipeline lifetimes (demos);
        long-lived hosts that reuse the pipeline must not call this.
        """
        release_one_shot_encoder_references(
            self,
            "text_encoder",
            "image_encoder",
            torch_module=torch,
        )

    @torch.no_grad()
    def generate(
        self,
        autoregressive_index: int,
        cache: OmnidreamsPipelineCache,
        hdmap: Tensor,
    ) -> Tensor:
        """Generate one decoded video chunk.

        Args:
            autoregressive_index: AR step index (0-based).
            cache: Per-rollout cache from ``initialize_cache``.
            hdmap: Per-AR-step HDMap pixels ``[B, V, T, 3, H, W]`` in
                ``[-1, 1]``. ``T`` must equal ``get_num_frames(autoregressive_index)``.

        Returns:
            Decoded video chunk ``[B, V, T, 3, H, W]`` in ``[-1, 1]``.
        """
        hdmap = split_inputs_cp(hdmap, seq_dim=1, cp_group=self.V_group)

        output = super().generate(
            autoregressive_index=autoregressive_index,
            cache=cache,
            input=hdmap,
        )

        output = cat_outputs_cp(output, seq_dim=1, cp_group=self.V_group)
        return output

    def get_num_frames(self, autoregressive_index: int) -> int:
        """Number of decoded video frames produced at this AR step."""
        if autoregressive_index == 0:
            return 1 + (self._len_t_latent - 1) * self._decoder_temporal_compression
        return self._len_t_latent * self._decoder_temporal_compression
