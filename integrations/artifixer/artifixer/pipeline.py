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

"""ArtiFixer inference pipeline.

Extends :class:`WanInferencePipeline` with the ArtiFixer-specific
conditioning surface: full-rollout opacity, Plucker camera rays, target
camera w2c/Ks, optional VAE-encoded neighbor frames + neighbor camera
matrices.

Phase 3.5a (this commit) lands the cache + ``initialize_cache``:

  - VAE-encoded condition latent and neighbor latent arrive pre-computed
    from the caller (the dreamfix-side driver in Phase 4) so this
    pipeline does not need its own VAE encoder.
  - On ``initialize_cache`` we:
      1. Run the base ``WanInferencePipeline.initialize_cache`` (text
         encoder + text K/V build) with ``image=None``.
      2. Project the neighbor latent through ``patch_embedding`` to get
         the per-token neighbor context, then push it into every block's
         :class:`ArtifixerCrossAttention.neighbor_kv_cache` via
         ``ArtifixerDiTNetwork.initialize_neighbor_kv_caches``.
      3. Update the patches_x/patches_y RoPE coefficients on both PRoPE
         modules and precompute the neighbor-side ``apply_fns`` (the
         neighbor cameras are static across AR steps).
      4. Stash the full-rollout opacity / camera_rays / w2cs / Ks on the
         cache for the per-AR ``generate`` slicing (Phase 3.5b).

Phase 3.5b adds ``generate``: per-AR-chunk PRoPE-src precompute,
opacity-weighted latent mix, manual denoise loop with prepare_latents
renoise, and decode.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from artifixer.transformer import ArtifixerWanTransformer
from torch import Tensor

from flashdreams.recipes.wan.pipeline import (
    WanInferencePipeline,
    WanInferencePipelineCache,
    WanInferencePipelineConfig,
)


@dataclass(kw_only=True)
class ArtifixerInferencePipelineCache(WanInferencePipelineCache):
    """Per-rollout cache for the ArtiFixer pipeline.

    Holds the full-rollout conditioning tensors so :meth:`generate` can
    slice the chunk's frame range on demand. All optional fields default
    to ``None`` so callers that do not have neighbors still construct a
    valid cache.
    """

    condition_latent: Tensor | None = None
    """VAE-encoded rendered RGB at latent resolution
    ``[*batch_shape, in_dim, T_lat, Hl, Wl]``. Pre-encoded by the
    caller."""

    opacity: Tensor | None = None
    """Per-pixel alpha at input resolution ``[*batch_shape, T_input, H, W]``,
    sliced per AR chunk inside ``generate`` and fed to the
    opacity-weighted latent mix."""

    camera_rays: Tensor | None = None
    """Plucker rays at input resolution
    ``[*batch_shape, T_input, H, W, 6]`` (or already at the latent rate;
    :func:`patchify_camera_rays` handles both)."""

    w2cs: Tensor | None = None
    """Target camera world-to-camera matrices ``[*batch_shape, T_lat, 4, 4]``
    indexed by latent-frame."""

    Ks: Tensor | None = None
    """Target camera intrinsics ``[*batch_shape, T_lat, 3, 3]``
    indexed by latent-frame."""

    neighbor_w2cs: Tensor | None = None
    """Optional neighbor camera w2cs ``[*batch_shape, N_neighbors, 4, 4]``."""

    neighbor_Ks: Tensor | None = None
    """Optional neighbor camera intrinsics ``[*batch_shape, N_neighbors, 3, 3]``."""


@dataclass(kw_only=True)
class ArtifixerInferencePipelineConfig(WanInferencePipelineConfig):
    """Config for the ArtiFixer inference pipeline."""

    _target: type["ArtifixerInferencePipeline"] = field(
        default_factory=lambda: ArtifixerInferencePipeline
    )


class ArtifixerInferencePipeline(WanInferencePipeline):
    """ArtiFixer inference pipeline.

    Subclasses :class:`WanInferencePipeline` to thread the per-rollout
    conditioning (opacity, camera rays, target cameras, neighbor latent +
    cameras) into the transformer and its per-block cross-attention
    modules.
    """

    @torch.no_grad()
    def initialize_cache(  # type: ignore[override]
        self,
        text: list[str],
        *,
        condition_latent: Tensor,
        opacity: Tensor,
        camera_rays: Tensor,
        w2cs: Tensor,
        Ks: Tensor,
        neighbor_latent: Tensor | None = None,
        neighbor_w2cs: Tensor | None = None,
        neighbor_Ks: Tensor | None = None,
        height: int,
        width: int,
    ) -> ArtifixerInferencePipelineCache:
        """Build a per-rollout cache and push static state into the network.

        Args:
            text: One prompt per batch element.
            condition_latent: VAE-encoded reconstruction-rendered RGB,
                ``[B, in_dim, T_lat, Hl, Wl]`` -- the caller (Phase 4
                driver) handles VAE encoding so this pipeline does not
                need its own encoder.
            opacity: Per-pixel alpha at input resolution
                ``[B, T_input, H, W]``.
            camera_rays: Plucker rays
                ``[B, T_lat or T_input, H, W, 6]``.
            w2cs: Target camera w2c ``[B, T_lat, 4, 4]``.
            Ks: Target camera intrinsics ``[B, T_lat, 3, 3]``.
            neighbor_latent: Optional VAE-encoded neighbor frames,
                ``[B, in_dim, T_n_lat, H_n_lat, W_n_lat]``. When ``None``
                the neighbor cross-attention branch is disabled.
            neighbor_w2cs: Required iff ``neighbor_latent`` is set.
            neighbor_Ks: Required iff ``neighbor_latent`` is set.
            height: Pre-patchify latent height (post-VAE), matching
                ``condition_latent.shape[-2]``.
            width: Pre-patchify latent width (post-VAE).
        """
        assert condition_latent.shape[-2] == height, (
            f"condition_latent height {condition_latent.shape[-2]} != "
            f"argument height {height}"
        )
        assert condition_latent.shape[-1] == width, (
            f"condition_latent width {condition_latent.shape[-1]} != "
            f"argument width {width}"
        )

        # 1. Base text / image cross-attn cache (image=None for ArtiFixer).
        base_cache = super().initialize_cache(text=text, height=height, width=width)

        # 2. Neighbor wiring. We patchify the neighbor latent through the
        #    network's ``patch_embedding`` so the cross-attention's
        #    ``add_k_proj`` sees per-token features in the transformer
        #    hidden dim, then push the result into every block's cross-attn
        #    neighbor cache.
        transformer = self.diffusion_model.transformer
        assert isinstance(transformer, ArtifixerWanTransformer), (
            f"ArtifixerInferencePipeline requires an ArtifixerWanTransformer "
            f"diffusion_model.transformer, got {type(transformer).__name__}"
        )

        if neighbor_latent is not None:
            assert neighbor_w2cs is not None and neighbor_Ks is not None, (
                "neighbor_w2cs and neighbor_Ks are required when "
                "neighbor_latent is provided"
            )
            neighbor_context = self._encode_neighbor_context(neighbor_latent)
            transformer.network.initialize_neighbor_kv_caches(neighbor_context)
        else:
            assert neighbor_w2cs is None and neighbor_Ks is None, (
                "neighbor_w2cs / neighbor_Ks must be None when "
                "neighbor_latent is None"
            )
            transformer.network.initialize_neighbor_kv_caches(None)

        # 3. PRoPE coefficient setup. ``update_coeffs`` builds the 2D
        #    cos/sin tables for the patches_x x patches_y latent grid;
        #    ``_precompute_and_cache_apply_fns`` on the target side bakes
        #    the static neighbor cameras into the apply fn callables.
        kt, kh, kw = transformer.config.network.patch_size
        patches_x = width // kw
        patches_y = height // kh
        device = transformer.device
        transformer.prope_cross_attn_src.update_coeffs(
            patches_x=patches_x, patches_y=patches_y, device=device
        )
        transformer.prope_cross_attn_tgt.update_coeffs(
            patches_x=patches_x, patches_y=patches_y, device=device
        )
        if neighbor_w2cs is not None:
            transformer.prope_cross_attn_tgt._precompute_and_cache_apply_fns(
                neighbor_w2cs, neighbor_Ks
            )

        return ArtifixerInferencePipelineCache(
            transformer_cache=base_cache.transformer_cache,
            encoder_cache=base_cache.encoder_cache,
            decoder_cache=base_cache.decoder_cache,
            image=base_cache.image,
            condition_latent=condition_latent,
            opacity=opacity,
            camera_rays=camera_rays,
            w2cs=w2cs,
            Ks=Ks,
            neighbor_w2cs=neighbor_w2cs,
            neighbor_Ks=neighbor_Ks,
        )

    def _encode_neighbor_context(self, neighbor_latent: Tensor) -> Tensor:
        """Run ``patch_embedding`` over neighbor latents and flatten to tokens.

        Mirrors dreamfix ``ArtifixerTransformer.forward`` L361-L362::

            neighbor_hidden_states = self.patch_embedding(neighbor_hidden_states)
            neighbor_hidden_states = neighbor_hidden_states.flatten(2).transpose(1, 2)

        Returns ``[*batch_shape, L_neighbor, dim]``.
        """
        transformer = self.diffusion_model.transformer
        assert isinstance(transformer, ArtifixerWanTransformer)
        network = transformer.network
        patched = network.patch_embedding(neighbor_latent)
        return patched.flatten(2).transpose(1, 2)


__all__ = [
    "ArtifixerInferencePipeline",
    "ArtifixerInferencePipelineCache",
    "ArtifixerInferencePipelineConfig",
]
