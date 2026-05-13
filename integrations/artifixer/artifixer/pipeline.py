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

``initialize_cache``:

  - VAE-encoded condition latent and neighbor latent arrive pre-computed
    from the caller so this pipeline does not need its own VAE encoder.
  - Runs the base ``WanInferencePipeline.initialize_cache`` (text
    encoder + text K/V build) with ``image=None``.
  - Projects the neighbor latent through ``patch_embedding`` to get the
    per-token neighbor context, then pushes it into every block's
    :class:`ArtifixerCrossAttention.neighbor_kv_cache` via
    ``ArtifixerDiTNetwork.initialize_neighbor_kv_caches``.
  - Updates the patches_x/patches_y RoPE coefficients on both PRoPE
    modules and precomputes the neighbor-side ``apply_fns`` (the
    neighbor cameras are static across AR steps).
  - Stashes the full-rollout opacity / camera_rays / w2cs / Ks on the
    cache for the per-AR ``generate`` slicing.

``generate`` runs per-AR-chunk PRoPE-src precompute, opacity-weighted
latent mix, manual denoise loop with prepare_latents renoise, and decode.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from artifixer.latent_mix import opacity_weighted_latent_mix
from artifixer.network.patches import patchify_camera_rays, patchify_opacity
from artifixer.transformer import ArtifixerCtrl, ArtifixerWanTransformer
from torch import Tensor

from flashdreams.infra.diffusion.model import DiffusionModel
from flashdreams.infra.diffusion.scheduler.fm import FlowMatchScheduler
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
        text: list[str] | None = None,
        *,
        text_embeddings: Tensor | None = None,
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
            text: One prompt per batch element. Mutually exclusive with
                ``text_embeddings``: pass either the raw strings (we run
                UMT5 internally) or the pre-encoded UMT5 embeddings (the
                external driver may pass them directly).
            text_embeddings: Pre-encoded UMT5 prompt embeddings
                ``[B, L, D]``. Skips the in-pipeline UMT5 forward when set.
            condition_latent: VAE-encoded reconstruction-rendered RGB,
                ``[B, in_dim, T_lat, Hl, Wl]`` -- the caller handles VAE
                encoding so this pipeline does not need its own encoder.
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
        assert (text is None) ^ (text_embeddings is None), (
            "Pass exactly one of ``text`` (raw prompts) or "
            "``text_embeddings`` (pre-encoded UMT5 output)."
        )

        # 1. Base text / image cross-attn cache (image=None for ArtiFixer).
        if text is not None:
            base_cache = super().initialize_cache(
                text=text, height=height, width=width
            )
        else:
            assert text_embeddings is not None
            base_cache = self._initialize_cache_from_text_embeddings(
                text_embeddings=text_embeddings, height=height, width=width
            )

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

    def _initialize_cache_from_text_embeddings(
        self,
        *,
        text_embeddings: Tensor,
        height: int,
        width: int,
    ) -> WanInferencePipelineCache:
        """Build the base cache when prompts are pre-encoded.

        Mirrors :meth:`WanInferencePipeline.initialize_cache` minus the UMT5
        forward / negative-prompt encoding / I2V-image plumbing. CFG is
        disabled per the ArtiFixer reference's kv-cache pipeline contract
        (``negative_prompt`` is ignored), so we don't build
        ``negative_text_embeddings`` even if ``guidance_scale > 1``.
        """
        parent_cache = super(
            WanInferencePipeline, self
        ).initialize_cache(
            transformer_context={
                "height": height,
                "width": width,
                "text_embeddings": text_embeddings,
                "negative_text_embeddings": None,
                "image_embeddings": None,
            },
        )
        return WanInferencePipelineCache(
            transformer_cache=parent_cache.transformer_cache,
            encoder_cache=parent_cache.encoder_cache,
            decoder_cache=parent_cache.decoder_cache,
            image=None,
        )

    def _encode_neighbor_context(self, neighbor_latent: Tensor) -> Tensor:
        """Run ``patch_embedding`` over neighbor latents and flatten to tokens.

        Mirrors the ArtiFixer reference's neighbor projection::

            neighbor_hidden_states = self.patch_embedding(neighbor_hidden_states)
            neighbor_hidden_states = neighbor_hidden_states.flatten(2).transpose(1, 2)

        Returns ``[*batch_shape, L_neighbor, dim]``.
        """
        transformer = self.diffusion_model.transformer
        assert isinstance(transformer, ArtifixerWanTransformer)
        network = transformer.network
        patched = network.patch_embedding(neighbor_latent)
        return patched.flatten(2).transpose(1, 2)

    def _chunk_frame_ranges(
        self, autoregressive_index: int, len_t: int, vae_t: int
    ) -> tuple[int, int, int, int]:
        """Return ``(lat_start, lat_end, input_start, input_end)`` for this AR chunk.

        Mirrors the per-chunk bookkeeping of the ArtiFixer reference kv-cache
        pipeline:

          - first chunk covers ``1 + vae_t * (len_t - 1)`` input frames
            (the Wan VAE 1+4 layout has one latent frame at the boundary
            covering one input frame);
          - subsequent chunks cover ``vae_t * len_t`` input frames each.
        """
        lat_start = autoregressive_index * len_t
        lat_end = lat_start + len_t
        first_chunk_inputs = 1 + vae_t * (len_t - 1)
        if autoregressive_index == 0:
            input_start = 0
            input_end = first_chunk_inputs
        else:
            input_start = first_chunk_inputs + (autoregressive_index - 1) * vae_t * len_t
            input_end = input_start + vae_t * len_t
        return lat_start, lat_end, input_start, input_end

    def _build_ctrl(
        self,
        *,
        chunk_opacity: Tensor,
        chunk_camera_rays: Tensor,
        chunk_post_patch_t: int,
        frame_offset: int,
        vae_t: int,
        vae_s: int,
        patch_size: tuple[int, int, int],
    ) -> ArtifixerCtrl:
        """Patchify the per-chunk opacity / camera_rays for ``ArtifixerCtrl``."""
        return ArtifixerCtrl(
            opacity_extra=patchify_opacity(
                chunk_opacity,
                vae_scale_factor_temporal=vae_t,
                vae_scale_factor_spatial=vae_s,
                patch_size=patch_size,
                frame_offset=frame_offset,
            ),
            camera_extra=patchify_camera_rays(
                chunk_camera_rays,
                hidden_post_patch_t=chunk_post_patch_t,
                vae_scale_factor_temporal=vae_t,
                vae_scale_factor_spatial=vae_s,
                patch_size=patch_size,
                frame_offset=frame_offset,
            ),
            ignore_neighbors=False,
        )

    @torch.no_grad()
    def generate(  # type: ignore[override]
        self,
        autoregressive_index: int,
        cache: ArtifixerInferencePipelineCache,
    ) -> Tensor:
        """Generate one decoded video chunk for AR step ``autoregressive_index``.

        Custom denoise loop (bypasses ``DiffusionModel.generate``) so we can
        renoise each step toward a fresh ``opacity_weighted_latent_mix`` of
        the condition + new noise (matches the ArtiFixer reference's
        kv-cache rollout).

        Steps:
          1. Slice the chunk's frame range out of the full-rollout cache.
          2. Update the source-side PRoPE ``apply_fns`` with the chunk's
             target cameras.
          3. Build the ``ArtifixerCtrl`` payload (patchified opacity +
             camera-ray features).
          4. Run the 4-step DMD denoise loop with prepare_latents renoise.
          5. Stash a ``FinalState`` on the cache so the standard
             ``WanInferencePipeline.finalize`` path closes the AR cache.
          6. Decode and return the chunk.
        """
        assert cache.condition_latent is not None, "initialize_cache must be called first"
        assert cache.opacity is not None
        assert cache.camera_rays is not None
        assert cache.w2cs is not None
        assert cache.Ks is not None

        transformer = self.diffusion_model.transformer
        assert isinstance(transformer, ArtifixerWanTransformer)
        scheduler = self.diffusion_model.scheduler
        assert isinstance(scheduler, FlowMatchScheduler), (
            f"ArtiFixer pipeline requires a FlowMatchScheduler, got "
            f"{type(scheduler).__name__}"
        )

        tcfg = transformer.config
        len_t = tcfg.len_t
        kt, kh, kw = tcfg.network.patch_size
        # Wan VAE scale factors. The decoder owns the public name.
        vae_t = self.decoder.temporal_compression_ratio if self.decoder is not None else 4
        vae_s = self.decoder.spatial_compression_ratio if self.decoder is not None else 8

        lat_start, lat_end, input_start, input_end = self._chunk_frame_ranges(
            autoregressive_index, len_t, vae_t
        )
        chunk_condition = cache.condition_latent[..., lat_start:lat_end, :, :]
        chunk_opacity = cache.opacity[..., input_start:input_end, :, :]
        chunk_w2cs = cache.w2cs[..., lat_start:lat_end, :, :]
        chunk_Ks = cache.Ks[..., lat_start:lat_end, :, :]
        # camera_rays may be at the latent rate or input rate; slice the
        # axis we're indexing on with the latent rate first, then fall back.
        if cache.camera_rays.shape[1] == cache.condition_latent.shape[2]:
            chunk_camera_rays = cache.camera_rays[:, lat_start:lat_end]
        else:
            chunk_camera_rays = cache.camera_rays[:, input_start:input_end]

        # Update the per-AR-chunk PRoPE source cameras (neighbor side was
        # set once at initialize_cache; it is static across AR steps).
        transformer.prope_cross_attn_src._precompute_and_cache_apply_fns(
            chunk_w2cs, chunk_Ks
        )

        # Build the ArtifixerCtrl payload (patchified opacity / camera).
        chunk_post_patch_t = (lat_end - lat_start) // kt
        ctrl = self._build_ctrl(
            chunk_opacity=chunk_opacity,
            chunk_camera_rays=chunk_camera_rays,
            chunk_post_patch_t=chunk_post_patch_t,
            frame_offset=lat_start,
            vae_t=vae_t,
            vae_s=vae_s,
            patch_size=tcfg.network.patch_size,
        )

        # Start the AR cache for this chunk (mirrors DiffusionModel.generate).
        cache.transformer_cache.start(autoregressive_index)

        # Initial mix: condition * opacity_lat + noise * (1 - opacity_lat).
        # We work in unpatchified latent space, then patchify before
        # feeding into ``predict_flow``.
        initial_noise = torch.randn(
            chunk_condition.shape,
            device=self.device,
            dtype=self.diffusion_model.dtype,
            generator=self.diffusion_model.rng,
        )
        is_first = autoregressive_index == 0
        latent_unpatched = opacity_weighted_latent_mix(
            condition=chunk_condition,
            opacity=chunk_opacity,
            noise=initial_noise,
            vae_scale_factor_temporal=vae_t,
            vae_scale_factor_spatial=vae_s,
            is_first_chunk=is_first,
        )
        # ``patchify_and_maybe_split_cp`` expects ``(B, T, C, H, W)`` (einops
        # pattern ``... (t kt) c (h kh) (w kw)``), but a diffusers-convention
        # VAE encoder emits ``(B, C, T, H, W)`` and we keep that convention
        # for ``cache.condition_latent`` / ``latent_unpatched``. Permute
        # before patchify.
        latent = transformer.patchify_and_maybe_split_cp(
            latent_unpatched.permute(0, 2, 1, 3, 4)
        )

        # 4-step DMD denoise with prepare_latents renoise. Matches the
        # ArtiFixer reference: at every non-exit step we replace the
        # next-iteration ``noisy`` with a fresh
        # ``opacity_weighted_latent_mix(...)`` rather than the base
        # FlowMatchScheduler.sample's ``clean``.
        sigmas = scheduler.denoising_sigmas
        timesteps = scheduler.denoising_step_list
        n_steps = timesteps.shape[0]
        input_dtype = latent.dtype
        clean: Tensor | None = None

        for i in range(n_steps):
            sigma = sigmas[i]
            timestep = timesteps[i].to(dtype=input_dtype)
            flow = transformer.predict_flow(
                noisy_latent=latent,
                timestep=timestep,
                cache=cache.transformer_cache,
                input=ctrl,
            )
            clean = latent - sigma * flow
            if i + 1 < n_steps:
                sigma_next = sigmas[i + 1]
                fresh_noise = torch.randn(
                    chunk_condition.shape,
                    device=self.device,
                    dtype=self.diffusion_model.dtype,
                    generator=self.diffusion_model.rng,
                )
                fresh_mix_unpatched = opacity_weighted_latent_mix(
                    condition=chunk_condition,
                    opacity=chunk_opacity,
                    noise=fresh_noise,
                    vae_scale_factor_temporal=vae_t,
                    vae_scale_factor_spatial=vae_s,
                    is_first_chunk=is_first,
                )
                # Same permute as the initial mix (see above): (B, C, T, H, W)
                # -> (B, T, C, H, W) before patchify.
                fresh_mix = transformer.patchify_and_maybe_split_cp(
                    fresh_mix_unpatched.permute(0, 2, 1, 3, 4)
                )
                latent = ((1.0 - sigma_next) * clean + sigma_next * fresh_mix).to(
                    input_dtype
                )

        assert clean is not None, "denoise loop produced no clean estimate"
        clean = transformer.postprocess_clean_latent(
            clean_latent=clean,
            cache=cache.transformer_cache,
            input=None,  # I2V stamp not used by ArtiFixer
        )

        # Stash a FinalState on the cache so the inherited ``finalize`` path
        # (WanInferencePipeline.finalize -> DiffusionModel.finalize) closes
        # the AR cache.
        final_state = DiffusionModel.FinalState(
            clean_latent=clean,
            autoregressive_index=autoregressive_index,
            cache=cache.transformer_cache,
            input=ctrl,
        )
        cache.final_state = final_state
        cache.autoregressive_index = autoregressive_index

        clean_unpatched = transformer.unpatchify_and_maybe_gather_cp(clean)

        if self.decoder is None:
            return clean_unpatched
        return self.decoder(
            clean_unpatched, autoregressive_index, cache.decoder_cache
        )


__all__ = [
    "ArtifixerInferencePipeline",
    "ArtifixerInferencePipelineCache",
    "ArtifixerInferencePipelineConfig",
]
