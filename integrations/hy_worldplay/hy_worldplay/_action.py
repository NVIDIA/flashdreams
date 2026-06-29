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

"""HY-WorldPlay action, camera, and memory conditioning for the Wan 2.2 TI2V 5B stack."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from flashdreams.recipes.wan.autoencoder.i2v import (
    I2VCtrl,
    I2VCtrlEncoder,
    I2VCtrlEncoderCache,
    WanI2VCtrlEncoderConfig,
)
from flashdreams.recipes.wan.transformer.impl.modules import (
    sinusoidal_embedding_1d,
)
from flashdreams.recipes.wan.transformer.impl.network import (
    Block,
    WanDiTNetwork,
    WanDiTNetworkCache,
    WanDiTNetworkTI2V5BConfig,
)
from flashdreams.recipes.wan.transformer.wan21 import (
    Wan21Transformer,
    Wan21TransformerCache,
    Wan21TransformerConfig,
)

_HY_STABILIZATION_TIMESTEP: int = 14
"""Near-clean AdaLN timestep applied to memory K/V during the memory KV
prefill, keeping memory tokens in the model's trained distribution."""


def _fp32_sequential(seq: nn.Sequential, x: Tensor) -> Tensor:
    """Run ``seq`` (chained ``nn.Linear`` + dtype-stable activations) in fp32.

    Upcasts input and Linear weights to fp32 and returns an fp32 output;
    callers ``.type_as(x)`` at the boundary. Only supports
    ``Linear -> Activation -> Linear`` / ``Activation -> Linear`` layouts;
    layers with dtype-coupled state (e.g. layer norms) need
    :func:`hy_worldplay._camera._fp32_layer_norm`.
    """
    out = x.to(torch.float32)
    for module in seq:
        if isinstance(module, nn.Linear):
            weight = module.weight.to(torch.float32)
            bias = module.bias.to(torch.float32) if module.bias is not None else None
            out = F.linear(out, weight, bias)
        else:
            out = module(out)
    return out


## Per-AR-step control payload


@dataclass(kw_only=True)
class HyWorldPlayCtrl(I2VCtrl):
    """I2V control payload extended with per-AR-step action, camera, and memory slices."""

    action: Tensor | None = None
    """Per-latent-frame action labels for the current AR chunk; shape
    ``[*batch_shape, len_t]``."""

    viewmats: Tensor | None = None
    """Per-latent-frame world-to-camera matrices for the current AR chunk; shape
    ``[*batch_shape, len_t, 4, 4]``. Consumed by the PRoPE self-attention branch."""

    Ks: Tensor | None = None
    """Per-latent-frame intrinsics with cx/cy renormalised to 0.5; shape
    ``[*batch_shape, len_t, 3, 3]``."""

    memory_frame_indices: list[int] | None = None
    """Sorted, deduplicated historical frame indices for the upcoming KV-prefill pass.
    Indexes into the per-rollout :attr:`rollout_viewmats` / :attr:`rollout_Ks` /
    :attr:`rollout_action` buffers (frame-granular, not token-granular). ``None`` on
    the first AR chunk and whenever memory selection is disabled."""

    rollout_viewmats: Tensor | None = None
    """Per-*rollout* world-to-camera matrices for the full trajectory; shape
    ``[*batch_shape, F_total, 4, 4]`` where ``F_total = num_chunk * len_t``. Read by
    :meth:`HyWorldPlayWan21Transformer.prefill_memory_kv_cache` to slice memory frames
    at :attr:`memory_frame_indices`. ``None`` when camera conditioning is disabled."""

    rollout_Ks: Tensor | None = None
    """Per-rollout intrinsics buffer; shape ``[*batch_shape, F_total, 3, 3]``. Bound and
    sliced alongside :attr:`rollout_viewmats`."""

    rollout_action: Tensor | None = None
    """Per-rollout action labels; shape ``[*batch_shape, F_total]``. Sliced at
    :attr:`memory_frame_indices` to feed the AdaLN modulation of the memory prefill.
    ``None`` when action conditioning is disabled."""


## I2V + action encoder


@dataclass(kw_only=True)
class HyWorldPlayWanCtrlEncoderConfig(WanI2VCtrlEncoderConfig):
    """Config for the action / camera-aware I2V control encoder."""

    _target: type = field(default_factory=lambda: HyWorldPlayWanCtrlEncoder)


class HyWorldPlayWanCtrlEncoder(I2VCtrlEncoder):
    """Wan I2V encoder that emits per-AR-step action, camera, and memory slices.

    Callers bind per-rollout sources via :meth:`set_action_labels`,
    :meth:`set_camera_data`, and :meth:`set_memory_config`. Each :meth:`forward`
    call slices the ``[ar_idx * len_t : (ar_idx + 1) * len_t]`` window from
    whichever sources are bound; unbound sources flow through as ``None`` so
    downstream consumers stay opt-in.
    """

    def __init__(self, config: HyWorldPlayWanCtrlEncoderConfig) -> None:
        super().__init__(config)
        self._action_labels: Tensor | None = None
        self._viewmats: Tensor | None = None
        self._intrinsics: Tensor | None = None
        # Bound via ``set_memory_config``; ``None`` disables selection.
        self._memory_config: _MemoryConfig | None = None

    def set_action_labels(self, labels: Tensor) -> None:
        """Bind the per-rollout action labels.

        Args:
            labels: Integer class labels with a trailing axis whose length is
                divisible by the transformer's ``len_t`` so successive AR
                steps see equal-sized slices.
        """
        if labels.ndim < 1:
            raise ValueError(
                f"action labels must have at least 1 dim, got shape "
                f"{tuple(labels.shape)}."
            )
        self._action_labels = labels

    def clear_action_labels(self) -> None:
        """Drop the per-rollout label tensor (used when reusing the encoder)."""
        self._action_labels = None

    def set_camera_data(self, viewmats: Tensor, Ks: Tensor) -> None:
        """Bind the per-rollout camera extrinsics and intrinsics.

        Args:
            viewmats: Per-latent-frame world-to-camera matrices, shape
                ``[*, n_latents, 4, 4]`` where ``n_latents`` is divisible by
                the transformer's ``len_t``.
            Ks: Per-latent-frame intrinsics with cx/cy renormalised to 0.5,
                shape ``[*, n_latents, 3, 3]``. Shares leading axes with
                ``viewmats``.
        """
        if viewmats.ndim < 3 or viewmats.shape[-2:] != (4, 4):
            raise ValueError(
                f"viewmats must have trailing shape (n_latents, 4, 4); "
                f"got {tuple(viewmats.shape)}."
            )
        if Ks.ndim < 3 or Ks.shape[-2:] != (3, 3):
            raise ValueError(
                f"Ks must have trailing shape (n_latents, 3, 3); got {tuple(Ks.shape)}."
            )
        if viewmats.shape[:-2] != Ks.shape[:-2]:
            raise ValueError(
                f"viewmats and Ks must share the leading dims preceding "
                f"the matrix axes; got viewmats={tuple(viewmats.shape)}, "
                f"Ks={tuple(Ks.shape)}."
            )
        self._viewmats = viewmats
        self._intrinsics = Ks

    def clear_camera_data(self) -> None:
        """Drop the per-rollout camera tensors (used when reusing the encoder)."""
        self._viewmats = None
        self._intrinsics = None

    def set_memory_config(
        self,
        *,
        points_local: Tensor,
        context_window_length: int,
        memory_frames: int,
        temporal_context_size: int,
        pred_latent_size: int,
        fov_h_deg: float,
        fov_v_deg: float,
        device: torch.device | str | None = None,
    ) -> None:
        """Arm reconstituted-context memory selection for this rollout.

        Stashes the Monte-Carlo point cloud and selection knobs so each
        :meth:`forward` call can compute ``memory_frame_indices`` from the
        bound camera history. AR steps with
        ``current_frame_idx < context_window_length`` bypass selection and
        emit ``memory_frame_indices=None``.

        Args:
            points_local: Pre-sampled cloud of 3D points, shape ``[N, 3]``.
                Build once via
                :func:`hy_worldplay._memory.generate_points_in_sphere` and
                reuse for the whole rollout.
            context_window_length: Frame-count threshold below which
                FOV-overlap selection is skipped (vendor default 16).
            memory_frames: Total budget of memory frames the selector
                returns once armed.
            temporal_context_size: Recent-frames portion of the memory
                budget (kept unconditionally).
            pred_latent_size: Length of the query clip the selector scores
                historical clips against.
            fov_h_deg: Horizontal FOV (degrees) for the overlap math.
            fov_v_deg: Vertical FOV (degrees) for the overlap math.
            device: Optional torch device for the overlap math; pass the
                runner's compute device for GPU rollouts.
        """
        if memory_frames < temporal_context_size:
            raise ValueError(
                f"memory_frames ({memory_frames}) must be >= "
                f"temporal_context_size ({temporal_context_size})."
            )
        if points_local.ndim != 2 or points_local.shape[-1] != 3:
            raise ValueError(
                f"points_local must have shape (N, 3); got {tuple(points_local.shape)}."
            )
        self._memory_config = _MemoryConfig(
            points_local=points_local,
            context_window_length=context_window_length,
            memory_frames=memory_frames,
            temporal_context_size=temporal_context_size,
            pred_latent_size=pred_latent_size,
            fov_h_deg=fov_h_deg,
            fov_v_deg=fov_v_deg,
            device=device,
        )

    def clear_memory_config(self) -> None:
        """Disarm memory selection (used when reusing the encoder)."""
        self._memory_config = None

    def initialize_autoregressive_cache(self) -> I2VCtrlEncoderCache:
        # Bound action / camera / memory state is owned by the runner across
        # rollouts and is intentionally not cleared here.
        return super().initialize_autoregressive_cache()

    @torch.no_grad()
    def forward(
        self,
        input: Tensor,
        autoregressive_index: int = 0,
        cache: I2VCtrlEncoderCache | None = None,
    ) -> HyWorldPlayCtrl:
        base = super().forward(
            input=input, autoregressive_index=autoregressive_index, cache=cache
        )
        len_t = base.latent.shape[-4]
        start = autoregressive_index * len_t
        end = start + len_t
        device = base.latent.device

        action_chunk: Tensor | None = None
        if self._action_labels is not None:
            if end > self._action_labels.shape[-1]:
                raise ValueError(
                    f"action labels exhausted at AR step {autoregressive_index}: "
                    f"need {end} entries but only "
                    f"{self._action_labels.shape[-1]} provided."
                )
            action_chunk = self._action_labels[..., start:end].to(device=device)

        viewmats_chunk: Tensor | None = None
        Ks_chunk: Tensor | None = None
        if self._viewmats is not None:
            assert self._intrinsics is not None, (
                "viewmats and Ks must be bound together via set_camera_data; "
                "found viewmats without intrinsics."
            )
            total = self._viewmats.shape[-3]
            if end > total:
                raise ValueError(
                    f"camera tensors exhausted at AR step {autoregressive_index}: "
                    f"need {end} frames but only {total} provided."
                )
            viewmats_chunk = self._viewmats[..., start:end, :, :].to(device=device)
            Ks_chunk = self._intrinsics[..., start:end, :, :].to(device=device)

        memory_indices: list[int] | None = self._compute_memory_indices(
            autoregressive_index=autoregressive_index, current_frame_idx=start
        )

        # Expose the full-trajectory tensors (in rollout coordinates) so the
        # prefill driver can index them at ``memory_frame_indices``.
        rollout_viewmats: Tensor | None = (
            self._viewmats.to(device=device) if self._viewmats is not None else None
        )
        rollout_Ks: Tensor | None = (
            self._intrinsics.to(device=device) if self._intrinsics is not None else None
        )
        rollout_action: Tensor | None = (
            self._action_labels.to(device=device)
            if self._action_labels is not None
            else None
        )

        return HyWorldPlayCtrl(
            latent=base.latent,
            mask=base.mask,
            action=action_chunk,
            viewmats=viewmats_chunk,
            Ks=Ks_chunk,
            memory_frame_indices=memory_indices,
            rollout_viewmats=rollout_viewmats,
            rollout_Ks=rollout_Ks,
            rollout_action=rollout_action,
        )

    def _compute_memory_indices(
        self, *, autoregressive_index: int, current_frame_idx: int
    ) -> list[int] | None:
        """Pick the historical frame indices for this AR step's KV prefill.

        Returns:
            ``None`` on AR step 0 or when camera data is not bound (no
            prefill to run). When memory selection is armed and past the
            warm-up window (``current_frame_idx >= context_window_length``),
            the FOV-overlap selector's choice. Otherwise the all-history
            fall-back ``list(range(0, current_frame_idx))``, required
            because the HY path wipes each block's rolling self-attention
            cache at chunk boundaries.
        """
        if autoregressive_index == 0 or current_frame_idx == 0:
            return None
        if self._viewmats is None:
            return None
        if (
            self._memory_config is not None
            and current_frame_idx >= self._memory_config.context_window_length
        ):
            cfg = self._memory_config
            # The FOV selector takes a flat ``[n_latents, 4, 4]`` history;
            # collapse leading batch axes by selecting slot 0.
            viewmats_history = self._viewmats
            while viewmats_history.ndim > 3:
                viewmats_history = viewmats_history[0]
            # Lazy import so numpy + FOV math stay out of the import graph
            # when memory selection is disabled.
            from hy_worldplay._memory import select_memory_frame_indices

            # ``.numpy()`` rejects bf16; cast to fp32.
            return select_memory_frame_indices(
                viewmats_history.detach().to(dtype=torch.float32).cpu().numpy(),
                current_frame_idx=current_frame_idx,
                points_local=cfg.points_local,
                memory_frames=cfg.memory_frames,
                temporal_context_size=cfg.temporal_context_size,
                pred_latent_size=cfg.pred_latent_size,
                fov_h_deg=cfg.fov_h_deg,
                fov_v_deg=cfg.fov_v_deg,
                device=cfg.device,
            )

        # All-history fall-back; see docstring.
        return list(range(0, current_frame_idx))


@dataclass(frozen=True)
class _MemoryConfig:
    """Memory-selection knobs bound on the encoder."""

    points_local: Tensor
    context_window_length: int
    memory_frames: int
    temporal_context_size: int
    pred_latent_size: int
    fov_h_deg: float
    fov_v_deg: float
    device: torch.device | str | None


## Action-aware DiT network


@dataclass
class HyWorldPlayWanDiTNetworkConfig(WanDiTNetworkTI2V5BConfig):
    """Config for the action / camera-aware Wan 2.2 TI2V 5B DiT."""

    _target: type = field(default_factory=lambda: HyWorldPlayWanDiTNetwork)

    use_prope_blocks: bool = False
    """Build dual-branch RoPE + PRoPE blocks instead of the standard
    :class:`Block`. When ``True`` the encoder must bind per-rollout camera
    data so each AR step's :class:`HyWorldPlayCtrl` carries ``viewmats`` and
    ``Ks`` slices. Defaults to ``False`` so action-only configurations keep
    the standard block stack."""


class HyWorldPlayWanDiTNetwork(WanDiTNetwork):
    """Wan DiT with action-modulated AdaLN and optional PRoPE self-attention blocks.

    Two extensions on top of :class:`WanDiTNetwork`:

    * ``action_embedding`` MLP (same shape as ``time_embedding``) consumes
      sinusoidally-encoded action class labels and produces a per-latent-frame
      additive term summed into the time embedding before ``time_projection``.
      ``linear_2`` is zero-initialised so the conditioner is an identity at init.
    * When :attr:`HyWorldPlayWanDiTNetworkConfig.use_prope_blocks` is set,
      :meth:`_build_block` returns :class:`HyWorldPlayPRoPEBlock` instances and
      :meth:`forward` threads ``viewmats`` / ``Ks`` through
      ``block_extra_kwargs``. ``o_prope`` is zero-init so the dual-branch path
      is also an identity until HY-WorldPlay weights are loaded.
    """

    def __init__(self, config: HyWorldPlayWanDiTNetworkConfig) -> None:
        # Stash ``use_prope_blocks`` before ``super().__init__()`` because the
        # base constructor calls ``self._build_block`` while wiring the stack.
        nn.Module.__init__(self)
        self._hy_use_prope_blocks = config.use_prope_blocks
        super().__init__(config)
        self.action_embedding = nn.Sequential(
            nn.Linear(self.freq_dim, self.dim),
            nn.SiLU(),
            nn.Linear(self.dim, self.dim),
        )
        # Zero-init the residual head so the action branch is an identity at
        # construction time.
        zero_linear = self.action_embedding[-1]
        assert isinstance(zero_linear, nn.Linear)
        nn.init.zeros_(zero_linear.weight)
        if zero_linear.bias is not None:
            nn.init.zeros_(zero_linear.bias)

    def _build_block(self, layer_idx: int) -> Block:
        if self._hy_use_prope_blocks:
            # Lazy import so action-only configs don't pay the PRoPE block's
            # import cost.
            from hy_worldplay._camera import HyWorldPlayPRoPEBlock

            return HyWorldPlayPRoPEBlock(
                dim=self.dim,
                ffn_dim=self.ffn_dim,
                num_heads=self.num_heads,
                cross_attn_norm=self.cross_attn_norm,
                eps=self.eps,
                i2v=self.cross_attn_enable_img,
                apply_rope_before_kvcache=self.apply_rope_before_kvcache,
            )
        return super()._build_block(layer_idx)

    def forward(
        self,
        x: Tensor,
        timesteps: Tensor,
        cache: WanDiTNetworkCache,
        rope_freqs: Tensor,
        current_chunk_idx: int = 0,
        eager_mode: bool = True,
        block_extra_kwargs: dict[str, Any] = {},
        action: Tensor | None = None,
        viewmats: Tensor | None = None,
        Ks: Tensor | None = None,
    ) -> Tensor:
        """Run the DiT forward with optional action conditioning and PRoPE routing.

        Adds the action embedding to the time embedding before the modulation
        projection, then threads ``viewmats`` / ``Ks`` through
        ``block_extra_kwargs`` when PRoPE blocks are active. With both
        ``action`` and ``viewmats`` ``None`` the modulation path is identical
        to :meth:`WanDiTNetwork.forward`.

        Raises:
            ValueError: ``use_prope_blocks`` is set but ``viewmats`` is
                ``None``.
        """
        assert self._parameters_updated_after_loading_checkpoint, (
            "We expect to have called update_parameters_after_loading_checkpoint() "
            "after loading the checkpoint"
        )
        batch_shape = x.shape[:-2]
        L = x.shape[-2]

        if self.patch_embedding_type == "linear":
            x = self.patch_embedding(x)
        elif self.patch_embedding_type == "conv3d":
            _weight = self.patch_embedding.weight.reshape(self.dim, -1)
            _bias = self.patch_embedding.bias
            x = torch.nn.functional.linear(x, _weight, _bias)
        else:
            raise ValueError(
                f"Invalid patch embedding type: {self.patch_embedding_type}"
            )

        per_token_timestep = (
            timesteps.ndim > len(batch_shape) and timesteps.shape[-1] == L
        )
        # ``time_embedding`` runs in fp32; ``time_projection`` stays in bf16.
        e_fp32 = _fp32_sequential(
            self.time_embedding,
            sinusoidal_embedding_1d(self.freq_dim, timesteps).to(torch.float32),
        )
        e = e_fp32.type_as(x)

        if action is not None:
            action_e = self._compute_action_embedding(action=action, x=x, L=L)
            # Add in bf16 (``e`` was cast to ``x.dtype`` by ``type_as`` above).
            e = e + action_e
            per_token_timestep = True

        e0 = self.time_projection(e).unflatten(-1, (6, self.dim))

        if per_token_timestep:
            block_e_shape = batch_shape + (L, 6, self.dim)
            head_e = torch.broadcast_to(e, batch_shape + (L, self.dim)).unsqueeze(-2)
        else:
            block_e_shape = batch_shape + (6, self.dim)
            head_e = torch.broadcast_to(e, batch_shape + (self.dim,)).unsqueeze(-2)
        block_e = torch.broadcast_to(e0, block_e_shape)

        # Thread camera data per-block when PRoPE blocks are active; copy
        # ``block_extra_kwargs`` rather than mutate the caller's dict.
        block_kwargs = dict(block_extra_kwargs)
        if self._hy_use_prope_blocks:
            if viewmats is None:
                raise ValueError(
                    "use_prope_blocks=True requires viewmats; "
                    "the encoder must bind camera data via "
                    "set_camera_data so HyWorldPlayCtrl.viewmats is populated."
                )
            block_kwargs["viewmats"] = viewmats
            block_kwargs["Ks"] = Ks

        from hy_worldplay import _debug_dump

        if eager_mode:
            cache.before_update(current_chunk_idx)
        for block_idx, block in enumerate(self.blocks):
            assert isinstance(block, Block)
            _debug_dump.set_context(phase="forward", block_idx=block_idx)
            x = block(
                x=x,
                e=block_e,
                rope_freqs=rope_freqs,
                cache=cache[block_idx],
                **block_kwargs,
            )
        _debug_dump.clear_context("phase", "block_idx")
        if eager_mode:
            cache.after_update(current_chunk_idx)

        x = self.head(x, head_e)
        return x

    def prefill_memory_kv_cache(
        self,
        x: Tensor,
        timesteps: Tensor,
        cache: WanDiTNetworkCache,
        rope_freqs: Tensor,
        block_extra_kwargs: dict[str, Any] | None = None,
        action: Tensor | None = None,
        viewmats: Tensor | None = None,
        Ks: Tensor | None = None,
    ) -> None:
        """Populate each block's memory KV cache.

        Runs :meth:`forward`'s patchify + time / action embedding + AdaLN
        preamble, then calls :meth:`HyWorldPlayPRoPEBlock.prefill_memory_kv`
        per block so each block's self-attention K/V land in its memory slot
        at the collapsed RoPE positions ``[0, K * tokens_per_frame)``.
        Cross-attention, FFN, and the head are skipped (not cached).

        The caller slices the per-rollout history at
        ``HyWorldPlayCtrl.memory_frame_indices`` and builds ``rope_freqs``
        against the same collapsed positions.

        Args:
            x: Patchified memory latents, shape ``[..., L_mem, in_dim]``.
            timesteps: Scalar broadcast or per-token clean-context timestep,
                applied so the AdaLN modulation stays in the trained
                distribution.
            cache: Per-block cache; only the ``memory`` slots are written.
            rope_freqs: RoPE frequencies for the collapsed memory positions
                ``[0, L_mem)``.
            block_extra_kwargs: Unused; kept for symmetry with :meth:`forward`.
            action: Optional action labels for the memory frames.
            viewmats: W2C extrinsics for the memory frames. Required when
                ``use_prope_blocks=True``.
            Ks: Per-frame intrinsics for the memory frames.

        Raises:
            RuntimeError: ``use_prope_blocks`` is not set.
            ValueError: ``viewmats`` is ``None``.
        """
        assert self._parameters_updated_after_loading_checkpoint, (
            "We expect to have called update_parameters_after_loading_checkpoint() "
            "after loading the checkpoint"
        )
        if not self._hy_use_prope_blocks:
            raise RuntimeError(
                "HyWorldPlayWanDiTNetwork.prefill_memory_kv_cache requires "
                "use_prope_blocks=True; the prefill executor only meaningfully "
                "writes the dual-branch memory caches owned by HyWorldPlayPRoPEBlock."
            )

        batch_shape = x.shape[:-2]
        L = x.shape[-2]

        if self.patch_embedding_type == "linear":
            x = self.patch_embedding(x)
        elif self.patch_embedding_type == "conv3d":
            _weight = self.patch_embedding.weight.reshape(self.dim, -1)
            _bias = self.patch_embedding.bias
            x = torch.nn.functional.linear(x, _weight, _bias)
        else:
            raise ValueError(
                f"Invalid patch embedding type: {self.patch_embedding_type}"
            )

        # Same per-token timestep dispatch and action injection as forward.
        per_token_timestep = (
            timesteps.ndim > len(batch_shape) and timesteps.shape[-1] == L
        )
        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, timesteps).type_as(x)
        )
        if action is not None:
            action_e = self._compute_action_embedding(action=action, x=x, L=L)
            e = e + action_e
            per_token_timestep = True
        e0 = self.time_projection(e).unflatten(-1, (6, self.dim))
        if per_token_timestep:
            block_e_shape = batch_shape + (L, 6, self.dim)
        else:
            block_e_shape = batch_shape + (6, self.dim)
        block_e = torch.broadcast_to(e0, block_e_shape)

        if viewmats is None:
            raise ValueError(
                "HyWorldPlayWanDiTNetwork.prefill_memory_kv_cache requires "
                "viewmats; the executor must slice the per-rollout viewmats "
                "by selected_frame_indices before calling."
            )

        # No before_update / after_update here: the prefill writes only into
        # cache[block_idx].memory, which the executor resets/writes separately.
        from hy_worldplay import _debug_dump

        for block_idx, block in enumerate(self.blocks):
            block_cache = cache[block_idx]
            from hy_worldplay._camera import (
                HyWorldPlayPRoPEBlock,
                HyWorldPlayPRoPEBlockCache,
            )

            assert isinstance(block, HyWorldPlayPRoPEBlock), (
                f"prefill expects HyWorldPlayPRoPEBlock, got {type(block).__name__}"
            )
            assert isinstance(block_cache, HyWorldPlayPRoPEBlockCache), (
                f"prefill expects HyWorldPlayPRoPEBlockCache, got "
                f"{type(block_cache).__name__}"
            )
            _debug_dump.set_context(phase="prefill", block_idx=block_idx)
            # Run the full block so the hidden state propagates block-to-block;
            # only the per-block ``cache.memory`` side effects are kept.
            x = block.prefill_memory_kv(
                x=x,
                e=block_e,
                rope_freqs=rope_freqs,
                viewmats=viewmats,
                Ks=Ks,
                cache=block_cache,
            )
        _debug_dump.clear_context("phase", "block_idx")

    def _compute_action_embedding(
        self,
        *,
        action: Tensor,
        x: Tensor,
        L: int,
    ) -> Tensor:
        """Lift per-latent-frame action labels to a per-token additive term.

        Sinusoidally encodes the integer labels, runs them through the action
        MLP, then ``repeat_interleave``s across the ``tokens_per_frame`` slots
        of each latent frame on the post-patchify token axis.

        Raises:
            NotImplementedError: ``cp_size > 1``; multi-rank action
                conditioning is not yet supported.
        """
        cp_group = getattr(self, "_cp_group", None)
        if cp_group is not None:
            raise NotImplementedError(
                "HyWorldPlayWanDiTNetwork does not yet support context-parallel "
                "(cp_size > 1) action conditioning."
            )
        n_latent = action.shape[-1]
        if L % n_latent != 0:
            raise ValueError(
                f"action.shape[-1]={n_latent} must divide the post-patchify "
                f"token count L={L}."
            )
        tokens_per_frame = L // n_latent
        action_freq = sinusoidal_embedding_1d(self.freq_dim, action).type_as(x)
        action_e = self.action_embedding(action_freq)
        return action_e.repeat_interleave(tokens_per_frame, dim=-2)


## Action-aware Wan 2.1 transformer


@dataclass(kw_only=True)
class HyWorldPlayWan21TransformerCache(Wan21TransformerCache):
    """Per-rollout cache for the HY-WorldPlay transformer.

    Adds clean-latent-history state and a prefill latch on top of the base
    :class:`Wan21TransformerCache`. :meth:`start` wipes each block's rolling
    self-attention cache at every chunk boundary past the first, so the
    rolling window only ever holds the current chunk's tokens; cross-chunk
    context arrives via the dedicated memory cache.
    """

    clean_latent_history: Tensor | None = None
    """Per-rollout patchified clean-latent history, concatenated along the
    post-patchify token axis (``dim=-2``). ``None`` until the first chunk's
    :meth:`HyWorldPlayWan21Transformer.finalize_kv_cache` call appends to it."""

    finished_chunks: int = 0
    """Count of chunks whose patchified clean latent has been appended to
    :attr:`clean_latent_history`."""

    hy_chunk_size_t: int = 0
    """Pre-patchify temporal chunk size (``len_t``) for the current rollout."""

    hy_tokens_per_frame: int = 0
    """Post-patchify tokens per latent frame, ``(height // kh) * (width // kw)``."""

    prefill_completed_for_chunk: int = -1
    """``autoregressive_index`` of the chunk whose memory KV prefill has already
    run; ``-1`` before the first prefill. One prefill per chunk suffices since
    memory K/V are stable across scheduler steps."""

    def start(self, autoregressive_index: int) -> None:
        # Reset per-block rolling self-attention caches at every chunk boundary
        # past the first; the dedicated memory cache supplies cross-chunk context.
        if autoregressive_index > 0:
            self._reset_per_block_rolling_caches(autoregressive_index)
        self.prefill_completed_for_chunk = -1
        super().start(autoregressive_index)

    def _reset_per_block_rolling_caches(self, autoregressive_index: int) -> None:
        """Wipe each block's ``self_attn`` / ``prope_self_attn`` for the new chunk.

        Sets ``_prev_chunk_idx`` to ``autoregressive_index - 1`` so the
        subsequent ``before_update`` in :meth:`Wan21TransformerCache.start`
        passes its monotonic-chunk-index assertion.
        """
        from hy_worldplay._camera import HyWorldPlayPRoPEBlockCache

        for net_cache in (self.network_cache, self.network_cache_uncond):
            if net_cache is None:
                continue
            for block_cache in net_cache.block_caches:
                if not isinstance(block_cache, HyWorldPlayPRoPEBlockCache):
                    continue
                block_cache.reset_current_chunk()
                block_cache.self_attn._prev_chunk_idx = autoregressive_index - 1
                block_cache.prope_self_attn._prev_chunk_idx = autoregressive_index - 1


@dataclass(kw_only=True)
class HyWorldPlayWan21TransformerConfig(Wan21TransformerConfig):
    """Config for the action / camera / memory-aware Wan 2.1 transformer."""

    _target: type = field(default_factory=lambda: HyWorldPlayWan21Transformer)

    network: HyWorldPlayWanDiTNetworkConfig = field(
        default_factory=HyWorldPlayWanDiTNetworkConfig
    )


class HyWorldPlayWan21Transformer(Wan21Transformer):
    """Wan 2.1 transformer that threads action, camera, and memory through the network.

    Extends :class:`Wan21Transformer` with the memory KV prefill executor and
    the plumbing to keep :class:`HyWorldPlayCtrl`'s extra fields alive across
    the patchify pass. The per-rollout cache accumulates patchified clean
    latents from past chunks; before the first denoising step of each chunk
    past the first, the prefill driver slices that history at
    ``memory_frame_indices`` and seeds each PRoPE block's memory KV cache.
    """

    def initialize_autoregressive_cache(
        self,
        *,
        height: int,
        width: int,
        text_embeddings: Tensor,
        image_embeddings: Tensor | None = None,
        negative_text_embeddings: Tensor | None = None,
        **_unused: Any,
    ) -> HyWorldPlayWan21TransformerCache:
        """Build a :class:`HyWorldPlayWan21TransformerCache` for a new rollout.

        Stamps the per-rollout spatial layout into ``hy_chunk_size_t`` /
        ``hy_tokens_per_frame`` so the prefill executor can map memory frame
        indices to post-patchify token ranges.
        """
        base = super().initialize_autoregressive_cache(
            height=height,
            width=width,
            text_embeddings=text_embeddings,
            image_embeddings=image_embeddings,
            negative_text_embeddings=negative_text_embeddings,
            **_unused,
        )
        cfg = self.config
        kt, kh, kw = cfg.network.patch_size
        # Post-patchify tokens per latent frame.
        tokens_per_frame = (height // kh) * (width // kw)
        return HyWorldPlayWan21TransformerCache(
            network_cache=base.network_cache,
            network_cache_uncond=base.network_cache_uncond,
            rope_adapter=base.rope_adapter,
            rope_freqs=base.rope_freqs,
            autoregressive_index=base.autoregressive_index,
            clean_latent_history=None,
            finished_chunks=0,
            hy_chunk_size_t=cfg.len_t // kt,
            hy_tokens_per_frame=tokens_per_frame,
        )

    def predict_flow(
        self,
        noisy_latent: Tensor,
        timestep: Tensor,
        cache: Wan21TransformerCache,
        input: I2VCtrl | None = None,
        network_extra_kwargs: dict[str, Any] | None = None,
    ) -> Tensor:
        from hy_worldplay import _debug_dump

        ar_idx = (
            cache.autoregressive_index if hasattr(cache, "autoregressive_index") else -1
        )
        is_first_step = (
            isinstance(cache, HyWorldPlayWan21TransformerCache)
            and cache.prefill_completed_for_chunk != ar_idx
        )
        # Bind chunk + step context so per-block dumps carry it.
        _debug_dump.set_context(
            ar_idx=ar_idx,
            is_first_step_of_chunk=is_first_step,
        )
        if _debug_dump.enabled():
            cfg_self = getattr(self, "config", None)
            extra_cfg = {}
            if cfg_self is not None:
                extra_cfg = {
                    "cfg_len_t": getattr(cfg_self, "len_t", None),
                    "cfg_window_size_t": getattr(cfg_self, "window_size_t", None),
                    "cfg_batch_shape": list(getattr(cfg_self, "batch_shape", ())),
                    "cfg_patch_size": list(getattr(cfg_self.network, "patch_size", ())),
                    "_cp_size": getattr(self, "_cp_size", None),
                    "_output_height": getattr(self, "_output_height", None),
                    "_output_width": getattr(self, "_output_width", None),
                }
            _debug_dump.dump(
                "predict_flow.entry",
                None,
                timestep_shape=list(timestep.shape),
                **extra_cfg,
            )
            _debug_dump.dump("predict_flow.noisy_latent", noisy_latent)
            _debug_dump.dump("predict_flow.timestep", timestep)
        network_extra_kwargs = dict(network_extra_kwargs or {})
        # Run the memory prefill once per chunk (latch suppresses re-runs).
        if (
            isinstance(cache, HyWorldPlayWan21TransformerCache)
            and isinstance(input, HyWorldPlayCtrl)
            and input.memory_frame_indices is not None
            and len(input.memory_frame_indices) > 0
            and cache.clean_latent_history is not None
            and cache.prefill_completed_for_chunk != ar_idx
        ):
            self.prefill_memory_kv_cache(cache=cache, input=input, timestep=timestep)
            cache.prefill_completed_for_chunk = ar_idx

        if isinstance(input, HyWorldPlayCtrl):
            if input.action is not None and "action" not in network_extra_kwargs:
                network_extra_kwargs["action"] = input.action
            if input.viewmats is not None and "viewmats" not in network_extra_kwargs:
                network_extra_kwargs["viewmats"] = input.viewmats
            if input.Ks is not None and "Ks" not in network_extra_kwargs:
                network_extra_kwargs["Ks"] = input.Ks
        return super().predict_flow(
            noisy_latent=noisy_latent,
            timestep=timestep,
            cache=cache,
            input=input,
            network_extra_kwargs=network_extra_kwargs,
        )

    def finalize_kv_cache(
        self,
        noisy_latent: Tensor,
        timestep: Tensor,
        cache: Wan21TransformerCache,
        input: Any = None,
    ) -> None:
        """Append the chunk's clean latent to the history and skip the rolling-cache update.

        The base implementation's rolling-window re-run is skipped: the HY
        path wipes that window at every chunk start and provides cross-chunk
        context through the dedicated memory cache.
        """
        if isinstance(cache, HyWorldPlayWan21TransformerCache):
            cache.clean_latent_history = self._append_clean_latent_to_history(
                history=cache.clean_latent_history,
                clean_latent=noisy_latent,
            )
            cache.finished_chunks += 1
            return
        # Defensive fall-through for any non-HY cache.
        super().finalize_kv_cache(
            noisy_latent=noisy_latent,
            timestep=timestep,
            cache=cache,
            input=input,
        )

    def patchify_and_maybe_split_cp(self, x: Any) -> Any:
        if isinstance(x, HyWorldPlayCtrl):
            if x._is_patchified:
                return x
            patched_latent = self.patchify_and_maybe_split_cp(x.latent)
            patched_mask = self.patchify_and_maybe_split_cp(x.mask)
            # action / viewmats / Ks / memory_frame_indices and the rollout_*
            # siblings are per-latent-frame metadata and don't participate in
            # the patchify reshape; pass them through unchanged.
            return HyWorldPlayCtrl(
                latent=patched_latent,
                mask=patched_mask,
                _is_patchified=True,
                action=x.action,
                viewmats=x.viewmats,
                Ks=x.Ks,
                memory_frame_indices=x.memory_frame_indices,
                rollout_viewmats=x.rollout_viewmats,
                rollout_Ks=x.rollout_Ks,
                rollout_action=x.rollout_action,
            )
        return super().patchify_and_maybe_split_cp(x)

    ## Reconstituted-context prefill driver

    def prefill_memory_kv_cache(  # noqa: C901 (debug instrumentation)
        self,
        cache: HyWorldPlayWan21TransformerCache,
        input: HyWorldPlayCtrl,
        timestep: Tensor,
    ) -> None:
        """Drive the memory KV prefill for the current chunk.

        Slices the clean-latent history at ``input.memory_frame_indices``,
        builds collapsed-position RoPE freqs, slices ``viewmats`` / ``Ks`` /
        ``action`` at the same indices, then dispatches into
        :meth:`HyWorldPlayWanDiTNetwork.prefill_memory_kv_cache` for the
        conditional (and unconditional, when CFG is enabled) branch. Each
        block's memory cache is reset before being repopulated.

        Args:
            cache: Active per-rollout cache; ``clean_latent_history`` must
                already contain the memory frames being indexed.
            input: Patchified per-AR-step ctrl payload with non-empty
                ``memory_frame_indices``.
            timestep: Current denoising-step tensor. Used only for its
                ``dtype`` / ``device`` / batch shape; memory positions are
                modulated at :data:`_HY_STABILIZATION_TIMESTEP` instead.
        """
        assert input.memory_frame_indices is not None, (
            "prefill_memory_kv_cache requires non-None memory_frame_indices"
        )
        selected = list(input.memory_frame_indices)
        K = len(selected)
        assert K > 0, "prefill_memory_kv_cache requires at least one memory frame"
        assert cache.clean_latent_history is not None, (
            "prefill_memory_kv_cache requires clean_latent_history; the executor "
            "must run after at least one chunk has finalized."
        )

        tokens_per_frame = cache.hy_tokens_per_frame
        history = cache.clean_latent_history  # [..., total_L, in_dim]
        total_L = history.shape[-2]
        max_frame = total_L // tokens_per_frame
        # Defensive bounds check; the selector should only emit in-range indices.
        for idx in selected:
            assert 0 <= idx < max_frame, (
                f"memory frame index {idx} out of range for history of "
                f"{max_frame} frames ({total_L} tokens / {tokens_per_frame} "
                f"tokens-per-frame)."
            )

        # Slice each selected frame's token range; the history is frame-major
        # along the token axis.
        token_ranges = [
            history[..., idx * tokens_per_frame : (idx + 1) * tokens_per_frame, :]
            for idx in selected
        ]
        memory_x = torch.cat(token_ranges, dim=-2)  # [..., K*TPF, in_dim]

        # Slice the per-rollout camera + action tensors (frame-granular) at the
        # same indices.
        selected_idx_t = torch.as_tensor(
            selected, dtype=torch.long, device=memory_x.device
        )
        memory_viewmats = self._index_rollout_buffer(
            rollout=input.rollout_viewmats,
            per_step=input.viewmats,
            selected=selected_idx_t,
            kind="viewmats",
        )
        memory_Ks = self._index_rollout_buffer(
            rollout=input.rollout_Ks,
            per_step=input.Ks,
            selected=selected_idx_t,
            kind="Ks",
        )
        memory_action = self._index_rollout_buffer(
            rollout=input.rollout_action,
            per_step=input.action,
            selected=selected_idx_t,
            kind="action",
        )

        # RoPE freqs for the collapsed memory positions ``[0, K)`` on the
        # temporal axis.
        rope_freqs = self._build_collapsed_rope_freqs(
            cache=cache,
            t_positions=torch.arange(K, dtype=torch.float32, device=memory_x.device),
        )

        # Clean-context timestep for memory positions; matches ``timestep``'s
        # dtype / device / batch.
        context_timestep = torch.full_like(
            timestep, fill_value=_HY_STABILIZATION_TIMESTEP
        )

        # Env-var-gated debug dump for prefill inputs.
        from hy_worldplay import _debug_dump

        if _debug_dump.enabled():
            _debug_dump.dump(
                "prefill.entry",
                None,
                selected=list(selected),
                K=K,
                tokens_per_frame=tokens_per_frame,
                stabilization_timestep=_HY_STABILIZATION_TIMESTEP,
            )
            _debug_dump.dump("prefill.memory_x", memory_x)
            _debug_dump.dump("prefill.rope_freqs", rope_freqs)
            _debug_dump.dump("prefill.context_timestep", context_timestep)
            _debug_dump.dump("prefill.timestep_input", timestep)
            if memory_viewmats is not None:
                _debug_dump.dump("prefill.memory_viewmats", memory_viewmats)
            if memory_Ks is not None:
                _debug_dump.dump("prefill.memory_Ks", memory_Ks)
            if memory_action is not None:
                _debug_dump.dump("prefill.memory_action", memory_action)

        # Reset each branch's per-block memory cache so a previous chunk's
        # content can't leak in.
        from hy_worldplay._camera import HyWorldPlayPRoPEBlockCache

        for net_cache in (cache.network_cache, cache.network_cache_uncond):
            if net_cache is None:
                continue
            for block_cache in net_cache.block_caches:
                if isinstance(block_cache, HyWorldPlayPRoPEBlockCache):
                    block_cache.memory.reset()

        # Narrow ``self.network`` to the HY-DiT type so the prefill entry
        # point resolves.
        network = self.network
        assert isinstance(network, HyWorldPlayWanDiTNetwork)

        # Conditional pass.
        network.prefill_memory_kv_cache(
            x=memory_x,
            timesteps=context_timestep,
            cache=cache.network_cache,
            rope_freqs=rope_freqs,
            action=memory_action,
            viewmats=memory_viewmats,
            Ks=memory_Ks,
        )
        # Unconditional pass (when CFG is enabled).
        if cache.network_cache_uncond is not None:
            network.prefill_memory_kv_cache(
                x=memory_x,
                timesteps=context_timestep,
                cache=cache.network_cache_uncond,
                rope_freqs=rope_freqs,
                action=memory_action,
                viewmats=memory_viewmats,
                Ks=memory_Ks,
            )

    def _append_clean_latent_to_history(
        self,
        history: Tensor | None,
        clean_latent: Tensor,
    ) -> Tensor:
        """Append the finalized chunk's patchified clean latent to the history.

        Detaches before concatenating (along the post-patchify token axis,
        ``dim=-2``) so the history outlives the chunk's autograd graph.
        """
        if history is None:
            return clean_latent.detach().clone()
        return torch.cat([history, clean_latent.detach()], dim=-2)

    def _index_rollout_buffer(
        self,
        *,
        rollout: Tensor | None,
        per_step: Tensor | None,
        selected: Tensor,
        kind: str,
    ) -> Tensor | None:
        """Slice a per-rollout metadata buffer at the selected memory-frame indices.

        Prefers the rollout-scoped buffer; falls back to the per-AR-step slice
        (unindexed) when ``rollout`` is ``None``, which only happens when the
        corresponding conditioner is disabled and the consumer treats the
        slice as a no-op.

        Args:
            rollout: Full-trajectory buffer (e.g.
                ``[*batch_shape, F_total, 4, 4]`` for viewmats), indexed at
                ``selected`` along the frame axis when present.
            per_step: Per-AR-step slice used as fallback when ``rollout`` is
                ``None``.
            selected: ``LongTensor`` of memory frame indices in rollout
                coordinates, shape ``[K]``.
            kind: Tensor kind name for error messages.

        Returns:
            Indexed tensor with shape ``[*batch_shape, K, ...]`` (matrices)
            or ``[*batch_shape, K]`` (action); ``None`` when both
            ``rollout`` and ``per_step`` are ``None``.

        Raises:
            ValueError: ``rollout`` has a zero-length frame axis or an
                unexpected rank.
        """
        if rollout is None and per_step is None:
            return None
        if rollout is None:
            return per_step

        # Action is integer-typed with the frame on the trailing axis;
        # viewmats / Ks are float matrices with the frame at -3.
        if rollout.dtype in (torch.int32, torch.int64):
            if rollout.shape[-1] == 0:
                raise ValueError(f"rollout {kind} buffer has zero-length frame axis")
            return rollout.index_select(-1, selected)
        if rollout.ndim < 3 or rollout.shape[-3] == 0:
            raise ValueError(
                f"rollout {kind} buffer must have shape "
                f"[..., F_total, M, N] with F_total > 0; got "
                f"{tuple(rollout.shape)}"
            )
        return rollout.index_select(-3, selected)

    def _build_collapsed_rope_freqs(
        self,
        cache: HyWorldPlayWan21TransformerCache,
        t_positions: Tensor,
    ) -> Tensor:
        """Compute RoPE frequencies for arbitrary temporal positions.

        Uses the ``_freq_components(seq_t)`` primitive to build freqs at the
        prefill's collapsed memory positions ``[0, K)``, which the base
        ``shift_t(autoregressive_index)`` (chunk-aligned only) can't express.

        Raises:
            NotImplementedError: ``rope_adapter`` is not
                :class:`RotaryPositionEmbedding3D` or has context parallel
                enabled.
        """
        rope = cache.rope_adapter
        from flashdreams.core.attention.rope import RotaryPositionEmbedding3D

        if not isinstance(rope, RotaryPositionEmbedding3D):
            raise NotImplementedError(
                f"Reconstituted-context prefill currently supports only "
                f"RotaryPositionEmbedding3D; got {type(rope).__name__}. "
                f"KVCacheRelativeRotaryPositionEmbedding3D support lands "
                f"with multi-resolution / extended-window rollouts."
            )
        if rope.is_context_parallel_enabled():
            raise NotImplementedError(
                "Reconstituted-context prefill does not yet support "
                "context-parallel; CP wiring lands with the multi-rank "
                "action expansion."
            )
        freqs_t, freqs_h, freqs_w = rope._freq_components(t_positions.to(rope.device))
        return rope._cat_freqs(freqs_t, freqs_h, freqs_w)

    def _is_first_step_of_chunk(
        self,
        cache: HyWorldPlayWan21TransformerCache,
    ) -> bool:
        """Return ``True`` on the first denoising step of the current chunk."""
        return cache.prefill_completed_for_chunk != cache.autoregressive_index
