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

"""HY-WorldPlay action + camera conditioner glue (phases 2b.3 + 2b.4).

Adds discrete 81-class action conditioning (2b.3) and the per-AR-step
camera data plumbing for the PRoPE branch (2b.4) on top of the Wan 2.2
TI2V 5B stack. The four pieces compose into a drop-in replacement of the
standard encoder + transformer + network used by
``PIPELINE_WAN22_TI2V_5B``:

* :class:`HyWorldPlayCtrl` extends :class:`I2VCtrl` with ``action``,
  ``viewmats``, and ``Ks`` fields. ``action`` carries the per-latent-
  frame discrete labels; ``viewmats`` / ``Ks`` carry the per-frame W2C
  extrinsics + intrinsics consumed by the PRoPE attention branch in
  :class:`hy_worldplay._camera.HyWorldPlayPRoPEBlock`.
* :class:`HyWorldPlayWanCtrlEncoder` wraps :class:`I2VCtrlEncoder` and
  slices the per-rollout action labels / camera tensors into the per-AR-
  step :class:`HyWorldPlayCtrl` payload via :meth:`set_action_labels` /
  :meth:`set_camera_data`. Either source can be bound independently;
  unbound sources flow through as ``None``.
* :class:`HyWorldPlayWanDiTNetwork` extends :class:`WanDiTNetwork` with a
  zero-residual ``action_embedding`` MLP summed into the time embedding
  before the AdaLN modulation projection (mirrors
  ``WanActionTimeTextImageEmbedding`` in upstream's
  ``arwan_w_action_w_mem_relative_rope.py``), and -- when
  :attr:`HyWorldPlayWanDiTNetworkConfig.use_prope_blocks` is set --
  builds :class:`HyWorldPlayPRoPEBlock` blocks so each self-attention
  runs the dual-branch RoPE + PRoPE path. Both injection paths are
  zero-init residuals so the network stays parity-safe at random /
  base-recipe init.
* :class:`HyWorldPlayWan21Transformer` re-wires :meth:`predict_flow` to
  forward ``action`` / ``viewmats`` / ``Ks`` through
  ``network_extra_kwargs``, and overrides
  :meth:`patchify_and_maybe_split_cp` so all three fields survive the
  patchify-rebuild of the I2V payload.

CP is intentionally restricted to size 1 here; multi-rank action +
PRoPE expansion lands in a follow-up.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
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

## ---------------------------------------------------------------------------
## Per-AR-step control payload
## ---------------------------------------------------------------------------


@dataclass(kw_only=True)
class HyWorldPlayCtrl(I2VCtrl):
    """I2V control payload extended with per-AR-step action + camera slices.

    The ``action`` field carries the integer class labels for the current
    chunk's ``len_t`` latent frames (shape ``[*batch_shape, len_t]``).
    The ``viewmats`` / ``Ks`` fields carry the per-frame world-to-camera
    extrinsic and intrinsic matrices consumed by the PRoPE attention
    branch (2b.4). All three fields survive the transformer's
    patchify-rebuild via
    :meth:`HyWorldPlayWan21Transformer.patchify_and_maybe_split_cp`.
    """

    action: Tensor | None = None
    """Per-latent-frame action labels for the current AR chunk."""

    viewmats: Tensor | None = None
    """Per-latent-frame world-to-camera matrices for the current AR chunk;
    shape ``[*batch_shape, len_t, 4, 4]``. Consumed by the PRoPE branch
    of :class:`HyWorldPlayPRoPEBlock` self-attention."""

    Ks: Tensor | None = None
    """Per-latent-frame intrinsics (with cx/cy renormalised to 0.5) for
    the current AR chunk; shape ``[*batch_shape, len_t, 3, 3]``."""


## ---------------------------------------------------------------------------
## I2V + action encoder
## ---------------------------------------------------------------------------


@dataclass(kw_only=True)
class HyWorldPlayWanCtrlEncoderConfig(WanI2VCtrlEncoderConfig):
    """Config for the action-aware I2V control encoder.

    The encoder slices the per-rollout label tensor (set externally via
    :meth:`HyWorldPlayWanCtrlEncoder.set_action_labels`) into the per-AR-step
    ``action`` field of :class:`HyWorldPlayCtrl`; ``latent`` / ``mask`` are
    produced by the inherited Wan I2V branch unchanged.
    """

    _target: type = field(default_factory=lambda: HyWorldPlayWanCtrlEncoder)


class HyWorldPlayWanCtrlEncoder(I2VCtrlEncoder):
    """Wan I2V encoder that also emits per-AR-step action + camera slices.

    Callers populate the full per-rollout label tensor via
    :meth:`set_action_labels` and (for the PRoPE camera branch added in
    2b.4) the per-rollout extrinsics + intrinsics via
    :meth:`set_camera_data`. Each :meth:`forward` call then slices the
    ``[ar_idx * len_t : (ar_idx + 1) * len_t]`` window from whichever
    sources are bound and attaches them to the :class:`HyWorldPlayCtrl`
    payload; unbound sources fall through as ``None`` so downstream
    consumers stay opt-in.
    """

    def __init__(self, config: HyWorldPlayWanCtrlEncoderConfig) -> None:
        super().__init__(config)
        self._action_labels: Tensor | None = None
        self._viewmats: Tensor | None = None
        self._intrinsics: Tensor | None = None

    def set_action_labels(self, labels: Tensor) -> None:
        """Bind the per-rollout action labels.

        ``labels`` must have a trailing axis whose length is divisible by the
        transformer's ``len_t`` so successive AR steps see equal-sized
        slices.
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
        """Bind the per-rollout camera extrinsics + intrinsics.

        Args:
            viewmats: Per-latent-frame world-to-camera matrices, shape
                ``[*, n_latents, 4, 4]`` where ``n_latents`` is divisible
                by the transformer's ``len_t``.
            Ks: Per-latent-frame intrinsics (with cx/cy renormalised to
                0.5 by :func:`hy_worldplay._pose.parse_pose_data`), shape
                ``[*, n_latents, 3, 3]``. Both tensors share the same
                leading axes.
        """
        if viewmats.ndim < 3 or viewmats.shape[-2:] != (4, 4):
            raise ValueError(
                f"viewmats must have trailing shape (n_latents, 4, 4); "
                f"got {tuple(viewmats.shape)}."
            )
        if Ks.ndim < 3 or Ks.shape[-2:] != (3, 3):
            raise ValueError(
                f"Ks must have trailing shape (n_latents, 3, 3); "
                f"got {tuple(Ks.shape)}."
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

    def initialize_autoregressive_cache(self) -> I2VCtrlEncoderCache:
        # Match the parent's per-rollout reset; action / camera tensors
        # are bound explicitly after ``initialize_cache`` so we
        # deliberately do *not* clear them here. The runner clears them
        # when it tears down the rollout.
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

        return HyWorldPlayCtrl(
            latent=base.latent,
            mask=base.mask,
            action=action_chunk,
            viewmats=viewmats_chunk,
            Ks=Ks_chunk,
        )


## ---------------------------------------------------------------------------
## Action-aware DiT network
## ---------------------------------------------------------------------------


@dataclass
class HyWorldPlayWanDiTNetworkConfig(WanDiTNetworkTI2V5BConfig):
    """Config for the action / camera-aware Wan 2.2 TI2V 5B DiT.

    Shares every field with :class:`WanDiTNetworkTI2V5BConfig`. Adds:

    * :attr:`_target` swap so the network gains the ``action_embedding``
      MLP and the action / camera-injecting forward.
    * :attr:`use_prope_blocks` (2b.4 knob) -- when ``True``,
      :meth:`HyWorldPlayWanDiTNetwork._build_block` returns
      :class:`HyWorldPlayPRoPEBlock` instances so each block runs the
      dual-branch RoPE + PRoPE self-attention. Defaults to ``False`` so
      enabling action conditioning alone (no camera) keeps the standard
      :class:`Block` stack and the network's behaviour stays bit-identical
      to 2b.3.
    """

    _target: type = field(default_factory=lambda: HyWorldPlayWanDiTNetwork)

    use_prope_blocks: bool = False
    """Build PRoPE-aware blocks (dual-branch RoPE + PRoPE self-attn)
    instead of the standard :class:`Block`. When ``True`` the encoder must
    bind per-rollout camera data so each AR step's :class:`HyWorldPlayCtrl`
    carries ``viewmats`` + ``Ks`` slices."""


class HyWorldPlayWanDiTNetwork(WanDiTNetwork):
    """Wan DiT with action-modulated AdaLN + optional PRoPE blocks.

    Two extensions on top of :class:`WanDiTNetwork`:

    * ``action_embedding`` MLP (same shape as ``time_embedding``) consumes
      sinusoidally-encoded action class labels and produces a per-latent-
      frame additive term summed into the time embedding before
      ``time_projection``. ``linear_2`` is zero-initialised so the
      conditioner is a strict identity at random / zero init.
    * When :attr:`HyWorldPlayWanDiTNetworkConfig.use_prope_blocks` is set,
      :meth:`_build_block` returns :class:`HyWorldPlayPRoPEBlock` instances
      so each block runs the dual-branch RoPE + PRoPE self-attention. The
      forward routes ``viewmats`` and ``Ks`` (per-AR-step camera data)
      through ``block_extra_kwargs`` so each PRoPE block sees the same
      per-frame extrinsics + intrinsics. ``o_prope`` in the PRoPE attention
      is zero-init so the dual-branch path also stays a strict identity
      until HY-WorldPlay weights are loaded.
    """

    def __init__(self, config: HyWorldPlayWanDiTNetworkConfig) -> None:
        # Pre-init nn.Module so we can stash ``use_prope_blocks`` *before*
        # super().__init__() runs ``_build_block`` via ``self``. The
        # outer ``super().__init__()`` re-initialises ``_parameters`` /
        # ``_modules`` etc. but leaves ordinary attributes (like
        # ``_hy_use_prope_blocks``) alone.
        nn.Module.__init__(self)
        self._hy_use_prope_blocks = config.use_prope_blocks
        super().__init__(config)
        self.action_embedding = nn.Sequential(
            nn.Linear(self.freq_dim, self.dim),
            nn.SiLU(),
            nn.Linear(self.dim, self.dim),
        )
        # Zero-init the residual head so the action branch is an identity at
        # construction time (matches upstream
        # ``add_discrete_action_parameters``).
        zero_linear = self.action_embedding[-1]
        assert isinstance(zero_linear, nn.Linear)
        nn.init.zeros_(zero_linear.weight)
        if zero_linear.bias is not None:
            nn.init.zeros_(zero_linear.bias)

    def _build_block(self, layer_idx: int) -> Block:
        if self._hy_use_prope_blocks:
            # Lazy-imported so the PRoPE block module is only pulled in
            # when the camera conditioner is actually enabled (keeps the
            # action-only path free of the extra import cost).
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
        """Action / camera-aware variant of :meth:`WanDiTNetwork.forward`.

        Extends the base forward by (a) adding the action embedding to the
        time embedding before the modulation projection and (b) threading
        ``viewmats`` + ``Ks`` through ``block_extra_kwargs`` when PRoPE
        blocks are active. When both ``action`` and ``viewmats`` are
        ``None`` the modulation path is bit-for-bit identical to the base;
        PRoPE blocks still require ``viewmats`` even when nothing else is
        bound.
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
        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, timesteps).type_as(x)
        )

        if action is not None:
            action_e = self._compute_action_embedding(action=action, x=x, L=L)
            # Adding ``[..., L, D]`` to a scalar-timestep ``[..., D]`` e
            # broadcasts e per-token; the rest of the path then takes the
            # per-token branch.
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

        # Thread camera data per-block when PRoPE blocks are active. We
        # copy ``block_extra_kwargs`` rather than mutate the caller's
        # dict so subsequent forwards (or other branches) see a clean
        # slate.
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

        if eager_mode:
            cache.before_update(current_chunk_idx)
        for block_idx, block in enumerate(self.blocks):
            assert isinstance(block, Block)
            x = block(
                x=x,
                e=block_e,
                rope_freqs=rope_freqs,
                cache=cache[block_idx],
                **block_kwargs,
            )
        if eager_mode:
            cache.after_update(current_chunk_idx)

        x = self.head(x, head_e)
        return x

    def _compute_action_embedding(
        self,
        *,
        action: Tensor,
        x: Tensor,
        L: int,
    ) -> Tensor:
        """Lift per-latent-frame action labels to a per-token additive term.

        Sinusoidally encodes the integer labels, runs them through the
        zero-residual MLP, then ``repeat_interleave``s the resulting
        per-frame embedding across the ``tokens_per_frame`` slots of each
        latent frame in the post-patchify token axis. Multi-rank CP is
        gated off here (PRoPE in phase 2b.4 lifts that restriction).
        """
        cp_group = getattr(self, "_cp_group", None)
        if cp_group is not None:
            raise NotImplementedError(
                "HyWorldPlayWanDiTNetwork does not yet support context-parallel "
                "(cp_size > 1) action conditioning; this is enabled together "
                "with PRoPE in phase 2b.4."
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


## ---------------------------------------------------------------------------
## Action-aware Wan 2.1 transformer
## ---------------------------------------------------------------------------


@dataclass(kw_only=True)
class HyWorldPlayWan21TransformerConfig(Wan21TransformerConfig):
    """Config for the action-aware Wan 2.1 transformer.

    Same set of knobs as :class:`Wan21TransformerConfig`; the only delta is
    that the default ``network`` is the action-aware DiT and the
    ``_target`` builds :class:`HyWorldPlayWan21Transformer` so the
    ``predict_flow`` + ``patchify_and_maybe_split_cp`` overrides take
    effect.
    """

    _target: type = field(default_factory=lambda: HyWorldPlayWan21Transformer)

    network: HyWorldPlayWanDiTNetworkConfig = field(
        default_factory=HyWorldPlayWanDiTNetworkConfig
    )


class HyWorldPlayWan21Transformer(Wan21Transformer):
    """Wan 2.1 transformer that threads the action label into the network.

    Two overrides:

    * :meth:`predict_flow` reads ``input.action`` (when ``input`` is a
      :class:`HyWorldPlayCtrl`) and forwards it through
      ``network_extra_kwargs={"action": ...}`` so it reaches
      :meth:`HyWorldPlayWanDiTNetwork.forward`.
    * :meth:`patchify_and_maybe_split_cp` preserves the ``action`` slice
      across the in-place patchify pass that the base implementation
      otherwise rebuilds via ``I2VCtrl(...)``, which would drop subclass
      fields. Action labels are per-latent-frame integers; they do not
      participate in patchify themselves and are passed through unchanged.
    """

    def predict_flow(
        self,
        noisy_latent: Tensor,
        timestep: Tensor,
        cache: Wan21TransformerCache,
        input: I2VCtrl | None = None,
        network_extra_kwargs: dict[str, Any] | None = None,
    ) -> Tensor:
        network_extra_kwargs = dict(network_extra_kwargs or {})
        if isinstance(input, HyWorldPlayCtrl):
            if input.action is not None and "action" not in network_extra_kwargs:
                network_extra_kwargs["action"] = input.action
            if (
                input.viewmats is not None
                and "viewmats" not in network_extra_kwargs
            ):
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

    def patchify_and_maybe_split_cp(self, x: Any) -> Any:
        if isinstance(x, HyWorldPlayCtrl):
            if x._is_patchified:
                return x
            patched_latent = self.patchify_and_maybe_split_cp(x.latent)
            patched_mask = self.patchify_and_maybe_split_cp(x.mask)
            # action / viewmats / Ks are per-latent-frame (not per-token)
            # and do not participate in the patchify reshape; pass them
            # through unchanged so the PRoPE branch sees the same
            # ``[..., len_t, 4, 4]`` / ``[..., len_t]`` layout it would
            # see on a fresh ctrl.
            return HyWorldPlayCtrl(
                latent=patched_latent,
                mask=patched_mask,
                _is_patchified=True,
                action=x.action,
                viewmats=x.viewmats,
                Ks=x.Ks,
            )
        return super().patchify_and_maybe_split_cp(x)
