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

"""HY-WorldPlay action conditioner (phase 2b.3).

Adds discrete 81-class action conditioning on top of the Wan 2.2 TI2V 5B
stack. The four pieces compose into a drop-in replacement of the standard
encoder + transformer + network used by ``PIPELINE_WAN22_TI2V_5B``:

* :class:`HyWorldPlayCtrl` extends :class:`I2VCtrl` with an ``action`` field
  carrying the per-latent-frame action labels for the current AR chunk.
* :class:`HyWorldPlayWanCtrlEncoder` wraps :class:`I2VCtrlEncoder`, slicing
  the per-rollout label tensor into the per-AR-step ``action`` payload that
  the transformer consumes.
* :class:`HyWorldPlayWanDiTNetwork` extends :class:`WanDiTNetwork` with a
  dedicated ``action_embedding`` MLP whose output is summed into the time
  embedding before the AdaLN modulation projection — mirroring
  ``WanActionTimeTextImageEmbedding`` in upstream
  ``arwan_w_action_w_mem_relative_rope.py``. The MLP's ``linear_2`` is
  zero-initialised so the conditioner is a strict identity at random / zero
  init, matching upstream's ``add_discrete_action_parameters``.
* :class:`HyWorldPlayWan21Transformer` re-wires :meth:`predict_flow` to
  extract ``input.action`` and forward it through
  ``network_extra_kwargs``, and overrides
  :meth:`patchify_and_maybe_split_cp` so the action survives the
  patchify-rebuild of the I2V payload.

CP is intentionally restricted to size 1 here; multi-rank action expansion
is added together with the PRoPE camera path in phase 2b.4.
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
    """I2V control payload extended with the per-AR-step action slice.

    The ``action`` field carries the integer class labels for the current
    chunk's ``len_t`` latent frames (shape ``[*batch_shape, len_t]``); it
    survives the transformer's patchify-rebuild via
    :meth:`HyWorldPlayWan21Transformer.patchify_and_maybe_split_cp`.
    """

    action: Tensor | None = None
    """Per-latent-frame action labels for the current AR chunk."""


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
    """Wan I2V encoder that also emits a per-AR-step action label slice.

    Callers populate the full per-rollout label tensor via
    :meth:`set_action_labels` (typically right after
    ``pipeline.initialize_cache``); each :meth:`forward` call then slices
    the ``[ar_idx * len_t : (ar_idx + 1) * len_t]`` window and attaches it
    to the :class:`HyWorldPlayCtrl` payload.
    """

    def __init__(self, config: HyWorldPlayWanCtrlEncoderConfig) -> None:
        super().__init__(config)
        self._action_labels: Tensor | None = None

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

    def initialize_autoregressive_cache(self) -> I2VCtrlEncoderCache:
        # Match the parent's per-rollout reset; action labels are bound
        # explicitly after ``initialize_cache`` so we deliberately do *not*
        # clear them here. The runner clears them when it tears down the
        # rollout.
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
        action_chunk: Tensor | None = None
        if self._action_labels is not None:
            start = autoregressive_index * len_t
            end = start + len_t
            full = self._action_labels
            if end > full.shape[-1]:
                raise ValueError(
                    f"action labels exhausted at AR step {autoregressive_index}: "
                    f"need {end} entries but only {full.shape[-1]} provided."
                )
            action_chunk = full[..., start:end].to(device=base.latent.device)
        return HyWorldPlayCtrl(
            latent=base.latent,
            mask=base.mask,
            action=action_chunk,
        )


## ---------------------------------------------------------------------------
## Action-aware DiT network
## ---------------------------------------------------------------------------


@dataclass
class HyWorldPlayWanDiTNetworkConfig(WanDiTNetworkTI2V5BConfig):
    """Config for the action-aware Wan 2.2 TI2V 5B DiT.

    Shares every field with :class:`WanDiTNetworkTI2V5BConfig`; the only
    delta is the ``_target`` swap so the network gains the
    ``action_embedding`` MLP and the action-injecting forward.
    """

    _target: type = field(default_factory=lambda: HyWorldPlayWanDiTNetwork)


class HyWorldPlayWanDiTNetwork(WanDiTNetwork):
    """Wan DiT with an additive action embedding on the modulation pathway.

    A dedicated ``action_embedding`` MLP (same shape as ``time_embedding``)
    consumes sinusoidally-encoded action class labels and produces a
    per-latent-frame additive term that is summed into the time embedding
    before the ``time_projection`` SiLU+Linear that builds the AdaLN
    modulation parameters. ``linear_2`` is zero-initialised so the
    conditioner is a strict identity at random / zero init, which keeps
    parity with the base recipe until HY-WorldPlay weights are loaded.
    """

    def __init__(self, config: HyWorldPlayWanDiTNetworkConfig) -> None:
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
    ) -> Tensor:
        """Action-aware variant of :meth:`WanDiTNetwork.forward`.

        Extends the base forward by adding the action embedding to the time
        embedding before the modulation projection. When ``action`` is
        ``None`` the path is bit-for-bit identical to the base, which keeps
        the no-pose / no-action callsite free.
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

        if eager_mode:
            cache.before_update(current_chunk_idx)
        for block_idx, block in enumerate(self.blocks):
            assert isinstance(block, Block)
            x = block(
                x=x,
                e=block_e,
                rope_freqs=rope_freqs,
                cache=cache[block_idx],
                **block_extra_kwargs,
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
        if (
            isinstance(input, HyWorldPlayCtrl)
            and input.action is not None
            and "action" not in network_extra_kwargs
        ):
            network_extra_kwargs["action"] = input.action
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
            return HyWorldPlayCtrl(
                latent=patched_latent,
                mask=patched_mask,
                _is_patchified=True,
                action=x.action,
            )
        return super().patchify_and_maybe_split_cp(x)
