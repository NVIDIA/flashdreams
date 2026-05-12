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

"""ArtiFixer transformer wrapper.

Subclasses :class:`Wan21Transformer` so the recipe can:

  - own the two ``PropeDotProductAttention`` modules (target/source and
    neighbor cameras) used by every transformer block's cross-attention
    branch;
  - hold per-rollout conditioning state (full-length opacity, Plucker rays,
    camera matrices, optional neighbor cameras) populated once by the
    pipeline at ``initialize_autoregressive_cache`` time;
  - thread per-AR-chunk extras (patchified opacity / camera-ray features,
    target-camera PRoPE precompute) into ``predict_flow`` ->
    ``WanDiTNetwork.forward`` via an :class:`ArtifixerCtrl` payload.

The pipeline (Phase 3.5) owns the per-chunk slicing + patchification +
PRoPE precompute updates and packages them into an ``ArtifixerCtrl``
that flows through ``DiffusionModel.generate(input=ctrl)``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from artifixer.network.prope import PropeDotProductAttention
from torch import Tensor

from flashdreams.recipes.wan import Wan21Transformer, Wan21TransformerConfig
from flashdreams.recipes.wan.autoencoder.i2v import I2VCtrl
from flashdreams.recipes.wan.transformer.wan21 import Wan21TransformerCache


@dataclass(kw_only=True)
class ArtifixerCtrl:
    """Per-AR-chunk conditioning payload for the ArtiFixer transformer.

    All tensors are already sliced to the current chunk's frame range; the
    patchifiable ones are also already passed through
    :func:`artifixer.network.patches.patchify_opacity` /
    :func:`patchify_camera_rays`. The pipeline owns this preparation
    (Phase 3.5); the transformer's ``predict_flow`` reads from here and
    forwards into ``network_extra_kwargs``.

    The cross-attention "neighbor" KV bank is *not* carried here -- it is
    set once per rollout on every block's cross-attention module via
    ``ArtifixerCrossAttention.initialize_neighbor_cache`` and stays there
    across AR chunks.
    """

    opacity_extra: Tensor
    """Patchified per-token opacity features
    ``[*batch_shape, L, opacity_embedding_dim]``."""

    camera_extra: Tensor
    """Patchified per-token Plucker camera-ray features
    ``[*batch_shape, L, camera_embedding_dim]``."""

    ignore_neighbors: bool = False
    """Diffusion-forcing CFG-dropout knob (matches dreamfix L936). Inference
    leaves this ``False``; training-time DMD distillation may toggle it."""

    _is_patchified: bool = True
    """Sentinel for ``Wan21Transformer.patchify_and_maybe_split_cp``:
    ``opacity_extra`` and ``camera_extra`` arrive already patchified, so the
    pipeline-level ``patchify_and_maybe_split_cp(input)`` dispatch in
    ``DiffusionModel.generate`` (base.py L163-164) is a no-op for us."""


@dataclass(kw_only=True)
class ArtifixerWanTransformerConfig(Wan21TransformerConfig):
    """Wan2.1 transformer config that constructs an :class:`ArtifixerWanTransformer`."""

    _target: type = field(default_factory=lambda: ArtifixerWanTransformer)

    prope_freq_base: float = 100.0
    """RoPE frequency base for the PRoPE cross-attention transform."""

    prope_freq_scale: float = 1.0
    """RoPE frequency scale for the PRoPE cross-attention transform."""


class ArtifixerWanTransformer(Wan21Transformer):
    """Wan 2.1 transformer with ArtiFixer's PRoPE cross-attention modules."""

    config: ArtifixerWanTransformerConfig

    def __init__(self, config: ArtifixerWanTransformerConfig) -> None:
        super().__init__(config)

        head_dim = config.network.dim // config.network.num_heads
        self.prope_cross_attn_src = PropeDotProductAttention(
            head_dim=head_dim,
            freq_base=config.prope_freq_base,
            freq_scale=config.prope_freq_scale,
        )
        self.prope_cross_attn_tgt = PropeDotProductAttention(
            head_dim=head_dim,
            freq_base=config.prope_freq_base,
            freq_scale=config.prope_freq_scale,
        )

        # Match the dreamfix dtype convention.
        self.prope_cross_attn_src = self.prope_cross_attn_src.to(dtype=config.dtype)
        self.prope_cross_attn_tgt = self.prope_cross_attn_tgt.to(dtype=config.dtype)

    def predict_flow(  # type: ignore[override]
        self,
        noisy_latent: Tensor,
        timestep: Tensor,
        cache: Wan21TransformerCache,
        input: ArtifixerCtrl | I2VCtrl | None = None,
        network_extra_kwargs: dict[str, Any] | None = None,
    ) -> Tensor:
        """Add ArtiFixer block_extra_kwargs to the base predict_flow.

        Reads patchified opacity/camera features and the ``ignore_neighbors``
        flag from :class:`ArtifixerCtrl` and forwards them to every block via
        ``network_extra_kwargs``. Also passes the PRoPE source/target modules
        themselves -- they are bound at construction time, but
        :class:`ArtifixerCrossAttention.forward` expects them as forward args
        so they show up uniformly under ``block_extra_kwargs``.

        When ``input`` is not an :class:`ArtifixerCtrl` (e.g. T2V smoke
        without any conditioning), falls through to the base behavior;
        :class:`ArtifixerBlock` treats every conditioning kwarg as optional.
        """
        network_extra_kwargs = dict(network_extra_kwargs or {})
        if isinstance(input, ArtifixerCtrl):
            network_extra_kwargs.setdefault("opacity_extra", input.opacity_extra)
            network_extra_kwargs.setdefault("camera_extra", input.camera_extra)
            network_extra_kwargs.setdefault("ignore_neighbors", input.ignore_neighbors)
            network_extra_kwargs.setdefault("prope_src", self.prope_cross_attn_src)
            network_extra_kwargs.setdefault("prope_tgt", self.prope_cross_attn_tgt)
            # The base ``predict_flow`` will dispatch ``_build_network_input``
            # on ``input`` -- ArtifixerCtrl isn't an I2VCtrl, so pass None
            # for the I2V mask-inject leg.
            i2v_input: I2VCtrl | None = None
        else:
            i2v_input = input

        return super().predict_flow(
            noisy_latent=noisy_latent,
            timestep=timestep,
            cache=cache,
            input=i2v_input,
            network_extra_kwargs=network_extra_kwargs,
        )


__all__ = [
    "ArtifixerCtrl",
    "ArtifixerWanTransformer",
    "ArtifixerWanTransformerConfig",
]
