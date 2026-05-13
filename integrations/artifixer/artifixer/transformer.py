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

The pipeline owns the per-chunk slicing + patchification + PRoPE
precompute updates and packages them into an ``ArtifixerCtrl`` that
flows through ``DiffusionModel.generate(input=ctrl)``.
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
    :func:`patchify_camera_rays`. The pipeline owns this preparation; the
    transformer's ``predict_flow`` reads from here and forwards into
    ``network_extra_kwargs``.

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

        Wraps the ArtiFixer per-block extras inside a ``block_extra_kwargs``
        dict so :meth:`Wan21Transformer._predict_flow`'s
        ``**network_extra_kwargs`` unpack lands them as
        ``WanDiTNetwork.forward(... block_extra_kwargs=...)`` -- a single
        kwarg, *not* individual ones. ``WanDiTNetwork.forward`` then forwards
        each entry to every block as keyword args via ``**block_extra_kwargs``
        (network.py L438-449), where :class:`ArtifixerBlock.forward` accepts
        ``opacity_extra`` / ``camera_extra`` / ``prope_src`` / ``prope_tgt`` /
        ``ignore_neighbors`` directly.

        The PRoPE source/target modules are bound at construction time, but
        :class:`ArtifixerCrossAttention.forward` expects them as forward
        kwargs so every per-block call gets them via the same path.

        When ``input`` is not an :class:`ArtifixerCtrl` (e.g. text-only smoke
        without any conditioning), falls through to the base behavior;
        :class:`ArtifixerBlock` treats every conditioning kwarg as optional.
        """
        network_extra_kwargs = dict(network_extra_kwargs or {})
        if isinstance(input, ArtifixerCtrl):
            block_extra_kwargs = dict(network_extra_kwargs.get("block_extra_kwargs", {}))
            block_extra_kwargs.setdefault("opacity_extra", input.opacity_extra)
            block_extra_kwargs.setdefault("camera_extra", input.camera_extra)
            block_extra_kwargs.setdefault("ignore_neighbors", input.ignore_neighbors)
            block_extra_kwargs.setdefault("prope_src", self.prope_cross_attn_src)
            block_extra_kwargs.setdefault("prope_tgt", self.prope_cross_attn_tgt)
            network_extra_kwargs["block_extra_kwargs"] = block_extra_kwargs
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

    def finalize_kv_cache(self, *args: Any, **kwargs: Any) -> None:
        """No-op: artifixer (dreamfix reference) does not finalize the KV cache.

        The dreamfix ``ArtifixerKvCachePipeline.generate_samples_from_batch``
        advances its KV cache *in-place* during the regular denoise forwards
        and never runs an extra "finalization" forward at AR-chunk
        boundaries. The FlashDreams default ``finalize_kv_cache`` runs one
        more ``predict_flow`` at the context-noise timestep to advance the
        cache; that extra forward writes a *different* KV state than
        dreamfix's last in-loop forward, which empirically opens a ~7-9 dB
        cross-backend PSNR gap at the start of every AR chunk past chunk 0
        (calls 4, 8 in ``scripts/parity_harness.py``'s per-step diff).

        Skipping the extra forward here aligns FlashDreams' KV-cache
        semantics with the dreamfix reference: the cache going into AR
        chunk ``N+1`` is the one written by the final denoise step of AR
        chunk ``N``, identical to what dreamfix uses. ``cache.finalize`` is
        still called by ``DiffusionModel.finalize`` after this (the
        bookkeeping that increments ``autoregressive_index``), so the
        rollout state advances correctly -- only the redundant predict is
        suppressed.
        """
        del args, kwargs  # intentionally no-op; see docstring


__all__ = [
    "ArtifixerCtrl",
    "ArtifixerWanTransformer",
    "ArtifixerWanTransformerConfig",
]
