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
  - thread that state into ``predict_flow`` -> ``WanDiTNetwork.forward``
    via ``network_extra_kwargs`` (Phase 3.3 wires the slicing).

This commit lands the static structure only -- the per-rollout state and
the ``predict_flow`` override that slices it per AR chunk are added in
later Phase 3 sub-commits.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from artifixer.network.prope import PropeDotProductAttention

from flashdreams.recipes.wan import Wan21Transformer, Wan21TransformerConfig


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


__all__ = [
    "ArtifixerWanTransformer",
    "ArtifixerWanTransformerConfig",
]
