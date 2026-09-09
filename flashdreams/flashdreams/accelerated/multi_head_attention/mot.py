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

"""Call-level routing between understanding and generation attention pathways."""

from __future__ import annotations

from typing import Any

from torch import Tensor, nn

from flashdreams.accelerated.multi_head_attention import (
    AttentionMask,
    MoTRoute,
    MultiHeadAttention,
)


class MoTMultiHeadAttention(nn.Module):
    """Route homogeneous attention calls to one of two independently weighted paths."""

    understanding: MultiHeadAttention[Any]
    """Attention module used for understanding tokens."""

    generation: MultiHeadAttention[Any]
    """Attention module used for generation tokens."""

    def __init__(
        self,
        understanding: MultiHeadAttention[Any],
        generation: MultiHeadAttention[Any],
    ) -> None:
        """Register the two model-of-thought attention pathways."""
        super().__init__()
        if not isinstance(understanding, MultiHeadAttention):
            raise TypeError("understanding must be a MultiHeadAttention module")
        if not isinstance(generation, MultiHeadAttention):
            raise TypeError("generation must be a MultiHeadAttention module")
        self.understanding = understanding
        self.generation = generation

    def branch(self, route: MoTRoute) -> MultiHeadAttention[Any]:
        """Return the attention module selected by ``route``."""
        self._validate_route(route)
        if route is MoTRoute.UNDERSTANDING:
            return self.understanding
        return self.generation

    def compute_kv(
        self,
        context: Tensor,
        rope_freqs: Tensor | None = None,
        *,
        route: MoTRoute,
    ) -> Any:
        """Project static context through one homogeneous pathway."""
        return self.branch(route).compute_kv(context, rope_freqs)

    def forward(
        self,
        x: Tensor,
        kv_cache: Any,
        rope_freqs: Tensor | None = None,
        *,
        route: MoTRoute,
        attn_mask: AttentionMask | None = None,
    ) -> Tensor:
        """Apply one homogeneous model-of-thought pathway.

        Args:
            x: Query tokens for a single pathway.
            kv_cache: Cache owned by the selected attention module.
            rope_freqs: Optional rotary-position data.
            route: Understanding or generation pathway for the complete call.
            attn_mask: Optional visibility mask supported by the selected backend.

        Returns:
            Output from the selected attention module.
        """
        branch = self.branch(route)
        return branch(x, kv_cache, rope_freqs, attn_mask=attn_mask)

    @staticmethod
    def _validate_route(route: MoTRoute) -> None:
        if not isinstance(route, MoTRoute):
            raise TypeError(f"route must be a MoTRoute; got {route!r}")


__all__ = ["MoTMultiHeadAttention"]
