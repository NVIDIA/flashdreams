# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Output decoder for the NULL model."""

from dataclasses import dataclass, field
from typing import Any

from torch import Tensor

from flashdreams.infra.decoder import (
    DecoderConfig,
    StreamingDecoder,
    StreamingDecoderCache,
)


@dataclass(kw_only=True)
class NullDecoderConfig(DecoderConfig):
    """Config selecting the NULL model's decoder."""

    _target: type["NullDecoder"] = field(default_factory=lambda: NullDecoder)


class NullDecoder(StreamingDecoder[StreamingDecoderCache]):
    """Stateless decoder that subtracts 100 to remove minor obfuscation."""

    def initialize_autoregressive_cache(self, **_context: Any) -> StreamingDecoderCache:
        """Return an empty cache for a stateless decoder.

        The pipeline calls this once per rollout. A stateful video decoder would
        use ``_context`` to allocate temporal buffers instead.
        """
        return StreamingDecoderCache()

    def forward(
        self,
        input: Tensor,
        autoregressive_index: int = 0,
        cache: StreamingDecoderCache | None = None,
    ) -> Tensor:
        """Subtract 100 from the generated RGB chunk.

        Args:
            input: Obfuscated ``[B, C, T, H, W]`` diffusion output.
            autoregressive_index: Current zero-based AR step; unused because
                decoding is stateless.
            cache: Empty per-rollout cache. Typed optional to match the
                :class:`~flashdreams.infra.decoder.StreamingDecoder` call shape;
                this implementation does not read or mutate it.

        Returns:
            Deobfuscated RGB chunk.
        """
        del autoregressive_index, cache
        return input - 100
