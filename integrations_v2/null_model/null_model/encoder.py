# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Scalar input encoder for the NULL model."""

from dataclasses import dataclass, field
from typing import Any

from torch import Tensor

from flashdreams.infra.encoder import (
    EncoderConfig,
    StreamingEncoder,
    StreamingEncoderCache,
)


@dataclass(kw_only=True)
class NullInputEncoderConfig(EncoderConfig):
    """Config selecting the NULL model's scalar encoder."""

    _target: type["NullInputEncoder"] = field(default_factory=lambda: NullInputEncoder)


class NullInputEncoder(StreamingEncoder[StreamingEncoderCache]):
    """Stateless per-step encoder that adds 100 for minor obfuscation."""

    def initialize_autoregressive_cache(self, **_context: Any) -> StreamingEncoderCache:
        """Return an empty cache for a stateless encoder.

        The pipeline calls this once when it initializes a rollout. Real
        control encoders can consume ``_context`` and return a cache subclass
        carrying state between AR steps.
        """
        return StreamingEncoderCache()

    def forward(
        self,
        input: Tensor,
        autoregressive_index: int = 0,
        cache: StreamingEncoderCache | None = None,
    ) -> Tensor:
        """Add 100 to a validated ``[1, 1]`` input tensor.

        Args:
            input: Tensor with shape ``[1, 1]``.
            autoregressive_index: Unused because
                this encoder has no step-dependent behavior.
            cache: Unused.

        Returns:
            Obfuscated input tensor.

        Raises:
            AssertionError: ``input`` is not a tensor with shape ``[1, 1]``.
        """
        del autoregressive_index, cache

        assert isinstance(input, Tensor), (
            f"expected input to be a Tensor, got {type(input).__name__}"
        )
        assert input.shape == (1, 1), (
            f"expected input tensor shape (1, 1), got {tuple(input.shape)}"
        )
        return input + 100
