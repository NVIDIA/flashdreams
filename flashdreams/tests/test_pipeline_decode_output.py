# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU unit test for the ``StreamInferencePipeline`` latent / decode seam.

Verifies the token-streaming hooks added to ``_generate``: ``decode_output``
gates the VAE decoder, and ``cache.clean_latent`` is populated either way.
Stages are replaced by fakes so no GPU, real DiT, or checkpoints are needed.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch
from torch import Tensor, nn

from flashdreams.infra.pipeline.base import (
    StreamInferencePipeline,
    StreamInferencePipelineCache,
)

pytestmark = pytest.mark.ci_cpu


class _FakeDiffusionModel:
    """Stands in for the diffusion model: returns a fixed clean latent."""

    def __init__(self, clean_latent: Tensor) -> None:
        self._clean_latent = clean_latent
        self.calls = 0

    def generate(
        self, *, autoregressive_index: int, cache: Any, input: Any
    ) -> tuple[Tensor, object]:
        self.calls += 1
        return self._clean_latent, object()


class _FakeDecoder:
    """Stands in for the VAE decoder: returns a fixed decoded tensor."""

    def __init__(self, decoded: Tensor) -> None:
        self._decoded = decoded
        self.calls = 0

    def __call__(self, *, input: Any, autoregressive_index: int, cache: Any) -> Tensor:
        self.calls += 1
        return self._decoded


class _SeamPipeline(StreamInferencePipeline):
    """A pipeline whose stages are fakes, bypassing config-driven setup."""

    def __init__(self, *, decoder: Any, diffusion_model: Any) -> None:
        nn.Module.__init__(self)
        self.encoder = None
        self.decoder = decoder
        self.diffusion_model = diffusion_model


def _make() -> tuple[_SeamPipeline, _FakeDecoder, Tensor, Tensor]:
    clean_latent = torch.randn(1, 4, 2, 8, 8)
    decoded = torch.randn(1, 3, 8, 64, 64)
    decoder = _FakeDecoder(decoded)
    pipe = _SeamPipeline(
        decoder=decoder,
        diffusion_model=_FakeDiffusionModel(clean_latent),
    )
    return pipe, decoder, clean_latent, decoded


def _cache() -> StreamInferencePipelineCache[Any, Any, Any]:
    # The fakes never read these caches; one Any-typed dummy stands in for both
    # the transformer and decoder cache slots.
    dummy: Any = object()
    return StreamInferencePipelineCache(
        transformer_cache=dummy,
        decoder_cache=dummy,
    )


def test_decode_output_true_runs_decoder_and_caches_latent() -> None:
    pipe, decoder, clean_latent, decoded = _make()
    cache = _cache()

    output = pipe.generate(0, cache, input=None, decode_output=True)

    assert output is decoded
    assert cache.clean_latent is clean_latent
    assert decoder.calls == 1


def test_decode_output_false_skips_decoder_and_returns_latent() -> None:
    pipe, decoder, clean_latent, _ = _make()
    cache = _cache()

    output = pipe.generate(0, cache, input=None, decode_output=False)

    assert output is clean_latent
    assert cache.clean_latent is clean_latent
    assert decoder.calls == 0


def test_default_keeps_decoding() -> None:
    # Backward compatibility: decode_output defaults to True, so the pixel
    # path is unchanged for every existing caller.
    pipe, decoder, _, decoded = _make()
    cache = _cache()

    assert pipe.generate(0, cache, input=None) is decoded
    assert decoder.calls == 1
