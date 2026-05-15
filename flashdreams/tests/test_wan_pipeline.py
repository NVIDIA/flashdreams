# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch

from flashdreams.infra.decoder import StreamingDecoderCache
from flashdreams.infra.diffusion.transformer import TransformerAutoregressiveCache
from flashdreams.infra.encoder import StreamingEncoderCache
from flashdreams.infra.pipeline import (
    StreamInferencePipeline,
    StreamInferencePipelineCache,
)
from flashdreams.recipes.wan import pipeline as wan_pipeline_module
from flashdreams.recipes.wan.pipeline import WanInferencePipeline
from flashdreams.recipes.wan.transformer.wan21 import Wan21TransformerConfig


class _FakeTextEncoder:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def __call__(self, prompts: list[str]) -> torch.Tensor:
        self.calls.append(list(prompts))
        return torch.full((len(prompts), 2, 3), float(len(self.calls)))


class _FakeStreamingVideoDecoder:
    spatial_compression_ratio = 2


def test_wan_initialize_cache_can_be_reused_for_multiple_rollouts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Long-lived hosts reuse one pipeline instance across session resets."""
    captured_contexts: list[dict[str, Any]] = []

    def capture_initialize_cache(
        self: StreamInferencePipeline,
        *,
        transformer_context: dict[str, Any] | None = None,
        encoder_context: dict[str, Any] | None = None,
        decoder_context: dict[str, Any] | None = None,
    ) -> StreamInferencePipelineCache[Any, Any, Any]:
        del self, encoder_context, decoder_context
        assert transformer_context is not None
        captured_contexts.append(transformer_context)
        return StreamInferencePipelineCache(
            transformer_cache=TransformerAutoregressiveCache(),
            encoder_cache=StreamingEncoderCache(),
            decoder_cache=StreamingDecoderCache(),
        )

    monkeypatch.setattr(
        StreamInferencePipeline,
        "initialize_cache",
        capture_initialize_cache,
    )
    monkeypatch.setattr(
        wan_pipeline_module,
        "StreamingVideoDecoder",
        _FakeStreamingVideoDecoder,
    )

    pipeline = WanInferencePipeline.__new__(WanInferencePipeline)
    torch.nn.Module.__init__(pipeline)
    text_encoder = _FakeTextEncoder()
    pipeline.text_encoder = text_encoder
    pipeline.image_encoder = None
    pipeline.encoder = object()
    pipeline.decoder = _FakeStreamingVideoDecoder()
    pipeline.diffusion_model = SimpleNamespace(
        transformer=SimpleNamespace(config=Wan21TransformerConfig(guidance_scale=1.0))
    )

    image = torch.zeros(1, 1, 1, 3, 4, 4)

    first_cache = pipeline.initialize_cache(text=["first prompt"], image=image)
    second_cache = pipeline.initialize_cache(text=["second prompt"], image=image)

    assert first_cache.image is image
    assert second_cache.image is image
    assert text_encoder.calls == [["first prompt"], ["second prompt"]]
    assert captured_contexts[0]["height"] == 2
    assert captured_contexts[0]["width"] == 2
    assert captured_contexts[1]["height"] == 2
    assert captured_contexts[1]["width"] == 2
