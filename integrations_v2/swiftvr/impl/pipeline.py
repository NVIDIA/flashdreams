# SPDX-FileCopyrightText: Copyright (c) 2026 SwiftVR Authors.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# This file has been modified by NVIDIA CORPORATION & AFFILIATES.

"""SwiftVR encoder-transformer-decoder streaming pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypeAlias, cast

import torch
from safetensors.torch import load_file
from torch import Tensor

from flashdreams.infra.pipeline import (
    StreamInferencePipeline,
    StreamInferencePipelineCache,
    StreamInferencePipelineConfig,
)
from swiftvr.impl.decoder import (
    SwiftVRDecoder,
    SwiftVRDecoderCache,
    SwiftVRDecoderConfig,
)
from swiftvr.impl.encoder import (
    SwiftVREncoder,
    SwiftVREncoderCache,
    SwiftVREncoderConfig,
)
from swiftvr.impl.transformer import (
    SwiftVRTransformer,
    SwiftVRTransformerCache,
)

SwiftVRPipelineCache: TypeAlias = StreamInferencePipelineCache[
    SwiftVREncoderCache,
    SwiftVRTransformerCache,
    SwiftVRDecoderCache,
]


@dataclass(kw_only=True)
class SwiftVRPipelineConfig(StreamInferencePipelineConfig):
    """Configuration for :class:`SwiftVRPipeline`."""

    _target: type["SwiftVRPipeline"] = field(  # type: ignore[assignment]
        default_factory=lambda: SwiftVRPipeline
    )

    encoder: SwiftVREncoderConfig = field(  # type: ignore[assignment]
        default_factory=SwiftVREncoderConfig
    )
    decoder: SwiftVRDecoderConfig = field(  # type: ignore[assignment]
        default_factory=SwiftVRDecoderConfig
    )
    prompt_path: str | None = None
    """Path to the frozen SwiftVR prompt embedding."""


class SwiftVRPipeline(
    StreamInferencePipeline[
        SwiftVREncoderCache,
        SwiftVRTransformerCache,
        SwiftVRDecoderCache,
    ]
):
    """SwiftVR streaming video-restoration pipeline."""

    encoder: SwiftVREncoder
    decoder: SwiftVRDecoder

    def __init__(self, config: SwiftVRPipelineConfig) -> None:
        super().__init__(config)
        self.config: SwiftVRPipelineConfig = config
        transformer = self.diffusion_model.transformer
        if not isinstance(transformer, SwiftVRTransformer):
            raise TypeError(
                "SwiftVRPipeline requires SwiftVRTransformer, got "
                f"{type(transformer).__name__}."
            )
        if not isinstance(self.encoder, SwiftVREncoder):
            raise TypeError(
                f"SwiftVRPipeline requires SwiftVREncoder, got {type(self.encoder).__name__}."
            )
        if not isinstance(self.decoder, SwiftVRDecoder):
            raise TypeError(
                f"SwiftVRPipeline requires SwiftVRDecoder, got {type(self.decoder).__name__}."
            )
        prompt = (
            load_file(config.prompt_path, device="cpu")["prompt_emb"][0]
            if config.prompt_path is not None
            else None
        )
        self.register_buffer("prompt_embedding", prompt, persistent=False)

    @property
    def transformer(self) -> SwiftVRTransformer:
        """Return the concrete SwiftVR transformer."""
        return cast(SwiftVRTransformer, self.diffusion_model.transformer)

    @torch.no_grad()
    def initialize_cache(  # type: ignore[override]
        self,
        *,
        output_height: int,
        output_width: int,
        overlap: int,
        prompt_embedding: Tensor | None = None,
    ) -> SwiftVRPipelineCache:
        """Build isolated encoder, transformer, and decoder stream caches."""
        prompt = (
            prompt_embedding if prompt_embedding is not None else self.prompt_embedding
        )
        if prompt is None:
            raise ValueError(
                "SwiftVRPipeline.initialize_cache requires prompt_embedding or "
                "SwiftVRPipelineConfig.prompt_path."
            )
        padded_height = output_height + (-output_height) % 32
        padded_width = output_width + (-output_width) % 32
        latent_height = padded_height // self.encoder.spatial_compression_ratio
        latent_width = padded_width // self.encoder.spatial_compression_ratio
        return super().initialize_cache(
            transformer_context={
                "height": latent_height,
                "width": latent_width,
                "prompt_embedding": prompt,
                "overlap": overlap,
            },
            encoder_context={
                "output_height": output_height,
                "output_width": output_width,
            },
            decoder_context={
                "output_height": output_height,
                "output_width": output_width,
            },
        )

    @torch.no_grad()
    def generate(  # type: ignore[override]
        self,
        autoregressive_index: int,
        cache: SwiftVRPipelineCache,
        input: Tensor,
    ) -> Tensor | None:
        """Run one input chunk through encoder, transformer, and decoder."""
        previous = cache.autoregressive_index
        expected = previous + 1 if previous is not None else 0
        if autoregressive_index != expected:
            raise AssertionError(
                f"AR step out of order: previous step was {previous}, expected "
                f"{expected}, got {autoregressive_index}"
            )
        cache.autoregressive_index = autoregressive_index
        cache.transformer_cache.start(autoregressive_index)
        assert cache.encoder_cache is not None
        latents = self.encoder(
            input=input,
            autoregressive_index=autoregressive_index,
            cache=cache.encoder_cache,
        )
        if latents is None:
            return None
        return self._restore_and_decode(latents, autoregressive_index, cache)

    @torch.no_grad()
    def finalize(  # type: ignore[override]
        self,
        autoregressive_index: int,
        cache: SwiftVRPipelineCache,
    ) -> None:
        """Close one public pipeline step without unused WAN self-KV updates."""
        if cache.autoregressive_index != autoregressive_index:
            raise AssertionError(
                "autoregressive_index mismatch: generate() ran with "
                f"{cache.autoregressive_index} but finalize() received "
                f"{autoregressive_index}."
            )
        cache.transformer_cache.finalize(autoregressive_index)

    @torch.no_grad()
    def flush(self, cache: SwiftVRPipelineCache) -> Tensor | None:
        """Flush the encoder's final partial group through the remaining stages."""
        assert cache.encoder_cache is not None
        latents = self.encoder.flush(cache.encoder_cache)
        if latents is None:
            return None
        autoregressive_index = cache.autoregressive_index or 0
        return self._restore_and_decode(latents, autoregressive_index, cache)

    def _restore_and_decode(
        self,
        latents: Tensor,
        autoregressive_index: int,
        cache: SwiftVRPipelineCache,
    ) -> Tensor | None:
        restored = self.transformer.restore(
            latents.permute(0, 2, 1, 3, 4).contiguous(),
            cache.transformer_cache,
        )
        assert cache.decoder_cache is not None
        return self.decoder(
            input=restored,
            autoregressive_index=autoregressive_index,
            cache=cache.decoder_cache,
        )

    @classmethod
    def from_pretrained(
        cls,
        checkpoint: str,
        *,
        revision: str | None,
        device: str,
        dtype: torch.dtype,
        attention_window: tuple[int, int],
        compile_blocks: bool,
        chunk_size: int = 8,
    ) -> "SwiftVRPipeline":
        """Resolve a checkpoint and construct the configured pipeline."""
        resolved_device = torch.device(device)
        if resolved_device.type != "cuda":
            raise ValueError("SwiftVR's FlashDreams WAN runtime requires CUDA.")
        from swiftvr.config import build_swiftvr_pipeline

        pipeline = build_swiftvr_pipeline(
            checkpoint=checkpoint,
            revision=revision,
            dtype=dtype,
            attention_window=attention_window,
            compile_blocks=compile_blocks,
            chunk_size=chunk_size,
        ).setup()
        assert isinstance(pipeline, cls)
        pipeline.to(device=resolved_device, dtype=dtype).eval()
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        return pipeline


__all__ = [
    "SwiftVRPipeline",
    "SwiftVRPipelineCache",
    "SwiftVRPipelineConfig",
]
