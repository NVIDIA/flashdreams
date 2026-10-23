# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stateful latent decoder for Cosmos Interactive hand video."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, cast

import torch
import torch.nn as nn
from torch import Tensor

from flashdreams.infra.decoder import (
    DecoderConfig,
    StreamingDecoderCache,
    StreamingVideoDecoder,
)
from flashdreams.recipes.taehv.impl import TAEHV, TAEHVCache
from flashdreams.recipes.wan.autoencoder.vae import (
    Wan22TI2V5BVAEDecoderConfig,
    WanVAECache,
    WanVAEDecoder,
)

from .. import action_format as _agibot
from ..patchify import crop_latent, unpatchify_latent
from ..spec import SPEC
from .network import (
    TAEHV2SuperCache,
    TAEHV2SuperDecoder,
    make_taehv2super_decoder,
)

_TAEW2_2_CHECKPOINT = (
    "https://huggingface.co/lightx2v/Autoencoders/resolve/main/taew2_2.pth"
)

DecoderBackendCache = WanVAECache | TAEHVCache | TAEHV2SuperCache
DecoderBackend = Literal["wan", "taehv", "taehv2super"]
DECODER_BACKENDS = ("wan", "taehv", "taehv2super")


@dataclass(kw_only=True)
class HandDecoderCache(StreamingDecoderCache):
    """Per-scene decoder state, primed with that scene's frame zero."""

    backend_cache: DecoderBackendCache
    autoregressive_index: int = 0


@dataclass(kw_only=True)
class HandDecoderConfig(DecoderConfig):
    """Config for the checkpoint-compatible streaming decoder."""

    _target: type["HandDecoder"] = field(default_factory=lambda: HandDecoder)
    backend: DecoderBackend = "wan"
    device: str = "cuda"
    credential_path: str | None = None


class HandDecoder(StreamingVideoDecoder[HandDecoderCache]):
    """Decode generated latents while preserving the causal decoder history."""

    def __init__(self, config: HandDecoderConfig) -> None:
        super().__init__(config)
        self.config: HandDecoderConfig = config
        if config.backend not in DECODER_BACKENDS:
            raise ValueError(f"unknown decoder {config.backend!r}")
        self.backend = self._setup_backend()

    @property
    def spatial_compression_ratio(self) -> int:
        return _agibot.SPATIAL_COMPRESSION_FACTOR

    @property
    def temporal_compression_ratio(self) -> int:
        return _agibot.TEMPORAL_COMPRESSION_FACTOR

    def get_output_temporal_size(
        self,
        autoregressive_index: int,
        input_temporal_size: int,
    ) -> int:
        del autoregressive_index
        return int(input_temporal_size) * self.temporal_compression_ratio

    def get_input_temporal_size(
        self,
        autoregressive_index: int,
        output_temporal_size: int,
    ) -> int:
        del autoregressive_index
        ratio = self.temporal_compression_ratio
        if output_temporal_size % ratio:
            raise ValueError(
                f"output_temporal_size must be divisible by {ratio}, "
                f"got {output_temporal_size}"
            )
        return int(output_temporal_size) // ratio

    def _setup_backend(self) -> nn.Module:
        if self.config.backend == "wan":
            backend = Wan22TI2V5BVAEDecoderConfig(
                dtype=torch.bfloat16,
                use_cuda_graph=False,
                use_compile=False,
            ).setup()
            return backend.to(self.config.device).eval().requires_grad_(False)
        if self.config.backend == "taehv":
            backend = TAEHV(
                checkpoint_path=_TAEW2_2_CHECKPOINT,
                model_type="wan22",
                use_cuda_graph=False,
                use_compile=False,
            )
            return (
                backend.to(device=self.config.device, dtype=torch.bfloat16)
                .eval()
                .requires_grad_(False)
            )
        return make_taehv2super_decoder(
            device=self.config.device,
            credential_path=self.config.credential_path,
        )

    def _new_backend_cache(self) -> DecoderBackendCache:
        if self.config.backend == "wan":
            return cast(WanVAEDecoder, self.backend).initialize_autoregressive_cache()
        if self.config.backend == "taehv":
            return cast(TAEHV, self.backend).prepare_cache()
        return cast(TAEHV2SuperDecoder, self.backend).prepare_cache()

    def _decode_chunk(self, latent: Tensor, cache: HandDecoderCache) -> Tensor:
        parameter = next(self.backend.parameters())
        device, dtype = parameter.device, parameter.dtype
        if self.config.backend == "taehv2super":
            pixels = cast(TAEHV2SuperDecoder, self.backend)(
                latent.unsqueeze(0).to(device=device, dtype=dtype),
                cache=cast(TAEHV2SuperCache, cache.backend_cache),
            )
            cache.autoregressive_index += int(latent.shape[1])
            return pixels[0].permute(1, 0, 2, 3).mul(2).sub(1).float()

        outputs: list[Tensor] = []
        for start in range(int(latent.shape[1])):
            frame = latent[:, start : start + 1]
            if self.config.backend == "wan":
                pixels = cast(WanVAEDecoder, self.backend)(
                    frame.permute(1, 0, 2, 3).to(device=device, dtype=dtype),
                    autoregressive_index=cache.autoregressive_index,
                    cache=cast(WanVAECache, cache.backend_cache),
                )
            else:
                pixels = (
                    cast(TAEHV, self.backend)
                    .decode(
                        frame.permute(1, 0, 2, 3)
                        .unsqueeze(0)
                        .to(device=device, dtype=dtype),
                        cache=cast(TAEHVCache, cache.backend_cache),
                    )
                    .mul(2)
                    .sub(1)[0]
                )
            outputs.append(pixels.float())
            cache.autoregressive_index += 1
        return torch.cat(outputs, dim=0)

    def initialize_autoregressive_cache(self, *, scene: Any) -> HandDecoderCache:
        """Reset decoder history and consume the scene's already-present frame zero."""
        cache = HandDecoderCache(backend_cache=self._new_backend_cache())
        seed = unpatchify_latent(
            scene.init_latent.to(self.config.device),
            channels=SPEC.vfm.vae_latent_channels,
            grid_t=1,
            grid_h=_agibot.TOKEN_GRID[0],
            grid_w=_agibot.TOKEN_GRID[1],
            patch_spatial=SPEC.vfm.latent_patch_size,
        )
        self._decode_chunk(self._crop_latent(seed), cache)
        return cache

    def _crop_latent(self, latent: Tensor) -> Tensor:
        return crop_latent(
            latent,
            height=_agibot.LATENT_HEIGHT,
            width=_agibot.LATENT_WIDTH,
        )

    @staticmethod
    def _crop_pixels(pixels: Tensor) -> Tensor:
        return pixels[..., : _agibot.OUTPUT_HEIGHT, : _agibot.OUTPUT_WIDTH].contiguous()

    def forward(
        self,
        input: Tensor,
        autoregressive_index: int = 0,
        cache: HandDecoderCache | None = None,
    ) -> Tensor:
        del autoregressive_index
        if cache is None:
            raise ValueError("HandDecoder requires a scene-initialized decoder cache")
        if input.ndim != 5 or input.shape[0] != 1:
            raise ValueError(
                "HandDecoder requires one [N,C,T,H,W] latent batch, got "
                f"{tuple(input.shape)}"
            )
        output = self._crop_pixels(
            self._decode_chunk(self._crop_latent(input[0]), cache)
        )
        expected = int(input.shape[2]) * self.temporal_compression_ratio
        if output.ndim != 4 or int(output.shape[0]) != expected:
            raise ValueError(
                "hand decoder broke temporal alignment: expected "
                f"[T={expected},C,H,W], got {tuple(output.shape)}"
            )
        return output.contiguous()


__all__ = [
    "DECODER_BACKENDS",
    "HandDecoder",
    "HandDecoderCache",
    "HandDecoderConfig",
]
