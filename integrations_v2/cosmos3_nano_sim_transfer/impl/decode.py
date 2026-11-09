# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Latents out: decode the rollout's clean latents a latent at a time.

The rollout's latents are in the Wan 2.2 VAE's normalized latent space, which
both decoders here read directly:

- ``wan2.2``: the Wan 2.2 VAE decoder, the one cosmos-framework decodes with
  (reference quality);
- ``taehv-super``: TAEHV-super, a small streaming decoder that approximates
  it at a fraction of the cost.

Both are causal and keep their own temporal cache: the first latent of a clip
decodes to one pixel frame, every later latent to four. :meth:`LatentDecoder.reset`
starts a new clip.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import torch
from torch import Tensor

DECODERS = ("wan2.2", "taehv-super")
"""Decoder kinds :class:`LatentDecoder` accepts."""


def to_uint8(pixels: Tensor) -> Tensor:
    """``[-1, 1]`` pixels to uint8 ``[0, 255]``, rounded to the nearest level."""
    return ((pixels.float().clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8)


class LatentDecoder:
    """Streaming decoder for the rollout's clean latents, one latent per call.

    Args:
        kind: One of :data:`DECODERS`.
        checkpoint_path: Decoder weights: the Wan 2.2 VAE (``Wan2.2_VAE.pth``)
            or the TAEHV-super safetensors; the FlashDreams defaults (fetched
            on first use) when ``None``.
        device: Where to run the decoder.
        dtype: Decoder dtype.
    """

    def __init__(
        self,
        kind: str = "wan2.2",
        *,
        checkpoint_path: str | None = None,
        device: torch.device | str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        if kind not in DECODERS:
            raise ValueError(f"unknown decoder {kind!r}; use one of {DECODERS}")
        self.kind = kind
        self.device = torch.device(device)
        self.dtype = dtype
        if kind == "wan2.2":
            from flashdreams.recipes.wan.autoencoder.vae import (
                Wan22TI2V5BVAEDecoderConfig,
            )

            # No CUDA graph: the first call takes one frame's latent and the
            # rest another shape, and a graph would pin the first.
            config = Wan22TI2V5BVAEDecoderConfig(dtype=dtype, use_cuda_graph=False)
            if checkpoint_path is not None:
                config = replace(config, checkpoint_path=checkpoint_path)
            self.model: Any = config.setup().to(self.device)
        else:
            from cosmos_interactive._taehv_super import SuperTAEHV
            from cosmos_interactive._vae import TAEW2_2_SUPER_CHECKPOINT

            self.model = (
                SuperTAEHV(
                    checkpoint_path=checkpoint_path or TAEW2_2_SUPER_CHECKPOINT,
                    use_cuda_graph=False,
                )
                .to(device=self.device, dtype=dtype)
                .eval()
            )
        self.reset()

    def reset(self) -> None:
        """Forget the previous clip."""
        if self.kind == "wan2.2":
            self._cache = self.model.initialize_autoregressive_cache()
        else:
            self._cache = self.model.prepare_cache()
        self._latents = 0

    @torch.no_grad()
    def decode(self, latent: Tensor) -> Tensor:
        """Decode the next ``[C, 1, h, w]`` latent to ``[T, 3, 16h, 16w]`` in ``[-1, 1]``.

        ``T`` is 1 for the first latent of a clip and 4 after it.
        """
        if latent.ndim != 4 or latent.shape[1] != 1:
            raise ValueError(
                f"expected a [C, 1, h, w] latent, got {tuple(latent.shape)}"
            )
        z = latent.to(device=self.device, dtype=self.dtype).transpose(0, 1)[None]
        if self.kind == "wan2.2":
            pixels = self.model(
                z, autoregressive_index=self._latents, cache=self._cache
            )
        else:
            # TAEHV emits [0, 1] and trims its warm-up frames on the first call.
            pixels = self.model.decode(z, cache=self._cache).mul(2).sub(1)
        self._latents += 1
        return pixels[0].float()
