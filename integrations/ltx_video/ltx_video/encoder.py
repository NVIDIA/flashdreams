# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import torch
from PIL import Image

if TYPE_CHECKING:
    from diffusers import LTXPipeline


@dataclass
class LTXConditionings:
    """Conditioning tensors computed once and reused across AR steps."""

    prompt_embeds: torch.Tensor
    prompt_attention_mask: torch.Tensor
    negative_prompt_embeds: torch.Tensor
    negative_prompt_attention_mask: torch.Tensor
    image_latents: Optional[torch.Tensor] = None


class LTXEncoder:
    """Thin wrapper around LTX T5 text encoder and VAE image encoder."""

    def __init__(self, pipe: LTXPipeline) -> None:
        self.pipe = pipe

    @staticmethod
    def _spatial_scale(pipe: LTXPipeline) -> int:
        return int(
            getattr(pipe, "vae_spatial_compression_ratio", None)
            or getattr(pipe.vae, "spatial_compression_ratio", 8)
        )

    @torch.no_grad()
    def encode(
        self,
        prompt: list[str],
        negative_prompt: list[str],
        device: torch.device,
        image: Optional[Image.Image] = None,
    ) -> LTXConditionings:
        pe, pm, npe, npm = self.pipe.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            device=device,
            do_classifier_free_guidance=True,
        )

        image_latents = None
        if image is not None:
            from diffusers.image_processor import VaeImageProcessor

            proc = VaeImageProcessor(vae_scale_factor=self._spatial_scale(self.pipe))
            img_t = proc.preprocess(image).to(device=device, dtype=torch.bfloat16)
            latent = self.pipe.vae.encode(img_t).latent_dist.sample()
            latent = latent * self.pipe.vae.config.scaling_factor
            image_latents = latent.unsqueeze(2)

        return LTXConditionings(pe, pm, npe, npm, image_latents)
