# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import torch
from PIL import Image

if TYPE_CHECKING:
    from diffusers import HeliosPyramidPipeline


@dataclass
class HeliosConditionings:
    """Conditioning tensors computed once and reused across AR steps."""

    prompt_embeds: torch.Tensor
    negative_prompt_embeds: torch.Tensor | None
    image: Optional[Image.Image] = None


class HeliosEncoder:
    """Thin wrapper around Helios T5 text encoder."""

    def __init__(self, pipe: HeliosPyramidPipeline) -> None:
        self.pipe = pipe

    @torch.no_grad()
    def encode(
        self,
        prompt: list[str],
        negative_prompt: list[str],
        device: torch.device,
        *,
        guidance_scale: float,
        image: Optional[Image.Image] = None,
    ) -> HeliosConditionings:
        do_cfg = guidance_scale > 1.0 and not getattr(
            self.pipe.config, "is_distilled", False
        )
        pe, npe = self.pipe.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            do_classifier_free_guidance=do_cfg,
            device=device,
        )
        return HeliosConditionings(
            prompt_embeds=pe,
            negative_prompt_embeds=npe if do_cfg else None,
            image=image,
        )
