# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Action and frame-zero encoders for Cosmos Interactive hand inference."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
from torch import Tensor

from flashdreams.infra.encoder import (
    EncoderConfig,
    StreamingEncoder,
    StreamingEncoderCache,
)
from flashdreams.recipes.wan.autoencoder.vae import Wan22TI2V5BVAEEncoderConfig

from .. import action_format as _agibot
from ..patchify import pad_latent, patchify_latent
from ..spec import SPEC


@dataclass(frozen=True)
class HandActionInput:
    """Normalized action rows with aligned camera or AgiBot domain metadata."""

    actions: Tensor
    """Normalized, zero-padded action values with shape ``[N,64]``."""

    domain_ids: Tensor
    """Int64 domain IDs with shape ``[N]``: camera 2 or AgiBot 15."""


@dataclass(kw_only=True)
class HandActionEncoderConfig(EncoderConfig):
    """Configuration for normalized, zero-padded AgiBot actions."""

    _target: type["HandActionEncoder"] = field(
        default_factory=lambda: HandActionEncoder
    )
    action_dim: int = _agibot.MAX_ACTION_DIM


class HandActionEncoder(StreamingEncoder[StreamingEncoderCache]):
    """Per-step encoder for normalized, zero-padded AgiBot actions."""

    def __init__(self, config: HandActionEncoderConfig) -> None:
        super().__init__(config)
        self.config: HandActionEncoderConfig = config

    def initialize_autoregressive_cache(self) -> StreamingEncoderCache:
        return StreamingEncoderCache()

    def forward(
        self,
        input: Tensor | HandActionInput,
        autoregressive_index: int = 0,
        cache: StreamingEncoderCache | None = None,
    ) -> HandActionInput:
        del autoregressive_index, cache
        actions = input.actions if isinstance(input, HandActionInput) else input
        if not isinstance(actions, Tensor):
            raise TypeError(f"hand actions must be a Tensor, got {type(actions)}")
        expected = self.config.action_dim
        if actions.ndim != 2 or actions.shape[0] == 0 or actions.shape[1] != expected:
            raise ValueError(
                f"hand actions must have shape [N,{expected}], got {tuple(actions.shape)}"
            )
        domain_ids = (
            input.domain_ids
            if isinstance(input, HandActionInput)
            else torch.full(
                (actions.shape[0],),
                _agibot.AGIBOT_DOMAIN_ID,
                dtype=torch.long,
                device=actions.device,
            )
        )
        if not isinstance(domain_ids, Tensor) or domain_ids.dtype != torch.long:
            raise TypeError("hand action domain_ids must be an int64 Tensor")
        if domain_ids.shape != (actions.shape[0],):
            raise ValueError(
                "hand action domain_ids must have shape [N], matching actions"
            )
        if not bool(
            (
                (domain_ids == _agibot.CAMERA_DOMAIN_ID)
                | (domain_ids == _agibot.AGIBOT_DOMAIN_ID)
            ).all()
        ):
            raise ValueError(
                "hand action domain_ids must contain only camera 2 or AgiBot 15"
            )
        return HandActionInput(
            actions.to(dtype=torch.float32).contiguous(),
            domain_ids.to(device=actions.device).contiguous(),
        )


class HandSceneEncoder(nn.Module):
    """Encode and patchify a scene's first frame with FlashDreams' Wan 2.2 VAE."""

    def __init__(self, *, device: str | torch.device) -> None:
        super().__init__()
        self.device = torch.device(device)
        config = Wan22TI2V5BVAEEncoderConfig(
            dtype=torch.bfloat16,
            use_cuda_graph=False,
        )
        self.vae = config.setup().to(self.device).eval()

    @torch.no_grad()
    def encode(self, image: Tensor) -> Tensor:
        """Encode one RGB ``HWC``/``CHW`` image to normalized ``[C,1,H,W]``."""
        if image.ndim != 3:
            raise ValueError(
                f"scene image must have three dimensions, got {image.shape}"
            )
        if image.shape[-1] == 3:
            image = image.permute(2, 0, 1)
        if image.shape[0] != 3:
            raise ValueError(
                f"scene image must contain three RGB channels, got {image.shape}"
            )
        if image.dtype == torch.uint8:
            image = image.float().div(127.5).sub(1.0)
        image = image.to(device=self.device, dtype=torch.bfloat16)
        latent = self.vae(image[None, None])
        return latent[0].transpose(0, 1).contiguous()

    def patchify_encoded(
        self,
        latent: Tensor,
        *,
        content_height: int,
        content_width: int,
    ) -> Tensor:
        """Crop a VAE latent to content geometry and patch it for the MoT."""
        logical_h, logical_w = _agibot.logical_latent_geometry(
            content_height,
            content_width,
            expected=(_agibot.LATENT_HEIGHT, _agibot.LATENT_WIDTH),
        )
        latent = latent[..., :logical_h, :logical_w].contiguous()
        patch = SPEC.vfm.latent_patch_size
        grid_h, grid_w = _agibot.TOKEN_GRID
        latent = pad_latent(
            latent,
            bottom=grid_h * patch - logical_h,
            right=grid_w * patch - logical_w,
        )
        return patchify_latent(latent, patch)


__all__ = [
    "HandActionEncoder",
    "HandActionEncoderConfig",
    "HandActionInput",
    "HandSceneEncoder",
]
