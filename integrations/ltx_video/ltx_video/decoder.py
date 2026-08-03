# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Causal VAE decode and optional TAEHV fast decoder."""

from __future__ import annotations

from typing import Any

import torch


class LTXDecoder:
    """Decode latent chunks to pixel frames."""

    def __init__(self, vae: Any, *, use_taehv: bool = False) -> None:
        self.vae = vae
        self.use_taehv = use_taehv
        self._taehv: Any = None
        if use_taehv:
            self._load_taehv()

    def _load_taehv(self) -> None:
        try:
            from diffusers import AutoencoderTiny

            device = next(self.vae.parameters()).device
            self._taehv = AutoencoderTiny.from_pretrained(
                "madebyollin/taehv",
                torch_dtype=torch.bfloat16,
            ).to(device)
            print("[LTX decoder] TAEHV fast decoder loaded")
        except Exception as exc:
            print(f"[LTX decoder] TAEHV load failed ({exc}), falling back to standard VAE")
            self.use_taehv = False

    @torch.no_grad()
    def decode_chunk(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode ``[B, C, T, H, W]`` latents to ``[B, T, C, H, W]`` in ``[0, 1]``."""
        latents = latents / self.vae.config.scaling_factor
        return self._decode_vae_output(latents)

    @torch.no_grad()
    def decode_from_denoised(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode unpacked latents after diffusers ``_denormalize_latents``."""
        latents = latents.to(self.vae.dtype)
        return self._decode_vae_output(latents)

    def _decode_vae_output(self, latents: torch.Tensor) -> torch.Tensor:
        if self.use_taehv and self._taehv is not None:
            frames = self._taehv.decode(latents).sample
        else:
            frames = self.vae.decode(latents, return_dict=False)[0]

        frames = (frames / 2 + 0.5).clamp(0, 1)
        return frames.permute(0, 2, 1, 3, 4).float()
