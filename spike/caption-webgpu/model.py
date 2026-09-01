# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# THROWAWAY spike (caption WebGPU de-risk) -- NOT product code. Delete before any PR.
#
# Candidate latent -> caption-bank classifier for the video-token caption path.
# Deliberately WebGPU-op-friendly: per-frame 2D convolutions over the latent grid
# + temporal mean pooling + a small MLP head. No 3D conv (limited onnxruntime-web
# WebGPU coverage). Input is a window of latent frames as the browser assembles
# them from the token stream; output is caption-bank logits (single forward).

from __future__ import annotations

import torch
from torch import nn


class CaptionBankClassifier(nn.Module):
    """latent window [B, Twin, C, H, W] -> caption-bank logits [B, num_captions]."""

    def __init__(
        self,
        latent_channels: int = 16,
        num_captions: int = 32,
        hidden: int = 256,
    ) -> None:
        super().__init__()
        c = (32, 64, 96, 128)
        # Per-frame 2D conv stack (stride-2 downsampling of the latent grid).
        self.features = nn.Sequential(
            nn.Conv2d(latent_channels, c[0], 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c[0], c[1], 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c[1], c[2], 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c[2], c[3], 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)  # -> GlobalAveragePool in ONNX
        self.head = nn.Sequential(
            nn.Linear(c[3], hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, num_captions),
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        b, t, c, h, w = latent.shape
        x = latent.reshape(b * t, c, h, w)  # per-frame
        x = self.features(x)
        x = self.pool(x).flatten(1)  # [B*T, C']
        x = x.reshape(b, t, -1).mean(dim=1)  # temporal mean -> [B, C']
        return self.head(x)  # [B, num_captions]
