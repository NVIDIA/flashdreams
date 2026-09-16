# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Latent-native caption CNN and the 6-class ego-motion bank.

The single source of truth for the model architecture (shared by training and
export) and the class order.

Ego-motion is a *temporal* signal: turning left vs right, and forward vs
reverse, differ only in how the scene moves between frames. So the window's
frames are stacked along the channel dimension (**early temporal fusion**) and a
2D conv reads frame-to-frame change at each spatial location — a per-frame conv
followed by a temporal *mean* would be motion-blind (mean pooling is order- and
mirror-invariant). Deliberately WebGPU-op-friendly for the browser: 2D conv +
global average pool + a small MLP head, no 3D conv, no attention. ~265K params.
"""

from __future__ import annotations

import torch
from torch import nn

#: Model output order. Index i of the logits corresponds to ``CAPTION_CLASSES[i]``.
CAPTION_CLASSES: tuple[str, ...] = (
    "driving_straight",
    "turning_left",
    "turning_right",
    "slowing_or_stopping",
    "stopped",
    "reversing",
)

#: Human-readable captions shown in the browser overlay (indexed like CLASSES).
CAPTION_BANK: tuple[str, ...] = (
    "Driving straight",
    "Turning left",
    "Turning right",
    "Slowing to a stop",
    "Stopped",
    "Reversing",
)


class CaptionBankClassifier(nn.Module):
    """Latent window ``[B, Twin, C, H, W]`` -> class logits ``[B, num_classes]``.

    The window's temporal mean is subtracted first so the static scene cancels
    and only motion survives (appearance-invariant → generalizes across
    sessions). The frames are then stacked into the channel dimension and run
    through a stride-2 2D conv stack (so it sees motion), a global average pool
    collapses space, and a small MLP (with dropout) maps to the class logits.
    Per-channel input normalization (buffers set from the training data)
    stabilizes optimization.
    """

    def __init__(
        self,
        latent_channels: int = 16,
        num_frames: int = 10,
        num_classes: int = 6,
        hidden: int = 256,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.register_buffer("in_mean", torch.zeros(latent_channels))
        self.register_buffer("in_std", torch.ones(latent_channels))
        c = (32, 64, 96, 128)
        # First conv consumes all Twin frames at once (early temporal fusion).
        self.features = nn.Sequential(
            nn.Conv2d(latent_channels * num_frames, c[0], 3, stride=2, padding=1),
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
            nn.Dropout(dropout),
            nn.Linear(c[3], hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def set_input_stats(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        """Set the per-channel input normalization buffers (from training data)."""
        self.in_mean.copy_(mean)
        self.in_std.copy_(std.clamp_min(1e-6))

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        b, t, c, h, w = latent.shape
        x = (latent - self.in_mean.view(1, 1, -1, 1, 1)) / self.in_std.view(
            1, 1, -1, 1, 1
        )
        # Remove the static scene (subtract the window's temporal mean) so the
        # model sees motion, not appearance -> generalizes across sessions.
        x = x - x.mean(dim=1, keepdim=True)
        x = x.reshape(b, t * c, h, w)  # early temporal fusion: frames -> channels
        x = self.features(x)
        x = self.pool(x).flatten(1)  # [B, C']
        return self.head(x)  # [B, num_classes]
