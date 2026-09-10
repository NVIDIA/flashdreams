# SPDX-FileCopyrightText: Copyright (c) 2026 SwiftVR Authors.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SwiftVR restoration-aware decoder network."""

from __future__ import annotations

import torch.nn.functional as F
from torch import Tensor, nn

from swiftvr.impl.autoencoder import MemoryBlock, convolution


class _Clamp(nn.Module):
    def forward(self, tensor: Tensor) -> Tensor:
        """Soft-clamp autoencoder latents."""
        return (tensor / 3).tanh() * 3


class _TemporalGrow(nn.Module):
    """Expand the temporal axis through nearest interpolation and projection."""

    def __init__(self, channels: int, stride: int) -> None:
        super().__init__()
        self.stride = stride
        self.proj = (
            nn.Conv2d(channels, channels, 1, bias=False) if stride == 1 else None
        )
        self.conv3d = (
            nn.Conv3d(
                channels,
                channels,
                kernel_size=(3, 1, 1),
                padding=(1, 0, 0),
                bias=False,
            )
            if stride != 1
            else None
        )

    def forward(self, tensor: Tensor) -> Tensor:
        """Grow one flattened frame batch."""
        if self.stride == 1:
            assert self.proj is not None
            return self.proj(tensor)
        assert self.conv3d is not None
        frames, channels, height, width = tensor.shape
        tensor = F.interpolate(
            tensor.unsqueeze(2),
            size=(self.stride, height, width),
            mode="nearest",
        )
        tensor = self.conv3d(tensor)
        return tensor.permute(0, 2, 1, 3, 4).reshape(
            frames * self.stride, channels, height, width
        )


class SwiftVRDecoderNetwork(nn.Sequential):
    """Decoder half of SwiftVR's restoration-aware autoencoder."""

    patch_size = 2
    frames_to_trim = 3

    def __init__(self) -> None:
        first, second, third, fourth = (512, 256, 128, 64)
        super().__init__(
            _Clamp(),
            convolution(48, first),
            nn.ReLU(inplace=True),
            *[MemoryBlock(first, first) for _ in range(3)],
            nn.Upsample(scale_factor=2),
            _TemporalGrow(first, 1),
            convolution(first, second, bias=False),
            *[MemoryBlock(second, second) for _ in range(3)],
            nn.Upsample(scale_factor=2),
            _TemporalGrow(second, 2),
            convolution(second, third, bias=False),
            *[MemoryBlock(third, third) for _ in range(3)],
            nn.Upsample(scale_factor=2),
            _TemporalGrow(third, 2),
            convolution(third, fourth, bias=False),
            nn.ReLU(inplace=True),
            convolution(fourth, 12),
        )


__all__ = ["SwiftVRDecoderNetwork"]
