# SPDX-FileCopyrightText: Copyright (c) 2026 SwiftVR Authors.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SwiftVR restoration-aware encoder network."""

from torch import nn

from swiftvr.impl.autoencoder import MemoryBlock, TemporalPool, convolution


class SwiftVREncoderNetwork(nn.Sequential):
    """Encoder half of SwiftVR's restoration-aware autoencoder."""

    patch_size = 2

    def __init__(self) -> None:
        channels = 64
        super().__init__(
            convolution(12, channels),
            nn.ReLU(inplace=True),
            TemporalPool(channels, 2),
            convolution(channels, channels, stride=2, bias=False),
            *[MemoryBlock(channels, channels) for _ in range(3)],
            TemporalPool(channels, 2),
            convolution(channels, channels, stride=2, bias=False),
            *[MemoryBlock(channels, channels) for _ in range(3)],
            TemporalPool(channels, 1),
            convolution(channels, channels, stride=2, bias=False),
            *[MemoryBlock(channels, channels) for _ in range(3)],
            convolution(channels, 48),
        )


__all__ = ["SwiftVREncoderNetwork"]
