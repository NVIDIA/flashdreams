# SPDX-FileCopyrightText: Copyright (c) 2026 SwiftVR Authors.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# This file has been modified by NVIDIA CORPORATION & AFFILIATES.

"""SwiftVR restoration-aware decoder network."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from flashdreams.recipes.taehv.checkpoint import legacy_to_blocks_keys
from flashdreams.recipes.taehv.impl import TAEHV, TGrow


class SwiftVRTemporalGrow(nn.Module):
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


class SwiftVRTAEHV(TAEHV):
    """Shared TAEHV configured for SwiftVR's ReAE checkpoint."""

    def __init__(self, checkpoint_path: str | None) -> None:
        super().__init__(
            checkpoint_path=None,
            model_type="wan22",
            channels=(512, 256, 128, 64),
            use_cuda_graph=False,
            use_compile=False,
        )
        with torch.device("meta"):
            for index, block in enumerate(self.decoder.blocks):
                if isinstance(block, TGrow):
                    self.decoder.blocks[index] = SwiftVRTemporalGrow(
                        int(block.conv.in_channels), block.stride
                    )

        if checkpoint_path is not None:
            self.load_from_checkpoint(
                checkpoint_path,
                state_dict_transform=legacy_to_blocks_keys,
            )


__all__ = ["SwiftVRTAEHV", "SwiftVRTemporalGrow"]
