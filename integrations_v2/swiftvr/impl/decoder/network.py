# SPDX-FileCopyrightText: Copyright (c) 2026 SwiftVR Authors.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# This file has been modified by NVIDIA CORPORATION & AFFILIATES.

"""SwiftVR restoration-aware decoder network."""

from __future__ import annotations

from typing import cast

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from flashdreams.infra.compile import compile_module
from flashdreams.infra.cuda_graph import set_or_copy
from flashdreams.recipes.taehv.checkpoint import legacy_to_blocks_keys
from flashdreams.recipes.taehv.impl import TAEHV, Decoder, MemBlock, TGrow


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


def _memblock_step_channels_last(
    block: MemBlock,
    tensor: Tensor,
    state: dict[int, Tensor],
    batch: int,
) -> Tensor:
    key = id(block)
    bt, channels, height, width = tensor.shape
    time_steps = bt // batch
    video = tensor.reshape(batch, time_steps, channels, height, width)
    past = torch.cat([state[key], video[:, :-1]], dim=1).reshape_as(tensor)
    set_or_copy(state, key, video[:, -1:])
    return block(
        tensor.contiguous(memory_format=torch.channels_last),
        past.contiguous(memory_format=torch.channels_last),
    )


def _temporal_grow_channels_last_3d(
    block: SwiftVRTemporalGrow, tensor: Tensor
) -> Tensor:
    if block.stride == 1:
        assert block.proj is not None
        return block.proj(tensor)
    assert block.conv3d is not None
    frames, channels, height, width = tensor.shape
    tensor = F.interpolate(
        tensor.unsqueeze(2),
        size=(block.stride, height, width),
        mode="nearest",
    ).contiguous(memory_format=torch.channels_last_3d)
    tensor = block.conv3d(tensor)
    return tensor.permute(0, 2, 1, 3, 4).reshape(
        frames * block.stride, channels, height, width
    )


class _SwiftVRCompiledDecoder(nn.Module):
    """SwiftVR decoder compute with compiler-friendly convolution layouts."""

    def __init__(
        self,
        decoder: Decoder,
        *,
        channels_last: bool = True,
        channels_last_3d: bool = True,
    ) -> None:
        super().__init__()
        self.decoder = decoder
        self.channels_last = channels_last
        self.channels_last_3d = channels_last_3d

        temporal_convs = {
            id(child)
            for block in decoder.blocks
            if isinstance(block, SwiftVRTemporalGrow)
            for child in block.modules()
            if isinstance(child, nn.Conv2d)
        }
        if channels_last:
            for child in decoder.modules():
                if isinstance(child, nn.Conv2d) and id(child) not in temporal_convs:
                    child.weight.data = child.weight.data.contiguous(
                        memory_format=torch.channels_last
                    )
        if channels_last_3d:
            for child in decoder.modules():
                if isinstance(child, nn.Conv3d):
                    child.weight.data = child.weight.data.contiguous(
                        memory_format=torch.channels_last_3d
                    )

    @torch.no_grad()
    def initialize_state(
        self,
        z_shape: tuple[int, int, int, int, int],
        dtype: torch.dtype,
        device: torch.device,
        state: dict[int, Tensor],
    ) -> None:
        """Initialize causal state through the wrapped decoder."""
        self.decoder.initialize_state(z_shape, dtype, device, state)

    def forward(self, tensor: Tensor, state: dict[int, Tensor], batch: int) -> Tensor:
        """Decode a latent chunk while preserving convolution memory formats."""
        _, time_steps, channels, height, width = tensor.shape
        tensor = tensor.reshape(batch * time_steps, channels, height, width)
        if self.channels_last:
            tensor = tensor.contiguous(memory_format=torch.channels_last)
        for block in self.decoder.blocks:
            if isinstance(block, MemBlock) and self.channels_last:
                tensor = _memblock_step_channels_last(block, tensor, state, batch)
            elif isinstance(block, MemBlock):
                tensor = block.cache_step(tensor, state, batch)
            elif isinstance(block, SwiftVRTemporalGrow):
                if self.channels_last_3d:
                    tensor = _temporal_grow_channels_last_3d(block, tensor.contiguous())
                else:
                    tensor = block(tensor.contiguous())
                if self.channels_last:
                    tensor = tensor.contiguous(memory_format=torch.channels_last)
            else:
                tensor = block(tensor)
        tensor = tensor.contiguous()
        _, channels, height, width = tensor.shape
        return tensor.reshape(batch, -1, channels, height, width)


class SwiftVRTAEHV(TAEHV):
    """Shared TAEHV configured for SwiftVR's ReAE checkpoint."""

    def __init__(
        self,
        checkpoint_path: str | None,
        *,
        use_compile: bool = False,
    ) -> None:
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
            if use_compile:
                self.decoder = cast(
                    Decoder,
                    compile_module(_SwiftVRCompiledDecoder(self.decoder)),
                )


__all__ = ["SwiftVRTAEHV", "SwiftVRTemporalGrow"]
