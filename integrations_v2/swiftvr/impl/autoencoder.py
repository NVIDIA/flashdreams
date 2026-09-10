# SPDX-FileCopyrightText: Copyright (c) 2026 SwiftVR Authors.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared SwiftVR restoration-autoencoder building blocks."""

from __future__ import annotations

from typing import Any, Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def convolution(input_channels: int, output_channels: int, **kwargs: Any) -> nn.Conv2d:
    """Build the 3x3 convolution used throughout the ReAE."""
    return nn.Conv2d(input_channels, output_channels, 3, padding=1, **kwargs)


class MemoryBlock(nn.Module):
    """Fuse each frame with the preceding frame."""

    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            convolution(input_channels * 2, output_channels),
            nn.ReLU(inplace=True),
            convolution(output_channels, output_channels),
            nn.ReLU(inplace=True),
            convolution(output_channels, output_channels),
        )
        self.skip = (
            nn.Conv2d(input_channels, output_channels, 1, bias=False)
            if input_channels != output_channels
            else nn.Identity()
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, tensor: Tensor, previous: Tensor) -> Tensor:
        """Apply the residual block with one preceding frame."""
        return self.act(
            self.conv(torch.cat([tensor, previous], dim=1)) + self.skip(tensor)
        )


class TemporalPool(nn.Module):
    """Reduce the temporal axis through a channel projection."""

    def __init__(self, channels: int, stride: int) -> None:
        super().__init__()
        self.stride = stride
        self.conv = nn.Conv2d(channels * stride, channels, 1, bias=False)

    def forward(self, tensor: Tensor) -> Tensor:
        """Pool ``stride`` adjacent frames."""
        _, channels, height, width = tensor.shape
        return self.conv(tensor.reshape(-1, self.stride * channels, height, width))


def apply_with_boundaries(
    model: nn.Sequential,
    tensor: Tensor,
    state: dict[str, Tensor | None] | None,
) -> tuple[Tensor | None, dict[str, Tensor | None]]:
    """Run one ReAE half while carrying temporal boundary tensors."""
    state = state or {}
    next_state: dict[str, Tensor | None] = {}
    batch, time, channels, height, width = tensor.shape
    tensor = tensor.reshape(batch * time, channels, height, width)
    for index, block in enumerate(model):
        if isinstance(block, MemoryBlock):
            _, channels, height, width = tensor.shape
            time = tensor.shape[0] // batch
            video = tensor.reshape(batch, time, channels, height, width)
            key = f"memory_{index}"
            previous_state = state.get(key)
            if previous_state is not None:
                previous = torch.cat([previous_state, video[:, :-1]], dim=1)
            else:
                previous = F.pad(video, (0, 0, 0, 0, 0, 0, 1, 0))[:, :time]
            next_state[key] = video[:, -1:].detach().clone()
            tensor = block(tensor, previous.reshape_as(tensor))
        elif isinstance(block, TemporalPool):
            _, channels, height, width = tensor.shape
            time = tensor.shape[0] // batch
            video = tensor.reshape(batch, time, channels, height, width)
            key = f"pool_{index}"
            previous = state.get(key)
            if previous is not None:
                video = torch.cat([previous, video], dim=1)
                time = video.shape[1]
            stride = int(block.stride)
            full_frames = time // stride * stride
            next_state[key] = (
                video[:, full_frames:].detach().clone() if full_frames != time else None
            )
            if full_frames == 0:
                return None, next_state
            tensor = block(
                video[:, :full_frames].reshape(
                    batch * full_frames, channels, height, width
                )
            )
        else:
            tensor = block(tensor)
    _, channels, height, width = tensor.shape
    return (
        tensor.view(batch, tensor.shape[0] // batch, channels, height, width),
        next_state,
    )


def split_reae_state_dict(
    state_dict: dict[str, Tensor],
) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
    """Split the upstream ReAE checkpoint into encoder and decoder weights."""
    components: dict[Literal["encoder", "decoder"], dict[str, Tensor]] = {
        "encoder": {},
        "decoder": {},
    }
    for key, value in state_dict.items():
        component, separator, relative_key = key.partition(".")
        if not separator or component not in ("encoder", "decoder"):
            raise KeyError(f"Unexpected SwiftVR ReAE checkpoint key: {key!r}")
        if component == "encoder":
            components["encoder"][relative_key] = value
        else:
            components["decoder"][relative_key] = value
    if not components["encoder"] or not components["decoder"]:
        raise KeyError("SwiftVR ReAE checkpoint must contain encoder and decoder keys")
    return components["encoder"], components["decoder"]


__all__ = [
    "MemoryBlock",
    "TemporalPool",
    "apply_with_boundaries",
    "convolution",
    "split_reae_state_dict",
]
