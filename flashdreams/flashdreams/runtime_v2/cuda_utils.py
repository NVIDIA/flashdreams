# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Small CUDA device helpers shared by the v2 presentation path."""

import torch


def resolve_cuda_device(device: torch.device) -> torch.device:
    """Return an indexed CUDA device, resolving ``cuda`` to the current device.

    Args:
        device: CUDA device to resolve.

    Returns:
        The same CUDA device with an explicit index.

    Raises:
        ValueError: ``device`` is not a CUDA device.
    """
    device = torch.device(device)
    if device.type != "cuda":
        raise ValueError(f"Expected a CUDA device, got {device}.")
    if device.index is None:
        return torch.device("cuda", torch.cuda.current_device())
    return device


__all__ = ["resolve_cuda_device"]
