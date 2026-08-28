# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Parameter-free RMS normalization operations used by Waypoint DiT blocks."""

from __future__ import annotations

import torch.nn.functional as F
from torch import Tensor


def adaptive_rms_norm(tokens: Tensor, scale: Tensor, bias: Tensor) -> Tensor:
    """Apply per-latent-frame adaptive RMS normalization.

    Args:
        tokens: Token tensor shaped ``[batch, frames * tokens_per_frame, channels]``.
        scale: Adaptive RMSNorm scale tensor shaped ``[batch, frames, channels]``.
        bias: Adaptive RMSNorm bias tensor shaped ``[batch, frames, channels]``.

    Returns:
        Adaptively normalized tokens with the same shape and dtype as ``tokens``.

    Raises:
        ValueError: The token and conditioner layouts are incompatible.
    """
    if tokens.ndim != 3 or scale.ndim != 3 or bias.ndim != 3:
        raise ValueError("tokens, scale, and bias must each have three dimensions")
    if scale.shape != bias.shape:
        raise ValueError("AdaRMSNorm scale and bias shapes must match")
    batch_size, token_count, channels = tokens.shape
    if scale.shape[0] != batch_size or scale.shape[2] != channels:
        raise ValueError("AdaRMSNorm conditioning must match token batch and channels")
    frames = scale.shape[1]
    if frames < 1 or token_count % frames:
        raise ValueError(
            "token count must be divisible by the number of conditioned frames"
        )

    tokens_per_frame = token_count // frames
    x = tokens.reshape(batch_size, frames, tokens_per_frame, channels)
    output = F.rms_norm(x, (channels,), weight=None, eps=None)
    output = output * (1 + scale[:, :, None]) + bias[:, :, None]
    return output.reshape_as(tokens)


def adaptive_gate(tokens: Tensor, gate: Tensor) -> Tensor:
    """Scale each latent frame's residual branch with a conditioner gate.

    Args:
        tokens: Residual tensor shaped ``[batch, frames * tokens_per_frame, channels]``.
        gate: Per-frame gate tensor shaped ``[batch, frames, channels]``.

    Returns:
        Gated residual tensor with the same shape and dtype as ``tokens``.

    Raises:
        ValueError: The residual and gate layouts are incompatible.
    """
    if tokens.ndim != 3 or gate.ndim != 3:
        raise ValueError("tokens and gate must each have three dimensions")
    batch_size, token_count, channels = tokens.shape
    if gate.shape[0] != batch_size or gate.shape[2] != channels:
        raise ValueError("AdaGate conditioning must match residual batch and channels")
    frames = gate.shape[1]
    if frames < 1 or token_count % frames:
        raise ValueError("token count must be divisible by the number of gated frames")
    tokens_per_frame = token_count // frames
    gated = tokens.reshape(batch_size, frames, tokens_per_frame, channels)
    return (gated * gate[:, :, None]).reshape_as(tokens)
