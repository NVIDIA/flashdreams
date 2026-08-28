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

"""Orthogonal three-axis rotary-angle construction for Waypoint attention."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from waypoint.spec import WAYPOINT_1_5, WaypointModelSpec


def apply_waypoint_ortho_rope(tokens: Tensor, cosine: Tensor, sine: Tensor) -> Tensor:
    """Apply Waypoint's half-head rotary transform to attention tensors.

    Args:
        tokens: Query or key tensor shaped ``[batch, tokens, heads, head_dim]``.
        cosine: Packed RoPE cosine factors shaped ``[tokens, 1, head_dim / 2]``.
        sine: Packed RoPE sine factors shaped ``[tokens, 1, head_dim / 2]``.

    Returns:
        The rotary-transformed tensor with the same shape and dtype as ``tokens``.

    Raises:
        ValueError: Tensor or angle shapes cannot represent the same token sequence.
    """
    if tokens.ndim != 4 or tokens.shape[-1] % 2:
        raise ValueError("tokens must have shape [batch, tokens, heads, even_head_dim]")
    expected_angles = (tokens.shape[1], 1, tokens.shape[-1] // 2)
    if cosine.shape != expected_angles or sine.shape != expected_angles:
        raise ValueError(
            "RoPE angles must have shape "
            f"{expected_angles}, got cosine={tuple(cosine.shape)}, sine={tuple(sine.shape)}"
        )
    # Projection channels arrive as adjacent real/imaginary pairs. The attention
    # kernel receives the rotated result in its packed real-half / imaginary-half
    # layout, matching Waypoint's grouped-query projections.
    promoted = tokens.float()
    first = promoted[..., 0::2]
    second = promoted[..., 1::2]
    cosine = cosine.unsqueeze(0).float()
    sine = sine.unsqueeze(0).float()
    rotated = torch.cat(
        (first * cosine - second * sine, first * sine + second * cosine), dim=-1
    )
    return rotated.to(dtype=tokens.dtype)


class WaypointOrthoRoPEAngles(nn.Module):
    """Construct the parameter-free three-axis rotary angles used by Waypoint.

    One quarter of each head's complex dimensions represents horizontal location,
    one quarter represents vertical location, and the remaining half represents
    autoregressive time. Spatial coordinates are patch centers relative to the
    latent-image center. The spatial frequency ceiling preserves circular
    frequency under the checkpoint's 16:32 aspect ratio.
    """

    spatial_frequencies: Tensor
    temporal_frequencies: Tensor

    def __init__(self, spec: WaypointModelSpec = WAYPOINT_1_5) -> None:
        """Initialize an angle generator from a checkpoint architecture contract.

        Args:
            spec: Published Waypoint architecture and RoPE constants.

        Raises:
            ValueError: The head dimension cannot be evenly partitioned.
        """
        super().__init__()
        if spec.head_dim % 8:
            raise ValueError(
                "Waypoint orthogonal RoPE requires a head dimension divisible "
                f"by 8, got {spec.head_dim}"
            )
        self.spec = spec
        spatial_dim = spec.head_dim // 8
        temporal_dim = spec.head_dim // 4
        spatial_frequency_count = (spatial_dim + 1) // 2
        max_frequency = min(spec.patch_grid_height, spec.patch_grid_width) * (
            spec.rope_nyquist_fraction
        )
        spatial = (
            torch.linspace(
                1.0,
                max_frequency / 2,
                spatial_frequency_count,
                dtype=torch.float32,
            )
            * torch.pi
        ).repeat_interleave(2)[:spatial_dim]
        temporal = torch.pow(
            torch.tensor(spec.rope_theta, dtype=torch.float32),
            -torch.arange(0, temporal_dim, 2, dtype=torch.float32) / temporal_dim,
        ).repeat_interleave(2)
        self.register_buffer("spatial_frequencies", spatial, persistent=False)
        self.register_buffer("temporal_frequencies", temporal, persistent=False)

    def _apply(self, fn):
        """Retain FP32 angle arithmetic after device precision conversion."""

        def keep_dtype(tensor: Tensor) -> Tensor:
            return fn(tensor).to(dtype=tensor.dtype)

        return super()._apply(keep_dtype)

    def forward(
        self,
        *,
        frame_index: Tensor,
        row_index: Tensor,
        column_index: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Return packed cosine and sine factors for token positions.

        Args:
            frame_index: Integer latent-frame positions with shape ``[tokens]``.
            row_index: Integer patch-row positions with shape ``[tokens]``.
            column_index: Integer patch-column positions with shape ``[tokens]``.

        Returns:
            Cosine and sine tensors, each shaped ``[tokens, 1, head_dim / 2]``.

        Raises:
            ValueError: Position tensors do not share one one-dimensional shape.
        """
        positions = (frame_index, row_index, column_index)
        token_count = frame_index.numel()
        if any(
            position.ndim != 1 or position.numel() != token_count
            for position in positions
        ):
            raise ValueError(
                "all RoPE position tensors must have matching [tokens] shape"
            )
        if not (frame_index.device == row_index.device == column_index.device):
            raise ValueError("all RoPE position tensors must be on one device")

        dtype = torch.float32
        device = frame_index.device
        column_position = (
            2.0 * column_index.to(dtype) + 1.0
        ) / self.spec.patch_grid_width - 1.0
        row_position = (
            2.0 * row_index.to(dtype) + 1.0
        ) / self.spec.patch_grid_height - 1.0
        temporal_position = frame_index.to(dtype)
        angles = torch.cat(
            (
                column_position[:, None] * self.spatial_frequencies.to(device=device),
                row_position[:, None] * self.spatial_frequencies.to(device=device),
                temporal_position[:, None]
                * self.temporal_frequencies.to(device=device),
            ),
            dim=-1,
        )
        return torch.cos(angles)[:, None], torch.sin(angles)[:, None]
