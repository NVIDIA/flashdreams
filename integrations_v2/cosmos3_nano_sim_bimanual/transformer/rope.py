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

"""Transformer 3D mRoPE position IDs and interleaved cosines/sines."""

from __future__ import annotations

import math

import torch
from torch import Tensor


def action_mrope_ids(
    num_frames: int,
    temporal_offset: float,
    stride: float,
    tcf: int,
    start_frame: int = 0,
) -> Tensor:
    """``[3, num_frames*tcf]`` mRoPE ids for forward-dynamics action sub-tokens.

    Action ``a_{f-1}`` drives vision frame ``f`` and contributes ``tcf`` sub-tokens
    on a ``tcf``×-finer temporal grid, positioned so the **last** sub-token
    co-locates with vision frame ``f``:

        t(f, s) = temporal_offset + (f-1)*stride + (s+1)*stride/tcf   (f >= 1)
        t(0, s) = temporal_offset                                     (frame 0 = null action)

    ``start_frame`` is the absolute index of the first frame (for streaming, where a
    chunk starts mid-sequence). Height/width are 0 (actions carry no spatial extent).
    ``stride = base_fps/fps`` matches the vision frame stride so the co-location holds.
    """
    ts: list[float] = []
    for i in range(num_frames):
        f = start_frame + i
        for s in range(tcf):
            if f == 0:
                ts.append(float(temporal_offset))
            else:
                ts.append(
                    float(temporal_offset) + (f - 1) * stride + (s + 1) * stride / tcf
                )
    t = torch.tensor(ts, dtype=torch.float32)
    zeros = torch.zeros_like(t)
    return torch.stack([t, zeros, zeros], dim=0)  # [3, num_frames*tcf]


def text_mrope_ids(num_tokens: int, temporal_offset: float) -> tuple[Tensor, float]:
    """``[3, num_tokens]`` position IDs for text (all axes equal), + next offset."""
    ids = torch.arange(num_tokens, dtype=torch.float32) + temporal_offset
    mrope_ids = ids.unsqueeze(0).expand(3, -1).contiguous()
    return mrope_ids, temporal_offset + num_tokens


def vae_mrope_ids(
    grid_t: int,
    grid_h: int,
    grid_w: int,
    temporal_offset: float,
    *,
    reset_spatial_indices: bool = True,
    fps: float | None = None,
    base_fps: float = 24.0,
    temporal_compression_factor: int = 4,
    start_frame_offset: int = 0,
) -> tuple[Tensor, float]:
    """``[3, T*H*W]`` mRoPE IDs for VAE vision tokens (T-major), + next offset.

    Supports integer and fps-modulated positions used by the interactive model.
    """
    if fps is not None:
        tps = fps / temporal_compression_factor
        base_tps = base_fps / temporal_compression_factor
        frames = torch.arange(grid_t, dtype=torch.float32)
        scaled_t = (frames + start_frame_offset) / tps * base_tps + temporal_offset
        t_index = scaled_t.view(-1, 1).expand(-1, grid_h * grid_w).flatten()
    else:
        t_index = (
            torch.arange(grid_t, dtype=torch.float32)
            .view(-1, 1)
            .expand(-1, grid_h * grid_w)
            .flatten()
            + temporal_offset
            + start_frame_offset
        )

    h_index = (
        torch.arange(grid_h, dtype=torch.float32)
        .view(1, -1, 1)
        .expand(grid_t, -1, grid_w)
        .flatten()
    )
    w_index = (
        torch.arange(grid_w, dtype=torch.float32)
        .view(1, 1, -1)
        .expand(grid_t, grid_h, -1)
        .flatten()
    )
    if not reset_spatial_indices:
        h_index = h_index + float(int(temporal_offset))
        w_index = w_index + float(int(temporal_offset))

    mrope_ids = torch.stack([t_index, h_index, w_index], dim=0)
    next_offset = float(math.ceil(mrope_ids.max().item()) + 1)
    return mrope_ids, next_offset


def mrope_cos_sin(
    position_ids: Tensor,
    head_dim: int,
    theta: float,
    mrope_section: tuple[int, int, int],
) -> tuple[Tensor, Tensor]:
    """Interleaved-mRoPE cos/sin from ``[3, S]`` position IDs → ``[S, head_dim]`` each.

    Matches ``Qwen3VLTextRotaryEmbedding``: per-axis freqs, interleaved T/H/W
    combine over the ``mrope_section`` split, ``cat(freqs, freqs)``, cos/sin.
    """
    assert position_ids.shape[0] == 3, "position_ids must be [3, S]"
    device = position_ids.device
    inv_freq = 1.0 / (
        theta
        ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim)
    )  # [head_dim/2]
    freqs = (
        position_ids[:, :, None].float() * inv_freq[None, None, :]
    )  # [3, S, head_dim/2]

    # Interleaved mRoPE: start with all-T, overwrite H/W at strided indices.
    combined = freqs[0].clone()  # [S, head_dim/2]
    for axis, offset in ((1, 1), (2, 2)):  # H, W
        length = mrope_section[axis] * 3
        idx = slice(offset, length, 3)
        combined[..., idx] = freqs[axis][..., idx]

    emb = torch.cat((combined, combined), dim=-1)  # [S, head_dim]
    return emb.cos(), emb.sin()
