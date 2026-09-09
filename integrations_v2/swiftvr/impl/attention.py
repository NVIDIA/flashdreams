# SPDX-FileCopyrightText: Copyright (c) 2026 SwiftVR Authors.
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

"""SwiftVR mask-free shifted-window attention for Diffusers WAN models.

Adapted from SwiftVR commit ``5ca168cef6ca7200f135fdfea85e5e13d12c5b53``.
The model topology and checkpoint loading remain in Diffusers; this module only
installs SwiftVR's inference-time spatial window processor.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MethodType
from typing import Any, cast

import torch
import torch.nn.functional as F
from diffusers.models.transformers.transformer_wan import WanAttention
from torch import Tensor, nn


def _axis_starts(
    size: int, window: int, *, shifted: bool, device: torch.device
) -> Tensor:
    if size <= window:
        return torch.zeros(1, dtype=torch.long, device=device)
    shift = window // 2 if shifted else 0
    maximum_start = size - window
    candidates = torch.arange(
        (size + window - 1) // window + 2,
        dtype=torch.long,
        device=device,
    )
    starts = torch.unique(
        (candidates * window - shift).clamp_(0, maximum_start), sorted=True
    )
    if starts.numel() > 2:
        keep = torch.ones_like(starts, dtype=torch.bool)
        keep[1:-1] = starts[2:] > starts[:-2] + window
        starts = starts[keep]
    return starts


def _window_indices(
    frames: int,
    height: int,
    width: int,
    window_height: int,
    window_width: int,
    *,
    shifted: bool,
    device: torch.device,
) -> Tensor:
    height_starts = _axis_starts(height, window_height, shifted=shifted, device=device)
    width_starts = _axis_starts(width, window_width, shifted=shifted, device=device)
    height_offset = torch.arange(window_height, device=device)
    width_offset = torch.arange(window_width, device=device)
    time_offset = torch.arange(frames, device=device)
    height_index = height_starts[:, None] + height_offset[None, :]
    width_index = width_starts[:, None] + width_offset[None, :]
    spatial = (
        height_index[:, None, :, None] * width + width_index[None, :, None, :]
    ).reshape(-1, window_height * window_width)
    indices = time_offset[None, :, None] * (height * width) + spatial[:, None, :]
    return indices.reshape(spatial.shape[0], frames * window_height * window_width)


@dataclass(frozen=True, slots=True)
class _WindowMetadata:
    flat_indices: Tensor
    owner_positions: Tensor
    window_count: int
    tokens_per_window: int


_WINDOW_CACHE: dict[tuple[object, ...], _WindowMetadata] = {}


def _window_metadata(
    frames: int,
    height: int,
    width: int,
    window_height: int,
    window_width: int,
    *,
    shifted: bool,
    device: torch.device,
) -> _WindowMetadata:
    key = (
        frames,
        height,
        width,
        window_height,
        window_width,
        shifted,
        device.type,
        device.index,
    )
    cached = _WINDOW_CACHE.get(key)
    if cached is not None:
        return cached

    indices = _window_indices(
        frames,
        height,
        width,
        window_height,
        window_width,
        shifted=shifted,
        device=device,
    )
    window_count, tokens_per_window = indices.shape
    owner = torch.empty(frames * height * width, dtype=torch.long)
    local = torch.arange(tokens_per_window, dtype=torch.long)
    order = range(window_count - 1, -1, -1) if not shifted else range(window_count)
    indices_cpu = indices.cpu()
    for window_index in order:
        owner[indices_cpu[window_index]] = window_index * tokens_per_window + local
    cached = _WindowMetadata(
        flat_indices=indices.reshape(-1).contiguous(),
        owner_positions=owner.to(device=device, non_blocking=True),
        window_count=window_count,
        tokens_per_window=tokens_per_window,
    )
    _WINDOW_CACHE[key] = cached
    return cached


def _apply_rotary_in_place(x: Tensor, cosine: Tensor, sine: Tensor) -> Tensor:
    cosine = cosine[..., 0::2].to(dtype=x.dtype)
    sine = sine[..., 1::2].to(dtype=x.dtype)
    pairs = x.view(*x.shape[:-1], -1, 2)
    even, odd = pairs[..., 0], pairs[..., 1]
    temporary = even * sine
    even.mul_(cosine)
    even.addcmul_(odd, sine, value=-1)
    odd.mul_(cosine)
    odd.add_(temporary)
    return x


def _qkv(
    attention: WanAttention, hidden_states: Tensor
) -> tuple[Tensor, Tensor, Tensor]:
    if getattr(attention, "fused_projections", False):
        return attention.to_qkv(hidden_states).chunk(3, dim=-1)
    return (
        attention.to_q(hidden_states),
        attention.to_k(hidden_states),
        attention.to_v(hidden_states),
    )


def _release_input_storage(tensor: Tensor) -> None:
    """Release a consumed CUDA temporary retained by the caller's frame."""
    try:
        if tensor.is_cuda and tensor._base is None and tensor.is_contiguous():
            tensor.untyped_storage().resize_(0)
    except RuntimeError:
        pass


def _infer_local_shape(
    global_shape: tuple[int, int, int], token_count: int
) -> tuple[int, int, int]:
    frames, height, width = global_shape
    if token_count == frames * height * width:
        return global_shape
    if token_count % (height * width) == 0:
        return token_count // (height * width), height, width
    raise RuntimeError(
        f"Cannot infer local SwiftVR shape from {global_shape} and {token_count} tokens."
    )


class ShiftedWindowAttentionProcessor:
    """Mask-free 2D spatial shifted-window self-attention."""

    def __init__(self, window: tuple[int, int] = (16, 16)) -> None:
        if min(window) <= 0:
            raise ValueError(
                f"SwiftVR attention window must be positive, got {window}."
            )
        self.window = window

    def __call__(
        self,
        attention: WanAttention,
        hidden_states: Tensor,
        encoder_hidden_states: Tensor | None = None,
        attention_mask: Tensor | None = None,
        rotary_emb: tuple[Tensor, Tensor] | None = None,
    ) -> Tensor:
        if encoder_hidden_states is not None or attention.is_cross_attention:
            raise RuntimeError("SwiftVR window attention only supports self-attention.")
        if attention_mask is not None:
            raise RuntimeError("SwiftVR window attention does not accept a mask.")
        global_shape = getattr(attention, "_swiftvr_shape", None)
        if global_shape is None:
            raise RuntimeError("SwiftVR attention shape was not initialized.")

        batch, tokens, _ = hidden_states.shape
        frames, height, width = _infer_local_shape(global_shape, tokens)
        window_height = min(self.window[0], height)
        window_width = min(self.window[1], width)
        shifted = bool(getattr(attention, "_swiftvr_shifted", False))
        metadata = _window_metadata(
            frames,
            height,
            width,
            window_height,
            window_width,
            shifted=shifted,
            device=hidden_states.device,
        )
        heads = attention.heads
        head_dim = attention.inner_dim // heads

        query, key, value = _qkv(attention, hidden_states)
        _release_input_storage(hidden_states)
        query = attention.norm_q(query).unflatten(2, (heads, head_dim))
        key = attention.norm_k(key).unflatten(2, (heads, head_dim))
        value = value.unflatten(2, (heads, head_dim))
        value = torch.index_select(value, 1, metadata.flat_indices).view(
            batch * metadata.window_count,
            metadata.tokens_per_window,
            heads,
            head_dim,
        )
        if rotary_emb is not None:
            query = _apply_rotary_in_place(query, *rotary_emb)
            key = _apply_rotary_in_place(key, *rotary_emb)
        query = torch.index_select(query, 1, metadata.flat_indices).view_as(value)
        key = torch.index_select(key, 1, metadata.flat_indices).view_as(value)

        output = F.scaled_dot_product_attention(
            query.transpose(1, 2),
            key.transpose(1, 2),
            value.transpose(1, 2),
            dropout_p=0.0,
            is_causal=False,
        ).transpose(1, 2)
        output = output.reshape(
            batch, metadata.window_count * metadata.tokens_per_window, heads, head_dim
        )
        output = torch.index_select(output, 1, metadata.owner_positions)
        output = output.reshape(batch, tokens, heads * head_dim)
        output = attention.to_out[0](output)
        if attention.training and isinstance(attention.to_out[1], nn.Dropout):
            output = attention.to_out[1](output)
        return output


def _swiftvr_block_forward(
    block: Any,
    hidden_states: Tensor,
    encoder_hidden_states: Tensor,
    timestep_projection: Tensor,
    rotary_embedding: tuple[Tensor, Tensor],
) -> Tensor:
    """Apply a WAN block with SwiftVR's inference-time BF16 operation order."""
    hidden_dtype = hidden_states.dtype
    if timestep_projection.ndim == 4:
        modulation = (
            block.scale_shift_table.unsqueeze(0) + timestep_projection.float()
        ).to(hidden_dtype)
        values = modulation.chunk(6, dim=2)
        shift, scale, gate, cross_shift, cross_scale, cross_gate = (
            value.squeeze(2) for value in values
        )
    else:
        modulation = (block.scale_shift_table + timestep_projection.float()).to(
            hidden_dtype
        )
        shift, scale, gate, cross_shift, cross_scale, cross_gate = modulation.chunk(
            6, dim=1
        )
    attention_output = block.attn1(
        block.norm1(hidden_states).mul_(1 + scale).add_(shift),
        None,
        None,
        rotary_embedding,
    )
    hidden_states.addcmul_(attention_output, gate)
    attention_output = block.attn2(
        block.norm2(hidden_states), encoder_hidden_states, None, None
    )
    hidden_states.add_(attention_output)
    feed_forward_output = block.ffn(
        block.norm3(hidden_states).mul_(1 + cross_scale).add_(cross_shift)
    )
    hidden_states.addcmul_(feed_forward_output, cross_gate)
    return hidden_states


def prepare_transformer(
    transformer: Any,
    *,
    window: tuple[int, int] = (16, 16),
    compile_blocks: bool = False,
) -> None:
    """Install SwiftVR attention, fuse projections, and optionally compile blocks."""
    processor = ShiftedWindowAttentionProcessor(window)
    blocks = getattr(transformer, "blocks")
    for index, block in enumerate(blocks):
        block.attn1._swiftvr_shifted = bool(index % 2)
        block.forward = MethodType(_swiftvr_block_forward, block)
    for module in transformer.modules():
        if isinstance(module, WanAttention):
            module.fuse_projections()
            if not module.is_cross_attention:
                cast(Any, module).set_processor(processor)
    _WINDOW_CACHE.clear()
    if compile_blocks:
        for index, block in enumerate(blocks):
            blocks[index] = torch.compile(block, mode="default", fullgraph=False)
    transformer.eval()


__all__ = ["ShiftedWindowAttentionProcessor", "prepare_transformer"]
