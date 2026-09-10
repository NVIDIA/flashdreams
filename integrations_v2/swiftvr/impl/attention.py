# SPDX-FileCopyrightText: Copyright (c) 2026 SwiftVR Authors.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# This file has been modified by NVIDIA CORPORATION & AFFILIATES.
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

"""SwiftVR mask-free shifted-window attention for FlashDreams WAN blocks."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

from flashdreams.core.attention.rope import apply_rope_freqs
from flashdreams.recipes.wan.transformer.impl.modules import (
    Block,
    BlockCache,
    SelfAttention,
)


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


class ShiftedWindowSelfAttention(SelfAttention):
    """FlashDreams WAN projections with SwiftVR's spatial window attention."""

    def __init__(
        self,
        *,
        query_dim: int,
        n_heads: int,
        head_dim: int,
        eps: float,
        window: tuple[int, int],
        shifted: bool,
    ) -> None:
        if min(window) <= 0:
            raise ValueError(
                f"SwiftVR attention window must be positive, got {window}."
            )
        super().__init__(
            query_dim=query_dim,
            n_heads=n_heads,
            head_dim=head_dim,
            eps=eps,
        )
        self.window = window
        self.shifted = shifted

    def forward(
        self,
        hidden_states: Tensor,
        *,
        rope_freqs: Tensor,
        shape: tuple[int, int, int],
    ) -> Tensor:
        """Apply mask-free window attention to one latent chunk."""
        batch, tokens, _ = hidden_states.shape
        frames, height, width = _infer_local_shape(shape, tokens)
        window_height = min(self.window[0], height)
        window_width = min(self.window[1], width)
        metadata = _window_metadata(
            frames,
            height,
            width,
            window_height,
            window_width,
            shifted=self.shifted,
            device=hidden_states.device,
        )

        query = self.norm_q(self.q(hidden_states)).unflatten(
            2, (self.n_heads, self.head_dim)
        )
        key = self.norm_k(self.k(hidden_states)).unflatten(
            2, (self.n_heads, self.head_dim)
        )
        value = self.v(hidden_states).unflatten(2, (self.n_heads, self.head_dim))
        _release_input_storage(hidden_states)
        query = apply_rope_freqs(query, rope_freqs, interleaved=True)
        key = apply_rope_freqs(key, rope_freqs, interleaved=True)
        value = torch.index_select(value, 1, metadata.flat_indices).view(
            batch * metadata.window_count,
            metadata.tokens_per_window,
            self.n_heads,
            self.head_dim,
        )
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
            batch, metadata.window_count * metadata.tokens_per_window, -1
        )
        output = torch.index_select(output, 1, metadata.owner_positions)
        return self.o(output.reshape(batch, tokens, -1))


class SwiftVRBlock(Block):
    """FlashDreams WAN block with SwiftVR shifted-window self-attention."""

    self_attn: ShiftedWindowSelfAttention

    def __init__(
        self,
        *,
        dim: int,
        ffn_dim: int,
        num_heads: int,
        cross_attn_norm: bool,
        eps: float,
        window: tuple[int, int],
        shifted: bool,
    ) -> None:
        super().__init__(
            dim=dim,
            ffn_dim=ffn_dim,
            num_heads=num_heads,
            cross_attn_norm=cross_attn_norm,
            eps=eps,
        )
        self.self_attn = ShiftedWindowSelfAttention(
            query_dim=dim,
            n_heads=num_heads,
            head_dim=dim // num_heads,
            eps=eps,
            window=window,
            shifted=shifted,
        )

    def forward(  # type: ignore[override]
        self,
        x: Tensor,
        e: Tensor,
        cache: BlockCache,
        rope_freqs: Tensor,
        shape: tuple[int, int, int],
    ) -> Tensor:
        """Apply a WAN block in SwiftVR's inference-time operation order."""
        assert self._parameters_updated_after_loading_checkpoint, (
            "Call update_parameters_after_loading_checkpoint() before inference."
        )
        hidden_dtype = x.dtype
        modulation = (self.modulation + e.float()).to(hidden_dtype)
        shift, scale, gate, cross_shift, cross_scale, cross_gate = modulation.chunk(
            6, dim=-2
        )
        attention_output = self.self_attn(
            self.norm1(x).mul_(1 + scale).add_(shift),
            rope_freqs=rope_freqs,
            shape=shape,
        )
        x.addcmul_(attention_output, gate)
        attention_output = self.cross_attn(self.norm3(x), kv_cache=cache.cross_attn)
        x.add_(attention_output)
        feed_forward_output = self.ffn(
            self.norm2(x).mul_(1 + cross_scale).add_(cross_shift)
        )
        x.addcmul_(feed_forward_output, cross_gate)
        return x


def prepare_transformer(
    transformer: torch.nn.Module,
    *,
    window: tuple[int, int] = (16, 16),
    compile_blocks: bool = False,
) -> None:
    """Validate SwiftVR blocks and optionally compile them."""
    blocks = getattr(transformer, "blocks")
    for block in blocks:
        if not isinstance(block, SwiftVRBlock):
            raise TypeError(f"Expected SwiftVRBlock, got {type(block).__name__}.")
        block.self_attn.window = window
    _WINDOW_CACHE.clear()
    if compile_blocks:
        for block in blocks:
            block.forward = torch.compile(  # type: ignore[method-assign]
                block.forward, mode="default", fullgraph=False
            )
    transformer.eval()


__all__ = [
    "ShiftedWindowSelfAttention",
    "SwiftVRBlock",
    "prepare_transformer",
]
