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

"""Token metadata and fixed-slot cache layout for multi-view prefill."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn.attention.flex_attention import BlockMask

from flashdreams.core.attention.kvcache import LayerKV
from flashdreams.core.attention.multiview.mask import (
    ROLE_CLEAN_TARGET,
    ROLE_CONTROL,
    ROLE_TARGET_CONDITION,
    AttentionPattern,
    AttentionScope,
    StreamFields,
    build_block_mask,
    visibility,
)
from flashdreams.core.attention.multiview.packing import (
    ClipGeometry,
    _check_ranges,
    _check_view_text_tokens,
    _stamp_view_major,
    _vision,
    concat,
    text_stream,
)


@dataclass(frozen=True)
class PrefillPack:
    """Describe the text and vision streams of a clean prefill pass.

    The vision stream contains one view-major section for every control range,
    followed by the target. ``memory_token_indexes`` selects the control and
    conditioning rows that enter the persistent K/V cache.
    """

    und: StreamFields
    """Understanding/text stream, shaped ``[num_und]``."""

    gen: StreamFields
    """Generation/vision stream, shaped ``[num_gen]``."""

    memory_token_indexes: Tensor
    """Rows of ``gen`` retained in persistent memory, in cache order."""

    control_ranges: tuple[tuple[int, int], ...]
    """Control-frame ranges represented by consecutive sections of ``gen``."""

    @property
    def num_und(self) -> int:
        """Return the number of understanding/text tokens."""
        return len(self.und)

    def mask(
        self,
        *,
        pattern: AttentionPattern = "causal",
        scope: AttentionScope = "all_views",
        decomposed_temporal_window_seconds: float | None = None,
    ) -> Tensor:
        """Build the dense ``[num_gen, num_und + num_gen]`` prefill mask."""
        return visibility(
            self.gen,
            concat(self.und, self.gen),
            pattern=pattern,
            scope=scope,
            decomposed_temporal_window_seconds=decomposed_temporal_window_seconds,
        )

    def block_mask(
        self,
        *,
        pattern: AttentionPattern = "causal",
        scope: AttentionScope = "all_views",
        decomposed_temporal_window_seconds: float | None = None,
        block_size: int | tuple[int, int] = 128,
    ) -> BlockMask:
        """Build the equivalent block-sparse prefill mask."""
        return build_block_mask(
            self.gen,
            concat(self.und, self.gen),
            pattern=pattern,
            scope=scope,
            decomposed_temporal_window_seconds=decomposed_temporal_window_seconds,
            block_size=block_size,
        )


def build_prefill_pack(
    geometry: ClipGeometry,
    *,
    text_tokens: int,
    view_text_tokens: Sequence[int] | None = None,
    target_frames: int | None = None,
    control_ranges: Sequence[tuple[int, int]] | None = None,
    device: torch.device | None = None,
) -> PrefillPack:
    """Lay out a clean prefill over text, controls, and available target frames.

    Args:
        geometry: Multi-view clip geometry.
        text_tokens: Number of understanding/text tokens prefixed to attention.
        view_text_tokens: Token count for each view's caption, in view order.
            ``None`` leaves every caption token visible to the whole rig.
        target_frames: Target frames represented by the pass. ``None`` keeps
            only the supplied conditioning frames.
        control_ranges: Control frames to prefill, one range per cache section.
            ``None`` includes the whole clip in one section.
        device: Device on which to construct the metadata.
    """
    if text_tokens < 0:
        raise ValueError(f"text_tokens must be non-negative, got {text_tokens}.")
    _check_view_text_tokens(geometry, view_text_tokens)
    device = device or torch.device("cpu")
    target_frames = (
        geometry.condition_frames if target_frames is None else target_frames
    )
    if not geometry.condition_frames <= target_frames <= geometry.frames_per_view:
        raise ValueError(
            f"target_frames must cover the {geometry.condition_frames} conditioning "
            f"frames and fit the clip's {geometry.frames_per_view}, got {target_frames}."
        )
    if control_ranges is None:
        control_ranges = [(0, geometry.frames_per_view)]
    control_ranges = tuple(control_ranges)
    _check_ranges(
        control_ranges,
        first_frame=0,
        last_frame=geometry.frames_per_view,
        what="Control",
    )
    if not control_ranges:
        raise ValueError("a prefill needs at least one control range.")

    num_views, spatial = geometry.num_views, geometry.spatial_tokens
    target_frame, target_view = _stamp_view_major(
        torch.arange(target_frames, device=device), num_views, spatial
    )
    is_condition = target_frame < geometry.condition_frames

    def control(start: int, end: int) -> StreamFields:
        frame, view = _stamp_view_major(
            torch.arange(start, end, device=device), num_views, spatial
        )
        return _vision(
            frame,
            view,
            role=ROLE_CONTROL,
            is_noisy=False,
            geometry=geometry,
        )

    gen = concat(
        *(control(start, end) for start, end in control_ranges),
        _vision(
            target_frame,
            target_view,
            role=torch.where(
                is_condition,
                torch.full_like(target_frame, ROLE_TARGET_CONDITION),
                torch.full_like(target_frame, ROLE_CLEAN_TARGET),
            ),
            is_noisy=~is_condition,
            geometry=geometry,
        ),
    )

    keep = gen.is_control | (gen.token_role_id == ROLE_TARGET_CONDITION)
    return PrefillPack(
        und=text_stream(text_tokens, device, view_text_tokens=view_text_tokens),
        gen=gen,
        memory_token_indexes=keep.nonzero().squeeze(-1),
        control_ranges=control_ranges,
    )


def pad_control_slots(
    memory_kv: Sequence[LayerKV],
    *,
    control_ranges: Sequence[tuple[int, int]],
    tokens_per_frame: int,
    slot_frames: int,
) -> list[LayerKV]:
    """Pad compact prefilled controls into fixed-width cache slots.

    Control sections precede any remaining cached tokens. Short sections leave
    zero-filled tails so later sections retain the physical offsets described
    by the multi-view memory layout.
    """
    if tokens_per_frame < 1:
        raise ValueError(f"tokens_per_frame must be positive, got {tokens_per_frame}.")
    if slot_frames < 1:
        raise ValueError(f"slot_frames must be positive, got {slot_frames}.")
    if any(end <= start for start, end in control_ranges):
        raise ValueError("control ranges must be non-empty.")

    sections = [(end - start) * tokens_per_frame for start, end in control_ranges]
    slot_tokens = slot_frames * tokens_per_frame
    if any(section > slot_tokens for section in sections):
        raise ValueError(
            f"control ranges {list(control_ranges)} do not fit {slot_frames}-frame slots."
        )
    real_control_tokens = sum(sections)
    for key, value in memory_kv:
        if key.ndim != 4 or value.ndim != 4:
            raise ValueError("prefilled keys and values must have shape [B, H, S, D].")
        if key.shape[2] != value.shape[2]:
            raise ValueError(
                "prefilled keys and values must have the same token count."
            )
        if key.shape[2] < real_control_tokens:
            raise ValueError(
                f"prefill has {key.shape[2]} tokens but its control ranges need "
                f"{real_control_tokens}."
            )
    if all(section == slot_tokens for section in sections):
        return list(memory_kv)

    control_tokens = len(sections) * slot_tokens
    padded: list[LayerKV] = []
    for key, value in memory_kv:
        rest = int(key.shape[2]) - real_control_tokens
        pair: list[Tensor] = []
        for source in (key, value):
            buffer = source.new_zeros(
                (
                    source.shape[0],
                    source.shape[1],
                    control_tokens + rest,
                    source.shape[3],
                )
            )
            read = 0
            for slot, section in enumerate(sections):
                start = slot * slot_tokens
                buffer[:, :, start : start + section] = source[
                    :, :, read : read + section
                ]
                read += section
            buffer[:, :, control_tokens:] = source[:, :, read:]
            pair.append(buffer)
        padded.append((pair[0], pair[1]))
    return padded


__all__ = ["PrefillPack", "build_prefill_pack", "pad_control_slots"]
