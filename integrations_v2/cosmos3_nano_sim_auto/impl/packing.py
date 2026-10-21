# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Token layout and metadata for multiview autoregressive inference.

Each token has a view, frame, role and causal step used by the attention mask.
Tokens are ordered by view, then frame, then spatial position.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, fields
from itertools import pairwise
from typing import Literal

import torch
from torch import Tensor
from torch.nn.attention.flex_attention import BlockMask

from cosmos3_nano_sim_auto.impl.transformer.attention import MasklessFold
from cosmos3_nano_sim_auto.impl.transformer.fold import FoldPlan, build_fold_plan
from cosmos3_nano_sim_auto.impl.transformer.mask import (
    ROLE_CLEAN_TARGET,
    ROLE_CONTROL,
    ROLE_CURRENT_TARGET,
    ROLE_PADDING,
    ROLE_TARGET_CONDITION,
    ROLE_UND,
    AttentionPattern,
    AttentionScope,
    StreamFields,
    build_block_mask,
    build_maskless_block_masks,
    visibility,
)

ChunkPassKind = Literal["noisy", "clean", "control"]
"""Which kind of pass some metadata is for. See :func:`build_chunk_metadata`."""

DEFAULT_CHUNK_FRAMES = 3
"""Latent frames generated per autoregressive step, per camera.

What the multiview checkpoint was trained with, confirmed by the model team
alongside the mRoPE convention and the decomposed scope. It differs from
``VFMSpec.ar_chunk_latent_frames``, which is four because that describes the
single-view chunk=4 checkpoint, so this rig cannot take the spec's word for it.

It travels with the four-step schedule rather than standing alone: the same
denoise budget spread over more autoregressive steps compounds error through
more of them, so three at two steps is visibly worse than four at two.
"""


@dataclass(frozen=True)
class ClipGeometry:
    """Shape of one multiview clip in latent-token space.

    Attributes:
        num_views: cameras in the rig. Eleven, for Hyperion.
        frames_per_view: latent frames each camera contributes over the whole
            clip, or ``None`` for a run with no end. Nothing about generating a
            chunk needs the total; it is here so that a clip of known length
            can say where it stops, and an endless run simply does not.
        patch_h, patch_w: the latent grid of one frame after 2x2 patching. Kept as
            two numbers rather than their product because mRoPE positions the two
            spatial axes separately.
        frames_per_chunk: latent frames generated per autoregressive step, which is
            also the causal partition's block size. See :func:`causal_steps`.
        condition_frames: leading latent frames per view that are supplied rather
            than generated. A count rather than a mask, since the only masks the
            model accepts are a contiguous prefix shared by every view.
        seconds_per_frame: real time between one camera's consecutive latent
            frames. Only the ``"decomposed"`` scope reads the timestamps this
            produces; the default of 1.0 makes a timestamp equal its frame index.
    """

    num_views: int
    frames_per_view: int | None
    patch_h: int
    patch_w: int
    frames_per_chunk: int
    condition_frames: int
    seconds_per_frame: float = 1.0
    latent_size: tuple[int, int] | None = None
    """Original latent height/width before bottom/right patch padding, if needed."""
    rig_view_ids: tuple[int, ...] | None = None
    """Canonical camera embedding rows, preserved when views are selected or reordered."""

    def __post_init__(self) -> None:
        if self.rig_view_ids is not None and (
            len(self.rig_view_ids) != self.num_views
            or any(view < 0 or view >= 11 for view in self.rig_view_ids)
        ):
            raise ValueError(
                "rig_view_ids must supply one canonical camera ID (0..10) per view"
            )
        if self.num_views < 1:
            raise ValueError(f"num_views must be >= 1, got {self.num_views}.")
        if self.frames_per_view is not None and self.frames_per_view < 1:
            raise ValueError(
                f"frames_per_view must be >= 1 or None, got {self.frames_per_view}."
            )
        if self.patch_h < 1 or self.patch_w < 1:
            raise ValueError(
                f"patch grid must be positive, got {self.patch_h}x{self.patch_w}."
            )
        if self.frames_per_chunk < 1:
            raise ValueError(
                f"frames_per_chunk must be >= 1, got {self.frames_per_chunk}."
            )
        if self.condition_frames < 0 or (
            self.frames_per_view is not None
            and self.condition_frames > self.frames_per_view
        ):
            raise ValueError(
                f"condition_frames must lie in [0, {self.frames_per_view}], got {self.condition_frames}."
            )
        if self.seconds_per_frame <= 0:
            raise ValueError(
                f"seconds_per_frame must be positive, got {self.seconds_per_frame}."
            )

    def latent_hw(self, patch: int) -> tuple[int, int]:
        """Return the unpadded latent extent and validate its token grid."""
        height, width = self.latent_size or (self.patch_h * patch, self.patch_w * patch)
        if min(height, width, patch) < 1 or (
            (height + patch - 1) // patch != self.patch_h
            or (width + patch - 1) // patch != self.patch_w
        ):
            raise ValueError("latent_size must patch to the geometry's patch_h/patch_w")
        return height, width

    @property
    def spatial_tokens(self) -> int:
        """Tokens one latent frame of one camera contributes."""
        return self.patch_h * self.patch_w

    @property
    def tokens_per_frame(self) -> int:
        """Tokens one latent frame contributes across the whole rig."""
        return self.num_views * self.spatial_tokens


def causal_steps(frame_id: Tensor, frames_per_chunk: int) -> Tensor:
    """Map frame indexes to the ``[1, C, C, ...]`` causal partition.

    Frame 0 is a singleton at step 0; frames ``1..C`` are step 1, ``C+1..2C`` step 2,
    and so on. Negative frames (padding) stay at -1.

    This is not ``frame // frames_per_chunk``: the singleton at the front pushes
    every later block along by one frame. Getting it wrong shifts the whole
    partition, and a chunk would then read its own step as history -- which the
    mask allows on a clean replay, so it goes wrong quietly.
    """
    if frames_per_chunk < 1:
        raise ValueError(f"frames_per_chunk must be >= 1, got {frames_per_chunk}.")
    nonnegative = frame_id.clamp_min(0)
    body_step = 1 + torch.div(nonnegative - 1, frames_per_chunk, rounding_mode="floor")
    step = torch.where(nonnegative == 0, torch.zeros_like(body_step), body_step)
    return torch.where(frame_id >= 0, step, torch.full_like(step, -1))


def ar_chunk_range(geometry: ClipGeometry, frame: int) -> tuple[int, int] | None:
    """The ``[start, end)`` a chunk starting at ``frame`` covers.

    One chunk at a time, so a rollout of unknown length has no list to hold.

    The same ``[1, C, C, ...]`` partition :func:`causal_steps` numbers. Frame
    0 on its own is not an artifact of the block size; training treated the
    first frame that way too. So a text2video clip generates frame 0 alone
    before settling into ``frames_per_chunk`` at a time.

    Returns:
        ``None`` when ``frame`` is at or past the clip's end, meaning there is
        no chunk there. A clip with no end always has one.
    """
    if frame < 0:
        raise ValueError(f"frame must be >= 0, got {frame}.")
    last = geometry.frames_per_view
    if last is not None and frame >= last:
        return None
    if frame == 0:
        end = 1
    else:
        blocks = (frame - 1) // geometry.frames_per_chunk + 1
        end = 1 + blocks * geometry.frames_per_chunk
    return frame, end if last is None else min(end, last)


def ar_chunk_ranges(geometry: ClipGeometry) -> list[tuple[int, int]]:
    """Every ``[start, end)`` a rollout generates, in order.

    :func:`ar_chunk_range` iterated from the first frame that is not supplied.
    The partition has to agree with :func:`causal_steps` or a chunk would span
    two steps, which ``test_packing.py`` checks.

    Raises:
        ValueError: the clip has no end, so this list has none either. Step
            through :func:`ar_chunk_range` instead.
    """
    if geometry.frames_per_view is None:
        raise ValueError(
            "a clip with no end has no list of chunks. Call ar_chunk_range one "
            "frame at a time."
        )
    ranges: list[tuple[int, int]] = []
    frame = geometry.condition_frames
    while (chunk := ar_chunk_range(geometry, frame)) is not None:
        ranges.append(chunk)
        frame = chunk[1]
    return ranges


def control_chunk_ranges(
    geometry: ClipGeometry, *, first_frame: int = 0, count: int | None = None
) -> list[tuple[int, int]]:
    """Control ranges following the same partition the rollout generates in.

    :func:`ar_chunk_range` from ``first_frame``, which is frame 0 rather than
    the first generated frame: controls exist for the supplied frames too.

    A bounded cache holds these ranges one to a slot, so a slot is exactly the
    frames one chunk reads and a top-up replaces exactly one chunk's worth.
    Slots anchored at multiples of ``frames_per_chunk`` instead straddle two
    chunks, which drops control frames whose history is still cached and holds
    frames no query may read yet.

    Args:
        count: how many ranges to take, or ``None`` for the rest of the clip,
            which a clip with no end has to say a number for.
    """
    if count is None and geometry.frames_per_view is None:
        raise ValueError(
            "a clip with no end has no rest of the clip. Ask for a count of "
            "control ranges."
        )
    ranges: list[tuple[int, int]] = []
    frame = first_frame
    while count is None or len(ranges) < count:
        chunk = ar_chunk_range(geometry, frame)
        if chunk is None:
            break
        ranges.append(chunk)
        frame = chunk[1]
    return ranges


def control_window(geometry: ClipGeometry, history_slots: int) -> list[tuple[int, int]]:
    """Control ranges reaching one chunk past the history a cache keeps.

    A chunk reads the map for its own frames, which are not history yet, so
    control covers the retained chunks and the one being generated. Counted in
    ranges rather than frames because a clip can open with a range shorter than
    a chunk, which still takes a whole slot.
    """
    generated = control_chunk_ranges(
        geometry, first_frame=geometry.condition_frames, count=history_slots + 1
    )
    reach = generated[-1][1] if generated else geometry.frames_per_view
    assert reach is not None
    window: list[tuple[int, int]] = []
    frame = 0
    while (chunk := ar_chunk_range(geometry, frame)) is not None:
        window.append(chunk)
        frame = chunk[1]
        if frame >= reach:
            break
    return window


@dataclass(frozen=True)
class ChunkPlan:
    """What a rollout has to know about its chunks before it generates any.

    Attributes:
        count: chunks the clip takes, or ``None`` when it has no end.
        committed_frames: frames the cache could be asked to hold, being every
            frame the clip commits. The most history it could want to keep, and
            ``None`` when the clip has no end and so no most.

    Built by walking the clip once and keeping none of it, so nothing holds a
    list as long as the run. The step path asks :func:`ar_chunk_range` for one
    range at a time instead.
    """

    count: int | None
    committed_frames: int | None


def ar_chunk_plan(geometry: ClipGeometry) -> ChunkPlan:
    """Summarize a clip's chunks without keeping them.

    The last chunk is left out of ``committed_frames``: nothing attends to it
    afterwards, so it is never committed and the cache never holds it.
    Reserving for it would keep a slot that is never read, which is 9% of the
    cache at eleven views.

    A clip with no end is summarized as two ``None``s rather than walked. There
    is no last chunk to leave out and no total to reach, so a caller wanting a
    cache size has to bring its own bound.
    """
    if geometry.frames_per_view is None:
        return ChunkPlan(count=None, committed_frames=None)
    committed_frames = 0
    count = 0
    frame = geometry.condition_frames
    while (chunk := ar_chunk_range(geometry, frame)) is not None:
        start, end = chunk
        count += 1
        frame = end
        if end < geometry.frames_per_view:
            committed_frames += end - start
    return ChunkPlan(count=count, committed_frames=committed_frames)


def _padding(count: int, device: torch.device) -> StreamFields:
    """Fields for ``count`` padding tokens, which the mask isolates from everything."""
    sentinel = torch.full((count,), -1, dtype=torch.long, device=device)
    return StreamFields(
        sample_id=sentinel.clone(),
        frame_id=sentinel.clone(),
        view_id=sentinel.clone(),
        is_noisy=torch.zeros(count, dtype=torch.bool, device=device),
        is_control=torch.zeros(count, dtype=torch.bool, device=device),
        timestamp=torch.full((count,), -1.0, dtype=torch.float32, device=device),
        token_role_id=torch.full(
            (count,), ROLE_PADDING, dtype=torch.long, device=device
        ),
        causal_step_id=sentinel.clone(),
    )


def concat(*streams: StreamFields) -> StreamFields:
    """Join streams field by field, in order."""
    return StreamFields(
        **{
            field.name: torch.cat([getattr(stream, field.name) for stream in streams])
            for field in fields(StreamFields)
        }
    )


def _pad_to(
    stream: StreamFields, capacity: int | None, device: torch.device
) -> StreamFields:
    if capacity is None or capacity == len(stream):
        return stream
    if capacity < len(stream):
        raise ValueError(
            f"Capacity {capacity} is smaller than the {len(stream)} tokens it must hold."
        )
    return concat(stream, _padding(capacity - len(stream), device))


def _stamp_view_major(
    frames: Tensor, num_views: int, spatial_tokens: int
) -> tuple[Tensor, Tensor]:
    """Expand a frame range to per-token ``(frame_id, view_id)``, view-outer."""
    frame_id = frames.repeat(num_views).repeat_interleave(spatial_tokens)
    view_id = torch.arange(num_views, device=frames.device).repeat_interleave(
        frames.numel() * spatial_tokens
    )
    return frame_id, view_id


def _vision(
    frame_id: Tensor,
    view_id: Tensor,
    *,
    role: int | Tensor,
    is_noisy: bool | Tensor,
    geometry: ClipGeometry,
) -> StreamFields:
    """Fields for a block of vision tokens.

    ``role`` and ``is_noisy`` take either a scalar, for a block that is all one
    thing, or a per-token tensor. The prefill pack needs the latter: its target
    item interleaves conditioning frames with generated ones view by view, so the
    two roles cannot be split into separate blocks without losing the order.
    """
    count = frame_id.numel()
    device = frame_id.device
    role_id = (
        torch.as_tensor(role, dtype=torch.long, device=device)
        .expand(count)
        .contiguous()
    )
    noisy = (
        torch.as_tensor(is_noisy, dtype=torch.bool, device=device)
        .expand(count)
        .contiguous()
    )
    return StreamFields(
        sample_id=torch.zeros(count, dtype=torch.long, device=device),
        frame_id=frame_id,
        view_id=view_id,
        is_noisy=noisy,
        is_control=role_id == ROLE_CONTROL,
        timestamp=frame_id.to(torch.float32) * geometry.seconds_per_frame,
        token_role_id=role_id,
        causal_step_id=causal_steps(frame_id, geometry.frames_per_chunk),
    )


def text_stream(
    text_tokens: int,
    device: torch.device | None = None,
    *,
    view_text_tokens: Sequence[int] | None = None,
) -> StreamFields:
    """Fields for ``text_tokens`` prompt tokens.

    They carry a sample id and nothing else. Leaving every multiview field at
    the padding sentinel keeps the vision rules from matching them at all, so
    the only rule that admits the prompt is the one written for it.

    Args:
        view_text_tokens: tokens in each camera's own caption, in rig order,
            for a checkpoint trained one caption per camera. The runs sit in
            that order and each takes its camera's view id, which is what
            scopes it to that camera in :func:`visibility`. ``None`` leaves
            every token at the ``-1`` sentinel, a caption written for the whole
            rig, which every camera reads.

    Raises:
        ValueError: the runs do not add up to ``text_tokens``. They index the
            same token stream, so a disagreement would tag the wrong tokens.
    """
    device = device or torch.device("cpu")
    if view_text_tokens is None:
        view_id = torch.full((text_tokens,), -1, dtype=torch.long, device=device)
    else:
        if sum(view_text_tokens) != text_tokens:
            raise ValueError(
                f"per-view captions of {list(view_text_tokens)} tokens make "
                f"{sum(view_text_tokens)}, but the prompt is {text_tokens} tokens."
            )
        view_id = torch.repeat_interleave(
            torch.arange(len(view_text_tokens), dtype=torch.long, device=device),
            torch.tensor(list(view_text_tokens), dtype=torch.long, device=device),
        )
    return StreamFields(
        sample_id=torch.zeros(text_tokens, dtype=torch.long, device=device),
        frame_id=torch.full((text_tokens,), -1, dtype=torch.long, device=device),
        view_id=view_id,
        is_noisy=torch.zeros(text_tokens, dtype=torch.bool, device=device),
        is_control=torch.zeros(text_tokens, dtype=torch.bool, device=device),
        timestamp=torch.full((text_tokens,), -1.0, dtype=torch.float32, device=device),
        token_role_id=torch.full(
            (text_tokens,), ROLE_UND, dtype=torch.long, device=device
        ),
        causal_step_id=torch.full((text_tokens,), -1, dtype=torch.long, device=device),
    )


@dataclass(frozen=True)
class MemoryLayout:
    """The cached tokens: controls, then supplied frames, then generated history.

    Attributes:
        stream: per-token metadata, ``[M]``, padded out to capacity.
        real_token_count: tokens before the padding starts. Also where the next
            completed chunk's keys and values get written.

    Which tokens of the prefill pack end up cached is
    :attr:`~cosmos3_nano_sim_auto.impl.prefill.PrefillPack.memory_token_indexes` and nothing
    else. Deriving it a second time from the geometry is a mistake this class
    once made: the two strides agree only when no view is trimmed.
    """

    stream: StreamFields
    real_token_count: int

    @property
    def capacity(self) -> int:
        """Key/value slots the memory reserves, padding included."""
        return len(self.stream)


def _check_ranges(
    ranges: Sequence[tuple[int, int]],
    *,
    first_frame: int,
    last_frame: int | None,
    what: str,
) -> None:
    """Refuse frame ranges that cannot describe a cache.

    Order is not checked, because a bounded cache reuses slots and so holds its
    chunks rotated rather than oldest-first. What must hold is that the ranges
    are in bounds, do not overlap, and together cover one unbroken span of
    frames: a gap would leave the model attending across a hole in time without
    anything saying so.
    """
    if not ranges:
        return
    for start, end in ranges:
        if (
            end <= start
            or start < first_frame
            or (last_frame is not None and end > last_frame)
        ):
            raise ValueError(
                f"{what} range {(start, end)} is empty or outside frames "
                f"{first_frame}-{last_frame}."
            )
    for (_, previous_end), (start, _) in pairwise(sorted(ranges)):
        if start != previous_end:
            raise ValueError(
                f"{what} ranges must cover an unbroken span; {previous_end} to "
                f"{start} is {'a gap' if start > previous_end else 'an overlap'}. "
                f"Got {sorted(ranges)}."
            )


def build_memory_layout(
    geometry: ClipGeometry,
    *,
    control_frame_ranges: Sequence[tuple[int, int]] | None = None,
    control_slot_frames: int | None = None,
    history_frame_ranges: Sequence[tuple[int, int]] = (),
    history_slot_frames: int | None = None,
    capacity: int | None = None,
    device: torch.device | None = None,
) -> MemoryLayout:
    """Describe the cached controls, supplied frames and generated history.

    Rebuilt after each chunk. Both sets of ranges must be given in the order the
    cache physically holds them, which is chronological until a bounded cache
    starts reusing its oldest slot.

    ``control_frame_ranges`` defaults to the whole clip in one range, which is
    what a prefill covering every control frame produces. A bounded cache holds
    a window instead and extends it as the rollout advances.

    ``control_slot_frames`` pads each control section out to that many frames.
    A bounded cache reuses fixed slots, and the last top-up of a finite clip
    can be short of one; the rest of that slot holds keys nothing should read,
    so it is described as padding, which no real query can attend to.

    ``history_slot_frames`` does the same for the generated chunks. A clip whose
    first chunk is shorter than the rest -- text2video's frame 0, or what two
    supplied frames leave over -- puts a short chunk in a full-width slot.

    ``capacity`` reserves padded slots past the real tokens. A dense cuDNN mask
    does not need them and can simply grow; they are here for FlexAttention,
    which needs the mask shape to stop changing from chunk to chunk or it
    recompiles for each one.
    """
    device = device or torch.device("cpu")
    num_views, spatial_tokens = geometry.num_views, geometry.spatial_tokens

    def stamped(start: int, end: int, role: int) -> StreamFields:
        frame, view = _stamp_view_major(
            torch.arange(start, end, device=device), num_views, spatial_tokens
        )
        return _vision(frame, view, role=role, is_noisy=False, geometry=geometry)

    if control_frame_ranges is None:
        if geometry.frames_per_view is None:
            raise ValueError(
                "a clip with no end has no whole control track to default to. "
                "Pass the window the cache holds as control_frame_ranges."
            )
        control_frame_ranges = [(0, geometry.frames_per_view)]
    # The conditioning frames' map is pinned like their RGB, so as history
    # rotates the chunk ranges can sit apart from it, exactly as RGB history
    # sits apart from the conditioning frames. Each side must still be unbroken.
    opening = [r for r in control_frame_ranges if r[1] <= geometry.condition_frames]
    _check_ranges(
        opening, first_frame=0, last_frame=geometry.condition_frames, what="Control"
    )
    _check_ranges(
        [r for r in control_frame_ranges if r[1] > geometry.condition_frames],
        first_frame=max((end for _start, end in opening), default=0),
        last_frame=geometry.frames_per_view,
        what="Control",
    )
    _check_ranges(
        history_frame_ranges,
        first_frame=geometry.condition_frames,
        last_frame=geometry.frames_per_view,
        what="History",
    )

    sections = []
    for start, end in control_frame_ranges:
        sections.append(stamped(start, end, ROLE_CONTROL))
        short = (
            0 if control_slot_frames is None else control_slot_frames - (end - start)
        )
        if short < 0:
            raise ValueError(
                f"Control range {(start, end)} is longer than the "
                f"{control_slot_frames}-frame slot holding it."
            )
        if short:
            sections.append(_padding(short * num_views * spatial_tokens, device))
    if geometry.condition_frames:
        sections.append(stamped(0, geometry.condition_frames, ROLE_TARGET_CONDITION))
    for start, end in history_frame_ranges:
        sections.append(stamped(start, end, ROLE_CLEAN_TARGET))
        short = (
            0 if history_slot_frames is None else history_slot_frames - (end - start)
        )
        if short < 0:
            raise ValueError(
                f"History range {(start, end)} is longer than the "
                f"{history_slot_frames}-frame slot holding it."
            )
        if short:
            sections.append(_padding(short * num_views * spatial_tokens, device))

    real = concat(*sections)
    return MemoryLayout(
        stream=_pad_to(real, capacity, device),
        real_token_count=len(real),
    )


@dataclass(frozen=True)
class ChunkMetadata:
    """Query and key/value metadata for denoising one autoregressive chunk.

    The key/value stream is ``[text | current chunk | memory]``, so the chunk sees
    the prompt, itself and everything already generated. Queries are the current
    chunk alone -- nothing else is ever denoised.
    """

    query: StreamFields
    kv: StreamFields
    num_und: int
    memory_offset: int

    def mask(
        self,
        *,
        pattern: AttentionPattern = "causal",
        scope: AttentionScope = "all_views",
        decomposed_temporal_window_seconds: float | None = None,
    ) -> Tensor:
        """Return the ``[Q, KV]`` boolean visibility mask for this chunk."""
        return visibility(
            self.query,
            self.kv,
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
        """Return the same visibility as :meth:`mask`, as blocks FlexAttention can skip.

        At eleven views roughly half the blocks are empty, because a target
        reads controls only in its own view and the controls are most of the key
        stream. That half is the work this saves over tiling the dense mask.

        Built here rather than inside the layers: it is the same mask for all of
        them, and materializing it synchronizes with the host, which cannot
        happen inside a compiled region.
        """
        return build_block_mask(
            self.query,
            self.kv,
            pattern=pattern,
            scope=scope,
            decomposed_temporal_window_seconds=decomposed_temporal_window_seconds,
            block_size=block_size,
        )

    def fold_plan(
        self,
        *,
        pattern: AttentionPattern = "causal",
        scope: AttentionScope = "decomposed",
        decomposed_temporal_window_seconds: float | None = None,
        deduplicate_cross_view: bool = False,
    ) -> FoldPlan:
        """the model's passes as rectangles for an unmasked kernel. See :mod:`cosmos3_nano_sim_auto.impl.transformer.fold`."""
        return build_fold_plan(
            self.query,
            self.kv,
            pattern=pattern,
            scope=scope,
            decomposed_temporal_window_seconds=decomposed_temporal_window_seconds,
            deduplicate_cross_view=deduplicate_cross_view,
        )

    def maskless_fold(
        self,
        *,
        pattern: AttentionPattern = "causal",
        scope: AttentionScope = "decomposed",
        block_size: int | tuple[int, int] = 128,
        decomposed_temporal_window_seconds: float | None = None,
        deduplicate_cross_view: bool = False,
    ) -> MasklessFold:
        """The same passes as :meth:`fold_plan`, each gated by a block mask.

        Kept as what :meth:`fold_plan` is checked against, not as a way to run
        a clip. The two are the same arithmetic on different kernels, so they
        must agree to rounding, and a disagreement is a bug in the rectangles
        -- which is the only cheap way to find one, since the fold's own
        output differs from the mask's by design.
        """
        return MasklessFold(
            build_maskless_block_masks(
                self.query,
                self.kv,
                pattern=pattern,
                scope=scope,
                block_size=block_size,
                decomposed_temporal_window_seconds=decomposed_temporal_window_seconds,
                deduplicate_cross_view=deduplicate_cross_view,
            )
        )


def build_chunk_metadata(
    geometry: ClipGeometry,
    memory: MemoryLayout,
    *,
    chunk_start: int,
    chunk_frames: int,
    text_tokens: int,
    view_text_tokens: Sequence[int] | None = None,
    pass_kind: ChunkPassKind = "noisy",
    text_capacity: int | None = None,
    chunk_capacity: int | None = None,
    query_range: tuple[int, int] | None = None,
    device: torch.device | None = None,
) -> ChunkMetadata:
    """Build the metadata for the chunk covering frames ``[chunk_start, +chunk_frames)``.

    Frames are numbered where they fall in the clip rather than from zero,
    because the mask compares them against the cache, which is numbered that way
    throughout.

    ``text_tokens`` counts the prompt tokens prefixed to the key stream, and
    ``view_text_tokens`` splits them per camera when the prompt is one caption
    each rather than one for the rig. See :func:`text_stream`.

    ``query_range`` restricts the *queries* to ``[start, end)`` of the chunk
    while leaving the key stream whole, which is what one context-parallel rank
    needs: it denoises its own token range and attends every key there is. The
    key axis is unaffected, so the mask this returns lines up with the same
    cache every other rank is reading.

    ``pass_kind`` picks which pass this is. ``"noisy"`` denoises: the tokens are
    noisy and may read strictly earlier history. ``"clean"`` replays the
    finished chunk into the cache, and may also read its own step, itself
    included; running the replay under the noisy roles would cut the chunk off
    from itself. ``"control"`` adds control frames to a bounded cache and reads
    the prompt, earlier control in its own view and strictly earlier clean RGB
    in scope.
    """
    device = device or torch.device("cpu")
    if chunk_frames < 1:
        raise ValueError(f"chunk_frames must be >= 1, got {chunk_frames}.")
    # Control covers the frames it describes, conditioned or not; only a target
    # is barred from generating over frames that were given to it.
    if pass_kind != "control" and chunk_start < geometry.condition_frames:
        raise ValueError(
            f"Chunk starts at frame {chunk_start}, inside the {geometry.condition_frames} "
            "conditioning frames it should be generating after."
        )
    if (
        geometry.frames_per_view is not None
        and chunk_start + chunk_frames > geometry.frames_per_view
    ):
        raise ValueError(
            f"Chunk [{chunk_start}, {chunk_start + chunk_frames}) runs past the clip's "
            f"{geometry.frames_per_view} frames per view."
        )

    if pass_kind not in ("noisy", "clean", "control"):
        raise ValueError(
            f"pass_kind must be 'noisy', 'clean' or 'control', got {pass_kind!r}."
        )
    noisy = pass_kind == "noisy"
    control = pass_kind == "control"

    global_frames = torch.arange(chunk_start, chunk_start + chunk_frames, device=device)
    gen_frame, gen_view = _stamp_view_major(
        global_frames, geometry.num_views, geometry.spatial_tokens
    )
    query = _pad_to(
        _vision(
            gen_frame,
            gen_view,
            role=ROLE_CONTROL
            if control
            else (ROLE_CURRENT_TARGET if noisy else ROLE_CLEAN_TARGET),
            is_noisy=noisy,
            geometry=geometry,
        ),
        chunk_capacity,
        device,
    )

    text = _pad_to(
        text_stream(text_tokens, device, view_text_tokens=view_text_tokens),
        text_capacity,
        device,
    )

    # The key stream keeps the whole chunk however the queries were cut: a rank
    # denoising the last camera still attends the first one's tokens, and the
    # cache behind them is the same on every rank.
    return ChunkMetadata(
        query=query if query_range is None else query.narrow(*query_range),
        kv=concat(text, query, memory.stream),
        num_und=len(text),
        memory_offset=len(text) + len(query),
    )
