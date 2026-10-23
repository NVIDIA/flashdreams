# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Multiview attention visibility for Cosmos3 Nano Sim Auto.

:func:`visibility` defines the dense correctness mask. Decomposed and maskless
plans preserve those rules using the token metadata supplied by
:mod:`cosmos3_nano_sim_auto.impl.packing`.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from dataclasses import dataclass, fields
from typing import Literal

import torch
from torch import Tensor
from torch.nn.attention.flex_attention import BlockMask, create_block_mask

LIDAR_VIEW_ID = 1_000_000
"""Sensor identity outside the camera rig; LiDAR reads every camera caption."""

# Stable role IDs used by token packing and attention visibility.
ROLE_PADDING = -1
ROLE_UND = 0
ROLE_CONTROL = 1
ROLE_TARGET_CONDITION = 2
ROLE_CURRENT_TARGET = 3
ROLE_CLEAN_TARGET = 4

AttentionScope = Literal["same_view", "decomposed", "all_views"]
AttentionPattern = Literal["causal", "current_control"]

# Timestamps are floats compared against a window bound, so the window edge needs
# a tolerance or a frame that lands exactly on it falls out at random.
_TIMESTAMP_EPS = 1e-4


@dataclass(frozen=True)
class StreamFields:
    """Per-token metadata for one side of the mask. Every tensor is ``[S]``."""

    sample_id: Tensor
    frame_id: Tensor
    view_id: Tensor
    is_noisy: Tensor
    is_control: Tensor
    timestamp: Tensor
    token_role_id: Tensor
    causal_step_id: Tensor

    def __len__(self) -> int:
        return int(self.sample_id.shape[0])

    def narrow(self, start: int, end: int) -> StreamFields:
        """Tokens ``[start, end)`` of this stream, field by field.

        How a context-parallel rank names the query rows it owns: the mask it
        builds is then ``[local queries, every key]``, which is the shape its
        share of the pass actually needs. Views throughout, so nothing is copied.

        Raises:
            ValueError: the range is empty or reaches past the stream.
        """
        if not 0 <= start < end <= len(self):
            raise ValueError(
                f"[{start}, {end}) is not a range of a {len(self)}-token stream."
            )
        return StreamFields(
            **{
                field.name: getattr(self, field.name)[start:end]
                for field in fields(StreamFields)
            }
        )


def visibility(
    q: StreamFields,
    kv: StreamFields,
    *,
    pattern: AttentionPattern = "causal",
    scope: AttentionScope = "all_views",
    decomposed_temporal_window_seconds: float | None = None,
    controls_read_clean_rgb: bool = True,
    joint: bool = False,
) -> Tensor:
    """Return the ``[Q, KV]`` boolean mask: ``True`` where the query may attend the key.

    The caller selects the checkpoint's attention scope and temporal pattern.

    ``controls_read_clean_rgb`` lets a control read the clean RGB of earlier
    steps in scope. Disabling it restricts controls to the prompt and earlier
    control tokens.
    """
    control_uses_history = pattern == "causal"
    reaches_every_view = scope == "all_views"
    is_decomposed = scope == "decomposed"
    window = decomposed_temporal_window_seconds

    # Broadcast to [Q, 1] against [1, KV] so every term below is [Q, KV].
    def q_col(tensor: Tensor) -> Tensor:
        return tensor.unsqueeze(1)

    def kv_row(tensor: Tensor) -> Tensor:
        return tensor.unsqueeze(0)

    same_sample = q_col(q.sample_id) == kv_row(kv.sample_id)
    same_frame = q_col(q.frame_id) == kv_row(kv.frame_id)
    same_view = q_col(q.view_id) == kv_row(kv.view_id)

    if is_decomposed and window is not None:
        gap = q_col(q.timestamp) - kv_row(kv.timestamp)
        reaches_own_instant = (gap >= -_TIMESTAMP_EPS) & (
            gap <= window + _TIMESTAMP_EPS
        )
    elif is_decomposed:
        reaches_own_instant = same_frame
    else:
        reaches_own_instant = torch.zeros_like(same_frame)

    in_scope = same_view | reaches_own_instant
    if reaches_every_view:
        in_scope = torch.ones_like(in_scope)

    q_step, kv_step = q_col(q.causal_step_id), kv_row(kv.causal_step_id)
    q_role, kv_role = q_col(q.token_role_id), kv_row(kv.token_role_id)

    q_is_current = q_role == ROLE_CURRENT_TARGET
    q_is_target = q_is_current | (q_role == ROLE_CLEAN_TARGET)
    q_is_control = q_role == ROLE_CONTROL
    q_is_target_condition = q_role == ROLE_TARGET_CONDITION
    q_is_condition = q_is_control | q_is_target_condition
    kv_is_noisy_rgb = kv_row(kv.is_noisy & ~kv.is_control)
    kv_is_clean_rgb = (kv_role == ROLE_TARGET_CONDITION) | (
        kv_role == ROLE_CLEAN_TARGET
    )

    # A chunk being denoised reads *strictly* earlier history; the same chunk on
    # its clean replay also reads its own step. That asymmetry is what makes the
    # replay worth doing, and the easiest thing here to get backwards.
    clean_step_allowed = (q_is_current & (kv_step < q_step)) | (
        ~q_is_current & (kv_step <= q_step)
    )
    control_step_allowed = (
        (kv_step <= q_step) if control_uses_history else (kv_step == q_step)
    )

    # A caption written for one camera is read by that camera alone; one written
    # for the whole rig carries no view and is read by all of them. Caption
    # tokens carry their own view scope.
    caption_reaches = (
        kv_row(kv.view_id < 0) | same_view | q_col(q.view_id == LIDAR_VIEW_ID)
    )

    allowed = (
        (q_is_target & (kv_role == ROLE_UND) & caption_reaches)
        | (q_is_condition & (kv_role == ROLE_UND) & caption_reaches)
        | (q_is_control & (kv_role == ROLE_CONTROL) & same_view & (kv_step <= q_step))
        | (
            q_is_target_condition
            & (kv_role == ROLE_TARGET_CONDITION)
            & same_frame
            & same_view
        )
        | (q_is_target_condition & kv_is_noisy_rgb & in_scope)
        | (
            q_is_current
            & (kv_role == ROLE_CURRENT_TARGET)
            & (kv_step == q_step)
            & in_scope
        )
        | (q_is_target & (kv_role == ROLE_CLEAN_TARGET) & clean_step_allowed & in_scope)
        | (q_is_target & (kv_role == ROLE_CONTROL) & same_view & control_step_allowed)
        | (
            q_is_target
            & (kv_role == ROLE_TARGET_CONDITION)
            & same_view
            & (kv_step <= q_step)
        )
    )
    if joint:
        allowed = (
            allowed
            | (
                q_is_target_condition
                & (kv_role == ROLE_CONTROL)
                & same_view
                & control_step_allowed
            )
            | (
                q_is_target
                & (kv_role == ROLE_TARGET_CONDITION)
                & in_scope
                & (kv_step <= q_step)
            )
        )
        allowed = allowed & ~(
            q_is_target_condition & kv_is_noisy_rgb & (kv_step >= q_step)
        )
    if controls_read_clean_rgb:
        allowed = allowed | (
            q_is_control & kv_is_clean_rgb & (kv_step < q_step) & in_scope
        )

    padding_to_padding = (q_role == ROLE_PADDING) & (kv_role == ROLE_PADDING)
    return (same_sample & allowed) | padding_to_padding


def visibility_mask_mod(
    q: StreamFields,
    kv: StreamFields,
    *,
    pattern: AttentionPattern = "causal",
    scope: AttentionScope = "all_views",
    decomposed_temporal_window_seconds: float | None = None,
    controls_read_clean_rgb: bool = True,
) -> Callable[[Tensor, Tensor, Tensor, Tensor], Tensor]:
    """The same rules as :func:`visibility`, as a FlexAttention ``mask_mod``.

    :func:`visibility` broadcasts the whole ``[Q, KV]`` tensor; this answers one
    token pair at a time, so FlexAttention can skip blocks instead of holding
    the matrix. Deliberately kept in this file and in the same order as the
    rules above: two statements of one thing drift, and ``test_mask.py`` checks
    them against each other rather than trusting that they cannot.

    Returns:
        ``mask_mod(b, h, q_idx, kv_idx)``, for ``create_block_mask``.
    """
    reaches_every_view = scope == "all_views"
    is_decomposed = scope == "decomposed"
    control_uses_history = pattern == "causal"
    window = decomposed_temporal_window_seconds

    def mask_mod(b: Tensor, h: Tensor, q_idx: Tensor, kv_idx: Tensor) -> Tensor:
        del b, h
        same_sample = q.sample_id[q_idx] == kv.sample_id[kv_idx]
        same_frame = q.frame_id[q_idx] == kv.frame_id[kv_idx]
        same_view = q.view_id[q_idx] == kv.view_id[kv_idx]

        # `scope` is configuration, not data, so these branch in Python and
        # `all_views` drops the term rather than anding against a true scalar.
        def scoped(term: Tensor) -> Tensor:
            if reaches_every_view:
                return term
            if is_decomposed and window is not None:
                gap = q.timestamp[q_idx] - kv.timestamp[kv_idx]
                near = (gap >= -_TIMESTAMP_EPS) & (gap <= window + _TIMESTAMP_EPS)
                return term & (same_view | near)
            if is_decomposed:
                return term & (same_view | same_frame)
            return term & same_view

        q_step, kv_step = q.causal_step_id[q_idx], kv.causal_step_id[kv_idx]
        q_role, kv_role = q.token_role_id[q_idx], kv.token_role_id[kv_idx]

        q_is_current = q_role == ROLE_CURRENT_TARGET
        q_is_target = q_is_current | (q_role == ROLE_CLEAN_TARGET)
        q_is_control = q_role == ROLE_CONTROL
        q_is_target_condition = q_role == ROLE_TARGET_CONDITION
        q_is_condition = q_is_control | q_is_target_condition
        kv_is_noisy_rgb = kv.is_noisy[kv_idx] & ~kv.is_control[kv_idx]

        clean_step_allowed = (q_is_current & (kv_step < q_step)) | (
            ~q_is_current & (kv_step <= q_step)
        )
        control_step_allowed = (
            (kv_step <= q_step) if control_uses_history else (kv_step == q_step)
        )

        # See :func:`visibility`, whose rule this has to match token for token.
        caption_reaches = (
            (kv.view_id[kv_idx] < 0) | same_view | (q.view_id[q_idx] == LIDAR_VIEW_ID)
        )

        allowed = (
            (q_is_target & (kv_role == ROLE_UND) & caption_reaches)
            | (q_is_condition & (kv_role == ROLE_UND) & caption_reaches)
            | (
                q_is_control
                & (kv_role == ROLE_CONTROL)
                & same_view
                & (kv_step <= q_step)
            )
            | (
                q_is_target_condition
                & (kv_role == ROLE_TARGET_CONDITION)
                & same_frame
                & same_view
            )
            | scoped(q_is_target_condition & kv_is_noisy_rgb)
            | scoped(
                q_is_current & (kv_role == ROLE_CURRENT_TARGET) & (kv_step == q_step)
            )
            | scoped(q_is_target & (kv_role == ROLE_CLEAN_TARGET) & clean_step_allowed)
            | (
                q_is_target
                & (kv_role == ROLE_CONTROL)
                & same_view
                & control_step_allowed
            )
            | (
                q_is_target
                & (kv_role == ROLE_TARGET_CONDITION)
                & same_view
                & (kv_step <= q_step)
            )
        )
        if controls_read_clean_rgb:
            clean_rgb = (kv_role == ROLE_TARGET_CONDITION) | (
                kv_role == ROLE_CLEAN_TARGET
            )
            allowed = allowed | scoped(q_is_control & clean_rgb & (kv_step < q_step))

        padding_to_padding = (q_role == ROLE_PADDING) & (kv_role == ROLE_PADDING)
        return (same_sample & allowed) | padding_to_padding

    return mask_mod


MASKLESS_SCOPES: tuple[AttentionScope, ...] = ("same_view", "decomposed")
"""Scopes the fold below expresses. ``all_views`` is absent for a reason.

``all_views`` is not a partition of the key stream but one unmasked pass over
the whole of it, so there is nothing to fold and nothing to merge. It stays a
mask rule.
"""

PartitionName = Literal["same_view", "cross_instant", "caption"]


def _rig_reaches_across(kv: StreamFields) -> bool:
    """Whether cross-view attention contributes keys beyond the same-view pass.

    A single-view rig has no additional cross-view keys; a second pass would
    reweight keys already read. View identities are taken from the token stream.
    """
    return int(kv.view_id[kv.view_id >= 0].unique().numel()) > 1


def maskless_partitions(
    q: StreamFields,
    kv: StreamFields,
    *,
    pattern: AttentionPattern = "causal",
    scope: AttentionScope = "decomposed",
    decomposed_temporal_window_seconds: float | None = None,
    joint: bool = False,
    deduplicate_cross_view: bool = False,
) -> tuple[tuple[PartitionName, Tensor], ...]:
    """Partition :func:`visibility` into named maskless attention passes.

    The same-view pass follows temporal causality. The cross-view pass covers the
    query's instant or the configured temporal window. With
    ``deduplicate_cross_view=True``, it excludes the query's own view and the
    cross-view control-query edges. Legacy checkpoints retain overlapping keys,
    so merging those passes counts an overlapping key once per pass.

    Returns:
        ``(name, [Q, KV] bool)`` per pass, in merge order.
    """
    if scope not in MASKLESS_SCOPES:
        raise ValueError(
            f"scope={scope!r} is not one the maskless folds express; they cover "
            f"{MASKLESS_SCOPES}. 'all_views' is one unmasked pass over the whole "
            "key stream rather than a partition of it, so it stays a mask rule"
        )

    # Widened to all_views deliberately: each pass below reimposes the spatial
    # rule itself, and taking the narrowed mask here would apply it twice.
    base = visibility(q, kv, pattern=pattern, scope="all_views", joint=joint)

    caption_key = (kv.token_role_id == ROLE_UND).unsqueeze(0)
    padding = (q.token_role_id == ROLE_PADDING).unsqueeze(1) & (
        kv.token_role_id == ROLE_PADDING
    ).unsqueeze(0)
    same_view = q.view_id.unsqueeze(1) == kv.view_id.unsqueeze(0)

    # Padding rides with the same-view pass so a padded row still has a key.
    # A row with no keys anywhere is a -inf log-sum-exp, and merging those is
    # the one way this returns NaN rather than a picture.
    partitions: list[tuple[PartitionName, Tensor]] = [
        ("same_view", base & ~caption_key & (same_view | padding))
    ]
    if scope == "decomposed" and _rig_reaches_across(kv):
        # the model's instant is a camera-anchored midpoint bucket, which collapses to
        # the frame index for a single sensor running at the camera's own rate.
        # That is this rig, and only this rig: a second sensor at a different
        # rate would need the bucket, and would be wrong read this way.
        if decomposed_temporal_window_seconds is None:
            same_instant = q.frame_id.unsqueeze(1) == kv.frame_id.unsqueeze(0)
        else:
            gap = q.timestamp.unsqueeze(1) - kv.timestamp.unsqueeze(0)
            same_instant = (gap >= -_TIMESTAMP_EPS) & (
                gap <= decomposed_temporal_window_seconds + _TIMESTAMP_EPS
            )
        # Upstream maskless replay excludes control queries from this pass.
        # They read strictly earlier clean history through their own view only.
        partitions.append(
            (
                "cross_instant",
                base
                & ~caption_key
                & ~padding
                & ~q.is_control.unsqueeze(1)
                & ~kv.is_control.unsqueeze(0)
                & same_instant
                & (
                    ~same_view if deduplicate_cross_view else torch.ones_like(same_view)
                ),
            )
        )
    partitions.append(("caption", base & caption_key))
    return tuple(partitions)


def build_maskless_block_masks(
    q: StreamFields,
    kv: StreamFields,
    *,
    pattern: AttentionPattern = "causal",
    scope: AttentionScope = "decomposed",
    block_size: int | tuple[int, int] = 128,
    decomposed_temporal_window_seconds: float | None = None,
    deduplicate_cross_view: bool = False,
) -> tuple[BlockMask, ...]:
    """:func:`maskless_partitions` as blocks, one ``BlockMask`` a pass.

    The passes are far sparser than the single mask they replace -- a frame's
    own instant is a thin diagonal stripe -- so the block skipping matters more
    here than it does for the union, not less.
    """
    dense = maskless_partitions(
        q,
        kv,
        pattern=pattern,
        scope=scope,
        decomposed_temporal_window_seconds=decomposed_temporal_window_seconds,
        deduplicate_cross_view=deduplicate_cross_view,
    )
    return tuple(
        _block_mask_of(partition, device=q.sample_id.device, block_size=block_size)
        for _name, partition in dense
    )


def _block_mask_of(
    dense: Tensor,
    *,
    device: torch.device,
    block_size: int | tuple[int, int],
) -> BlockMask:
    """A ``BlockMask`` reading one already-built ``[Q, KV]`` tensor.

    The partitions are cheaper to derive from each other than to restate as
    three predicates, so unlike :func:`build_block_mask` this indexes the dense
    form rather than recomputing the rules per pair. The dense tensor is the
    same one :func:`maskless_partitions` returns, so the two can never disagree.
    """

    def mask_mod(b: Tensor, h: Tensor, q_idx: Tensor, kv_idx: Tensor) -> Tensor:
        del b, h
        return dense[q_idx, kv_idx]

    return _compiled_create_block_mask()(
        mask_mod,
        B=None,
        H=None,
        Q_LEN=int(dense.shape[0]),
        KV_LEN=int(dense.shape[1]),
        device=device,
        BLOCK_SIZE=block_size,
    )


@functools.cache
def _compiled_create_block_mask() -> Callable[..., BlockMask]:
    """``create_block_mask``, compiled once.

    Compiled because eager ``create_block_mask`` calls ``create_mask``, which
    materializes the whole ``[Q, KV]`` bool and then runs the predicate's dozen
    boolean terms across it -- the exact cost the block mask exists to avoid.
    Inductor fuses the predicate into the block reduction without writing
    the dense grid.
    """
    return torch.compile(create_block_mask)


def build_block_mask(
    q: StreamFields,
    kv: StreamFields,
    *,
    pattern: AttentionPattern = "causal",
    scope: AttentionScope = "all_views",
    decomposed_temporal_window_seconds: float | None = None,
    block_size: int | tuple[int, int] = 128,
) -> BlockMask:
    """The rules of :func:`visibility`, as blocks FlexAttention can skip.

    Args:
        q, kv: the two token streams, as :func:`visibility` takes them.
        block_size: the square tile, or independent query and key/value tiles,
            the kernel skips at. A mask smaller than one block has nothing to
            skip, which only matters to tests.

    Returns:
        A ``BlockMask`` over ``[len(q), len(kv)]``.
    """
    return _compiled_create_block_mask()(
        visibility_mask_mod(
            q,
            kv,
            pattern=pattern,
            scope=scope,
            decomposed_temporal_window_seconds=decomposed_temporal_window_seconds,
        ),
        B=None,
        H=None,
        Q_LEN=len(q),
        KV_LEN=len(kv),
        device=q.sample_id.device,
        BLOCK_SIZE=block_size,
    )
