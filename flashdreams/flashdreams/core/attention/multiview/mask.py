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

"""Role-based visibility masks for multi-view autoregressive attention."""

from __future__ import annotations

import functools
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor
from torch.nn.attention.flex_attention import BlockMask, create_block_mask

# Stable token-role identifiers shared by packing and visibility rules.
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


def visibility(
    q: StreamFields,
    kv: StreamFields,
    *,
    pattern: AttentionPattern = "causal",
    scope: AttentionScope = "all_views",
    decomposed_temporal_window_seconds: float | None = None,
) -> Tensor:
    """Return the ``[Q, KV]`` boolean mask: ``True`` where the query may attend the key.

    The defaults use causal history with visibility across all views. Alternate
    scopes support view-local and time-bounded cross-view attention.
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

    # A chunk being denoised reads *strictly* earlier history; the same chunk on
    # its clean replay also reads its own step. That asymmetry is what makes the
    # replay worth doing, and the easiest thing here to get backwards.
    clean_step_allowed = (q_is_current & (kv_step < q_step)) | (
        ~q_is_current & (kv_step <= q_step)
    )
    control_step_allowed = (
        (kv_step <= q_step) if control_uses_history else (kv_step == q_step)
    )

    allowed = (
        (q_is_target & (kv_role == ROLE_UND))
        | (q_is_condition & (kv_role == ROLE_UND))
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

    padding_to_padding = (q_role == ROLE_PADDING) & (kv_role == ROLE_PADDING)
    return (same_sample & allowed) | padding_to_padding


def visibility_mask_mod(
    q: StreamFields,
    kv: StreamFields,
    *,
    pattern: AttentionPattern = "causal",
    scope: AttentionScope = "all_views",
    decomposed_temporal_window_seconds: float | None = None,
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

        allowed = (
            (q_is_target & (kv_role == ROLE_UND))
            | (q_is_condition & (kv_role == ROLE_UND))
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

        padding_to_padding = (q_role == ROLE_PADDING) & (kv_role == ROLE_PADDING)
        return (same_sample & allowed) | padding_to_padding

    return mask_mod


@functools.cache
def _compiled_create_block_mask() -> Callable[..., BlockMask]:
    """``create_block_mask``, compiled once.

    Compiled because eager ``create_block_mask`` calls ``create_mask``, which
    materializes the whole ``[Q, KV]`` bool and then runs the predicate's dozen
    boolean terms across it -- the exact cost the block mask exists to avoid.
    Measured at four views: 7.2 GiB eager against 0.77 GiB compiled, for a mask
    whose dense form is 0.66 GiB. Compiling fuses the predicate into the block
    reduction and avoids writing the full grid.
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
        block_size: query and key/value tiles the kernel skips at. One integer
            selects square blocks; a pair selects asymmetric ``(Q, KV)`` blocks.
            A mask smaller than one block has nothing to skip, which only
            matters to tests.

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
