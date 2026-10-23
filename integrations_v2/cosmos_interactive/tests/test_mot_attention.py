# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the MoT joint-attention visibility mask + masked SDPA.

Validates (GPU-free, ci_cpu):
  * the visibility rule (und causal-among-und, no gen; gen sees all und + gen
    up to its own frame), and
  * that ``mot_joint_attention`` (SDPA + mask) matches an explicit-softmax
    reference on random inputs.
"""

from __future__ import annotations

import pytest
import torch
from cosmos_interactive._attention import (
    build_mot_visibility_mask,
    mot_joint_attention,
    reference_masked_attention,
)

pytestmark = pytest.mark.ci_cpu


def _toy_layout() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """2 text tokens, then 2 frames x 2 spatial gen tokens (S=6)."""
    is_gen = torch.tensor([False, False, True, True, True, True])
    # Text order 0,1; gen tokens after text keep increasing order (unused for gen rows).
    order = torch.tensor([0, 1, 2, 2, 3, 3])
    # Frame ids: text = -1 (irrelevant); gen frame0 (idx 2,3), frame1 (idx 4,5).
    frame = torch.tensor([-1, -1, 0, 0, 1, 1])
    return is_gen, order, frame


def test_visibility_rule() -> None:
    is_gen, order, frame = _toy_layout()
    m = build_mot_visibility_mask(is_gen, order, frame)

    # Text token 0: sees only text 0 (causal, no gen).
    assert m[0].tolist() == [True, False, False, False, False, False]
    # Text token 1: sees text 0,1; no gen.
    assert m[1].tolist() == [True, True, False, False, False, False]
    # Gen frame0 token (idx 2): sees both text + gen frame0 (idx 2,3); NOT frame1.
    assert m[2].tolist() == [True, True, True, True, False, False]
    # Gen frame1 token (idx 4): sees text + gen frame0 + gen frame1 (all gen so far).
    assert m[4].tolist() == [True, True, True, True, True, True]


def test_no_gen_query_sees_future_frame() -> None:
    is_gen, order, frame = _toy_layout()
    m = build_mot_visibility_mask(is_gen, order, frame)
    # No gen query attends to a strictly-later frame.
    for qi in range(6):
        if not is_gen[qi]:
            continue
        for kj in range(6):
            if is_gen[kj] and frame[kj] > frame[qi]:
                assert not m[qi, kj]


def test_masked_sdpa_matches_reference() -> None:
    torch.manual_seed(0)
    is_gen, order, frame = _toy_layout()
    mask = build_mot_visibility_mask(is_gen, order, frame)
    b, h, s, d = 2, 4, 6, 16
    q = torch.randn(b, h, s, d, dtype=torch.float32)
    k = torch.randn(b, h, s, d, dtype=torch.float32)
    v = torch.randn(b, h, s, d, dtype=torch.float32)

    out = mot_joint_attention(q, k, v, mask)
    ref = reference_masked_attention(q, k, v, mask)
    max_abs = (out - ref).abs().max().item()
    assert max_abs < 1e-5, f"masked SDPA diverges from reference: {max_abs}"


def test_all_same_frame_reduces_to_two_way() -> None:
    """With all gen in one frame, gen sees all und + all gen (two-way full)."""
    is_gen = torch.tensor([False, True, True])
    order = torch.tensor([0, 1, 1])
    frame = torch.tensor([-1, 0, 0])
    m = build_mot_visibility_mask(is_gen, order, frame)
    # Both gen tokens see everything.
    assert m[1].tolist() == [True, True, True]
    assert m[2].tolist() == [True, True, True]
