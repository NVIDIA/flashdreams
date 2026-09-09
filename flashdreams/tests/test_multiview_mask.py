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

"""CPU tests for multi-view attention visibility."""

from __future__ import annotations

import pytest
import torch

from flashdreams.core.attention.multiview.mask import (
    ROLE_CLEAN_TARGET,
    ROLE_CONTROL,
    ROLE_CURRENT_TARGET,
    ROLE_PADDING,
    ROLE_TARGET_CONDITION,
    ROLE_UND,
    StreamFields,
    _compiled_create_block_mask,
    visibility,
    visibility_mask_mod,
)

pytestmark = pytest.mark.ci_cpu

SECONDS_PER_FRAME = 0.5


def build_stream(
    tokens: list[tuple[int, int, int, int]], sample: int = 0
) -> StreamFields:
    """Build a stream from ``(role, frame, view, causal_step)`` rows.

    ``is_noisy`` follows the role the way the packer will set it: current targets
    are the noisy RGB being denoised, everything else is clean or conditioning.
    Timestamps derive from the frame index at a fixed ``seconds_per_frame``.
    """
    roles = torch.tensor([t[0] for t in tokens])
    frames = torch.tensor([t[1] for t in tokens])
    views = torch.tensor([t[2] for t in tokens])
    steps = torch.tensor([t[3] for t in tokens])
    return StreamFields(
        sample_id=torch.full((len(tokens),), sample, dtype=torch.long),
        frame_id=frames,
        view_id=views,
        is_noisy=roles == ROLE_CURRENT_TARGET,
        is_control=roles == ROLE_CONTROL,
        timestamp=frames.to(torch.float32) * SECONDS_PER_FRAME,
        token_role_id=roles,
        causal_step_id=steps,
    )


# One two-view, three-frame sample at one frame per chunk, so causal_step == frame.
def _sample_tokens() -> list[tuple[int, int, int, int]]:
    tokens: list[tuple[int, int, int, int]] = [(ROLE_UND, 0, 0, 0)]
    for frame in range(3):
        for view in range(2):
            tokens.append((ROLE_CONTROL, frame, view, frame))
            tokens.append((ROLE_CURRENT_TARGET, frame, view, frame))
            tokens.append((ROLE_CLEAN_TARGET, frame, view, frame))
    for view in range(2):
        tokens.append((ROLE_TARGET_CONDITION, 0, view, 0))
    return tokens


def index_of(
    tokens: list[tuple[int, int, int, int]], role: int, frame: int, view: int
) -> int:
    matches = [
        i
        for i, t in enumerate(tokens)
        if t[0] == role and t[1] == frame and t[2] == view
    ]
    assert len(matches) == 1, (
        f"expected exactly one {role=} {frame=} {view=}, got {matches}"
    )
    return matches[0]


@pytest.fixture
def tokens() -> list[tuple[int, int, int, int]]:
    return _sample_tokens()


def dense_from_mask_mod(stream: StreamFields, **kwargs) -> torch.Tensor:
    """Expand the FlexAttention predicate back to a ``[Q, KV]`` tensor.

    Only a test does this -- expanding is the cost the predicate exists to
    avoid -- but it is the one way to compare the two against each other.
    """
    flex = pytest.importorskip(
        "torch.nn.attention.flex_attention", reason="needs FlexAttention"
    )
    mod = visibility_mask_mod(stream, stream, **kwargs)
    tokens = len(stream)
    expanded = flex.create_mask(
        mod, B=None, H=None, Q_LEN=tokens, KV_LEN=tokens, device=stream.sample_id.device
    )
    return expanded[0, 0]


@pytest.mark.parametrize("pattern", ["causal", "current_control"])
@pytest.mark.parametrize("scope", ["same_view", "decomposed", "all_views"])
def test_the_mask_mod_says_exactly_what_the_dense_mask_says(
    tokens, pattern, scope
) -> None:
    """The FlexAttention predicate and the dense tensor are one set of rules.

    They are written twice, in the same file and the same order, because the
    kernel wants one token pair at a time and cuDNN wants the matrix. Nothing
    but this test stops the two from drifting, and a drift would not raise --
    it would quietly change what the model can see.
    """
    stream = build_stream(tokens)
    dense = visibility(stream, stream, pattern=pattern, scope=scope)

    assert torch.equal(dense_from_mask_mod(stream, pattern=pattern, scope=scope), dense)


def test_the_mask_mod_agrees_with_the_dense_mask_on_padding() -> None:
    """Padding is the one rule that ignores the others, so it is checked apart.

    A padded key is not merely invisible: pad attends pad, or a query whose row
    is entirely false makes softmax produce NaN rather than zero.
    """
    tokens = [
        (ROLE_UND, 0, 0, 0),
        (ROLE_CURRENT_TARGET, 0, 0, 0),
        (ROLE_PADDING, -1, -1, -1),
    ]
    stream = build_stream(tokens)

    assert torch.equal(dense_from_mask_mod(stream), visibility(stream, stream))


@pytest.mark.parametrize(
    ("pattern", "sees_control_history"), [("causal", True), ("current_control", False)]
)
def test_current_target_visibility(tokens, pattern, sees_control_history) -> None:
    """Current targets read same-step noise and strictly earlier clean history.

    Exercise noisy and clean replay visibility at the same causal step.
    """
    stream = build_stream(tokens)
    mask = visibility(stream, stream, pattern=pattern, scope="all_views")
    q = index_of(tokens, ROLE_CURRENT_TARGET, frame=2, view=0)

    assert mask[q, 0].item() is True  # the understanding token
    assert mask[q, index_of(tokens, ROLE_CONTROL, 1, 0)].item() is sees_control_history
    assert mask[q, index_of(tokens, ROLE_CONTROL, 2, 0)].item() is True
    assert mask[q, index_of(tokens, ROLE_CURRENT_TARGET, 2, 1)].item() is True
    assert mask[q, index_of(tokens, ROLE_CURRENT_TARGET, 1, 0)].item() is False
    assert mask[q, index_of(tokens, ROLE_CLEAN_TARGET, 1, 1)].item() is True
    assert mask[q, index_of(tokens, ROLE_CLEAN_TARGET, 2, 0)].item() is False


def test_control_queries_are_causal_and_view_local(tokens) -> None:
    """A control token sees its own view's control history, never other views or its own step."""
    stream = build_stream(tokens)
    mask = visibility(stream, stream, pattern="causal", scope="all_views")
    q = index_of(tokens, ROLE_CONTROL, frame=1, view=0)

    assert mask[q, index_of(tokens, ROLE_CONTROL, 0, 0)].item() is True
    assert mask[q, index_of(tokens, ROLE_CONTROL, 1, 0)].item() is True
    assert mask[q, index_of(tokens, ROLE_CONTROL, 2, 0)].item() is False
    assert mask[q, index_of(tokens, ROLE_CONTROL, 1, 1)].item() is False


def test_target_condition_reads_noisy_rgb_but_not_clean_memory(tokens) -> None:
    """Let conditioning queries read current noisy targets but not clean replay."""
    stream = build_stream(tokens)
    mask = visibility(stream, stream, pattern="causal", scope="all_views")
    q = index_of(tokens, ROLE_TARGET_CONDITION, frame=0, view=0)

    assert mask[q, index_of(tokens, ROLE_CURRENT_TARGET, 2, 1)].item() is True
    assert mask[q, index_of(tokens, ROLE_CLEAN_TARGET, 1, 1)].item() is False


@pytest.mark.parametrize(
    ("scope", "sees_same_frame_other_view", "sees_other_view_same_step"),
    [
        ("same_view", False, False),
        ("decomposed", True, False),
        ("all_views", True, True),
    ],
)
def test_scope_restricts_cross_view_relations(
    scope, sees_same_frame_other_view, sees_other_view_same_step
) -> None:
    """Replay causality intersects with the configured view scope.

    Exercise current and clean target relations under each attention scope.
    Two frames per chunk here, so ``causal_step = frame // 2`` and frames 2 and 3
    share a step while frame 1 sits in the one before.
    """
    tokens: list[tuple[int, int, int, int]] = [(ROLE_UND, 0, 0, 0)]
    for frame in range(4):
        for view in range(2):
            tokens.append((ROLE_CURRENT_TARGET, frame, view, frame // 2))
            tokens.append((ROLE_CLEAN_TARGET, frame, view, frame // 2))
    stream = build_stream(tokens)
    mask = visibility(stream, stream, pattern="causal", scope=scope)

    q = index_of(tokens, ROLE_CURRENT_TARGET, frame=2, view=0)
    same_frame_other_view = index_of(tokens, ROLE_CURRENT_TARGET, 3, 1)
    other_view_earlier_step = index_of(tokens, ROLE_CURRENT_TARGET, 1, 1)

    # Frame 3 view 1 shares the causal step with the query but is a different
    # frame and view, so only all_views admits it; decomposed needs same frame.
    assert mask[q, same_frame_other_view].item() is sees_other_view_same_step
    assert mask[q, index_of(tokens, ROLE_CURRENT_TARGET, 2, 1)].item() is (
        sees_same_frame_other_view or sees_other_view_same_step
    )
    assert mask[q, other_view_earlier_step].item() is False
    assert mask[q, index_of(tokens, ROLE_CURRENT_TARGET, 2, 0)].item() is True


# Not covered here: the timestamp-window branch of ``decomposed`` scope. It only
# becomes observable when replayed clean history sits at a strictly lower causal
# step than the current chunk, and that step assignment comes from the memory
# layout rather than from the frame index -- hand-writing it here would be
# inventing the packer's output and testing the invention. It is covered against
# real metadata in ``test_packing.py``.


def test_cross_sample_attention_is_blocked(tokens) -> None:
    """Two packed samples never see each other, whatever their roles say."""
    assert not visibility(
        build_stream(tokens, sample=0),
        build_stream(tokens, sample=1),
        scope="all_views",
    ).any()


def test_padding_attends_only_padding() -> None:
    """Padding rows must stay valid so softmax never sees an all-false row."""
    tokens = [
        (ROLE_PADDING, 0, 0, 0),
        (ROLE_CURRENT_TARGET, 0, 0, 0),
        (ROLE_PADDING, 0, 0, 0),
    ]
    stream = build_stream(tokens)
    mask = visibility(stream, stream, scope="all_views")
    assert mask[0].tolist() == [True, False, True]
    assert mask[1, 0].item() is False


def test_the_block_mask_builder_stays_compiled() -> None:
    """Correctness does not depend on this. Memory does, by about 9x.

    Eager ``create_block_mask`` calls ``create_mask``, which materializes the
    whole ``[Q, KV]`` bool and runs the predicate's dozen boolean terms over it
    -- the cost the block mask exists to avoid. Measured at four views: 7.2 GiB
    eager against 0.77 GiB compiled, above a 0.66 GiB dense mask.

    Dropping the ``torch.compile`` would therefore pass every other test in this
    file and every parity test, and quietly cost tens of GiB at eleven views.
    Hence a test that asserts nothing about behaviour.

    ``_torchdynamo_orig_callable`` is private, and the public-looking
    alternative does not apply: ``torch.compile`` returns an ``OptimizedModule``
    only for an ``nn.Module``, and ``create_block_mask`` is a function, so the
    wrapper's type is plain ``function``. If a future release renames the
    attribute this test fails rather than passing quietly, which is the
    direction to fail in -- it would be read as the compile having gone.
    """
    assert hasattr(_compiled_create_block_mask(), "_torchdynamo_orig_callable")


def test_the_block_mask_builder_compiles_once() -> None:
    """A fresh predicate closure per call must not mean a fresh compile.

    The rollout builds one per chunk per branch, so recompiling each time would
    cost more than the kernel saves.
    """
    assert _compiled_create_block_mask() is _compiled_create_block_mask()
