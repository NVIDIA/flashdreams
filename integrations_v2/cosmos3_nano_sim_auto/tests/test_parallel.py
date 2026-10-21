# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Multiview mask slicing and checkpoint defaults for the core process mesh."""

from __future__ import annotations

from typing import Any, cast

import pytest
import torch
from cosmos3_nano_sim_auto.impl.distributed import parallel
from cosmos3_nano_sim_auto.impl.packing import (
    ClipGeometry,
    build_chunk_metadata,
    build_memory_layout,
)
from cosmos3_nano_sim_auto.impl.transformer.mask import StreamFields

pytestmark = pytest.mark.ci_cpu

GEOMETRY = ClipGeometry(
    num_views=11,
    frames_per_view=9,
    patch_h=3,
    patch_w=2,
    frames_per_chunk=4,
    condition_frames=1,
)


def test_stream_fields_narrow_keeps_every_field_in_step() -> None:
    """A field left unnarrowed would mislabel the rows rather than fail."""
    memory = build_memory_layout(GEOMETRY)
    metadata = build_chunk_metadata(
        GEOMETRY, memory, chunk_start=1, chunk_frames=4, text_tokens=8
    )
    whole: StreamFields = metadata.query
    part = whole.narrow(100, 140)
    assert len(part) == 40
    for field in ("view_id", "frame_id", "token_role_id", "causal_step_id", "is_noisy"):
        assert torch.equal(getattr(part, field), getattr(whole, field)[100:140])


def test_a_restricted_mask_is_narrower_and_no_shorter() -> None:
    """The queries are one rank's; the keys are still everyone's."""
    memory = build_memory_layout(GEOMETRY)
    args: dict[str, Any] = {"chunk_start": 1, "chunk_frames": 4, "text_tokens": 8}
    whole = build_chunk_metadata(GEOMETRY, memory, **args).mask()
    part = build_chunk_metadata(GEOMETRY, memory, query_range=(0, 66), **args).mask()
    assert part.shape == (66, whole.shape[1])
    assert torch.equal(part, whole[:66])


@pytest.mark.parametrize(
    ("explicit", "environment", "world_size", "expected"),
    [
        (None, None, "8", 2),
        (None, None, "3", 1),
        (None, None, None, 1),
        (None, "4", "8", 4),
        (4, "2", "8", 4),
    ],
)
def test_launch_options_are_passed_to_core(
    monkeypatch, explicit, environment, world_size, expected
) -> None:
    if world_size is None:
        monkeypatch.delenv("WORLD_SIZE", raising=False)
    else:
        monkeypatch.setenv("WORLD_SIZE", world_size)
    if environment is None:
        monkeypatch.delenv("COSMOS3_NANO_SIM_AUTO_TENSOR_PARALLEL", raising=False)
    else:
        monkeypatch.setenv("COSMOS3_NANO_SIM_AUTO_TENSOR_PARALLEL", environment)
    received = []
    monkeypatch.setattr(
        parallel,
        "init_core_parallel",
        lambda tp, *, head_groups: received.append((tp, head_groups)),
    )
    parallel.init_parallel(explicit)
    assert received == [(expected, parallel.SPEC.llm.num_key_value_heads)]


def test_owned_views_deal_cameras_round_robin() -> None:
    """Every camera has exactly one owner, and the owner is ``view % world_size``."""

    class Mesh:
        def __init__(self, rank: int, world_size: int) -> None:
            self.rank, self.world_size = rank, world_size

    assert parallel.owned_views(None, 3) == range(3)
    for world in (1, 2, 4):
        owners = {
            view: rank
            for rank in range(world)
            for view in parallel.owned_views(cast(Any, Mesh(rank, world)), 3)
        }
        assert owners == {view: view % world for view in range(3)}
