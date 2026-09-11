# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Mesh arithmetic and distributed projection/token equivalence.

Everything here is a pure function of two or three integers, and every rank
computes it independently and has to reach the same answer -- so a disagreement
is a hang rather than an exception, which is the reason to pin the arithmetic
down here rather than notice it on eight GPUs.
"""

from __future__ import annotations

from datetime import timedelta
from unittest.mock import call, patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from flashdreams.core.distributed.context_parallel import (
    build_shard,
    gather_tokens,
    local_query_range,
)
from flashdreams.core.distributed.parallel import (
    ParallelContext,
    balanced_ranges,
    build_context,
    init_parallel,
    plan_mesh,
)
from flashdreams.core.distributed.tensor_parallel import (
    ColumnParallelLinear,
    RowParallelLinear,
)

pytestmark = pytest.mark.ci_cpu


def context(*, tp: int = 1, cp: int = 1, cp_rank: int = 0) -> ParallelContext:
    """A context with the right shape and no process group behind it.

    Every function under test reads the sizes and the rank and nothing else, so
    the groups can stay ``None``; anything that would actually collectivize is
    in :mod:`test_parallel_equivalence`, which does launch processes.
    """
    return ParallelContext(
        tp_group=None,
        tp_rank=0,
        tp_size=tp,
        cp_group=None,
        cp_rank=cp_rank,
        cp_size=cp,
        device=torch.device("cpu"),
    )


@pytest.mark.parametrize(
    ("world", "expected"),
    [
        (1, (1, 1)),
        (2, (2, 1)),
        (3, (1, 3)),
        (4, (4, 1)),
        (5, (1, 5)),
        (6, (2, 3)),
        (7, (1, 7)),
        (8, (8, 1)),
        (12, (4, 3)),
        (16, (8, 2)),
        (64, (8, 8)),
    ],
)
def test_the_default_tensor_axis_divides_the_head_groups(
    world: int, expected: tuple[int, int]
) -> None:
    """The rule is ``gcd(world, 8)``, and it is a latency choice.

    A tensor shard holds whole head groups, so a layer costs ``ceil(8/tp)``
    rather than ``tp``. Choosing a tensor axis that does not divide the head
    groups leaves uneven work, so the default selects only an exact factor.
    """
    assert plan_mesh(world, head_groups=8) == expected


@pytest.mark.parametrize("world", [3, 5, 6, 7, 12, 24])
def test_the_default_never_leaves_a_rank_holding_two_head_groups_alone(
    world: int,
) -> None:
    """The property behind the table above, stated as the property.

    Whatever the rank count, the default splits the eight groups evenly or not
    at all -- never 3/3/2 or 2/2/1/1/1/1, which are the shapes that cost.
    """
    tp, _cp = plan_mesh(world, head_groups=8)
    assert 8 % tp == 0


def test_default_mesh_stops_at_the_head_groups() -> None:
    """Sixty-four ranks cannot put more than eight of themselves on the model."""
    tp, cp = plan_mesh(64, head_groups=8)
    assert tp == 8
    assert tp * cp == 64


def test_a_prime_world_falls_back_to_one_copy_per_rank() -> None:
    """Eleven ranks share no factor with eight but one, so the mesh is 1x11."""
    assert plan_mesh(11, head_groups=8) == (1, 11)


def test_a_smaller_checkpoint_is_a_smaller_ceiling() -> None:
    """``head_groups`` is an argument so a four-head model is not a crash after the load."""
    assert plan_mesh(16, head_groups=4) == (4, 4)
    with pytest.raises(ValueError, match="head groups"):
        plan_mesh(16, 8, head_groups=4)


def test_a_tensor_axis_that_does_not_divide_is_refused() -> None:
    with pytest.raises(ValueError, match="rectangular"):
        plan_mesh(6, 4, head_groups=8)


def test_a_tensor_axis_past_the_head_groups_is_refused() -> None:
    with pytest.raises(ValueError, match="head groups"):
        plan_mesh(16, 16, head_groups=8)


@pytest.mark.parametrize("parts", range(1, 9))
def test_head_groups_split_evenly_to_within_one(parts: int) -> None:
    ranges = balanced_ranges(8, parts)
    sizes = [end - start for start, end in ranges]
    assert sum(sizes) == 8
    assert max(sizes) - min(sizes) <= 1
    # Contiguous and in order, which is what makes a weight slice a slice.
    assert [start for start, _ in ranges] == [0, *[end for _, end in ranges[:-1]]]


def test_the_remainder_goes_to_the_low_ranks() -> None:
    """Eight groups over three ranks is 3/3/2, not 2/2/4."""
    assert balanced_ranges(8, 3) == [(0, 3), (3, 6), (6, 8)]


@pytest.mark.parametrize("size", [1, 2, 3, 4, 5, 6, 7, 8, 16, 64])
def test_a_chunk_shards_at_every_size_and_loses_no_token(size: int) -> None:
    """A prime-sized chunk shards unevenly without losing a token.

    This is the whole reason the package does not use FlashDreams'
    ``split_inputs_cp``, which asserts the sequence divides the rank count.
    """
    total = 67
    shards = [
        build_shard(total, context(cp=size, cp_rank=rank)) for rank in range(size)
    ]
    sizes = [shard.local_size for shard in shards]
    assert sum(sizes) == total
    assert max(sizes) - min(sizes) <= 1
    # Contiguous, in rank order, so a gather in rank order is the original order.
    assert [shard.local for shard in shards] == list(shards[0].ranges)


def test_a_shard_narrower_than_the_ranks_is_refused() -> None:
    with pytest.raises(ValueError, match="past the 3 tokens"):
        build_shard(3, context(cp=4))


def test_an_inert_axis_asks_for_no_query_range() -> None:
    """``None`` is what every mask builder reads as 'all of them'."""
    assert local_query_range(1000, context()) is None
    assert local_query_range(1000, context(cp=4, cp_rank=2)) == (500, 750)


def test_select_and_ranges_agree() -> None:
    shard = build_shard(10, context(cp=3, cp_rank=1))
    x = torch.arange(10).reshape(1, 10, 1)
    assert shard.local == (4, 7)
    assert shard.select(x, dim=1).flatten().tolist() == [4, 5, 6]


@torch.no_grad()
def _projection_worker(rank: int, world: int, tp: int, rendezvous: str) -> None:
    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=world,
        timeout=timedelta(seconds=60),
    )
    try:
        # An initialized group must work without torchrun environment variables.
        with patch("torch.cuda.is_available", return_value=False):
            ctx = init_parallel(tp, head_groups=8)
        assert ctx.rank == rank
        assert ctx.world_size == world
        if tp > 1:
            assert dist.get_process_group_ranks(ctx.tp_group) == list(
                range(ctx.cp_rank * tp, (ctx.cp_rank + 1) * tp)
            )
        if ctx.cp_size > 1:
            assert dist.get_process_group_ranks(ctx.cp_group) == list(
                range(ctx.tp_rank, world, tp)
            )

        torch.manual_seed(7)
        up = torch.nn.Linear(5, 7)
        down = torch.nn.Linear(7, 5)
        # Context ranks receive different inputs, exposing a TP world-group leak.
        x = torch.randn(2, 5) + ctx.cp_rank
        expected = down(torch.nn.functional.gelu(up(x)))
        start, end = balanced_ranges(7, tp)[ctx.tp_rank]
        if tp > 1:
            column = ColumnParallelLinear(up, start, end)
            row = RowParallelLinear(
                down, start, end, group=ctx.tp_group, keeps_bias=ctx.tp_rank == 0
            )
            actual = row(torch.nn.functional.gelu(column(x)))
            torch.testing.assert_close(actual, expected)
            assert column.weight.numel() < up.weight.numel()
            assert not column.weight.requires_grad

        shard = build_shard(7, ctx)
        sequence = torch.arange(42).reshape(2, 7, 3) + ctx.tp_rank * 100
        for dim in (1, -2):
            actual = gather_tokens(shard.select(sequence, dim), shard, ctx, dim=dim)
            torch.testing.assert_close(actual, sequence)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize(("tp", "cp"), [(3, 1), (1, 3), (2, 2)])
def test_uneven_projections_and_tokens_match_unsharded_values(tp, cp, tmp_path) -> None:
    mp.spawn(
        _projection_worker,
        args=(tp * cp, tp, str(tmp_path / "rendezvous")),
        nprocs=tp * cp,
    )


def test_single_rank_gather_is_a_noop() -> None:
    ctx = ParallelContext.single()
    x = torch.arange(5)
    assert gather_tokens(x, build_shard(5, ctx), ctx, dim=-1) is x


def test_collectives_require_the_axis_group() -> None:
    ctx = context(cp=2)
    shard = build_shard(5, ctx)
    with pytest.raises(ValueError, match="context process group"):
        gather_tokens(torch.arange(shard.local_size), shard, ctx, dim=0)
    with pytest.raises(ValueError, match="tensor process group"):
        RowParallelLinear(torch.nn.Linear(5, 5), 0, 3, group=None, keeps_bias=True)


def test_projection_bounds_are_validated() -> None:
    with pytest.raises(ValueError, match="output shard"):
        ColumnParallelLinear(torch.nn.Linear(5, 5), 0, 6)


def test_single_rank_initialization_validates_the_requested_mesh(monkeypatch) -> None:
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert init_parallel(head_groups=4) == ParallelContext.single()
    with pytest.raises(ValueError, match="rectangular"):
        init_parallel(2, head_groups=4)


@pytest.mark.parametrize(("device", "backend"), [("cpu", "gloo"), ("cuda:1", "nccl")])
@pytest.mark.parametrize("world_backend", ["gloo", "nccl", "cpu:gloo,cuda:nccl"])
@pytest.mark.parametrize("reuse_world", [False, True])
def test_mesh_axis_backends_follow_device(
    device, backend, world_backend, reuse_world, monkeypatch
) -> None:
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    monkeypatch.setenv("LOCAL_RANK", "1")
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_world_size", lambda: 4)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: device != "cpu")
    with (
        patch.object(dist, "get_backend", return_value=world_backend),
        patch.object(dist, "init_process_group") as initialize,
        patch.object(dist, "destroy_process_group") as destroy,
        patch.object(torch.cuda, "set_device"),
    ):
        for rank in range(4):
            with (
                patch.object(dist, "get_rank", return_value=rank),
                patch.object(
                    dist, "new_group", side_effect=[object() for _ in range(4)]
                ) as groups,
            ):
                ctx = (
                    init_parallel(2, head_groups=8)
                    if reuse_world
                    else build_context(2, 2, torch.device(device))
                )
                assert ctx.rank == rank
                assert ctx.device == torch.device(device)
                assert groups.call_args_list == [
                    call([0, 1], backend=backend),
                    call([2, 3], backend=backend),
                    call([0, 2], backend=backend),
                    call([1, 3], backend=backend),
                ]
        initialize.assert_not_called()
        destroy.assert_not_called()


def test_mesh_rejects_unsupported_device_before_creating_groups() -> None:
    with (
        patch.object(dist, "get_world_size", return_value=4),
        patch.object(dist, "get_rank", return_value=0),
        patch.object(dist, "new_group") as groups,
    ):
        with pytest.raises(ValueError, match="CPU or CUDA"):
            build_context(2, 2, torch.device("meta"))
        groups.assert_not_called()
