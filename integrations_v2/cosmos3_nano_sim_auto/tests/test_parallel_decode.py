# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Does ``--parallel-decode`` hand rank 0 the same cameras, in the same order?

Real processes over gloo, no GPU, a stub in place of TAEHV-super. What can go
wrong in ``decoded_views`` is wiring: a camera decoded on the wrong rank, sent
in the wrong order, or a rank with no cameras taking part in a collective it
should sit out. A stub decode that stamps each row with its camera index and the
rank that decoded it is enough to catch all three.
"""

from __future__ import annotations

import os
import tempfile
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from cosmos3_nano_sim_auto.impl.decode import decoded_views
from cosmos3_nano_sim_auto.impl.distributed.parallel import owned_views

from flashdreams.core.distributed.parallel import build_context

pytestmark = pytest.mark.ci_cpu

VIEWS, CHANNELS, FRAMES, HEIGHT, WIDTH = 4, 2, 3, 2, 2


def _latent() -> torch.Tensor:
    # Camera index baked into every element, so a misrouted camera shows up as a value.
    return (
        torch.arange(VIEWS, dtype=torch.float32)
        .view(VIEWS, 1, 1, 1, 1)
        .expand(VIEWS, CHANNELS, FRAMES, HEIGHT, WIDTH)
        .clone()
    )


def _stub_decode(rank: int):
    """Camera value times ten plus the decoding rank, as a ``[v, T, 3, H, W]`` stub."""

    def decode(latent: torch.Tensor) -> torch.Tensor:
        views = latent.shape[0]
        return (
            latent[:, :1, :, :1, :1]
            .expand(views, 1, FRAMES, 3, 3)
            .permute(0, 2, 1, 3, 4)
            .reshape(views, FRAMES, 3, 3)
            * 10
            + rank
        )

    return decode


def _worker(rank: int, world: int, init_file: str, out_dir: str) -> None:
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world,
        timeout=timedelta(seconds=120),
    )
    try:
        ctx = build_context(1, world, torch.device("cpu"))
        decode = _stub_decode(rank) if len(owned_views(ctx, VIEWS)) else None
        pixels = decoded_views(_latent(), decode, parallel=ctx)
        torch.save(pixels.clone(), os.path.join(out_dir, f"rank{rank}.pt"))
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("world", [2, 3, 5])
def test_rank_zero_gets_every_camera_in_order_from_its_owner(world: int) -> None:
    with tempfile.TemporaryDirectory() as directory:
        init_file = os.path.join(directory, "init")
        mp.start_processes(
            _worker,
            args=(world, init_file, directory),
            nprocs=world,
            join=True,
            start_method="spawn",
        )
        pixels = torch.load(os.path.join(directory, "rank0.pt"))
    assert pixels.shape == (VIEWS, FRAMES, 3, 3)
    for view in range(VIEWS):
        owner = view % world
        assert torch.equal(
            pixels[view], torch.full((FRAMES, 3, 3), float(view * 10 + owner))
        )


def test_without_a_mesh_it_is_just_decode() -> None:
    latent = _latent()
    assert torch.equal(
        decoded_views(latent, _stub_decode(0), parallel=None), _stub_decode(0)(latent)
    )


class _Ludus:
    """Draws camera ``v``'s frame ``f`` as the value ``10 * v + f``."""

    cameras = tuple(range(VIEWS))

    def render(self, first: int, last: int, views) -> torch.Tensor:
        drawn = [
            torch.arange(first, last).float().view(-1, 1, 1, 1).expand(-1, 2, 2, 3)
            + 10 * v
            for v in views
        ]
        return torch.cat(drawn) if drawn else torch.empty(0, 2, 2, 3)


def _drive_worker(rank: int, world: int, init_file: str, out_dir: str) -> None:
    """The drive's draw: each rank draws its own cameras and rank 0 queues the rig."""
    from cosmos3_nano_sim_auto.apps.interactive_drive.backend import _DrawnRig

    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world,
        timeout=timedelta(seconds=120),
    )
    try:
        ctx = build_context(1, world, torch.device("cpu"))
        rig = _DrawnRig(_Ludus(), parallel=ctx)
        rig.render(0, 2, views=list(owned_views(ctx, VIEWS)))
        queued = [[float(view[0, 0, 0]) for view in frame] for frame in rig._queued]
        torch.save(queued, os.path.join(out_dir, f"rank{rank}.pt"))
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("world", [2, 5])
def test_the_drive_queues_every_drawn_camera_in_rig_order_on_rank_zero_alone(
    world: int,
) -> None:
    with tempfile.TemporaryDirectory() as directory:
        init_file = os.path.join(directory, "init")
        mp.start_processes(
            _drive_worker,
            args=(world, init_file, directory),
            nprocs=world,
            join=True,
            start_method="spawn",
        )
        queued = [
            torch.load(os.path.join(directory, f"rank{rank}.pt"))
            for rank in range(world)
        ]
    assert queued[0] == [
        [10.0 * view + frame for view in range(VIEWS)] for frame in range(2)
    ]
    assert all(one == [] for one in queued[1:])
