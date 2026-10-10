# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A HUD button on a mesh, where only one rank has a HUD to press.

Input events do not need this: they arrive at rank zero and the runtime
broadcasts them before any rank steps, so the gamepad's reset button already
agrees. The HUD reaches the model loop another way, through ``invoke_async``
from the thread drawing the window, and that thread exists on one rank.

A rank that restarted alone would rebuild its scene, and enter the
conditioning's collectives, while the others were still generating, and the two
sequences do not meet again. So these spawn real processes over a real gloo
group: a stub would agree with whatever the code did.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from dataclasses import replace
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch

import pytest
import torch.distributed as dist
import torch.multiprocessing as mp
from interactive_drive import InteractiveDriveConfig, InteractiveDriveModelState
from interactive_drive.config import AppConfig

from flashdreams.core.distributed.parallel import ParallelContext

pytestmark = pytest.mark.ci_cpu

SPAWN_TIMEOUT_S = 120
"""A deadlock guard, not a speed limit.

The bug under test is ranks taking different paths, and the shape that takes
without a timeout is a suite that never returns rather than one that fails.
"""

_REPO_ROOT = str(Path(__file__).resolve().parents[3])

FIRST = Path("/scenes/first.usdz")
SECOND = Path("/scenes/second.usdz")


def _state(rank: int, world: int) -> InteractiveDriveModelState:
    """The model state alone, which is all of the app a queued action touches."""
    return InteractiveDriveModelState(
        backend_factory=lambda _config: None,
        config=InteractiveDriveConfig(
            app=AppConfig(scene_path=FIRST, variant="dry"),
            total_blocks=0,
            view_mode="rgb",
        ),
        desc=None,
        scene_loader=lambda *_args, **_kwargs: None,
        parallel_context=replace(
            ParallelContext.single("cpu"), cp_rank=rank, cp_size=world
        ),
    )


def _worker(rank: int, world: int, asked: str, init_file: str, directory: str) -> None:
    """One rank, told what the window's rank was asked for, agreeing on it."""
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    # A spawned process inherits its parent's environment, so a suite run from
    # inside a launcher would otherwise have every rank join a second world.
    for variable in ("WORLD_SIZE", "RANK", "LOCAL_RANK"):
        os.environ.pop(variable, None)
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world,
        timeout=timedelta(seconds=60),
    )
    try:
        state = _state(rank, world)
        if rank == 0:
            if asked == "scene":
                state.select_scene(SECOND, "wet")
            elif asked == "variant":
                state.select_variant("wet")
            elif asked == "restart":
                state.restart("Drive through snow.")
        # Three times, because this is asked once a step for the life of the
        # run. A group remade each time would leak until the world refused
        # another, and a rank that fell out of order would hang here.
        for _ in range(3):
            state.agree_on_queued()
        Path(directory, f"rank{rank}.json").write_text(
            json.dumps(
                {
                    "reset_pending": state.reset_pending,
                    "scene": str(state.config.app.scene_path),
                    "variant": state.config.app.variant,
                    "prompt": state.pending_prompt,
                    "queued": state.queued is not None,
                }
            )
        )
    finally:
        dist.destroy_process_group()


def _ranks(asked: str, world: int = 2) -> list[dict]:
    """Run ``world`` ranks and read back what each decided.

    Through files rather than a queue, because a queued object outlives the
    sending process only by luck, and every rank is joined before any is
    judged so one slow rank does not report as several broken ones.
    """
    context = mp.get_context("spawn")
    with tempfile.TemporaryDirectory() as directory:
        init_file = os.path.join(directory, "rendezvous")
        processes = [
            context.Process(
                target=_worker, args=(rank, world, asked, init_file, directory)
            )
            for rank in range(world)
        ]
        deadline = time.monotonic() + SPAWN_TIMEOUT_S
        try:
            with patch.object(sys, "path", [_REPO_ROOT, *sys.path]):
                for process in processes:
                    process.start()
            for process in processes:
                process.join(timeout=max(0, deadline - time.monotonic()))
            codes = [process.exitcode for process in processes]
            assert all(code == 0 for code in codes), f"rank exit codes were {codes}"
        finally:
            for process in processes:
                if process.is_alive():
                    process.kill()
            for process in processes:
                if process.pid is not None:
                    process.join()
        return [
            json.loads(Path(directory, f"rank{rank}.json").read_text())
            for rank in range(world)
        ]


def test_a_restart_asked_of_the_window_reaches_every_rank() -> None:
    """The flags Jon measured: [True, False] before this, [True, True] after."""
    ranks = _ranks("restart")

    assert [rank["reset_pending"] for rank in ranks] == [True, True]
    assert [rank["prompt"] for rank in ranks] == ["Drive through snow."] * 2


def test_a_scene_chosen_on_the_window_is_the_scene_every_rank_loads() -> None:
    """The payload is why the runtime's own reset event could not carry this."""
    ranks = _ranks("scene")

    assert [rank["scene"] for rank in ranks] == [str(SECOND)] * 2
    assert [rank["variant"] for rank in ranks] == ["wet"] * 2
    assert all(rank["reset_pending"] for rank in ranks)


def test_a_variant_chosen_on_the_window_keeps_every_rank_on_the_same_scene() -> None:
    ranks = _ranks("variant")

    assert [rank["scene"] for rank in ranks] == [str(FIRST)] * 2
    assert [rank["variant"] for rank in ranks] == ["wet"] * 2


def test_a_step_that_was_asked_for_nothing_leaves_every_rank_alone() -> None:
    """Agreeing happens every step, so the common case has to agree on nothing."""
    ranks = _ranks("nothing")

    assert [rank["reset_pending"] for rank in ranks] == [False, False]
    assert not any(rank["queued"] for rank in ranks)


def test_a_drive_on_one_card_queues_and_applies_without_a_collective() -> None:
    """No mesh, no process group, and the same queued action still lands."""
    state = _state(0, 1)
    state.parallel_context = None

    state.select_scene(SECOND, "wet")
    state.agree_on_queued()

    assert state.reset_pending
    assert state.config.app.scene_path == SECOND
    assert state.queued is None
