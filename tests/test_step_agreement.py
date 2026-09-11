# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Do the ranks of a mesh leave a run together? Real processes, gloo, no GPU.

What :mod:`flashdreams.runtime_v2.coordination` claims is that every rank of a mesh takes
the same number of steps whatever ends the run, and that is not a claim one
process can check. So these spawn real ranks over gloo and compare what each of
them heard.

The scripts below are rank 0's exits, one test each, because they are not one
exit. The runtime checks rank 0's shutdown event *before* and *after* the call
the decision rides on, and a step that raises leaves from inside the step, so
rank 0 leaves the loop from four places and only one of them reaches
:meth:`~flashdreams.runtime_v2.coordination.StepAgreement.agree`. A script that runs out
before the cap is rank 0 taking one of the other three, and what the followers
hear then comes from ``close`` alone.

A follower that hears nothing hangs rather than fails, which is the bug being
fixed, so every wait here is bounded twice: the control group has a timeout of
its own, and the harness joins each rank with one.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Sequence
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from flashdreams.runtime_v2.coordination import (
    AGREEMENT_TIMEOUT,
    StepAction,
    StepAgreement,
)
from flashdreams.core.distributed.parallel import ParallelContext, build_context

pytestmark = pytest.mark.ci_cpu

ITERATION_CAP = 12
"""Iterations a rank runs before giving up on being told to stop.

Every script here is shorter than this, so reaching it means a rank heard
nothing -- which is the failure this module removes. It is a guard that turns
that failure into a readable assertion rather than a hung process.
"""

CONTROL_TIMEOUT = timedelta(seconds=300)
"""How long a spawned rank waits for the decision, in place of the real day.

:data:`~flashdreams.runtime_v2.coordination.AGREEMENT_TIMEOUT` is sized for a stream
nobody is watching. A test is watched, and a rank that would wait a day is a
test that never reports.
"""

SPAWN_TIMEOUT_S = 900
"""How long a spawned rank gets. A deadlock guard, not a speed limit."""


# --- one process, nothing to agree with -------------------------------------


def test_the_wire_values_are_pinned() -> None:
    """These travel between processes, so renumbering them is a protocol change."""
    assert [int(action) for action in StepAction] == [0, 1, 2]
    assert (StepAction.CONTINUE, StepAction.RESET, StepAction.STOP) == (0, 1, 2)


def test_one_rank_agrees_with_itself_and_takes_no_collective() -> None:
    """The inert path, which is every run launched without ``torchrun``."""
    agreement = StepAgreement(ParallelContext.single())

    assert not agreement.distributed
    assert agreement.agree(StepAction.CONTINUE) is StepAction.CONTINUE
    assert agreement.agree(StepAction.RESET) is StepAction.RESET
    assert not agreement.stopped
    assert agreement.agree(StepAction.STOP) is StepAction.STOP
    assert agreement.stopped


def test_announcing_a_stop_twice_says_it_once() -> None:
    """``close`` announces the stop that ``agree`` never reached, and only that one.

    A second announcement would be a broadcast nobody is expecting, and the rank
    expecting the *next* one would read it as that.
    """
    agreement = StepAgreement(ParallelContext.single())

    agreement.announce_stop()
    assert agreement.stopped
    # Reaching here at all is the assertion: a second one would have to be
    # matched, and on a real mesh there is nothing left to match it.
    agreement.announce_stop()


def test_a_mesh_with_no_process_group_behind_it_decides_for_itself() -> None:
    """A context built by hand is what a test has, and it must not go collective.

    The package's rule is that anything unlaunched takes the path it took before
    any of this existed. A hand-built multi-rank context has no group to
    broadcast over, so the honest answer is the caller's own.
    """
    ctx = ParallelContext(
        tp_group=None,
        tp_rank=1,
        tp_size=2,
        cp_group=None,
        cp_rank=0,
        cp_size=1,
        device=torch.device("cpu"),
    )
    agreement = StepAgreement(ctx)

    assert ctx.world_size == 2
    assert not ctx.is_main
    assert not agreement.distributed
    assert agreement.agree(StepAction.STOP) is StepAction.STOP


def test_the_real_timeout_is_sized_for_a_person_not_a_kernel() -> None:
    """A rank between chunks of a served stream waits as long as nobody watches."""
    assert AGREEMENT_TIMEOUT >= timedelta(hours=1)


# --- real ranks over gloo ----------------------------------------------------


def _worker(
    rank: int,
    world: int,
    script: list[int],
    follower_cap: int,
    init_file: str,
    directory: str,
) -> None:
    """One rank running the shape of the runtime's model loop.

    ``script`` is what rank 0 wants at each iteration, one entry per iteration.
    Running off its end is rank 0 leaving the loop *without* deciding -- the
    shutdown-event path, whose only announcement is the loop's ``close``, which
    is :meth:`StepAgreement.announce_stop` here.

    ``follower_cap`` is how many iterations a rank that is not rank 0 runs before
    leaving on its own. It exists for one case: ``run_session(steps=N)`` gives
    every rank the same bound, so they all leave together having agreed nothing
    and rank 0 announces to an empty room.
    """
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    dist.init_process_group(
        backend="gloo", init_method=f"file://{init_file}", rank=rank, world_size=world
    )
    try:
        ctx = build_context(1, world, torch.device("cpu"))
        agreement = StepAgreement(ctx, timeout=CONTROL_TIMEOUT)
        heard: list[int] = []
        cap = len(script) if ctx.is_main else follower_cap
        for iteration in range(cap):
            wanted = (
                StepAction(script[iteration]) if ctx.is_main else StepAction.CONTINUE
            )
            action = agreement.agree(wanted)
            heard.append(int(action))
            if action is StepAction.STOP:
                break
        agreement.announce_stop()
        with open(os.path.join(directory, f"rank{rank}.json"), "w") as file:
            json.dump(
                {"rank": ctx.rank, "heard": heard, "stopped": agreement.stopped}, file
            )
    finally:
        dist.destroy_process_group()


def run_mesh(
    world: int, script: Sequence[StepAction], *, follower_cap: int = ITERATION_CAP
) -> list[dict]:
    """Spawn ``world`` ranks running ``script`` and collect what each heard.

    Results come back through **files** rather than a queue, for the reason
    ``test_parallel_equivalence.py`` sets out at length: a queued object outlives
    the sending process only by luck. Every rank is joined before any of them is
    judged, so one slow rank does not report as several broken ones.
    """
    context = mp.get_context("spawn")
    with tempfile.TemporaryDirectory() as directory:
        init_file = os.path.join(directory, "rendezvous")
        processes = [
            context.Process(
                target=_worker,
                args=(
                    rank,
                    world,
                    [int(action) for action in script],
                    follower_cap,
                    init_file,
                    directory,
                ),
            )
            for rank in range(world)
        ]
        for process in processes:
            process.start()
        for process in processes:
            process.join(timeout=SPAWN_TIMEOUT_S)
        codes = [process.exitcode for process in processes]
        for process in processes:
            if process.exitcode is None:
                process.kill()
        assert all(code == 0 for code in codes), f"rank exit codes were {codes}"
        results = []
        for rank in range(world):
            with open(os.path.join(directory, f"rank{rank}.json")) as file:
                results.append(json.load(file))
        return results


@pytest.mark.parametrize("world", [2, 3])
def test_a_stop_rank_zero_decided_reaches_every_rank(world: int) -> None:
    """The ordinary end of a clip: rank 0 knows, and says so where it is heard."""
    script = [StepAction.CONTINUE, StepAction.CONTINUE, StepAction.STOP]

    for result in run_mesh(world, script):
        assert result["heard"] == [int(action) for action in script]
        assert result["stopped"]


def test_a_rank_zero_that_leaves_without_deciding_still_stops_the_others() -> None:
    """The hang this module exists for.

    A browser closing sets rank 0's shutdown event, and the runtime checks it
    outside the call the decision rides on -- so rank 0 leaves the loop having
    said nothing. Its ``close`` is the only thing left to say it, and the ranks
    parked in the broadcast are the only ones who need to hear it. Without that
    announcement they run on alone into a collective nobody else is in.
    """
    script = [StepAction.CONTINUE, StepAction.CONTINUE]

    results = run_mesh(3, script)

    leader, *followers = results
    assert leader["heard"] == [0, 0]
    assert leader["stopped"]
    for follower in followers:
        assert follower["heard"] == [0, 0, int(StepAction.STOP)]


def test_a_window_closed_before_anything_was_generated_stops_the_others() -> None:
    """Rank 0 never enters the loop, so its first message is the last one."""
    results = run_mesh(2, [])

    leader, follower = results
    assert leader["heard"] == []
    assert follower["heard"] == [int(StepAction.STOP)]


def test_a_run_that_ends_before_any_rank_agrees_does_not_hang_rank_zero() -> None:
    """Nobody reaches ``agree`` at all, which is a shutdown before iteration one.

    Every rank goes straight to ``close``: the followers return from
    ``announce_stop`` at once because they are not rank 0, and rank 0 is left
    holding the only announcement. With the control group built on first use
    that call was rank 0 alone inside ``new_group`` -- a collective over the
    world -- waiting for ranks that had already gone, for the group's timeout,
    which is a *day*. Measured, before the group moved into ``__init__``: rank 1
    returned in 0.0 s and rank 0 never did.

    The bound here is what makes the test an assertion rather than a hang: a
    rank still inside ``new_group`` misses it and reports as a failure to exit.
    """
    results = run_mesh(2, [], follower_cap=0)

    for record in results:
        assert record["heard"] == []
        # Rank 0 records the stop it could not send; the followers were never
        # rank 0 and have nothing to record.
        assert record["stopped"] is (record["rank"] == 0)


def test_announcing_a_stop_nobody_is_waiting_for_does_not_fail_the_run() -> None:
    """``run_session(steps=N)`` bounds every rank, so they all leave together.

    Nothing reaches ``agree`` on that path, so rank 0 announces a stop into a
    group whose other ranks have already gone. Gloo does not wait for a receive
    that is never posted -- which is the behaviour worth having, because waiting
    would hold a finished run open for the group's whole timeout, and that is a
    day -- but it does report the closed connection, and this ran as written
    until ``announce_stop`` stopped letting that through: the loop's ``close``
    raised, the runtime queued it, and a clean bounded run reported a failure.

    Pinned because it is the one place the protocol's "one call per iteration"
    invariant does not hold, and because what happens there is a property of the
    backend rather than of anything here.
    """
    script = [StepAction.CONTINUE, StepAction.CONTINUE]

    results = run_mesh(2, script, follower_cap=len(script))

    for result in results:
        assert result["heard"] == [0, 0]
    assert results[0]["stopped"]
    assert not results[1]["stopped"]


def test_a_reset_only_rank_zero_saw_reaches_every_rank() -> None:
    """A client restarting the clip is a stop and a start, and hangs the same way.

    The generation counter a reset travels on is bumped by rank 0's event buffer
    alone, so without this the other ranks carry on with a rollout rank 0 has
    already thrown away -- and the chunk after it is a collective over two
    different shapes.
    """
    script = [
        StepAction.CONTINUE,
        StepAction.RESET,
        StepAction.CONTINUE,
        StepAction.STOP,
    ]

    for result in run_mesh(2, script):
        assert result["heard"] == [int(action) for action in script]
