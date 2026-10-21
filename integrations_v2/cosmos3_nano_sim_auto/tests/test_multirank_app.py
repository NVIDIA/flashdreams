# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The real model loop on a real mesh: who publishes, and who leaves when.

The root ``tests/test_step_agreement.py`` checks runtime coordination. This checks it is
wired into the thing the runtime actually drives -- ``IModelLoop._run_model_loop``
-- over a context-parallel mesh whose collectives are real, so a rank that
carried on alone would sit in one rather than merely count differently.

Two claims, one per test, and both are about a mesh rather than about a model:

- **Only rank 0 publishes.** Every rank generates the same chunk, so the rest
  return no channel and the runtime presents nothing for them.
- **Every rank leaves on the same step.** A shutdown event reaches rank 0 alone,
  which is what a browser closing looks like from here, and the ranks that never
  saw it used to run on into the next chunk's all-gather.

The network is the two-layer stand-in from ``test_parallel_equivalence.py`` and
the decoder is a stub: nothing here is about the picture, and a checkpoint would
cost minutes an assertion.
"""

from __future__ import annotations

import json
import os
import queue
import sys
import tempfile
import threading
import time
from dataclasses import replace
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from cosmos3_nano_sim_auto.impl.app import (
    Cosmos3NanoSimAutoApplication,
    Cosmos3NanoSimAutoModelLoop,
    Cosmos3NanoSimAutoModelState,
)
from cosmos3_nano_sim_auto.impl.packing import ClipGeometry
from cosmos3_nano_sim_auto.impl.rollout import ChunkRollout
from cosmos3_nano_sim_auto.inputs import MultiviewSample
from cosmos_interactive._network import (
    CosmosMoTNetwork,
    CosmosMoTNetworkConfig,
)
from cosmos_interactive._spec import Cosmos3InteractiveSpec, LLMSpec, VFMSpec

from flashdreams.core.distributed.parallel import build_context
from flashdreams.runtime_v2.coordination import StepAgreement
from flashdreams.runtime_v2.event_buffer import EventBuffer
from flashdreams.runtime_v2.session_desc import SessionDesc
from flashdreams.runtime_v2.step_result import StepResult

TEST_SAMPLE = Path(__file__).with_name("sample.jsonl")


pytestmark = pytest.mark.ci_cpu

SEED = 7
SPAWN_TIMEOUT_S = 900
"""How long a spawned rank gets. A deadlock guard, not a speed limit.

A rank that ran on past its peers blocks in an all-gather rather than failing,
so without this the bug under test reports as a test run that never returns.
"""

TINY = LLMSpec(
    hidden_size=32,
    num_layers=2,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=8,
    intermediate_size=16,
    vocab_size=128,
    mrope_section=(2, 1, 1),
)
SPEC = Cosmos3InteractiveSpec(
    llm=TINY,
    vfm=VFMSpec(vae_latent_channels=4, latent_patch_size=2, time_embed_freq_dim=8),
)
GEOMETRY = ClipGeometry(
    num_views=2,
    frames_per_view=13,
    patch_h=1,
    patch_w=2,
    frames_per_chunk=4,
    condition_frames=0,
)
CHUNKS = 4
"""Chunks this geometry takes: frame 0 alone, then three groups of four.

More than the step a run is stopped at, so "stopped early" and "ran out" are
different numbers rather than the same one.
"""

STOP_AFTER = 2
"""Steps rank 0 publishes before its shutdown event is set, as a close would."""

_REPO_ROOT = str(Path(__file__).resolve().parents[4])


def _decode(latent: torch.Tensor) -> torch.Tensor:
    """``[V, C, T, H, W]`` -> ``[V, T, 3, H, W]``. Only the frame count matters."""
    views, _channels, frames, height, width = latent.shape
    return torch.zeros(views, frames, 3, height, width)


def _state(
    ctx, network: CosmosMoTNetwork, sample_dir: str
) -> Cosmos3NanoSimAutoModelState:
    """A model-loop state seated on this rank's rollout, so no checkpoint is read.

    ``_build`` is what loads the VAE and reads the control videos, and it runs
    only when the state has no rollout. Seating one skips all of it and leaves
    the part under test -- the loop's own bookkeeping -- exactly as it is.
    """
    application = Cosmos3NanoSimAutoApplication()
    with patch("cosmos3_nano_sim_auto.impl.app.init_parallel", return_value=ctx):
        application.init(
            [
                "--views",
                str(GEOMETRY.num_views),
                "--frames",
                "33",
                "--sample",
                sample_dir,
            ]
        )
    assert application._config is not None
    config = replace(application._config, device="cpu", parallel=ctx)

    generator = torch.Generator().manual_seed(SEED + 2)
    assert GEOMETRY.frames_per_view is not None
    tokens = GEOMETRY.num_views * GEOMETRY.frames_per_view * GEOMETRY.spatial_tokens
    rollout = ChunkRollout(
        network,
        geometry=GEOMETRY,
        control_tokens=torch.randn(
            tokens, SPEC.vfm.patch_latent_dim, generator=generator
        ),
        text_ids=torch.randint(0, TINY.vocab_size, (3,), generator=generator),
        parallel=ctx,
    )
    return Cosmos3NanoSimAutoModelState(
        config=config,
        session_desc=SessionDesc(),
        sample=MultiviewSample.load(TEST_SAMPLE).trimmed_to(GEOMETRY.num_views),
        network=network,
        rollout=rollout,
        decode=_decode,
    )


def _worker(
    rank: int, world: int, stop_after: int, init_file: str, directory: str
) -> None:
    """One rank running the runtime's own model loop over a ``1 x world`` mesh."""
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    # The mesh here is built by hand, and `Cosmos3NanoSimAutoApplication.init` calls
    # `init_parallel`, which reads these. A spawned process inherits its parent's
    # environment, so a suite run from inside a launcher would otherwise have
    # every rank try to join a second, different world.
    for variable in (
        "WORLD_SIZE",
        "RANK",
        "LOCAL_RANK",
        "COSMOS3_NANO_SIM_AUTO_TENSOR_PARALLEL",
    ):
        os.environ.pop(variable, None)
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world,
        timeout=timedelta(seconds=120),
    )
    try:
        ctx = build_context(1, world, torch.device("cpu"))
        torch.manual_seed(SEED)
        network = (
            CosmosMoTNetworkConfig(
                spec=SPEC, dtype=torch.float32, include_vision_tower=False
            )
            .setup()
            .eval()
        )
        shutdown = threading.Event()
        failures: queue.Queue[BaseException] = queue.Queue()
        channels: list[int] = []

        class RecordingLoop(Cosmos3NanoSimAutoModelLoop):
            """Counts channels per step; the runtime publishes only non-empty steps."""

            def step(self, step_index, events):
                results = super().step(step_index, events)
                channels.append(len(results))
                # The close a browser sends, which reaches the rank holding the
                # window and no other: after a step, before the next one.
                if ctx.is_main and stop_after and len(channels) == stop_after:
                    shutdown.set()
                return results

        loop = RecordingLoop()
        loop.register_session_loop_objects(
            state=_state(ctx, network, directory),
            frequency=0,
            shutdown_event=shutdown,
            failure_queue=failures,
        )
        event_buffer = EventBuffer()
        event_buffer.register(0)

        def publish(
            generation: int, results: list[StepResult], step_elapsed_s: float
        ) -> None:
            del generation, results, step_elapsed_s

        agreement = StepAgreement(ctx, timeout=timedelta(seconds=30))
        try:
            loop._run_model_loop(
                event_buffer=event_buffer,
                reader_id=0,
                publish=publish,
                max_steps=None,
                step_control=agreement,
            )
        finally:
            agreement.close()
        with open(os.path.join(directory, f"rank{rank}.json"), "w") as file:
            json.dump(
                {
                    "rank": ctx.rank,
                    "channels": channels,
                    "failures": failures.qsize(),
                },
                file,
            )
    finally:
        dist.destroy_process_group()


def run_mesh(world: int, stop_after: int) -> list[dict]:
    """Spawn ``world`` ranks running the model loop and collect what each did.

    Results come back through **files** rather than a queue, for the reason
    ``test_parallel_equivalence.py`` sets out: a queued object outlives the
    sending process only by luck. Every rank is joined before any is judged, so
    one slow rank does not report as several broken ones.
    """
    context = mp.get_context("spawn")
    with tempfile.TemporaryDirectory() as directory:
        (Path(directory) / "sample.json").write_text(TEST_SAMPLE.read_text())
        init_file = os.path.join(directory, "rendezvous")
        processes = [
            context.Process(
                target=_worker, args=(rank, world, stop_after, init_file, directory)
            )
            for rank in range(world)
        ]
        deadline = time.monotonic() + SPAWN_TIMEOUT_S
        try:
            # pytest imports this module by its repository package path
            # root, so a spawned rank needs that root to unpickle its target.
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
        results = []
        for rank in range(world):
            with open(os.path.join(directory, f"rank{rank}.json")) as file:
                results.append(json.load(file))
        return results


def test_only_rank_zero_publishes_a_channel() -> None:
    """The redundant decode and the redundant file, both gone.

    Every rank generates the same chunk -- the rollout gathers it before
    returning -- so a worker returning no channel loses nothing. It also writes
    no file, since the runtime creates no client window on workers,
    which is what lets every rank of a launch share one ``--output-path``.
    """
    results = run_mesh(2, stop_after=0)

    leader, follower = results
    assert leader["channels"] == [1] * CHUNKS
    assert follower["channels"] == [0] * CHUNKS
    assert (leader["failures"], follower["failures"]) == (0, 0)


def test_every_rank_leaves_on_the_step_rank_zero_left_on() -> None:
    """The hang: a stop only rank 0 heard, and the ranks it never reached.

    Rank 0's shutdown event is set after two chunks and the runtime breaks out of
    the loop on it, without going through the call the decision rides on. Before
    the agreement, the other ranks carried on to the clip's own end and blocked
    in the third chunk's all-gather with rank 0 already gone -- a hang, with
    nothing in any log to say why. Now they stop on the same step.
    """
    results = run_mesh(3, stop_after=STOP_AFTER)

    assert STOP_AFTER < CHUNKS
    for result in results:
        assert len(result["channels"]) == STOP_AFTER
        assert result["failures"] == 0
    assert results[0]["channels"] == [1] * STOP_AFTER
    for follower in results[1:]:
        assert follower["channels"] == [0] * STOP_AFTER
