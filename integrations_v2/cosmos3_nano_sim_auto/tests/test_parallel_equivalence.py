# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Does a mesh compute what one rank computes? Real processes, gloo, no GPU.

The claim both axes rest on is that they change *where* the arithmetic happens
and nothing else. That is not a claim a single process can check, so these tests
spawn real ranks over gloo and compare against the same layers run whole in the
parent. A tiny randomly-initialised network stands in for the 15.2B-parameter
one, which is enough: what can go wrong here is a slice taken on the wrong axis,
a partial sum never reduced, or a gather that reassembles the sequence in the
wrong order, and all three are wiring rather than arithmetic.

Two things are deliberately different from a GPU run and neither weakens the
check. The backend is gloo, which is the only one a CPU has, and every collective
used here -- ``all_reduce`` and ``all_gather`` -- is backend-agnostic in this
package. And the dtype is float32, so the comparison can be tight; bf16 on a GPU
would agree to a tolerance instead, for the ordinary reason that its reductions
are not associative.
"""

from __future__ import annotations

import os
import sys
import tempfile
import time
from collections.abc import Callable
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from cosmos3_nano_sim_auto.impl.distributed.parallel import owned_views
from cosmos3_nano_sim_auto.impl.distributed.tensor_parallel import shard_layers
from cosmos3_nano_sim_auto.impl.packing import ClipGeometry
from cosmos3_nano_sim_auto.impl.rollout import run_rollout
from cosmos3_nano_sim_auto.impl.transformer.forward import run_chunk_layers
from cosmos_interactive._network import (
    CosmosMoTNetwork,
    CosmosMoTNetworkConfig,
    _MoTLayer,
)
from cosmos_interactive._spec import Cosmos3InteractiveSpec, LLMSpec, VFMSpec

from flashdreams.core.distributed.parallel import (
    ParallelContext,
    balanced_ranges,
    build_context,
)

pytestmark = pytest.mark.ci_cpu

# Four query heads over two key/value heads, so grouped-query attention is real
# and the tensor axis has two head groups to hand out.
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
CHUNK, TEXT, MEMORY = 13, 3, 7
SEED = 7
SPAWN_TIMEOUT_S = 900
"""How long a spawned rank gets. It is a deadlock guard, not a speed limit."""

_REPO_ROOT = str(Path(__file__).resolve().parents[4])


def build_layers() -> list[_MoTLayer]:
    """The same two layers on every rank, from the same seed."""
    torch.manual_seed(SEED)
    return [_MoTLayer(TINY, torch.float32).eval() for _ in range(TINY.num_layers)]


def build_inputs() -> tuple[torch.Tensor, torch.Tensor, list, torch.Tensor]:
    """One chunk, one prompt cache, one memory cache, and a mask over the lot."""
    generator = torch.Generator().manual_seed(SEED + 1)
    h = torch.randn(1, CHUNK, TINY.hidden_size, generator=generator)
    positions = torch.rand(3, CHUNK, generator=generator) * 10

    def cache(length: int) -> tuple[torch.Tensor, torch.Tensor]:
        shape = (1, TINY.num_key_value_heads, length, TINY.head_dim)
        return (
            torch.randn(shape, generator=generator),
            torch.randn(shape, generator=generator),
        )

    caches = [(cache(TEXT), cache(MEMORY)) for _ in range(TINY.num_layers)]
    # Something less than everything, so a mask that was silently row-shifted
    # would change the answer rather than not matter.
    mask = torch.rand(CHUNK, TEXT + CHUNK + MEMORY, generator=generator) > 0.25
    return h, positions, caches, mask


def kv_slice(pair: tuple[torch.Tensor, torch.Tensor], tp_rank: int, tp_size: int):
    """This rank's head slice of a cache entry, matching how the weights were cut."""
    heads = TINY.num_key_value_heads // tp_size
    start = tp_rank * heads
    return tuple(x[:, start : start + heads] for x in pair)


@torch.no_grad()
def reference() -> tuple[torch.Tensor, list]:
    """The whole computation in one process, which is what a mesh has to match."""
    h, positions, caches, mask = build_inputs()
    return run_chunk_layers(
        build_layers(),
        TINY,
        h,
        positions,
        text_kv=[text for text, _ in caches],
        memory_kv=[memory for _, memory in caches],
        mask=mask,
    )


@torch.no_grad()
def _worker(rank: int, world: int, tp: int, init_file: str, directory: str) -> None:
    """One rank: join the mesh, run its share, and write down what it got back."""
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world,
        timeout=timedelta(seconds=120),
    )
    try:
        ctx = build_context(tp, world // tp, torch.device("cpu"))
        layers = shard_layers(build_layers(), TINY, ctx)
        h, positions, caches, mask = build_inputs()

        rows = mask
        if ctx.context_parallel:
            base, extra = divmod(CHUNK, ctx.cp_size)
            start = ctx.cp_rank * base + min(ctx.cp_rank, extra)
            end = start + base + (1 if ctx.cp_rank < extra else 0)
            rows = mask[start:end]

        out, chunk_kv = run_chunk_layers(
            layers,
            TINY,
            h,
            positions,
            text_kv=[kv_slice(text, ctx.tp_rank, ctx.tp_size) for text, _ in caches],
            memory_kv=[
                kv_slice(memory, ctx.tp_rank, ctx.tp_size) for _, memory in caches
            ],
            mask=rows,
            ctx=ctx,
        )
        kept = sum(p.numel() for p in torch.nn.ModuleList(list(layers)).parameters())
        torch.save(
            (
                ctx.rank,
                ctx.tp_rank,
                ctx.cp_rank,
                out.clone(),
                [(k.clone(), v.clone()) for k, v in chunk_kv],
                local_kv_heads_for(ctx),
                kept,
            ),
            os.path.join(directory, f"rank{rank}.pt"),
        )
    finally:
        dist.destroy_process_group()


def local_kv_heads_for(ctx) -> int:
    """What ``local_kv_heads`` would say, without a whole network to ask it about."""
    if not ctx.tensor_parallel:
        return TINY.num_key_value_heads
    start, end = balanced_ranges(TINY.num_key_value_heads, ctx.tp_size)[ctx.tp_rank]
    return end - start


def run_mesh_with(worker: Callable, tp: int, cp: int) -> list:
    """Spawn ``tp * cp`` ranks running ``worker`` and collect one result each.

    Results come back through **files** rather than a queue. Putting a tensor on
    a ``multiprocessing.Queue`` sends a shared-memory handle rather than the
    bytes, and the handle stops resolving the moment the sending process exits --
    which shows up as ``FileNotFoundError`` on a ``/tmp/pymp-*`` socket in the
    parent, at whatever moment the race happens to be lost. A file is the whole
    tensor, and it outlives the rank that wrote it.

    Every rank is joined before any of them is judged. Asserting inside the loop
    would leave the temporary directory -- which holds the rendezvous file every
    *other* rank is still waiting on -- to be deleted on the way out, and the
    ranks that were fine would then fail with a missing file store. One slow
    rank would report as several broken ones.
    """
    world = tp * cp
    context = mp.get_context("spawn")
    with tempfile.TemporaryDirectory() as directory:
        init_file = os.path.join(directory, "rendezvous")
        processes = [
            context.Process(target=worker, args=(rank, world, tp, init_file, directory))
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
        return [
            torch.load(os.path.join(directory, f"rank{rank}.pt"), weights_only=False)
            for rank in range(world)
        ]


def run_mesh(tp: int, cp: int) -> list:
    """The layer-level mesh, which is what most of these tests want."""
    return run_mesh_with(_worker, tp, cp)


# A whole tiny network, for the end-to-end rollout check. Two views over five
# latent frames: small, and still the shape of a real clip -- frame 0 is a chunk
# on its own and frames 1..4 follow, so both the short opening chunk and a full
# one get sharded.
ROLLOUT_SPEC = Cosmos3InteractiveSpec(
    llm=TINY,
    vfm=VFMSpec(vae_latent_channels=4, latent_patch_size=2, time_embed_freq_dim=8),
)
ROLLOUT_GEOMETRY = ClipGeometry(
    num_views=2,
    frames_per_view=5,
    patch_h=1,
    patch_w=2,
    frames_per_chunk=4,
    condition_frames=0,
)


def build_tiny_network() -> CosmosMoTNetwork:
    torch.manual_seed(SEED)
    return (
        CosmosMoTNetworkConfig(
            spec=ROLLOUT_SPEC, dtype=torch.float32, include_vision_tower=False
        )
        .setup()
        .eval()
    )


def rollout_inputs():
    generator = torch.Generator().manual_seed(SEED + 2)
    assert ROLLOUT_GEOMETRY.frames_per_view is not None
    tokens = (
        ROLLOUT_GEOMETRY.num_views
        * ROLLOUT_GEOMETRY.frames_per_view
        * ROLLOUT_GEOMETRY.spatial_tokens
    )
    controls = torch.randn(
        tokens, ROLLOUT_SPEC.vfm.patch_latent_dim, generator=generator
    )
    text_ids = torch.randint(0, TINY.vocab_size, (3,), generator=generator)
    return controls, text_ids


@pytest.mark.parametrize(("tp", "cp"), [(2, 1), (1, 2), (1, 4), (1, 3), (2, 2)])
def test_a_mesh_computes_what_one_rank_computes(tp: int, cp: int) -> None:
    """Hidden states and cache keys over five meshes with uneven token shards.

    Thirteen tokens over three ranks produces shards of 5, 4, and 4 tokens.
    Gathering pads the smaller shards for transport and trims them afterward;
    equal-size shards would hide a padding bug.
    """
    expected_h, expected_kv = reference()
    results = run_mesh(tp, cp)

    for _rank, tp_rank, cp_rank, out, chunk_kv, kv_heads, kept in results:
        base, extra = divmod(CHUNK, cp)
        start = cp_rank * base + min(cp_rank, extra)
        end = start + base + (1 if cp_rank < extra else 0)
        torch.testing.assert_close(out, expected_h[:, start:end], atol=1e-5, rtol=1e-5)

        # Every rank's keys cover the whole chunk, because every rank attends it.
        for (k, v), (expected_k, expected_v) in zip(chunk_kv, expected_kv, strict=True):
            assert k.shape[2] == CHUNK
            assert k.shape[1] == kv_heads
            heads = TINY.num_key_value_heads // tp
            head_start = tp_rank * heads
            torch.testing.assert_close(
                k, expected_k[:, head_start : head_start + heads], atol=1e-5, rtol=1e-5
            )
            torch.testing.assert_close(
                v, expected_v[:, head_start : head_start + heads], atol=1e-5, rtol=1e-5
            )

        # A shard that agrees but copied the whole model would pass everything
        # above. The norms and the two per-head RMSNorms stay whole, so this is
        # not exactly 1/tp; it is the projections that had to move, and they are
        # almost all of it.
        if tp > 1:
            assert kept < whole_parameters() * 0.6


@torch.no_grad()
def _rollout_worker(
    rank: int, world: int, tp: int, init_file: str, directory: str
) -> None:
    """One rank of a whole rollout: prefill, every chunk, the finished clip."""
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world,
        timeout=timedelta(seconds=120),
    )
    try:
        ctx = build_context(tp, world // tp, torch.device("cpu"))
        network = build_tiny_network()
        shard_layers(network.layers, network.config.spec.llm, ctx)
        controls, text_ids = rollout_inputs()
        result = run_rollout(
            network,
            geometry=ROLLOUT_GEOMETRY,
            control_tokens=controls,
            text_ids=text_ids,
            parallel=ctx,
        )
        torch.save(
            (ctx.rank, result.tokens.clone()), os.path.join(directory, f"rank{rank}.pt")
        )
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize(("tp", "cp"), [(1, 2), (2, 1), (2, 2), (1, 4), (1, 3)])
def test_a_whole_rollout_matches_a_single_rank(tp: int, cp: int) -> None:
    """The end-to-end claim, and the one a unit test of a layer cannot make.

    A rollout is a prefill, a cache, a sampler and a clean replay wired together,
    and each of them has its own idea of whether it holds the whole chunk or a
    piece of it. Getting that wrong does not raise -- it shards something twice,
    or denoises against noise drawn for somebody else's tokens, and comes out
    finite and wrong. So the check is the clip.
    """
    network = build_tiny_network()
    controls, text_ids = rollout_inputs()
    expected = run_rollout(
        network, geometry=ROLLOUT_GEOMETRY, control_tokens=controls, text_ids=text_ids
    ).tokens

    for _rank, tokens in run_mesh_with(_rollout_worker, tp, cp):
        torch.testing.assert_close(tokens, expected, atol=1e-4, rtol=1e-4)


def whole_parameters() -> int:
    """Parameters in the two layers before anything is sharded."""
    return sum(p.numel() for p in torch.nn.ModuleList(build_layers()).parameters())


def test_a_live_tensor_axis_without_its_group_is_refused() -> None:
    """The all-reduce would silently run over the world group instead.

    Reachable only through a hand-built context -- which is exactly what a test
    reaches for -- so it is worth one check at the point of surgery rather than
    a wrong answer thirty-six layers later.
    """
    ctx = ParallelContext(
        tp_group=None,
        tp_rank=0,
        tp_size=2,
        cp_group=None,
        cp_rank=0,
        cp_size=1,
        device=torch.device("cpu"),
    )
    with pytest.raises(ValueError, match="no tensor process group"):
        shard_layers(build_layers(), TINY, ctx)


def _encode_worker(
    rank: int, world: int, tp: int, init_file: str, directory: str
) -> None:
    import cosmos3_nano_sim_auto.inputs as inputs

    dist.init_process_group(
        backend="gloo", init_method=f"file://{init_file}", rank=rank, world_size=world
    )
    try:
        ctx = build_context(tp, world // tp, torch.device("cpu"))
        geometry = ClipGeometry(
            num_views=3,
            frames_per_view=3,
            patch_h=1,
            patch_w=1,
            frames_per_chunk=2,
            condition_frames=1,
        )
        clips = [
            torch.arange(27).reshape(9, 3, 1, 1).float() + 100 * view
            for view in owned_views(ctx, 3)
        ]
        calls = []

        def encode(clip):
            calls.append(int(clip[0, 0, 0, 0] // 100))
            return clip[::4].permute(1, 0, 2, 3)

        def tokenize(*args, **kwargs):
            assert rank == 0, "only rank 0 should load the tokenizer"
            assert (
                kwargs["system_prompt"]
                == inputs.AV_JOINT_CAMERA_LIDAR_TRANSFER_SYSTEM_PROMPT
            )
            return torch.tensor([4, 8, 15, 16, 23, 42])

        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setattr(inputs, "tokenize_prompt", tokenize)
        sample = inputs.MultiviewSample(
            name="test",
            model_mode="text2video",
            prompt="test",
            resolution="480",
            cameras=("front",),
            control_paths=("unused",),
        )
        prompt = inputs.prompt_token_ids(
            sample, pixel_frames=9, per_view_captions=False, lidar=True, parallel=ctx
        )
        monkeypatch.undo()

        controls = inputs.control_latents(
            clips, encode, geometry=geometry, parallel=ctx
        )
        conditions = inputs.condition_latents(
            [clip[:1] for clip in clips], encode, geometry=geometry, parallel=ctx
        )
        torch.save(
            (prompt, controls, conditions, calls),
            os.path.join(directory, f"rank{rank}.pt"),
        )
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize(("tp", "cp"), [(1, 2), (2, 2)])
def test_parallel_encode_preserves_camera_order_with_uneven_or_empty_shards(tp, cp):
    """Three cameras over two or four ranks: one rank owns two, one owns none."""
    results = run_mesh_with(_encode_worker, tp, cp)
    expected = torch.stack(
        [
            (torch.arange(27).reshape(9, 3, 1, 1).float() + 100 * view)[::4].permute(
                1, 0, 2, 3
            )
            for view in range(3)
        ]
    )
    for rank, (prompt, controls, conditions, calls) in enumerate(results):
        assert torch.equal(prompt.text_ids, torch.tensor([4, 8, 15, 16, 23, 42]))
        assert torch.equal(controls, expected)
        assert torch.equal(conditions, expected[:, :, :1])
        # Spelled out, not owned_views(): this is the independent check on it.
        assert calls == list(range(rank, 3, tp * cp)) * 2


def _rendered_controls(parallel):
    """Three cameras of drawn control whose encoders remember what they saw."""
    from cosmos3_nano_sim_auto.impl.conditioning.control_provider import (
        RenderedControls,
    )

    geometry = ClipGeometry(
        num_views=3,
        frames_per_view=9,
        patch_h=1,
        patch_w=1,
        frames_per_chunk=2,
        condition_frames=1,
    )

    drawn: list[int] = []

    def render(first, last, views=(0, 1, 2)):
        drawn.extend(views)
        if not views:  # LudusControls.render's contract for a rank with no camera
            return torch.empty((0, 0, 0, 3), dtype=torch.uint8)
        frames = torch.arange(first, last).float().view(1, -1, 1, 1, 1)
        offset = torch.tensor(list(views)).float().view(-1, 1, 1, 1, 1) * 100
        return (
            (frames + offset)
            .expand(len(views), last - first, 1, 1, 4)
            .reshape(-1, 1, 1, 4)
            .byte()
        )

    calls: list[int] = []

    def stream(view):
        seen = [0.0]

        def encode(clip):
            calls.append(view)
            # Temporal state: each output depends on every earlier call.
            seen[0] += float(clip.sum())
            latent = clip[(clip.shape[0] - 1) % 4 :: 4, :1].permute(1, 0, 2, 3)
            return latent + seen[0]

        return encode

    controls = RenderedControls(
        render,
        [stream(view) for view in range(3)],
        geometry=geometry,
        latent_patch_size=1,
        parallel=parallel,
    )
    tokens = [controls.tokens(start, end) for start, end in ((0, 1), (1, 3), (3, 5))]
    return tokens, calls, drawn


def _rendered_controls_worker(
    rank: int, world: int, tp: int, init_file: str, directory: str
) -> None:
    dist.init_process_group(
        backend="gloo", init_method=f"file://{init_file}", rank=rank, world_size=world
    )
    try:
        ctx = build_context(tp, world // tp, torch.device("cpu"))
        torch.save(_rendered_controls(ctx), os.path.join(directory, f"rank{rank}.pt"))
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize(("tp", "cp"), [(1, 2), (2, 2)])
def test_drawn_control_encode_is_split_by_camera_and_matches_one_rank(tp, cp):
    """Each camera is drawn and encoded on one owner only; every rank gets all."""
    expected, _, _ = _rendered_controls(None)
    for rank, (tokens, calls, drawn) in enumerate(
        run_mesh_with(_rendered_controls_worker, tp, cp)
    ):
        for got, want in zip(tokens, expected, strict=True):
            assert torch.equal(got, want)
        assert drawn == list(range(rank, 3, tp * cp)) * 3
        assert calls == list(range(rank, 3, tp * cp)) * 3
