# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cosmos MoT projection and head mapping onto the core tensor-parallel layers."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import cast

import torch
from cosmos_interactive._network import (
    CosmosMoTNetwork,
    _DecoderHalf,
    _MoTLayer,
)
from cosmos_interactive._spec import LLMSpec
from torch import nn

from flashdreams.core.distributed.parallel import ParallelContext, balanced_ranges
from flashdreams.core.distributed.tensor_parallel import (
    ColumnParallelLinear,
    RowParallelLinear,
)


def _halves(layers: Sequence[_MoTLayer]) -> Iterator[_DecoderHalf]:
    """Every decoder half of every layer: the understanding one, then the generating one.

    Both are sharded. The generating half is the one a rollout spends its time
    in, but the prompt still runs through the understanding half at startup, and
    a half left whole would hold weights this axis exists to be rid of.
    """
    for layer in layers:
        yield layer.und
        yield layer.gen


def shard_network(network: CosmosMoTNetwork, ctx: ParallelContext) -> CosmosMoTNetwork:
    """Replace every layer's projections with this rank's slice of them, in place.

    A no-op at ``tp_size == 1``, so a caller need not branch.

    The embedding table, the patch adapters, the timestep embedder and the final
    norms are left whole. They are about 2.5 GiB together against the layers'
    27.8, they carry no head axis to cut along, and replicating them costs one
    all-gather less per pass than sharding them would.

    Returns:
        The same network object, for chaining onto a load.
    """
    shard_layers(network.layers, network.config.spec.llm, ctx)
    if ctx.tensor_parallel and ctx.device.type == "cuda":
        # The originals are unreferenced now, but the caching allocator holds
        # their blocks, and the whole point of this axis is the memory. Returning
        # them here means the peak reported for a run is the run's rather than
        # the load's.
        torch.cuda.empty_cache()
    return network


def shard_layers(
    layers: Sequence[_MoTLayer], llm: LLMSpec, ctx: ParallelContext
) -> Sequence[_MoTLayer]:
    """The surgery itself, over a bare layer list.

    Separate from :func:`shard_network` so that the equivalence tests can shard
    two tiny layers without a vocabulary-sized embedding table behind them --
    which is the only way this axis gets checked anywhere but on a GPU.
    """
    if not ctx.tensor_parallel:
        return layers
    groups = llm.num_key_value_heads
    # Validate the loaded model before replacing any of its projections.
    if ctx.tp_size > groups:
        raise ValueError(
            f"tp_size={ctx.tp_size} is past the {groups} key/value head groups this "
            "checkpoint has to hand out."
        )
    # A live tensor axis with no group of its own would all-reduce over the world
    # group, silently summing across context ranks as well. Unreachable through
    # build_context and one line away through a hand-built context.
    if ctx.tp_group is None:
        raise ValueError(
            f"tp_size={ctx.tp_size} with no tensor process group; an all-reduce would "
            "run over the world group and mix the context axis into it."
        )

    per_group = llm.num_attention_heads // groups
    head_dim = llm.head_dim
    group_start, group_end = balanced_ranges(groups, ctx.tp_size)[ctx.tp_rank]
    mlp_start, mlp_end = balanced_ranges(llm.intermediate_size, ctx.tp_size)[
        ctx.tp_rank
    ]

    q_start, q_end = (
        group_start * per_group * head_dim,
        group_end * per_group * head_dim,
    )
    kv_start, kv_end = group_start * head_dim, group_end * head_dim
    keeps_bias = ctx.tp_rank == 0

    for half in _halves(layers):
        attn = half.self_attn
        attn.q_proj = ColumnParallelLinear(cast(nn.Linear, attn.q_proj), q_start, q_end)
        attn.k_proj = ColumnParallelLinear(
            cast(nn.Linear, attn.k_proj), kv_start, kv_end
        )
        attn.v_proj = ColumnParallelLinear(
            cast(nn.Linear, attn.v_proj), kv_start, kv_end
        )
        attn.o_proj = RowParallelLinear(
            cast(nn.Linear, attn.o_proj),
            q_start,
            q_end,
            group=ctx.tp_group,
            keeps_bias=keeps_bias,
        )
        # The network reshapes projections with these, so they have to follow the
        # weights down. ``num_kv_groups`` deliberately does not move: grouped-query
        # attention still binds four query heads to each key/value head, and that
        # ratio is what makes a whole-group split the only legal one.
        attn.num_heads = (group_end - group_start) * per_group
        attn.num_kv_heads = group_end - group_start

        mlp = half.mlp
        mlp.gate_proj = ColumnParallelLinear(
            cast(nn.Linear, mlp.gate_proj), mlp_start, mlp_end
        )
        mlp.up_proj = ColumnParallelLinear(
            cast(nn.Linear, mlp.up_proj), mlp_start, mlp_end
        )
        mlp.down_proj = RowParallelLinear(
            cast(nn.Linear, mlp.down_proj),
            mlp_start,
            mlp_end,
            group=ctx.tp_group,
            keeps_bias=keeps_bias,
        )
    return layers


def local_kv_heads(network: CosmosMoTNetwork, ctx: ParallelContext) -> int:
    """Key/value heads this rank's cache has to hold per layer.

    The whole count when the tensor axis is inert. Asked for rather than derived
    at each call site, because the empty cache a prefill starts from has to be
    shaped for it and a mismatch there is a shape error thirty-six layers in.
    """
    llm = network.config.spec.llm
    if not ctx.tensor_parallel:
        return llm.num_key_value_heads
    start, end = balanced_ranges(llm.num_key_value_heads, ctx.tp_size)[ctx.tp_rank]
    return end - start


def parameter_bytes(network: CosmosMoTNetwork) -> int:
    """Bytes of parameter this rank holds, for the memory column of a report."""
    return sum(p.numel() * p.element_size() for p in network.parameters())
