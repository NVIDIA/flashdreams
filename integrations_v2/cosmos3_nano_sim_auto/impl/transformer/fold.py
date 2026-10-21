# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Execute attention partitions as unmasked dense attention.

:mod:`cosmos3_nano_sim_auto.impl.transformer.mask` defines visibility.
This module groups each partition's keys into contiguous runs for fused
attention, then merges the results. Plans are derived from masks over runs
rather than individual tokens to keep planning memory bounded.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass

import torch
from torch import Tensor

from cosmos3_nano_sim_auto.impl.transformer.mask import (
    AttentionPattern,
    AttentionScope,
    PartitionName,
    StreamFields,
    maskless_partitions,
)

# The metadata the visibility rules actually branch on. `timestamp` is absent
# because it is `frame_id` times a constant, so a run of equal frames is a run
# of equal timestamps, and comparing the floats would only add a way to split a
# run that should not be split.
_RUN_FIELDS = (
    "sample_id",
    "frame_id",
    "view_id",
    "is_noisy",
    "is_control",
    "token_role_id",
    "causal_step_id",
)


@dataclass(frozen=True)
class Span:
    """A half-open ``[start, stop)`` run of one stream's tokens."""

    start: int
    stop: int

    def __len__(self) -> int:
        return self.stop - self.start


@dataclass(frozen=True)
class FoldGroup:
    """Queries that read the same keys, and those keys.

    One dense rectangle: every query here reads every key here, so the kernel
    that runs it needs no mask. Held as spans rather than index tensors because
    most of them turn out to be single runs -- our tokens are laid out camera
    by camera, so a camera's own keys are already contiguous -- and a single
    span is a view rather than a copy.
    """

    queries: tuple[Span, ...]
    keys: tuple[Span, ...]

    @property
    def query_count(self) -> int:
        return sum(len(span) for span in self.queries)

    @property
    def key_count(self) -> int:
        return sum(len(span) for span in self.keys)


@dataclass(frozen=True)
class FoldPlan:
    """Every pass of a fold, as rectangles a maskless kernel can run.

    Within a pass the groups partition the queries: a query's key set puts it in
    exactly one of them, so a pass's outputs can be written straight into place
    with nothing to merge. Merging happens once, across passes, where the
    checkpoint selects whether those passes overlap.
    """

    passes: tuple[tuple[PartitionName, tuple[FoldGroup, ...]], ...]
    query_length: int
    key_length: int
    fused_merge: bool = False
    """Opt into the qualified CUDA inference merge; other inputs stay native."""

    @property
    def group_count(self) -> int:
        """How many kernel launches a chunk pass costs."""
        return sum(len(groups) for _name, groups in self.passes)

    def to(self, device: torch.device | str) -> FoldPlan:
        """Answer the move a mask would, by having nothing to move.

        :func:`~cosmos3_nano_sim_auto.impl.transformer.forward.run_chunk_layers` moves what it is given
        before the first layer reads it. A plan is spans of integers rather than
        tensors, so there is nothing on a device to relocate -- but it stands
        where a mask stood, so it has to answer the call.
        """
        del device
        return self


def _runs(fields: StreamFields) -> tuple[StreamFields, list[int]]:
    """Collapse the stream to one representative token a run of equal metadata.

    The visibility rules read only the fields in :data:`_RUN_FIELDS`, so tokens
    agreeing on all of them are interchangeable to every rule and can be decided
    together. That is what keeps this planner cheap: a four-camera chunk is tens
    of runs where it is tens of thousands of tokens, and the partition masks are
    built at run resolution below.

    Returns:
        The representative fields, ``[R]``, and the ``R + 1`` token boundaries.
    """
    stacked = torch.stack(
        [getattr(fields, name).long() for name in _RUN_FIELDS], dim=1
    )  # [S, len(_RUN_FIELDS)]
    changed = (stacked[1:] != stacked[:-1]).any(dim=1)
    starts = [0, *(index + 1 for index in changed.nonzero().flatten().tolist())]
    boundaries = [*starts, len(fields)]

    at = torch.tensor(starts, device=fields.sample_id.device)
    representative = StreamFields(
        sample_id=fields.sample_id[at],
        frame_id=fields.frame_id[at],
        view_id=fields.view_id[at],
        is_noisy=fields.is_noisy[at],
        is_control=fields.is_control[at],
        timestamp=fields.timestamp[at],
        token_role_id=fields.token_role_id[at],
        causal_step_id=fields.causal_step_id[at],
    )
    return representative, boundaries


def _spans(runs: list[int], boundaries: list[int]) -> tuple[Span, ...]:
    """Turn run indices into token spans, joining any that abut.

    Joining matters more than it looks: a camera's keys arrive as one run a
    frame, and left unjoined that would be one kernel launch a frame instead of
    one for the camera.
    """
    merged: list[Span] = []
    for index in sorted(runs):
        start, stop = boundaries[index], boundaries[index + 1]
        if merged and merged[-1].stop == start:
            merged[-1] = Span(merged[-1].start, stop)
        else:
            merged.append(Span(start, stop))
    return tuple(merged)


def build_fold_plan(
    q: StreamFields,
    kv: StreamFields,
    *,
    pattern: AttentionPattern = "causal",
    scope: AttentionScope = "decomposed",
    decomposed_temporal_window_seconds: float | None = None,
    joint: bool = False,
    deduplicate_cross_view: bool = False,
) -> FoldPlan:
    """Plan one chunk pass as dense rectangles, one group a distinct key set.

    Queries reading nothing in a pass are left out rather than launched as an
    empty kernel; they come back zeroed from the merge, which is what the
    single masked pass gives them too.
    """
    query_runs, query_bounds = _runs(q)
    key_runs, key_bounds = _runs(kv)

    passes: list[tuple[PartitionName, tuple[FoldGroup, ...]]] = []
    for name, dense in maskless_partitions(
        query_runs,
        key_runs,
        pattern=pattern,
        joint=joint,
        scope=scope,
        decomposed_temporal_window_seconds=decomposed_temporal_window_seconds,
        deduplicate_cross_view=deduplicate_cross_view,
    ):
        grouped: dict[tuple[int, ...], list[int]] = {}
        # The planner builds Python spans, so transfer the compact mask once.
        for index, row in enumerate(dense.cpu()):
            keys = tuple(row.nonzero().flatten().tolist())
            if keys:
                grouped.setdefault(keys, []).append(index)
        passes.append(
            (
                name,
                tuple(
                    FoldGroup(
                        queries=_spans(queries, query_bounds),
                        keys=_spans(list(keys), key_bounds),
                    )
                    for keys, queries in grouped.items()
                ),
            )
        )

    return FoldPlan(passes=tuple(passes), query_length=len(q), key_length=len(kv))


def _take(tensor: Tensor, spans: tuple[Span, ...]) -> Tensor:
    """Gather ``[B, H, S, D]`` down to the spans, as a view where it can be."""
    if len(spans) == 1:
        return tensor[:, :, spans[0].start : spans[0].stop]
    return torch.cat([tensor[:, :, span.start : span.stop] for span in spans], dim=2)


def _dense_attention(q: Tensor, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
    """Unmasked attention and its log-sum-exp, from the fused kernel.

    No mask, no score modifier, nothing for the kernel to check per element --
    which is the entire reason the keys were rearranged to get here. The CPU
    entry is a separate operator rather than a fallback; both return the
    log-sum-exp the merge needs, which ``scaled_dot_product_attention`` does
    not expose.
    """
    if q.is_cuda:
        out, lse, *_rest = torch.ops.aten._scaled_dot_product_flash_attention(q, k, v)
        return out, lse
    out, lse = torch.ops.aten._scaled_dot_product_flash_attention_for_cpu(q, k, v)
    return out, lse


# Batched fold passes: every group of a pass with the same query and key count
# runs as one batched launch.
#
#
# Per group, the plain path does a ``torch.cat`` gather of K and V and one flash
# launch (:func:`_take` + :func:`_dense_attention`). Groups with equal query and
# key counts can ride the kernel's batch axis together: one ``index_select``
# gather of K and V into ``[B*G, Hkv, L, D]`` and one launch. At this rig's
# steady chunk the same-view pass (4 groups) and the windowed cross-view pass
# (3 groups, one per frame instant) each become one launch; pre-steady plans
# split into a few buckets. The kernel computes each query row against its own
# key rows as it would in a separate launch; the outputs are scattered back to
# the query axis and the merge (eager or fused) receives the same
# ``[B, H, Q, d]`` and ``[B, H, Q]`` inputs the per-group path produces.
# Caption groups (single-span keys of unequal length) stay on the plain path.


@dataclass(frozen=True)
class _Bucket:
    """Groups of one pass with equal query and key counts, as index tensors."""

    groups: int
    query_count: int
    key_count: int
    query_index: Tensor | None
    """``None`` when the groups' queries are the whole query axis in order (no reordering)."""
    key_index: Tensor


def _span_index(spans: tuple[Span, ...], device: torch.device) -> Tensor:
    return torch.cat(
        [torch.arange(span.start, span.stop, device=device) for span in spans]
    )


@functools.cache
def _sm_count(device: torch.device) -> int:
    return torch.cuda.get_device_properties(device).multi_processor_count


@functools.lru_cache(maxsize=64)
def _pass_buckets(
    groups: tuple[FoldGroup, ...], query_length: int, heads: int, device: torch.device
) -> tuple[tuple[tuple[int, ...], _Bucket | None], ...]:
    """Partition a pass's groups by ``(query_count, key_count)``; built once per distinct plan.

    Plans repeat (a bounded cache rotates through a few slot orders and the
    frozen groups hash by value), so this runs a handful of times a clip.
    Returns ``(group numbers, bucket)`` pairs; a lone group in its shape class
    gets ``None`` and runs the plain path, and so does a class whose separate
    launches would take the flash kernel's split-KV path: that path sums the
    key blocks in another order, so one batched launch (more work units, no
    split) would differ from them at the last bit. The kernel splits when
    ``batch * heads * ceil(Lq / 128) < 0.8 * SMs`` (measured: 390-query groups
    differ, 780 and up are bit-equal for any batch).
    ponytail: this mirrors the kernel's heuristic (query block 128, 0.8 x SMs); if
    it moves, the paired GPU runs stop being byte-identical (the CPU test cannot see it).
    """
    by_shape: dict[tuple[int, int], list[int]] = {}
    for number, group in enumerate(groups):
        by_shape.setdefault((group.query_count, group.key_count), []).append(number)
    out = []
    for (count_q, count_k), numbers in by_shape.items():
        if len(numbers) < 2 or heads * -(-count_q // 128) < 0.8 * _sm_count(device):
            out.append((tuple(numbers), None))
            continue
        members = [groups[n] for n in numbers]
        query_index = torch.cat([_span_index(g.queries, device) for g in members])
        key_index = torch.cat([_span_index(g.keys, device) for g in members])
        identity = int(query_index.numel()) == query_length and bool(
            torch.equal(query_index, torch.arange(query_length, device=device))
        )
        out.append(
            (
                tuple(numbers),
                _Bucket(
                    len(numbers),
                    count_q,
                    count_k,
                    None if identity else query_index,
                    key_index,
                ),
            )
        )
    return tuple(out)


def _batched_pass(
    q: Tensor, k: Tensor, v: Tensor, bucket: _Bucket
) -> tuple[Tensor, Tensor]:
    """One batched kernel launch for a bucket; returns ``[B, H, G*Lq, d]`` and ``[B, H, G*Lq]``."""
    batch, heads, _length, width = q.shape
    kv_heads = k.shape[1]
    groups, count_q, count_k = bucket.groups, bucket.query_count, bucket.key_count
    q_groups = (
        q if bucket.query_index is None else q.index_select(2, bucket.query_index)
    )
    q_groups = (
        q_groups.reshape(batch, heads, groups, count_q, width)
        .transpose(1, 2)
        .reshape(batch * groups, heads, count_q, width)
    )

    def gather(tensor: Tensor) -> Tensor:
        return (
            tensor.index_select(2, bucket.key_index)
            .view(batch, kv_heads, groups, count_k, width)
            .transpose(1, 2)
            .reshape(batch * groups, kv_heads, count_k, width)
        )

    out, lse = _dense_attention(q_groups, gather(k), gather(v))
    out = out.view(batch, groups, heads, count_q, width).transpose(1, 2)
    lse = lse.view(batch, groups, heads, count_q).transpose(1, 2)
    return out.reshape(batch, heads, groups * count_q, width), lse.reshape(
        batch, heads, groups * count_q
    )


def gathered_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    plan: FoldPlan,
    *,
    enable_gqa: bool = False,
) -> Tensor:
    """Run a :class:`FoldPlan`, merging its passes by log-sum-exp.

    Args:
        q: ``[B, Hq, Q, d]``, the chunk's queries.
        k, v: ``[B, Hq, KV, d]``, or ``[B, Hkv, KV, d]`` with ``enable_gqa``.
        plan: from :func:`build_fold_plan`, for these exact lengths.
        enable_gqa: Let query heads share key/value heads. Both fused kernels
            read grouped heads as they are, so nothing is expanded here; the
            flag only states the layout ``k`` and ``v`` arrive in.

    Returns:
        ``[B, Hq, Q, d]``, in ``q``'s dtype.
    """
    from cosmos3_nano_sim_auto.impl.transformer.attention import merge_attentions

    if (plan.query_length, plan.key_length) != (q.shape[-2], k.shape[-2]):
        raise ValueError(
            f"Plan for {plan.query_length} queries and {plan.key_length} keys "
            f"does not match the {q.shape[-2]} and {k.shape[-2]} it must run."
        )

    batched = q.is_cuda
    parts: list[tuple[Tensor, Tensor]] = []
    for _name, groups in plan.passes:
        # -inf, not zero: a query this pass cannot read has to weigh nothing in
        # the merge, and zero is a weight of one.
        out = torch.zeros_like(q)
        lse = q.new_full(q.shape[:-1], float("-inf"), dtype=torch.float32)
        singles: tuple[int, ...] | None = None
        if batched:
            singles = ()
            for numbers, bucket in _pass_buckets(
                groups, plan.query_length, q.shape[1], q.device
            ):
                if bucket is None:
                    singles += numbers
                    continue
                b_out, b_lse = _batched_pass(q, k, v, bucket)
                if bucket.query_index is None:
                    # copy_, not rebind: the merge's inputs keep q's strides
                    # exactly as on the per-group path (the fused kernel checks them).
                    out.copy_(b_out)
                    lse.copy_(b_lse)
                else:
                    out.index_copy_(2, bucket.query_index, b_out.to(out.dtype))
                    lse.index_copy_(2, bucket.query_index, b_lse.to(lse.dtype))
        for group_no, group in enumerate(groups):
            if singles is not None and group_no not in singles:
                continue
            gathered, gathered_lse = _dense_attention(
                _take(q, group.queries), _take(k, group.keys), _take(v, group.keys)
            )
            at = 0
            for span in group.queries:
                width = len(span)
                out[:, :, span.start : span.stop] = gathered[:, :, at : at + width].to(
                    out.dtype
                )
                lse[:, :, span.start : span.stop] = gathered_lse[
                    :, :, at : at + width
                ].to(lse.dtype)
                at += width
        parts.append((out, lse))

    if plan.fused_merge:
        return merge_attentions(parts, fused=True, output_dtype=q.dtype)
    return merge_attentions(parts).to(q.dtype)
