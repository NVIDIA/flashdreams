# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Populate the key/value cache before autoregressive generation.

Prefill encodes clean controls and supplied opening frames. ``control_frames``
bounds the initial control window; :func:`run_control_topup` supplies later
controls. ``target_frames`` must not include unknown future images: noise in
those positions would corrupt the attention sink.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from cosmos_interactive._network import _MoTLayer
from cosmos_interactive._spec import LLMSpec
from torch import Tensor
from torch.nn.attention.flex_attention import BlockMask

from cosmos3_nano_sim_auto.impl.packing import (
    ClipGeometry,
    _check_ranges,
    _stamp_view_major,
    _vision,
    concat,
    text_stream,
)
from cosmos3_nano_sim_auto.impl.transformer.attention import ChunkMask, MasklessFold
from cosmos3_nano_sim_auto.impl.transformer.fold import FoldPlan, build_fold_plan
from cosmos3_nano_sim_auto.impl.transformer.forward import (
    LayerKV,
    run_chunk_layers,
    run_text_layers,
)
from cosmos3_nano_sim_auto.impl.transformer.mask import (
    ROLE_CLEAN_TARGET,
    ROLE_CONTROL,
    ROLE_TARGET_CONDITION,
    AttentionScope,
    StreamFields,
    build_block_mask,
    build_maskless_block_masks,
    visibility,
)
from flashdreams.accelerated.multi_head_attention.optimized import OptimizedImplConfig
from flashdreams.core.distributed.parallel import ParallelContext


@dataclass(frozen=True)
class PrefillPack:
    """Token layout for the clean pass that fills the cache.

    Attributes:
        und: the prompt, ``[num_und]``.
        gen: the vision tokens, ``[N]`` -- one section per control range, then
            the target. Each section is view-outer, frame-inner,
            spatial-innermost, over its own frames alone.
        memory_token_indexes: which rows of ``gen`` are kept, in the order the
            cache wants them. Worked out here because the strides are this
            pack's rather than the geometry's: the target is only
            ``target_frames`` deep per view, the control track deeper.
        control_ranges: the frames each control section covers, in order. The
            caller reads back what it asked for, because the cache has to be
            told the same ranges and the two agreeing is the whole point.

    One section per range rather than one block over all the control frames.
    The two are the same tokens in a different order, and the order has to be
    the one :func:`~cosmos3_nano_sim_auto.impl.packing.build_memory_layout` describes, or
    the mask admits keys belonging to another camera and another instant.
    """

    und: StreamFields
    gen: StreamFields
    memory_token_indexes: Tensor
    control_ranges: tuple[tuple[int, int], ...]

    @property
    def num_und(self) -> int:
        return len(self.und)

    def queries(self, query_range: tuple[int, int] | None) -> StreamFields:
        """The vision tokens this pass asks about: all of them, or one rank's range.

        The key stream stays whole either way, so a restricted pass computes the
        same keys and values for its own rows that an unrestricted one would --
        which is what lets the ranges be gathered back into one cache.
        """
        return self.gen if query_range is None else self.gen.narrow(*query_range)

    def mask(
        self,
        *,
        scope: AttentionScope = "all_views",
        query_range: tuple[int, int] | None = None,
        decomposed_temporal_window_seconds: float | None = None,
    ) -> Tensor:
        """The ``[N, num_und + N]`` visibility mask for the pass.

        Queries are the vision tokens; the prompt is keys only, exactly as during
        the rollout. Nothing is cached yet, so the key stream is
        ``[text | vision]`` with the memory section empty.

        The scope has to be the one the checkpoint was trained with and has to
        match the rollout's, since the cache this pass writes is read by every
        chunk after it. The pattern stays at :mod:`cosmos3_nano_sim_auto.impl.transformer.mask`'s
        default, which no delivered configuration has varied.

        ``query_range`` narrows the rows to one context-parallel rank's share;
        the key axis stays whole. :meth:`block_mask` and :meth:`fold_plan` take
        it too.
        """
        return visibility(
            self.queries(query_range),
            concat(self.und, self.gen),
            scope=scope,
            decomposed_temporal_window_seconds=decomposed_temporal_window_seconds,
        )

    def block_mask(
        self,
        *,
        scope: AttentionScope = "all_views",
        block_size: int | tuple[int, int] = 128,
        query_range: tuple[int, int] | None = None,
        decomposed_temporal_window_seconds: float | None = None,
    ) -> BlockMask:
        """The same visibility as :meth:`mask`, as blocks FlexAttention can skip.

        Far emptier than a chunk's mask: a control reads its own view here and,
        across the rig, only the conditioning frames inside the temporal window,
        so the mask is close to block-diagonal by view. At eleven views 94% of blocks are empty, against
        53% for a chunk.

        That matters twice over. The dense form of this mask is 5.0 GiB at eleven
        views and :func:`~cosmos3_nano_sim_auto.impl.transformer.mask.visibility` holds about a dozen
        live, which is most of the memory floor -- and the floor, not the
        rollout, is what caps how long a clip can be.
        """
        return build_block_mask(
            self.queries(query_range),
            concat(self.und, self.gen),
            scope=scope,
            block_size=block_size,
            decomposed_temporal_window_seconds=decomposed_temporal_window_seconds,
        )

    def fold_plan(
        self,
        *,
        scope: AttentionScope = "decomposed",
        query_range: tuple[int, int] | None = None,
        decomposed_temporal_window_seconds: float | None = None,
        deduplicate_cross_view: bool = False,
    ) -> FoldPlan:
        """the model's passes as rectangles for an unmasked kernel.

        The prefill needs this as much as the chunks do. The cache it writes is
        what every chunk reads, so a run folding its chunks and masking its
        prefill would be two attentions deep before its first frame.
        """
        return build_fold_plan(
            self.queries(query_range),
            concat(self.und, self.gen),
            scope=scope,
            decomposed_temporal_window_seconds=decomposed_temporal_window_seconds,
            deduplicate_cross_view=deduplicate_cross_view,
        )

    def maskless_fold(
        self,
        *,
        scope: AttentionScope = "decomposed",
        block_size: int | tuple[int, int] = 128,
        decomposed_temporal_window_seconds: float | None = None,
        deduplicate_cross_view: bool = False,
    ) -> MasklessFold:
        """The same passes as :meth:`fold_plan`, each gated by a block mask.

        Kept as what :meth:`fold_plan` is checked against rather than as a way
        to run a clip; see :meth:`~cosmos3_nano_sim_auto.impl.packing.ChunkMetadata.maskless_fold`.
        """
        return MasklessFold(
            build_maskless_block_masks(
                self.gen,
                concat(self.und, self.gen),
                scope=scope,
                block_size=block_size,
                decomposed_temporal_window_seconds=decomposed_temporal_window_seconds,
                deduplicate_cross_view=deduplicate_cross_view,
            )
        )


def build_prefill_pack(
    geometry: ClipGeometry,
    *,
    text_tokens: int,
    view_text_tokens: Sequence[int] | None = None,
    target_frames: int | None = None,
    control_ranges: Sequence[tuple[int, int]] | None = None,
    device: torch.device | None = None,
) -> PrefillPack:
    """Lay out the clean pass.

    Args:
        view_text_tokens: tokens in each camera's own caption, in rig order,
            when the prompt is one caption each rather than one for the rig.
            See :func:`~cosmos3_nano_sim_auto.impl.packing.text_stream`.
        target_frames: latent frames the target spans. Defaults to the supplied
            frames, which is all a demo has. Pass ``geometry.frames_per_view``
            to reproduce the model's pack, which the module docstring qualifies.
        control_ranges: the control frames to prefill, one range per cache slot.
            Defaults to the whole clip in one range. A bounded cache prefills a
            window of :func:`~cosmos3_nano_sim_auto.impl.packing.control_chunk_ranges`
            instead and tops the rest up as the rollout reaches it, since a
            stream has no whole clip to front-load.
    """
    device = device or torch.device("cpu")
    target_frames = (
        geometry.condition_frames if target_frames is None else target_frames
    )
    last = geometry.frames_per_view
    if target_frames < geometry.condition_frames or (
        last is not None and target_frames > last
    ):
        raise ValueError(
            f"target_frames must cover the {geometry.condition_frames} conditioning frames "
            f"and fit the clip's {last}, got {target_frames}."
        )
    if control_ranges is None:
        if last is None:
            raise ValueError(
                "a clip with no end has no whole control track to prefill. Pass "
                "the window the cache holds as control_ranges."
            )
        control_ranges = [(0, last)]
    control_ranges = tuple(control_ranges)
    _check_ranges(
        control_ranges,
        first_frame=0,
        last_frame=last,
        what="Control",
    )
    if not control_ranges:
        raise ValueError("a prefill needs at least one control range.")

    num_views, spatial = geometry.num_views, geometry.spatial_tokens
    target_frame, target_view = _stamp_view_major(
        torch.arange(target_frames, device=device), num_views, spatial
    )
    is_condition = target_frame < geometry.condition_frames

    def control(start: int, end: int) -> StreamFields:
        frame, view = _stamp_view_major(
            torch.arange(start, end, device=device), num_views, spatial
        )
        return _vision(
            frame, view, role=ROLE_CONTROL, is_noisy=False, geometry=geometry
        )

    gen = concat(
        *(control(start, end) for start, end in control_ranges),
        _vision(
            target_frame,
            target_view,
            role=torch.where(
                is_condition,
                torch.full_like(target_frame, ROLE_TARGET_CONDITION),
                torch.full_like(target_frame, ROLE_CLEAN_TARGET),
            ),
            is_noisy=~is_condition,
            geometry=geometry,
        ),
    )

    # The cache wants the controls whole and then the supplied frames, and
    # ``nonzero`` returns ascending indexes, which is that order already.
    keep = gen.is_control | (gen.token_role_id == ROLE_TARGET_CONDITION)
    return PrefillPack(
        und=text_stream(text_tokens, device, view_text_tokens=view_text_tokens),
        gen=gen,
        memory_token_indexes=keep.nonzero().squeeze(-1),
        control_ranges=control_ranges,
    )


def pad_control_slots(
    memory_kv: Sequence[LayerKV],
    *,
    control_ranges: Sequence[tuple[int, int]],
    tokens_per_frame: int,
    slot_frames: int,
) -> list[LayerKV]:
    """Spread the prefilled control sections over fixed-width cache slots.

    The pass computes real tokens only, while a bounded cache holds slots of one
    chunk each so a top-up can overwrite the oldest whole. A range shorter than
    a slot -- the frame the clip opens with -- leaves the rest of its slot
    holding keys nothing reads, which the layout describes as padding.

    Whatever follows the control sections, being the supplied frames, keeps its
    place after the last slot.
    """
    sections = [(end - start) * tokens_per_frame for start, end in control_ranges]
    slot_tokens = slot_frames * tokens_per_frame
    if any(section > slot_tokens for section in sections):
        raise ValueError(
            f"control ranges {list(control_ranges)} do not fit {slot_frames}-frame slots."
        )
    if all(section == slot_tokens for section in sections):
        return list(memory_kv)

    control_tokens = len(sections) * slot_tokens
    padded: list[LayerKV] = []
    for k, v in memory_kv:
        rest = int(k.shape[2]) - sum(sections)
        pair = []
        for source in (k, v):
            buffer = source.new_zeros(
                (
                    source.shape[0],
                    source.shape[1],
                    control_tokens + rest,
                    source.shape[3],
                )
            )
            read = 0
            for slot, section in enumerate(sections):
                start = slot * slot_tokens
                buffer[:, :, start : start + section] = source[
                    :, :, read : read + section
                ]
                read += section
            buffer[:, :, control_tokens:] = source[:, :, read:]
            pair.append(buffer)
        padded.append((pair[0], pair[1]))
    return padded


def run_control_topup(
    layers: Sequence[_MoTLayer],
    llm: LLMSpec,
    *,
    control_h: Tensor,
    control_position_ids: Tensor,
    text_kv: Sequence[LayerKV],
    memory_kv: Sequence[LayerKV],
    mask: ChunkMask,
    attention_implementation: OptimizedImplConfig | None = None,
    native_gqa_attention: bool = False,
    accelerated_rms_norm: bool = False,
    ctx: ParallelContext | None = None,
) -> list[LayerKV]:
    """Keys and values for control frames the prefill did not cover.

    A prefill holding the whole control track needs none of this. A bounded one
    has to add frames as the rollout reaches them, which is also all a live
    renderer can do, having nothing beyond the current moment to give.

    Control reads the prompt, earlier control in its own view and the clean
    RGB of earlier steps in scope -- under ``decomposed``, every view within
    the temporal window (the model's ``controls_read_strict_past_clean_rgb``) -- never
    its own step's target. That is why this runs after the previous chunk is
    committed and before the chunk it serves. A prefill's window computes the
    controls one chunk beyond the first before that chunk exists; at the
    default history that is the second chunk alone.

    Args:
        control_h: ``[B, N, D]`` embedded control patches for the new frames.
        control_position_ids: ``[3, N]``, from
            :func:`~cosmos3_nano_sim_auto.impl.transformer.rope.chunk_mrope_ids` over the same frames.
        mask: from ``build_chunk_metadata(..., pass_kind="control")``.

    Returns:
        One entry per layer, for the caller to write into its control window.
    """
    _out, control_kv = run_chunk_layers(
        layers,
        llm,
        control_h,
        control_position_ids,
        text_kv=text_kv,
        memory_kv=memory_kv,
        mask=mask,
        attention_implementation=attention_implementation,
        native_gqa_attention=native_gqa_attention,
        accelerated_rms_norm=accelerated_rms_norm,
        ctx=ctx,
    )
    return control_kv


def run_prefill(
    layers: Sequence[_MoTLayer],
    llm: LLMSpec,
    *,
    text_h: Tensor,
    text_position_ids: Tensor,
    view_text_tokens: Sequence[int] | None = None,
    vision_h: Tensor,
    vision_position_ids: Tensor,
    mask: ChunkMask,
    memory_token_indexes: Tensor,
    attention_implementation: OptimizedImplConfig | None = None,
    native_gqa_attention: bool = False,
    accelerated_rms_norm: bool = False,
    ctx: ParallelContext | None = None,
    kv_heads: int | None = None,
) -> tuple[list[LayerKV], list[LayerKV]]:
    """Run the clean pass and return the caches the rollout needs.

    Args:
        text_h: ``[B, num_und, D]`` embedded prompt tokens.
        text_position_ids: ``[3, num_und]``, from
            ``cosmos_interactive._rope.text_mrope_ids``, or from
            :func:`~cosmos3_nano_sim_auto.impl.transformer.rope.caption_mrope_ids` for one caption per
            camera.
        view_text_tokens: the length of each camera's own caption, passed on to
            :func:`~cosmos3_nano_sim_auto.impl.transformer.forward.run_text_layers`, which explains what
            it does to the prompt's self-attention.
        vision_h: ``[B, N, D]`` embedded vision tokens -- the control track then
            the target, already patch-embedded by the caller.
        vision_position_ids: ``[3, N]``, from
            :func:`~cosmos3_nano_sim_auto.impl.transformer.rope.chunk_mrope_ids`. Control and target
            share their positions, so the two halves agree wherever their
            frames do.
        mask: from :meth:`PrefillPack.mask`, or :meth:`PrefillPack.block_mask`
            to gate the pass with FlexAttention instead of cuDNN.
        memory_token_indexes: from :attr:`PrefillPack.memory_token_indexes`.
        ctx: the mesh. Under a live context axis ``mask`` must have been built
            with this rank's ``query_range``; the keys and values still come back
            whole, so the cache is the same on every rank.
        kv_heads: key/value heads this rank's slice of the model has, from
            :func:`~cosmos3_nano_sim_auto.impl.distributed.tensor_parallel.local_kv_heads`. Defaults to
            the whole count, which is right when the tensor axis is inert. It
            sizes the empty memory the pass starts from, and getting it wrong is
            a shape error inside the first layer rather than a wrong answer.

    Returns:
        ``(text_kv, memory_kv)``, one entry per layer, ready for
        :func:`~cosmos3_nano_sim_auto.impl.transformer.forward.run_chunk_layers`. The cache holds real
        tokens only, so a :class:`~cosmos3_nano_sim_auto.impl.packing.MemoryLayout` built
        with a padded ``capacity`` will not line up -- which the chunk's mask
        shape check catches.
    """
    _text_out, text_kv = run_text_layers(
        layers,
        llm,
        text_h,
        text_position_ids,
        view_text_tokens=view_text_tokens,
        native_gqa_attention=native_gqa_attention,
    )

    empty = torch.zeros(
        vision_h.shape[0],
        llm.num_key_value_heads if kv_heads is None else kv_heads,
        0,
        llm.head_dim,
        device=vision_h.device,
        dtype=vision_h.dtype,
    )
    _vision_out, gen_kv = run_chunk_layers(
        layers,
        llm,
        vision_h,
        vision_position_ids,
        text_kv=text_kv,
        memory_kv=[(empty, empty)] * len(layers),
        mask=mask,
        attention_implementation=attention_implementation,
        native_gqa_attention=native_gqa_attention,
        accelerated_rms_norm=accelerated_rms_norm,
        ctx=ctx,
    )

    indexes = memory_token_indexes.to(vision_h.device)
    memory_kv = [
        (k.index_select(2, indexes), v.index_select(2, indexes)) for k, v in gen_kv
    ]
    return text_kv, memory_kv
