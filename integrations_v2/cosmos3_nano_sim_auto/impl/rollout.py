# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Autoregressive generation with bounded key/value history.

Prefill the prompt and controls, denoise each chunk, then replay it cleanly to
populate history. The default is four denoising passes. The final chunk skips
the clean replay because no later chunk consumes its keys. Noise seeds are
stable across chunks and denoising steps.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from math import prod
from typing import Protocol, TypedDict

import torch
from cosmos_interactive._compile_modes import torch_compile_mode_arg
from cosmos_interactive._network import CosmosMoTNetwork
from cosmos_interactive._patchify import unpatchify_latent
from cosmos_interactive._rope import text_mrope_ids
from cosmos_interactive._sampler import fixed_step_denoise
from torch import Tensor

from cosmos3_nano_sim_auto.impl.conditioning.control_provider import (
    ControlProvider,
    EncodedControls,
)
from cosmos3_nano_sim_auto.impl.distributed.tensor_parallel import local_kv_heads
from cosmos3_nano_sim_auto.impl.packing import (
    ChunkPassKind,
    ClipGeometry,
    MemoryLayout,
    ar_chunk_plan,
    ar_chunk_range,
    build_chunk_metadata,
    build_memory_layout,
    control_window,
)
from cosmos3_nano_sim_auto.impl.prefill import (
    build_prefill_pack,
    pad_control_slots,
    run_control_topup,
    run_prefill,
)
from cosmos3_nano_sim_auto.impl.transformer.attention import (
    ChunkMask,
    attention_implementation,
)
from cosmos3_nano_sim_auto.impl.transformer.embedding import (
    flow_prediction,
    text_embeddings,
    vision_embeddings,
)
from cosmos3_nano_sim_auto.impl.transformer.forward import LayerKV, run_chunk_layers
from cosmos3_nano_sim_auto.impl.transformer.mask import AttentionScope
from cosmos3_nano_sim_auto.impl.transformer.rope import (
    caption_mrope_ids,
    chunk_mrope_ids,
    vision_temporal_offset,
)
from cosmos3_nano_sim_auto.inputs import control_tokens as rgb_tokens
from flashdreams.accelerated.multi_head_attention.kv_arena import (
    KVSegmentLayout,
    allocate_kv_arena,
)
from flashdreams.accelerated.multi_head_attention.optimized import (
    OptimizedImplConfig,
    SDPABackend,
)
from flashdreams.core.distributed.context_parallel import (
    build_shard,
    gather_tokens,
    local_query_range,
)
from flashdreams.core.distributed.parallel import ParallelContext

SDE_CHUNK_STRIDE = 1_000_003
"""the model's per-chunk seed stride for the re-noise between denoising steps."""

SDE_STEP_STRIDE = 9_176
"""the model's per-step stride. Coprime with the chunk stride, so no two pairs collide."""

COMPILE_DENOISE_MODES = (
    "default",
    "max-autotune",
    "max-autotune-no-cudagraphs",
)
"""Compilation modes safe for every Hyperion caller.

``reduce-overhead`` is intentionally absent: two real Blackwell runs reached an
illegal memory access when nested CUDA graphs wrapped FlexAttention, including
one with explicit graph iteration boundaries.
"""


@dataclass(frozen=True)
class ChunkTrace:
    """What one autoregressive step did, for tests and for progress reporting.

    Attributes:
        start, end: the frames the chunk covered, ``[start, end)``.
        denoising_passes: network passes spent denoising, one per sampler step.
        clean_pass: whether the chunk was replayed cleanly to fill the cache.
            False on the last chunk, so total passes are ``denoising_passes``
            plus one when this is set.
        memory_tokens: cached tokens the chunk was denoised against, before its
            own were added.
    """

    start: int
    end: int
    denoising_passes: int
    clean_pass: bool
    memory_tokens: int


class _ChunkMaskArgs(TypedDict):
    """Shared mask arguments for the noisy and clean passes of one chunk."""

    chunk_start: int
    chunk_frames: int
    device: torch.device
    attention_impl: OptimizedImplConfig
    scope: AttentionScope
    maskless: bool
    deduplicate_cross_view: bool
    decomposed_temporal_window_seconds: float | None
    query_range: tuple[int, int] | None


class _KVCache:
    """One branch's per-layer key/value cache, allocated once and never grown.

    Allocated outright because appending with ``torch.cat`` held every layer's
    old copy beside its new one, roughly doubling the whole cache.

    The prefill's tokens are a fixed prefix. Generated chunks go after it into
    ``slots`` slots, each holding one chunk, reused oldest-first once full. That
    is what keeps the cache from growing with the length of the run. Given
    enough slots for the whole clip, nothing is reused and the contents are
    identical to an unbounded cache.

    :meth:`layers` hands out views rather than copies, so a reader sees the
    tokens written so far and none of the reserve.
    """

    def __init__(
        self,
        prefilled: Sequence[LayerKV],
        *,
        capacity: int,
        slots: int,
        slot_tokens: int = 0,
        control_slots: int = 0,
        control_slot_tokens: int = 0,
        control_pinned_slots: int = 0,
        text_prefilled: Sequence[LayerKV] | None = None,
        attention_current_tokens: int = 0,
    ) -> None:
        filled = int(prefilled[0][0].shape[2])
        if capacity < filled:
            raise ValueError(
                f"the prefill wrote {filled} tokens, past a capacity of {capacity}."
            )
        if slots < 0:
            raise ValueError(f"a cache cannot have {slots} chunk slots.")
        if slots and slot_tokens <= 0:
            raise ValueError(
                f"{slots} chunk slots need a slot size; got {slot_tokens} tokens."
            )
        # Checked here rather than on the write that would overflow, so a sizing
        # mistake is a refusal at build time and not a raise chunks later.
        if filled + slots * slot_tokens > capacity:
            raise ValueError(
                f"{slots} slots of {slot_tokens} tokens after the prefill's "
                f"{filled} are past a capacity of {capacity}."
            )
        if control_slots and not 0 <= control_pinned_slots < control_slots:
            raise ValueError(
                f"{control_pinned_slots} pinned of {control_slots} control slots "
                "leaves none to rotate."
            )
        if control_slots and control_slots * control_slot_tokens > filled:
            raise ValueError(
                f"{control_slots} control slots of {control_slot_tokens} tokens do not "
                f"fit the {filled} the prefill wrote."
            )
        self._capacity = capacity
        self._length = filled
        self._base = filled
        self._slots = slots
        self._slot_tokens = slot_tokens
        self._next_slot = 0
        # Control sits at the front, where the prefill put it, and rotates in
        # place. Zero slots means the whole track was front-loaded and none of
        # it is ever replaced. The pinned slots lead and never rotate: they hold
        # the conditioning frames' map, which stays as long as their RGB does.
        self._control_slots = control_slots
        self._control_slot_tokens = control_slot_tokens
        self._control_pinned_slots = control_pinned_slots
        self._next_control_slot = control_pinned_slots
        self._k: list[Tensor] = []
        self._v: list[Tensor] = []
        self._arena_layout: KVSegmentLayout | None = None
        if text_prefilled is not None:
            if len(text_prefilled) != len(prefilled):
                raise ValueError(
                    f"{len(text_prefilled)} text layers for {len(prefilled)} memory layers"
                )
            if attention_current_tokens <= 0:
                raise ValueError("an attention arena needs a positive current capacity")
            for text, memory in zip(text_prefilled, prefilled, strict=True):
                arena, layout = allocate_kv_arena(
                    text,
                    memory,
                    current_tokens=attention_current_tokens,
                    memory_tokens=capacity,
                )
                if self._arena_layout is not None and layout != self._arena_layout:
                    raise ValueError("all attention arena layers must share one layout")
                self._arena_layout = layout
                self._k.append(arena[0])
                self._v.append(arena[1])
        else:
            for k, v in prefilled:
                for source, buffers in ((k, self._k), (v, self._v)):
                    buffer = source.new_zeros(
                        (source.shape[0], source.shape[1], capacity, source.shape[3])
                    )
                    buffer[:, :, :filled] = source
                    buffers.append(buffer)

    @property
    def _memory_start(self) -> int:
        return 0 if self._arena_layout is None else self._arena_layout.memory_start

    @property
    def length(self) -> int:
        """Tokens written so far."""
        return self._length

    def layers(self) -> list[LayerKV]:
        """Per-layer ``(k, v)`` views over the written tokens."""
        return [
            (
                k[:, :, self._memory_start : self._memory_start + self._length],
                v[:, :, self._memory_start : self._memory_start + self._length],
            )
            for k, v in zip(self._k, self._v, strict=True)
        ]

    def text_layers(self) -> list[LayerKV]:
        """Text-prefix views owned by an attention arena."""
        if self._arena_layout is None:
            raise ValueError("this cache has no attention arena")
        end = self._arena_layout.static_tokens
        return [
            (k[:, :, :end], v[:, :, :end])
            for k, v in zip(self._k, self._v, strict=True)
        ]

    @property
    def attention_layout(self) -> KVSegmentLayout | None:
        """Shared arena layout, if this cache owns one."""
        return self._arena_layout

    def attention_layers(self, current_tokens: int) -> list[LayerKV] | None:
        """Full contiguous arenas only after both transient axes have settled."""
        layout = self._arena_layout
        if (
            layout is None
            or current_tokens != layout.current_tokens
            or self._length != self._capacity
        ):
            return None
        return list(zip(self._k, self._v, strict=True))

    def replace_control(self, control: Sequence[LayerKV]) -> int:
        """Overwrite the oldest unpinned control slot with newly computed control keys.

        Control frames past the prefill's window arrive as the rollout reaches
        them, and the oldest go once the window is full. The pinned slots, the
        conditioning frames' map, are never overwritten. A short final group
        leaves the tail of its slot holding keys nothing should read, which the
        layout describes as padding.

        Returns:
            The control slot written, so the caller can say which frames are
            where.
        """
        if not self._control_slots:
            raise ValueError(
                "this cache front-loaded the whole control track, so there is "
                "nothing to top up."
            )
        count = int(control[0][0].shape[2])
        if count > self._control_slot_tokens:
            raise ValueError(
                f"{count} control tokens for a {self._control_slot_tokens}-token slot."
            )
        slot = self._next_control_slot
        self._write(control, slot * self._control_slot_tokens)
        pinned = self._control_pinned_slots
        self._next_control_slot = pinned + (slot + 1 - pinned) % (
            self._control_slots - pinned
        )
        return slot

    def extend(self, chunk: Sequence[LayerKV]) -> int:
        """Write a finished chunk's keys and values into a reusable slot.

        Chunks land end to end while there is room, matching the order
        :func:`~cosmos3_nano_sim_auto.impl.packing.build_memory_layout` lays the cache out
        in. Once the slots are full the oldest is overwritten rather than the
        cache grown, which is what keeps its size fixed.

        A chunk shorter than its slot leaves the tail holding keys nothing
        should read, which the layout describes as padding. Text2video's frame
        0 is one such chunk, being its own causal step.

        Returns:
            The slot the chunk went into. The caller needs it to tell the layout
            which frames now sit where.
        """
        if not self._slots:
            raise ValueError(
                "this cache reserved no reusable slots, so it can hold no "
                "generated chunk."
            )
        count = int(chunk[0][0].shape[2])
        if count > self._slot_tokens:
            raise ValueError(f"{count} tokens for a {self._slot_tokens}-token slot.")

        slot = self._next_slot
        self._write(chunk, self._base + slot * self._slot_tokens)
        # The whole slot counts as written even when the chunk is short, because
        # the layout describes the tail as padding and the mask isolates it.
        self._length = max(self._length, self._base + (slot + 1) * self._slot_tokens)
        self._next_slot = (slot + 1) % self._slots
        return slot

    def _write(self, chunk: Sequence[LayerKV], start: int) -> int:
        """Copy one chunk's keys and values in at ``start``, returning its end."""
        if len(chunk) != len(self._k):
            raise ValueError(
                f"{len(chunk)} layers of chunk for {len(self._k)} of memory."
            )
        end = start + int(chunk[0][0].shape[2])
        if end > self._capacity:
            raise ValueError(
                f"a chunk ending at {end} is past the {self._capacity} this "
                "cache was built for."
            )
        physical_start = self._memory_start + start
        physical_end = self._memory_start + end
        for index, (ck, cv) in enumerate(chunk):
            self._k[index][:, :, physical_start:physical_end] = ck
            self._v[index][:, :, physical_start:physical_end] = cv
        return end


class _TokenWindow:
    """Generated tokens for the most recent frames, keyed by absolute frame.

    An output buffer on the way to the decoder. Generation never reads it back,
    so how many frames it keeps cannot change the picture; it can only make an
    old frame unreadable.

    Pass ``frames`` the whole clip's length to store every frame at its own
    index, which keeps :meth:`~ChunkRollout.result` available. Pass fewer for a
    run of unknown length, and old frames are dropped as new ones arrive.
    """

    def __init__(
        self,
        *,
        num_views: int,
        frames: int,
        spatial: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        if frames < 1:
            raise ValueError(f"a window must hold at least one frame, got {frames}.")
        self._frames = frames
        # [V, T, S, D]: the packer's order, with the frame axis exposed so that
        # a range of frames is an index into one axis.
        self._store = torch.zeros(
            num_views, frames, spatial, width, device=device, dtype=dtype
        )
        self._end = 0

    @property
    def capacity(self) -> int:
        """Frames held at once."""
        return self._frames

    @property
    def start(self) -> int:
        """The oldest frame still readable."""
        return max(0, self._end - self._frames)

    @property
    def end(self) -> int:
        """One past the newest frame written."""
        return self._end

    def append(self, start: int, end: int, tokens: Tensor) -> None:
        """Write frames ``[start, end)``, which must be the ones that come next.

        Raises:
            ValueError: the range does not carry on from what is already
                written, or is longer than the window holds.
        """
        if start != self._end:
            raise ValueError(
                f"frames [{start}, {end}) do not carry on from {self._end}; a "
                "window is written in order."
            )
        if end - start > self._frames:
            raise ValueError(
                f"{end - start} frames at once do not fit a {self._frames}-frame window."
            )
        self._store[:, self._ring(start, end)] = tokens
        self._end = end

    def read(self, start: int, end: int) -> Tensor:
        """``[V, t, S, D]`` for frames ``[start, end)``.

        Raises:
            ValueError: the range is empty, not written yet, or has left the
                window. Refused rather than served from the reserve, which
                would decode to whatever those frames held two wraps ago.
        """
        if start >= end:
            raise ValueError(f"[{start}, {end}) is not a frame range.")
        if end > self._end:
            raise ValueError(
                f"frames up to {self._end} are written; [{start}, {end}) reaches past that."
            )
        if start < self.start:
            raise ValueError(
                f"[{start}, {end}) reaches back past frame {self.start}, which is "
                f"as far as a {self._frames}-frame window goes."
            )
        return self._store[:, self._ring(start, end)]

    def _ring(self, start: int, end: int) -> Tensor:
        """Where frames ``[start, end)`` sit in the store.

        Modular, so the buffer wraps rather than shuffling its contents along.
        It is the identity when the buffer holds the whole clip.
        """
        frames = torch.arange(start, end, device=self._store.device)
        return frames % self._frames


@dataclass(frozen=True)
class Rollout:
    """A finished rollout.

    Attributes:
        latent: ``[V, C, T, H, W]`` per camera, in the sample's rig order, ready
            for the decoder. Mean/std-normalized, as the encoder left it.
        tokens: the same thing before unpatchifying, ``[V*T*S, patch_latent_dim]``
            in the packer's view-outer, frame-inner order.
        trace: one entry per autoregressive step.
    """

    latent: Tensor
    tokens: Tensor
    trace: tuple[ChunkTrace, ...]


class ChunkRollout:
    """A rollout driven one autoregressive chunk at a time.

    :func:`run_rollout` generates a whole clip in one call, which suits a probe.
    A serving loop wants the opposite: ask for the next chunk, hand back frames,
    ask again. So the loop body is :meth:`step` here, and ``run_rollout`` wraps
    it.

    Most of the cost is in ``__init__``, which prefills the prompt and as much
    of the control track as the cache holds. With ``history_frames`` set, the
    rest arrives a chunk at a time during :meth:`step`.

    Stateful and not re-entrant. One instance generates one clip, and
    :meth:`step` past the end raises.
    """

    def __init__(
        self,
        network: CosmosMoTNetwork,
        *,
        geometry: ClipGeometry,
        control_tokens: Tensor | ControlProvider,
        text_ids: Tensor,
        view_text_tokens: Sequence[int] | None = None,
        condition_tokens: Tensor | None = None,
        fps: float = 30.0,
        seed: int = 0,
        t_list: Sequence[float] | None = None,
        native_gqa_attention: bool = True,
        flex_attention: bool = False,
        attention_impl: OptimizedImplConfig | None = None,
        accelerated_rms_norm: bool = False,
        attention_kv_arena: bool = False,
        history_frames: int | None = None,
        token_frames: int | None = None,
        compile_denoise: str | None = None,
        attention_scope: AttentionScope = "all_views",
        maskless_attention: bool = False,
        fused_attention_merge: bool = False,
        decomposed_temporal_window_seconds: float | None = None,
        parallel: ParallelContext | None = None,
    ) -> None:
        """Validate the inputs and prefill the control track.

        Args:
            parallel: the mesh this rollout runs on, from
                :func:`~flashdreams.core.distributed.parallel.init_parallel`. ``None`` is one
                rank and no collectives. The network must already have been
                through :func:`~cosmos3_nano_sim_auto.impl.distributed.tensor_parallel.shard_network` for
                the same context, since the cache is sized from what the model
                actually holds.

            token_frames: How many frames of finished tokens to keep for the
                decoder. ``None`` keeps the whole clip, which :meth:`result`
                needs; fewer suits a run of unknown length and makes
                :meth:`result` unavailable. Always raised to at least the
                conditioning frames plus one chunk, which is the least a single
                decode can use. Generation never reads these back, so this
                cannot change the picture.

            Everything else: see :func:`run_rollout`, which takes the same set.

        Raises:
            ValueError: a token shape disagrees with ``geometry``, or the
                geometry leaves nothing to generate.
        """
        if fused_attention_merge and not maskless_attention:
            raise ValueError("fused_attention_merge requires maskless_attention")
        vfm = network.config.spec.vfm
        width = vfm.patch_latent_dim
        geometry.latent_hw(vfm.latent_patch_size)
        # A whole encoded track is the common case and reads better at the call
        # site than wrapping it, so it is accepted and wrapped here. The rollout
        # itself only ever asks a provider for a range.
        if isinstance(control_tokens, Tensor):
            if geometry.frames_per_view is None:
                raise ValueError(
                    "an encoded control track is as long as it is, so it cannot "
                    "feed a clip with no end. Pass a ControlProvider, which is "
                    "asked for a range at a time."
                )
            _check_tokens(
                control_tokens,
                geometry.frames_per_view,
                geometry,
                width,
                "control_tokens",
            )
            controls: ControlProvider = EncodedControls(
                control_tokens.to(network.embed_tokens.weight.device), geometry
            )
        else:
            controls = control_tokens
        if geometry.condition_frames:
            if condition_tokens is None:
                raise ValueError(
                    f"the geometry has {geometry.condition_frames} conditioning frames, "
                    "so condition_tokens is required."
                )
            _check_tokens(
                condition_tokens,
                geometry.condition_frames,
                geometry,
                width,
                "condition_tokens",
            )
        elif condition_tokens is not None:
            raise ValueError(
                "the geometry has no conditioning frames, so condition_tokens is unused."
            )

        plan = ar_chunk_plan(geometry)
        # Every latent frame already supplied leaves nothing to generate, which
        # one pixel frame of image2video reaches. Refused here, where the
        # geometry is known, rather than as an empty schedule a caller finds by
        # stepping into it.
        if plan.count == 0:
            raise ValueError(
                f"all {geometry.frames_per_view} latent frame(s) are conditioning "
                "frames, so there is nothing to generate. Ask for more frames, or "
                "fewer conditioning frames."
            )

        if history_frames is not None and history_frames <= 0:
            raise ValueError(
                f"history_frames must be positive or None, got {history_frames}"
            )

        # Everything an endless run needs is something it cannot have from the
        # clip's length, so the two unbounded things have to be bounded by hand.
        if geometry.frames_per_view is None:
            if history_frames is None:
                raise ValueError(
                    "a clip with no end needs history_frames: a cache that keeps "
                    "everything grows until the GPU is full."
                )
            if token_frames is None:
                raise ValueError(
                    "a clip with no end needs token_frames: finished tokens kept "
                    "for the whole clip are a buffer with no size."
                )

        if compile_denoise is not None and compile_denoise not in COMPILE_DENOISE_MODES:
            raise ValueError(
                "compile_denoise must be one of "
                f"{COMPILE_DENOISE_MODES} or None, got {compile_denoise!r}"
            )

        # Refused rather than allowed to be slow: the key stream would be a new
        # shape every chunk, so every chunk would recompile.
        if compile_denoise is not None and history_frames is None:
            raise ValueError(
                "compiling the denoise step needs history_frames to bound the "
                "cache; an unbounded key stream recompiles every chunk."
            )
        if attention_kv_arena and history_frames is None:
            raise ValueError(
                "an attention K/V arena needs history_frames to reach a reusable "
                "full-cache shape"
            )
        if attention_kv_arena and compile_denoise is not None:
            raise ValueError(
                "attention K/V arena cannot be combined with compiled denoising; "
                "the combination failed multi-seed quality qualification"
            )
        # RowParallelLinear.forward all-reduces in place, and a mutating
        # collective inside a compiled region is not something Inductor can be
        # trusted to keep. Refused rather than left to produce a wrong answer or
        # a graph break nobody looks at; the functional collectives are the fix,
        # and until they are in, these two are exclusive.
        if (
            compile_denoise is not None
            and parallel is not None
            and parallel.tensor_parallel
        ):
            raise ValueError(
                "compile_denoise and tensor parallelism are not compatible: the "
                "tensor axis all-reduces inside the compiled region. Set "
                "COSMOS3_NANO_SIM_AUTO_TENSOR_PARALLEL=1, or run without compilation."
            )

        self._network = network
        self._vfm = vfm
        self._llm = network.config.spec.llm
        self._device = network.embed_tokens.weight.device
        self._ctx = parallel or ParallelContext.single(self._device)
        self._kv_heads = local_kv_heads(network, self._ctx)
        self._geometry = geometry
        self._width = width
        self._spatial = geometry.spatial_tokens
        self._schedule = list(t_list if t_list is not None else vfm.fixed_step_t_list)
        self._fps = fps
        self._seed = seed
        self._native_gqa_attention = native_gqa_attention
        self._attention_impl = attention_impl or attention_implementation(
            flex=flex_attention
        )
        self._accelerated_rms_norm = accelerated_rms_norm
        self._deduplicate_cross_view = hasattr(network, "rig_view_embed")
        if self._deduplicate_cross_view and geometry.rig_view_ids is None:
            raise ValueError(
                "The final Sim-Auto checkpoint requires canonical camera identities"
            )
        self._scope = attention_scope
        self._maskless = maskless_attention
        self._fused_merge = fused_attention_merge
        self._window = decomposed_temporal_window_seconds
        # Built here rather than per chunk: compiling is only worth its cost if
        # one compilation serves every chunk.
        self._flow = _flow_fn(
            network,
            native_gqa_attention=native_gqa_attention,
            attention_impl=self._attention_impl,
            accelerated_rms_norm=accelerated_rms_norm,
            compile_denoise=compile_denoise,
            rig_view_ids=geometry.rig_view_ids,
            ctx=self._ctx,
        )

        committed = plan.committed_frames
        if history_frames is None:
            kept = committed
        else:
            kept = (
                history_frames if committed is None else min(history_frames, committed)
            )
        assert kept is not None
        self._slots = -(-kept // geometry.frames_per_chunk)

        chunk = geometry.frames_per_chunk
        per_frame = geometry.num_views * self._spatial
        # Control has to reach a chunk further than history does: a chunk reads
        # the map for its own frames, which are not history yet. Its slots
        # follow the same partition the rollout generates in, so a slot holds
        # exactly what one chunk reads. If that covers the clip there is nothing
        # to stream and the whole track is front-loaded, which is the unbounded
        # case and the identical one.
        window = (
            None if history_frames is None else control_window(geometry, self._slots)
        )
        streaming = window is not None and (
            geometry.frames_per_view is None or window[-1][1] < geometry.frames_per_view
        )
        self._control_slots = len(window) if streaming and window else 0
        self._control_ranges: list[tuple[int, int]] | None = (
            list(window) if streaming and window is not None else None
        )
        self._control_frames = (
            self._control_ranges[-1][1]
            if self._control_ranges
            else geometry.frames_per_view
        )

        assert self._control_frames is not None

        # A short opening range still occupies a whole slot, so the control
        # region is sized by its slots rather than by the frames it holds.
        control_capacity = per_frame * (
            self._control_slots * chunk if streaming else self._control_frames
        )
        capacity = control_capacity + per_frame * (
            geometry.condition_frames + self._slots * chunk
        )

        # Slot -> the frames it holds, in the order the cache holds them, which
        # is the order `build_memory_layout` is given. Chronological until the
        # slots are full and the oldest starts being reused.
        self._slot_frames: list[tuple[int, int] | None] = [None] * self._slots
        self._control_next = self._control_frames
        self._controls = controls

        # Parameters still require grad after .eval(), so prefilling with
        # autograd on keeps every layer's activations over the whole control
        # track: 81 GiB at eleven views, against 276 available. Held here rather
        # than left for a caller to remember.
        with torch.no_grad():
            text_kv, memory_kv, num_und, offset = _prefill(
                network,
                geometry=geometry,
                controls=controls,
                condition_tokens=condition_tokens,
                text_ids=text_ids,
                view_text_tokens=view_text_tokens,
                fps=fps,
                native_gqa_attention=native_gqa_attention,
                attention_impl=self._attention_impl,
                accelerated_rms_norm=accelerated_rms_norm,
                control_ranges=self._control_ranges,
                control_slot_frames=chunk if streaming else None,
                scope=attention_scope,
                maskless=maskless_attention,
                deduplicate_cross_view=self._deduplicate_cross_view,
                fused_merge=fused_attention_merge,
                decomposed_temporal_window_seconds=decomposed_temporal_window_seconds,
                ctx=self._ctx,
                kv_heads=self._kv_heads,
            )
        self._num_und = num_und
        self._view_und = None if view_text_tokens is None else tuple(view_text_tokens)
        # Where vision begins on the temporal axis, past the prompt. Taken from
        # the prefill rather than recomputed, since per-camera captions share
        # their positions and so occupy fewer than they have tokens.
        self._offset = offset
        self._cache = _KVCache(
            memory_kv,
            capacity=capacity,
            slots=self._slots,
            slot_tokens=chunk * per_frame,
            control_slots=self._control_slots,
            control_slot_tokens=chunk * per_frame,
            # The conditioning frames' map is kept as long as their RGB sink is:
            # cameras and controls are retained and evicted together.
            control_pinned_slots=sum(
                1
                for _start, end in self._control_ranges or ()
                if end <= geometry.condition_frames
            ),
            text_prefilled=text_kv if attention_kv_arena else None,
            attention_current_tokens=chunk * per_frame if attention_kv_arena else 0,
        )
        self._text_kv = self._cache.text_layers() if attention_kv_arena else text_kv

        # The last thing in the rollout still sized by the run's length. Kept
        # that way by default, because `result` can then hand back the whole
        # clip; a window instead is what a run of unknown length asks for.
        token_capacity = (
            geometry.frames_per_view
            if token_frames is None
            else max(token_frames, geometry.condition_frames + chunk)
        )
        assert token_capacity is not None
        self._token_frames = token_capacity
        self._tokens = _TokenWindow(
            num_views=geometry.num_views,
            frames=self._token_frames,
            spatial=self._spatial,
            width=width,
            device=self._device,
            dtype=network.config.dtype,
        )
        if condition_tokens is not None:
            self._tokens.append(
                0,
                geometry.condition_frames,
                condition_tokens.to(
                    device=self._device, dtype=network.config.dtype
                ).reshape(
                    geometry.num_views, geometry.condition_frames, self._spatial, width
                ),
            )

        self._chunk_count = plan.count
        self._next = 0
        self._frame = geometry.condition_frames
        self._stopped = False
        self._keeps_trace = (
            geometry.frames_per_view is not None
            and self._token_frames >= geometry.frames_per_view
        )
        self._trace: list[ChunkTrace] = []

    @property
    def _history(self) -> list[tuple[int, int]]:
        """Cached history in the order the cache holds it, empty slots dropped."""
        return [frames for frames in self._slot_frames if frames is not None]

    def _memory(self) -> MemoryLayout:
        """What the cache holds right now, in the form the mask builders read.

        Derived twice a chunk rather than stored: the control pass needs the
        cache as it stands before it writes, and the chunk after needs the slot
        that write landed in.
        """
        return build_memory_layout(
            self._geometry,
            control_frame_ranges=self._control_ranges,
            control_slot_frames=(
                self._geometry.frames_per_chunk if self._control_ranges else None
            ),
            history_frame_ranges=self._history,
            history_slot_frames=self._geometry.frames_per_chunk,
            device=self._device,
        )

    def _clean_embeddings(self, patches: Tensor) -> Tensor:
        """Embed patches with no timestep, for a pass that writes to the cache.

        Later chunks attend to those keys as settled history, so neither the
        control pass nor the clean replay may leave a noise level in them.
        """
        return vision_embeddings(
            self._network,
            patches,
            view_ids=_rig_token_ids(
                self._geometry.rig_view_ids, patches.shape[0], self._device
            ),
            is_noisy=torch.zeros(
                patches.shape[0], dtype=torch.bool, device=self._device
            ),
        )

    @property
    def geometry(self) -> ClipGeometry:
        """What this rollout was built for."""
        return self._geometry

    @property
    def chunk_count(self) -> int | None:
        """Chunks the whole clip takes, known before any of them run."""
        return self._chunk_count

    @property
    def chunks_done(self) -> int:
        """Chunks generated so far."""
        return self._next

    @property
    def warmup_chunks(self) -> int:
        """Chunks generated while the cache is still filling.

        Each of these attends to fewer cached keys than the one after it, so
        they run faster than the rest and a frame rate averaged over them
        flatters a long run. Reporting the count lets a caller leave them out.

        One per cache slot, since a chunk fills one slot and the cost stops
        climbing when the last is full. A cache that was never bounded keeps
        growing and has no settled rate, so it reports every chunk.
        """
        return self._slots

    @property
    def next_range(self) -> tuple[int, int] | None:
        """The frames the next :meth:`step` will generate, or ``None`` at the end.

        Worked out from where the last chunk left off rather than read from a
        list, so that nothing here is sized by how long the run turns out to be.
        """
        if self._stopped:
            return None
        return ar_chunk_range(self._geometry, self._frame)

    @property
    def is_finished(self) -> bool:
        """Whether anything is left to generate.

        Two ways to be finished: the clip ran out, or a driver called
        :meth:`stop`. A run with no end of its own has only the second.
        """
        return self.next_range is None

    def stop(self) -> None:
        """Finish after the chunk just generated, whatever the clip's length says.

        How a caller ends a run with no fixed length. When control arrives a
        chunk at a time from outside, whoever supplies it is the only thing that
        knows when there is no more. Frames already generated stay readable:
        this refuses further chunks, it does not tear anything down.
        """
        self._stopped = True

    def _velocity_fn(
        self, positions: Tensor, noisy_mask: ChunkMask
    ) -> Callable[[Tensor, Tensor], Tensor]:
        """The sampler's callback: one noisy pass, its flow returned unchanged."""
        memory_kv = self._cache.layers()
        attention_arenas = self._cache.attention_layers(int(positions.shape[1]))
        attention_layout = (
            self._cache.attention_layout if attention_arenas is not None else None
        )

        def velocity(latent: Tensor, timestep: Tensor) -> Tensor:
            return self._flow(
                latent,
                timestep,
                positions,
                text_kv=self._text_kv,
                memory_kv=memory_kv,
                mask=noisy_mask,
                attention_arenas=attention_arenas,
                attention_arena_layout=attention_layout,
            )

        return velocity

    def _top_up_control(self, needed_end: int) -> None:
        """Bring the control window up to frame ``needed_end``, evicting as it goes.

        Nothing to do when the whole track was front-loaded, which is the
        unbounded case. Otherwise a chunk's worth of map is computed and written
        over the oldest slot, once per chunk once the window is passed.
        """
        if self._control_ranges is None:
            return

        geometry = self._geometry
        while self._control_next < needed_end:
            # The same partition the window was built on, so a top-up replaces
            # one slot with exactly one chunk's worth of map.
            following = ar_chunk_range(geometry, self._control_next)
            if following is None:
                break
            start, end = following
            # Against the cache as it stands: control reads the prompt, earlier
            # control in its own view and the clean RGB committed before it in
            # scope, so this runs after the previous chunk's commit, and what it
            # must not see is a slot this loop is about to replace.
            memory = self._memory()
            control_kv = run_control_topup(
                self._network.layers,
                self._llm,
                control_h=self._clean_embeddings(self._controls.tokens(start, end)),
                control_position_ids=chunk_mrope_ids(
                    geometry,
                    chunk_start=start,
                    chunk_frames=end - start,
                    fps=self._fps,
                    temporal_offset=self._offset,
                ),
                text_kv=self._text_kv,
                memory_kv=self._cache.layers(),
                mask=_chunk_mask(
                    geometry,
                    memory,
                    text_tokens=self._num_und,
                    view_text_tokens=self._view_und,
                    pass_kind="control",
                    chunk_start=start,
                    chunk_frames=end - start,
                    device=self._device,
                    attention_impl=self._attention_impl,
                    scope=self._scope,
                    maskless=self._maskless,
                    deduplicate_cross_view=self._deduplicate_cross_view,
                    fused_merge=self._fused_merge,
                    decomposed_temporal_window_seconds=self._window,
                    query_range=local_query_range(
                        geometry.num_views * (end - start) * self._spatial, self._ctx
                    ),
                ),
                attention_implementation=self._attention_impl,
                native_gqa_attention=self._native_gqa_attention,
                accelerated_rms_norm=self._accelerated_rms_norm,
                ctx=self._ctx,
            )
            slot = self._cache.replace_control(control_kv)
            self._control_ranges[slot] = (start, end)
            self._control_next = end
        # The chunk after this one reads the map for frames the clip and the
        # car decide, never this chunk's pixels, so a provider that can draw
        # ahead is told to start now and has the whole denoise to do it in. Its
        # cross-rank gather still happens in `tokens`, on this thread, in order.
        # A no-op for a provider that cannot, or that is already drawing it.
        ahead = ar_chunk_range(geometry, self._control_next)
        prefetch = getattr(self._controls, "prefetch", None)
        if ahead is not None and prefetch is not None:
            prefetch(*ahead)

    @torch.no_grad()
    def step(self) -> tuple[ChunkTrace, Tensor]:
        """Generate the next chunk.

        Returns:
            Its trace, and its ``[V, frames, S, D]`` tokens. The tokens are a
            view into this rollout's buffer, not a copy, so decode or clone them
            before the next step if you need them to persist.

        Raises:
            RuntimeError: the clip is already complete, or a driver has called
                :meth:`stop`.
        """
        chunk = self.next_range
        if chunk is None:
            raise RuntimeError(
                f"nothing left to generate after {self._next} of "
                f"{self._chunk_count} chunk(s)."
                if self._stopped
                else f"all {self._chunk_count} chunks are generated; call result()."
            )

        geometry = self._geometry
        start, end = chunk
        frames = end - start
        count = geometry.num_views * frames * self._spatial

        self._top_up_control(end)
        memory = self._memory()
        # One query range for every pass this chunk takes -- the denoising steps
        # and the clean replay -- because they run over the same tokens and their
        # keys have to land in the same places. Derived from the token count
        # alone, so the layers rebuild the identical split without being told.
        query_range = local_query_range(count, self._ctx)
        mask_args: _ChunkMaskArgs = {
            "chunk_start": start,
            "chunk_frames": frames,
            "device": self._device,
            "attention_impl": self._attention_impl,
            "scope": self._scope,
            "maskless": self._maskless,
            "deduplicate_cross_view": self._deduplicate_cross_view,
            "decomposed_temporal_window_seconds": self._window,
            "query_range": query_range,
        }

        positions = chunk_mrope_ids(
            geometry,
            chunk_start=start,
            chunk_frames=frames,
            fps=self._fps,
            temporal_offset=self._offset,
        )
        noisy_mask = _chunk_mask(
            geometry,
            memory,
            text_tokens=self._num_und,
            view_text_tokens=self._view_und,
            pass_kind="noisy",
            fused_merge=self._fused_merge,
            **mask_args,
        )

        # The sampler stays whole on every rank and only the *network* is
        # sharded: the flow comes back one range at a time and is gathered
        # inside the flow function. Two reasons. The sampler's own arithmetic is
        # elementwise over 17,160x128 floats, which is nothing beside a pass over
        # 36 layers, so replicating it is cheaper than another shard to keep in
        # step. And the noise is then drawn from the clip's seeds rather than the
        # mesh's, which is what lets a sharded run be compared against a
        # single-rank one at all.
        velocity = self._velocity_fn(positions, noisy_mask)
        patch = self._vfm.latent_patch_size
        latent_hw = geometry.latent_hw(patch)
        padded = latent_hw != (geometry.patch_h * patch, geometry.patch_w * patch)
        noise_count, noise_width = count, self._width
        if padded:
            # Only real latent pixels participate in noise and scheduler updates.
            shape = (
                self._vfm.vae_latent_channels,
                geometry.num_views,
                frames,
                *latent_hw,
            )
            noise_count, noise_width = prod(shape), 1
            token_velocity = velocity

            def pack(latent: Tensor) -> Tensor:
                return rgb_tokens(
                    latent.reshape(shape).transpose(0, 1), latent_patch_size=patch
                )

            def velocity(latent: Tensor, timestep: Tensor) -> Tensor:
                predicted = token_velocity(pack(latent), timestep)
                return (
                    _to_latent(
                        predicted.reshape(
                            geometry.num_views, frames, self._spatial, self._width
                        ),
                        geometry,
                        channels=shape[0],
                        patch=patch,
                    )
                    .transpose(0, 1)
                    .reshape(-1, 1)
                )

        denoised = fixed_step_denoise(
            velocity,
            _chunk_noise(
                noise_count,
                noise_width,
                seed=self._seed,
                start=start,
                device=self._device,
            ),
            t_list=self._schedule,
            sde_noise_fn=_sde_noise_fn(
                noise_count,
                noise_width,
                seed=self._seed,
                start=start,
                device=self._device,
            ),
            fp32_math=True,
        )
        if padded:
            denoised = pack(denoised)
        self._tokens.append(
            start,
            end,
            denoised.to(self._network.config.dtype).reshape(
                geometry.num_views, frames, self._spatial, self._width
            ),
        )

        # A clip with no end has no last chunk, so there is never one to skip:
        # something always follows to read these keys.
        commit = geometry.frames_per_view is None or end < geometry.frames_per_view
        if commit:
            self._commit(
                denoised,
                positions=positions,
                memory=memory,
                mask_args=mask_args,
                frames=chunk,
            )

        step = ChunkTrace(
            start=start,
            end=end,
            denoising_passes=len(self._schedule),
            clean_pass=commit,
            memory_tokens=memory.real_token_count,
        )
        if self._keeps_trace:
            # Only :meth:`result` reads the accumulated trace, and a windowed
            # rollout refuses to be one. Keeping it there would be a list that
            # grows with the run and nothing ever looks at -- the one shape of
            # bookkeeping this class is built to avoid.
            self._trace.append(step)
        self._next += 1
        self._frame = end
        return step, self._tokens.read(start, end)

    def _commit(
        self,
        denoised: Tensor,
        *,
        positions: Tensor,
        memory: MemoryLayout,
        mask_args: _ChunkMaskArgs,
        frames: tuple[int, int],
    ) -> None:
        """Replay a finished chunk clean, so later chunks read settled keys.

        The denoising passes left a noise level in the keys they wrote, and to
        the chunks that follow these frames are history. So the chunk runs once
        more with no timestep and those keys are kept instead. Skipped on the
        final chunk, where nothing follows to read it.
        """
        _hidden, chunk_kv = run_chunk_layers(
            self._network.layers,
            self._llm,
            self._clean_embeddings(denoised),
            positions,
            text_kv=self._text_kv,
            memory_kv=self._cache.layers(),
            mask=_chunk_mask(
                self._geometry,
                memory,
                text_tokens=self._num_und,
                view_text_tokens=self._view_und,
                pass_kind="clean",
                fused_merge=self._fused_merge,
                **mask_args,
            ),
            attention_implementation=self._attention_impl,
            native_gqa_attention=self._native_gqa_attention,
            accelerated_rms_norm=self._accelerated_rms_norm,
            ctx=self._ctx,
        )
        self._slot_frames[self._cache.extend(chunk_kv)] = frames

    def latent_for(self, start: int, end: int) -> Tensor:
        """``[V, C, t, H, W]`` for frames ``[start, end)``, ready for the decoder.

        The frames have to exist already, whether a :meth:`step` generated them
        or they were supplied. This is how a caller decodes as it goes, and why
        the first call usually starts at 0 rather than at the first chunk: an
        image2video clip's supplied frames are real pixels that belong in the
        output, and no step produced them.

        Raises:
            ValueError: the range is empty, out of bounds, not generated yet, or
                has left the token window.
        """
        last = self._geometry.frames_per_view
        if not 0 <= start < end or (last is not None and end > last):
            raise ValueError(
                f"[{start}, {end}) is not a frame range of a {last}-frame clip."
            )
        return _to_latent(
            self._tokens.read(start, end),
            self._geometry,
            channels=self._vfm.vae_latent_channels,
            patch=self._vfm.latent_patch_size,
        )

    def result(self) -> Rollout:
        """The finished clip.

        Raises:
            RuntimeError: chunks remain. Unpatchifying a partly-zero buffer
                would decode to a clip that ends in grey rather than an error,
                and a driver that stopped early leaves the same hole. Or the
                rollout keeps a window rather than the whole clip, in which case
                the earlier frames are gone and :meth:`latent_for` as the clip
                goes is the only way to see them.
        """
        if self._chunk_count is None:
            raise RuntimeError(
                "this rollout has no end, so it has no finished clip. Decode "
                "with latent_for() as the chunks arrive."
            )
        if self._next < self._chunk_count:
            raise RuntimeError(
                f"{self._chunk_count - self._next} of {self._chunk_count} chunks "
                "are still to generate."
            )
        assert self._geometry.frames_per_view is not None
        if self._token_frames < self._geometry.frames_per_view:
            raise RuntimeError(
                f"this rollout keeps {self._token_frames} of the clip's "
                f"{self._geometry.frames_per_view} frames, so there is no whole clip "
                "to return. Decode with latent_for() as the chunks arrive."
            )
        tokens = self._tokens.read(0, self._geometry.frames_per_view)
        return Rollout(
            latent=_to_latent(
                tokens,
                self._geometry,
                channels=self._vfm.vae_latent_channels,
                patch=self._vfm.latent_patch_size,
            ),
            tokens=tokens.reshape(-1, self._width),
            trace=tuple(self._trace),
        )


def run_rollout(
    network: CosmosMoTNetwork,
    *,
    geometry: ClipGeometry,
    control_tokens: Tensor | ControlProvider,
    text_ids: Tensor,
    view_text_tokens: Sequence[int] | None = None,
    condition_tokens: Tensor | None = None,
    fps: float = 30.0,
    seed: int = 0,
    t_list: Sequence[float] | None = None,
    native_gqa_attention: bool = True,
    flex_attention: bool = False,
    attention_impl: OptimizedImplConfig | None = None,
    accelerated_rms_norm: bool = False,
    attention_kv_arena: bool = False,
    history_frames: int | None = None,
    compile_denoise: str | None = None,
    attention_scope: AttentionScope = "all_views",
    maskless_attention: bool = False,
    fused_attention_merge: bool = False,
    decomposed_temporal_window_seconds: float | None = None,
    parallel: ParallelContext | None = None,
    on_chunk: Callable[[ChunkTrace, Tensor], None] | None = None,
) -> Rollout:
    """Generate the whole clip, in one call.

    A :class:`ChunkRollout` built and drained. Use that directly when something
    else owns the loop, such as a server asking for one chunk at a time.

    Args:
        geometry: From :meth:`~cosmos3_nano_sim_auto.inputs.MultiviewSample.geometry`.
        control_tokens: ``[V*T*S, patch_latent_dim]`` for the whole clip, from
            :func:`~cosmos3_nano_sim_auto.inputs.control_tokens`. Or a
            :class:`~cosmos3_nano_sim_auto.impl.conditioning.control_provider.ControlProvider`, for
            control that is not read from a file.
        text_ids: ``[num_und]``, from
            :func:`~cosmos3_nano_sim_auto.inputs.prompt_token_ids`.
        condition_tokens: The target's opening frames, if the sample has any.
            ``text2video`` has none and this stays None.
        t_list: The distilled sigma schedule. Defaults to the checkpoint's
            ``fixed_step_t_list``, which is taken as written.
        native_gqa_attention: Hand SDPA the model's key/value heads as they
            are, rather than repeating each to fill the query heads first.
        flex_attention: Gate attention with a block mask through compiled
            FlexAttention instead of a dense mask through cuDNN. Same rules
            either way.
        attention_impl: Shared backend and kernel tuning policy. When supplied,
            it supersedes ``flex_attention``.
        accelerated_rms_norm: Use FlashDreams Accelerated's fused RMSNorm for
            generation hidden states and attention queries/keys. False keeps the
            checkpoint modules unchanged.
        attention_kv_arena: Reuse stable contiguous text/current/memory K/V
            storage. False by default in this API. Requires bounded
            ``history_frames`` and cannot be combined with ``compile_denoise``.

        history_frames: How many generated frames the cache keeps, rounded up
            to whole chunks. ``None`` keeps the whole clip, which a run of
            unknown length cannot afford.
        compile_denoise: A ``torch.compile`` mode for the noisy pass, or
            ``None`` to run it eager. Needs ``history_frames`` set, since an
            unbounded key stream is a new shape every chunk.
        fused_attention_merge: Opt into the CUDA vector merge for eager
            maskless attention. Native weights and reduction order are retained;
            unsupported layouts, autograd and compiler tracing stay native.
        parallel: the mesh to run on. ``None`` is one rank. Every rank calls
            this with the same arguments and gets the same clip back, so the
            caller decides which of them writes it.
        on_chunk: Called with each chunk's trace and its ``[V, frames, S, D]``
            tokens as it completes, for streaming the decode.

    Returns:
        A :class:`Rollout` covering the whole clip, opening frames copied
        through unchanged.
    """
    rollout = ChunkRollout(
        network,
        geometry=geometry,
        control_tokens=control_tokens,
        text_ids=text_ids,
        view_text_tokens=view_text_tokens,
        condition_tokens=condition_tokens,
        fps=fps,
        seed=seed,
        t_list=t_list,
        native_gqa_attention=native_gqa_attention,
        flex_attention=flex_attention,
        attention_impl=attention_impl,
        accelerated_rms_norm=accelerated_rms_norm,
        attention_kv_arena=attention_kv_arena,
        history_frames=history_frames,
        compile_denoise=compile_denoise,
        attention_scope=attention_scope,
        maskless_attention=maskless_attention,
        fused_attention_merge=fused_attention_merge,
        decomposed_temporal_window_seconds=decomposed_temporal_window_seconds,
        parallel=parallel,
    )
    while not rollout.is_finished:
        trace, chunk = rollout.step()
        if on_chunk is not None:
            on_chunk(trace, chunk)
    return rollout.result()


class FlowFn(Protocol):
    """One noisy pass, from a noisy latent to the flow the sampler steps along.

    The caches and the mask are keyword-only on purpose. They are all sequences
    of per-layer tensors of the same type, so passing them in the wrong order
    reads fine, type-checks fine, and builds a key stream of exactly the length
    the mask expects -- wrong content, right shape, nothing raised.
    """

    def __call__(
        self,
        latent: Tensor,
        timestep: Tensor,
        positions: Tensor,
        *,
        text_kv: Sequence[LayerKV],
        memory_kv: Sequence[LayerKV],
        mask: ChunkMask,
        attention_arenas: Sequence[LayerKV] | None,
        attention_arena_layout: KVSegmentLayout | None,
    ) -> Tensor: ...


def _rig_token_ids(
    camera_ids: tuple[int, ...] | None, count: int, device: torch.device
) -> Tensor | None:
    """Expand physical camera IDs over a camera-major chunk."""
    if camera_ids is None:
        return None
    if count % len(camera_ids):
        raise ValueError("Camera-major tokens must divide evenly across the rig")
    return torch.tensor(camera_ids, device=device).repeat_interleave(
        count // len(camera_ids)
    )


def _flow_fn(
    network: CosmosMoTNetwork,
    *,
    native_gqa_attention: bool,
    attention_impl: OptimizedImplConfig,
    accelerated_rms_norm: bool,
    compile_denoise: str | None,
    rig_view_ids: tuple[int, ...] | None = None,
    ctx: ParallelContext | None = None,
) -> FlowFn:
    """Embed a noisy latent, run the stack, read the flow off the end.

    Everything that varies between passes is an argument rather than a closure,
    so one compilation serves every chunk. Closing over the chunk would compile
    per chunk instead.

    Compiling also needs the key stream to reach a fixed size, which is what
    ``history_frames`` does. Left unbounded, every chunk is a new shape and
    recompiles.

    **The whole chunk goes in and the whole flow comes out**, however the mesh is
    shaped. The layers run over this rank's token range and the gather at the end
    puts the flow back together, so the sampler on the other side of this
    function never learns that anything was split. That gather is the chunk's
    tokens by ``patch_latent_dim`` in float32 -- 8.8 MB at eleven views, against
    a pass over 36 layers -- and it buys a sampler with no shard to keep in step
    and noise drawn from the clip's seeds rather than the mesh's.

    Args:
        compile_denoise: A ``torch.compile`` mode, or ``None`` to run eager.
        ctx: the mesh. ``None`` or an inert context makes the gather a no-op.
    """
    llm = network.config.spec.llm

    def flow(
        latent: Tensor,
        timestep: Tensor,
        positions: Tensor,
        *,
        text_kv: Sequence[LayerKV],
        memory_kv: Sequence[LayerKV],
        mask: ChunkMask,
        attention_arenas: Sequence[LayerKV] | None,
        attention_arena_layout: KVSegmentLayout | None,
    ) -> Tensor:
        hidden, _kv = run_chunk_layers(
            network.layers,
            llm,
            vision_embeddings(
                network,
                latent,
                timestep=timestep,
                view_ids=_rig_token_ids(rig_view_ids, latent.shape[0], latent.device),
            ),
            positions,
            text_kv=text_kv,
            memory_kv=memory_kv,
            mask=mask,
            final_norm=network.norm_moe_gen,
            attention_implementation=attention_impl,
            native_gqa_attention=native_gqa_attention,
            accelerated_rms_norm=accelerated_rms_norm,
            attention_arenas=attention_arenas,
            attention_arena_layout=attention_arena_layout,
            collect_kv=False,
            ctx=ctx,
        )
        predicted = flow_prediction(network, hidden)
        if ctx is not None and ctx.context_parallel:
            predicted = gather_tokens(
                predicted, build_shard(int(latent.shape[0]), ctx), ctx, dim=1
            )
        return predicted[0]

    if compile_denoise is None:
        return flow
    return torch.compile(flow, mode=torch_compile_mode_arg(compile_denoise))


def _chunk_mask(
    geometry: ClipGeometry,
    memory: MemoryLayout,
    *,
    chunk_start: int,
    chunk_frames: int,
    text_tokens: int,
    view_text_tokens: Sequence[int] | None,
    pass_kind: ChunkPassKind,
    device: torch.device,
    attention_impl: OptimizedImplConfig,
    scope: AttentionScope = "all_views",
    maskless: bool = False,
    deduplicate_cross_view: bool = False,
    fused_merge: bool = False,
    query_range: tuple[int, int] | None = None,
    decomposed_temporal_window_seconds: float | None = None,
) -> ChunkMask:
    """The mask for one chunk pass, in the form the chosen kernel takes.

    ``query_range`` is this rank's share of the queries; the key axis stays
    whole, so a sharded mask is narrower and no shorter.
    """
    metadata = build_chunk_metadata(
        geometry,
        memory,
        chunk_start=chunk_start,
        chunk_frames=chunk_frames,
        text_tokens=text_tokens,
        view_text_tokens=view_text_tokens,
        pass_kind=pass_kind,
        query_range=query_range,
        device=device,
    )
    if maskless:
        return replace(
            metadata.fold_plan(
                scope=scope,
                deduplicate_cross_view=deduplicate_cross_view,
                decomposed_temporal_window_seconds=decomposed_temporal_window_seconds,
            ),
            fused_merge=fused_merge,
        )
    if attention_impl.sdpa_backend is SDPABackend.FLEX:
        return metadata.block_mask(
            scope=scope,
            block_size=attention_impl.flex_attention.mask_block_size,
            decomposed_temporal_window_seconds=decomposed_temporal_window_seconds,
        )
    return metadata.mask(
        scope=scope,
        decomposed_temporal_window_seconds=decomposed_temporal_window_seconds,
    )


def _prefill(
    network: CosmosMoTNetwork,
    *,
    geometry: ClipGeometry,
    controls: ControlProvider,
    condition_tokens: Tensor | None,
    text_ids: Tensor,
    view_text_tokens: Sequence[int] | None,
    fps: float,
    native_gqa_attention: bool,
    attention_impl: OptimizedImplConfig,
    accelerated_rms_norm: bool,
    control_ranges: Sequence[tuple[int, int]] | None = None,
    control_slot_frames: int | None = None,
    scope: AttentionScope = "all_views",
    maskless: bool = False,
    deduplicate_cross_view: bool = False,
    fused_merge: bool = False,
    decomposed_temporal_window_seconds: float | None = None,
    ctx: ParallelContext | None = None,
    kv_heads: int | None = None,
) -> tuple[list[LayerKV], list[LayerKV], int, float]:
    """One clean pass over prompt and control track; returns their caches.

    The vision pack is the control track and then the supplied frames, of which
    ``text2video`` has none -- so its pack is the control track alone.
    :mod:`cosmos3_nano_sim_auto.impl.prefill` has why feeding it anything longer corrupts
    the cache rather than just wasting the compute.

    ``control_ranges`` arrive one to a cache slot and are laid out that way
    throughout -- tokens, positions and the metadata describing them -- because
    the cache is told the same ranges and reorders nothing.

    Returns:
        the prompt cache, the memory cache, the prompt's token count, and where
        vision starts on the temporal axis. The last is returned rather than
        left to the caller because per-camera captions share their positions,
        so the prompt occupies fewer of them than it has tokens.
    """
    device = network.embed_tokens.weight.device
    num_und = int(text_ids.shape[0])

    pack = build_prefill_pack(
        geometry,
        text_tokens=num_und,
        view_text_tokens=view_text_tokens,
        control_ranges=control_ranges,
        device=device,
    )
    if view_text_tokens is None:
        text_positions, _next = text_mrope_ids(num_und, 0.0)
        text_positions_used = num_und
    else:
        text_positions, text_positions_used = caption_mrope_ids(view_text_tokens)
    offset = vision_temporal_offset(text_positions_used)

    # Control and target share their positions exactly, so the target's ids are
    # the clip's, restricted to the frames it covers.
    positions = [
        chunk_mrope_ids(
            geometry,
            chunk_start=start,
            chunk_frames=end - start,
            fps=fps,
            temporal_offset=offset,
        )
        for start, end in pack.control_ranges
    ]
    patches = [controls.tokens(start, end) for start, end in pack.control_ranges]
    if geometry.condition_frames:
        assert condition_tokens is not None
        positions.append(
            chunk_mrope_ids(
                geometry,
                chunk_start=0,
                chunk_frames=geometry.condition_frames,
                fps=fps,
                temporal_offset=offset,
            )
        )
        patches.append(condition_tokens)

    vision = torch.cat(patches).to(device=device)
    ctx = ctx or ParallelContext.single(device)
    query_range = local_query_range(int(vision.shape[0]), ctx)
    text_kv, memory_kv = run_prefill(
        network.layers,
        network.config.spec.llm,
        text_h=text_embeddings(network, text_ids),
        text_position_ids=text_positions,
        view_text_tokens=view_text_tokens,
        # The whole pack is clean: controls always, and the target covers only
        # supplied frames. So no timestep is embedded anywhere in it.
        vision_h=vision_embeddings(
            network,
            vision,
            is_noisy=pack.gen.is_noisy,
            view_ids=(
                torch.tensor(geometry.rig_view_ids, device=device)[pack.gen.view_id]
                if geometry.rig_view_ids is not None
                else None
            ),
        ),
        vision_position_ids=torch.cat(positions, dim=1),
        mask=(
            replace(
                pack.fold_plan(
                    scope=scope,
                    deduplicate_cross_view=deduplicate_cross_view,
                    query_range=query_range,
                    decomposed_temporal_window_seconds=decomposed_temporal_window_seconds,
                ),
                fused_merge=fused_merge,
            )
            if maskless
            else pack.block_mask(
                scope=scope,
                block_size=attention_impl.flex_attention.mask_block_size,
                query_range=query_range,
                decomposed_temporal_window_seconds=decomposed_temporal_window_seconds,
            )
            if attention_impl.sdpa_backend is SDPABackend.FLEX
            else pack.mask(
                scope=scope,
                query_range=query_range,
                decomposed_temporal_window_seconds=decomposed_temporal_window_seconds,
            )
        ),
        memory_token_indexes=pack.memory_token_indexes,
        attention_implementation=attention_impl,
        native_gqa_attention=native_gqa_attention,
        accelerated_rms_norm=accelerated_rms_norm,
        ctx=ctx,
        kv_heads=kv_heads,
    )
    if control_slot_frames is not None:
        memory_kv = pad_control_slots(
            memory_kv,
            control_ranges=pack.control_ranges,
            tokens_per_frame=geometry.num_views * geometry.spatial_tokens,
            slot_frames=control_slot_frames,
        )
    return text_kv, memory_kv, num_und, offset


def _check_tokens(
    tokens: Tensor, frames: int, geometry: ClipGeometry, width: int, name: str
) -> None:
    expected = (geometry.num_views * frames * geometry.spatial_tokens, width)
    if tuple(tokens.shape) != expected:
        raise ValueError(
            f"{name} is {tuple(tokens.shape)}; for {geometry.num_views} views x {frames} frames "
            f"x {geometry.spatial_tokens} spatial tokens it must be {expected}."
        )


def _chunk_noise(
    count: int, width: int, *, seed: int, start: int, device: torch.device
) -> Tensor:
    """The chunk's initial noise, seeded as the reference implementation seeds it."""
    generator = torch.Generator(device=device).manual_seed(seed + start)
    return torch.empty(count, width, device=device, dtype=torch.float32).normal_(
        generator=generator
    )


def _sde_noise_fn(
    count: int, width: int, *, seed: int, start: int, device: torch.device
) -> Callable[[int], Tensor]:
    """The re-noise added between denoising steps, seeded as the reference implementation seeds it."""

    def noise(step_idx: int) -> Tensor:
        step_seed = seed + start * SDE_CHUNK_STRIDE + step_idx * SDE_STEP_STRIDE
        generator = torch.Generator(device=device).manual_seed(step_seed)
        return torch.empty(count, width, device=device, dtype=torch.float32).normal_(
            generator=generator
        )

    return noise


def _to_latent(
    tokens: Tensor, geometry: ClipGeometry, *, channels: int, patch: int
) -> Tensor:
    """``[V, T, S, D]`` tokens -> ``[V, C, T, H, W]`` latents, per camera.

    ``grid_t`` comes from the tokens rather than from ``geometry`` so that a
    slice of the clip converts as readily as the whole of it, which is what
    decoding a chunk at a time needs.
    """
    height, width = geometry.latent_hw(patch)
    return torch.stack(
        [
            unpatchify_latent(
                view.reshape(-1, tokens.shape[-1]),
                channels=channels,
                grid_t=tokens.shape[1],
                grid_h=geometry.patch_h,
                grid_w=geometry.patch_w,
                patch_spatial=patch,
            )[..., :height, :width]
            for view in tokens
        ]
    )
