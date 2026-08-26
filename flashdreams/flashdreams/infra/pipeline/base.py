# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Autoregressive inference pipeline: encode → diffuse → decode."""

from __future__ import annotations

import copy
import os
from dataclasses import dataclass, field
from typing import Any, Generic

import torch
import torch.nn as nn
from loguru import logger
from torch import Tensor

from flashdreams.infra.config import InstantiateConfig
from flashdreams.infra.decoder import (
    DecoderConfig,
    StreamingDecoder,
    StreamingDecoderCacheT,
)
from flashdreams.infra.diffusion.model import (
    DiffusionModel,
    DiffusionModelConfig,
)
from flashdreams.infra.diffusion.transformer import (
    TransformerCacheT,
)
from flashdreams.infra.encoder import (
    EncoderConfig,
    StreamingEncoder,
    StreamingEncoderCacheT,
)
from flashdreams.infra.profiler import EventProfiler

# --- TEMPORARY (step 1 of the pixel-path SAS work). Bounded so a long run cannot
# flood the log; remove along with the probe block in __call__. ---
_PROBE_LIMIT = 6
_probe_calls = 0


def _latent_probe_enabled() -> bool:
    """True for the first few AR steps when FD_LATENT_PROBE is 1 or 2."""
    global _probe_calls
    if os.environ.get("FD_LATENT_PROBE") not in ("1", "2"):
        return False
    _probe_calls += 1
    return _probe_calls <= _PROBE_LIMIT


def _mem_range(t: Any) -> tuple[int, int, int] | None:
    """(storage_start, storage_end, tensor_start) in bytes, or None for a non-tensor.

    ``data_ptr()`` on its own does NOT establish independence: two tensors can share a
    single storage at different offsets, in which case their ``data_ptr()`` values differ
    while a write through one is still visible through the other. Only comparing the
    underlying storage byte ranges answers "can these alias".
    """
    if not isinstance(t, Tensor):
        return None
    storage = t.untyped_storage()
    start = storage.data_ptr()
    return (start, start + storage.nbytes(), t.data_ptr())


def _overlaps(a: tuple[int, int, int] | None, b: tuple[int, int, int] | None) -> Any:
    """True when two storage ranges intersect at all (i.e. the tensors can alias)."""
    if a is None or b is None:
        return None
    return a[0] < b[1] and b[0] < a[1]


_codec_resolved = False
_codec_instance: Any = None


def _resolve_latent_codec() -> Any:
    """Resolve the optional pixel-path latent codec from ``FD_LATENT_CODEC``, once.

    Unset means no seam and a path identical to the original. Resolved lazily so the
    SAS codec's Triton dependency is only imported when actually selected.
    """
    global _codec_resolved, _codec_instance
    if not _codec_resolved:
        from flashdreams.infra.pipeline.latent_codec import get_latent_codec

        _codec_instance = get_latent_codec(os.environ.get("FD_LATENT_CODEC"))
        if _codec_instance is not None:
            logger.info(
                "Latent-codec seam ACTIVE: {} -- the pixel decoder will see "
                "round-tripped latents, not the DiT's originals.",
                _codec_instance.name,
            )
        _codec_resolved = True
    return _codec_instance


def _assert_branches_isolated(dec_a: Any, dec_b: Any, cache: Any) -> None:
    """Verify the two decode branches share no mutable state. Step 8 of the plan.

    Checked in the live pipeline rather than only in a standalone harness, because the
    pipeline's cache lifecycle differs from a synthetic one. Everything here has a
    failure mode that produces plausible-looking video rather than an error, so none of
    it is safe to assume:

      * shared weight storage would mean deepcopy silently aliased
      * shared MemBlock ids would let a cache mix-up go undetected (dec_state is keyed
        by id(module), so distinct ids turn a mix-up into a loud KeyError)
      * shared cache-slot addresses would let one branch's captured graph write through
        the other's history
    """
    sa = {p.untyped_storage().data_ptr() for p in dec_a.parameters()}
    sb = {p.untyped_storage().data_ptr() for p in dec_b.parameters()}
    if sa & sb:
        raise RuntimeError(
            f"dual-decode branches share {len(sa & sb)} weight storage(s); the codec "
            f"branch is not an independent instance."
        )

    def _memblock_ids(mod: Any) -> set[int]:
        return {
            id(m) for m in mod.modules() if type(m).__name__ == "MemBlock"
        }

    ia, ib = _memblock_ids(dec_a), _memblock_ids(dec_b)
    if ia & ib:
        raise RuntimeError(
            f"dual-decode branches share {len(ia & ib)} MemBlock instance(s); their "
            f"caches would collide on id(module) keys."
        )

    ca, cb = cache.decoder_cache, cache.decoder_cache_codec
    if ca is cb:
        raise RuntimeError("dual-decode branches were handed the same cache object.")
    sa_slots = {
        v.untyped_storage().data_ptr() for v in getattr(ca, "dec_state", {}).values()
    }
    sb_slots = {
        v.untyped_storage().data_ptr() for v in getattr(cb, "dec_state", {}).values()
    }
    if sa_slots & sb_slots:
        raise RuntimeError(
            f"dual-decode caches share {len(sa_slots & sb_slots)} slot storage "
            f"address(es); a captured graph would write across branches."
        )


def _log_graph_capture(dec_a: Any, dec_b: Any) -> None:
    """Report whether each branch actually captured a CUDA graph.

    If neither did, an isolation result says nothing about graph safety -- the hazard
    was simply never exercised. Worth stating explicitly rather than inferring a pass.
    """

    def _captured(mod: Any) -> bool:
        for m in mod.modules():
            w = getattr(m, "_decoder_wrapper", None)
            if w is not None and getattr(w, "_graph", None) is not None:
                return True
        return False

    a, b = _captured(dec_a), _captured(dec_b)
    logger.info(
        "[DUAL-DECODE] CUDA graph captured: reference={} codec={}{}",
        a,
        b,
        "" if (a or b) else "  <- graph hazard NOT exercised by this run",
    )


def _assert_latent_disjoint(
    new: Tensor, source: Tensor, final_state: Any, codec_name: str
) -> None:
    """Fail loudly if a codec returned a view rather than independent storage.

    ``source`` is the same tensor object as ``cache.clean_latent``, which is what the
    token stream emits -- so a view here would corrupt the token path and double-apply
    quantization, with no visible error. Storage ranges are compared rather than
    ``data_ptr()``, since two tensors can share one storage at different offsets and
    still alias. Cheap enough to check every step, and the failure it guards against
    would be very hard to spot in the output.
    """
    r_new = _mem_range(new)
    if _overlaps(r_new, _mem_range(source)):
        raise RuntimeError(
            f"latent codec {codec_name!r} returned a tensor sharing storage with the "
            f"DiT output; it must allocate. This would corrupt the token stream, which "
            f"emits the same buffer."
        )
    if _overlaps(r_new, _mem_range(getattr(final_state, "clean_latent", None))):
        raise RuntimeError(
            f"latent codec {codec_name!r} returned a tensor sharing storage with the "
            f"DiT feedback tensor; this would alter the generation trajectory."
        )
    if new.shape != source.shape or new.dtype != source.dtype:
        raise RuntimeError(
            f"latent codec {codec_name!r} changed shape/dtype: "
            f"{tuple(source.shape)}/{source.dtype} -> {tuple(new.shape)}/{new.dtype}"
        )


@dataclass(kw_only=True)
class StreamInferencePipelineConfig(InstantiateConfig):
    """Config for the streaming inference pipeline.

    Set ``encoder=None`` when the pipeline has no per-AR-step control input
    (pure T2V). Set ``decoder=None`` to return the clean latent directly
    (training, latent-space evaluation, or pipelines that own decoding).
    """

    _target: type["StreamInferencePipeline"] = field(
        default_factory=lambda: StreamInferencePipeline
    )

    name: str
    """Stable slug for this pipeline variant; the primary key of
    ``<NAME>_CONFIGS``. Runners mirror it as ``runner_name`` so
    ``flashdreams-run <slug>`` resolves to this pipeline."""

    diffusion_model: DiffusionModelConfig
    """Transformer + scheduler config."""

    decoder: DecoderConfig | None = None
    """Optional output :class:`StreamingDecoder` with a per-rollout cache,
    called as ``decoder(input, autoregressive_index, cache)``. Use
    ``None`` to return the clean latent unchanged."""

    encoder: EncoderConfig | None = None
    """Optional per-AR-step input encoder. Must be a
    :class:`StreamingEncoder`; one-shot encoders go on
    ``transformer.context_encoder`` instead."""

    enable_sync_and_profile: bool = False
    """Record per-stage CUDA events and log timing per AR step. Calls
    ``torch.cuda.synchronize()`` once per step, which hurts throughput."""


@dataclass(kw_only=True)
class StreamInferencePipelineCache(
    Generic[StreamingEncoderCacheT, TransformerCacheT, StreamingDecoderCacheT]
):
    """Per-rollout cache held by the pipeline."""

    transformer_cache: TransformerCacheT
    """Long-lived transformer AR cache (always present)."""

    encoder_cache: StreamingEncoderCacheT | None = None
    """Encoder AR cache; ``None`` iff the pipeline has no encoder."""

    decoder_cache: StreamingDecoderCacheT | None = None
    """Decoder AR cache; ``None`` iff the pipeline has no decoder."""

    decoder_cache_codec: StreamingDecoderCacheT | None = None
    """Second decoder AR cache, for the latent-codec comparison branch.

    Non-``None`` only while the ``FD_LATENT_CODEC`` seam is active. It belongs to
    ``pipeline.decoder_codec`` and must NEVER be passed to ``pipeline.decoder``: a
    captured CUDA graph binds to the cache-slot storage addresses it was captured
    with, so crossing the pairing would replay writes into the other branch's slots
    and silently merge the two histories."""

    decoder_output_codec: "torch.Tensor | None" = None
    """Pixels decoded from the round-tripped latent by the comparison branch.

    ``__call__`` still returns the reference pixels, so the served video is unchanged;
    this carries the second branch's frames out for the offline comparison."""

    final_state: "DiffusionModel.FinalState[TransformerCacheT] | None" = None
    """Diffusion-model state from the most recent ``generate``, consumed
    by ``finalize``."""

    clean_latent: "torch.Tensor | None" = None
    """Pre-decode latent from the most recent ``generate`` (before the VAE
    decoder), exposed for consumers that stream latents instead of pixels."""

    autoregressive_index: int | None = None
    """AR step index of the most recent ``generate``."""

    event_profiler: EventProfiler | None = None
    """Per-step profiler, populated only when profiling is on."""


class StreamInferencePipeline(
    nn.Module,
    Generic[
        StreamingEncoderCacheT,
        TransformerCacheT,
        StreamingDecoderCacheT,
    ],
):
    """End-to-end streaming inference pipeline.

    Generic over the encoder, transformer, and decoder cache types. The
    encoder's input/output types are forwarded as ``Any`` so the
    transformer's ``predict_flow`` / ``postprocess_clean_latent`` overrides
    own the typing on the ``input`` argument they receive.

    Examples:

        cache = pipeline.initialize_cache(transformer_context={...})
        output = pipeline.generate(0, cache, input=...)
        pipeline.finalize(0, cache)
        output = pipeline.generate(1, cache, input=...)
        pipeline.finalize(1, cache)  # optional for the last rollout
    """

    encoder: StreamingEncoder[StreamingEncoderCacheT] | None
    decoder: StreamingDecoder[StreamingDecoderCacheT] | None
    decoder_codec: StreamingDecoder[StreamingDecoderCacheT] | None
    diffusion_model: DiffusionModel[TransformerCacheT]

    def __init__(self, config: StreamInferencePipelineConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = config.encoder.setup() if config.encoder is not None else None
        self.decoder = config.decoder.setup() if config.decoder is not None else None
        self.diffusion_model = config.diffusion_model.setup()

        # Second decoder for the latent-codec comparison. A separate INSTANCE, not just
        # a separate cache: the decoder captures a CUDA graph bound to its parameters,
        # its static input buffers and the cache slots it was captured with. One shared
        # instance driven from two caches would replay into the first cache's slots and
        # silently merge the branches -- which is exactly the bias this experiment must
        # not have. deepcopy rather than a second setup() so the weights are bit-identical
        # without re-reading the checkpoint.
        self.decoder_codec = None
        if self.decoder is not None and _resolve_latent_codec() is not None:
            self.decoder_codec = copy.deepcopy(self.decoder)
            logger.info(
                "Dual-decode ACTIVE: reference branch on pipeline.decoder, codec branch "
                "on pipeline.decoder_codec (separate instance, graph and cache)."
            )

    @property
    def device(self) -> torch.device:
        return self.diffusion_model.device

    def initialize_cache(
        self,
        transformer_context: dict[str, Any] | None = None,
        encoder_context: dict[str, Any] | None = None,
        decoder_context: dict[str, Any] | None = None,
    ) -> StreamInferencePipelineCache[
        StreamingEncoderCacheT, TransformerCacheT, StreamingDecoderCacheT
    ]:
        """Build a fresh per-rollout cache.

        Each ``*_context`` dict is forwarded as keyword arguments to the
        corresponding component's ``initialize_autoregressive_cache``.

        Args:
            transformer_context: Per-rollout state for the transformer
                (e.g. ``{"text_embeddings": ..., "image_embeddings": ...}``).
            encoder_context: Per-rollout state for the encoder. Ignored
                when there is no encoder.
            decoder_context: Per-rollout state for the decoder. Ignored
                when there is no decoder.

        Returns:
            A fresh cache to thread through ``generate`` / ``finalize``.
        """
        transformer_context = transformer_context or {}
        encoder_context = encoder_context or {}
        decoder_context = decoder_context or {}
        return StreamInferencePipelineCache(
            encoder_cache=(
                self.encoder.initialize_autoregressive_cache(**encoder_context)
                if self.encoder is not None
                else None
            ),
            decoder_cache=(
                self.decoder.initialize_autoregressive_cache(**decoder_context)
                if self.decoder is not None
                else None
            ),
            # Minted from decoder_codec, not decoder, so the cache and the instance that
            # will drive it are paired from birth.
            decoder_cache_codec=(
                self.decoder_codec.initialize_autoregressive_cache(**decoder_context)
                if self.decoder_codec is not None
                else None
            ),
            transformer_cache=self.diffusion_model.transformer.initialize_autoregressive_cache(
                **transformer_context
            ),
        )

    @torch.no_grad()
    def generate(
        self,
        autoregressive_index: int,
        cache: StreamInferencePipelineCache[
            StreamingEncoderCacheT, TransformerCacheT, StreamingDecoderCacheT
        ],
        input: Any = None,
    ) -> Tensor:
        """Generate one chunk for this AR step.

        Args:
            autoregressive_index: Must be ``cache.autoregressive_index + 1``,
                or ``0`` for the first call after ``initialize_cache``.
            cache: Per-rollout cache from ``initialize_cache``.
            input: Raw input fed to the encoder. Required when an encoder
                is configured, must be ``None`` otherwise. Use
                ``NullEncoderConfig`` to pass an already-encoded tensor
                straight through.

        Returns:
            Decoded tensor (e.g. RGB video) when a decoder is configured;
            otherwise the unpatchified clean latent from the diffusion model.
        """
        prev = cache.autoregressive_index
        expected = (prev + 1) if prev is not None else 0
        assert autoregressive_index == expected, (
            f"AR step out of order: previous step was {prev}, expected next "
            f"{expected}, got {autoregressive_index}"
        )
        cache.autoregressive_index = autoregressive_index

        events: EventProfiler | None = None
        if self.config.enable_sync_and_profile:
            events = EventProfiler()
            cache.event_profiler = events

        if input is not None:
            assert self.encoder is not None, (
                "input was provided but the pipeline has no encoder. "
                "Configure StreamInferencePipelineConfig.encoder (e.g. with "
                "NullEncoderConfig() for an identity passthrough)."
            )
            assert cache.encoder_cache is not None  # invariant: paired with encoder
            input = self.encoder(
                input=input,
                autoregressive_index=autoregressive_index,
                cache=cache.encoder_cache,
            )

        if events is not None:
            events.record("encode")

        clean_latent, final_state = self.diffusion_model.generate(
            autoregressive_index=autoregressive_index,
            cache=cache.transformer_cache,
            input=input,
        )
        cache.final_state = final_state
        cache.clean_latent = clean_latent

        if events is not None:
            events.record("diffuse")

        # --- TEMPORARY probe (set FD_LATENT_PROBE=1). Step 1 of the pixel-path SAS
        # work: prove from a live run that the DiT feedback tensor, the token-stream
        # tensor and the decoder input are the same buffer, and that the decoder does
        # not mutate it. Remove once step 2 lands. ---
        _probe = _latent_probe_enabled()
        _probe_sum = None
        if _probe:
            _fs = getattr(final_state, "clean_latent", None)
            _r_cl = _mem_range(clean_latent)
            _r_fs = _mem_range(_fs)
            _r_ca = _mem_range(cache.clean_latent)
            _probe_sum = clean_latent.float().sum().item()
            logger.info(
                "[PROBE ar={}] shape={} dtype={}\n"
                "    clean_latent      storage=[{}, {}) tensor@{}\n"
                "    final_state.clean storage=[{}, {}) tensor@{}  OVERLAPS={}\n"
                "    cache.clean_latent storage=[{}, {}) tensor@{}  OVERLAPS={}\n"
                "    sum={:.8f}",
                autoregressive_index,
                tuple(clean_latent.shape),
                clean_latent.dtype,
                _r_cl[0], _r_cl[1], _r_cl[2],
                *(_r_fs if _r_fs else (None, None, None)),
                _overlaps(_r_cl, _r_fs),
                _r_ca[0], _r_ca[1], _r_ca[2],
                _overlaps(_r_cl, _r_ca),
                _probe_sum,
            )

            # Definitive empirical check (FD_LATENT_PROBE=2): write through
            # clean_latent and see whether the DiT feedback tensor observes it.
            # Storage-range analysis should already answer this; this catches any case
            # where the ranges are misleading. The write is undone, but bf16 rounding
            # means the restore is not exact -- so treat a PROBE=2 run as throwaway.
            if os.environ.get("FD_LATENT_PROBE") == "2" and isinstance(_fs, Tensor):
                _fs_before = _fs.float().sum().item()
                clean_latent.add_(1.0)
                _fs_after = _fs.float().sum().item()
                clean_latent.sub_(1.0)
                logger.info(
                    "[PROBE ar={}] MUTATION TEST: wrote +1.0 through clean_latent -> "
                    "final_state.clean sum {:.8f} -> {:.8f}  ALIASED={}",
                    autoregressive_index,
                    _fs_before,
                    _fs_after,
                    _fs_before != _fs_after,
                )

        # --- Optional latent-codec seam. Inert unless FD_LATENT_CODEC is set, so the
        # default path stays exactly as before. The round-tripped latent goes to the
        # comparison decoder ONLY: clean_latent is left untouched, which keeps both the
        # DiT feedback and the token stream (which shares this buffer) on the
        # originals. ---
        _codec = _resolve_latent_codec()
        codec_latent = None
        if _codec is not None:
            codec_latent = _codec.roundtrip(clean_latent)
            _assert_latent_disjoint(
                codec_latent, clean_latent, final_state, _codec.name
            )

        if self.decoder is not None:
            assert cache.decoder_cache is not None  # invariant: paired with decoder
            # Branch A -- reference. Always the pristine latent, so what is served and
            # returned is unaffected by the experiment.
            output = self.decoder(
                input=clean_latent,
                autoregressive_index=autoregressive_index,
                cache=cache.decoder_cache,
            )
            # Branch B -- what a client reconstructing from the codec would see. Its own
            # instance and its own cache, so its temporal state evolves purely from
            # round-tripped latents. Sharing branch A's cache would let this branch
            # inherit clean-latent history and understate the codec's effect.
            if self.decoder_codec is not None:
                assert cache.decoder_cache_codec is not None
                assert codec_latent is not None
                _assert_branches_isolated(
                    self.decoder, self.decoder_codec, cache
                )
                cache.decoder_output_codec = self.decoder_codec(
                    input=codec_latent,
                    autoregressive_index=autoregressive_index,
                    cache=cache.decoder_cache_codec,
                )
                if autoregressive_index in (0, 4):
                    # Once before capture and once after (warmup_iters=2), so the log
                    # shows whether graph replay was actually reached.
                    _log_graph_capture(self.decoder, self.decoder_codec)

                # Both branches dumped from the same call site, so the two sequences are
                # frame-aligned by construction -- no index bookkeeping to get wrong.
                from flashdreams.infra.pipeline.frame_dump import dump_branch

                dump_branch(output, "reference", autoregressive_index)
                dump_branch(
                    cache.decoder_output_codec, "codec", autoregressive_index
                )
        else:
            output = clean_latent
            cache.decoder_output_codec = codec_latent

        if _probe:
            # If the decoder mutates its input in place, a SAS round-trip fed to it
            # would corrupt the DiT feedback even via a separate tensor -- so this
            # has to be checked, not assumed.
            _post = clean_latent.float().sum().item()
            logger.info(
                "[PROBE ar={}] post-decode sum={:.8f} decoder_mutated_input={}",
                autoregressive_index,
                _post,
                _post != _probe_sum,
            )

        if events is not None:
            events.record("decode")

        return output

    @torch.no_grad()
    def finalize(
        self,
        autoregressive_index: int,
        cache: StreamInferencePipelineCache[
            StreamingEncoderCacheT, TransformerCacheT, StreamingDecoderCacheT
        ],
    ) -> dict[str, float] | None:
        """Advance the diffusion AR cache for the next AR step.

        Args:
            autoregressive_index: Must match the index passed to the most
                recent ``generate`` (asserted).
            cache: Same cache used by ``generate``. Consumes
                ``cache.final_state``.

        Returns:
            ``None`` when profiling is disabled. Otherwise a snapshot of this
            AR step's per-stage timings (ms) and GPU memory (GiB):
            ``{<stage>_ms, total_ms, total_ms_wo_finalize, mem_alloc_gib,
            mem_reserved_gib, mem_peak_gib}``. The same numbers are also
            logged via ``logger.info``.
        """
        assert cache.autoregressive_index == autoregressive_index, (
            f"autoregressive_index mismatch: generate() ran with "
            f"{cache.autoregressive_index} but finalize() was called with "
            f"{autoregressive_index}."
        )
        assert cache.final_state is not None, (
            "finalize() called before generate() — no FinalState on the cache."
        )
        self.diffusion_model.finalize(final_state=cache.final_state)
        if not self.config.enable_sync_and_profile:
            return None

        assert cache.event_profiler is not None, (
            "finalize() called before any generate() — no EventProfiler on the cache."
        )
        cache.event_profiler.record("finalize")
        stats_ms = cache.event_profiler.sync_and_summarize()
        total_ms = sum(stats_ms.values())
        total_ms_wo_finalize = total_ms - stats_ms.get("finalize", 0.0)
        stages_str = " ".join(f"{stage} {ms:.3f} ms" for stage, ms in stats_ms.items())

        stats: dict[str, float] = {f"{stage}_ms": ms for stage, ms in stats_ms.items()}
        stats["total_ms"] = total_ms
        stats["total_ms_wo_finalize"] = total_ms_wo_finalize

        mem_str = ""
        if torch.cuda.is_available():
            device = torch.cuda.current_device()
            gib = 1024**3
            mem_alloc_gib = torch.cuda.memory_allocated(device) / gib
            mem_reserved_gib = torch.cuda.memory_reserved(device) / gib
            mem_peak_gib = torch.cuda.max_memory_allocated(device) / gib
            stats["mem_alloc_gib"] = mem_alloc_gib
            stats["mem_reserved_gib"] = mem_reserved_gib
            stats["mem_peak_gib"] = mem_peak_gib
            mem_str = (
                f" | GPU mem alloc {mem_alloc_gib:.3f} GiB "
                f"reserved {mem_reserved_gib:.3f} GiB "
                f"peak {mem_peak_gib:.3f} GiB"
            )
        logger.info(
            f"AR {autoregressive_index} {stages_str} | "
            f"total(w/o finalize) {total_ms_wo_finalize:.3f} ms "
            f"total {total_ms:.3f} ms"
            f"{mem_str}"
        )
        return stats
