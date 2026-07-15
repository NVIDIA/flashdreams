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

"""FlashVSR streaming inference pipeline."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import TypeAlias

import torch
from torch import Tensor

from flashdreams.core.distributed.context_parallel import cat_outputs_cp
from flashdreams.core.io.download import download_to_cache
from flashdreams.infra.diffusion.model import DiffusionModel
from flashdreams.infra.pipeline import (
    StreamInferencePipeline,
    StreamInferencePipelineCache,
    StreamInferencePipelineConfig,
)
from flashdreams.infra.profiler import EventProfiler, record_event
from flashdreams.recipes.wan.transformer.wan21 import Wan21TransformerCache
from flashvsr.decoder import (
    FlashVSRDecoder,
    FlashVSRDecoderCache,
    FlashVSRDecoderConfig,
)
from flashvsr.encoder import (
    FlashVSREncoder,
    FlashVSREncoderCache,
    FlashVSREncoderConfig,
)
from flashvsr.transformer import (
    FlashVSRTransformer,
    FlashVSRTransformerConfig,
)

FlashVSRPipelineCache: TypeAlias = StreamInferencePipelineCache[
    FlashVSREncoderCache,
    Wan21TransformerCache,
    FlashVSRDecoderCache,
]


PROMPT_CACHE_DIR = (
    Path(os.path.expanduser(os.getenv("FLASHDREAMS_CACHE_DIR", "~/.cache/flashdreams")))
    / "flashvsr"
)
"""User-writable cache for downloaded frozen UMT5 prompt tensors."""


def _load_prompt_tensor(prompt_path: str) -> Tensor:
    """Load FlashVSR's frozen UMT5 prompt tensor.

    URL inputs are downloaded once into :data:`PROMPT_CACHE_DIR` through
    :func:`download_to_cache`; filesystem paths are loaded directly.
    ``posi_prompt.pth`` is a bare ``Tensor`` saved with ``torch.save`` rather
    than a checkpoint state dict, so the pipeline deserializes it with
    ``torch.load`` and validates the object type here.

    Args:
        prompt_path: Local filesystem path or HTTP(S) URL for
            ``posi_prompt.pth``.

    Returns:
        Prompt embedding loaded on CPU.
    """
    if prompt_path.startswith(("http://", "https://")):
        local_path = download_to_cache(prompt_path, cache_dir=PROMPT_CACHE_DIR)
    else:
        local_path = Path(prompt_path)
    tensor = torch.load(local_path, map_location="cpu", weights_only=True)
    assert isinstance(tensor, Tensor), (
        f"Expected {prompt_path} to pickle a Tensor; got {type(tensor).__name__}."
    )
    return tensor


@dataclass(kw_only=True)
class FlashVSRPipelineConfig(StreamInferencePipelineConfig):
    """Configuration for :class:`FlashVSRPipeline`.

    The required ``diffusion_model`` field is inherited from
    :class:`StreamInferencePipelineConfig`. The encoder and decoder fields are
    narrowed to the FlashVSR implementations because :class:`FlashVSRPipeline`
    relies on their cache side channels: the encoder records the chunk's
    bicubic upres and internal-iteration count, and the decoder consumes that
    upres as both conditioning and color reference.
    """

    _target: type["FlashVSRPipeline"] = field(  # type: ignore[assignment]
        default_factory=lambda: FlashVSRPipeline
    )

    encoder: FlashVSREncoderConfig = field(  # type: ignore[assignment]
        default_factory=FlashVSREncoderConfig
    )
    """LR projector + bicubic upres encoder."""

    decoder: FlashVSRDecoderConfig = field(  # type: ignore[assignment]
        default_factory=FlashVSRDecoderConfig
    )
    """TC decoder + AdaIN color corrector."""

    prompt_path: str | None = None
    """Path or URL for ``posi_prompt.pth``.

    The tensor is the frozen UMT5 prompt embedding used by the FlashVSR DiT,
    typically shaped ``[1, 512, 4096]`` for the shipped checkpoint. When this
    is set, the pipeline loads the tensor once during construction and reuses
    it for every :meth:`FlashVSRPipeline.initialize_cache` call. Leave it as
    ``None`` only when callers will pass ``prompt_tensor=...`` explicitly.
    """


class FlashVSRPipeline(
    StreamInferencePipeline[
        FlashVSREncoderCache,
        Wan21TransformerCache,
        FlashVSRDecoderCache,
    ]
):
    """FlashVSR streaming video super-resolution pipeline.

    Construct through ``flashvsr.config.build_flashvsr_v1_1`` or one of the
    shipped config literals so the encoder dimensions, prompt tensor, DiT
    checkpoint, and decoder checkpoint stay in sync. A cache represents one
    rollout over a video; call :meth:`generate` and then :meth:`finalize` for
    each chunk in increasing ``autoregressive_index`` order.

    Examples:

        from flashvsr.config import build_flashvsr_v1_1
        pipeline = build_flashvsr_v1_1(input_H=704, input_W=1280).setup().to("cuda")

        cache = pipeline.initialize_cache()
        for chunk_idx, (start, size) in enumerate(chunks):
            clip = video[..., start:start + size]
            out = pipeline.generate(chunk_idx, cache, clip)
            pipeline.finalize(chunk_idx, cache)
    """

    encoder: FlashVSREncoder
    decoder: FlashVSRDecoder

    def __init__(self, config: FlashVSRPipelineConfig) -> None:
        super().__init__(config)
        self.config: FlashVSRPipelineConfig = config

        transformer = self.diffusion_model.transformer
        assert isinstance(transformer, FlashVSRTransformer), (
            "FlashVSRPipeline requires a FlashVSRTransformer as the diffusion "
            f"model's transformer; got {type(transformer).__name__}."
        )
        assert isinstance(self.encoder, FlashVSREncoder), (
            "FlashVSRPipeline requires a FlashVSREncoder; got "
            f"{type(self.encoder).__name__}."
        )
        assert isinstance(self.decoder, FlashVSRDecoder), (
            "FlashVSRPipeline requires a FlashVSRDecoder; got "
            f"{type(self.decoder).__name__}."
        )

        # Keep the configured prompt on CPU and move/cast it per rollout in
        # ``initialize_cache``. Callers that pass ``prompt_tensor=...`` bypass
        # this cached copy.
        self._prompt_tensor: Tensor | None = (
            _load_prompt_tensor(config.prompt_path)
            if config.prompt_path is not None
            else None
        )

    @torch.no_grad()
    def initialize_cache(  # type: ignore[override]
        self,
        prompt_tensor: Tensor | None = None,
    ) -> FlashVSRPipelineCache:
        """Build the cache state for one FlashVSR rollout.

        The returned cache owns the encoder projector tail buffers, the rolling
        transformer KV cache, the decoder streaming state, and the per-step
        side-channel slots used inside :meth:`generate`. A cache should be
        threaded through all chunks from one video stream and not reused for an
        unrelated stream.

        The prompt tensor is moved to ``self.device`` and cast to the DiT dtype
        before it is forwarded as ``text_embeddings``. The transformer's latent
        height and width are derived from the encoder's cropped high-resolution
        target dimensions using Wan's 8x spatial compression.

        Args:
            prompt_tensor: UMT5 prompt embedding
                ``[1, text_len, text_dim]`` for the DiT cross-attention cache;
                ``None`` uses the tensor loaded from ``config.prompt_path``.
                One of these two sources must be available.

        Returns:
            Fresh cache to thread through one video stream.
        """
        prompt = prompt_tensor if prompt_tensor is not None else self._prompt_tensor
        assert prompt is not None, (
            "FlashVSRPipeline.initialize_cache requires a prompt tensor: "
            "pass prompt_tensor=... or set FlashVSRPipelineConfig.prompt_path."
        )
        prompt = prompt.to(device=self.device, dtype=self.diffusion_model.dtype)
        # The Wan cache stores latent spatial size, not pixel size.
        latent_height = self.encoder.target_H // 8
        latent_width = self.encoder.target_W // 8
        return super().initialize_cache(
            transformer_context={
                "text_embeddings": prompt,
                "height": latent_height,
                "width": latent_width,
            },
            encoder_context={},
            decoder_context={},
        )

    @staticmethod
    def reset_cache_in_place(cache: FlashVSRPipelineCache) -> None:
        """Reset a rollout while preserving transformer KV-buffer identities.

        CUDA graphs in :class:`FlashVSRTransformer` capture the transformer's
        per-rollout KV-buffer addresses. Replacing ``cache`` after warmup
        invalidates those graphs and repeats their expensive warmup and capture
        during the measured rollout. Reset the nested caches in place so the
        transformer graph remains reusable; the encoder and decoder restore
        their cold-start semantics independently.

        Args:
            cache: Completed warmup cache to reuse for the real rollout.
        """
        assert cache.encoder_cache is not None
        assert cache.decoder_cache is not None
        cache.encoder_cache.reset()
        cache.transformer_cache.reset()
        cache.decoder_cache.reset()
        cache.final_state = None
        cache.autoregressive_index = None
        cache.event_profiler = None

    @torch.no_grad()
    def generate(  # type: ignore[override]
        self,
        autoregressive_index: int,
        cache: FlashVSRPipelineCache,
        input: Tensor,
    ) -> Tensor:
        """Upsample one complete FlashVSR chunk.

        The first chunk may contain 5 or 13 raw frames; steady-state chunks
        contain 8 or 16 raw frames. The encoder pads cold-start chunks
        internally, reports whether the padded chunk requires one or two
        internal DiT iterations, and leaves the bicubic upres on its cache for
        the decoder.

        Callers must invoke :meth:`finalize` with the same
        ``autoregressive_index`` after consuming the returned tensor. For
        FlashVSR, finalization advances the transformer's rolling KV cache and,
        when profiling is enabled, records the final timing event.

        Args:
            autoregressive_index: Must be ``cache.autoregressive_index + 1``,
                or ``0`` for the first call after ``initialize_cache``.
            cache: Per-rollout cache from ``initialize_cache``.
            input: Low-resolution frames ``[B, 3, T, H, W]`` in ``[-1, 1]``.
                ``T`` must be one of the chunk sizes accepted by
                :class:`FlashVSREncoderConfig`.

        Returns:
            Upsampled RGB frames ``[B, 3, T_out, target_H, target_W]`` in
            ``[-1, 1]``. ``T_out`` matches the unpadded input frame count.

        Note:
            With ``enable_sync_and_profile=True``, this method contributes
            ``pad``, ``bicubic``, ``projector``, ``dit_concat``, ``denoise``,
            ``decoder``, and ``color`` events. The inherited
            :meth:`StreamInferencePipeline.finalize` appends ``finalize`` and
            returns the summarized timings.
        """
        prev = cache.autoregressive_index
        expected = (prev + 1) if prev is not None else 0
        assert autoregressive_index == expected, (
            f"AR step out of order: previous step was {prev}, expected next "
            f"{expected}, got {autoregressive_index}"
        )
        cache.autoregressive_index = autoregressive_index

        # One profiler lives on the parent cache for the whole public AR step.
        # The encoder and decoder record into it through explicit kwargs.
        if self.config.enable_sync_and_profile:
            cache.event_profiler = EventProfiler()
        event_profiler = cache.event_profiler

        ## Encoder

        # Produces per-block LR token tensors and records two side channels:
        # ``last_n_iters`` for the DiT loop below and ``last_upres`` for the
        # decoder conditioning path.
        assert cache.encoder_cache is not None  # invariant: paired with encoder
        per_block_latents = self.encoder(
            input=input,
            autoregressive_index=autoregressive_index,
            cache=cache.encoder_cache,
            event_profiler=event_profiler,
        )
        n_iters = cache.encoder_cache.last_n_iters
        assert n_iters in (1, 2), (
            f"FlashVSREncoder.last_n_iters must be 1 or 2 (got {n_iters})."
        )

        # The decoder reads this as both TC-decoder conditioning and the AdaIN
        # color reference. Keep the tensor unpadded so output length matches
        # the user-visible input length.
        assert cache.decoder_cache is not None  # invariant: paired with decoder
        cache.decoder_cache.last_upres = cache.encoder_cache.last_upres

        ## Chunk noise

        # Legacy FlashVSR samples ``[B, C, n_latent, H, W]`` before patchify.
        # Preserve that order; patchify interleaves space and time, so slicing
        # a previously patchified full-chunk noise tensor would not be equal.
        transformer = self.diffusion_model.transformer
        # Narrow from the abstract Wan base to the subclass asserted in
        # ``__init__`` so the FlashVSR-specific config fields are visible.
        assert isinstance(transformer, FlashVSRTransformer)
        cfg = transformer.config
        assert isinstance(cfg, FlashVSRTransformerConfig)
        # ``initialize_cache`` seeds these per-rollout latent dimensions via
        # ``transformer_context``.
        latent_h = transformer._output_height
        latent_w = transformer._output_width
        assert latent_h is not None and latent_w is not None, (
            "FlashVSRPipeline.generate called before initialize_cache: "
            "transformer._output_height/_width must be populated."
        )
        _kt, kh, kw = cfg.network.patch_size
        pH = latent_h // kh
        pW = latent_w // kw
        len_t = cfg.len_t
        n_latent = len_t * n_iters
        full_noise = torch.randn(
            (1, cfg.network.in_dim, n_latent, latent_h, latent_w),
            device=transformer.device,
            dtype=transformer.dtype,
            generator=self.diffusion_model.rng,
        )

        ## Internal DiT iterations

        # Each iteration consumes two latent frames and one matching slice of
        # every per-block LR-token tensor. The internal AR index advances once
        # per iteration, matching the legacy ``chunk_idx * n_iters + idx``.
        #
        # Context parallelism must split each DiT call independently. If a
        # 16-frame chunk were split after concatenating both iterations, ranks
        # could receive different iteration ranges instead of shards of the
        # same iteration. Gather each iteration's clean tokens before joining.
        L_per_iter = len_t * pH * pW
        # FlashVSR's distilled DiT uses the same sigma=1 / t=1000 point for
        # every internal iteration.
        timestep = torch.tensor(
            [1000.0], device=transformer.device, dtype=transformer.dtype
        )
        clean_parts: list[Tensor] = []
        for idx in range(n_iters):
            internal_ar_idx = autoregressive_index * n_iters + idx
            cache.transformer_cache.start(autoregressive_index=internal_ar_idx)

            per_iter_lq = [
                L[:, idx * L_per_iter : (idx + 1) * L_per_iter, :]
                for L in per_block_latents
            ]
            per_iter_lq = transformer.patchify_and_maybe_split_cp(per_iter_lq)

            # Slice before patchify for legacy parity; the transformer's hook
            # then rearranges and CP-splits this iteration only.
            noise_slice = full_noise[:, :, idx * len_t : (idx + 1) * len_t, :, :]
            noisy_patched = transformer.patchify_and_maybe_split_cp(
                noise_slice.transpose(1, 2)
            )
            flow = transformer.predict_flow(
                noisy_latent=noisy_patched,
                timestep=timestep,
                cache=cache.transformer_cache,
                input=per_iter_lq,
            )
            clean_iter = noisy_patched - flow
            clean_parts.append(
                cat_outputs_cp(clean_iter, seq_dim=-2, cp_group=transformer._cp_group)
            )

            # Pair every internal ``start`` with an ``after_update``. The last
            # one is deferred to the public ``finalize`` call via ``FinalState``
            # so the caller-facing lifecycle remains generate -> finalize.
            if idx < n_iters - 1:
                cache.transformer_cache.finalize(autoregressive_index=internal_ar_idx)

        record_event(event_profiler, "dit_concat")

        ## Clean latent reassembly

        clean_patched = torch.cat(clean_parts, dim=-2)
        clean_latent = transformer.network.unpatchify_and_maybe_gather_cp(
            pH=pH,
            pW=pW,
            x=clean_patched,
        )

        # Stash enough state for the inherited ``finalize`` to close the final
        # internal DiT iteration. FlashVSR's ``finalize_kv_cache`` is a no-op
        # and ``context_noise == 0`` skips re-noising, so the meaningful side
        # effect is ``cache.finalize(last_internal_ar_idx)``.
        cache.final_state = DiffusionModel.FinalState(
            clean_latent=clean_patched,
            autoregressive_index=autoregressive_index * n_iters + (n_iters - 1),
            cache=cache.transformer_cache,
        )
        record_event(event_profiler, "denoise")

        ## Decoder

        # TC decoder + color corrector. Both consume the bicubic upres that was
        # forwarded to ``cache.decoder_cache.last_upres`` above.
        return self.decoder(
            input=clean_latent,
            autoregressive_index=autoregressive_index,
            cache=cache.decoder_cache,
            event_profiler=event_profiler,
        )
