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

"""FlashVSR streaming inference pipeline (LR projector + DiT + TC decoder).

Subclasses :class:`StreamInferencePipeline`; one
``pipeline.generate(autoregressive_index, cache, input)`` call processes
one full FlashVSR chunk. The 5-step algorithm (encoder -> sample noise
-> per-iter DiT -> noise - flow -> decoder) mirrors the legacy
``UltraFlashVSRUpsampler.forward`` exactly. See the sibling ``README.md``
for the full streaming chunk contract and per-component knob list.

Three FlashVSR-specific quirks the abstract pipeline doesn't natively
handle, and why ``generate`` is fully inlined instead of calling
``super().generate``:

- The encoder's output is a ``list[Tensor]`` of per-block LR latent
  slices; :class:`FlashVSRTransformer` overrides
  ``patchify_and_maybe_split_cp`` to pass lists through.
- The TC decoder needs the bicubic upres as ``cond``; the encoder
  stashes it on its own cache and the pipeline forwards it.
- Sampling noise once per chunk (then slicing pre-patchify) is needed
  for parity with the legacy ``generate_noise(n_latent)`` -- the
  inherited ``DiffusionModel.generate`` re-samples per AR step.

FlashVSR's distilled DiT was trained with KV cache K/V derived from the
**noisy** latent (sigma=1, t=1000); we preserve this by overriding
:meth:`FlashVSRTransformer.finalize_kv_cache` to a no-op.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import TypeAlias

import torch
from torch import Tensor

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
"""User-writable cache for the precomputed UMT5 prompt tensor."""


def _load_prompt_tensor(prompt_path: str) -> Tensor:
    """Load FlashVSR's frozen UMT5 prompt tensor from a local path or HTTP URL.

    ``http(s)://`` strings are atomically fetched into
    :data:`PROMPT_CACHE_DIR` (via :func:`download_to_cache`) on first use;
    local paths pass through unchanged. ``posi_prompt.pth`` pickles a
    bare ``Tensor`` (not a state dict), so we deserialize with
    ``torch.load`` directly rather than routing through
    ``load_checkpoint``.
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

    Required fields are inherited from :class:`StreamInferencePipelineConfig`
    (``diffusion_model``); the encoder / decoder slots are pinned to the
    FlashVSR-specific configs because the pipeline subclass relies on the
    encoder's ``last_upres`` side-channel and the decoder's
    ``cond``-aware forward.
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
    """Path to ``posi_prompt.pth`` (``[1, 512, 4096]`` UMT5 embedding).
    Loaded once at pipeline construction and forwarded to
    :meth:`FlashVSRTransformer.initialize_autoregressive_cache` via the
    ``text_embeddings=`` kwarg on every :meth:`FlashVSRPipeline.initialize_cache`
    call. Set to ``None`` to require callers to pass ``prompt_tensor``
    explicitly."""


class FlashVSRPipeline(
    StreamInferencePipeline[
        FlashVSREncoderCache,
        Wan21TransformerCache,
        FlashVSRDecoderCache,
    ]
):
    """FlashVSR streaming video super-resolution pipeline.

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

        # Loaded once on CPU and reused on every ``initialize_cache`` call;
        # ``initialize_cache`` moves it to ``self.device``. ``posi_prompt.pth``
        # is a precomputed UMT5 prompt tensor (a bare Tensor pickled with
        # ``torch.save``), not a checkpoint state dict, so we deserialize via
        # ``download_to_cache`` + ``torch.load`` rather than ``load_checkpoint``.
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
        """Build a fresh per-rollout cache.

        Args:
            prompt_tensor: Optional ``[1, text_len, text_dim]`` UMT5 prompt
                embedding to seed the DiT cross-attention KV cache. Falls
                back to the prompt tensor loaded from ``config.prompt_path``
                if not provided. Required if neither is set.
        """
        prompt = prompt_tensor if prompt_tensor is not None else self._prompt_tensor
        assert prompt is not None, (
            "FlashVSRPipeline.initialize_cache requires a prompt tensor: "
            "pass prompt_tensor=... or set FlashVSRPipelineConfig.prompt_path."
        )
        prompt = prompt.to(device=self.device, dtype=self.diffusion_model.dtype)
        # Per-rollout latent (height, width) for the DiT cache, derived from
        # the encoder's pixel-space upres dims via Wan VAE's 8x spatial
        # compression. ``Wan21Transformer.initialize_autoregressive_cache``
        # consumes them via ``transformer_context``.
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

    @torch.no_grad()
    def generate(  # type: ignore[override]
        self,
        autoregressive_index: int,
        cache: FlashVSRPipelineCache,
        input: Tensor,
    ) -> Tensor:
        """Process one full FlashVSR chunk (8 or 16 raw frames).

        See module docstring for the 5-step algorithm. The profiler emits
        seven events (``pad`` / ``bicubic`` / ``projector`` / ``dit_concat``
        / ``denoise`` / ``decoder`` / ``color``) because steps 1 and 3 each
        record sub-stages. Mirrors the legacy
        ``UltraFlashVSRUpsampler.forward`` exactly.

        Args:
            autoregressive_index: Must be ``cache.autoregressive_index + 1``,
                or ``0`` for the first call after ``initialize_cache``.
            cache: Per-rollout cache from ``initialize_cache``.
            input: Low-resolution frames ``[B, 3, T, H, W]`` in ``[-1, 1]``.
                See :class:`FlashVSREncoderConfig` for the per-step ``T``
                contract.

        Returns:
            Upsampled RGB frames ``[B, 3, T_out, target_H, target_W]`` in
            ``[-1, 1]``. ``T_out`` matches the un-padded input frame count.
        """
        prev = cache.autoregressive_index
        expected = (prev + 1) if prev is not None else 0
        assert autoregressive_index == expected, (
            f"AR step out of order: previous step was {prev}, expected next "
            f"{expected}, got {autoregressive_index}"
        )
        cache.autoregressive_index = autoregressive_index

        # Profiling. Allocate one shared :class:`EventProfiler` on the
        # parent cache; both encoder and decoder receive it as an explicit
        # ``event_profiler`` kwarg below (no per-sub-cache duplication).
        # Per-AR-step record order is the legacy 7-stage breakdown
        # (``pad`` -> ``bicubic`` -> ``projector`` -> ``dit_concat``
        # -> ``denoise`` -> ``decoder`` -> ``color``); the inherited
        # :meth:`StreamInferencePipeline.finalize` appends the ``finalize``
        # event and sync-summarizes, so the runner's existing summary
        # block consumes it unchanged.
        if self.config.enable_sync_and_profile:
            cache.event_profiler = EventProfiler()
        event_profiler = cache.event_profiler

        # ----- 1. Encoder -----
        # Bicubic + projector. Side-stashes ``last_upres`` and
        # ``last_n_iters`` on cache.encoder_cache. Records ``pad``,
        # ``bicubic`` and ``projector`` against the shared profiler.
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

        # Forward the encoder's upres to the decoder cache. The TC decoder
        # reads ``last_upres`` as ``cond`` and the color corrector uses it
        # as the AdaIN reference.
        assert cache.decoder_cache is not None  # invariant: paired with decoder
        cache.decoder_cache.last_upres = cache.encoder_cache.last_upres

        # ----- 2. Sample noise once for the full chunk -----
        # Pre-patchify shape ``[B, in_dim, n_latent, latent_H, latent_W]``,
        # matching the legacy ``UltraFlashVSRUpsampler.generate_noise(n_latent)``.
        # Sampling once and slicing in pre-patchify space (vs sampling per
        # iter post-patchify) is required for parity with the legacy: the
        # patchify rearrange interleaves spatial + temporal axes, so the
        # per-iter noise vectors differ between the two approaches.
        transformer = self.diffusion_model.transformer
        # Narrow from the abstract base ``Wan21Transformer`` (with
        # ``config: InstantiateConfig[Any]``) to the concrete subclass
        # asserted in ``__init__``. Required for ty + readability.
        assert isinstance(transformer, FlashVSRTransformer)
        cfg = transformer.config
        assert isinstance(cfg, FlashVSRTransformerConfig)
        # Per-rollout latent (height, width) and patchified (pH, pW) live on
        # the transformer instance after ``initialize_autoregressive_cache``;
        # ``initialize_cache`` above seeds them via ``transformer_context``.
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

        # ----- 3. Loop n_iters internal DiT iterations -----
        # Each iter consumes a 2-latent-frame slice of the noise and a
        # ``L_per_iter``-token slice of the per-block LR latents. The
        # internal AR step index advances by 1 per iter so the rolling
        # KV cache rolls at the right cadence (matches legacy
        # ``cur_process_idx = chunk_idx * n_iters + idx``).
        #
        # ``flow_full`` is pre-allocated once and each iter's flow is
        # ``copy_``-ed into its slot immediately after ``predict_flow``.
        # This is required for ``compile_network=True`` paths that fall
        # back into Inductor cudagraphs (``mode="max-autotune"`` or
        # similar): the DiT's compiled output lives in a static
        # cudagraph buffer that gets clobbered by the next iter's call,
        # so the original ``flow_parts.append(flow)`` + post-loop
        # ``torch.cat`` pattern crashes with "accessing tensor output
        # of CUDAGraphs that has been overwritten by a subsequent run".
        # The slice-copy materialises each iter's flow into a stable,
        # caller-owned tensor before the next call. Hoisting
        # ``full_noise_patched`` lets us use it as the ``empty_like``
        # template and reuse it for the post-loop ``noise - flow``
        # subtract.
        L_per_iter = len_t * pH * pW
        # FlashVSR's distilled DiT is fixed at t=1000 every chunk.
        timestep = torch.tensor(
            [1000.0], device=transformer.device, dtype=transformer.dtype
        )
        full_noise_patched = transformer.patchify_and_maybe_split_cp(
            full_noise.transpose(1, 2)
        )
        flow_full = torch.empty_like(full_noise_patched)
        for idx in range(n_iters):
            internal_ar_idx = autoregressive_index * n_iters + idx
            cache.transformer_cache.start(autoregressive_index=internal_ar_idx)

            per_iter_lq = [
                L[:, idx * L_per_iter : (idx + 1) * L_per_iter, :]
                for L in per_block_latents
            ]

            # Slice the chunk-shared noise to this iter's 2 latent frames
            # (pre-patchify), then route through the transformer's standard
            # patchify hook to get ``[B, L_per_iter, D]``. Note: this is
            # *not* equivalent to slicing ``full_noise_patched`` post-
            # patchify -- the patchify rearrange interleaves spatial and
            # temporal axes, so per-iter parity with the legacy upsampler
            # requires slicing pre-patchify and patchifying each slice
            # independently.
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
            flow_full[..., idx * L_per_iter : (idx + 1) * L_per_iter, :].copy_(flow)

            # Per-iter cache lifecycle: ``start()`` ran before predict_flow
            # (hoisted ``before_update``); pair it with ``finalize()``
            # (``after_update``) here for all but the last iter. The last
            # iter's ``after_update`` is deferred to the inherited
            # ``StreamInferencePipeline.finalize`` -> ``DiffusionModel.finalize``
            # -> ``final_state.cache.finalize(...)``, which lands on the
            # right index thanks to the ``FinalState`` we stash below.
            if idx < n_iters - 1:
                cache.transformer_cache.finalize(autoregressive_index=internal_ar_idx)

        record_event(event_profiler, "dit_concat")

        # ----- 4. clean = noise - flow at sigma=1 -----
        clean_patched = full_noise_patched - flow_full
        clean_latent = transformer.unpatchify_and_maybe_gather_cp(clean_patched)

        # Stash ``FinalState`` so the inherited ``StreamInferencePipeline.finalize``
        # routes to ``DiffusionModel.finalize``, which for FlashVSR collapses
        # to a single ``cache.finalize(autoregressive_index)`` call
        # (``context_noise == 0`` skips re-noise; ``FlashVSRTransformer.finalize_kv_cache``
        # is a no-op). The ``autoregressive_index`` is the **last internal**
        # iter index so that call advances the per-block KV cache at the
        # right cadence (one ``after_update`` per internal iter). ``clean_latent``
        # is never read by ``DiffusionModel.finalize`` for FlashVSR (both branches
        # that consume it are short-circuited), but the dataclass requires it; we
        # pass the patchified clean latent for honesty.
        cache.final_state = DiffusionModel.FinalState(
            clean_latent=clean_patched,
            autoregressive_index=autoregressive_index * n_iters + (n_iters - 1),
            cache=cache.transformer_cache,
        )
        record_event(event_profiler, "denoise")

        # ----- 5. Decoder -----
        # TC decoder + color corrector. Reads ``last_upres`` from
        # cache.decoder_cache. Records ``decoder`` and ``color`` against the
        # shared profiler from inside :meth:`FlashVSRDecoder.forward`.
        return self.decoder(
            input=clean_latent,
            autoregressive_index=autoregressive_index,
            cache=cache.decoder_cache,
            event_profiler=event_profiler,
        )
