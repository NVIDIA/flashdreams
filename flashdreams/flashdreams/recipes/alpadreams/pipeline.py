"""Alpadreams streaming inference pipeline (Cosmos DiT + I2V + HDMap).

Project-level :class:`StreamInferencePipeline` subclass (no wrapper
indirection) for the alpadreams Cosmos DiT checkpoints. The pipeline
wires:

- :class:`CosmosReason1TextEncoder` — one-shot at rollout start.
- A first-frame Wan VAE encoder (project-level ``image_encoder``) —
  one-shot at rollout start; its output seeds the long-lived AR cache
  (mask injection at AR step 0 only).
- The infra :attr:`StreamInferencePipelineConfig.encoder` —
  per-AR-step HDMap encoder (Wan VAE *or* PixelShuffle pseudo-VAE),
  required.
- The infra :attr:`StreamInferencePipelineConfig.decoder` —
  per-AR-step video decoder (Wan VAE *or* TAEHV).
- A :class:`DiffusionModelConfig` over :class:`CosmosTransformerConfig`.

I2V is implemented by :class:`CosmosTransformer` itself (mask injection
into ``noisy_latent`` and ``postprocess_clean_latent``); the per-AR-step
``input`` is the encoded HDMap chunk routed through the infra
encoder slot.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypeAlias

import torch
from torch import Tensor

from flashdreams.infra.decoder import DecoderAutoregressiveCache
from flashdreams.infra.encoder import EncoderAutoregressiveCache
from flashdreams.infra.pipeline import (
    StreamInferencePipeline,
    StreamInferencePipelineCache,
    StreamInferencePipelineConfig,
)
from flashdreams.infra.encoder.text.cosmos_qwen import (
    CosmosReason1TextEncoder,
    CosmosReason1TextEncoderConfig,
)
from flashdreams.recipes.alpadreams.transformer import (
    CosmosTransformerCache,
)
from flashdreams.recipes.wan.autoencoder.vae import (
    WanVAEEncoder,
    WanVAEEncoderConfig,
)


AlpadreamsPipelineCache: TypeAlias = StreamInferencePipelineCache[
    EncoderAutoregressiveCache,  # EncCacheT
    CosmosTransformerCache,  # TransformerCacheT
    DecoderAutoregressiveCache,  # DecCacheT
]


@dataclass(kw_only=True)
class AlpadreamsPipelineConfig(StreamInferencePipelineConfig):
    """Hyperparameters for :class:`AlpadreamsPipeline`.

    Extends :class:`StreamInferencePipelineConfig` (inherits
    ``diffusion_model``, ``encoder``, ``decoder``) with the project-level
    text and first-frame image encoders. ``encoder`` MUST be set (the
    per-AR-step HDMap encoder); ``diffusion_model.transformer`` must be
    a :class:`CosmosTransformerConfig`.
    """

    _target: type["StreamInferencePipeline"] = field(
        default_factory=lambda: AlpadreamsPipeline
    )

    text_encoder: CosmosReason1TextEncoderConfig = field(
        default_factory=CosmosReason1TextEncoderConfig
    )
    """Cosmos-Reason1 text encoder, run once at the start of each rollout."""

    image_encoder: WanVAEEncoderConfig = field(default_factory=WanVAEEncoderConfig)
    """One-shot Wan VAE encoder used to encode the first-frame image
    (per-rollout, NOT per-AR-step). Pin its checkpoint to the same VAE
    that produced the network's training distribution."""


class AlpadreamsPipeline(
    StreamInferencePipeline[
        EncoderAutoregressiveCache,  # EncCacheT
        CosmosTransformerCache,  # TransformerCacheT
        DecoderAutoregressiveCache,  # DecCacheT
    ]
):
    """Streaming alpadreams inference pipeline (Cosmos DiT + HDMap + I2V mask).

    Usage::

        config = ALPADREAMS_CONFIG_BUILDERS["sv_..."](device=device)
        pipeline = config.setup().to(device=device)

        cache = pipeline.initialize_cache(
            text=[["A driving scene..."]],  # [B, V] (V=1 for single-view)
            image=first_frames,             # [B, V, 1, 3, H, W]
            view_names=["camera_front_..."],
        )
        for i in range(num_blocks):
            chunk = pipeline.generate(i, cache, hdmap=hdmap_chunk_i)
            pipeline.finalize(i, cache)
    """

    text_encoder: CosmosReason1TextEncoder
    image_encoder: WanVAEEncoder

    def __init__(self, config: AlpadreamsPipelineConfig) -> None:
        super().__init__(config)
        self.text_encoder = config.text_encoder.setup()
        self.image_encoder = config.image_encoder.setup()

        assert self.encoder is not None, (
            "AlpadreamsPipeline requires a per-AR-step HDMap encoder; "
            "set StreamInferencePipelineConfig.encoder."
        )

        transformer = self.diffusion_model.transformer
        self._len_t_latent: int = transformer.config.len_t
        # Pre-patchify number of latent frames committed per AR step, i.e.
        # ``len_t - kv_drop_t``. Used downstream to split the decoder input
        # window from the DiT query window when ``kv_drop_t > 0``.
        self._len_t_commit: int = transformer.config._len_t_commit
        self._kv_drop_t: int = transformer.config.kv_drop_t

        # FIXME: stateful HDMap encoders (Wan VAE) advance their conv
        # state by the number of pixel-frames they see per call. With
        # ``kv_drop_t > 0`` consecutive HDMap input windows OVERLAP by
        # ``kv_drop_t * temporal_compression`` frames (see ``get_num_input_frames``
        # vs ``get_num_output_frames``), and the encoder cache would
        # double-process the overlapping region, corrupting the
        # conditioning. PixelShuffle is stateless w.r.t. temporal context
        # and is safe. Until a proper fix lands (cache reset / reuse),
        # gate ``kv_drop_t > 0`` to PixelShuffle-encoder configs only.
        if self._kv_drop_t > 0 and isinstance(self.encoder, WanVAEEncoder):
            raise NotImplementedError(
                "kv_drop_t > 0 is not yet supported with a stateful Wan VAE "
                "HDMap encoder. Use a PixelShuffle encoder recipe (e.g. "
                "build_sv_2steps_chunk4_loc8_pshuffle_lighttae) or set "
                "kv_drop_t=0."
            )

        decoder = self.decoder
        assert decoder is not None and hasattr(decoder, "TEMPORAL_COMPRESSION_RATIO"), (
            f"Decoder {type(decoder).__name__} must expose "
            "TEMPORAL_COMPRESSION_RATIO (e.g. WanVAEDecoder, TeahvVAEDecoder)."
        )
        self._decoder_temporal_compression: int = decoder.TEMPORAL_COMPRESSION_RATIO

    @property
    def device(self) -> torch.device:
        return self.diffusion_model.device

    # ------------------------------------------------------------------
    # AR rollout API
    # ------------------------------------------------------------------

    @torch.no_grad()
    def initialize_cache(  # type: ignore[override]
        self,
        text: list[list[str]],
        image: Tensor,
        view_names: list[str] | None = None,
    ) -> AlpadreamsPipelineCache:
        """Initialize the per-rollout cache.

        Args:
            text: ``[B, V]`` nested list of prompts (one per view).
            image: First frame pixel tensor of shape ``[B, V, 1, 3, H, W]``
                in ``[-1, 1]``. ``H`` / ``W`` must match
                ``transformer.config.height/width *
                WanVAEDecoder.SPATIAL_COMPRESSION_RATIO``.
            view_names: List of view names (length ``V``); required when
                ``num_views > 1``.
        """
        assert isinstance(text, list) and len(text) > 0 and isinstance(text[0], list), (
            f"text must be a [B, V] nested list of prompts, got {type(text)}"
        )

        text_embeddings = torch.stack(
            [self.text_encoder(t) for t in text], dim=0
        )  # [B, V, L, D]
        image_embeddings = self.image_encoder(image)  # [B, V, 1, Cl, Hl, Wl]

        return super().initialize_cache(
            transformer_context={
                "text_embeddings": text_embeddings,
                "image_embeddings": image_embeddings,
                "view_names": view_names,
            },
        )

    @torch.no_grad()
    def generate(  # type: ignore[override]
        self,
        autoregressive_index: int,
        cache: AlpadreamsPipelineCache,
        hdmap: Tensor,
    ) -> Tensor:
        """Generate one decoded video chunk.

        Args:
            autoregressive_index: AR step index (``0``-based).
            cache: The per-rollout pipeline cache.
            hdmap: Per-AR-step HDMap pixel tensor of shape
                ``[B, V, T, 3, H, W]`` in ``[-1, 1]``. ``T`` must equal
                :meth:`get_num_frames(autoregressive_index)`.

        Returns:
            Decoded video chunk of shape ``[B, V, T, 3, H, W]`` in
            ``[-1, 1]``.
        """
        return super().generate(
            autoregressive_index=autoregressive_index,
            cache=cache,
            input=hdmap,
        )

    def get_num_input_frames(self, autoregressive_index: int) -> int:
        """Number of HDMap pixel frames the pipeline expects as input per AR step.

        The DiT always consumes ``len_t`` latent frames per AR step
        regardless of ``kv_drop_t`` (the trailing ``kv_drop_t`` latents
        are denoised and used as a transient self-attention tail before
        being discarded). So the per-AR-step HDMap input window is
        always sized by ``len_t``:

        - AR step 0: ``1 + (len_t - 1) * temporal_compression`` frames
          (anchor frame + denoised tail).
        - Steady-state: ``len_t * temporal_compression`` frames.

        Consecutive input windows overlap by
        ``kv_drop_t * temporal_compression`` frames when ``kv_drop_t > 0``;
        callers should advance their AR pointer using
        :meth:`get_num_output_frames` instead.
        """
        if autoregressive_index == 0:
            return 1 + (self._len_t_latent - 1) * self._decoder_temporal_compression
        return self._len_t_latent * self._decoder_temporal_compression

    def get_num_output_frames(self, autoregressive_index: int) -> int:
        """Number of decoded video frames emitted per AR step (= AR pointer advance).

        Sized by ``_len_t_commit = len_t - kv_drop_t`` because only the
        committed prefix of each AR step's clean latent reaches the
        streaming decoder:

        - AR step 0: ``1 + (_len_t_commit - 1) * temporal_compression``
          frames (anchor frame + committed tail).
        - Steady-state: ``_len_t_commit * temporal_compression`` frames.

        Equals :meth:`get_num_input_frames` when ``kv_drop_t == 0``.
        """
        if autoregressive_index == 0:
            return 1 + (self._len_t_commit - 1) * self._decoder_temporal_compression
        return self._len_t_commit * self._decoder_temporal_compression

    # Back-compat alias: most callers want the output count (it doubles
    # as the AR pointer advance). Pre-``kv_drop_t`` callers wrote loops
    # like ``end = start + get_num_frames(i); ...; start = end`` which
    # only stays correct when ``kv_drop_t == 0``; with the experiment
    # turned on, callers should split into
    # ``get_num_input_frames`` / ``get_num_output_frames`` (see
    # ``examples/run_alpadreams.py``).
    def get_num_frames(self, autoregressive_index: int) -> int:
        return self.get_num_output_frames(autoregressive_index)

    def _pre_decode_hook(
        self,
        clean_latent: Tensor,
        autoregressive_index: int,
    ) -> Tensor:
        """Slice the clean latent to the committed prefix before the decoder.

        ``DiffusionModel.generate`` returns the unpatchified clean latent
        for all ``len_t`` denoised positions, but with ``kv_drop_t > 0``
        only the first ``_len_t_commit`` are committed to the decoder.
        Streaming decoders (TAEHV, Wan VAE) advance their conv-state cache
        by however many latent frames they see per call, so we must drop
        the trailing ``kv_drop_t`` frames here -- otherwise the decoder
        time index would race past the AR pointer by ``kv_drop_t`` per
        step.

        For ``kv_drop_t == 0`` this is a no-op (slice covers the full T).
        """
        del autoregressive_index  # decoder slicing is uniform across steps
        if self._len_t_commit == self._len_t_latent:
            return clean_latent
        # Latent layout from ``unpatchify_and_maybe_gather_cp``: [B, V, T, C, H, W].
        return clean_latent[:, :, : self._len_t_commit]
