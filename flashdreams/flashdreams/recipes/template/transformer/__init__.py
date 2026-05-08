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

"""Reference :class:`Transformer` subclass for the template recipe."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from einops import rearrange
from torch import Tensor

from flashdreams.core.attention.rope import RotaryPositionEmbedding3D
from flashdreams.core.checkpoint.load import load_checkpoint
from flashdreams.core.distributed.context_parallel import (
    cat_outputs_cp,
    split_inputs_cp,
)
from flashdreams.infra.compile import compile_module
from flashdreams.infra.config import InstantiateConfig
from flashdreams.infra.cuda_graph import CUDAGraphWrapper
from flashdreams.infra.diffusion.transformer import (
    Transformer,
    TransformerAutoregressiveCache,
)
from flashdreams.infra.encoder import Encoder, NullEncoderConfig

from .network import TemplateDiT, TemplateDiTCache, TemplateDiTConfig


@dataclass(kw_only=True)
class TemplateTransformerCache(TransformerAutoregressiveCache):
    """Long-lived AR cache for :class:`TemplateTransformer`.

    Cond and uncond branches own independent KV buffers: the residual
    stream diverges at the first context-bias addition.
    """

    network_cache: TemplateDiTCache
    """Conditional per-block KV cache and conditional context tokens."""

    network_cache_uncond: TemplateDiTCache | None = None
    """Unconditional cache. ``None`` disables CFG."""

    rope_adapter: RotaryPositionEmbedding3D
    """3D RoPE adapter, advanced via ``shift_t`` each step."""

    rope_freqs: Tensor | None = None
    """Self-attention RoPE frequencies, refreshed each AR step by
    :meth:`start` via :meth:`RotaryPositionEmbedding3D.shift_t`.

    Shape ``[L_per_chunk / cp_size, 1, 1, d // 2]`` after
    :meth:`RotaryPositionEmbedding3D.set_context_parallel_group` has
    sharded the table along the sequence axis (``cp_size == 1`` when CP
    is disabled, so the shape collapses to ``[L_per_chunk, 1, 1, d //
    2]``)."""

    autoregressive_index: int = -1
    """AR step index for the chunk currently being processed; ``-1``
    before the first :meth:`start` call."""

    def start(self, autoregressive_index: int) -> None:
        """Snapshot the AR index and run per-block pre-update hooks.

        Hoisting ``before_update`` out of the network forward keeps the
        captured region shape-stable across AR steps.
        """
        self.rope_freqs = self.rope_adapter.shift_t(autoregressive_index)

        self.autoregressive_index = autoregressive_index
        self.network_cache.before_update(autoregressive_index)
        if self.network_cache_uncond is not None:
            self.network_cache_uncond.before_update(autoregressive_index)

    def finalize(self, autoregressive_index: int) -> None:
        """Run per-block post-update hooks."""
        self.network_cache.after_update(autoregressive_index)
        if self.network_cache_uncond is not None:
            self.network_cache_uncond.after_update(autoregressive_index)


@dataclass(kw_only=True)
class TemplateTransformerConfig(InstantiateConfig["TemplateTransformer"]):
    """Config for the template transformer.

    Bakes in the temporal layout (``len_t``, ``window_size_t``,
    ``sink_size_t``) and the CFG / compile knobs. Per-rollout spatial
    layout (``batch_size``, ``height``, ``width``) is supplied to
    :meth:`TemplateTransformer.initialize_autoregressive_cache` so one
    instance can serve multiple resolutions. CP size auto-detects from
    ``torch.distributed.get_world_size()`` at construction.
    """

    _target: type["TemplateTransformer"] = field(
        default_factory=lambda: TemplateTransformer
    )

    network: TemplateDiTConfig = field(default_factory=TemplateDiTConfig)
    """Underlying DiT network config."""

    patch_size: tuple[int, int, int] = (1, 1, 1)
    """3D ``(kt, kh, kw)`` patch size folded into the token-channel dim.
    :attr:`TemplateDiTConfig.in_channels` must equal
    ``raw_channels * prod(patch_size)``. Defaults to ``(1, 1, 1)`` (no
    packing) so the bare config is self-consistent with
    :attr:`TemplateDiTConfig`'s default ``in_channels=4``; builders
    that want patch packing must set both fields together."""

    context_encoder: InstantiateConfig[Any] = field(default_factory=NullEncoderConfig)
    """One-shot encoder applied to raw ``context`` inside
    :meth:`TemplateTransformer.initialize_autoregressive_cache`. The
    default :class:`~flashdreams.infra.encoder.NullEncoder` is identity;
    swap in a text or CLIP image encoder here."""

    # rope adapter config
    h_extrapolation_ratio: float = 3.0
    """RoPE extrapolation along H. Default to 3.0 for 720p."""
    w_extrapolation_ratio: float = 3.0
    """RoPE extrapolation along W. Default to 3.0 for 720p."""

    dtype: torch.dtype = torch.bfloat16
    """Parameter and activation dtype."""

    checkpoint_path: str | None = None
    """Network checkpoint path for
    :func:`flashdreams.core.checkpoint.load.load_checkpoint`. ``None``
    keeps the random init."""

    len_t: int = 2
    """Per-AR-chunk temporal length, in **pre-patchify** latent frames.
    Must be divisible by ``patch_size[0]``."""

    window_size_t: int = 4
    """Sliding-window length, in **pre-patchify** latent frames. Must be
    divisible by ``patch_size[0]``."""

    sink_size_t: int = 0
    """Sink length, in **pre-patchify** latent frames. Must be
    divisible by ``patch_size[0]``."""

    guidance_scale: float = 1.0
    """CFG scale. ``1.0`` disables CFG; ``> 1.0`` requires a
    ``negative_context`` at cache build time."""

    compile_network: bool = False
    """Compile the network via :func:`flashdreams.infra.compile.compile_module`."""

    use_cuda_graph: bool = False
    """Wrap the network in :class:`CUDAGraphWrapper` for steady-state
    replay. The wrapper is built lazily inside
    :meth:`TemplateTransformer.initialize_autoregressive_cache` so each
    rollout gets static buffers sized to its ``(height, width)``."""

    cuda_graph_warmup_iters: int = 2
    """Eager calls before CUDA-graph capture; see :class:`CUDAGraphWrapper`."""

    @property
    def requires_negative_context_embeddings(self) -> bool:
        """``True`` when CFG is on (``guidance_scale > 1.0``)."""
        return self.guidance_scale > 1.0


class TemplateTransformer(Transformer[TemplateTransformerCache]):
    """Reference :class:`Transformer` subclass used by the template recipe."""

    config: TemplateTransformerConfig
    network: TemplateDiT
    context_encoder: Encoder

    def __init__(self, config: TemplateTransformerConfig) -> None:
        super().__init__(config)
        self.config = config

        if torch.distributed.is_initialized():
            self._cp_size = torch.distributed.get_world_size()
            self._cp_group = (
                torch.distributed.group.WORLD if self._cp_size > 1 else None
            )
        else:
            self._cp_size = 1
            self._cp_group = None

        self.network = config.network.setup()
        self.network = self.network.to(dtype=config.dtype)
        self.network.eval()
        self.network.set_context_parallel_group(cp_group=self._cp_group)

        if config.checkpoint_path is not None:
            state_dict = load_checkpoint(config.checkpoint_path)
            self.network.load_state_dict(state_dict)

        if config.compile_network:
            self.network = compile_module(self.network)

        self.context_encoder = config.context_encoder.setup()

        self._batch_size: int | None = None
        self._output_height: int | None = None
        self._output_width: int | None = None

        self._use_cuda_graph = config.use_cuda_graph
        self._network_call: CUDAGraphWrapper | None = None
        self._network_call_uncond: CUDAGraphWrapper | None = None
        self._cuda_graph_capture_ar_idx: int = 0

    @property
    def latent_shape(self) -> tuple[int, ...]:
        """Per-rank post-patchify latent shape ``[B, L/cp, in_channels]``.

        Populated by :meth:`initialize_autoregressive_cache`; reading
        it earlier asserts.
        """
        assert (
            self._batch_size is not None
            and self._output_height is not None
            and self._output_width is not None
        ), (
            "latent_shape requires an initialized rollout; call "
            "initialize_autoregressive_cache(..., height=..., width=...) "
            "first."
        )
        cfg = self.config
        kt, kh, kw = cfg.patch_size
        L = (cfg.len_t // kt) * (self._output_height // kh) * (self._output_width // kw)
        return (self._batch_size, L // self._cp_size, cfg.network.in_channels)

    def patchify_and_maybe_split_cp(self, x: Tensor) -> Tensor:
        """Pack 3D patches into tokens and shard the token axis across CP ranks.

        Each ``(kt, kh, kw)`` cube becomes one token, with its voxels
        flattened into the channel dim. The resulting token sequence is
        then split evenly across CP ranks (no-op when CP is disabled).

        Args:
            x: Pre-patchify latent ``[B, C, T, H, W]``. ``T``, ``H`` and
                ``W`` must each be divisible by the matching
                ``patch_size`` entry.

        Returns:
            Per-rank token tensor ``[B, L/cp, C']`` where each token
            carries one packed patch.
        """
        assert x.ndim == 5, f"Expected [B, C, T, H, W], got {tuple(x.shape)}."
        _, _, T, H, W = x.shape
        kt, kh, kw = self.config.patch_size
        assert T % kt == 0 and H % kh == 0 and W % kw == 0, (
            f"(T, H, W) = ({T}, {H}, {W}) must be divisible by "
            f"patch_size={(kt, kh, kw)}."
        )
        x = rearrange(
            x,
            "b c (pT kt) (pH kh) (pW kw) -> b (pT pH pW) (c kt kh kw)",
            kt=kt,
            kh=kh,
            kw=kw,
        )
        return split_inputs_cp(x, seq_dim=1, cp_group=self._cp_group)

    def unpatchify_and_maybe_gather_cp(self, x: Tensor) -> Tensor:
        """Gather token shards across CP ranks and unpack patches back to voxels.

        Inverse of :meth:`patchify_and_maybe_split_cp`. Requires the
        per-rollout ``(height, width)``, so it asserts if called before
        :meth:`initialize_autoregressive_cache`.

        Args:
            x: Per-rank token tensor ``[B, L/cp, C']``.

        Returns:
            Pre-patchify latent ``[B, C, T, H, W]``.
        """
        assert self._output_height is not None and self._output_width is not None, (
            "unpatchify_and_maybe_gather_cp requires an initialized "
            "rollout; call initialize_autoregressive_cache(..., "
            "height=..., width=...) first."
        )
        cfg = self.config
        kt, kh, kw = cfg.patch_size

        x = cat_outputs_cp(x, seq_dim=1, cp_group=self._cp_group)
        return rearrange(
            x,
            "b (pT pH pW) (c kt kh kw) -> b c (pT kt) (pH kh) (pW kw)",
            pT=cfg.len_t // kt,
            pH=self._output_height // kh,
            pW=self._output_width // kw,
            kt=kt,
            kh=kh,
            kw=kw,
        )

    def initialize_autoregressive_cache(
        self,
        *,
        height: int,
        width: int,
        context: Tensor,
        negative_context: Tensor | None = None,
    ) -> TemplateTransformerCache:
        """Build a fully seeded cache for a new rollout.

        Runs ``context`` (and ``negative_context`` when CFG is on)
        through :attr:`context_encoder`, stashes the per-rollout
        ``(batch_size, height, width)``, and — when
        ``config.use_cuda_graph`` is set — builds fresh
        :class:`CUDAGraphWrapper` instances sized to this rollout.

        Args:
            height: Pre-patchify latent height.
            width: Pre-patchify latent width.
            context: Raw conditional context passed through
                :attr:`context_encoder`. The leading dim defines the
                rollout's ``batch_size``.
            negative_context: Raw unconditional context. Required iff
                ``config.guidance_scale > 1.0``.

        Returns:
            Seeded :class:`TemplateTransformerCache`.
        """
        cfg = self.config
        context_embeddings = self.context_encoder(input=context)
        batch_size, _, _ = context_embeddings.shape

        kt, kh, kw = cfg.patch_size
        assert cfg.len_t % kt == 0 and height % kh == 0 and width % kw == 0, (
            f"(len_t, height, width) = ({cfg.len_t}, {height}, {width}) "
            f"must be divisible by patch_size={cfg.patch_size}."
        )
        assert cfg.window_size_t % kt == 0 and cfg.sink_size_t % kt == 0, (
            f"(window_size_t, sink_size_t) = "
            f"({cfg.window_size_t}, {cfg.sink_size_t}) must be divisible "
            f"by patch_size[0]={kt}; otherwise the // kt truncation below "
            f"silently mis-sizes the KV cache and breaks window/sink "
            f"semantics."
        )
        pHW = (height // kh) * (width // kw)
        chunk_tokens = cfg.len_t // kt * pHW
        window_tokens = cfg.window_size_t // kt * pHW
        sink_tokens = cfg.sink_size_t // kt * pHW
        assert chunk_tokens % self._cp_size == 0, (
            f"per-chunk token count {chunk_tokens} (= len_t/kt * pH * pW) "
            f"must be divisible by cp_size={self._cp_size}; otherwise "
            f"split_inputs_cp would truncate. Pad len_t / height / width."
        )
        chunk_size = chunk_tokens // self._cp_size
        window_size = window_tokens // self._cp_size
        sink_size = sink_tokens // self._cp_size

        device = context_embeddings.device
        dtype = cfg.dtype

        network_cache = self.network.initialize_cache(
            chunk_size=chunk_size,
            window_size=window_size,
            sink_size=sink_size,
            context=context_embeddings,
            batch_size=batch_size,
            device=device,
            dtype=dtype,
        )

        network_cache_uncond: TemplateDiTCache | None = None
        if cfg.requires_negative_context_embeddings:
            assert negative_context is not None, (
                f"guidance_scale={cfg.guidance_scale} > 1.0 requires "
                "negative_context_embeddings."
            )
            negative_context_embeddings = self.context_encoder(input=negative_context)
            network_cache_uncond = self.network.initialize_cache(
                chunk_size=chunk_size,
                window_size=window_size,
                sink_size=sink_size,
                context=negative_context_embeddings,
                batch_size=batch_size,
                device=device,
                dtype=dtype,
            )

        # initialize the rope adapter
        rope_adapter = RotaryPositionEmbedding3D(
            len_t=cfg.len_t // kt,
            len_h=height // kh,
            len_w=width // kw,
            head_dim=cfg.network.model_channels // cfg.network.num_heads,
            h_extrapolation_ratio=cfg.h_extrapolation_ratio,
            w_extrapolation_ratio=cfg.w_extrapolation_ratio,
            device=device,
        )
        rope_adapter.set_context_parallel_group(cp_group=self._cp_group)

        self._batch_size = batch_size
        self._output_height = height
        self._output_width = width

        # One wrapper per rollout: static buffers and the captured
        # graph are bound to this cache's KV pointers. The dispatch
        # threshold matches the KV cache's filling → steady transition
        # so the captured region only sees steady-state paths.
        if self._use_cuda_graph:
            self._cuda_graph_capture_ar_idx = (
                cfg.sink_size_t + cfg.window_size_t
            ) // cfg.len_t
            self._network_call = CUDAGraphWrapper(
                self.network, warmup_iters=cfg.cuda_graph_warmup_iters
            )
            self._network_call_uncond = CUDAGraphWrapper(
                self.network, warmup_iters=cfg.cuda_graph_warmup_iters
            )

        return TemplateTransformerCache(
            network_cache=network_cache,
            network_cache_uncond=network_cache_uncond,
            rope_adapter=rope_adapter,
        )

    def _select_network(self, autoregressive_index: int, *, uncond: bool) -> Any:
        # Filling phase: eager ``.drain`` (drains Inductor autotune and
        # exercises the KV cache's slice-returning filling path).
        # Steady phase: ``wrapper.__call__`` (warmup + capture + replay).
        # Capturing in the filling phase would bake slice pointers into
        # the graph and read stale storage after the cache rolls.
        if not self._use_cuda_graph:
            return self.network
        network_call = self._network_call_uncond if uncond else self._network_call
        assert isinstance(network_call, CUDAGraphWrapper), (
            "predict_flow called before initialize_autoregressive_cache "
            "while use_cuda_graph=True."
        )
        return (
            network_call.drain
            if autoregressive_index < self._cuda_graph_capture_ar_idx
            else network_call
        )

    def _predict_flow(
        self,
        noisy_latent: Tensor,
        timestep: Tensor,
        cache: TemplateTransformerCache,
        control: Tensor | None,
        *,
        uncond: bool,
    ) -> Tensor:
        autoregressive_index = cache.autoregressive_index
        assert autoregressive_index >= 0, (
            "Cache.start(autoregressive_index) must be called before "
            "predict_flow (DiffusionModel.generate handles this)."
        )
        network_cache = cache.network_cache_uncond if uncond else cache.network_cache
        assert network_cache is not None, (
            "uncond=True requires cache.network_cache_uncond, but it is None "
            "(CFG was not enabled at cache build time)."
        )
        return self._select_network(autoregressive_index, uncond=uncond)(
            noisy_latent,
            timesteps=timestep,
            cache=network_cache,
            rope_freqs=cache.rope_freqs,
            control=control,
        )

    def predict_flow(
        self,
        noisy_latent: Tensor,
        timestep: Tensor,
        cache: TemplateTransformerCache,
        input: Tensor | None = None,
    ) -> Tensor:
        """Run the cond branch and merge in the uncond branch when CFG is on.

        Args:
            noisy_latent: ``[B, L/cp, in_channels]`` per-rank noisy latent.
            timestep: Scalar timestep.
            cache: Per-rollout cache; ``cache.autoregressive_index`` must
                be set by a prior :meth:`TemplateTransformerCache.start`.
            input: Encoded control latent, or ``None`` to skip the
                per-token control bias.

        Returns:
            ``[B, L/cp, in_channels]`` flow prediction with CFG applied
            when ``cache.network_cache_uncond`` is populated.
        """
        flow_cond = self._predict_flow(
            noisy_latent=noisy_latent,
            timestep=timestep,
            cache=cache,
            control=input,
            uncond=False,
        )
        if cache.network_cache_uncond is None:
            return flow_cond
        else:
            flow_uncond = self._predict_flow(
                noisy_latent=noisy_latent,
                timestep=timestep,
                cache=cache,
                control=input,
                uncond=True,
            )
            return flow_uncond + self.config.guidance_scale * (flow_cond - flow_uncond)
