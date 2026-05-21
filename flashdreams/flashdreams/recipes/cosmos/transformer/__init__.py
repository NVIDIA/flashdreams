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

"""Single-view Cosmos DiT for streaming Cosmos-Predict2 inference."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

from flashdreams.core.attention.rope import (
    RotaryPositionEmbedding3D,
)
from flashdreams.core.checkpoint.load import load_checkpoint
from flashdreams.infra.compile import compile_module
from flashdreams.infra.cuda_graph import CUDAGraphWrapper
from flashdreams.infra.diffusion.transformer import (
    Transformer,
    TransformerAutoregressiveCache,
    TransformerConfig,
)

from .impl.network import (
    CosmosDiTNetwork,
    CosmosDiTNetworkCache,
    CosmosDiTNetworkConfig,
)


@dataclass(kw_only=True)
class CosmosTransformerCache(TransformerAutoregressiveCache):
    """Long-lived AR cache for the Cosmos transformer."""

    network_cache: CosmosDiTNetworkCache
    """Per-block self-attn KV + (text-only) cross-attn KV."""

    network_cache_uncond: CosmosDiTNetworkCache | None = None
    """Unconditional cache for CFG; ``None`` disables CFG."""

    rope_adapter: RotaryPositionEmbedding3D
    """3D RoPE adapter, advanced via ``shift_t`` each step."""

    rope_freqs: Tensor | None = None
    """Self-attention RoPE frequencies for the current AR step.
    Shape ``[L, 1, 1, head_dim // 2]`` after CP. Recomputed once per
    AR step in :meth:`start` and reused across cond and uncond branches
    (and across all scheduler steps within the AR step)."""

    image: Tensor | None
    """First-frame VAE latent, T-padded to ``len_t`` and patchified.
    Injects into the noisy / predicted latent at AR step 0."""

    mask_first_block: Tensor
    """``[B, T, 1, H, W]`` mask with ones on the first temporal latent
    frame; used at AR step 0."""

    mask_other_blocks: Tensor
    """All-zero counterpart used at AR step >= 1."""

    autoregressive_index: int = -1
    """AR step index for the chunk currently being processed; ``-1`` before the first ``start``."""

    def start(self, autoregressive_index: int) -> None:
        # Hoist per-block KV pre-update and the RoPE shift out of the
        # (graph-captured) network forward. ``predict_flow`` runs with
        # ``eager_mode=False``; the cond/uncond passes share ``rope_freqs``.
        self.rope_freqs = self.rope_adapter.shift_t(autoregressive_index)

        self.autoregressive_index = autoregressive_index
        self.network_cache.before_update(autoregressive_index)
        if self.network_cache_uncond is not None:
            self.network_cache_uncond.before_update(autoregressive_index)

    def finalize(self, autoregressive_index: int) -> None:
        self.network_cache.after_update(autoregressive_index)
        if self.network_cache_uncond is not None:
            self.network_cache_uncond.after_update(autoregressive_index)


@dataclass(kw_only=True)
class CosmosTransformerConfig(TransformerConfig):
    """Config for the Cosmos transformer.

    Bakes in the temporal layout (``len_t``, ``window_size_t``,
    ``sink_size_t``) and CFG / compile knobs. Per-rollout spatial
    layout (``height``, ``width``) is supplied to
    :meth:`CosmosTransformer.initialize_autoregressive_cache` so one
    instance can serve multiple resolutions. CP size is auto-detected
    from ``torch.distributed.get_world_size()`` at construction; build
    the pipeline under a ``torch.distributed`` initialization with the
    desired world size to opt in.
    """

    _target: type["CosmosTransformer"] = field(
        default_factory=lambda: CosmosTransformer
    )

    network: CosmosDiTNetworkConfig = field(default_factory=CosmosDiTNetworkConfig)
    """Backbone Cosmos DiT network config."""

    dtype: torch.dtype = torch.bfloat16
    """Network parameter / activation dtype."""

    checkpoint_path: str | None = None
    """Optional path to a pretrained checkpoint; ``None`` keeps the random init."""

    state_dict_transform: Callable[[dict[str, Tensor]], dict[str, Tensor]] | None = None
    """Pre-load state-dict remap. Defaults to a ``net.`` prefix stripper."""

    batch_shape: tuple[int, ...] = (1,)
    """Batch dims of the latent (excluding ``T, HW, D``)."""

    len_t: int = 21
    """Latent frames per AR chunk."""

    h_extrapolation_ratio: float = 3.0
    """RoPE extrapolation along H (3.0 @ 720p)."""

    w_extrapolation_ratio: float = 3.0
    """RoPE extrapolation along W."""

    window_size_t: int = 8
    """Self-attention sliding window (pre-patchify T)."""

    sink_size_t: int = 0
    """Sink-token count (pre-patchify T)."""

    compile_network: bool = True
    """``torch.compile`` the network."""

    use_cuda_graph: bool = True
    """Wrap in ``CUDAGraphWrapper`` for steady-state replay. Caller must
    keep non-staged inputs at stable storage addresses across calls."""

    cuda_graph_warmup_iters: int = 2
    """Eager calls before capture (>= 2 to drain Inductor autotune)."""

    skip_finalize_kv_cache: bool = False
    """Skip the KV cache finalize step."""

    guidance_scale: float = 1.0
    """CFG scale. ``1.0`` disables CFG; ``> 1.0`` requires negative text embeddings."""

    conditional_frame_timestep: float | None = None
    """Scheduler-scale timestep fed to the network at the conditional frame.
    ``None`` disables the override.
    """

    @property
    def requires_negative_text_embeddings(self) -> bool:
        """Whether cache initialization must receive negative text embeddings."""
        return self.guidance_scale > 1.0


class CosmosTransformer(Transformer[CosmosTransformerCache]):
    """Cosmos DiT as an infra transformer."""

    config: CosmosTransformerConfig
    network: CosmosDiTNetwork

    def __init__(self, config: CosmosTransformerConfig) -> None:
        super().__init__(config)
        self.config = config

        # Auto-detect CP world size from torch.distributed; non-distributed
        # mode short-circuits to a singleton group set.
        if torch.distributed.is_initialized():
            self._cp_size = torch.distributed.get_world_size()
            self._cp_group = (
                torch.distributed.group.WORLD if self._cp_size > 1 else None
            )
        else:
            self._cp_size = 1
            self._cp_group = None

        # Pre-patchify temporal divisibility check; per-rollout
        # (height, width) is populated by initialize_autoregressive_cache.
        kt, _, _ = config.network.patch_size
        assert config.len_t % kt == 0, (
            f"len_t ({config.len_t}) must be divisible by patch_temporal ({kt})."
        )
        self._output_height: int | None = None
        self._output_width: int | None = None

        self.network = CosmosDiTNetwork(config=config.network)
        self.network = self.network.to(dtype=config.dtype)
        self.network.eval()
        self.network.set_context_parallel_group(cp_group=self._cp_group)

        if config.checkpoint_path is not None:
            state_dict = load_checkpoint(config.checkpoint_path)
            if config.state_dict_transform is not None:
                state_dict = config.state_dict_transform(state_dict)
            self.network.load_state_dict(state_dict)
        self.network.update_parameters_after_loading_checkpoint()

        if config.compile_network:
            self.network = compile_module(self.network)

        # Per-rollout dispatch when use_cuda_graph=True:
        # filling phase -> wrapper.drain (eager, drains Inductor autotune);
        # steady-state -> wrapper.__call__ (warmup + capture + replay).
        # First AR step that runs on the KV cache's steady-state code
        # path. The cache fills at AR step ``chunks_total // len_t - 1``;
        # the *next* step is the first one whose ``before_update`` sees
        # ``is_steady_state() == True`` and whose forward takes the steady
        # branches.
        self._use_cuda_graph = config.use_cuda_graph
        chunks_total = config.sink_size_t + config.window_size_t
        assert chunks_total % config.len_t == 0, (
            f"sink_size_t + window_size_t ({chunks_total}) must be "
            f"divisible by len_t ({config.len_t}) so the BlockKVCache can "
            f"fit a whole number of AR chunks."
        )
        self._cuda_graph_capture_ar_idx: int = chunks_total // config.len_t
        self._network_call: CUDAGraphWrapper | CosmosDiTNetwork = (
            CUDAGraphWrapper(self.network, warmup_iters=config.cuda_graph_warmup_iters)
            if config.use_cuda_graph
            else self.network
        )
        self._network_call_uncond: CUDAGraphWrapper | CosmosDiTNetwork = (
            CUDAGraphWrapper(self.network, warmup_iters=config.cuda_graph_warmup_iters)
            if config.use_cuda_graph
            else self.network
        )

    @property
    def latent_shape(self) -> tuple[int, ...]:
        """Per-rank latent shape ``[..., L, D]``.

        Per-rollout ``(height, width)`` is populated by
        :meth:`initialize_autoregressive_cache`; reading earlier asserts.
        """
        assert self._output_height is not None and self._output_width is not None, (
            "latent_shape requires an initialized rollout; call "
            "initialize_autoregressive_cache(..., height=..., width=...) first."
        )
        cfg = self.config
        kt, kh, kw = cfg.network.patch_size
        L = (cfg.len_t // kt) * (self._output_height // kh) * (self._output_width // kw)
        return (
            *cfg.batch_shape,
            L // self._cp_size,
            cfg.network.out_channels * kt * kh * kw,
        )

    @torch.no_grad()
    def _build_network_cache(
        self,
        *,
        text_embeddings: Tensor,
    ) -> CosmosDiTNetworkCache:
        """Build one network cache (cond or uncond branch).

        Caller must have populated ``self._output_height/_output_width``
        (done by :meth:`initialize_autoregressive_cache`) before invoking
        this.
        """
        assert self._output_height is not None and self._output_width is not None, (
            "_build_network_cache called before height/width were stashed."
        )
        cfg = self.config
        kt, kh, kw = cfg.network.patch_size
        pHW = (self._output_height // kh) * (self._output_width // kw)
        cp_size = self._cp_size
        chunk_size = self.latent_shape[-2]  # already CP-divided
        window_size = (cfg.window_size_t // kt * pHW) // cp_size
        sink_size = (cfg.sink_size_t // kt * pHW) // cp_size
        return self.network.initialize_cache(
            chunk_size=chunk_size,
            window_size=window_size,
            sink_size=sink_size,
            text_embeddings=text_embeddings,
        )

    @torch.no_grad()
    def initialize_autoregressive_cache(
        self,
        *,
        height: int,
        width: int,
        text_embeddings: Tensor,
        image_embeddings: Tensor | None = None,
        negative_text_embeddings: Tensor | None = None,
        **_unused: Any,
    ) -> CosmosTransformerCache:
        """Build a fully seeded cache for a new rollout.

        Args:
            height: Pre-patchify latent height (post-VAE).
            width: Pre-patchify latent width (post-VAE).
            text_embeddings: ``[..., L, D]`` text embeddings.
            image_embeddings: ``[..., 1, C, H, W]`` first-frame VAE latent.
                ``H``/``W`` must equal ``height``/``width``.
        """
        # Stash the per-rollout spatial layout. ``latent_shape``,
        # ``unpatchify_and_maybe_gather_cp`` and the network-cache /
        # RoPE setup below all read these.
        cfg = self.config
        kt, kh, kw = cfg.network.patch_size
        assert height % kh == 0 and width % kw == 0, (
            f"(height, width) = ({height}, {width}) must be divisible by "
            f"patch_spatial ({kh})."
        )
        self._output_height = height
        self._output_width = width
        total_tokens = (cfg.len_t // kt) * (height // kh) * (width // kw)
        assert total_tokens % self._cp_size == 0, (
            f"Cosmos token length ({total_tokens} from len_t={cfg.len_t}, "
            f"height={height}, width={width}, "
            f"patch_temporal={kt}, patch_spatial={kh}) must be divisible by "
            f"cp_size={self._cp_size}"
        )

        network_cache = self._build_network_cache(text_embeddings=text_embeddings)
        network_cache_uncond: CosmosDiTNetworkCache | None = None
        if cfg.requires_negative_text_embeddings:
            assert negative_text_embeddings is not None, (
                f"{type(cfg).__name__}.guidance_scale={cfg.guidance_scale} > 1.0 "
                "requires negative_text_embeddings."
            )
            network_cache_uncond = self._build_network_cache(
                text_embeddings=negative_text_embeddings,
            )

        head_dim = cfg.network.model_channels // cfg.network.num_heads
        rope_adapter = RotaryPositionEmbedding3D(
            len_t=cfg.len_t // kt,
            len_h=height // kh,
            len_w=width // kw,
            head_dim=head_dim,
            h_extrapolation_ratio=cfg.h_extrapolation_ratio,
            w_extrapolation_ratio=cfg.w_extrapolation_ratio,
            device=self.device,
        )
        rope_adapter.set_context_parallel_group(cp_group=self._cp_group)

        # for I2V
        if image_embeddings is not None:
            H, W = image_embeddings.shape[-2:]
            assert H == height and W == width, (
                f"image_embeddings spatial dims ({H}, {W}) must match "
                f"(height, width) = ({height}, {width})."
            )
            # Pad first-frame image latent along T (zeros for steady state).
            image = F.pad(image_embeddings, (0, 0, 0, 0, 0, 0, 0, cfg.len_t - 1))
            image_patched = self.patchify_and_maybe_split_cp(image)
        else:
            image_patched = None

        mask_first_block = torch.zeros(
            *cfg.batch_shape,
            cfg.len_t,
            1,
            height,
            width,
            device=self.device,
            dtype=cfg.dtype,
        )
        mask_first_block[..., :1, :, :, :] = 1.0
        mask_other_blocks = torch.zeros(
            *cfg.batch_shape,
            cfg.len_t,
            1,
            height,
            width,
            device=self.device,
            dtype=cfg.dtype,
        )

        # Patchify masks once at rollout start (image was patchified above).
        mask_first_patched = self.patchify_and_maybe_split_cp(mask_first_block)
        mask_other_patched = self.patchify_and_maybe_split_cp(mask_other_blocks)

        # Reset any prior CUDA graph: it refers to slot pointers from the
        # previous cache, which the new cache invalidates.
        if self._use_cuda_graph:
            assert isinstance(self._network_call, CUDAGraphWrapper)
            self._network_call.reset()
            assert isinstance(self._network_call_uncond, CUDAGraphWrapper)
            self._network_call_uncond.reset()

        return CosmosTransformerCache(
            network_cache=network_cache,
            network_cache_uncond=network_cache_uncond,
            rope_adapter=rope_adapter,
            image=image_patched,
            mask_first_block=mask_first_patched,
            mask_other_blocks=mask_other_patched,
        )

    def _maybe_inject_image(
        self,
        latent: Tensor,
        cache: CosmosTransformerCache,
    ) -> Tensor:
        """Replace the first-temporal-frame latent with the encoded image at AR step 0."""
        if cache.image is None or cache.autoregressive_index != 0:
            return latent
        mask = cache.mask_first_block[..., :1]
        return latent * (1.0 - mask) + cache.image * mask

    def _select_mask(self, cache: CosmosTransformerCache) -> Tensor:
        if cache.image is None or cache.autoregressive_index != 0:
            return cache.mask_other_blocks
        return cache.mask_first_block

    def _select_network(self, autoregressive_index: int, *, uncond: bool) -> Any:
        # Filling phase: eager ``.drain`` (drains Inductor autotune and
        # exercises the KV cache's slice-returning filling path).
        # Steady phase: ``wrapper.__call__`` (warmup + capture + replay).
        # Cond and CFG-uncond branches both mutate their rolling KV cache,
        # so neither branch can be graph-captured until the cache is steady.
        if not self._use_cuda_graph:
            return self.network
        network_call = self._network_call_uncond if uncond else self._network_call
        assert isinstance(network_call, CUDAGraphWrapper)
        return (
            network_call.drain
            if autoregressive_index < self._cuda_graph_capture_ar_idx
            else network_call
        )

    def _build_per_token_timesteps(
        self, timestep: Tensor, cache: CosmosTransformerCache
    ) -> Tensor:
        """Return scalar (T2V/non-AR0) or per-token (I2V at AR=0) timesteps.

        For the cosmos 2.5 2B base post-trained checkpoint the conditional
        frame must be fed at ``conditional_frame_timestep`` (~0.1, near-clean)
        regardless of the current denoising step, because that is how the
        per-frame adaLN was trained. Other frames keep ``timestep``.

        Note: per-token timesteps is not the most efficient way (should be per-frame).
        But it is per-token timesteps is convenient to implement.
        """
        cft = self.config.conditional_frame_timestep
        if (
            cache.image is None
            or cft is None
            or cft < 0.0
            or cache.autoregressive_index != 0
        ):
            return timestep
        # mask_first_block is patchified ``[..., L, D]``.
        override_mask = cache.mask_first_block[..., 0]
        override_timestep = timestep.new_tensor(cft)
        return override_timestep * override_mask + timestep * (1.0 - override_mask)

    def _predict_flow(
        self,
        noisy_latent: Tensor,
        timestep: Tensor,
        cache: CosmosTransformerCache,
        network_cache: CosmosDiTNetworkCache,
        input: Tensor | None,
        *,
        uncond: bool,
    ) -> Tensor:
        ar_idx = cache.autoregressive_index
        assert ar_idx >= 0 and cache.rope_freqs is not None, (
            "Cache.start(autoregressive_index) must be called before "
            "predict_flow (DiffusionModel.generate handles this)."
        )
        noisy_latent = self._maybe_inject_image(noisy_latent, cache)
        mask = self._select_mask(cache)
        timesteps = self._build_per_token_timesteps(timestep, cache)
        return self._select_network(ar_idx, uncond=uncond)(
            x=torch.cat([noisy_latent, mask], dim=-1),
            timesteps=timesteps,
            rope_freqs=cache.rope_freqs,
            cache=network_cache,
            current_chunk_idx=ar_idx,
            eager_mode=False,
        )

    def predict_flow(
        self,
        noisy_latent: Tensor,
        timestep: Tensor,
        cache: CosmosTransformerCache,
        input: Tensor | None = None,
    ) -> Tensor:
        flow_cond = self._predict_flow(
            noisy_latent=noisy_latent,
            timestep=timestep,
            cache=cache,
            network_cache=cache.network_cache,
            input=input,
            uncond=False,
        )
        if cache.network_cache_uncond is None:
            return flow_cond
        flow_uncond = self._predict_flow(
            noisy_latent=noisy_latent,
            timestep=timestep,
            cache=cache,
            network_cache=cache.network_cache_uncond,
            input=input,
            uncond=True,
        )
        return flow_uncond + self.config.guidance_scale * (flow_cond - flow_uncond)

    def postprocess_clean_latent(
        self,
        clean_latent: Tensor,
        cache: CosmosTransformerCache,
        input: Tensor | None = None,
    ) -> Tensor:
        return self._maybe_inject_image(clean_latent, cache)

    def finalize_kv_cache(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        if self.config.skip_finalize_kv_cache:
            return
        super().finalize_kv_cache(*args, **kwargs)

    def patchify_and_maybe_split_cp(self, x: Tensor) -> Tensor:
        """Patchify and CP-split a video-shaped tensor ``[..., T, C, H, W]``.

        Returns the post-patchify, post-CP-split layout ``[..., L/cp, D]``.
        """
        return self.network.patchify_and_maybe_split_cp(
            x,
            process_groups=[self._cp_group],
            cp_dims=[-2],
        )

    def unpatchify_and_maybe_gather_cp(self, x: Tensor) -> Tensor:
        """Inverse of :meth:`patchify_and_maybe_split_cp`.

        Expects ``[..., L/cp, D]`` and returns ``[..., T, C, H, W]``.
        """
        assert self._output_height is not None and self._output_width is not None, (
            "unpatchify_and_maybe_gather_cp requires an initialized rollout; "
            "call initialize_autoregressive_cache(..., height=..., width=...) first."
        )
        _, kh, kw = self.config.network.patch_size
        return self.network.unpatchify_and_maybe_gather_cp(
            pH=self._output_height // kh,
            pW=self._output_width // kw,
            x=x,
            process_groups=[self._cp_group],
            cp_dims=[-2],
        )
