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

"""Wan 2.1 DiT."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, overload

import torch
from torch import Tensor

from flashdreams.core.attention.rope import RotaryPositionEmbedding3D
from flashdreams.core.checkpoint.load import load_checkpoint
from flashdreams.infra.compile import compile_module
from flashdreams.infra.config import InstantiateConfig
from flashdreams.infra.cuda_graph import CUDAGraphWrapper
from flashdreams.infra.diffusion.transformer import (
    Transformer,
    TransformerAutoregressiveCache,
)
from flashdreams.recipes.wan.autoencoder.i2v import I2VCtrl
from flashdreams.recipes.wan.transformer.impl.network import (
    WanDiTNetwork,
    WanDiTNetwork1pt3BConfig,
    WanDiTNetworkCache,
    WanDiTNetworkConfig,
)

## Autoregressive cache (per-rollout, mutated across AR steps)


@dataclass(kw_only=True)
class Wan21TransformerCache(TransformerAutoregressiveCache):
    """Per-rollout AR cache for the Wan 2.1 transformer.

    Holds an always-present conditional network cache and an optional
    unconditional one for classifier-free guidance (``None`` disables CFG).
    Both branches own independent per-block self-attention KV buffers since
    the residual stream diverges after the first cross-attention layer.
    """

    network_cache: WanDiTNetworkCache
    """Conditional per-block KV / cross-attention caches."""

    network_cache_uncond: WanDiTNetworkCache | None = None
    """Unconditional caches; ``None`` disables CFG."""

    rope_adapter: RotaryPositionEmbedding3D
    """3D RoPE adapter; advances along T per AR step."""

    rope_freqs: Tensor | None = None
    """Self-attention RoPE frequencies for the current AR step.
    Shape ``[L, 1, 1, head_dim // 2]`` after CP. Recomputed once per
    AR step in :meth:`start` and reused across cond and uncond branches
    (and across all scheduler steps within the AR step)."""

    autoregressive_index: int = -1
    """Current AR step index, set by ``start``."""

    def start(self, autoregressive_index: int) -> None:
        # Hoist per-block KV pre-update out of the (graph-captured) network
        # forward; predict_flow runs with eager_mode=False so the network
        # itself does not call before_update. Same for shift_t: tying the
        # AR index into the captured graph as a Python int would re-trigger
        # cat/repeat on every cond/uncond pass.
        self.rope_freqs = self.rope_adapter.shift_t(autoregressive_index)

        self.autoregressive_index = autoregressive_index
        self.network_cache.before_update(autoregressive_index)
        if self.network_cache_uncond is not None:
            self.network_cache_uncond.before_update(autoregressive_index)

    def finalize(self, autoregressive_index: int) -> None:
        self.network_cache.after_update(autoregressive_index)
        if self.network_cache_uncond is not None:
            self.network_cache_uncond.after_update(autoregressive_index)


## Transformer


@dataclass(kw_only=True)
class Wan21TransformerConfig(InstantiateConfig["Wan21Transformer"]):
    """Config for the Wan 2.1 transformer.

    Bakes in the temporal layout (``len_t``, ``window_size_t``,
    ``sink_size_t``) and the CFG / compile knobs. Per-rollout spatial
    layout (``height``, ``width``) is supplied to
    :meth:`Wan21Transformer.initialize_autoregressive_cache` so one
    instance can serve multiple resolutions. Wan flattens ``T*H*W`` into
    one token axis and shards it across the THW CP group; the CP size
    is auto-detected from ``torch.distributed.get_world_size()`` at
    construction time, so the launcher
    (``torchrun --nproc_per_node=N``) is the single source of truth.

    The two I2V flags are independent and composable:

    - ``stamp_image_latent``: overwrite the noisy latent with the clean
      image latent at masked positions every denoising step, and re-stamp
      the predicted ``x0`` the same way. ``network.in_dim`` unchanged.
      (flashdreams mask-inject recipe; used by causal_wan21.)
    - ``concat_image_mask_to_latent``: append the 4-channel mask and
      16-channel image latent along the channel dim. Builders that set
      this flag must also set ``network.in_dim = 16 + 4 + 16`` to match
      the official Wan 2.1 14B I2V layout.

    With both enabled, the stamp runs first and the result is then
    concatenated with the mask + image latent.
    """

    _target: type["Wan21Transformer"] = field(default_factory=lambda: Wan21Transformer)

    network: WanDiTNetworkConfig = field(default_factory=WanDiTNetwork1pt3BConfig)
    dtype: torch.dtype = torch.bfloat16
    checkpoint_path: str | None = None

    state_dict_transform: Callable[[dict[str, Tensor]], dict[str, Tensor]] | None = None
    """Pre-load state-dict remap (e.g. Self-Forcing's
    ``generator_ema.model.…`` layout)."""

    batch_shape: tuple[int, ...] = (1,)
    """Batch dims of the latent (excluding the L, D dims)."""

    len_t: int = 21
    """Latent frames per AR chunk (post-VAE)."""

    guidance_scale: float = 1.0
    """CFG scale ``s``: ``flow = uncond + s * (cond - uncond)``. ``1.0``
    disables CFG; ``> 1.0`` requires negative-text embeddings at cache
    build time."""

    window_size_t: int = 21
    """Self-attention sliding-window size (pre-patchify T frames)."""

    sink_size_t: int = 0
    """Number of sink tokens preserved across the window."""

    h_extrapolation_ratio: float = 1.0
    w_extrapolation_ratio: float = 1.0

    compile_network: bool = True
    """``torch.compile`` the network on init."""

    use_cuda_graph: bool = True
    """Wrap the network in ``CUDAGraphWrapper`` for steady-state replay.
    Caller must keep non-staged inputs at stable storage addresses across
    calls. ``predict_flow`` dispatches to ``wrapper.drain`` while the KV
    cache is still filling and to ``wrapper`` once it reaches steady state."""

    cuda_graph_warmup_iters: int = 2
    """Eager calls before capture (>= 2 to drain Inductor autotune)."""

    stamp_image_latent: bool = False
    """See class docstring (mask-inject I2V recipe)."""

    concat_image_mask_to_latent: bool = False
    """See class docstring (channel-concat I2V layout)."""


class Wan21Transformer(Transformer[Wan21TransformerCache]):
    """Wan 2.1 DiT adapted to the infra Transformer interface."""

    config: Wan21TransformerConfig
    network: WanDiTNetwork

    def __init__(self, config: Wan21TransformerConfig) -> None:
        super().__init__(config)
        self.config = config

        # Auto-detect CP size from the launcher (``torchrun
        # --nproc_per_node=N``) — the single source of truth. Wan flattens
        # T*H*W into one token axis and shards it across the WORLD group.
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
            f"len_t ({config.len_t}) must be divisible by patch_size[0] ({kt})."
        )
        self._output_height: int | None = None
        self._output_width: int | None = None

        self.network = config.network.setup()
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
        # Cond and CFG-uncond branches each get their own wrapper since each
        # mutates an independent rolling KV cache. The dispatch threshold
        # matches the KV cache's filling -> steady transition so the captured
        # region only sees steady-state paths.
        self._use_cuda_graph = config.use_cuda_graph
        chunks_total = config.sink_size_t + config.window_size_t
        assert chunks_total % config.len_t == 0, (
            f"sink_size_t + window_size_t ({chunks_total}) must be "
            f"divisible by len_t ({config.len_t}) so the BlockKVCache can "
            f"fit a whole number of AR chunks."
        )
        self._cuda_graph_capture_ar_idx: int = chunks_total // config.len_t
        self._network_call: CUDAGraphWrapper | WanDiTNetwork = (
            CUDAGraphWrapper(self.network, warmup_iters=config.cuda_graph_warmup_iters)
            if config.use_cuda_graph
            else self.network
        )
        self._network_call_uncond: CUDAGraphWrapper | WanDiTNetwork = (
            CUDAGraphWrapper(self.network, warmup_iters=config.cuda_graph_warmup_iters)
            if config.use_cuda_graph
            else self.network
        )

    @property
    def latent_shape(self) -> tuple[int, ...]:
        """Per-rank post-patchify latent shape ``[*batch_shape, L/cp, D]``.

        Wan flattens THW into one token axis and shards across the THW
        CP group. ``D = network.out_dim * prod(patch_size)`` is the
        noise channel count; the mask / image-latent channels added by
        ``concat_image_mask_to_latent`` come from ``input`` in
        ``predict_flow``, not from the noise tensor.

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
            cfg.network.out_dim * kt * kh * kw,
        )

    @torch.no_grad()
    def _build_network_cache(
        self,
        *,
        text_embeddings: Tensor,
        image_embeddings: Tensor | None = None,
    ) -> WanDiTNetworkCache:
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
            img_embeddings=image_embeddings,
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
    ) -> Wan21TransformerCache:
        """Build a seeded transformer cache for a new rollout.

        I2V state is *not* baked into the cache; the latent + injection mask
        are passed per AR step as the ``input`` argument to ``predict_flow`` /
        ``postprocess_clean_latent``.

        Args:
            height: Pre-patchify latent height (post-VAE).
            width: Pre-patchify latent width (post-VAE).
            text_embeddings: Conditional UMT5 embeddings ``[..., text_len, text_dim]``.
            image_embeddings: Conditional CLIP image embeddings (only used by
                networks with ``cross_attn_enable_img=True``). Shared with the
                uncond branch.
            negative_text_embeddings: Negative-prompt embeddings. Required iff
                ``config.guidance_scale > 1.0``; must be ``None`` otherwise.

        Returns:
            Populated cache. ``network_cache_uncond`` is ``None`` iff CFG is
            disabled.
        """
        # Stash the per-rollout spatial layout. ``_build_network_cache``
        # below and ``latent_shape`` / ``unpatchify_and_maybe_gather_cp``
        # at AR-step time read these.
        cfg = self.config
        kt, kh, kw = cfg.network.patch_size
        assert height % kh == 0 and width % kw == 0, (
            f"(height, width) = ({height}, {width}) must be divisible by "
            f"patch_size={cfg.network.patch_size[1:]}."
        )
        self._output_height = height
        self._output_width = width
        total_tokens = (cfg.len_t // kt) * (height // kh) * (width // kw)
        assert total_tokens % self._cp_size == 0, (
            f"Wan token length ({total_tokens} from len_t={cfg.len_t}, "
            f"height={height}, width={width}, "
            f"patch_size={cfg.network.patch_size}) must be divisible by "
            f"cp_size={self._cp_size}"
        )

        network_cache = self._build_network_cache(
            text_embeddings=text_embeddings,
            image_embeddings=image_embeddings,
        )
        network_cache_uncond: WanDiTNetworkCache | None = None
        if self.config.guidance_scale > 1.0:
            assert negative_text_embeddings is not None, (
                f"WanTransformerConfig.guidance_scale="
                f"{self.config.guidance_scale} > 1.0 requires "
                f"negative_text_embeddings."
            )
            network_cache_uncond = self._build_network_cache(
                text_embeddings=negative_text_embeddings,
                image_embeddings=image_embeddings,
            )

        head_dim = self.config.network.dim // self.config.network.num_heads
        rope_adapter = RotaryPositionEmbedding3D(
            len_t=cfg.len_t // kt,
            len_h=height // kh,
            len_w=width // kw,
            head_dim=head_dim,
            h_extrapolation_ratio=self.config.h_extrapolation_ratio,
            w_extrapolation_ratio=self.config.w_extrapolation_ratio,
            interleaved=True,
            device=self.device,
        )
        rope_adapter.set_context_parallel_group(cp_group=self._cp_group)

        # Reset any prior CUDA graph: it refers to slot pointers from the
        # previous cache, which the new cache invalidates.
        if self._use_cuda_graph:
            assert isinstance(self._network_call, CUDAGraphWrapper)
            self._network_call.reset()
            assert isinstance(self._network_call_uncond, CUDAGraphWrapper)
            self._network_call_uncond.reset()

        return Wan21TransformerCache(
            network_cache=network_cache,
            network_cache_uncond=network_cache_uncond,
            rope_adapter=rope_adapter,
        )

    def _stamp_image_latent(
        self,
        latent: Tensor,
        control: I2VCtrl,
    ) -> Tensor:
        """Overwrite ``latent`` with the image latent at masked positions.

        All three tensors share the same patchified + CP-split shape, so this
        is a plain per-token blend ``(1 - m) * latent + m * control.latent``.
        """
        return latent * (1.0 - control.mask) + control.latent * control.mask

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

    def _build_network_input(
        self,
        noisy_latent: Tensor,
        input: I2VCtrl | None,
    ) -> Tensor:
        """Apply the (optional) I2V stamp / channel-concat to the noisy latent.

        See :class:`Wan21TransformerConfig` for the two composable I2V
        modes. T2V (``input is None``) takes neither path.
        """
        network_input = noisy_latent
        if self.config.stamp_image_latent:
            assert isinstance(input, I2VCtrl), (
                "stamp_image_latent requires input to be an "
                f"I2VCtrl (got {type(input).__name__})"
            )
            network_input = self._stamp_image_latent(network_input, input)
        if self.config.concat_image_mask_to_latent:
            assert isinstance(input, I2VCtrl), (
                "concat_image_mask_to_latent requires input to be "
                f"an I2VCtrl (got {type(input).__name__})"
            )
            # The patchified mask carries the encoder's 16-channel uniform
            # tag. Slicing the leading 16 entries recovers the 4-channel mask
            # the official 14B I2V network expects (4 ch * K=4 patch entries).
            mask = input.mask[..., :16]
            network_input = torch.cat([network_input, mask, input.latent], dim=-1)
        return network_input

    def _predict_flow(
        self,
        network_input: Tensor,
        timestep: Tensor,
        cache: Wan21TransformerCache,
        autoregressive_index: int,
        network_extra_kwargs: dict[str, Any],
        *,
        uncond: bool,
    ) -> Tensor:
        network_cache = cache.network_cache_uncond if uncond else cache.network_cache
        assert network_cache is not None, (
            "uncond=True requires cache.network_cache_uncond, but it is None "
            "(CFG was not enabled at cache build time)."
        )
        return self._select_network(autoregressive_index, uncond=uncond)(
            x=network_input,
            timesteps=timestep,
            cache=network_cache,
            rope_freqs=cache.rope_freqs,
            current_chunk_idx=autoregressive_index,
            eager_mode=False,
            **network_extra_kwargs,
        )

    def predict_flow(
        self,
        noisy_latent: Tensor,
        timestep: Tensor,
        cache: Wan21TransformerCache,
        input: I2VCtrl | None = None,
        network_extra_kwargs: dict[str, Any] | None = None,
    ) -> Tensor:
        ar_idx = cache.autoregressive_index
        assert ar_idx >= 0, (
            "Wan21TransformerCache.start(autoregressive_index) must be called "
            "before predict_flow (DiffusionModel.generate handles this)."
        )
        network_extra_kwargs = network_extra_kwargs or {}
        network_input = self._build_network_input(noisy_latent, input)

        flow_cond = self._predict_flow(
            network_input=network_input,
            timestep=timestep,
            cache=cache,
            autoregressive_index=ar_idx,
            network_extra_kwargs=network_extra_kwargs,
            uncond=False,
        )
        if cache.network_cache_uncond is None:
            return flow_cond
        flow_uncond = self._predict_flow(
            network_input=network_input,
            timestep=timestep,
            cache=cache,
            autoregressive_index=ar_idx,
            network_extra_kwargs=network_extra_kwargs,
            uncond=True,
        )
        return flow_uncond + self.config.guidance_scale * (flow_cond - flow_uncond)

    def postprocess_clean_latent(
        self,
        clean_latent: Tensor,
        cache: Wan21TransformerCache,
        input: I2VCtrl | None = None,
    ) -> Tensor:
        """Re-stamp ``x0`` masked positions with the image latent (mask-inject I2V only).

        T2V and the channel-concat I2V mode fall through unchanged.
        """
        if input is None or not self.config.stamp_image_latent:
            return clean_latent
        return self._stamp_image_latent(clean_latent, input)

    @overload
    def patchify_and_maybe_split_cp(self, x: Tensor) -> Tensor: ...
    @overload
    def patchify_and_maybe_split_cp(self, x: I2VCtrl) -> I2VCtrl: ...
    def patchify_and_maybe_split_cp(self, x: Tensor | I2VCtrl) -> Tensor | I2VCtrl:
        """Patchify and CP-split a noisy latent or an I2V control payload.

        Tensors delegate to the network helper; I2V payloads patchify the
        ``latent`` and ``mask`` fields independently so the per-field channel
        layouts are preserved for the mask-inject blend downstream.
        """
        if isinstance(x, I2VCtrl):
            if x._is_patchified:
                return x
            return I2VCtrl(
                latent=self.patchify_and_maybe_split_cp(x.latent),
                mask=self.patchify_and_maybe_split_cp(x.mask),
                _is_patchified=True,
            )
        return self.network.patchify_and_maybe_split_cp(
            x,
            process_groups=[self._cp_group],
            cp_dims=[-2],
        )

    def unpatchify_and_maybe_gather_cp(self, x: Tensor) -> Tensor:
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
