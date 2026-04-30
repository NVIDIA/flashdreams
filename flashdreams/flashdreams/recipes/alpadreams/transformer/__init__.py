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

"""Cosmos DiT adapted to the infra :class:`Transformer` interface.

The Cosmos DiT (:class:`CosmosDiTNetwork`) is a multi-view video DiT
with optional HDMap conditioning, cross-view attention, and AdaLN-LoRA.
It is the backbone used by the alpadreams driving-scene video
generation project. This module bridges that backbone to the
:mod:`flashdreams.infra.diffusion` interfaces.

Conditioning model (per-AR-step):

- The HDMap chunk for the current AR step is the per-step *control*
  (encoded once per AR step to a latent and routed through the
  infra encoder slot).
- The first-frame VAE-encoded latent + first-block / other-block masks
  are *cache-only* state populated once at rollout start.
- Mask injection happens at AR step ``0`` only, both before
  :meth:`predict_flow` (overriding noisy latent) and inside
  :meth:`postprocess_clean_latent` (overriding the predicted ``x0``).
  Subsequent AR steps emit zero masks so the network ignores the
  image-latent slot.

A single ``CosmosTransformer`` instance is bound to one ``(batch_shape,
num_views, height, width, len_t)`` resolution and one ``cp_size``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

from flashdreams.core.checkpoint.load import load_checkpoint
from flashdreams.infra.cuda_graph import CUDAGraphWrapper
from flashdreams.infra.diffusion.transformer import (
    Transformer,
    TransformerAutoregressiveCache,
    TransformerConfig,
)
from flashdreams.recipes.wan.transformer.impl.rope import (
    RotaryPositionEmbedding3D,
)

from .impl.context_parallel import (
    HierarchicalCPGroups,
    create_hierarchical_cp_groups,
)
from .impl.network import (
    CosmosDiTNetwork,
    CosmosDiTNetworkCache,
    CosmosDiTNetworkConfig,
)

# ---------------------------------------------------------------------------
# Default camera names / view index mapping (matches projects/alpadreams)
# ---------------------------------------------------------------------------

DEFAULT_CAMERAS: tuple[str, ...] = (
    "camera_front_wide_120fov",
    "camera_cross_right_120fov",
    "camera_rear_right_70fov",
    "camera_rear_tele_30fov",
    "camera_rear_left_70fov",
    "camera_cross_left_120fov",
    "camera_front_tele_30fov",
)

DEFAULT_CAMERA_VIEW_MAPPING: dict[str, int] = dict(
    zip(DEFAULT_CAMERAS, range(len(DEFAULT_CAMERAS)))
)


# ---------------------------------------------------------------------------
# Long-lived per-rollout cache (image latent, masks, KV cache, RoPE).
# ---------------------------------------------------------------------------


@dataclass(kw_only=True)
class CosmosTransformerCache(TransformerAutoregressiveCache):
    """Long-lived AR cache for ``CosmosTransformer``.

    Holds:

    - ``network_cache``: per-block self-attention KV + cross-attention
      KV (latter is text-only, set once at rollout start).
    - ``rope_adapter``: 3D RoPE adapter, advanced via ``shift_t`` per
      AR step.
    - ``image``: VAE-encoded first-frame latent, padded along T to
      ``len_t`` and patchified. Used to override the noisy / predicted
      latent at AR step 0 only.
    - ``mask_first_block`` / ``mask_other_blocks``: binary masks (also
      patchified). The first-block mask has 1s on the first temporal
      latent frame and 0 elsewhere; the other-blocks mask is all
      zeros. These are concatenated with the noisy latent inside the
      network forward (``condition_video_input_mask``) and used for
      mask injection at AR step 0.
    - ``view_indices``: optional per-view index tensor for AdaLN view
      modulation (``None`` when ``num_views == 1``).
    """

    network_cache: CosmosDiTNetworkCache
    rope_adapter: RotaryPositionEmbedding3D
    image: Tensor
    mask_first_block: Tensor
    mask_other_blocks: Tensor
    view_indices: Tensor | None = None
    autoregressive_index: int = -1

    def start(self, autoregressive_index: int) -> None:
        # Hoist the per-block KV pre-update hook out of the network
        # forward (predict_flow runs the network with eager_mode=False)
        # so capture-time / replay-time of the network never executes
        # cache pointer-swap bookkeeping.
        self.autoregressive_index = autoregressive_index
        self.network_cache.before_update(autoregressive_index)

    def finalize(self, autoregressive_index: int) -> None:
        # Counterpart of start(): per-block KV post-update hook now
        # lives at the AR-step boundary instead of inside the captured
        # network forward.
        self.network_cache.after_update(autoregressive_index)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(kw_only=True)
class CosmosTransformerConfig(TransformerConfig):
    """Configuration for :class:`CosmosTransformer`.

    Each instance is bound to one ``(batch_shape, num_views, height,
    width, len_t)`` layout AND one ``cp_size``. The HDMap condition's
    in-channel count is propagated to the underlying network via
    ``additional_concat_ch`` in ``__post_init__``.
    """

    _target: type["CosmosTransformer"] = field(
        default_factory=lambda: CosmosTransformer
    )

    network: CosmosDiTNetworkConfig = field(default_factory=CosmosDiTNetworkConfig)
    dtype: torch.dtype = torch.bfloat16
    checkpoint_path: str | None = None
    state_dict_transform: Callable[[dict[str, Tensor]], dict[str, Tensor]] | None = None
    """Optional callback applied to the loaded state-dict before
    ``network.load_state_dict``. Defaults to a ``net.``-prefix stripper
    matching the upstream alpadreams checkpoints."""

    # Per-rollout layout ------------------------------------------------
    batch_shape: tuple[int, ...] = (1,)
    """Batch dims of the latent (without ``V, T, HW, D``)."""
    num_views: int = 1
    """Number of camera views (``V``). Cross-view attention is enabled
    automatically when ``num_views > 1``."""
    height: int = 90
    """Latent (post-VAE) height in pixels."""
    width: int = 160
    """Latent (post-VAE) width in pixels."""
    len_t: int = 4
    """Latent (post-VAE) frames per AR chunk."""

    # Conditioning ------------------------------------------------------
    enable_hdmap_condition: bool = True
    """Whether to enable the HDMap conditioning branch."""
    encode_with_pixel_shuffle: bool = False
    """If True, the HDMap is encoded via pixel-shuffle (192 channels);
    otherwise via a Wan VAE (16 channels). Determines
    ``network.additional_concat_ch``."""

    # Context-parallel --------------------------------------------------
    cp_size: int = 1
    """Size of the THW context-parallel group. Validated against
    ``torch.distributed.get_world_size()`` in ``__init__``."""

    # RoPE extrapolation (3.0 for 720P; 2.0 for 480P).
    h_extrapolation_ratio: float = 3.0
    w_extrapolation_ratio: float = 3.0

    # Self-attention sliding window (in pre-patchify T frames).
    window_size_t: int = 8
    sink_size_t: int = 0

    # Speedup.
    compile_network: bool = True
    use_cuda_graph: bool = True
    """Wrap the (optionally compiled) network in :class:`CUDAGraphWrapper`
    so steady-state ``predict_flow`` calls replay a captured graph
    instead of re-launching kernels every denoising step. Caller is
    responsible for keeping all non-staged inputs (timestep, rope_freqs,
    masks, hdmap_condition, view_indices) at stable storage addresses
    across calls -- the wrapper only stages the noisy latent."""
    warmup_iters: int = 2
    """Number of eager calls per rollout before the wrapper captures
    (only consulted when :attr:`use_cuda_graph` is True). Two is the
    minimum that drains Inductor autotune (call 1) AND gives a clean
    no-sync stream (call 2) before ``cudaStreamBeginCapture``."""

    def __post_init__(self) -> None:
        # Wire HDMap conditioning channel-count.
        if self.enable_hdmap_condition:
            self.network.additional_concat_ch = (
                192 if self.encode_with_pixel_shuffle else 16
            )
        else:
            self.network.additional_concat_ch = 0
        # Cross-view attention iff multi-view.
        self.network.enable_cross_view_attn = self.num_views > 1

        kt = self.network.patch_temporal
        kh = kw = self.network.patch_spatial
        assert (
            self.len_t % kt == 0 and self.height % kh == 0 and self.width % kw == 0
        ), (
            f"({self.len_t}, {self.height}, {self.width}) must be divisible by "
            f"patch_size ({kt}, {kh}, {kw})"
        )
        self._pT = self.len_t // kt
        self._pH = self.height // kh
        self._pW = self.width // kw

        # First AR step at which the per-block KV cache's `cached_k()`
        # reaches its steady (= window-full) shape. Before this step,
        # the attention sequence length grows by `_pT` tokens per AR
        # step (filling phase); from this step onwards every call sees
        # the same `(sink_size + window_size)` tokens, so the network
        # forward is shape-stable and safe for CUDA-graph capture.
        # Counted in chunks (one chunk per AR step):
        #   chunks_per_window = (sink_size_t + window_size_t) // _pT
        # The last filling step (chunks_per_window - 1) ALREADY makes
        # cached_k() return the full prefix, so it counts as steady.
        chunks_total = self.sink_size_t + self.window_size_t
        assert chunks_total % self._pT == 0, (
            f"sink_size_t + window_size_t ({chunks_total}) must be "
            f"divisible by _pT ({self._pT}) so the BlockKVCache can fit "
            f"a whole number of AR chunks."
        )
        self._steady_ar_idx = chunks_total // self._pT - 1


# ---------------------------------------------------------------------------
# Default state-dict transform (alpadreams-style "net." prefix stripper).
# ---------------------------------------------------------------------------


def _strip_net_prefix(state_dict: dict[str, Tensor]) -> dict[str, Tensor]:
    """Strip the ``net.`` prefix added by the legacy alpadreams training stack."""
    out: dict[str, Tensor] = {}
    for k, v in state_dict.items():
        out[k[len("net.") :] if k.startswith("net.") else k] = v
    return out


# ---------------------------------------------------------------------------
# Transformer
# ---------------------------------------------------------------------------


class CosmosTransformer(Transformer[CosmosTransformerCache]):
    """Cosmos DiT (multi-view, HDMap-conditioned) as a infra Transformer."""

    network: CosmosDiTNetwork

    def __init__(
        self,
        config: CosmosTransformerConfig,
        device: torch.device | None = None,
    ) -> None:
        super().__init__(config)
        self.config: CosmosTransformerConfig = config

        if torch.distributed.is_initialized():
            world_size = torch.distributed.get_world_size()
            assert config.cp_size == world_size, (
                f"CosmosTransformerConfig.cp_size ({config.cp_size}) must match "
                f"torch.distributed.get_world_size() ({world_size})"
            )
            self.cp_groups = create_hierarchical_cp_groups(
                world_size=world_size,
                rank=torch.distributed.get_rank(),
                V=config.num_views,
                T=config.len_t,
                single_group_as_none=True,
            )
        else:
            assert config.cp_size == 1, (
                f"CosmosTransformerConfig.cp_size must be 1 in non-distributed "
                f"mode (got {config.cp_size})"
            )
            self.cp_groups = HierarchicalCPGroups(rank=0)

        self.network = CosmosDiTNetwork(config=config.network)
        if device is not None:
            self.network = self.network.to(device=device)
        self.network = self.network.to(dtype=config.dtype)
        self.network.eval()
        self.network.set_context_parallel_group(
            self_attn_group=self.cp_groups.THW_group,
            cross_view_attn_group=self.cp_groups.V_group,
        )

        if config.checkpoint_path is not None:
            transform = config.state_dict_transform or _strip_net_prefix
            state_dict = load_checkpoint(config.checkpoint_path)
            state_dict = transform(state_dict)  # ty:ignore[invalid-argument-type]
            self.network.load_state_dict(state_dict)
        self.network.update_parameters_after_loading_checkpoint()

        if config.compile_network:
            self.network = torch.compile(  # type: ignore[assignment]  # ty:ignore[invalid-assignment]
                self.network, mode="max-autotune-no-cudagraphs"
            )

        # Per-rollout dispatch (when use_cuda_graph=True):
        # - AR step < cfg._steady_ar_idx (filling phase: cached_k()
        #   length grows each step): wrapper.drain -- eager through
        #   the wrapper's static input buffer. Each new shape drains
        #   its own Inductor autotune so the same shape, when revisited
        #   in a later rollout, no longer syncs during capture.
        # - AR step >= cfg._steady_ar_idx (KV window first full and
        #   stays full): wrapper.__call__ -- 2 warmups + 1 capture,
        #   then pure replays for every same-shape predict_flow call.
        self._use_cuda_graph = config.use_cuda_graph
        self._network_call: Callable[..., Tensor] = (
            CUDAGraphWrapper(self.network, warmup_iters=config.warmup_iters)
            if config.use_cuda_graph
            else self.network
        )

    # ------------------------------------------------------------------
    # Patchify / CP plumbing
    # ------------------------------------------------------------------

    @property
    def latent_shape(self) -> tuple[int, ...]:
        """Per-rank latent shape ``[..., V/cp_V, pT/cp_T, HW/cp_HW, D]``.

        Derived from :attr:`cp_groups` so the broadcast against the
        patchified+CP-split mask/image inside :meth:`_maybe_inject_image`
        always lines up, no matter which axes the hierarchical V→T→HW
        splitter ended up sharding.
        """
        cfg = self.config
        kt = cfg.network.patch_temporal
        kh = kw = cfg.network.patch_spatial
        D = cfg.network.in_channels * kt * kh * kw
        return (
            *cfg.batch_shape,
            cfg.num_views // self.cp_groups.V_size,
            self.config._pT // self.cp_groups.T_size,
            (self.config._pH * self.config._pW) // self.cp_groups.HW_size,
            D,
        )

    @property
    def _process_groups(self) -> list[Any]:
        return [self.cp_groups.V_group, self.cp_groups.T_group, self.cp_groups.HW_group]

    @property
    def _cp_dims(self) -> list[int]:
        return [-5, -4, -3]  # V, T, HW (counted from the end to support arbitrary B...)

    def patchify_and_maybe_split_cp(self, x: Any) -> Any:
        # Cosmos network patchify expects 6D ``[B, V, T, C, H, W]`` (asserted in
        # CosmosDiTNetwork.patchify_and_maybe_split_cp). We pass cp_dims as
        # positive [1, 2, 3] for compatibility with that layout.
        if isinstance(x, Tensor):
            return self.network.patchify_and_maybe_split_cp(
                x,
                process_groups=self._process_groups,
                cp_dims=[1, 2, 3],
            )
        raise TypeError(
            f"CosmosTransformer.patchify_and_maybe_split_cp got unsupported "
            f"input type {type(x).__name__}; expected Tensor."
        )

    def unpatchify_and_maybe_gather_cp(self, x: Tensor) -> Tensor:
        return self.network.unpatchify_and_maybe_gather_cp(
            pH=self.config._pH,
            pW=self.config._pW,
            x=x,
            process_groups=self._process_groups,
            cp_dims=[1, 2, 3],
        )

    # ------------------------------------------------------------------
    # Condition / cache plumbing
    # ------------------------------------------------------------------

    @torch.no_grad()
    def initialize_autoregressive_cache(  # type: ignore[override]
        self,
        *,
        text_embeddings: Tensor,
        image_embeddings: Tensor,
        view_names: list[str] | None = None,
        **_unused: Any,
    ) -> CosmosTransformerCache:
        """Build a fully seeded :class:`CosmosTransformerCache` for a new rollout.

        Args:
            text_embeddings: Text embeddings of shape ``[B, V, L, D]``.
            image_embeddings: VAE-encoded first-frame latent of shape
                ``[B, V, 1, C, H, W]``.
            view_names: List of view names (length ``V``); required when
                ``num_views > 1``.
        """
        from flashdreams.core.distributed.context_parallel import split_inputs_cp

        if self.cp_groups.V_group is not None:
            text_embeddings = split_inputs_cp(
                text_embeddings, seq_dim=1, cp_group=self.cp_groups.V_group
            )

        cfg = self.config
        head_dim = cfg.network.model_channels // cfg.network.num_heads
        rope_adapter = RotaryPositionEmbedding3D(
            len_t=cfg._pT,
            len_h=cfg._pH,
            len_w=cfg._pW,
            head_dim=head_dim,
            h_extrapolation_ratio=cfg.h_extrapolation_ratio,
            w_extrapolation_ratio=cfg.w_extrapolation_ratio,
            device=self.device,
        )
        rope_adapter.set_context_parallel_group(cp_group=self.cp_groups.THW_group)

        num_tokens_per_view_per_step = cfg._pH * cfg._pW
        if self.cp_groups.THW_group is not None:
            num_tokens_per_view_per_step //= self.cp_groups.THW_group.size()
        network_cache = self.network.initialize_cache(
            chunk_size=num_tokens_per_view_per_step * cfg._pT,
            window_size=num_tokens_per_view_per_step * cfg.window_size_t,
            sink_size=num_tokens_per_view_per_step * cfg.sink_size_t,
            context=text_embeddings,
        )

        view_indices: Tensor | None = None
        if cfg.network.enable_cross_view_attn:
            assert view_names is not None and len(view_names) == cfg.num_views, (
                f"view_names of length {cfg.num_views} required when "
                f"num_views > 1 (got {view_names})"
            )
            batch_size = image_embeddings.shape[0]
            view_indices = torch.tensor(
                [DEFAULT_CAMERA_VIEW_MAPPING[name] for name in view_names],
                device=self.device,
                dtype=torch.long,
            )
            view_indices = view_indices.repeat(batch_size, 1)
            if self.cp_groups.V_group is not None:
                view_indices = split_inputs_cp(
                    view_indices, seq_dim=1, cp_group=self.cp_groups.V_group
                )

        B, V, _, _, H, W = image_embeddings.shape
        mask_first_block = torch.zeros(
            B, V, cfg.len_t, 1, H, W, device=self.device, dtype=cfg.dtype
        )
        mask_first_block[:, :, :1, :, :, :] = 1.0
        mask_other_blocks = torch.zeros(
            B, V, cfg.len_t, 1, H, W, device=self.device, dtype=cfg.dtype
        )

        # Pad first-frame image latent along T (zeros for steady state).
        image = F.pad(image_embeddings, (0, 0, 0, 0, 0, 0, 0, cfg.len_t - 1))

        # Patchify image and masks once at rollout start.
        image_patched = self.network.patchify_and_maybe_split_cp(
            image, process_groups=self._process_groups, cp_dims=[1, 2, 3]
        )
        mask_first_patched = self.network.patchify_and_maybe_split_cp(
            mask_first_block, process_groups=self._process_groups, cp_dims=[1, 2, 3]
        )
        mask_other_patched = self.network.patchify_and_maybe_split_cp(
            mask_other_blocks, process_groups=self._process_groups, cp_dims=[1, 2, 3]
        )

        # Drop any captured CUDA graph from the previous rollout: its
        # kernels reference the old network_cache's slot pointers,
        # which the freshly-initialised network_cache invalidates.
        # Warmup + capture re-run on the next steady-state predict_flow.
        if self._use_cuda_graph:
            self._network_call.reset()  # type: ignore[union-attr]  # ty:ignore[unresolved-attribute]

        return CosmosTransformerCache(
            network_cache=network_cache,
            rope_adapter=rope_adapter,
            image=image_patched,
            mask_first_block=mask_first_patched,
            mask_other_blocks=mask_other_patched,
            view_indices=view_indices,
        )

    # ------------------------------------------------------------------
    # Mask injection helpers
    # ------------------------------------------------------------------

    def _maybe_inject_image(
        self,
        latent: Tensor,
        cache: CosmosTransformerCache,
    ) -> Tensor:
        """Override the first-temporal-frame latent with the encoded image at AR step 0."""
        if cache.autoregressive_index != 0:
            return latent
        # ``mask_first_block`` is shape [B, V, pT, HW, 1] post-patchify;
        # ``image`` and ``latent`` are shape [B, V, pT, HW, D]. Element-wise.
        mask = cache.mask_first_block[..., :1]
        return latent * (1.0 - mask) + cache.image * mask

    def _select_mask(self, cache: CosmosTransformerCache) -> Tensor:
        return (
            cache.mask_first_block
            if cache.autoregressive_index == 0
            else cache.mask_other_blocks
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def predict_flow(
        self,
        noisy_latent: Tensor,
        timestep: Tensor,
        cache: CosmosTransformerCache,
        input: Tensor | None = None,
    ) -> Tensor:
        ar_idx = cache.autoregressive_index
        assert ar_idx >= 0, (
            "CosmosTransformerCache.start(autoregressive_index) must be called "
            "before predict_flow (DiffusionModel.generate handles this)."
        )
        rope_freqs = cache.rope_adapter.shift_t(offset=ar_idx * self.config._pT)

        # AR step 0: inject the encoded first-frame latent into the noisy input.
        noisy_latent = self._maybe_inject_image(noisy_latent, cache)
        condition_video_input_mask = self._select_mask(cache)

        # eager_mode=False: per-block KV before_update / after_update
        # hooks are invoked at the AR-step boundary by
        # CosmosTransformerCache.start() / .finalize(), not inside the
        # network forward. This keeps the network forward graph-capture
        # clean (no cache pointer-swap bookkeeping inside).
        #
        # AR step < _steady_ar_idx -> wrapper.drain (filling phase;
        # cached_k() shape changes every step). AR step >=
        # _steady_ar_idx -> wrapper.__call__ (window full, shape
        # stable; warmup -> capture -> replay). See `__init__` for the
        # per-rollout dispatch contract.
        if self._use_cuda_graph:
            network = (
                self._network_call.drain  # type: ignore[union-attr]  # ty:ignore[unresolved-attribute]
                if ar_idx < self.config._steady_ar_idx
                else self._network_call
            )
        else:
            network = self.network
        return network(
            noisy_latent,
            timesteps=timestep,
            rope_freqs=rope_freqs,
            cache=cache.network_cache,
            condition_video_input_mask=condition_video_input_mask,
            current_chunk_idx=ar_idx,
            hdmap_condition=input,
            view_indices=cache.view_indices,
            eager_mode=False,
        )

    def postprocess_clean_latent(
        self,
        clean_latent: Tensor,
        cache: CosmosTransformerCache,
        input: Tensor | None = None,
    ) -> Tensor:
        return self._maybe_inject_image(clean_latent, cache)
