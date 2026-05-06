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

"""Multi-view, HDMap-conditioned Cosmos DiT for streaming alpadreams."""

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
from flashdreams.core.distributed.context_parallel import split_inputs_cp
from flashdreams.infra.compile import compile_module
from flashdreams.infra.config import InstantiateConfig
from flashdreams.infra.cuda_graph import CUDAGraphWrapper
from flashdreams.infra.diffusion.transformer import (
    Transformer,
    TransformerAutoregressiveCache,
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

## Default camera names / view-index mapping

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


## Per-rollout cache


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

    image: Tensor
    """First-frame VAE latent, T-padded to ``len_t`` and patchified.
    Injects into the noisy / predicted latent at AR step 0."""

    mask_first_block: Tensor
    """``[B, V, T, 1, H, W]`` mask with ones on the first temporal latent
    frame; used at AR step 0."""

    mask_other_blocks: Tensor
    """All-zero counterpart used at AR step >= 1."""

    view_indices: Tensor | None = None
    """Per-view index tensor for AdaLN view modulation; ``None`` when
    ``num_views == 1``."""

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


## Config


@dataclass(kw_only=True)
class CosmosTransformerConfig(InstantiateConfig["CosmosTransformer"]):
    """Config for the Cosmos transformer.

    Bakes in the temporal layout (``len_t``, ``window_size_t``,
    ``sink_size_t``), CFG / compile knobs, and the multi-view layout.
    Per-rollout spatial layout (``height``, ``width``) is supplied to
    :meth:`CosmosTransformer.initialize_autoregressive_cache` so one
    instance can serve multiple resolutions. CP size is auto-detected
    from ``torch.distributed.get_world_size()`` at construction; build
    the pipeline under a ``torch.distributed`` initialization with the
    desired world size to opt in.

    Builders are responsible for wiring the embedded
    :class:`CosmosDiTNetworkConfig`:

    - ``network.additional_concat_ch`` — ``16`` for the Wan-VAE HDMap
      branch, ``192`` for the pixel-shuffle HDMap branch, ``0`` to
      disable HDMap conditioning.
    - ``network.enable_cross_view_attn`` — ``True`` iff ``num_views > 1``.
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
    """Batch dims of the latent (excluding ``V, T, HW, D``)."""

    num_views: int = 1
    """Number of camera views; >1 enables cross-view attention."""

    len_t: int = 4
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

    @property
    def requires_negative_text_embeddings(self) -> bool:
        """Whether cache initialization must receive negative text embeddings."""
        return self.guidance_scale > 1.0


## Default state-dict transform


def _strip_net_prefix(state_dict: dict[str, Tensor]) -> dict[str, Tensor]:
    """Strip the ``net.`` prefix added by the upstream training stack."""
    out: dict[str, Tensor] = {}
    for k, v in state_dict.items():
        out[k[len("net.") :] if k.startswith("net.") else k] = v
    return out


## Transformer


class CosmosTransformer(Transformer[CosmosTransformerCache]):
    """Multi-view, HDMap-conditioned Cosmos DiT as an infra transformer."""

    network: CosmosDiTNetwork

    def __init__(self, config: CosmosTransformerConfig) -> None:
        super().__init__(config)
        self.config: CosmosTransformerConfig = config

        # Auto-detect CP world size from torch.distributed; non-distributed
        # mode short-circuits to a singleton group set.
        if torch.distributed.is_initialized():
            world_size = torch.distributed.get_world_size()
            self.cp_groups = create_hierarchical_cp_groups(
                world_size=world_size,
                rank=torch.distributed.get_rank(),
                V=config.num_views,
                T=config.len_t,
                single_group_as_none=True,
            )
        else:
            self.cp_groups = HierarchicalCPGroups(rank=0)

        # Pre-patchify temporal divisibility check; per-rollout
        # (height, width) is populated by initialize_autoregressive_cache.
        kt = config.network.patch_temporal
        assert config.len_t % kt == 0, (
            f"len_t ({config.len_t}) must be divisible by patch_temporal ({kt})."
        )
        self._output_height: int | None = None
        self._output_width: int | None = None

        self.network = CosmosDiTNetwork(config=config.network)
        self.network = self.network.to(dtype=config.dtype)
        self.network.eval()
        self.network.set_context_parallel_group(
            self_attn_group=self.cp_groups.THW_group,
            cross_view_attn_group=self.cp_groups.V_group,
        )

        if config.checkpoint_path is not None:
            transform = config.state_dict_transform or _strip_net_prefix
            state_dict = load_checkpoint(config.checkpoint_path)
            state_dict = transform(state_dict)
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

        # In the case of single view, we always flatten the latent tensor into
        # 4D [B, V, L, D]. This makes CP easier: just directly apply on L dimension.
        # For multi-view, we keep the original 5D [B, V, T, HW, D] shape so we can apply
        # dedicated hierarchical CP groups.
        self.flatten_thw = config.num_views == 1

    ## Patchify / CP plumbing

    @property
    def latent_shape(self) -> tuple[int, ...]:
        """Per-rank latent shape ``[..., V/cp_V, pT/cp_T, HW/cp_HW, D]``.

        Per-rollout ``(height, width)`` is populated by
        :meth:`initialize_autoregressive_cache`; reading earlier asserts.
        """
        assert self._output_height is not None and self._output_width is not None, (
            "latent_shape requires an initialized rollout; call "
            "initialize_autoregressive_cache(..., height=..., width=...) first."
        )
        cfg = self.config
        kt = cfg.network.patch_temporal
        kh = kw = cfg.network.patch_spatial
        D = cfg.network.in_channels * kt * kh * kw
        pT = cfg.len_t // kt
        pH = self._output_height // kh
        pW = self._output_width // kw
        if self.flatten_thw:
            return (
                *cfg.batch_shape,
                cfg.num_views // self.cp_groups.V_size,
                (pT * pH * pW) // self.cp_groups.THW_size,
                D,
            )
        else:
            return (
                *cfg.batch_shape,
                cfg.num_views // self.cp_groups.V_size,
                pT // self.cp_groups.T_size,
                (pH * pW) // self.cp_groups.HW_size,
                D,
            )

    def patchify_and_maybe_split_cp(self, x: Tensor) -> Tensor:
        # x expected to be [B, V, T, C, H, W]
        assert x.ndim == 6, f"x must be a 6D tensor, but got shape {x.shape}"

        if self.flatten_thw:
            return self.network.patchify_and_maybe_split_cp(
                x,
                process_groups=[self.cp_groups.V_group, self.cp_groups.THW_group],
                cp_dims=[-3, -2],
                flatten_thw=True,
            )  # [B, V, L, D]
        else:
            return self.network.patchify_and_maybe_split_cp(
                x,
                process_groups=[
                    self.cp_groups.V_group,
                    self.cp_groups.T_group,
                    self.cp_groups.HW_group,
                ],
                cp_dims=[-4, -3, -2],
                flatten_thw=False,
            )  # [B, V, T, HW, D]

    def unpatchify_and_maybe_gather_cp(self, x: Tensor) -> Tensor:
        assert self._output_height is not None and self._output_width is not None, (
            "unpatchify_and_maybe_gather_cp requires an initialized rollout; "
            "call initialize_autoregressive_cache(..., height=..., width=...) first."
        )
        kh = kw = self.config.network.patch_spatial
        pH = self._output_height // kh
        pW = self._output_width // kw
        if self.flatten_thw:
            # x expected to be [B, V, L, D]
            assert x.ndim == 4, f"x must be a 4D tensor, but got shape {x.shape}"
            return self.network.unpatchify_and_maybe_gather_cp(
                pH=pH,
                pW=pW,
                x=x,
                process_groups=[self.cp_groups.V_group, self.cp_groups.THW_group],
                cp_dims=[-3, -2],
                flatten_thw=True,
            )  # [B, V, T, C, H, W]
        else:
            # x expected to be [B, V, T, HW, D]
            assert x.ndim == 5, f"x must be a 5D tensor, but got shape {x.shape}"
            return self.network.unpatchify_and_maybe_gather_cp(
                pH=pH,
                pW=pW,
                x=x,
                process_groups=[
                    self.cp_groups.V_group,
                    self.cp_groups.T_group,
                    self.cp_groups.HW_group,
                ],
                cp_dims=[-4, -3, -2],
                flatten_thw=False,
            )  # [B, V, T, C, H, W]

    ## Condition / cache plumbing

    @torch.no_grad()
    def initialize_autoregressive_cache(
        self,
        *,
        height: int,
        width: int,
        text_embeddings: Tensor,
        image_embeddings: Tensor,
        negative_text_embeddings: Tensor | None = None,
        view_names: list[str] | None = None,
        **_unused: Any,
    ) -> CosmosTransformerCache:
        """Build a fully seeded cache for a new rollout.

        Args:
            height: Pre-patchify latent height (post-VAE).
            width: Pre-patchify latent width (post-VAE).
            text_embeddings: ``[B, V, L, D]`` text embeddings.
            image_embeddings: ``[B, V, 1, C, H, W]`` first-frame VAE latent.
                ``H``/``W`` must equal ``height``/``width``.
            view_names: Length-``V`` view names; required when
                ``num_views > 1``.
        """
        # Stash the per-rollout spatial layout. ``latent_shape``,
        # ``unpatchify_and_maybe_gather_cp`` and the network-cache /
        # RoPE setup below all read these.
        cfg = self.config
        kt = cfg.network.patch_temporal
        kh = kw = cfg.network.patch_spatial
        assert height % kh == 0 and width % kw == 0, (
            f"(height, width) = ({height}, {width}) must be divisible by "
            f"patch_spatial ({kh})."
        )
        self._output_height = height
        self._output_width = width
        pT = cfg.len_t // kt
        pH = height // kh
        pW = width // kw

        if self.cp_groups.V_group is not None:
            text_embeddings = split_inputs_cp(
                text_embeddings, seq_dim=1, cp_group=self.cp_groups.V_group
            )
            if negative_text_embeddings is not None:
                negative_text_embeddings = split_inputs_cp(
                    negative_text_embeddings,
                    seq_dim=1,
                    cp_group=self.cp_groups.V_group,
                )

        head_dim = cfg.network.model_channels // cfg.network.num_heads
        rope_adapter = RotaryPositionEmbedding3D(
            len_t=pT,
            len_h=pH,
            len_w=pW,
            head_dim=head_dim,
            h_extrapolation_ratio=cfg.h_extrapolation_ratio,
            w_extrapolation_ratio=cfg.w_extrapolation_ratio,
            device=self.device,
        )
        rope_adapter.set_context_parallel_group(cp_group=self.cp_groups.THW_group)

        num_tokens_per_view_per_step = pH * pW
        if self.cp_groups.THW_group is not None:
            num_tokens_per_view_per_step //= self.cp_groups.THW_group.size()
        network_cache = self.network.initialize_cache(
            chunk_size=num_tokens_per_view_per_step * pT,
            window_size=num_tokens_per_view_per_step * cfg.window_size_t,
            sink_size=num_tokens_per_view_per_step * cfg.sink_size_t,
            context=text_embeddings,
        )
        network_cache_uncond: CosmosDiTNetworkCache | None = None
        if cfg.requires_negative_text_embeddings:
            assert negative_text_embeddings is not None, (
                f"{type(cfg).__name__}.guidance_scale={cfg.guidance_scale} > 1.0 "
                "requires negative_text_embeddings."
            )
            network_cache_uncond = self.network.initialize_cache(
                chunk_size=num_tokens_per_view_per_step * pT,
                window_size=num_tokens_per_view_per_step * cfg.window_size_t,
                sink_size=num_tokens_per_view_per_step * cfg.sink_size_t,
                context=negative_text_embeddings,
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
        assert H == height and W == width, (
            f"image_embeddings spatial dims ({H}, {W}) must match "
            f"(height, width) = ({height}, {width})."
        )
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
        image_patched = self.patchify_and_maybe_split_cp(image)
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
            view_indices=view_indices,
        )

    ## Mask-injection helpers

    def _maybe_inject_image(
        self,
        latent: Tensor,
        cache: CosmosTransformerCache,
    ) -> Tensor:
        """Replace the first-temporal-frame latent with the encoded image at AR step 0."""
        if cache.autoregressive_index != 0:
            return latent
        mask = cache.mask_first_block[..., :1]
        return latent * (1.0 - mask) + cache.image * mask

    def _select_mask(self, cache: CosmosTransformerCache) -> Tensor:
        return (
            cache.mask_first_block
            if cache.autoregressive_index == 0
            else cache.mask_other_blocks
        )

    ## Forward

    def _select_network(self, autoregressive_index: int, *, uncond: bool) -> Any:
        if not self._use_cuda_graph:
            return self.network

        network_call = self._network_call_uncond if uncond else self._network_call
        assert isinstance(network_call, CUDAGraphWrapper)
        # Cond and CFG-uncond branches both mutate the rolling KV cache, so
        # neither branch can be graph-captured until the cache is steady.
        return (
            network_call.drain
            if autoregressive_index < self._cuda_graph_capture_ar_idx
            else network_call
        )

    def _predict_branch(
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
        return self._select_network(ar_idx, uncond=uncond)(
            noisy_latent,
            timesteps=timestep,
            rope_freqs=cache.rope_freqs,
            cache=network_cache,
            condition_video_input_mask=self._select_mask(cache),
            current_chunk_idx=ar_idx,
            hdmap_condition=input,
            view_indices=cache.view_indices,
            eager_mode=False,
        )

    def predict_flow(
        self,
        noisy_latent: Tensor,
        timestep: Tensor,
        cache: CosmosTransformerCache,
        input: Tensor | None = None,
    ) -> Tensor:
        flow_cond = self._predict_branch(
            noisy_latent=noisy_latent,
            timestep=timestep,
            cache=cache,
            network_cache=cache.network_cache,
            input=input,
            uncond=False,
        )
        if cache.network_cache_uncond is None:
            return flow_cond
        flow_uncond = self._predict_branch(
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
        if not self.config.skip_finalize_kv_cache:
            super().finalize_kv_cache(*args, **kwargs)
        else:
            print("Skipping KV cache finalize")
