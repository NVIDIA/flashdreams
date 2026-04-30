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

"""Wan 2.1 DiT"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor

from flashdreams.core.checkpoint.load import load_checkpoint
from flashdreams.infra.diffusion.transformer import (
    Transformer,
    TransformerAutoregressiveCache,
    TransformerConfig,
)
from flashdreams.recipes.wan.autoencoder.i2v import I2VCtrl

from .impl.network import (
    WanDiTNetwork1pt3BConfig,
    WanDiTNetworkCache,
    WanDiTNetworkConfig,
)
from .impl.rope import RotaryPositionEmbedding3D

# ---------------------------------------------------------------------------
# Autoregressive cache (per-rollout, mutated across AR steps)
# ---------------------------------------------------------------------------


@dataclass(kw_only=True)
class Wan21TransformerCache(TransformerAutoregressiveCache):
    """Long-lived AR cache for ``Wan21Transformer``.

    Holds the conditional ``WanDiTNetworkCache`` (always present) and an
    *optional* unconditional ``WanDiTNetworkCache`` for classifier-free
    guidance (``None`` ⇒ CFG disabled, no extra memory). Both branches own
    independent per-block self-attention KV buffers because the residual
    stream diverges after the first cross-attention layer; this is the
    standard arrangement for streaming/autoregressive video DiTs.

    Built by :meth:`Wan21Transformer.initialize_autoregressive_cache` for the
    common cases, or constructed directly by the user for full control.
    Mutated in-place across AR steps via :meth:`start`.
    """

    network_cache_cond: WanDiTNetworkCache
    """Conditional per-block KV / cross-attention caches."""

    network_cache_uncond: WanDiTNetworkCache | None = None
    """Unconditional per-block caches; ``None`` disables CFG. The
    guidance scale lives on :class:`WanTransformerConfig` rather than
    here, since it is a model-level hyperparameter rather than per-rollout
    state."""

    rope_adapter: RotaryPositionEmbedding3D
    """3D RoPE adapter; ``shift_t`` advances along the time axis per AR step."""

    len_t: int
    """Tokens along T per AR chunk (post-patchify, pre-CP)."""
    len_h: int
    """Tokens along H (post-patchify, pre-CP)."""
    len_w: int
    """Tokens along W (post-patchify, pre-CP)."""

    autoregressive_index: int = -1
    """Current AR step index. Set by :meth:`start`."""

    def start(self, autoregressive_index: int) -> None:
        self.autoregressive_index = autoregressive_index

    def finalize(self, autoregressive_index: int) -> None:
        # The per-block KV update happens inside the network forward
        # (eager_mode=True). DiffusionModel.finalize already issued the
        # extra context-noise forward that re-keys both branches' caches
        # with the clean representation, so there is nothing extra to do.
        return


# ---------------------------------------------------------------------------
# Transformer
# ---------------------------------------------------------------------------


@dataclass(kw_only=True)
class Wan21TransformerConfig(TransformerConfig):
    """Configuration for :class:`Wan21Transformer`.

    Each instance is bound to one ``(batch_shape, height, width, len_t)``
    layout AND to one context-parallel size (``cp_size``). The per-rank
    token shape lives on :attr:`Wan21Transformer.latent_shape` (a property
    derived from runtime CP groups), since Wan shards the flattened
    ``(T*H*W)`` token axis with one THW context-parallel group.
    """

    _target: type["Wan21Transformer"] = field(default_factory=lambda: Wan21Transformer)

    # Network -----------------------------------------------------------
    network: WanDiTNetworkConfig = field(default_factory=WanDiTNetwork1pt3BConfig)
    dtype: torch.dtype = torch.bfloat16
    checkpoint_path: str | None = None
    state_dict_transform: Callable[[dict[str, Tensor]], dict[str, Tensor]] | None = None
    """Optional callback applied to the loaded state dict before
    ``network.load_state_dict``. Use it when the checkpoint comes from a
    fine-tuner that wraps / re-prefixes weights (e.g. Self-Forcing's
    ``generator_ema.model.…`` layout)."""

    # Per-rollout layout ------------------------------------------------
    batch_shape: tuple[int, ...] = (1,)
    """Batch dims of the latent (without the L, D dims)."""
    height: int = 60
    """Latent (post-VAE) height in pixels."""
    width: int = 104
    """Latent (post-VAE) width in pixels."""
    len_t: int = 21
    """Latent (post-VAE) frames per AR chunk."""

    # Context-parallel ---------------------------------------------------
    cp_size: int = 1
    """Size of the THW context-parallel group. Validated against
    ``torch.distributed.get_world_size()`` in ``Wan21Transformer.__init__``."""

    # Classifier-free guidance ------------------------------------------
    guidance_scale: float = 1.0
    """CFG scale ``s``: ``flow = uncond + s * (cond - uncond)``. ``1.0``
    disables CFG (single forward, no uncond cache allocated). Any value
    ``> 1.0`` requires the caller to provide ``negative_text_embeddings``
    when building the cache (asserted at construction time)."""

    # Self-attention sliding-window / sink (in pre-patchify T frames).
    window_size_t: int = 21
    sink_size_t: int = 0

    # RoPE extrapolation ratios.
    h_extrapolation_ratio: float = 1.0
    w_extrapolation_ratio: float = 1.0

    # Speedup.
    compile_network: bool = False

    # I2V conditioning. The two flags are independent and can be combined:
    #
    # * ``stamp_image_latent``: at every denoising step, overwrite the
    #   noisy latent with the (clean) image-encoded latent at the
    #   positions flagged by the control mask, and re-stamp the
    #   predicted ``x0`` the same way in :meth:`postprocess_clean_latent`. The
    #   network's ``in_dim`` is unchanged. This is the flashdreams
    #   mask-inject recipe (used by ``causal_wan21``).
    # * ``concat_image_mask_to_latent``: append the 4-channel mask and
    #   the 16-channel image latent along the channel dim of the
    #   network input (``in_dim`` grows by ``4 + 16``). This matches the
    #   official Wan 2.1 14B I2V network layout.
    #
    # When both are enabled, the stamp runs first and the resulting
    # tensor is then concatenated with the mask + image latent.
    stamp_image_latent: bool = False
    concat_image_mask_to_latent: bool = False

    def __post_init__(self) -> None:
        assert self.guidance_scale >= 1.0, (
            f"guidance_scale must be >= 1.0 (got {self.guidance_scale})"
        )

        if self.concat_image_mask_to_latent:
            self.network.in_dim += 4 + 16


class Wan21Transformer(Transformer[Wan21TransformerCache]):
    """Wan 2.1 DiT adapted to the infra :class:`Transformer` interface."""

    def __init__(
        self,
        config: Wan21TransformerConfig,
        device: torch.device | None = None,
    ) -> None:
        super().__init__(config)
        self.config: Wan21TransformerConfig = config

        # Context-parallel groups -------------------------------------------------
        #
        # Same launcher contract as AlpaDreams (cp_size == world_size), but now wire
        # Wan's THW CP group to WORLD so all existing CP-aware Wan plumbing works.
        self.cp_group = None
        self.cp_size = 1
        if torch.distributed.is_initialized():
            world_size = torch.distributed.get_world_size()
            assert config.cp_size == world_size, (
                f"WanTransformerConfig.cp_size ({config.cp_size}) must match "
                f"torch.distributed.get_world_size() ({world_size})"
            )
            self.cp_size = world_size
            self.cp_group = torch.distributed.group.WORLD if world_size > 1 else None
        else:
            assert config.cp_size == 1, (
                f"WanTransformerConfig.cp_size must be 1 in non-distributed mode "
                f"(got {config.cp_size})"
            )

        # Token layout (pre-CP). Per-rank token count is on self.latent_shape[-2].
        kt, kh, kw = config.network.patch_size
        self._pT = config.len_t // kt
        self._pH = config.height // kh
        self._pW = config.width // kw
        total_tokens = self._pT * self._pH * self._pW
        assert total_tokens % self.cp_size == 0, (
            f"Wan token length ({total_tokens} from len_t={config.len_t}, "
            f"height={config.height}, width={config.width}, "
            f"patch_size={config.network.patch_size}) must be divisible by "
            f"cp_size={self.cp_size}"
        )

        # Network ----------------------------------------------------------------
        self.network = config.network.setup()
        if device is not None:
            self.network = self.network.to(device=device)
        self.network = self.network.to(dtype=config.dtype)
        self.network.eval()
        self.network.set_context_parallel_group(cp_group=self.cp_group)

        if config.checkpoint_path is not None:
            state_dict = load_checkpoint(config.checkpoint_path)
            if config.state_dict_transform is not None:
                state_dict = config.state_dict_transform(state_dict)  # ty:ignore[invalid-argument-type]
            self.network.load_state_dict(state_dict)  # ty:ignore[invalid-argument-type]
        self.network.update_parameters_after_loading_checkpoint()

        if config.compile_network:
            self.network = torch.compile(  # type: ignore[assignment]
                self.network, mode="max-autotune-no-cudagraphs"
            )

    # ------------------------------------------------------------------
    # Abstract method implementations
    # ------------------------------------------------------------------

    @property
    def latent_shape(self) -> tuple[int, ...]:
        """Per-rank latent shape ``[..., L, D]`` where ``L = pT*pH*pW / THW_size``.

        Wan 2.1 collapses ``T*H*W`` into one token dim and shards along
        ``THW_group``, so the per-rank token count is the full token
        grid divided by ``THW_size``.

        ``D`` reports the *noise* channel count (pre-patchify), not the
        full network input width: the mask / image latent appended by
        ``concat_image_mask_to_latent`` come from ``input`` inside
        :meth:`predict_flow`, not from the noise tensor sized here.
        """
        cfg = self.config
        kt, kh, kw = cfg.network.patch_size
        L = (self._pT * self._pH * self._pW) // self.cp_size
        # Strip the +(4 mask + 16 image latent) bump applied in
        # ``WanTransformerConfig.__post_init__`` so the bumped ``in_dim``
        # only reshapes the patch-embedding weight, not the noise tensor.
        noise_in_dim = cfg.network.in_dim
        if cfg.concat_image_mask_to_latent:
            noise_in_dim -= 4 + 16
        D = noise_in_dim * kt * kh * kw
        return (*cfg.batch_shape, L, D)

    @torch.no_grad()
    def _build_network_cache(
        self,
        *,
        text_embeddings: Tensor,
        image_embeddings: Tensor | None = None,
    ) -> WanDiTNetworkCache:
        """Build one ``WanDiTNetworkCache`` (cond *or* uncond branch).

        Exposed so users can construct exotic CFG variants (different chunk
        sizes, different image embedding, etc.) themselves and stitch them
        directly into a :class:`Wan21TransformerCache`.
        """
        cp_size = self.cp_size
        chunk_size = self.latent_shape[-2]  # already CP-divided
        window_size = (self.config.window_size_t * self._pH * self._pW) // cp_size
        sink_size = (self.config.sink_size_t * self._pH * self._pW) // cp_size
        return self.network.initialize_cache(  # ty:ignore[unresolved-attribute]
            chunk_size=chunk_size,
            window_size=window_size,
            sink_size=sink_size,
            text_embeddings=text_embeddings,
            img_embeddings=image_embeddings,
        )

    @torch.no_grad()
    def initialize_autoregressive_cache(  # type: ignore[override]
        self,
        *,
        text_embeddings: Tensor,
        image_embeddings: Tensor | None = None,
        negative_text_embeddings: Tensor | None = None,
        **_unused: Any,
    ) -> Wan21TransformerCache:
        """Build a fully seeded :class:`Wan21TransformerCache` for a new rollout.

        Carries no I2V state. The entire I2V signal (latent + binary
        injection mask) is plumbed per-AR-step through the infra
        as the ``input`` argument to :meth:`predict_flow` /
        :meth:`postprocess_clean_latent`; see :class:`I2VCtrl`
        for the channel layout.

        The CFG ``guidance_scale`` lives on :class:`WanTransformerConfig`,
        so this method only accepts the conditioning *content*. Whether to
        build an uncond branch is determined by ``self.config.guidance_scale``:

        - ``guidance_scale == 1.0``: no uncond branch is built;
          ``negative_text_embeddings`` MUST be ``None``.
        - ``guidance_scale  > 1.0``: ``negative_text_embeddings`` MUST be
          provided so the uncond branch can be built.

        For Wan we only support a *negative text prompt*; the same
        ``image_embeddings`` (when applicable for I2V) is shared by both
        branches — Wan's official inference does the same. Users who need
        a fully custom uncond branch should build it via
        :meth:`_build_network_cache` and construct
        :class:`Wan21TransformerCache` directly.

        Args:
            text_embeddings: Conditional UMT5 text embeddings, shape
                ``[..., text_len, text_dim]``.
            image_embeddings: Conditional CLIP image embeddings (only used
                by I2V networks where ``cross_attn_enable_img=True``).
                Shared by the uncond branch when CFG is enabled.
            negative_text_embeddings: Negative-prompt UMT5 embeddings.
                Required iff ``self.config.guidance_scale > 1.0``.

        Returns:
            A populated :class:`Wan21TransformerCache`. ``network_cache_uncond``
            is ``None`` iff CFG is disabled.
        """
        network_cache_cond = self._build_network_cache(
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
            len_t=self._pT,
            len_h=self._pH,
            len_w=self._pW,
            head_dim=head_dim,
            h_extrapolation_ratio=self.config.h_extrapolation_ratio,
            w_extrapolation_ratio=self.config.w_extrapolation_ratio,
            interleaved=True,
            device=self.device,
        )
        rope_adapter.set_context_parallel_group(cp_group=self.cp_group)

        return Wan21TransformerCache(
            network_cache_cond=network_cache_cond,
            network_cache_uncond=network_cache_uncond,
            rope_adapter=rope_adapter,
            len_t=self._pT,
            len_h=self._pH,
            len_w=self._pW,
        )

    def _stamp_image_latent(
        self,
        latent: Tensor,
        control: I2VCtrl,
    ) -> Tensor:
        """Overwrite ``latent`` with the image latent at masked positions.

        ``latent``, ``control.latent`` and ``control.mask`` all share the
        same patchified+CP-split shape (``[..., L, in_dim * K]``), so the
        per-token blend is a plain elementwise multiply::

            out = (1 - mask) * latent + mask * control.latent

        Returns the same shape as ``latent``. Used symmetrically by
        :meth:`predict_flow` (on the noisy input) and
        :meth:`postprocess_clean_latent` (on the predicted ``x0``).
        """
        return latent * (1.0 - control.mask) + control.latent * control.mask

    def predict_flow(
        self,
        noisy_latent: Tensor,
        timestep: Tensor,
        cache: Wan21TransformerCache,
        input: I2VCtrl | None = None,
        network_extra_kwargs: dict[str, Any] = {},
    ) -> Tensor:
        ar_idx = cache.autoregressive_index
        assert ar_idx >= 0, (
            "Wan21TransformerCache.start(autoregressive_index) must be called "
            "before predict_flow (DiffusionModel.generate handles this)."
        )
        rope_freqs = cache.rope_adapter.shift_t(offset=ar_idx * cache.len_t)

        # I2V conditioning. Two independent, composable modes:
        #
        # * ``stamp_image_latent`` (flashdreams mask-inject recipe used by
        #   ``causal_wan21``): overwrite ``noisy_latent`` with the clean
        #   image-encoded latent at the positions flagged by
        #   ``input.mask`` (e.g. the first temporal frame at AR step
        #   0). The network's ``in_dim`` is unchanged.
        # * ``concat_image_mask_to_latent`` (official Wan 2.1 14B I2V):
        #   the image latent + mask are appended along the channel dim.
        #   The network was trained with this layout
        #   (``in_dim=36 = 16+4+16``).
        #
        # When both are enabled, the stamp runs first and the resulting
        # tensor is then concatenated with the mask + image latent.
        # T2V (``input is None``) takes neither path.
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
            # Patchified mask carries the encoder's 16-channel uniform
            # tag; slicing the leading 16 entries gives the 4-channel
            # mask the official 14B I2V network expects (4 ch * K=4
            # patch entries = 16 trailing channels per token).
            mask = input.mask[..., :16]
            network_input = torch.cat([network_input, mask, input.latent], dim=-1)

        flow_cond = self.network(
            x=network_input,
            timesteps=timestep,
            cache=cache.network_cache_cond,
            rope_freqs=rope_freqs,
            current_chunk_idx=ar_idx,
            eager_mode=True,
            **network_extra_kwargs,
        )
        if cache.network_cache_uncond is None:
            return flow_cond

        flow_uncond = self.network(
            x=network_input,
            timesteps=timestep,
            cache=cache.network_cache_uncond,
            rope_freqs=rope_freqs,
            current_chunk_idx=ar_idx,
            eager_mode=True,
            **network_extra_kwargs,
        )
        return flow_uncond + self.config.guidance_scale * (flow_cond - flow_uncond)

    def postprocess_clean_latent(
        self,
        clean_latent: Tensor,
        cache: Wan21TransformerCache,
        input: I2VCtrl | None = None,
    ) -> Tensor:
        """Re-stamp the masked positions of ``x0`` with the image latent.

        Active only when the network is configured for the mask-inject
        I2V recipe (``stamp_image_latent=True``). The official Wan 2.1
        14B I2V (pure channel-concat mode) and pure T2V both fall
        through unchanged.
        """
        if input is None or not self.config.stamp_image_latent:
            return clean_latent
        return self._stamp_image_latent(clean_latent, input)

    def patchify_and_maybe_split_cp(self, x: Tensor | I2VCtrl) -> Tensor | I2VCtrl:
        """Patchify + CP-split a noisy latent or an I2V control payload.

        Dispatches on input type:

          - ``Tensor`` (the noisy latent or any plain-tensor control):
            delegates to :meth:`WanDiTNetwork.patchify_and_maybe_split_cp`.
          - :class:`I2VCtrl` (the I2V control payload from
            :class:`I2VCtrlEncoder`): patchifies both ``latent``
            and ``mask`` independently and returns a new
            :class:`I2VCtrl` carrying the patchified tensors.
            Splitting per field preserves the channel layouts (``in_dim``
            for ``latent``, ``1`` for ``mask``) so downstream
            :meth:`_stamp_image_latent` can broadcast the mask over the
            latent's channels without re-derivation.
        """
        if isinstance(x, I2VCtrl):
            if x._is_patchified:
                return x
            else:
                return I2VCtrl(
                    latent=self.patchify_and_maybe_split_cp(x.latent),  # ty:ignore[invalid-argument-type]
                    mask=self.patchify_and_maybe_split_cp(x.mask),  # ty:ignore[invalid-argument-type]
                    _is_patchified=True,
                )
        return self.network.patchify_and_maybe_split_cp(  # ty:ignore[unresolved-attribute]
            x,
            process_groups=[self.cp_group],
            cp_dims=[-2],
        )

    def unpatchify_and_maybe_gather_cp(self, x: Tensor) -> Tensor:
        return self.network.unpatchify_and_maybe_gather_cp(  # ty:ignore[unresolved-attribute]
            pH=self._pH,
            pW=self._pW,
            x=x,
            process_groups=[self.cp_group],
            cp_dims=[-2],
        )
