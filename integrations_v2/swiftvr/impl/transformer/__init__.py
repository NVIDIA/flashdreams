# SPDX-FileCopyrightText: Copyright (c) 2026 SwiftVR Authors.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# This file has been modified by NVIDIA CORPORATION & AFFILIATES.

"""SwiftVR transformer and per-stream overlap state."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor

from flashdreams.core.attention import RotaryPositionEmbedding3D
from flashdreams.core.checkpoint.load import load_checkpoint
from flashdreams.infra.diffusion.transformer import (
    Transformer,
    TransformerAutoregressiveCache,
    TransformerConfig,
)
from flashdreams.recipes.wan import wan_dit_state_dict_from_diffusers
from flashdreams.recipes.wan.transformer.impl.network import WanDiTNetworkCache
from swiftvr.impl.attention import prepare_transformer
from swiftvr.impl.transformer.network import (
    SwiftVRDiTNetwork,
    SwiftVRDiTNetworkConfig,
)

_INFERENCE_TIMESTEP = 1000.0


@dataclass(kw_only=True)
class SwiftVRTransformerConfig(TransformerConfig):
    """Configuration for the native WAN SwiftVR transformer."""

    _target: type["SwiftVRTransformer"] = field(
        default_factory=lambda: SwiftVRTransformer
    )

    network: SwiftVRDiTNetworkConfig = field(default_factory=SwiftVRDiTNetworkConfig)
    checkpoint_path: str | None = None
    """Diffusers-format WAN checkpoint path; ``None`` uses random test weights."""

    dtype: torch.dtype = torch.bfloat16
    """Transformer compute dtype."""

    compile_blocks: bool = False
    """Compile each SwiftVR block forward method."""

    batch_shape: tuple[int, ...] = (1,)
    """Batch dimensions used by the steady-state latent-shape contract."""

    latent_frames: int = 2
    """Steady-state latent frames; tail calls may contain fewer frames."""


@dataclass(kw_only=True)
class SwiftVRTransformerCache(TransformerAutoregressiveCache):
    """Per-stream prompt, overlap, and temporal-position state."""

    network_cache: WanDiTNetworkCache
    overlap: int
    previous_input: Tensor | None = None
    previous_output: Tensor | None = None
    time_offset: int = 0
    rope_freqs: Tensor | None = None
    shape: tuple[int, int, int] | None = None
    autoregressive_index: int = -1

    def start(self, autoregressive_index: int) -> None:
        self.autoregressive_index = autoregressive_index


class SwiftVRTransformer(Transformer[SwiftVRTransformerCache]):
    """One-step SwiftVR flow predictor over native FlashDreams WAN blocks."""

    def __init__(self, config: SwiftVRTransformerConfig) -> None:
        super().__init__(config)
        self.config: SwiftVRTransformerConfig = config
        self._output_height: int | None = None
        self._output_width: int | None = None

        if config.checkpoint_path is None:
            self.network = config.network.setup().to(dtype=config.dtype)
        else:
            with torch.device("meta"):
                self.network = config.network.setup()
            self.network.load_state_dict(
                wan_dit_state_dict_from_diffusers(
                    load_checkpoint(config.checkpoint_path, map_location="cpu")
                ),
                strict=True,
                assign=True,
            )
            self.network.to(dtype=config.dtype)
        self.network.update_parameters_after_loading_checkpoint()
        self.network.eval().requires_grad_(False)
        prepare_transformer(
            self.network,
            window=config.network.attention_window,
            compile_blocks=config.compile_blocks,
        )

    @property
    def latent_shape(self) -> tuple[int, ...]:
        """Return the steady-state post-patchify latent shape."""
        assert self._output_height is not None and self._output_width is not None, (
            "latent_shape requires initialize_autoregressive_cache() first"
        )
        kt, kh, kw = self.config.network.patch_size
        tokens = (
            self.config.latent_frames
            // kt
            * (self._output_height // kh)
            * (self._output_width // kw)
        )
        return (
            *self.config.batch_shape,
            tokens,
            self.config.network.out_dim * kt * kh * kw,
        )

    def initialize_autoregressive_cache(
        self,
        *,
        height: int,
        width: int,
        prompt_embedding: Tensor,
        overlap: int,
        **_unused: Any,
    ) -> SwiftVRTransformerCache:
        """Build one prompt-conditioned cache without unused full-size self KV."""
        if overlap < 0:
            raise ValueError(f"SwiftVR overlap must be non-negative, got {overlap}.")
        _, kh, kw = self.config.network.patch_size
        if height % kh or width % kw:
            raise ValueError(
                f"SwiftVR latent size {height}x{width} must be divisible by "
                f"patch size {(kh, kw)}."
            )
        self._output_height = height
        self._output_width = width
        prompt = prompt_embedding.to(device=self.device, dtype=self.dtype)
        if prompt.ndim == 2:
            prompt = prompt.unsqueeze(0).expand(*self.config.batch_shape, -1, -1)
        network_cache = self.network.initialize_cache(
            chunk_size=1,
            window_size=1,
            sink_size=0,
            text_embeddings=prompt,
        )
        return SwiftVRTransformerCache(
            network_cache=network_cache,
            overlap=overlap,
        )

    def patchify_and_maybe_split_cp(self, x: Tensor) -> Tensor:
        """Patchify ``[B,C,T,H,W]`` latents without context parallelism."""
        return self.network.patchify_and_maybe_split_cp(x.permute(0, 2, 1, 3, 4))

    def unpatchify_and_maybe_gather_cp(self, x: Tensor) -> Tensor:
        """Unpatchify tokens using the current rollout's spatial shape."""
        assert self._output_height is not None and self._output_width is not None
        _, kh, kw = self.config.network.patch_size
        return self.network.unpatchify_and_maybe_gather_cp(
            self._output_height // kh,
            self._output_width // kw,
            x,
        ).permute(0, 2, 1, 3, 4)

    def predict_flow(
        self,
        noisy_latent: Tensor,
        timestep: Tensor,
        cache: SwiftVRTransformerCache,
        input: Any = None,
    ) -> Tensor:
        """Predict one SwiftVR flow field for the active dynamic chunk."""
        del input
        assert cache.rope_freqs is not None and cache.shape is not None, (
            "SwiftVRTransformer.restore() must prepare shape and RoPE first"
        )
        return self.network(
            noisy_latent,
            timestep,
            cache.network_cache,
            cache.rope_freqs,
            eager_mode=False,
            block_extra_kwargs={"shape": cache.shape},
        )

    def finalize_kv_cache(
        self,
        noisy_latent: Tensor,
        timestep: Tensor,
        cache: SwiftVRTransformerCache,
        input: Any = None,
    ) -> None:
        """Skip WAN self-KV finalization; SwiftVR self-attention is window-local."""
        del noisy_latent, timestep, cache, input

    @torch.inference_mode()
    def restore(self, tensor: Tensor, cache: SwiftVRTransformerCache) -> Tensor:
        """Restore one ``[B,C,T,H,W]`` latent chunk with optional overlap."""
        batch, _, frames, height, width = tensor.shape
        overlap = 0
        if cache.previous_input is not None and cache.overlap:
            overlap = cache.previous_input.shape[2]
            extended = torch.cat(
                [cache.previous_input.to(tensor.device), tensor], dim=2
            )
        else:
            extended = tensor

        patch_time, patch_height, patch_width = self.network.patch_size
        patch_shape = (
            extended.shape[2] // patch_time,
            height // patch_height,
            width // patch_width,
        )
        cache.shape = patch_shape
        cache.rope_freqs = _rope_with_offset(
            self.network,
            *patch_shape,
            time_offset=cache.time_offset - overlap,
        )
        hidden = self.patchify_and_maybe_split_cp(extended)
        timestep = torch.full(
            (batch,),
            _INFERENCE_TIMESTEP,
            device=tensor.device,
            dtype=torch.float32,
        )
        prediction = self.predict_flow(hidden, timestep, cache)
        prediction = self.network.unpatchify_and_maybe_gather_cp(
            patch_shape[1], patch_shape[2], prediction
        ).permute(0, 2, 1, 3, 4)
        restored = extended - prediction

        if overlap and cache.previous_output is not None:
            blend = torch.linspace(
                0, 1, overlap, device=tensor.device, dtype=tensor.dtype
            ).view(1, 1, overlap, 1, 1)
            restored[:, :, :overlap] = (
                cache.previous_output.to(tensor.device) * (1 - blend)
                + restored[:, :, :overlap] * blend
            )
            restored = restored[:, :, overlap:]

        retained = min(cache.overlap, frames)
        if retained:
            cache.previous_input = tensor[:, :, -retained:].detach().cpu().clone()
            cache.previous_output = restored[:, :, -retained:].detach().cpu().clone()
        cache.time_offset += frames
        return restored


def _rope_with_offset(
    transformer: SwiftVRDiTNetwork,
    frames: int,
    height: int,
    width: int,
    *,
    time_offset: int,
) -> Tensor:
    rope = RotaryPositionEmbedding3D(
        head_dim=transformer.dim // transformer.num_heads,
        len_t=1,
        len_h=height,
        len_w=width,
        interleaved=True,
        device=transformer.patch_embedding.weight.device,
    )
    return torch.cat(
        [rope.shift_t(time_offset + frame) for frame in range(frames)], dim=0
    ).contiguous()


__all__ = [
    "SwiftVRTransformer",
    "SwiftVRTransformerCache",
    "SwiftVRTransformerConfig",
]
