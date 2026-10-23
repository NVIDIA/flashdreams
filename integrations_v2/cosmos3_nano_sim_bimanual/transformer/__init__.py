# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""AgiBot MoT transformer implementation for the generic V2 diffusion model."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from flashdreams.infra.diffusion.transformer import Transformer

from .. import action_format as _agibot
from ..checkpoint import load_exported_checkpoint, resolve_credential_path
from ..encoder import HandActionInput
from ..fetch_checkpoint import ensure_checkpoint
from ..sdpa_backends import sdpa_kernel_context
from ..spec import SPEC
from .cache import HandTransformerCache
from .config import (
    DEFAULT_CHECKPOINT,
    HandTransformerConfig,
    validate_transformer_config,
)
from .network import CosmosMoTNetwork


class HandTransformer(Transformer[HandTransformerCache]):
    """Own the loaded MoT network and implement the generic transformer hooks."""

    def __init__(
        self,
        config: HandTransformerConfig,
        *,
        network: CosmosMoTNetwork | None = None,
    ) -> None:
        super().__init__(config)
        self.config: HandTransformerConfig = config
        validate_transformer_config(config, check_runtime=True)
        self.net = network if network is not None else self._load_network()
        self._active_cache: HandTransformerCache | None = None
        self._prepared_caches: dict[str, HandTransformerCache] = {}

    def _load_network(self) -> CosmosMoTNetwork:
        if self.config.compile_denoise:
            os.environ.setdefault("TORCHINDUCTOR_FX_GRAPH_CACHE", "1")
            os.environ.setdefault(
                "TORCHINDUCTOR_CACHE_DIR",
                str(Path.home() / ".cache" / "flashdreams" / "torchinductor"),
            )
        source = ensure_checkpoint(
            self.config.checkpoint or DEFAULT_CHECKPOINT,
            credential_path=self.config.credential_path,
        )
        credential = resolve_credential_path(self.config.credential_path)
        network = load_exported_checkpoint(
            source,
            device=self.config.device,
            credential_path=credential,
        ).eval()
        if not network.config.action_gen:
            raise RuntimeError(
                "checkpoint has no action pathway; hand inference requires an "
                "AgiBot action-conditioned checkpoint"
            )
        return network

    @property
    def device(self) -> torch.device:
        return self.net.embed_tokens.weight.device

    @property
    def dtype(self) -> torch.dtype:
        return self.net.embed_tokens.weight.dtype

    @property
    def latent_shape(self) -> tuple[int, ...]:
        return (
            1,
            SPEC.vfm.vae_latent_channels,
            self.config.chunk_size,
            _agibot.LATENT_HEIGHT,
            _agibot.LATENT_WIDTH,
        )

    def _new_cache(self, scene: Any) -> HandTransformerCache:
        return HandTransformerCache(
            self.net,
            scene=scene,
            text_ids=scene.text_ids,
            init_latent=scene.init_latent,
            grid_hw=_agibot.TOKEN_GRID,
            chunk_size=self.config.chunk_size,
            tcf=_agibot.TEMPORAL_COMPRESSION_FACTOR,
            action_domain_id=_agibot.AGIBOT_DOMAIN_ID,
            fps=scene.fps if self.config.model_fps is None else self.config.model_fps,
            logical_latent_hw=(_agibot.LATENT_HEIGHT, _agibot.LATENT_WIDTH),
            window_frames=self.config.window_frames,
            native_gqa_attention=self.config.native_gqa_attention,
            sdpa_backend=self.config.sdpa_backend,
            cache_mode=self.config.cache_mode,
            cache_refresh_mode=self.config.cache_refresh_mode,
            overlap_cache_update=self.config.overlap_cache_update,
            compile_denoise=self.config.compile_denoise,
            compile_denoise_mode=self.config.compile_denoise_mode,
            fixed_shape_cache_fill=self.config.fixed_shape_cache_fill,
            valid_length_cache_attention=self.config.valid_length_cache_attention,
            cuda_graph_denoise=self.config.cuda_graph_denoise,
            cuda_graph_warmup_iters=self.config.cuda_graph_warmup_iters,
            profile=self.config.profile,
        )

    def has_prepared_cache(self, scene: Any) -> bool:
        return str(scene.name) in self._prepared_caches

    def initialize_autoregressive_cache(self, *, scene: Any) -> HandTransformerCache:
        self.release_active_cache()
        cache = self._prepared_caches.pop(str(scene.name), None) or self._new_cache(
            scene
        )
        self._active_cache = cache
        return cache

    def release_active_cache(
        self,
        expected_cache: HandTransformerCache | None = None,
    ) -> None:
        cache = self._active_cache
        if cache is None or (
            expected_cache is not None and cache is not expected_cache
        ):
            return
        self._active_cache = None
        cache.reset_rollout()
        self._prepared_caches[str(cache.scene.name)] = cache

    def initial_noise(
        self,
        *,
        latent_shape: tuple[int, ...],
        rng: torch.Generator | None,
        cache: HandTransformerCache,
        input: Any = None,
    ) -> Tensor:
        del latent_shape
        if isinstance(input, HandActionInput):
            cache.prepare(input.actions, input.domain_ids)
        elif isinstance(input, Tensor):
            cache.prepare(input)
        else:
            raise TypeError(
                "the hand transformer requires HandActionInput or an action Tensor"
            )
        if cache._steps == 0 and rng is not None:
            rng.manual_seed(rng.initial_seed())
        return torch.randn(
            (1, cache.channels, cache._frames, cache.h_lat, cache.w_lat),
            device=self.device,
            dtype=torch.bfloat16,
            generator=rng,
        ).float()

    def predict_flow(
        self,
        noisy_latent: Tensor,
        timestep: Tensor,
        cache: HandTransformerCache,
        input: Any = None,
    ) -> Tensor:
        """Predict velocity in the latent dtype for checkpoint-faithful updates."""
        del input
        if noisy_latent.ndim != 5 or noisy_latent.shape[0] != 1:
            raise ValueError(
                f"expected latent [1,C,T,H,W], got {tuple(noisy_latent.shape)}"
            )
        latent = noisy_latent[0]
        if cache._compiled_args is not None:
            if cache._compiled_fill_active:
                call = cache._denoise_fill_fn
                if cache._denoise_fill_graph is not None:
                    call = cache._denoise_fill_graph
            else:
                call = cache._denoise_step_fn
            if cache._denoise_graph is not None and not cache._compiled_fill_active:
                call = (
                    cache._denoise_graph
                    if cache._steps >= cache._compile_start_step
                    else cache._denoise_graph.drain
                )
            sigma = timestep.to(device=self.device, dtype=torch.float32).reshape(1)
            with sdpa_kernel_context(cache.sdpa_backend):
                return (
                    call(latent, sigma, *cache._compiled_args)
                    .unsqueeze(0)
                    .to(noisy_latent.dtype)
                )

        assert cache._actions is not None
        assert cache._action_domain_ids is not None
        assert cache._positions_current is not None
        assert cache._is_gen is not None
        assert cache._is_action is not None
        h = self.net._assemble_input(
            None,
            cache._to_tokens(latent),
            timestep.to(device=self.device, dtype=torch.float32)
            .reshape(1)
            .expand(cache._vision_tokens),
            cache._is_gen,
            vision_noisy_mask=None,
            action_tokens=cache._actions,
            is_action=cache._is_action,
            action_domain_id=cache._action_domain_ids,
            uniform_vision_timestep=True,
        )
        with sdpa_kernel_context(cache.sdpa_backend):
            if cache.cache_mode == "block":
                h = cache._stream_layers(h, cache._positions_current, None)
            else:
                h = self.net._layers_step(
                    h,
                    cache._is_gen,
                    cache._positions_current,
                    cache._cat_cache(),
                    append=False,
                    within_mask=None,
                    native_gqa_attention=cache.native_gqa_attention,
                )
        h = self.net.norm_moe_gen(h)
        flow = cache._to_logical(
            self.net.llm2vae(h)[0][~cache._is_action], cache._frames
        )
        # The checkpoint predicts in BF16 but its denoising updates use FP32.
        return flow.unsqueeze(0).to(noisy_latent.dtype)

    def postprocess_clean_latent(
        self,
        clean_latent: Tensor,
        cache: HandTransformerCache,
        input: Any = None,
    ) -> Tensor:
        del input
        return cache.complete(clean_latent)

    def finalize_kv_cache(
        self,
        noisy_latent: Tensor,
        timestep: Tensor,
        cache: HandTransformerCache,
        input: Any = None,
    ) -> None:
        # Hands refreshes before decode so it can overlap with presentation.
        del noisy_latent, timestep, cache, input

    def patchify_and_maybe_split_cp(self, x: Any) -> Any:
        return x

    def unpatchify_and_maybe_gather_cp(self, x: Tensor) -> Tensor:
        return x


__all__ = ["HandTransformer", "HandTransformerCache", "HandTransformerConfig"]
