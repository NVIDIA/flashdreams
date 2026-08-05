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

"""FlashDreams adapter for the SANA-WM Stage-1 DiT."""

from __future__ import annotations

import gc
import time
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Literal, cast

import torch
import torch.nn as nn
import yaml
from loguru import logger
from torch import Tensor

from flashdreams.core.checkpoint.load import load_checkpoint
from flashdreams.core.io.hf import maybe_download_hf_repo_on_rank0
from flashdreams.infra.diffusion.transformer import (
    Transformer,
    TransformerAutoregressiveCache,
    TransformerConfig,
)
from sana_wm._tools import resolve_hf_path
from sana_wm.constants import (
    DEFAULT_VIDEO_HEIGHT,
    DEFAULT_VIDEO_WIDTH,
    SANA_WM_CONFIG_PATH,
    SANA_WM_MODEL_PATH,
    SANA_WM_STREAMING_CAUSAL_VAE_ROOT,
    SANA_WM_STREAMING_CONFIG_PATH,
    SANA_WM_STREAMING_LATENT_CHUNK_SIZE,
    SANA_WM_STREAMING_MODEL_PATH,
)
from sana_wm.quant import (
    TorchScaledMMFP4Recipe,
    TorchScaledMMFP8Recipe,
    replace_linear_with_quant,
)
from sana_wm.stage1_model import (
    SANA_WM_STREAMING_STAGE1_SPEC,
    SanaWMStage1Model,
    linearize_stage1_ffn_for_quant,
)

Precision = Literal["bf16", "fp8", "fp4"]
QuantBackend = Literal["auto", "torch", "torch-fp8", "torch-fp4"]

_STAGE1_QUANT_SKIP_DEFAULTS = (
    "^x_embedder",
    "^raymap_embedder",
    "^plucker_embedder",
    "^t_embedder",
    "^t_block",
    "^y_embedder",
    "^final_layer",
)
_STAGE1_QUANT_INCLUDE_DEFAULTS = (
    r"^blocks\.\d+\.attn\.qkv$",
    r"^blocks\.\d+\.attn\.proj$",
    r"^blocks\.\d+\.attn\.beta_proj$",
    r"^blocks\.\d+\.attn\.gate_proj$",
    r"^blocks\.\d+\.attn\.output_gate$",
    r"^blocks\.\d+\.cross_attn\.",
    r"^blocks\.\d+\.mlp\.inverted_conv\.linear$",
    r"^blocks\.\d+\.mlp\.point_conv\.linear$",
)


def _stage1_state_dict(state: Any) -> dict[str, Tensor]:
    """Normalize a Stage-1 checkpoint payload into a plain state dict.

    The bidirectional release ships flat safetensors weights, while the
    streaming release ships a ``.pt`` that nests the tensors under
    ``generator`` and/or ``state_dict`` and prefixes keys with ``model.``.

    Args:
        state: Payload returned by ``load_checkpoint``.

    Returns:
        Tensor state dict accepted by ``load_state_dict``.
    """
    if "generator" in state:
        state = state["generator"]
    if "state_dict" in state:
        return cast("dict[str, Tensor]", state["state_dict"])
    return {
        (key[len("model.") :] if key.startswith("model.") else key): value
        for key, value in state.items()
    }


@dataclass(kw_only=True)
class SanaWMStage1Conditioning:
    """Per-rollout inputs needed by the SANA-WM Stage-1 sampler."""

    condition: Tensor
    uncondition: Tensor | None
    model_kwargs: dict[str, object]
    first_latent: Tensor
    latent_shape: tuple[int, int, int, int, int]
    cfg_scale: float
    flow_shift: float
    steps: int
    seed: int


@dataclass(kw_only=True)
class SanaWMStreamingStage1Conditioning(SanaWMStage1Conditioning):
    """Per-chunk Stage-1 inputs for SANA-WM streaming inference."""

    total_latent_shape: tuple[int, int, int, int, int]
    start_frame: int
    end_frame: int
    chunk_index: int
    chunk_boundaries: tuple[int, ...]

    @property
    def num_chunks(self) -> int:
        """Return the number of AR chunks in this rollout."""
        return len(self.chunk_boundaries) - 1


@dataclass(kw_only=True)
class SanaWMTransformerCache(TransformerAutoregressiveCache):
    """AR cache for the one-shot SANA-WM Stage-1 rollout."""

    conditioning: SanaWMStage1Conditioning | None = None


@dataclass(kw_only=True)
class SanaWMStreamingTransformerCache(SanaWMTransformerCache):
    """AR cache for streaming SANA-WM Stage-1 prefix sampling."""

    latent_state: Tensor | None = None
    initial_latent_state: Tensor | None = None


@dataclass(frozen=True)
class _StreamingStage1Window:
    """Local Stage-1 frame window for one streaming AR step."""

    frame_indices: tuple[int, ...]
    active_start: int
    active_end: int


@dataclass(kw_only=True)
class SanaWMTransformerConfig(TransformerConfig):
    """Config for the SANA-WM Stage-1 transformer adapter."""

    _target: type["SanaWMTransformer"] = field(
        default_factory=lambda: SanaWMTransformer
    )

    config_path: str = SANA_WM_CONFIG_PATH
    """SANA-WM inference YAML path or ``hf://`` URI."""

    checkpoint_path: str = SANA_WM_MODEL_PATH
    """SANA-WM Stage-1 checkpoint path, ``s3://`` URI, or Hugging Face file URL."""

    stage1_precision: Precision = "bf16"
    """Stage-1 precision requested by the runner."""

    quant_backend: QuantBackend = "auto"
    """Low-precision linear replacement backend for FP8/FP4 modes."""

    height: int = DEFAULT_VIDEO_HEIGHT
    """Pixel height used by the public SANA-WM bidirectional release."""

    width: int = DEFAULT_VIDEO_WIDTH
    """Pixel width used by the public SANA-WM bidirectional release."""

    offload_stage1: bool = False
    """Tear down the Stage-1 DiT after sampling, before decode/refine. Default
    keeps it resident (fastest when memory is available); enable only to free
    Stage-1 memory for the VAE/refiner on memory-constrained GPUs."""


@dataclass(kw_only=True)
class SanaWMStreamingTransformerConfig(SanaWMTransformerConfig):
    """Config for the SANA-WM streaming Stage-1 transformer adapter."""

    _target: type["SanaWMStreamingTransformer"] = field(
        default_factory=lambda: SanaWMStreamingTransformer
    )

    config_path: str = SANA_WM_STREAMING_CONFIG_PATH
    """SANA-WM streaming inference config path or built-in identifier."""

    checkpoint_path: str = SANA_WM_STREAMING_MODEL_PATH
    """Streaming Stage-1 checkpoint path, ``s3://`` URI, or Hugging Face file URL."""

    num_frame_per_block: int = SANA_WM_STREAMING_LATENT_CHUNK_SIZE
    """Latent frames generated per steady-state AR block."""

    num_cached_blocks: int = 2
    """Stage-1 streaming context-window hint retained on the config surface."""

    sink_token: bool = True
    """Keep the first chunk as the Stage-1 streaming sink anchor."""


class SanaWMTransformer(Transformer[SanaWMTransformerCache]):
    """FlashDreams adapter for the SANA-WM Stage-1 model call."""

    def __init__(self, config: SanaWMTransformerConfig) -> None:
        super().__init__(config)
        self.config: SanaWMTransformerConfig = config
        self._dummy = nn.Parameter(torch.empty(0))
        self._runtime_config: Any | None = None
        self._model_path: str | None = None
        self.weight_dtype: torch.dtype | None = None
        self._model_built = False
        self._stage1_quantized = False

    @property
    def latent_shape(self) -> tuple[int, ...]:
        """Return the current rollout latent shape or the public default."""
        if self._runtime_config is None:
            return (1, 128, 21, 22, 40)
        cfg = self._ensure_runtime_config()
        latent_t = (161 - 1) // int(cfg.vae.vae_stride[0]) + 1
        return (
            1,
            int(cfg.vae.vae_latent_dim),
            latent_t,
            self.config.height // int(cfg.vae.vae_stride[-1]),
            self.config.width // int(cfg.vae.vae_stride[-1]),
        )

    def initialize_autoregressive_cache(
        self,
        *,
        conditioning: SanaWMStage1Conditioning | None = None,
        **_: Any,
    ) -> SanaWMTransformerCache:
        """Build a cache containing per-rollout SANA-WM conditioning."""
        return SanaWMTransformerCache(conditioning=conditioning)

    def predict_flow(
        self,
        noisy_latent: Tensor,
        timestep: Tensor,
        cache: SanaWMTransformerCache,
        input: Any = None,
        model_kwargs: dict[str, object] | None = None,
    ) -> Tensor:
        """Execute one SANA-WM DiT flow prediction."""
        if isinstance(input, SanaWMStage1Conditioning):
            cache.conditioning = input
            return self._predict_conditioned(
                noisy_latent=noisy_latent,
                timestep=timestep,
                conditioning=input,
            )

        conditioning = _require_conditioning(cache)
        prompt_embeds = (
            cast(Tensor, input) if input is not None else conditioning.condition
        )
        kwargs = conditioning.model_kwargs if model_kwargs is None else model_kwargs
        return self._predict_with_prompt(
            noisy_latent,
            timestep,
            prompt_embeds,
            kwargs,
        )

    def finalize_kv_cache(
        self,
        noisy_latent: Tensor,
        timestep: Tensor,
        cache: SanaWMTransformerCache,
        input: Any = None,
    ) -> None:
        """SANA-WM bidirectional inference has no streaming KV cache to advance."""
        del noisy_latent, timestep, cache, input

    def patchify_and_maybe_split_cp(self, x: Any) -> Any:
        """SANA-WM Stage-1 latents are already in model layout."""
        return x

    def unpatchify_and_maybe_gather_cp(self, x: Tensor) -> Tensor:
        """SANA-WM Stage-1 latents are already in model layout."""
        return x

    def initial_noise(
        self,
        *,
        latent_shape: tuple[int, ...],
        rng: torch.Generator | None,
        cache: SanaWMTransformerCache,
        input: Any = None,
    ) -> Tensor:
        """Draw SANA-WM's first-frame-pinned Stage-1 initial latent."""
        del latent_shape, rng
        if isinstance(input, SanaWMStage1Conditioning):
            cache.conditioning = input
            conditioning = input
        else:
            conditioning = _require_conditioning(cache)
        return self.initial_latents(conditioning)

    def postprocess_clean_latent(
        self,
        clean_latent: Tensor,
        cache: SanaWMTransformerCache,
        input: Any = None,
    ) -> Tensor:
        """Release Stage-1 runtime before decode/refine only when offloading.

        By default the Stage-1 DiT stays resident so a reused pipeline never
        reloads and re-quantizes it; ``offload_stage1`` restores the old
        free-before-decode behavior for memory-constrained GPUs.
        """
        del input
        if self.config.offload_stage1:
            self.release_stage1_runtime(cache)
        elif cache is not None:
            cache.conditioning = None
        return clean_latent

    def initial_latents(self, conditioning: SanaWMStage1Conditioning) -> Tensor:
        """Draw Stage-1 initial noise and pin the encoded first frame."""
        generator = torch.Generator(device=self.device).manual_seed(conditioning.seed)
        latents = torch.randn(
            conditioning.latent_shape,
            dtype=self._ensure_weight_dtype(),
            device=self.device,
            generator=generator,
        )
        latents[:, :, :1] = conditioning.first_latent
        return latents

    def release_stage1_runtime(
        self,
        cache: SanaWMTransformerCache | None = None,
    ) -> None:
        """Release Stage-1-only tensors before VAE/refiner work."""
        free_before_gib: float | None = None
        if torch.cuda.is_available():
            try:
                free_before, _total = torch.cuda.mem_get_info()
                free_before_gib = free_before / (1024**3)
            except RuntimeError:
                free_before_gib = None
        if cache is not None:
            cache.conditioning = None

        for attr in ("model",):
            module = getattr(self, attr, None)
            if module is None:
                continue
            try:
                module.to("meta")
            except Exception:
                try:
                    module.to("cpu")
                except Exception:
                    pass
            setattr(self, attr, None)

        self._model_built = False
        self._stage1_quantized = False
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            try:
                free_after, _total = torch.cuda.mem_get_info()
                if free_before_gib is not None:
                    logger.info(
                        "[stage1] released Stage-1 runtime before decode/refine "
                        "(free CUDA memory: {:.2f} -> {:.2f} GiB)",
                        free_before_gib,
                        free_after / (1024**3),
                    )
                else:
                    logger.info(
                        "[stage1] released Stage-1 runtime before decode/refine"
                    )
            except RuntimeError:
                logger.info("[stage1] released Stage-1 runtime before decode/refine")
        gc.collect()

    def _predict_conditioned(
        self,
        *,
        noisy_latent: Tensor,
        timestep: Tensor,
        conditioning: SanaWMStage1Conditioning,
    ) -> Tensor:
        self._prepare_static_model_kwargs(conditioning)
        model_timestep = _conditioned_frame_timestep(
            noisy_latent=noisy_latent,
            timestep=timestep,
            conditioning=conditioning,
            num_train_timesteps=1000,
        )
        if conditioning.cfg_scale <= 1.0:
            flow = self._predict_with_prompt(
                noisy_latent,
                model_timestep,
                conditioning.condition,
                _condition_model_kwargs(conditioning.model_kwargs),
            )
            return _zero_conditioned_frame_flow(flow, conditioning)
        if conditioning.uncondition is None:
            raise RuntimeError("CFG was requested without negative prompt embeds.")

        noise_pred = self._predict_with_prompt(
            torch.cat([noisy_latent, noisy_latent], dim=0),
            torch.cat([model_timestep, model_timestep], dim=0),
            torch.cat([conditioning.uncondition, conditioning.condition], dim=0),
            _batched_cfg_model_kwargs(conditioning.model_kwargs),
        )
        flow = _cfg_guidance(noise_pred, conditioning.cfg_scale)
        return _zero_conditioned_frame_flow(flow, conditioning)

    def _prepare_static_model_kwargs(
        self,
        conditioning: SanaWMStage1Conditioning,
    ) -> None:
        """Cache model projections for conditioning that is constant per rollout."""
        model_kwargs = conditioning.model_kwargs
        self._ensure_model()
        chunk_plucker = model_kwargs.get("chunk_plucker")
        prepare_plucker = getattr(self.model, "prepare_plucker_embedding", None)
        if (
            "chunk_plucker_emb" not in model_kwargs
            and isinstance(chunk_plucker, Tensor)
            and callable(prepare_plucker)
        ):
            with torch.inference_mode():
                model_kwargs["chunk_plucker_emb"] = prepare_plucker(chunk_plucker)
            del model_kwargs["chunk_plucker"]

        camera_conditions = model_kwargs.get("camera_conditions")
        prepare_camera = getattr(self.model, "prepare_camera_projection_cache", None)
        if (
            "camera_cache" not in model_kwargs
            and isinstance(camera_conditions, Tensor)
            and camera_conditions.shape[0] == 1
            and callable(prepare_camera)
        ):
            _batch, _channels, frames, height, width = conditioning.latent_shape
            with torch.inference_mode():
                rotary_emb, camera_cache = prepare_camera(
                    camera_conditions,
                    frames=frames,
                    height=height,
                    width=width,
                )
            model_kwargs["rotary_emb"] = rotary_emb
            model_kwargs["camera_cache"] = camera_cache

    def _predict_with_prompt(
        self,
        noisy_latent: Tensor,
        timestep: Tensor,
        prompt_embeds: Tensor,
        model_kwargs: dict[str, object],
    ) -> Tensor:
        self._ensure_model()
        output = self.model(
            noisy_latent,
            timestep,
            prompt_embeds,
            **model_kwargs,
        )
        if isinstance(output, tuple):
            output = output[0]
        return cast(Tensor, output)

    def _ensure_runtime_config(self) -> Any:
        if self._runtime_config is not None:
            return self._runtime_config
        self._runtime_config = _load_inference_config(self.config.config_path)
        return self._runtime_config

    def _ensure_weight_dtype(self) -> torch.dtype:
        if self.weight_dtype is None:
            self.weight_dtype = _get_weight_dtype(
                self._ensure_runtime_config().model.mixed_precision
            )
        return self.weight_dtype

    def _ensure_model(self) -> None:
        if self._model_built:
            self._prepare_stage1_quant()
            return
        t0 = time.perf_counter()
        cfg = self._ensure_runtime_config()
        weight_dtype = self._ensure_weight_dtype()
        model = SanaWMStage1Model().to(self.device)
        logger.info(
            "[Sana] Loaded {} ({:,} params)",
            cfg.model.model,
            sum(p.numel() for p in model.parameters()),
        )
        self._model_path = self.config.checkpoint_path
        state_dict = _stage1_state_dict(load_checkpoint(self._model_path))
        missing, unexpected = model.load_state_dict(state_dict, strict=True)
        if missing:
            logger.warning("[Sana] Missing keys: {}", missing)
        if unexpected:
            logger.warning("[Sana] Unexpected keys: {}", unexpected)
        self.model = model.eval().to(weight_dtype)
        self._model_built = True
        self._prepare_stage1_quant()
        logger.info(
            "[timing] stage1 build+load+quant: {:.3f}s (precision={})",
            time.perf_counter() - t0,
            self.config.stage1_precision,
        )

    def _prepare_stage1_quant(self) -> None:
        if self._stage1_quantized or self.config.stage1_precision == "bf16":
            return
        if self.config.stage1_precision == "fp8":
            recipe = TorchScaledMMFP8Recipe()
        else:
            recipe = TorchScaledMMFP4Recipe()
        linearized, linearize_skipped = linearize_stage1_ffn_for_quant(self.model)
        if linearized > 0 or linearize_skipped > 0:
            logger.info(
                "[stage1-quant] linearized {} FFN pointwise convs (skipped {})",
                linearized,
                linearize_skipped,
            )
        converted, skipped = replace_linear_with_quant(
            self.model,
            recipe=recipe,
            params_dtype=self._ensure_weight_dtype(),
            skip_patterns=_STAGE1_QUANT_SKIP_DEFAULTS,
            include_patterns=_stage1_quant_include_patterns(),
        )
        if converted <= 0:
            raise RuntimeError(
                f"SANA-WM {self.config.stage1_precision} converted no "
                f"Stage-1 Linear layers; skipped={skipped}."
            )
        self._stage1_quantized = True
        recipe_detail = ""
        if isinstance(recipe, TorchScaledMMFP4Recipe):
            recipe_detail = (
                f" rht={recipe.use_rht}"
                f" global_scale={recipe.use_global_scale}"
                f" weight_scale_2d={recipe.weight_scale_2d}"
            )
        logger.info(
            "[stage1-quant] precision={}{} converted {} Linear layers (skipped {})",
            self.config.stage1_precision,
            recipe_detail,
            converted,
            skipped,
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class SanaWMStreamingTransformer(SanaWMTransformer):
    """FlashDreams adapter for SANA-WM streaming Stage-1 chunks.

    This implementation keeps the full latent rollout in the transformer
    cache and recomputes the available prefix for each chunk. That preserves
    the FlashDreams AR lifecycle without depending on upstream cached module
    classes.
    """

    config: SanaWMStreamingTransformerConfig

    def __init__(self, config: SanaWMStreamingTransformerConfig) -> None:
        super().__init__(config)
        self.config = config
        self._current_latent_shape: tuple[int, ...] | None = None

    @property
    def latent_shape(self) -> tuple[int, ...]:
        """Return the current streaming chunk latent shape."""
        if self._current_latent_shape is not None:
            return self._current_latent_shape
        return (
            1,
            128,
            int(self.config.num_frame_per_block) + 1,
            self.config.height // 32,
            self.config.width // 32,
        )

    def initialize_autoregressive_cache(
        self,
        *,
        conditioning: SanaWMStreamingStage1Conditioning | None = None,
        **_: Any,
    ) -> SanaWMStreamingTransformerCache:
        """Build a cache for a streaming SANA-WM rollout."""
        self._current_latent_shape = (
            conditioning.latent_shape if conditioning is not None else None
        )
        return SanaWMStreamingTransformerCache(conditioning=conditioning)

    def patchify_and_maybe_split_cp(self, x: Any) -> Any:
        """SANA-WM streaming latents are already in model layout."""
        if isinstance(x, SanaWMStreamingStage1Conditioning):
            self._current_latent_shape = x.latent_shape
        return x

    def initial_noise(
        self,
        *,
        latent_shape: tuple[int, ...],
        rng: torch.Generator | None,
        cache: SanaWMTransformerCache,
        input: Any = None,
    ) -> Tensor:
        """Return this AR chunk's initial noisy latent."""
        if not isinstance(input, SanaWMStreamingStage1Conditioning):
            return super().initial_noise(
                latent_shape=latent_shape,
                rng=rng,
                cache=cache,
                input=input,
            )
        streaming_cache = _require_streaming_cache(cache)
        streaming_cache.conditioning = input
        self._current_latent_shape = input.latent_shape
        latent_state = self._ensure_streaming_latent_state(streaming_cache, input)
        return latent_state[:, :, input.start_frame : input.end_frame].clone()

    def predict_flow(
        self,
        noisy_latent: Tensor,
        timestep: Tensor,
        cache: SanaWMTransformerCache,
        input: Any = None,
        model_kwargs: dict[str, object] | None = None,
    ) -> Tensor:
        """Execute one streaming SANA-WM Stage-1 flow prediction."""
        if not isinstance(input, SanaWMStreamingStage1Conditioning):
            return super().predict_flow(
                noisy_latent=noisy_latent,
                timestep=timestep,
                cache=cache,
                input=input,
                model_kwargs=model_kwargs,
            )
        del model_kwargs
        streaming_cache = _require_streaming_cache(cache)
        streaming_cache.conditioning = input
        return self._predict_streaming_conditioned(
            noisy_latent=noisy_latent,
            timestep=timestep,
            cache=streaming_cache,
            conditioning=input,
        )

    def postprocess_clean_latent(
        self,
        clean_latent: Tensor,
        cache: SanaWMTransformerCache,
        input: Any = None,
    ) -> Tensor:
        """Commit a finished streaming chunk to the prefix latent cache."""
        if not isinstance(input, SanaWMStreamingStage1Conditioning):
            return super().postprocess_clean_latent(clean_latent, cache, input)
        streaming_cache = _require_streaming_cache(cache)
        latent_state = self._ensure_streaming_latent_state(streaming_cache, input)
        latent_state[:, :, input.start_frame : input.end_frame] = clean_latent
        if input.start_frame == 0:
            latent_state[:, :, :1] = input.first_latent.to(
                device=latent_state.device,
                dtype=latent_state.dtype,
            )
            clean_latent[:, :, :1] = latent_state[:, :, :1]

        if input.chunk_index == input.num_chunks - 1 and self.config.offload_stage1:
            self.release_stage1_runtime(streaming_cache)
        else:
            streaming_cache.conditioning = None
        return clean_latent

    def finalize_kv_cache(
        self,
        noisy_latent: Tensor,
        timestep: Tensor,
        cache: SanaWMTransformerCache,
        input: Any = None,
    ) -> None:
        """Prefix recomputation keeps streaming context in ``latent_state``."""
        del noisy_latent, timestep, cache, input

    def _ensure_model(self) -> None:
        if self._model_built:
            self._prepare_stage1_quant()
            return
        t0 = time.perf_counter()
        cfg = self._ensure_runtime_config()
        weight_dtype = self._ensure_weight_dtype()
        model = SanaWMStage1Model(SANA_WM_STREAMING_STAGE1_SPEC).to(self.device)
        logger.info(
            "[Sana] Loaded {} ({:,} params)",
            cfg.model.model,
            sum(p.numel() for p in model.parameters()),
        )
        self._model_path = self.config.checkpoint_path
        state_dict = _stage1_state_dict(load_checkpoint(self._model_path))
        missing, unexpected = model.load_state_dict(state_dict, strict=True)
        if missing:
            logger.warning("[Sana] Missing keys: {}", missing)
        if unexpected:
            logger.warning("[Sana] Unexpected keys: {}", unexpected)
        self.model = model.eval().to(weight_dtype)
        self._model_built = True
        self._prepare_stage1_quant()
        logger.info(
            "[timing] streaming stage1 build+load+quant: {:.3f}s (precision={})",
            time.perf_counter() - t0,
            self.config.stage1_precision,
        )

    def _ensure_streaming_latent_state(
        self,
        cache: SanaWMStreamingTransformerCache,
        conditioning: SanaWMStreamingStage1Conditioning,
    ) -> Tensor:
        if cache.latent_state is None:
            generator = torch.Generator(device=self.device).manual_seed(
                conditioning.seed
            )
            cache.latent_state = torch.randn(
                conditioning.total_latent_shape,
                dtype=self._ensure_weight_dtype(),
                device=self.device,
                generator=generator,
            )
            cache.latent_state[:, :, :1] = conditioning.first_latent.to(
                device=self.device,
                dtype=cache.latent_state.dtype,
            )
            cache.initial_latent_state = cache.latent_state.clone()
        return cache.latent_state

    def _predict_streaming_conditioned(
        self,
        *,
        noisy_latent: Tensor,
        timestep: Tensor,
        cache: SanaWMStreamingTransformerCache,
        conditioning: SanaWMStreamingStage1Conditioning,
    ) -> Tensor:
        latent_state = self._ensure_streaming_latent_state(cache, conditioning)
        start = int(conditioning.start_frame)
        end = int(conditioning.end_frame)
        window = _streaming_stage1_window(conditioning, self.config)
        window_index = torch.tensor(
            window.frame_indices,
            dtype=torch.long,
            device=latent_state.device,
        )
        model_latent = latent_state.index_select(2, window_index).clone()
        model_latent[:, :, window.active_start : window.active_end] = noisy_latent
        if 0 in window.frame_indices:
            sink_local = window.frame_indices.index(0)
            model_latent[:, :, sink_local : sink_local + 1] = (
                conditioning.first_latent.to(
                    device=model_latent.device,
                    dtype=model_latent.dtype,
                )
            )
        model_timestep = _streaming_window_timestep(
            noisy_latent=noisy_latent,
            timestep=timestep,
            conditioning=conditioning,
            frame_indices=window.frame_indices,
            active_start=window.active_start,
            active_end=window.active_end,
        )
        kwargs = _streaming_window_model_kwargs(
            conditioning.model_kwargs,
            frame_indices=window.frame_indices,
            total_frames=int(conditioning.total_latent_shape[2]),
            chunk_size=int(self.config.num_frame_per_block),
        )
        if conditioning.cfg_scale <= 1.0:
            flow_window = self._predict_with_prompt(
                model_latent,
                model_timestep,
                conditioning.condition,
                _condition_model_kwargs(kwargs),
            )
        else:
            if conditioning.uncondition is None:
                raise RuntimeError("CFG was requested without negative prompt embeds.")
            flow_window = self._predict_with_prompt(
                torch.cat([model_latent, model_latent], dim=0),
                torch.cat([model_timestep, model_timestep], dim=0),
                torch.cat([conditioning.uncondition, conditioning.condition], dim=0),
                _batched_cfg_model_kwargs(kwargs),
            )
            flow_window = _cfg_guidance(flow_window, conditioning.cfg_scale)

        flow = flow_window[:, :, window.active_start : window.active_end]
        if start == 0:
            flow[:, :, :1] = 0
        return flow.to(dtype=noisy_latent.dtype)


def _stage1_quant_include_patterns() -> tuple[str, ...]:
    """Return Stage-1 Linear names eligible for FP8/FP4 quantization."""
    return _STAGE1_QUANT_INCLUDE_DEFAULTS


def _require_conditioning(
    cache: SanaWMTransformerCache,
) -> SanaWMStage1Conditioning:
    if cache.conditioning is None:
        raise RuntimeError("SANA-WM cache was initialized without conditioning.")
    return cache.conditioning


def _require_streaming_cache(
    cache: SanaWMTransformerCache,
) -> SanaWMStreamingTransformerCache:
    if not isinstance(cache, SanaWMStreamingTransformerCache):
        raise TypeError(
            "SANA-WM streaming transformer requires SanaWMStreamingTransformerCache."
        )
    return cache


def _condition_model_kwargs(model_kwargs: dict[str, object]) -> dict[str, object]:
    """Return model kwargs for the positive prompt branch."""
    return {
        key: value
        for key, value in model_kwargs.items()
        if key != "negative_mask" and not key.startswith("_")
    }


def _batched_cfg_model_kwargs(model_kwargs: dict[str, object]) -> dict[str, object]:
    """Return model kwargs for a single batched negative/positive CFG forward."""
    kwargs = _condition_model_kwargs(model_kwargs)
    mask = kwargs.get("mask")
    negative_mask = model_kwargs.get("negative_mask")
    if isinstance(mask, Tensor):
        mask_uncond = negative_mask if isinstance(negative_mask, Tensor) else mask
        kwargs["mask"] = torch.cat([mask_uncond, mask], dim=0)
    camera_conditions = kwargs.get("camera_conditions")
    if isinstance(camera_conditions, Tensor) and camera_conditions.shape[0] != 1:
        kwargs["camera_conditions"] = torch.cat(
            [camera_conditions, camera_conditions],
            dim=0,
        )
    chunk_plucker = kwargs.get("chunk_plucker")
    if isinstance(chunk_plucker, Tensor):
        kwargs["chunk_plucker"] = torch.cat([chunk_plucker, chunk_plucker], dim=0)
    return kwargs


def _streaming_prefix_model_kwargs(
    model_kwargs: dict[str, object],
    prefix_frames: int,
) -> dict[str, object]:
    """Slice full-rollout model kwargs to the currently visible prefix."""
    result: dict[str, object] = {}
    for key, value in model_kwargs.items():
        if isinstance(value, Tensor):
            if value.ndim == 5 and value.shape[2] >= prefix_frames:
                result[key] = value[:, :, :prefix_frames].contiguous()
            elif value.ndim >= 3 and value.shape[1] >= prefix_frames:
                result[key] = value[:, :prefix_frames].contiguous()
            else:
                result[key] = value
        elif isinstance(value, dict):
            result[key] = dict(value)
        else:
            result[key] = value
    return result


def _streaming_stage1_window(
    conditioning: SanaWMStreamingStage1Conditioning,
    config: SanaWMStreamingTransformerConfig,
) -> _StreamingStage1Window:
    """Return the sink-plus-tail frame window for one streaming Stage-1 call."""
    start = int(conditioning.start_frame)
    end = int(conditioning.end_frame)
    active_frames = end - start
    if start < 0 or active_frames <= 0:
        raise ValueError(
            "SANA-WM streaming chunk bounds must describe a positive chunk; "
            f"got start={start}, end={end}."
        )
    if start == 0:
        return _StreamingStage1Window(
            frame_indices=tuple(range(end)),
            active_start=0,
            active_end=end,
        )

    sink_frames = 1 if bool(config.sink_token) else 0
    sink_frames = min(sink_frames, start)
    cached_frames = max(0, int(config.num_cached_blocks)) * max(
        1,
        int(config.num_frame_per_block),
    )
    history_start = max(sink_frames, start - cached_frames)
    if history_start <= sink_frames:
        frame_indices = tuple(range(end))
        active_start = start
    else:
        frame_indices = tuple(range(sink_frames)) + tuple(range(history_start, end))
        active_start = len(frame_indices) - active_frames
    return _StreamingStage1Window(
        frame_indices=frame_indices,
        active_start=active_start,
        active_end=active_start + active_frames,
    )


def _streaming_window_model_kwargs(
    model_kwargs: dict[str, object],
    *,
    frame_indices: tuple[int, ...],
    total_frames: int,
    chunk_size: int,
) -> dict[str, object]:
    """Slice full-rollout model kwargs to a possibly non-contiguous frame window."""
    result: dict[str, object] = {}
    for key, value in model_kwargs.items():
        if key in {"camera_cache", "rotary_emb", "chunk_plucker_emb"}:
            continue
        if key == "data_info" and isinstance(value, dict):
            data_info = cast(dict[object, object], value)
            result[key] = _streaming_window_data_info(data_info, frame_indices)
        elif key == "chunk_index":
            result[key] = _chunk_index_from_chunk_size(
                len(frame_indices),
                chunk_size,
                strategy="first_chunk_plus_one",
            )
        elif isinstance(value, Tensor):
            result[key] = _slice_streaming_window_tensor(
                value,
                frame_indices=frame_indices,
                total_frames=total_frames,
            )
        elif isinstance(value, dict):
            result[key] = dict(cast(dict[object, object], value))
        else:
            result[key] = value
    return result


def _slice_streaming_window_tensor(
    value: Tensor,
    *,
    frame_indices: tuple[int, ...],
    total_frames: int,
) -> Tensor:
    """Index latent-frame-aligned tensors along their temporal axis."""
    index = torch.tensor(frame_indices, dtype=torch.long, device=value.device)
    if value.ndim == 5 and value.shape[2] >= total_frames:
        return value.index_select(2, index).contiguous()
    if value.ndim >= 3 and value.shape[1] >= total_frames:
        return value.index_select(1, index).contiguous()
    return value


def _streaming_window_data_info(
    data_info: dict[object, object],
    frame_indices: tuple[int, ...],
) -> dict[object, object]:
    """Remap absolute conditioned-frame metadata into the local frame window."""
    result: dict[object, object] = dict(data_info)
    frame_to_local = {frame: local for local, frame in enumerate(frame_indices)}
    condition_frame_info = data_info.get("condition_frame_info")
    if isinstance(condition_frame_info, dict):
        remapped: dict[object, object] = {}
        for frame, value in condition_frame_info.items():
            frame_index = _condition_frame_index(frame)
            if frame_index is None:
                continue
            local = frame_to_local.get(frame_index)
            if local is not None:
                remapped[local] = value
        result["condition_frame_info"] = remapped
    return result


def _condition_frame_index(value: object) -> int | None:
    """Convert conditioned-frame metadata keys to integer frame indices."""
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value) if value.is_integer() else None
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None
    return None


def _streaming_prefix_timestep(
    *,
    noisy_latent: Tensor,
    timestep: Tensor,
    conditioning: SanaWMStreamingStage1Conditioning,
    prefix_frames: int,
) -> Tensor:
    """Build a prefix timestep table with clean context and noisy current chunk."""
    current = _streaming_current_timestep(
        noisy_latent=noisy_latent,
        timestep=timestep,
        conditioning=conditioning,
    )
    batch = noisy_latent.shape[0]
    chunk_frames = current.shape[2]
    active_start = int(conditioning.start_frame)
    active_end = active_start + chunk_frames
    if active_end > prefix_frames:
        raise ValueError(
            "SANA-WM streaming prefix is shorter than the active chunk end: "
            f"prefix_frames={prefix_frames}, active_end={active_end}."
        )
    prefix = torch.zeros(
        batch,
        1,
        prefix_frames,
        dtype=torch.float32,
        device=noisy_latent.device,
    )
    prefix[:, :, active_start:active_end] = current
    return prefix


def _streaming_window_timestep(
    *,
    noisy_latent: Tensor,
    timestep: Tensor,
    conditioning: SanaWMStreamingStage1Conditioning,
    frame_indices: tuple[int, ...],
    active_start: int,
    active_end: int,
) -> Tensor:
    """Build a local timestep table for a bounded streaming frame window."""
    current = _streaming_current_timestep(
        noisy_latent=noisy_latent,
        timestep=timestep,
        conditioning=conditioning,
    )
    batch = noisy_latent.shape[0]
    if active_end - active_start != current.shape[2]:
        raise ValueError(
            "SANA-WM streaming window active span does not match the latent chunk: "
            f"active=({active_start}, {active_end}), chunk={current.shape[2]}."
        )
    window = torch.zeros(
        batch,
        1,
        len(frame_indices),
        dtype=torch.float32,
        device=noisy_latent.device,
    )
    window[:, :, active_start:active_end] = current
    condition_frame_info = _streaming_condition_frame_info(conditioning)
    frame_to_local = {frame: local for local, frame in enumerate(frame_indices)}
    for frame_idx in condition_frame_info:
        frame_index = _condition_frame_index(frame_idx)
        if frame_index is None:
            continue
        local = frame_to_local.get(frame_index)
        if local is not None:
            window[:, :, local] = 0
    return window


def _streaming_current_timestep(
    *,
    noisy_latent: Tensor,
    timestep: Tensor,
    conditioning: SanaWMStreamingStage1Conditioning,
) -> Tensor:
    """Normalize scheduler timestep input to ``[B, 1, active_T]``."""
    batch = noisy_latent.shape[0]
    chunk_frames = int(conditioning.end_frame - conditioning.start_frame)
    if timestep.ndim == 0:
        current = timestep.reshape(1, 1, 1).expand(batch, 1, chunk_frames)
    elif timestep.ndim == 1:
        if timestep.numel() == 1:
            current = timestep.reshape(1, 1, 1).expand(batch, 1, chunk_frames)
        else:
            current = timestep.reshape(batch, 1, 1).expand(batch, 1, chunk_frames)
    elif timestep.ndim == 3:
        current = timestep
    elif timestep.ndim == noisy_latent.ndim:
        current = timestep[:, :1, :, 0, 0]
    else:
        raise ValueError(
            "SANA-WM streaming timestep must be scalar, [B], [B, 1, T], "
            f"or latent-shaped; got shape={tuple(timestep.shape)}."
        )
    if current.shape != (batch, 1, chunk_frames):
        raise ValueError(
            "SANA-WM streaming timestep shape is incompatible with the chunk: "
            f"got {tuple(current.shape)}, expected {(batch, 1, chunk_frames)}."
        )
    current = current.to(device=noisy_latent.device, dtype=torch.float32).clone()
    condition_frame_info = _streaming_condition_frame_info(conditioning)
    for frame_idx in condition_frame_info:
        index = _condition_frame_index(frame_idx)
        if index is None:
            continue
        if conditioning.start_frame <= index < conditioning.end_frame:
            current[:, :, index - conditioning.start_frame] = 0
    return current


def _streaming_condition_frame_info(
    conditioning: SanaWMStreamingStage1Conditioning,
) -> dict[object, object]:
    """Return conditioned-frame metadata from a streaming conditioning payload."""
    data_info = conditioning.model_kwargs.get("data_info", {})
    if not isinstance(data_info, dict):
        return {}
    condition_frame_info = cast(dict[object, object], data_info).get(
        "condition_frame_info",
        {},
    )
    return (
        cast(dict[object, object], condition_frame_info)
        if isinstance(condition_frame_info, dict)
        else {}
    )


def _cfg_guidance(noise_pred: Tensor, cfg_scale: float) -> Tensor:
    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2, dim=0)
    return noise_pred_uncond + cfg_scale * (noise_pred_text - noise_pred_uncond)


def _conditioned_frame_timestep(
    *,
    noisy_latent: Tensor,
    timestep: Tensor,
    conditioning: SanaWMStage1Conditioning,
    num_train_timesteps: int,
) -> Tensor:
    """Expand scheduler timesteps and pin conditioned frames to timestep zero."""
    batch, _channels, frames, _height, _width = noisy_latent.shape
    if timestep.ndim == 0:
        frame_timestep = timestep.reshape(1, 1, 1).expand(batch, 1, frames)
    elif timestep.ndim == 1:
        frame_timestep = timestep.reshape(batch, 1, 1).expand(batch, 1, frames)
    elif timestep.ndim == 3:
        frame_timestep = timestep
    elif timestep.ndim == noisy_latent.ndim:
        frame_timestep = timestep[:, :1, :, 0, 0]
    else:
        raise ValueError(
            "SANA-WM timestep must be scalar, [B], [B, 1, T], or latent-shaped; "
            f"got shape={tuple(timestep.shape)}."
        )
    if frame_timestep.shape != (batch, 1, frames):
        raise ValueError(
            "SANA-WM timestep shape is incompatible with the latent: "
            f"got {tuple(frame_timestep.shape)}, expected {(batch, 1, frames)}."
        )

    frame_timestep = frame_timestep.to(device=noisy_latent.device, dtype=torch.float32)
    condition_frame_mask = _condition_frame_mask(
        conditioning,
        batch=batch,
        frames=frames,
        device=noisy_latent.device,
    )
    max_timestep = (1.0 - condition_frame_mask) * float(num_train_timesteps)
    return torch.minimum(frame_timestep, max_timestep)


def _zero_conditioned_frame_flow(
    flow: Tensor,
    conditioning: SanaWMStage1Conditioning,
) -> Tensor:
    """Prevent scalar scheduler updates from moving pinned condition frames."""
    batch, _channels, frames, _height, _width = flow.shape
    condition_frame_mask = _condition_frame_mask(
        conditioning,
        batch=batch,
        frames=frames,
        device=flow.device,
    ).to(dtype=torch.bool)
    return torch.where(condition_frame_mask[:, :, :, None, None], 0, flow)


def _condition_frame_mask(
    conditioning: SanaWMStage1Conditioning,
    *,
    batch: int,
    frames: int,
    device: torch.device,
) -> Tensor:
    data_info = conditioning.model_kwargs.get("data_info", {})
    mask = torch.zeros(batch, 1, frames, dtype=torch.float32, device=device)
    if not isinstance(data_info, dict):
        return mask
    condition_frame_info = cast(dict[object, object], data_info).get(
        "condition_frame_info",
        {},
    )
    if not isinstance(condition_frame_info, dict):
        return mask
    for frame_idx in condition_frame_info:
        index = _condition_frame_index(frame_idx)
        if index is None:
            continue
        if 0 <= index < frames:
            mask[:, :, index] = 1.0
    return mask


def _avoid_degenerate_tile_tail(
    *,
    sample_extent: int,
    sample_tile_min: int,
    sample_stride: int,
    compression_ratio: int,
) -> int:
    """Choose a tile stride that avoids final latent tiles of size one."""
    if compression_ratio <= 1:
        return sample_stride
    latent_extent = max(1, sample_extent // compression_ratio)
    latent_tile_min = max(1, sample_tile_min // compression_ratio)
    requested_latent_stride = max(1, sample_stride // compression_ratio)

    def tail_size(latent_stride: int) -> int:
        last_start = ((latent_extent - 1) // latent_stride) * latent_stride
        return latent_extent - last_start

    candidates = range(
        min(requested_latent_stride, latent_tile_min),
        1,
        -1,
    )
    for latent_stride in candidates:
        if tail_size(latent_stride) > 1:
            return latent_stride * compression_ratio

    for latent_stride in range(
        requested_latent_stride + 1,
        latent_tile_min + 1,
    ):
        if tail_size(latent_stride) > 1:
            return latent_stride * compression_ratio

    return sample_stride


def _load_inference_config(config_path: str) -> Any:
    if config_path == SANA_WM_STREAMING_CONFIG_PATH:
        return _to_namespace(_builtin_streaming_config())
    with open(resolve_hf_path(config_path), encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise TypeError(f"SANA-WM config must be a mapping, got {type(raw).__name__}.")
    raw.setdefault("work_dir", "")
    return _to_namespace(raw)


def _builtin_streaming_config() -> dict[str, object]:
    """Return the FlashDreams-owned SANA-WM streaming config data."""
    return {
        "work_dir": "",
        "model": {
            "model": "SanaMSVideoCamCtrlStreaming_1600M_P1_D20",
            "mixed_precision": "bf16",
            "chunk_size": SANA_WM_STREAMING_LATENT_CHUNK_SIZE,
            "chunk_split_strategy": "first_chunk_plus_one",
            "softmax_every_n": 4,
        },
        "vae": {
            "vae_type": "LTX2VAE_diffusers_causal",
            "vae_pretrained": SANA_WM_STREAMING_CAUSAL_VAE_ROOT,
            "weight_dtype": "bfloat16",
            "vae_latent_dim": 128,
            "vae_stride": [8, 32, 32],
            "use_framewise_encoding": True,
            "use_framewise_decoding": True,
        },
        "text_encoder": {
            "text_encoder_name": "gemma-2-2b-it",
            "model_max_length": 300,
            "chi_prompt": [
                (
                    "Given a user prompt, generate an enhanced visual "
                    "description suitable for image generation. User Prompt: "
                )
            ],
        },
        "scheduler": {
            "flow_shift": 9.95,
            "inference_flow_shift": 8.0,
        },
    }


def _get_vae(*args: Any, **kwargs: Any) -> nn.Module:
    name, model_path = args[:2]
    device = kwargs["device"]
    dtype = kwargs["dtype"]
    if "LTX2VAE_diffusers" not in str(name):
        raise ValueError(f"Unsupported SANA-WM VAE type: {name!r}")
    from diffusers import AutoencoderKLLTX2Video

    maybe_download_hf_repo_on_rank0(str(model_path))

    try:
        vae = AutoencoderKLLTX2Video.from_pretrained(
            model_path,
            subfolder="vae",
            torch_dtype=dtype,
            local_files_only=True,
        )
    except OSError:
        vae = AutoencoderKLLTX2Video.from_pretrained(
            model_path,
            torch_dtype=dtype,
            local_files_only=True,
        )
    return vae.to(device).eval()


def _get_tokenizer_and_text_encoder(*args: Any, **kwargs: Any) -> tuple[Any, nn.Module]:
    name = kwargs.get("name", args[0] if args else "T5")
    device = kwargs.get("device", "cuda")
    model_id = _TEXT_ENCODER_MODEL_IDS.get(str(name))
    if model_id is None:
        raise ValueError(f"Unsupported SANA-WM text encoder: {name!r}")
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        T5EncoderModel,
        T5Tokenizer,
    )

    maybe_download_hf_repo_on_rank0(model_id)

    if "T5" in str(name):
        tokenizer = T5Tokenizer.from_pretrained(model_id, local_files_only=True)
        text_encoder = T5EncoderModel.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            local_files_only=True,
        ).to(device)
        return tokenizer, text_encoder.eval()

    tokenizer = cast(
        Any, AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    )
    tokenizer.padding_side = "right"
    text_encoder = (
        AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            local_files_only=True,
        )
        .get_decoder()
        .to(device)
        .eval()
    )
    return tokenizer, text_encoder


def _get_weight_dtype(value: str) -> torch.dtype:
    normalized = str(value).lower()
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp16", "float16", "half"}:
        return torch.float16
    if normalized in {"fp32", "float32", "float"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype value: {value!r}")


class _ConfigNamespace(SimpleNamespace):
    """Simple YAML-backed config object with dict-like ``get`` support."""

    def get(self, name: str, default: Any = None) -> Any:
        return getattr(self, name, default)


_TEXT_ENCODER_MODEL_IDS = {
    "T5": "DeepFloyd/t5-v1_1-xxl",
    "T5-small": "google/t5-v1_1-small",
    "T5-base": "google/t5-v1_1-base",
    "T5-large": "google/t5-v1_1-large",
    "T5-xl": "google/t5-v1_1-xl",
    "T5-xxl": "google/t5-v1_1-xxl",
    "gemma-2b": "google/gemma-2b",
    "gemma-2b-it": "google/gemma-2b-it",
    "gemma-2-2b": "google/gemma-2-2b",
    "gemma-2-2b-it": "Efficient-Large-Model/gemma-2-2b-it",
    "gemma-2-9b": "google/gemma-2-9b",
    "gemma-2-9b-it": "google/gemma-2-9b-it",
}


def _vae_encode_ltx2(
    name: str, vae: Any, images: Tensor, *, device: torch.device
) -> Tensor:
    if "LTX2VAE_diffusers" not in name:
        raise ValueError(f"Unsupported SANA-WM VAE encode type: {name!r}")
    dtype = images.dtype
    vae_device = next(vae.parameters()).device
    vae_dtype = next(vae.parameters()).dtype
    posterior = vae.encode(images.to(device=vae_device, dtype=vae_dtype)).latent_dist
    z = posterior.mode()
    latents_mean = vae.latents_mean.view(1, -1, 1, 1, 1).to(z.device, z.dtype)
    latents_std = vae.latents_std.view(1, -1, 1, 1, 1).to(z.device, z.dtype)
    z = (z - latents_mean) * vae.config.scaling_factor / latents_std
    return z.to(device=device, dtype=dtype)


def _vae_decode_ltx2(name: str, vae: Any, latents: Tensor) -> Tensor:
    if "LTX2VAE_diffusers" not in name:
        raise ValueError(f"Unsupported SANA-WM VAE decode type: {name!r}")
    vae_device = next(vae.parameters()).device
    vae_dtype = next(vae.parameters()).dtype
    latents_mean = vae.latents_mean.view(1, -1, 1, 1, 1).to(
        latents.device,
        latents.dtype,
    )
    latents_std = vae.latents_std.view(1, -1, 1, 1, 1).to(
        latents.device,
        latents.dtype,
    )
    scaled = latents * latents_std / vae.config.scaling_factor + latents_mean
    return vae.decode(
        scaled.to(device=vae_device, dtype=vae_dtype),
        temb=None,
        return_dict=False,
    )[0]


def _to_namespace(value: Any) -> Any:
    if isinstance(value, dict):
        return _ConfigNamespace(
            **{str(key): _to_namespace(child) for key, child in value.items()}
        )
    if isinstance(value, list):
        return [_to_namespace(child) for child in value]
    return value


def _chunk_index_from_config(config: Any, *, num_frames: int) -> list[int] | None:
    model = getattr(config, "model", None)
    if model is None:
        return None
    chunk_index = model.get("chunk_index", None)
    chunk_size = model.get("chunk_size", None)
    strategy = model.get("chunk_split_strategy", "uniform")
    if chunk_index is not None:
        if not isinstance(chunk_index, (list, tuple)):
            raise TypeError(
                f"chunk_index must be a list, got {type(chunk_index).__name__}"
            )
        if len(chunk_index) == 0:
            raise ValueError("chunk_index cannot be empty.")
        return [int(index) for index in chunk_index]
    if chunk_size is None:
        return None
    return _chunk_index_from_chunk_size(
        num_frames,
        int(chunk_size),
        strategy=str(strategy),
    )


def _chunk_index_from_chunk_size(
    num_frames: int,
    chunk_size: int,
    *,
    strategy: str = "uniform",
) -> list[int]:
    if num_frames <= 0:
        raise ValueError(f"num_frames must be > 0, got {num_frames}.")
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be > 0, got {chunk_size}.")
    normalized = strategy.lower()
    if normalized in {"uniform", "default"}:
        indices = list(range(0, num_frames, chunk_size))
        if len(indices) > 1 and (num_frames - indices[-1]) < chunk_size:
            indices.pop()
        return indices
    if normalized in {"first_frame", "first_frame_alone", "first_frame_only"}:
        if num_frames <= 1:
            return [0]
        indices = [0] + list(range(1, num_frames, chunk_size))
        if len(indices) > 2 and (num_frames - indices[-1]) < chunk_size:
            indices.pop()
        return indices
    if normalized in {"first_plus_one", "first_chunk_plus_one"}:
        if num_frames <= chunk_size + 1:
            return [0]
        indices = [0] + list(range(chunk_size + 1, num_frames, chunk_size))
        if len(indices) > 1 and (num_frames - indices[-1]) < chunk_size:
            indices.pop()
        return indices
    raise ValueError(f"Unknown chunk_split_strategy {strategy!r}.")


__all__ = [
    "SanaWMStage1Conditioning",
    "SanaWMStreamingStage1Conditioning",
    "SanaWMStreamingTransformer",
    "SanaWMStreamingTransformerCache",
    "SanaWMStreamingTransformerConfig",
    "SanaWMTransformer",
    "SanaWMTransformerCache",
    "SanaWMTransformerConfig",
]
