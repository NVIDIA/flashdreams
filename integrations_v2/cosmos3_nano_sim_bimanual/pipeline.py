# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""V2 pipeline orchestration for Cosmos Interactive hand generation."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, cast

import torch

from flashdreams.infra.diffusion.model import DiffusionModel, DiffusionModelConfig
from flashdreams.infra.diffusion.scheduler import FlowMatchSchedulerConfig
from flashdreams.infra.pipeline import (
    StreamInferencePipeline,
    StreamInferencePipelineCache,
    StreamInferencePipelineConfig,
)

from . import action_format as _agibot
from .decoder import DECODER_BACKENDS, HandDecoder, HandDecoderConfig
from .encoder import HandActionEncoder, HandActionEncoderConfig, HandSceneEncoder
from .transformer import (
    HandTransformer,
    HandTransformerCache,
    HandTransformerConfig,
)
from .transformer.config import validate_transformer_config


@dataclass(kw_only=True)
class HandStreamInferencePipelineConfig(StreamInferencePipelineConfig):
    """Config for the native Cosmos Interactive hand pipeline."""

    _target: type["HandStreamInferencePipeline"] = field(
        default_factory=lambda: HandStreamInferencePipeline
    )
    compile_prewarm_steps: int = 0

    def __post_init__(self) -> None:
        validate_pipeline_config(self, check_runtime=False)


def validate_pipeline_config(
    config: HandStreamInferencePipelineConfig,
    *,
    check_runtime: bool,
) -> HandTransformerConfig:
    """Validate component compatibility before constructing heavyweight models."""
    if config.compile_prewarm_steps < 0:
        raise ValueError("compile_prewarm_steps must be non-negative")
    if not isinstance(config.encoder, HandActionEncoderConfig):
        raise TypeError("hand pipeline requires HandActionEncoderConfig")
    if not isinstance(config.decoder, HandDecoderConfig):
        raise TypeError("hand pipeline requires HandDecoderConfig")
    if config.decoder.backend not in DECODER_BACKENDS:
        raise ValueError(f"unknown decoder {config.decoder.backend!r}")
    if not isinstance(config.diffusion_model, DiffusionModelConfig):
        raise TypeError("hand pipeline requires DiffusionModelConfig")
    transformer = config.diffusion_model.transformer
    if not isinstance(transformer, HandTransformerConfig):
        raise TypeError("hand diffusion model requires HandTransformerConfig")
    scheduler = config.diffusion_model.scheduler
    if not isinstance(scheduler, FlowMatchSchedulerConfig):
        raise TypeError("hand diffusion model requires FlowMatchSchedulerConfig")
    if scheduler.num_inference_steps != len(scheduler.denoising_timesteps):
        raise ValueError(
            "num_inference_steps must equal the number of denoising_timesteps"
        )
    if config.compile_prewarm_steps and not transformer.compile_denoise:
        raise ValueError("compile prewarm requires compile denoise")
    validate_transformer_config(transformer, check_runtime=check_runtime)
    return transformer


def _compile_prewarm_plan(
    config: HandStreamInferencePipelineConfig,
) -> tuple[int, int, int]:
    """Return fill chunks, repeated fill chunks, and total hidden warmup chunks."""
    transformer = cast(HandTransformerConfig, config.diffusion_model.transformer)
    if not transformer.compile_denoise:
        return 0, 0, 0
    scheduler = cast(FlowMatchSchedulerConfig, config.diffusion_model.scheduler)
    denoise_steps = scheduler.num_inference_steps
    fill_steps = (
        0
        if transformer.valid_length_cache_attention
        else -(-transformer.window_frames // transformer.chunk_size)
    )
    # Warmup counts denoise calls; the following call captures the CUDA graph.
    capture_steps = (
        transformer.cuda_graph_warmup_iters // denoise_steps + 1
        if transformer.cuda_graph_denoise
        else 1
    )
    repeated_fill_steps = (
        fill_steps
        * (transformer.cuda_graph_warmup_iters // (fill_steps * denoise_steps))
        if fill_steps
        and transformer.fixed_shape_cache_fill
        and transformer.cuda_graph_denoise
        else 0
    )
    return (
        fill_steps,
        repeated_fill_steps,
        repeated_fill_steps + fill_steps + capture_steps,
    )


class HandStreamInferencePipeline(StreamInferencePipeline):
    """Own every model component used by a hand-demo rollout."""

    encoder: HandActionEncoder
    decoder: HandDecoder
    diffusion_model: DiffusionModel[HandTransformerCache]

    def __init__(self, config: HandStreamInferencePipelineConfig) -> None:
        transformer_config = validate_pipeline_config(config, check_runtime=True)
        super().__init__(config)
        self.config: HandStreamInferencePipelineConfig = config
        if not isinstance(self.encoder, HandActionEncoder):
            raise TypeError("hand pipeline requires HandActionEncoder")
        if not isinstance(self.decoder, HandDecoder):
            raise TypeError("hand pipeline requires HandDecoder")
        if not isinstance(self.diffusion_model.transformer, HandTransformer):
            raise TypeError("hand pipeline requires HandTransformer")
        self.scene_encoder = HandSceneEncoder(device=transformer_config.device)
        self.diffusion_model.scheduler.to(self.device)

    @property
    def transformer(self) -> HandTransformer:
        return cast(HandTransformer, self.diffusion_model.transformer)

    @property
    def actions_per_step(self) -> int:
        return self.transformer.config.chunk_size * _agibot.TEMPORAL_COMPRESSION_FACTOR

    @property
    def temporal_compression_ratio(self) -> int:
        return _agibot.TEMPORAL_COMPRESSION_FACTOR

    def encode_scene(self, image: torch.Tensor) -> torch.Tensor:
        return self.scene_encoder.encode(image)

    def patchify_scene_latent(
        self,
        latent: torch.Tensor,
        *,
        content_height: int,
        content_width: int,
    ) -> torch.Tensor:
        return self.scene_encoder.patchify_encoded(
            latent,
            content_height=content_height,
            content_width=content_width,
        )

    @torch.no_grad()
    def prepare_scene(self, scene: Any) -> None:
        """Warm generation and decoding, retaining only the reset transformer cache."""
        steps = self.config.compile_prewarm_steps
        transformer = self.transformer
        if steps == 0 or transformer.has_prepared_cache(scene):
            return
        print(
            f"[prewarm][hand:{scene.name}] running {steps} hidden chunk(s)...",
            flush=True,
        )
        started = time.perf_counter()
        context = {"scene": scene}
        pipeline_cache = self.initialize_cache(
            transformer_context=context, decoder_context=context
        )
        cache = cast(HandTransformerCache, pipeline_cache.transformer_cache)
        profile = cache.profile
        cache.profile = True
        prepared = False
        try:
            fill_steps, repeated_fill_steps, _ = _compile_prewarm_plan(self.config)
            action = torch.zeros(
                self.actions_per_step,
                _agibot.MAX_ACTION_DIM,
                device=self.device,
            )
            for index in range(steps):
                rollout_index = (
                    index % fill_steps
                    if index < repeated_fill_steps
                    else index - repeated_fill_steps
                )
                self.generate(rollout_index, pipeline_cache, input=action)
                self.finalize(rollout_index, pipeline_cache)
                timing = cache.last_timing
                print(
                    f"[prewarm][hand:{scene.name}] step {index + 1}/{steps}: "
                    f"model={timing.get('model_step_s', 0.0) * 1e3:.0f}ms "
                    f"denoise={timing.get('denoise_s', 0.0) * 1e3:.0f}ms "
                    f"cache={timing.get('cache_seed_prune_s', 0.0) * 1e3:.0f}ms "
                    f"graph={timing.get('cuda_graph_captured', 0.0):.0f}",
                    flush=True,
                )
                if (
                    index < repeated_fill_steps
                    and (index + 1) % fill_steps == 0
                    and index + 1 < steps
                ):
                    # Short windows may need another fill pass before capture.
                    cache.profile = profile
                    transformer.release_active_cache(cache)
                    pipeline_cache = self.initialize_cache(
                        transformer_context=context, decoder_context=context
                    )
                    cache = cast(HandTransformerCache, pipeline_cache.transformer_cache)
                    cache.profile = True
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            prepared = True
        finally:
            cache.profile = profile
            try:
                transformer.release_active_cache(cache)
            finally:
                if not prepared:
                    transformer._prepared_caches.pop(str(scene.name), None)
        print(
            f"[prewarm][hand:{scene.name}] ready in "
            f"{time.perf_counter() - started:.1f}s; rollout reset to frame 0",
            flush=True,
        )

    @torch.no_grad()
    def generate(
        self,
        autoregressive_index: int,
        cache: StreamInferencePipelineCache,
        input: Any = None,
        *,
        on_diffusion_complete: Callable[[], None] | None = None,
    ) -> torch.Tensor:
        return self._generate(
            autoregressive_index,
            cache,
            input=input,
            on_diffusion_complete=on_diffusion_complete,
        )

    @torch.no_grad()
    def finalize(
        self,
        autoregressive_index: int,
        cache: StreamInferencePipelineCache,
    ) -> dict[str, float] | None:
        transformer_cache = cast(HandTransformerCache, cache.transformer_cache)
        rollout_timings = dict(transformer_cache.last_timing)
        timings = super().finalize(autoregressive_index, cache)
        if timings is None:
            return rollout_timings or None
        return {**rollout_timings, **timings}

    def release_session(
        self,
        cache: StreamInferencePipelineCache | None = None,
    ) -> None:
        expected = (
            None
            if cache is None
            else cast(HandTransformerCache, cache.transformer_cache)
        )
        self.transformer.release_active_cache(expected)

    def close(self) -> None:
        self.release_session()

    def to(self, *args: Any, **kwargs: Any) -> "HandStreamInferencePipeline":
        requested = kwargs.get("device")
        if requested is None and args and isinstance(args[0], (str, torch.device, int)):
            requested = args[0]
        if requested is not None and torch.device(requested) != self.device:
            raise ValueError(
                "hand inference allocates decoder state while loading; configure "
                f"component devices instead of moving the pipeline to {requested!s}"
            )
        return super().to(*args, **kwargs)


__all__ = [
    "HandActionEncoder",
    "HandActionEncoderConfig",
    "HandDecoder",
    "HandDecoderConfig",
    "HandStreamInferencePipeline",
    "HandStreamInferencePipelineConfig",
    "HandTransformer",
    "HandTransformerCache",
    "HandTransformerConfig",
    "validate_pipeline_config",
]
