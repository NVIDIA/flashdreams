# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration and validation for the AgiBot transformer runtime."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, get_args

import torch

from flashdreams.infra.compile import CompileMode
from flashdreams.infra.diffusion.transformer import TransformerConfig

from ..attention import varlen_attention_available
from ..fetch_checkpoint import DEFAULT_CHECKPOINT
from ..sdpa_backends import normalize_sdpa_backend

if TYPE_CHECKING:
    from . import HandTransformer

DEFAULT_WINDOW_FRAMES = 64

CacheMode = Literal["cat", "block"]
CacheRefreshMode = Literal["chunk", "framewise"]
CACHE_MODES = get_args(CacheMode)
CACHE_REFRESH_MODES = get_args(CacheRefreshMode)
COMPILE_DENOISE_MODES = get_args(CompileMode)


def _target() -> type[HandTransformer]:
    from . import HandTransformer

    return HandTransformer


@dataclass(kw_only=True)
class HandTransformerConfig(TransformerConfig):
    """Checkpoint, cache, and optimized execution settings for the hand MoT."""

    _target: type[HandTransformer] = field(default_factory=_target)
    checkpoint: str | None = DEFAULT_CHECKPOINT
    device: str = "cuda"
    credential_path: str | None = None
    chunk_size: int = 2
    window_frames: int = DEFAULT_WINDOW_FRAMES
    native_gqa_attention: bool = False
    sdpa_backend: str = "default"
    cache_mode: CacheMode = "cat"
    cache_refresh_mode: CacheRefreshMode = "chunk"
    overlap_cache_update: bool = False
    compile_denoise: bool = False
    compile_denoise_mode: CompileMode = "max-autotune-no-cudagraphs"
    fixed_shape_cache_fill: bool = False
    valid_length_cache_attention: bool = False
    cuda_graph_denoise: bool = False
    cuda_graph_warmup_iters: int = 2
    profile: bool = False
    model_fps: float | None = None


def validate_transformer_config(
    config: HandTransformerConfig,
    *,
    check_runtime: bool,
) -> None:
    """Reject invalid cache and execution combinations before loading weights."""
    if config.chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if config.window_frames < 0:
        raise ValueError("window_frames must be non-negative")
    if config.cuda_graph_warmup_iters < 0:
        raise ValueError("cuda_graph_warmup_iters must be non-negative")
    if config.model_fps is not None and (
        not math.isfinite(config.model_fps) or config.model_fps <= 0.0
    ):
        raise ValueError("model_fps must be finite and positive")
    normalize_sdpa_backend(config.sdpa_backend)

    validate_cache_options(
        cache_mode=config.cache_mode,
        cache_refresh_mode=config.cache_refresh_mode,
        window_frames=config.window_frames,
        overlap_cache_update=config.overlap_cache_update,
        compile_denoise=config.compile_denoise,
        compile_denoise_mode=config.compile_denoise_mode,
        fixed_shape_cache_fill=config.fixed_shape_cache_fill,
        valid_length_cache_attention=config.valid_length_cache_attention,
        cuda_graph_denoise=config.cuda_graph_denoise,
        device=config.device,
        check_runtime=check_runtime,
    )


def validate_cache_options(
    *,
    cache_mode: str,
    cache_refresh_mode: str,
    window_frames: int,
    overlap_cache_update: bool,
    compile_denoise: bool,
    compile_denoise_mode: str,
    fixed_shape_cache_fill: bool,
    valid_length_cache_attention: bool,
    cuda_graph_denoise: bool,
    device: str,
    check_runtime: bool = True,
) -> None:
    """Validate cache invariants shared by config and per-scene cache setup."""
    if cache_mode not in CACHE_MODES:
        raise ValueError(f"cache_mode must be one of {CACHE_MODES}, got {cache_mode!r}")
    if cache_refresh_mode not in CACHE_REFRESH_MODES:
        raise ValueError(
            f"cache_refresh_mode must be one of {CACHE_REFRESH_MODES}, "
            f"got {cache_refresh_mode!r}"
        )
    if compile_denoise_mode not in COMPILE_DENOISE_MODES:
        raise ValueError(
            "compile denoise mode must be one of: " + ", ".join(COMPILE_DENOISE_MODES)
        )

    requires_block = {
        "cache_refresh_mode='framewise'": cache_refresh_mode == "framewise",
        "overlap_cache_update": overlap_cache_update,
        "compile_denoise": compile_denoise,
        "fixed_shape_cache_fill": fixed_shape_cache_fill,
        "valid_length_cache_attention": valid_length_cache_attention,
    }
    for name, requested in requires_block.items():
        if requested and cache_mode != "block":
            raise ValueError(f"{name} requires cache_mode='block'")
    if cache_mode == "block" and window_frames <= 0:
        raise ValueError("cache_mode='block' requires window_frames > 0")
    if fixed_shape_cache_fill and not compile_denoise:
        raise ValueError("fixed_shape_cache_fill requires compile_denoise=True")
    if valid_length_cache_attention and not compile_denoise:
        raise ValueError("valid_length_cache_attention requires compile_denoise=True")
    if valid_length_cache_attention and fixed_shape_cache_fill:
        raise ValueError(
            "valid_length_cache_attention and fixed_shape_cache_fill are mutually exclusive"
        )
    if cuda_graph_denoise and not compile_denoise:
        raise ValueError("cuda_graph_denoise requires compile_denoise=True")

    resolved_device = torch.device(device)
    if check_runtime and valid_length_cache_attention:
        if not varlen_attention_available():
            raise ValueError(
                "valid_length_cache_attention requires PyTorch varlen FlashAttention"
            )
        if resolved_device.type != "cuda" or not torch.cuda.is_available():
            raise ValueError("valid_length_cache_attention requires a CUDA device")
    if (
        check_runtime
        and cuda_graph_denoise
        and (resolved_device.type != "cuda" or not torch.cuda.is_available())
    ):
        raise ValueError("cuda_graph_denoise requires a CUDA device")


__all__ = [
    "CACHE_MODES",
    "CACHE_REFRESH_MODES",
    "COMPILE_DENOISE_MODES",
    "DEFAULT_CHECKPOINT",
    "DEFAULT_WINDOW_FRAMES",
    "HandTransformerConfig",
    "validate_cache_options",
    "validate_transformer_config",
]
