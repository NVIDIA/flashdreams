# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SwiftVR pipeline configuration builder."""

from __future__ import annotations

from pathlib import Path

import torch
from huggingface_hub import snapshot_download

from flashdreams.core.io.hf import maybe_download_hf_repo_on_rank0
from flashdreams.infra.diffusion.model import DiffusionModelConfig
from flashdreams.infra.diffusion.scheduler.fm import FlowMatchSchedulerConfig
from swiftvr.impl.decoder import SwiftVRDecoderConfig
from swiftvr.impl.encoder import SwiftVREncoderConfig
from swiftvr.impl.pipeline import SwiftVRPipelineConfig
from swiftvr.impl.transformer import SwiftVRTransformerConfig
from swiftvr.impl.transformer.network import SwiftVRDiTNetworkConfig

_TRANSFORMER_CHECKPOINT = "transformer/diffusion_pytorch_model.safetensors"
_TRANSFORMER_CHECKPOINT_INDEX = f"{_TRANSFORMER_CHECKPOINT}.index.json"
_CHECKPOINT_PATTERNS = (
    "reae.safetensors",
    "prompt_embedding.safetensors",
    "transformer/*.safetensors",
    _TRANSFORMER_CHECKPOINT_INDEX,
)


def build_swiftvr_pipeline(
    *,
    checkpoint: str = "H-oliday/SwiftVR",
    revision: str | None = None,
    dtype: torch.dtype = torch.bfloat16,
    attention_window: tuple[int, int] = (16, 16),
    compile_blocks: bool = False,
    chunk_size: int = 8,
    name: str = "swiftvr",
) -> SwiftVRPipelineConfig:
    """Build the nested SwiftVR encoder/transformer/decoder configuration."""
    if chunk_size <= 0 or chunk_size % 4:
        raise ValueError(
            f"SwiftVR chunk_size must be a positive multiple of 4, got {chunk_size}."
        )
    checkpoint_root = _resolve_checkpoint(checkpoint, revision=revision)
    reae_checkpoint = str(checkpoint_root / "reae.safetensors")
    return SwiftVRPipelineConfig(
        name=name,
        prompt_path=str(checkpoint_root / "prompt_embedding.safetensors"),
        encoder=SwiftVREncoderConfig(
            checkpoint_path=reae_checkpoint,
            dtype=dtype,
        ),
        diffusion_model=DiffusionModelConfig(
            transformer=SwiftVRTransformerConfig(
                network=SwiftVRDiTNetworkConfig(
                    attention_window=attention_window,
                ),
                checkpoint_path=str(_transformer_checkpoint(checkpoint_root)),
                dtype=dtype,
                compile_blocks=compile_blocks,
                latent_frames=chunk_size // 4,
            ),
            scheduler=FlowMatchSchedulerConfig(
                num_inference_steps=1,
                denoising_timesteps=[1000],
                shift=8.0,
                sigma_min=0.0,
                extra_one_step=True,
            ),
            context_noise=0,
        ),
        decoder=SwiftVRDecoderConfig(
            checkpoint_path=reae_checkpoint,
            dtype=dtype,
        ),
    )


def _transformer_checkpoint(checkpoint_root: Path) -> Path:
    for relative_path in (_TRANSFORMER_CHECKPOINT, _TRANSFORMER_CHECKPOINT_INDEX):
        path = checkpoint_root / relative_path
        if path.is_file():
            return path
    raise FileNotFoundError(
        "SwiftVR transformer checkpoint not found; expected "
        f"{_TRANSFORMER_CHECKPOINT!r} or {_TRANSFORMER_CHECKPOINT_INDEX!r} "
        f"under {checkpoint_root}."
    )


def _resolve_checkpoint(checkpoint: str, *, revision: str | None) -> Path:
    local = Path(checkpoint).expanduser()
    if local.is_dir():
        return local
    maybe_download_hf_repo_on_rank0(
        checkpoint,
        revision=revision,
        allow_patterns=_CHECKPOINT_PATTERNS,
    )
    return Path(
        snapshot_download(
            checkpoint,
            revision=revision,
            allow_patterns=list(_CHECKPOINT_PATTERNS),
            local_files_only=True,
        )
    )


__all__ = ["build_swiftvr_pipeline"]
