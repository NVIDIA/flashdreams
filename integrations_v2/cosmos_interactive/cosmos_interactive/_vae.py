# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Wan 2.2 encoder configuration and TAEHV-super decoder checkpoint."""

from __future__ import annotations

from typing import Any

import torch

from flashdreams.recipes.wan.autoencoder.vae import Wan22TI2V5BVAEEncoderConfig


def wan22_vae_encoder_config(
    *,
    dtype: torch.dtype = torch.bfloat16,
    use_cuda_graph: bool = False,
    use_compile: bool = False,
    checkpoint_path: str | None = None,
) -> Wan22TI2V5BVAEEncoderConfig:
    """Wan 2.2 (48-channel) VAE encoder — the video tokenizer / first-frame I2V encode."""
    kwargs: dict[str, Any] = {
        "dtype": dtype,
        "use_cuda_graph": use_cuda_graph,
        "use_compile": use_compile,
    }
    if checkpoint_path is not None:
        kwargs["checkpoint_path"] = checkpoint_path
    return Wan22TI2V5BVAEEncoderConfig(**kwargs)


TAEW2_2_SUPER_CHECKPOINT = (
    "https://raw.githubusercontent.com/madebyollin/taehv/"
    "62f7591f59dfbb4c3c02b7a621d180a9eeaba26c/"
    "safetensors/taew2_2_super.safetensors"
)
