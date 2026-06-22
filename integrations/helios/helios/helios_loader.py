# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any, Type

import torch


def get_helios_pipeline_class() -> Type[Any]:
    try:
        from diffusers import HeliosPyramidPipeline

        return HeliosPyramidPipeline
    except ImportError as exc:
        raise ImportError(
            "HeliosPyramidPipeline requires diffusers from source or a recent release. "
            "Install with: pip install git+https://github.com/huggingface/diffusers.git"
        ) from exc


def get_helios_vae_class() -> Type[Any]:
    try:
        from diffusers import AutoencoderKLWan

        return AutoencoderKLWan
    except ImportError:
        from diffusers.models import AutoencoderKLWan

        return AutoencoderKLWan


def load_helios_pipeline(
    checkpoint: str,
    device: str | torch.device,
    *,
    dtype: torch.dtype = torch.bfloat16,
    enable_parallelism: bool = False,
    cp_backend: str = "ulysses",
) -> Any:
    """Load Helios pyramid pipeline with float32 VAE and bfloat16 transformer."""
    pipeline_cls = get_helios_pipeline_class()
    vae_cls = get_helios_vae_class()

    vae = vae_cls.from_pretrained(
        checkpoint,
        subfolder="vae",
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    pipe = pipeline_cls.from_pretrained(
        checkpoint,
        vae=vae,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    dev = torch.device(device)
    vae = vae.to(dev)
    pipe.vae = vae
    pipe.text_encoder = pipe.text_encoder.to(dev)
    pipe.transformer = pipe.transformer.to(dev, dtype=dtype)

    if enable_parallelism:
        _try_enable_parallelism(pipe, cp_backend)

    return pipe


def _try_enable_parallelism(pipe: Any, cp_backend: str) -> None:
    """Enable diffusers context parallelism when the runtime supports it."""
    enable_fn = getattr(pipe, "enable_parallelism", None)
    if enable_fn is not None:
        enable_fn(backend=cp_backend)
        print(f"[Helios loader] Context parallelism enabled ({cp_backend})")
        return

    transformer = getattr(pipe, "transformer", None)
    if transformer is not None:
        xf_enable = getattr(transformer, "enable_parallelism", None)
        if xf_enable is not None:
            xf_enable(backend=cp_backend)
            print(
                f"[Helios loader] Transformer context parallelism enabled ({cp_backend})"
            )
            return

    print(
        "[Helios loader] Context parallelism requested but not supported by this "
        "diffusers build; run with torchrun and upgrade diffusers if needed."
    )
