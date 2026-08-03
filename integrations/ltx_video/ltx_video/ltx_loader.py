# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any, Type

import torch


def get_ltx_pipeline_class() -> Type[Any]:
    try:
        from diffusers import LTXPipeline

        return LTXPipeline
    except ImportError:
        from diffusers.pipelines.ltx.pipeline_ltx import LTXPipeline

        return LTXPipeline


def load_ltx_pipeline(
    checkpoint: str,
    device: str | torch.device,
    *,
    dtype: torch.dtype = torch.bfloat16,
) -> Any:
    pipeline_cls = get_ltx_pipeline_class()
    pipe = pipeline_cls.from_pretrained(
        checkpoint,
        torch_dtype=dtype,
        low_cpu_mem_usage=False,
    )
    return pipe.to(device)
