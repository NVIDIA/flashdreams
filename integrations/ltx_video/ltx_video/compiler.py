# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""torch.compile, CUDA graphs, and FlashAttention helpers."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import contextmanager
from typing import Any

import torch
import torch.nn as nn


def enable_flash_attention() -> None:
    """Route SDPA through FlashAttention / memory-efficient kernels."""
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(False)
    print("[LTX compiler] Flash attention (cuDNN) enabled")


def compile_transformer(transformer: nn.Module, *, use_kv_cache: bool = False) -> nn.Module:
    """Compile the LTX DiT for repeated same-shape AR calls."""
    # reduce-overhead enables Inductor CUDA graphs that alias buffers and break KV-cache.
    mode = "default" if use_kv_cache else "reduce-overhead"
    compiled = torch.compile(
        transformer,
        mode=mode,
        fullgraph=False,
        dynamic=False,
    )
    print(f"[LTX compiler] torch.compile applied to DiT transformer (mode={mode})")
    return compiled


class CUDAGraphRunner:
    """Capture one denoising step as a CUDA graph and replay it."""

    def __init__(self, n_warmup: int = 3) -> None:
        self._graph: torch.cuda.CUDAGraph | None = None
        self._input_buffers: dict[str, torch.Tensor] = {}
        self._output_buffer: Any = None
        self._n_warmup = n_warmup

    def _warmup(self, fn: Callable[..., Any], **kwargs: Any) -> Any:
        out = None
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            for _ in range(self._n_warmup):
                out = fn(**kwargs)
        torch.cuda.current_stream().wait_stream(stream)
        return out

    def __call__(self, fn: Callable[..., Any], **kwargs: Any) -> Any:
        if self._graph is None:
            self._warmup(fn, **kwargs)
            self._input_buffers = {
                k: v.clone()
                for k, v in kwargs.items()
                if isinstance(v, torch.Tensor)
            }
            self._graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self._graph):
                self._output_buffer = fn(**self._input_buffers)
            print("[LTX compiler] CUDA graph captured")

        for key, value in kwargs.items():
            if isinstance(value, torch.Tensor) and key in self._input_buffers:
                self._input_buffers[key].copy_(value)

        assert self._graph is not None
        self._graph.replay()
        return self._output_buffer

    def reset(self) -> None:
        self._graph = None
        self._input_buffers = {}
        self._output_buffer = None


@contextmanager
def sdpa_flash_context():
    """Context manager toggling Flash SDPA for a block."""
    prev_flash = torch.backends.cuda.flash_sdp_enabled()
    prev_mem = torch.backends.cuda.mem_efficient_sdp_enabled()
    prev_math = torch.backends.cuda.math_sdp_enabled()
    enable_flash_attention()
    try:
        yield
    finally:
        torch.backends.cuda.enable_flash_sdp(prev_flash)
        torch.backends.cuda.enable_mem_efficient_sdp(prev_mem)
        torch.backends.cuda.enable_math_sdp(prev_math)
