# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Isolated GPU tests for each optimization layer (run in order)."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import pytest
import torch
from flashdreams.infra.config import derive_config
from ltx_video.config import PIPELINE_LTX_T2V_2B
from ltx_video.pipeline import LTXVideoStreamingPipeline

pytestmark = pytest.mark.ci_gpu

PROMPT = "A coastal road at dusk, waves breaking on rocks"
WIDTH, HEIGHT = 768, 512


def _require_cuda() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")


@dataclass
class OptResult:
    name: str
    seconds: float
    shape: tuple[int, ...]
    optimizations: dict[str, bool]
    kv_seq_len: int = 0


def _run_one_chunk(cfg_overrides: dict[str, Any]) -> OptResult:
    _require_cuda()
    torch.cuda.empty_cache()
    overrides = dict(cfg_overrides)
    label = overrides.pop("_label", "run")
    manual_denoise = overrides.pop("manual_denoise", False)
    base = derive_config(
        PIPELINE_LTX_T2V_2B,
        device="cuda:0",
        num_inference_steps=8,
        manual_denoise=manual_denoise,
        **overrides,
    )
    pipe = LTXVideoStreamingPipeline(base)
    cache = pipe.initialize_cache(text=[PROMPT])
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = pipe.generate(0, cache, width=WIDTH, height=HEIGHT)
    pipe.finalize(0, cache)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    kv_len = cache.kv.seq_len if pipe.use_kv_cache else 0
    return OptResult(
        name=label,
        seconds=elapsed,
        shape=tuple(out.shape),
        optimizations=pipe.active_optimizations,
        kv_seq_len=kv_len,
    )


def test_baseline_streaming_pipe() -> None:
    """Streaming path: native pipe() per chunk, no manual denoise loop."""
    r = _run_one_chunk({"_label": "streaming_pipe", "manual_denoise": False})
    assert len(r.shape) == 4 and r.shape[0] > 0
    assert r.optimizations["manual_denoise"] is False


def test_manual_denoise_only() -> None:
    r = _run_one_chunk({"_label": "manual_denoise", "manual_denoise": True})
    assert r.shape[0] > 0
    assert r.optimizations["manual_denoise"] is True
    assert r.optimizations["kv_cache"] is False
    assert r.optimizations["compile"] is False


def test_cuda_graphs_isolated() -> None:
    r = _run_one_chunk(
        {
            "_label": "cuda_graphs",
            "manual_denoise": True,
            "cuda_graphs": True,
            "compile": False,
            "kv_cache": False,
        }
    )
    assert r.shape[0] > 0
    # Graphs may self-disable if capture fails; test must still produce frames.
    assert r.optimizations["manual_denoise"] is True


def test_compile_isolated() -> None:
    r = _run_one_chunk(
        {
            "_label": "compile",
            "manual_denoise": True,
            "compile": True,
            "cuda_graphs": False,
            "kv_cache": False,
        }
    )
    assert r.shape[0] > 0
    assert r.optimizations["compile"] is True


def test_kv_cache_isolated() -> None:
    r = _run_one_chunk(
        {
            "_label": "kv_cache",
            "manual_denoise": True,
            "kv_cache": True,
            "compile": False,
            "cuda_graphs": False,
        }
    )
    assert r.shape[0] > 0
    assert r.optimizations["kv_cache"] is True


def test_kv_cache_accumulates_across_ar_steps() -> None:
    _require_cuda()
    cfg = derive_config(
        PIPELINE_LTX_T2V_2B,
        device="cuda:0",
        num_inference_steps=6,
        manual_denoise=True,
        kv_cache=True,
        compile=False,
        cuda_graphs=False,
    )
    pipe = LTXVideoStreamingPipeline(cfg)
    cache = pipe.initialize_cache(text=[PROMPT])
    pipe.generate(0, cache, width=WIDTH, height=HEIGHT)
    pipe.finalize(0, cache)
    seq_after_0 = cache.kv.seq_len
    pipe.generate(1, cache, width=WIDTH, height=HEIGHT)
    pipe.finalize(1, cache)
    seq_after_1 = cache.kv.seq_len
    assert seq_after_0 > 0
    assert seq_after_1 > seq_after_0


def test_full_optimization_stack() -> None:
    from ltx_video.config import PIPELINE_LTX_T2V_2B_OPTIMIZED

    _require_cuda()
    cfg = derive_config(PIPELINE_LTX_T2V_2B_OPTIMIZED, device="cuda:0", num_inference_steps=8)
    pipe = LTXVideoStreamingPipeline(cfg)
    cache = pipe.initialize_cache(text=[PROMPT])
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = pipe.generate(0, cache, width=WIDTH, height=HEIGHT)
    pipe.finalize(0, cache)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    opts = pipe.active_optimizations
    assert out.shape[0] > 0
    assert opts["manual_denoise"] is True
    assert opts["kv_cache"] is True
    assert opts["compile"] is True
    print(f"full_stack: {elapsed:.2f}s opts={opts}")
