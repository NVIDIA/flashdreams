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

"""Per-stage profiling for the FlashVSR streaming pipeline.

Drives :class:`FlashVSRPipeline` through one cold chunk + ``--n_steady``
steady chunks, instruments the projector / DiT / decoder / color-corrector
forwards with ``cuda.Event`` pairs, and reports the median per-stage share
of the steady-chunk wall time. The cold chunk fills caches (and primes
``CUDAGraphWrapper.drain``); the first ``--n_warmup_skip`` steady chunks
are discarded before the median so the reported numbers are pure replay.

Marked ``manual`` + ``slow`` so it stays opt-in. Run with::

    uv run pytest integrations/flashvsr/tests/test_projector_benchmark.py \\
        -m "manual and slow" -v -s
"""

from __future__ import annotations

import statistics
import time

import pytest
import torch
from flashvsr.config import build_flashvsr_v1_1
from flashvsr.pipeline import FlashVSRPipeline
from flashvsr.transformer import FlashVSRTransformer

pytestmark = pytest.mark.manual

# (cold first-chunk frames, steady frame count) keyed by chunk_size.
# Mirrors the legacy ``_CHUNK_TARGET = {5: 8, 13: 16, 8: 8, 16: 16}``.
_CHUNK_MODES: dict[int, tuple[int, int]] = {16: (13, 16), 8: (5, 8)}

_GPU_REASON = "FlashVSR projector benchmark requires CUDA"


class _StageTimer:
    """Accumulates GPU elapsed time for a named stage across many calls.

    Records a ``cuda.Event`` pair around each wrapped invocation; totals
    are summed after a ``cuda.synchronize()``.
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self.events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []

    def wrap(self, fn):
        def timed(*args, **kwargs):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            out = fn(*args, **kwargs)
            end.record()
            self.events.append((start, end))
            return out

        return timed

    def total_ms(self) -> float:
        return sum(s.elapsed_time(e) for s, e in self.events)

    def reset(self) -> None:
        self.events.clear()


def _instrument(pipeline: FlashVSRPipeline) -> dict[str, _StageTimer]:
    """Wrap the per-stage forward methods with ``cuda.Event`` timers.

    Patching the bound methods (rather than the classes) keeps the
    instrumentation scoped to this benchmark run. The projector dispatch
    lives on ``pipeline.encoder.projector.forward_streaming``; the DiT
    forward lives on ``pipeline.diffusion_model.transformer.network``;
    the TC decoder + color corrector live on ``pipeline.decoder``.
    """
    encoder = pipeline.encoder
    decoder = pipeline.decoder
    transformer = pipeline.diffusion_model.transformer
    assert encoder is not None and decoder is not None, (
        "FlashVSRPipeline must be built with both encoder and decoder for this benchmark"
    )
    # Narrow ``transformer`` from the abstract ``Wan21Transformer`` to the
    # concrete subclass so ``transformer.network`` resolves to
    # ``FlashVSRDiTNetwork`` (with a typed ``forward``) instead of going
    # through ``nn.Module.__getattr__``.
    assert isinstance(transformer, FlashVSRTransformer)

    timers = {
        "projector": _StageTimer("projector"),
        "dit": _StageTimer("dit"),
        "decoder": _StageTimer("decoder"),
        "color": _StageTimer("color"),
    }

    encoder.projector.forward_streaming = timers["projector"].wrap(  # type: ignore[method-assign]
        encoder.projector.forward_streaming
    )
    transformer.network.forward = timers["dit"].wrap(transformer.network.forward)  # type: ignore[method-assign]
    decoder.tcdecoder.forward = timers["decoder"].wrap(decoder.tcdecoder.forward)  # type: ignore[method-assign]
    decoder.color_corrector.forward = timers["color"].wrap(  # type: ignore[method-assign]
        decoder.color_corrector.forward
    )
    return timers


@pytest.mark.skipif(not torch.cuda.is_available(), reason=_GPU_REASON)
@pytest.mark.parametrize("chunk_size", [16])
def test_flashvsr_per_stage_breakdown(chunk_size: int) -> None:
    """Report median per-stage wall time over the steady-replay window.

    Runs one cold chunk + 10 steady chunks, discards the first 3 steady
    chunks (which absorb wrapper warmup + capture), and prints the
    projector / DiT / decoder / color median ms for the remainder.
    Checkpoints flow through ``build_flashvsr_v1_1(...).setup()``, which
    routes the URLs in :data:`AVAILABLE_FLASHVSR_CHECKPOINT_PATHS`
    through :func:`flashdreams.core.checkpoint.load.load_checkpoint`
    (i.e. the ``~/.cache/huggingface/hub/`` cache) -- a cached run or
    network access is required.
    """
    n_steady = 10
    n_warmup_skip = 3
    input_H, input_W = 384, 640
    scale = 2
    dtype = torch.bfloat16

    pipeline_config = build_flashvsr_v1_1(
        input_H=input_H,
        input_W=input_W,
        scale=scale,
        dtype=dtype,
    )
    pipeline = pipeline_config.setup().to(device="cuda")
    cache = pipeline.initialize_cache()
    # ``pipeline_config.setup()`` is typed as the abstract
    # :class:`StreamInferencePipeline`; we know it's a
    # :class:`FlashVSRPipeline` (asserted by the builder).
    assert isinstance(pipeline, FlashVSRPipeline)
    timers = _instrument(pipeline)

    first_size, subseq_size = _CHUNK_MODES[chunk_size]
    cold = torch.randn(
        (1, 3, first_size, input_H, input_W), device="cuda", dtype=dtype
    ).clamp_(-1, 1)
    steady = torch.randn(
        (1, 3, subseq_size, input_H, input_W), device="cuda", dtype=dtype
    ).clamp_(-1, 1)

    # Cold chunk: fills caches + drains autotune. Discard timing.
    out = pipeline.generate(autoregressive_index=0, cache=cache, input=cold)
    pipeline.finalize(autoregressive_index=0, cache=cache)
    torch.cuda.synchronize()
    del out
    for t in timers.values():
        t.reset()

    per_chunk_totals: list[float] = []
    per_chunk_breakdown: list[dict[str, float]] = []
    for i in range(n_steady):
        for t in timers.values():
            t.reset()
        torch.cuda.synchronize()
        wall_start = time.perf_counter()
        out = pipeline.generate(autoregressive_index=1 + i, cache=cache, input=steady)
        pipeline.finalize(autoregressive_index=1 + i, cache=cache)
        torch.cuda.synchronize()
        wall_end = time.perf_counter()
        del out

        breakdown = {name: t.total_ms() for name, t in timers.items()}
        total_ms = (wall_end - wall_start) * 1000.0
        breakdown["other"] = max(0.0, total_ms - sum(breakdown.values()))
        per_chunk_totals.append(total_ms)
        per_chunk_breakdown.append(breakdown)
        print(
            f"  steady chunk {i + 1}: total={total_ms:.2f}ms "
            + " ".join(f"{k}={v:.2f}" for k, v in breakdown.items())
        )

    median_window = per_chunk_totals[n_warmup_skip:]
    median_breakdowns = per_chunk_breakdown[n_warmup_skip:]
    n_kept = len(median_window)
    assert n_kept > 0, (
        f"no steady samples kept (n_steady={n_steady}, n_warmup_skip={n_warmup_skip})"
    )
    median_total = statistics.median(median_window)
    print(
        f"\n=== Median over {n_kept} chunks (skipped first "
        f"{n_warmup_skip} of {n_steady}) ==="
    )
    print(f"total: {median_total:.2f}ms")
    for name in ("projector", "dit", "decoder", "color", "other"):
        median_stage = statistics.median(b[name] for b in median_breakdowns)
        share = 100.0 * median_stage / median_total if median_total > 0 else 0.0
        print(f"  {name:10s}: {median_stage:7.2f}ms  ({share:5.1f}%)")

    assert median_total > 0
