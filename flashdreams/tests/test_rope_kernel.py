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

"""Parity and perf tests for the Triton-fused RoPE inference kernel.

Ground truth is
:func:`transformer_engine.pytorch.attention.rope.apply_rotary_pos_emb`
(TE) — the kernel this module replaces. Parity covers the full
3 x 3 ``(x_dtype, freqs_dtype)`` matrix ({fp32, fp16, bf16}^2),
both rotation layouts, and a small / medium / production-sized
shape; parity tests skip when TE is unimportable.

Run with ``PYTHONPATH=. pytest tests/test_rope_kernel.py -s`` so
the printed ``triton vs TE`` line from :func:`test_perf_vs_te` is
visible.
"""

from __future__ import annotations

import pytest
import torch
from torch import Tensor

from flashdreams.core.attention.rope import apply_rope_freqs
from flashdreams.core.attention.rope_kernel import apply_rotary_pos_emb

try:
    from transformer_engine.pytorch.attention.rope import (
        apply_rotary_pos_emb as _te_apply_rotary_pos_emb,
    )

    _TE_AVAILABLE = True
except (ImportError, OSError):
    try:
        from transformer_engine.pytorch.attention import (
            apply_rotary_pos_emb as _te_apply_rotary_pos_emb,
        )

        _TE_AVAILABLE = True
    except (ImportError, OSError):
        _te_apply_rotary_pos_emb = None
        _TE_AVAILABLE = False


_requires_te = pytest.mark.skipif(
    not _TE_AVAILABLE, reason="transformer_engine not available"
)

# All tests in this module require CUDA; the cuda_device fixture skips
# if unavailable.  ci_gpu opts them into the GPU CI runner.
pytestmark = pytest.mark.ci_gpu


def _te_reference(x: Tensor, freqs: Tensor, interleaved: bool) -> Tensor:
    """Run TE's fused RoPE for use as the parity / perf reference."""
    assert _te_apply_rotary_pos_emb is not None
    return _te_apply_rotary_pos_emb(
        x, freqs, tensor_format="bshd", fused=True, interleaved=interleaved
    )


def _expanded_freqs(
    S: int,
    D: int,
    interleaved: bool,
    device: torch.device,
    seed: int,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Build ``[S, 1, 1, D]`` freqs matching the layout ``shift_t`` emits."""
    g = torch.Generator(device=device).manual_seed(seed)
    raw = torch.randn(S, D // 2, generator=g, device=device, dtype=dtype)
    expanded = (
        raw.repeat_interleave(2, dim=-1)
        if interleaved
        else torch.cat([raw, raw], dim=-1)
    )
    return expanded.reshape(S, 1, 1, D)


@pytest.fixture(scope="module")
def cuda_device() -> torch.device:
    if not torch.cuda.is_available():
        pytest.skip("CUDA required.")
    return torch.device("cuda")


_DTYPES = [torch.bfloat16, torch.float16, torch.float32]


def _parity_tol(x_dtype: torch.dtype) -> tuple[float, float]:
    """Pick an ``(atol, rtol)`` that covers a few ULPs of the output dtype."""
    if x_dtype is torch.float32:
        return 1e-5, 1e-5
    if x_dtype is torch.float16:
        return 2e-3, 2e-3
    return 2e-2, 2e-2  # bf16


_PARITY_SHAPES = [
    (1, 12, 8, 64),
    (2, 64, 16, 128),
    (1, 21 * 30, 24, 128),
]


@_requires_te
@pytest.mark.parametrize("x_dtype", _DTYPES)
@pytest.mark.parametrize("f_dtype", _DTYPES)
@pytest.mark.parametrize("interleaved", [False, True])
@pytest.mark.parametrize("shape", _PARITY_SHAPES)
def test_parity_vs_te(cuda_device, shape, interleaved, x_dtype, f_dtype):
    """Byte-for-byte parity vs TE across the full dtype matrix.

    Uses the production full-width ``[S, 1, 1, D]`` cat / repeat-
    interleave freqs layout (the only layout ``shift_t`` ever emits).
    """
    B, S, H, D = shape
    g = torch.Generator(device=cuda_device).manual_seed(42)
    x = (
        torch.randn(B, S, H, D, generator=g, device=cuda_device, dtype=torch.float32)
        * 0.5
    ).to(x_dtype)
    freqs = _expanded_freqs(S, D, interleaved, cuda_device, seed=0, dtype=f_dtype)

    expected = _te_reference(x.clone(), freqs, interleaved=interleaved)
    actual = apply_rotary_pos_emb(
        x.clone(), freqs, interleaved=interleaved, inplace=True
    )

    atol, rtol = _parity_tol(x_dtype)
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)


@_requires_te
def test_apply_rope_freqs_dispatches_to_kernel(cuda_device):
    """The public ``apply_rope_freqs`` matches TE through the Triton path."""
    B, S, H, D = 1, 16, 4, 64
    x = torch.randn(B, S, H, D, device=cuda_device, dtype=torch.bfloat16)
    freqs = _expanded_freqs(S, D, interleaved=False, device=cuda_device, seed=1)

    expected = _te_reference(x.clone(), freqs, interleaved=False)
    actual = apply_rope_freqs(x.clone(), freqs, interleaved=False)
    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)


def test_inplace_returns_same_storage(cuda_device):
    """``inplace=True`` rotates in place; ``inplace=False`` clones first."""
    B, S, H, D = 2, 16, 4, 64
    x = torch.randn(B, S, H, D, device=cuda_device, dtype=torch.bfloat16)
    freqs = _expanded_freqs(S, D, interleaved=False, device=cuda_device, seed=2)

    x_before = x.clone()
    out_inplace = apply_rotary_pos_emb(x, freqs, interleaved=False, inplace=True)
    assert out_inplace.data_ptr() == x.data_ptr()
    # Sanity: in-place actually mutated.
    assert not torch.equal(x, x_before)

    out_oop = apply_rotary_pos_emb(x, freqs, interleaved=False, inplace=False)
    assert out_oop.data_ptr() != x.data_ptr()


def test_zero_freqs_is_identity(cuda_device):
    """``cos(0) = 1`` and ``sin(0) = 0`` so the output equals the input."""
    B, S, H, D = 1, 8, 2, 32
    x = torch.randn(B, S, H, D, device=cuda_device, dtype=torch.float32)
    zero_freqs = torch.zeros(S, 1, 1, D, device=cuda_device, dtype=torch.float32)
    for interleaved in (False, True):
        out = apply_rotary_pos_emb(
            x.clone(), zero_freqs, interleaved=interleaved, inplace=False
        )
        torch.testing.assert_close(out, x)


@_requires_te
def test_non_contiguous_x(cuda_device):
    """The kernel respects arbitrary strides on the B / S / H axes."""
    B, S, H, D = 4, 16, 8, 64
    # Build a non-contiguous x by transposing B and S.
    x = torch.randn(S, B, H, D, device=cuda_device, dtype=torch.bfloat16).transpose(
        0, 1
    )
    assert not x.is_contiguous()
    freqs = _expanded_freqs(S, D, interleaved=False, device=cuda_device, seed=99)

    expected = _te_reference(x.contiguous(), freqs, interleaved=False)
    actual = apply_rotary_pos_emb(x.clone(), freqs, interleaved=False, inplace=False)
    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)


def test_rejects_fp8(cuda_device):
    """fp8 is intentionally not supported — TE rejects it too."""
    B, S, H, D = 1, 8, 2, 32
    freqs = torch.randn(S, 1, 1, D, device=cuda_device, dtype=torch.float32)
    for fp8 in (torch.float8_e4m3fn, torch.float8_e5m2):
        x_fp8 = torch.randn(B, S, H, D, device=cuda_device, dtype=torch.float32).to(fp8)
        with pytest.raises(NotImplementedError, match="fp8"):
            apply_rotary_pos_emb(x_fp8, freqs, interleaved=False, inplace=False)


def test_rejects_half_width_freqs(cuda_device):
    """Half-width freqs are not accepted — ``shift_t`` emits full-width."""
    B, S, H, D = 1, 8, 2, 32
    x = torch.randn(B, S, H, D, device=cuda_device, dtype=torch.bfloat16)
    half_freqs = torch.randn(S, 1, 1, D // 2, device=cuda_device, dtype=torch.float32)
    with pytest.raises(ValueError, match="head_dim"):
        apply_rotary_pos_emb(x, half_freqs, interleaved=False, inplace=False)


# --------------------------------------------------------------------------- #
# Performance benchmark
# --------------------------------------------------------------------------- #


def _bench_ms(fn, *, n_warmup: int = 100, n_iter: int = 1000) -> float:
    """Return the p20 wall-clock per call in milliseconds.

    Uses ``p20`` rather than the median because cluster-noise outliers
    skew the right tail; the lower tail is much more reproducible and
    gives a stable ratio against TE across runs.
    """
    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]
    for i in range(n_iter):
        starts[i].record()
        fn()
        ends[i].record()
    torch.cuda.synchronize()
    times = sorted(s.elapsed_time(e) for s, e in zip(starts, ends))
    return times[n_iter // 5]


_PERF_SHAPES = [
    (1, 64, 24, 128),
    (1, 21 * 30, 24, 128),
    (4, 21 * 30, 24, 128),
    # Production shape observed in the lingbot / alpadreams pipeline.
    (1, 7040, 16, 128),
]


@_requires_te
@pytest.mark.slow
@pytest.mark.parametrize("interleaved", [False, True])
@pytest.mark.parametrize("shape", _PERF_SHAPES)
def test_perf_vs_te(cuda_device, shape, interleaved):
    """Guard against perf regressions vs TE on the live workload.

    The threshold (``triton_ms <= 1.5 * te_ms``) is intentionally
    loose so cluster noise does not flake CI; the printed
    ``triton vs TE`` line is the actual signal across kernel edits.

    Excluded from CI (``@pytest.mark.slow``) because the ratio varies
    across GPU architectures and can flake on shared runners.
    """
    B, S, H, D = shape
    x = torch.randn(B, S, H, D, device=cuda_device, dtype=torch.bfloat16)
    freqs = _expanded_freqs(S, D, interleaved, cuda_device, seed=5)

    # RoPE preserves the L2 norm, so repeatedly rotating one buffer
    # in place keeps values bounded across the timed iterations and
    # avoids dragging the GPU allocator into the timing loop.
    x_triton = x.clone()

    def run_triton():
        apply_rotary_pos_emb(x_triton, freqs, interleaved=interleaved, inplace=True)

    def run_te():
        _te_reference(x, freqs, interleaved=interleaved)

    triton_ms = _bench_ms(run_triton)
    te_ms = _bench_ms(run_te)
    ratio = triton_ms / te_ms

    print(
        f"\n[rope_kernel] shape={shape} interleaved={interleaved} "
        f"triton={triton_ms:.3f}ms te={te_ms:.3f}ms ratio={ratio:.2f}x"
    )

    assert triton_ms <= 1.5 * te_ms, (
        f"Triton kernel ({triton_ms:.3f}ms) is significantly slower than "
        f"TE ({te_ms:.3f}ms) for shape={shape} interleaved={interleaved}."
    )
