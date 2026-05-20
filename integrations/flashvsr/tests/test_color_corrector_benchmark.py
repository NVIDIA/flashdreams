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

"""AdaIN color-corrector benchmark + numerical parity test.

For each parametrized ``(dtype, shape)`` row, builds a torch-backend
reference and a cuda-backend candidate corrector, times both with the
standard ``cuda.Event`` warmup / iters protocol, asserts ``max_abs``
against a dtype-appropriate tolerance, and prints the wall time and
speedup.

Marked ``manual`` + ``slow``: the first run compiles the CUDA
extension (~10s) and each parametrized row does ``warmup + iters``
GPU launches at the configured shape. Run with::

    uv run pytest \\
        integrations/flashvsr/tests/test_color_corrector_benchmark.py \\
        -m "manual and slow" -v -s
"""

from __future__ import annotations

from typing import Callable

import pytest
import torch
from flashvsr import corrector as cc_module

pytestmark = pytest.mark.manual

_GPU_REASON = "color corrector benchmark requires CUDA"


def _time_cuda(
    fn: Callable[[], torch.Tensor], warmup: int, iters: int
) -> tuple[float, torch.Tensor]:
    """Run ``fn`` ``warmup`` times then ``iters`` timed iterations.

    Returns ``(per_iter_ms, last_output)``. Mirrors the timing harness
    from the legacy ``benchmark_color_corrector.py``.
    """
    if iters < 1:
        raise ValueError("iters must be >= 1")
    out = fn()
    for _ in range(max(warmup - 1, 0)):
        out = fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        out = fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters, out


# (dtype, frames, height, width, max_abs_tol). The 1536 x 2560 row is
# the production-ish 2x-of-FlashVSR-720p target; the 128 x 128 row is the
# legacy smoke shape. ``max_abs_tol`` is dtype-driven: bf16 / fp16 carry
# ~1e-2 noise from the AdaIN reduction, fp32 should match much tighter.
_BENCH_CASES = [
    pytest.param(torch.float32, 1, 128, 128, 1e-4, id="fp32-1f-128x128-smoke"),
    pytest.param(torch.bfloat16, 8, 1536, 2560, 5e-2, id="bf16-8f-1536x2560-prod"),
    pytest.param(torch.bfloat16, 16, 1536, 2560, 5e-2, id="bf16-16f-1536x2560-prod"),
]


@pytest.mark.skipif(not torch.cuda.is_available(), reason=_GPU_REASON)
@pytest.mark.parametrize(
    ("dtype", "frames", "height", "width", "max_abs_tol"), _BENCH_CASES
)
def test_color_corrector_cuda_vs_torch(
    dtype: torch.dtype,
    frames: int,
    height: int,
    width: int,
    max_abs_tol: float,
) -> None:
    """CUDA AdaIN matches the torch reference within tolerance + reports speedup.

    ``warmup`` and ``iters`` are fixed (5 / 20) so the test runs
    deterministically; the legacy CLI exposed them as flags but the
    pytest variant doesn't need that knob.
    """
    if dtype is torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("GPU does not support bfloat16")

    reference_corrector = cc_module.FlashVSRColorCorrector(
        levels=5, implementation="torch"
    ).cuda()
    test_corrector = cc_module.FlashVSRColorCorrector(
        levels=5, implementation="cuda"
    ).cuda()

    native_cuda_available = cc_module._load_adain_cuda_extension() is not None
    if not native_cuda_available:
        err = cc_module._ADAIN_CUDA_EXTENSION_LOAD_ERROR
        pytest.skip(f"native CUDA AdaIN extension unavailable: {err}")

    shape = (1, 3, frames, height, width)
    hq = torch.randn(shape, device="cuda", dtype=dtype)
    lq = torch.randn(shape, device="cuda", dtype=dtype)

    def torch_path() -> torch.Tensor:
        return reference_corrector(
            hq, lq, clip_range=(-1.0, 1.0), method="adain", chunk_size=None
        )

    def cuda_path() -> torch.Tensor:
        return test_corrector(
            hq, lq, clip_range=(-1.0, 1.0), method="adain", chunk_size=None
        )

    torch_ms, torch_out = _time_cuda(torch_path, warmup=5, iters=20)
    cuda_ms, cuda_out = _time_cuda(cuda_path, warmup=5, iters=20)

    diff = (cuda_out.float() - torch_out.float()).abs()
    max_abs = diff.max().item()
    mean_abs = diff.mean().item()
    speedup = torch_ms / cuda_ms if cuda_ms > 0 else float("inf")
    print(
        f"shape={shape} dtype={dtype} torch_ms={torch_ms:.3f} "
        f"cuda_ms={cuda_ms:.3f} speedup={speedup:.2f}x "
        f"max_abs={max_abs:.6g} mean_abs={mean_abs:.6g}"
    )

    assert max_abs <= max_abs_tol, (
        f"CUDA AdaIN diverged from torch reference: "
        f"max_abs={max_abs:.6g} > tol {max_abs_tol:.6g}"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason=_GPU_REASON)
def test_color_corrector_cuda_extension_loads() -> None:
    """The renamed CUDA extension (item 8) loads + reports device caps."""
    ext = cc_module._load_adain_cuda_extension()
    if ext is None:
        err = cc_module._ADAIN_CUDA_EXTENSION_LOAD_ERROR
        pytest.skip(f"native CUDA AdaIN extension unavailable: {err}")

    caps = cc_module.adain_cuda_caps()
    assert isinstance(caps, dict) and caps, "extension loaded but caps() returned empty"
    expected_keys = {
        "cooperative_launch",
        "persisting_l2",
        "num_sms",
        "access_policy_max_window_size",
        "persisting_l2_max_size",
    }
    assert expected_keys.issubset(caps.keys()), (
        f"caps missing expected keys: {expected_keys - caps.keys()}"
    )
    print(f"native_cuda_caps={caps}")
