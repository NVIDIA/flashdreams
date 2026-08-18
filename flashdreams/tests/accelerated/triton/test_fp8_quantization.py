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

"""Reference tests for Triton row-wise FP8 quantization kernels."""

from __future__ import annotations

import pytest
import torch

from flashdreams.accelerated.triton import fp8_quantization

pytestmark = pytest.mark.ci_gpu


def test_fused_fp8_row_quantization_matches_torch(
    tma_device: torch.device,
) -> None:
    """Match fused row-wise E4M3 quantization against PyTorch reference math.

    Exercise a positively strided input, the minimum scale for an all-zero row,
    NaN propagation, and an empty row axis that must not launch a kernel.

    Args:
        tma_device: CUDA device satisfying the shared TMA capability gate.
    """
    generator = torch.Generator(device=tma_device).manual_seed(5)
    # Transpose an owned ``[1536, 7]`` allocation to exercise a non-contiguous
    # ``[7, 1536]`` view, then reserve row zero for the scale-floor check.
    x = torch.randn(
        (1536, 7),
        generator=generator,
        device=tma_device,
        dtype=torch.bfloat16,
    ).T
    x[0].zero_()
    actual, actual_scales = fp8_quantization._quantize_fp8_rows(x)

    # Reproduce the kernel's per-row scale and normalized E4M3 cast in FP32.
    x_float = x.to(torch.float32)
    expected_scales = (
        (x_float.abs().amax(dim=1, keepdim=True) / fp8_quantization._FP8_MAX)
        .clamp_min(1e-12)
        .contiguous()
    )
    expected = (
        (x_float / expected_scales)
        .clamp(-fp8_quantization._FP8_MAX, fp8_quantization._FP8_MAX)
        .to(torch.float8_e4m3fn)
    )

    # Exact arithmetic parity is expected, and contiguous output is part of the
    # wrapper contract even when the source view is strided.
    assert actual.is_contiguous()
    assert torch.equal(actual, expected)
    assert torch.equal(actual_scales, expected_scales)
    assert torch.count_nonzero(actual[0]) == 0
    assert actual_scales[0, 0] == 1e-12

    # Pin IEEE NaN propagation separately from finite-row equality.
    nan_input = x.clone()
    nan_input[0, 0] = float("nan")
    nan_output, _ = fp8_quantization._quantize_fp8_rows(nan_input)
    assert torch.isnan(nan_output[0, 0].to(torch.float32))

    # An empty row axis returns correctly shaped storage without a Triton launch.
    empty_output, empty_scales = fp8_quantization._quantize_fp8_rows(x[:0])
    assert empty_output.shape == x[:0].shape
    assert empty_scales.shape == (0, 1)
