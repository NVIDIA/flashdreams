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

"""CUDA smoke tests for SANA-WM low-precision linear replacements."""

from __future__ import annotations

import pytest
import torch
from sana_wm.quant import TorchScaledMMFP4Linear, TorchScaledMMFP8Linear

pytestmark = pytest.mark.ci_gpu


def _require_quant_cuda() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for quantization smokes")
    missing = [
        name
        for name in ("_scaled_mm", "float8_e4m3fn", "float4_e2m1fn_x2")
        if not hasattr(torch, name)
    ]
    if missing:
        pytest.skip(f"PyTorch lacks quantization primitive(s): {missing}")


def test_fp8_linear_runs_scaled_mm_on_cuda() -> None:
    """Exercise the TE-free E4M3 FP8 Linear replacement on CUDA."""
    _require_quant_cuda()
    source = torch.nn.Linear(32, 64, bias=True, device="cuda", dtype=torch.bfloat16)
    quantized = TorchScaledMMFP8Linear.from_linear(
        source,
        out_dtype=torch.bfloat16,
    )
    inputs = torch.randn(4, 5, 32, device="cuda", dtype=torch.bfloat16)

    output = quantized(inputs)

    assert quantized.weight.shape == source.weight.shape
    assert quantized.bias is not None
    assert output.shape == (4, 5, 64)
    assert output.dtype == torch.bfloat16
    assert torch.isfinite(output.float()).all()


def test_fp4_linear_runs_scaled_mm_on_blackwell() -> None:
    """Exercise the TE-free E2M1 NVFP4 Linear replacement on Blackwell CUDA."""
    _require_quant_cuda()
    major, minor = torch.cuda.get_device_capability()
    if major < 10:
        pytest.skip(f"NVFP4 requires Blackwell-class CUDA, got sm_{major}{minor}")

    source = torch.nn.Linear(32, 64, bias=True, device="cuda", dtype=torch.bfloat16)
    quantized = TorchScaledMMFP4Linear.from_linear(
        source,
        out_dtype=torch.bfloat16,
    )
    inputs = torch.randn(4, 5, 32, device="cuda", dtype=torch.bfloat16)

    output = quantized(inputs)

    assert quantized.weight.shape == source.weight.shape
    assert quantized.bias is not None
    assert output.shape == (4, 5, 64)
    assert output.dtype == torch.bfloat16
    assert torch.isfinite(output.float()).all()


def test_quantized_linears_roughly_track_source_layer() -> None:
    """Catch gross scale/layout regressions beyond expected FP4/FP8 noise."""
    _require_quant_cuda()
    major, minor = torch.cuda.get_device_capability()
    if major < 10:
        pytest.skip(f"NVFP4 requires Blackwell-class CUDA, got sm_{major}{minor}")

    torch.manual_seed(123)
    source = torch.nn.Linear(32, 64, bias=True, device="cuda", dtype=torch.bfloat16)
    inputs = torch.randn(4, 5, 32, device="cuda", dtype=torch.bfloat16)
    reference = source(inputs).float()

    fp8 = TorchScaledMMFP8Linear.from_linear(
        source,
        out_dtype=torch.bfloat16,
    )
    fp4 = TorchScaledMMFP4Linear.from_linear(
        source,
        out_dtype=torch.bfloat16,
    )

    fp8_error = (fp8(inputs).float() - reference).abs()
    fp4_error = (fp4(inputs).float() - reference).abs()
    fp8_rel_mae = fp8_error.mean() / reference.abs().mean().clamp_min(1e-6)
    fp4_rel_mae = fp4_error.mean() / reference.abs().mean().clamp_min(1e-6)

    assert fp8_error.max().item() <= 0.10
    assert fp8_rel_mae.item() <= 0.06
    assert fp4_error.max().item() <= 0.35
    assert fp4_rel_mae.item() <= 0.20
