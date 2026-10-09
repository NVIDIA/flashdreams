# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the shared RMSNorm inference primitive."""

import pytest
import torch

from flashdreams.accelerated.common.rms_norm import rms_norm

pytestmark = pytest.mark.ci_cpu


def test_rms_norm_matches_pytorch_module() -> None:
    torch.manual_seed(0)
    eps = 1e-6
    module = torch.nn.RMSNorm(16, eps=eps)
    input = torch.randn(2, 7, 16)

    assert torch.allclose(rms_norm(input, module.weight, eps), module(input))


def test_rms_norm_rejects_a_mismatched_weight() -> None:
    with pytest.raises(ValueError, match="does not match"):
        rms_norm(torch.randn(2, 7, 16), torch.ones(8), 1e-6)
