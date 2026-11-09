# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Control scaling gives the same values on the GPU as on the CPU."""

from __future__ import annotations

import pytest
import torch
from cosmos3_nano_sim_transfer.impl.control import to_model_range

pytestmark = [
    pytest.mark.ci_gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device"),
]


def test_model_range_is_the_same_on_the_gpu() -> None:
    levels = torch.arange(256, dtype=torch.uint8).float()
    on_cpu = to_model_range(levels)
    on_gpu = to_model_range(levels.cuda()).cpu()
    assert torch.equal(on_cpu, on_gpu)
    assert torch.equal(on_cpu.bfloat16(), on_gpu.bfloat16())
