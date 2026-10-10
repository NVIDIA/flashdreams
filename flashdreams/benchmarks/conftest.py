# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared environment metadata for manual GPU benchmarks."""

from __future__ import annotations

from functools import cache

import pytest
import torch
import triton
from torch.utils.collect_env import get_nvidia_driver_version, run


@cache
def _gpu_environment() -> dict[str, object]:
    """Return the active GPU software and hardware environment."""
    if not torch.cuda.is_available():
        return {}

    device = torch.cuda.current_device()
    capability = torch.cuda.get_device_capability(device)
    return {
        "gpu": torch.cuda.get_device_name(device),
        "device_index": device,
        "device_capability": ".".join(map(str, capability)),
        "driver_version": get_nvidia_driver_version(run),
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "triton": triton.__version__,
    }


@pytest.fixture(autouse=True)
def _record_gpu_environment(request: pytest.FixtureRequest) -> None:
    """Attach the shared GPU environment to every benchmark record."""
    if "benchmark" not in request.fixturenames:
        return
    benchmark = request.getfixturevalue("benchmark")
    benchmark.extra_info.update(_gpu_environment())
