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

"""Shared fixtures for Triton kernel tests."""

from __future__ import annotations

import pytest
import torch


@pytest.fixture(scope="module")
def tma_device() -> torch.device:
    """Provide a CUDA device capable of launching TMA kernels.

    Returns:
        Active CUDA device with compute capability 9.0 or newer.
    """
    # Gate the shared fixture once so every TMA test skips consistently on CPU
    # or pre-Hopper hosts instead of failing during kernel compilation.
    if not torch.cuda.is_available():
        pytest.skip("CUDA required.")
    device = torch.device("cuda")
    if torch.cuda.get_device_capability(device)[0] < 9:
        pytest.skip("TMA kernels require compute capability 9.0 or newer.")
    return device
