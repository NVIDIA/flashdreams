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

"""Shared pytest fixtures for LingBot benchmarks."""

import gc
from collections.abc import Iterator

import pytest
import torch


@pytest.fixture(autouse=True)
def _release_cuda_memory_between_benchmarks() -> Iterator[None]:
    """Release compiler and CUDA allocator state after each benchmark."""
    yield
    if not torch.cuda.is_available():
        return
    torch.cuda.synchronize()
    torch.compiler.reset()
    gc.collect()
    torch.cuda.empty_cache()
