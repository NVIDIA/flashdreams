# SPDX-FileCopyrightText: Copyright (c) 2026 Praneeth Samineni.
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

"""Opt-in NVTX ranges for Nsight Systems traces."""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager

import torch

_ENABLED = os.environ.get("FLASHDREAMS_NVTX") == "1" and torch.cuda.is_available()
"""Resolved once at import, so the variable must be set before FlashDreams loads.

``torch.cuda.nvtx.range_push`` raises on CPU-only builds, hence the CUDA check.
"""


@contextmanager
def nvtx_range(name: str) -> Iterator[None]:
    """Wrap the block in an NVTX range named ``name`` when profiling is enabled."""
    if not _ENABLED:
        yield
        return
    torch.cuda.nvtx.range_push(name)
    try:
        yield
    finally:
        torch.cuda.nvtx.range_pop()


__all__ = ["nvtx_range"]
