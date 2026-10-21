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

"""PyTorch SDPA backend selection for AgiBot inference."""

from __future__ import annotations

from contextlib import AbstractContextManager, nullcontext

from torch.nn.attention import SDPBackend, sdpa_kernel

SDPA_BACKENDS = ("default", "cudnn", "flash", "efficient", "math")


def normalize_sdpa_backend(value: str | None) -> str:
    """Return the canonical SDPA backend name."""
    raw = (value or "default").strip().lower().replace("-", "_")
    aliases = {
        "none": "default",
        "auto": "default",
        "mem_efficient": "efficient",
        "memory_efficient": "efficient",
        "efficient_attention": "efficient",
        "flash_attention": "flash",
        "cudnn_attention": "cudnn",
    }
    backend = aliases.get(raw, raw)
    if backend not in SDPA_BACKENDS:
        allowed = ", ".join(SDPA_BACKENDS)
        raise ValueError(f"SDPA backend must be one of: {allowed}; got {value!r}")
    return backend


def sdpa_kernel_context(backend: str) -> AbstractContextManager[None]:
    """Constrain scaled-dot-product attention to one backend."""
    backend = normalize_sdpa_backend(backend)
    if backend == "default":
        return nullcontext()
    selected = {
        "cudnn": SDPBackend.CUDNN_ATTENTION,
        "flash": SDPBackend.FLASH_ATTENTION,
        "efficient": SDPBackend.EFFICIENT_ATTENTION,
        "math": SDPBackend.MATH,
    }[backend]
    return sdpa_kernel(selected)
