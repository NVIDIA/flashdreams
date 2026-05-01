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

"""``torch.compile`` helper that preserves the wrapped module's static type."""

from __future__ import annotations

from typing import Literal, TypeVar, cast

import torch
import torch.nn as nn

M = TypeVar("M", bound=nn.Module)

CompileMode = Literal[
    "default",
    "reduce-overhead",
    "max-autotune",
    "max-autotune-no-cudagraphs",
]
"""Valid ``mode`` values accepted by ``torch.compile``.

- ``default``: balanced; no Inductor autotune.
- ``reduce-overhead``: Inductor with CUDA-graph capture; lower per-call overhead.
- ``max-autotune``: full Inductor autotune + CUDA graphs.
- ``max-autotune-no-cudagraphs``: full Inductor autotune, skip CUDA graphs
  (use this when the caller wraps the result in its own
  :class:`flashdreams.infra.cuda_graph.CUDAGraphWrapper`).
"""


def compile_module(
    module: M,
    *,
    mode: CompileMode = "max-autotune-no-cudagraphs",
) -> M:
    """``torch.compile`` returning the same static type as ``module``.

    ``torch.compile`` wraps a module in an ``OptimizedModule`` proxy whose
    forward signature mirrors the wrapped module; the static type widens.
    This helper hides that single cast at one site so callers stay clean.

    Args:
        module: ``nn.Module`` to compile.
        mode: One of the four ``torch.compile`` modes; see :data:`CompileMode`.

    Returns:
        The compiled module, statically typed as the same ``M`` so attribute
        access on the wrapped module continues to type-check at call sites.
    """
    return cast(M, torch.compile(module, mode=mode))
