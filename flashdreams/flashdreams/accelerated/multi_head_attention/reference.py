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

"""Small explicit attention oracle for backend correctness tests."""

from __future__ import annotations

import torch
from torch import Tensor


def reference_masked_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    mask: Tensor,
) -> Tensor:
    """Compute masked attention explicitly, without an SDPA backend.

    Args:
        query: Queries shaped ``[B, H, Q, D]``.
        key: Keys shaped ``[B, H, K, D]``.
        value: Values shaped ``[B, H, K, Dv]``.
        mask: Boolean visibility mask shaped ``[Q, K]``.

    Returns:
        Attention values shaped ``[B, H, Q, Dv]``.

    Raises:
        ValueError: Q/K/V or mask geometry is inconsistent.
        TypeError: ``mask`` is not boolean.
    """
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("query, key, and value must have shape [B, H, S, D].")
    if query.shape[:2] != key.shape[:2] or key.shape[:3] != value.shape[:3]:
        raise ValueError("query, key, and value batch/head/token axes must agree.")
    expected = (query.shape[-2], key.shape[-2])
    if tuple(mask.shape) != expected:
        raise ValueError(f"mask shape must be {expected}; got {tuple(mask.shape)}.")
    if mask.dtype is not torch.bool:
        raise TypeError(f"mask must be boolean; got {mask.dtype}.")

    scores = torch.matmul(query, key.transpose(-1, -2))
    scores = scores * query.shape[-1] ** -0.5
    scores = scores.masked_fill(~mask[None, None], torch.finfo(scores.dtype).min)
    return torch.matmul(torch.softmax(scores, dim=-1), value)


__all__ = ["reference_masked_attention"]
