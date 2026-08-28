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

"""PyTorch scaled-dot-product attention forced to cuDNN."""

import torch
import torch.nn.functional as F
from torch import Tensor


def torch_cudnn_sdpa(query: Tensor, key: Tensor, value: Tensor) -> Tensor:
    """Apply PyTorch scaled-dot-product attention with the cuDNN backend.

    Args:
        query: Queries in ``[B, H, L, D]`` layout.
        key: Keys in ``[B, H, S, D]`` layout.
        value: Values in ``[B, H, S, D]`` layout.

    Returns:
        Attention output shaped like ``query``.
    """
    with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.CUDNN_ATTENTION):
        return F.scaled_dot_product_attention(query, key, value)
