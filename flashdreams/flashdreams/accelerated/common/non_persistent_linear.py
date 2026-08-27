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

"""Nonpersistent linear transformation for accelerated inference."""

from torch import Tensor, nn


class NonPersistentLinear(nn.Linear):
    """Linear transformation backed by nonpersistent weight and bias buffers."""

    def __init__(self, weight: Tensor, bias: Tensor | None) -> None:
        """Initialize the transformation from existing tensors.

        Args:
            weight: Projection weight shaped ``[out_features, in_features]``.
            bias: Optional projection bias shaped ``[out_features]``.
        """
        nn.Module.__init__(self)
        self.in_features = weight.shape[1]
        self.out_features = weight.shape[0]
        self.register_buffer("weight", weight, persistent=False)
        self.register_buffer("bias", bias, persistent=False)
