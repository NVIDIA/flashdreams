# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

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
