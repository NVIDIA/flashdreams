# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Minimal ``mmcv.utils`` namespace for SANA inference imports."""

from __future__ import annotations

import torch.nn as nn

_BatchNorm = nn.modules.batchnorm._BatchNorm
_InstanceNorm = nn.modules.instancenorm._InstanceNorm

__all__ = ["_BatchNorm", "_InstanceNorm"]
