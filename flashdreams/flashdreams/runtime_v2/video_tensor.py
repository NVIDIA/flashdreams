# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Video tensor layout implementation."""

from enum import Enum


class VideoTensorLayout(Enum):
    """Supported/Recognized tensor layouts."""

    tchw = "tchw"
    btchw = "btchw"
    bcthw = "bcthw"
    bvtchw = "bvtchw"
