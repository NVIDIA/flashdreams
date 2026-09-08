# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tensor layouts a generated video result can take."""

from enum import Enum


class VideoTensorLayout(Enum):
    """Axis order of a generated video tensor.

    ``b`` is batch, ``v`` view, ``t`` time, ``c`` colour channel, and ``h`` and
    ``w`` the frame. Presentation reads one frame at a time and so needs a batch
    of one, and one view, from the layouts that carry them.
    """

    tchw = "tchw"
    """Frames, channels, height, width. The default, and what the text-to-video
    models emit."""

    btchw = "btchw"
    """Batch of ``tchw``."""

    bcthw = "bcthw"
    """Batch of channel-major clips, as a diffusion pipeline usually holds them."""

    bvtchw = "bvtchw"
    """Batch of multi-view clips, one ``tchw`` per camera."""
