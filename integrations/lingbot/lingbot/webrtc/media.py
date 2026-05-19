# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compatibility exports for shared FlashDreams WebRTC media helpers."""

from __future__ import annotations

from flashdreams.serving.webrtc.media import (
    BufferedVideoTrack as LingbotVideoTrack,
)
from flashdreams.serving.webrtc.media import (
    tensor_chunk_to_rgb_frames,
)

__all__ = ["LingbotVideoTrack", "tensor_chunk_to_rgb_frames"]
