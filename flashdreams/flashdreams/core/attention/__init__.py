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

"""Attention primitives and KV cache for streaming inference."""

from flashdreams.core.attention.kvcache import BlockKVCache
from flashdreams.core.attention.native import NativeAttention
from flashdreams.core.attention.ring import RingAttention
from flashdreams.core.attention.rope import (
    KVCacheRelativeRotaryPositionEmbedding3D,
    RotaryPositionEmbedding3D,
    apply_rope_freqs,
)

__all__ = [
    "RotaryPositionEmbedding3D",
    "KVCacheRelativeRotaryPositionEmbedding3D",
    "BlockKVCache",
    "NativeAttention",
    "RingAttention",
    "apply_rope_freqs",
]
