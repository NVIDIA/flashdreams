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

"""Attention metadata, packing, and position IDs for multi-view inference."""

from flashdreams.core.attention.kvcache import (
    FixedSlotKVCache,
    LayerKV,
    SlotRegion,
    TokenWindow,
)
from flashdreams.core.attention.multiview.mask import (
    ROLE_CLEAN_TARGET,
    ROLE_CONTROL,
    ROLE_CURRENT_TARGET,
    ROLE_PADDING,
    ROLE_TARGET_CONDITION,
    ROLE_UND,
    AttentionPattern,
    AttentionScope,
    StreamFields,
    build_block_mask,
    visibility,
    visibility_mask_mod,
)
from flashdreams.core.attention.multiview.packing import (
    ChunkMetadata,
    ChunkPassKind,
    ChunkPlan,
    ClipGeometry,
    MemoryLayout,
    ar_chunk_plan,
    ar_chunk_range,
    ar_chunk_ranges,
    build_chunk_metadata,
    build_memory_layout,
    causal_steps,
    control_chunk_ranges,
    control_window,
    pack_cross_view_attention,
    text_stream,
    unpack_cross_view_attention,
)
from flashdreams.core.attention.multiview.prefill import (
    PrefillPack,
    build_prefill_pack,
    pad_control_slots,
)
from flashdreams.core.attention.multiview.rollout import (
    ChunkMask,
    ChunkRollout,
    ChunkTrace,
    ControlSource,
    MultiViewRolloutModel,
    MultiViewRolloutState,
    Rollout,
    run_rollout,
)
from flashdreams.core.attention.multiview.rope import (
    BASE_FPS,
    MODALITY_MARGIN,
    caption_mrope_ids,
    chunk_mrope_ids,
    clip_mrope_ids,
    temporal_positions,
    vision_temporal_offset,
)

__all__ = [
    "BASE_FPS",
    "MODALITY_MARGIN",
    "ROLE_CLEAN_TARGET",
    "ROLE_CONTROL",
    "ROLE_CURRENT_TARGET",
    "ROLE_PADDING",
    "ROLE_TARGET_CONDITION",
    "ROLE_UND",
    "AttentionPattern",
    "AttentionScope",
    "ChunkMask",
    "ChunkMetadata",
    "ChunkPassKind",
    "ChunkPlan",
    "ChunkRollout",
    "ChunkTrace",
    "ClipGeometry",
    "ControlSource",
    "FixedSlotKVCache",
    "LayerKV",
    "MemoryLayout",
    "MultiViewRolloutModel",
    "MultiViewRolloutState",
    "PrefillPack",
    "Rollout",
    "SlotRegion",
    "StreamFields",
    "TokenWindow",
    "ar_chunk_plan",
    "ar_chunk_range",
    "ar_chunk_ranges",
    "build_block_mask",
    "build_chunk_metadata",
    "build_memory_layout",
    "build_prefill_pack",
    "caption_mrope_ids",
    "causal_steps",
    "chunk_mrope_ids",
    "clip_mrope_ids",
    "control_chunk_ranges",
    "control_window",
    "pack_cross_view_attention",
    "pad_control_slots",
    "run_rollout",
    "temporal_positions",
    "text_stream",
    "unpack_cross_view_attention",
    "visibility",
    "visibility_mask_mod",
    "vision_temporal_offset",
]
