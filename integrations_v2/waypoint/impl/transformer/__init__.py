# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native Waypoint DiT topology and conditioning primitives."""

from waypoint.impl.transformer.cache import (
    WaypointAttentionPolicy,
    WaypointKVCache,
    WaypointKVView,
)
from waypoint.impl.transformer.impl import (
    WaypointTransformer,
    WaypointTransformerCache,
    WaypointTransformerConfig,
)
from waypoint.impl.transformer.network import (
    WaypointDiT,
    WaypointDiTConfig,
    sinusoidal_noise_embedding,
)
from waypoint.impl.transformer.norm import adaptive_gate, adaptive_rms_norm
from waypoint.impl.transformer.rope import (
    WaypointOrthoRoPEAngles,
    apply_waypoint_ortho_rope,
)

__all__ = [
    "WaypointDiT",
    "WaypointDiTConfig",
    "WaypointAttentionPolicy",
    "WaypointKVCache",
    "WaypointKVView",
    "WaypointTransformer",
    "WaypointTransformerCache",
    "WaypointTransformerConfig",
    "WaypointOrthoRoPEAngles",
    "adaptive_gate",
    "adaptive_rms_norm",
    "apply_waypoint_ortho_rope",
    "sinusoidal_noise_embedding",
]
