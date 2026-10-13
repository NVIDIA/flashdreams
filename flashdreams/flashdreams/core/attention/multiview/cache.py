# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compatibility imports for multi-view cache types now owned by core attention."""

from flashdreams.core.attention.kvcache import (
    FixedSlotKVCache,
    LayerKV,
    SlotRegion,
    TokenWindow,
)

__all__ = ["FixedSlotKVCache", "LayerKV", "SlotRegion", "TokenWindow"]
