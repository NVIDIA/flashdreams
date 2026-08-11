# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FlashVSR demos built on the shared inference runtime API."""

from flashvsr.demo.adapter import FlashVSRDemoAdapter
from flashvsr.demo.providers import FlashVSRVideoInputProvider
from flashvsr.demo.spec import (
    DEFAULT_FLASHVSR_INPUT_URL,
    FlashVSRVideoScenario,
    PreparedFlashVSRVideo,
)
from flashvsr.runtime import (
    DEFAULT_FLASHVSR_PRESET,
    FLASHVSR_MODEL_ID,
)

__all__ = [
    "DEFAULT_FLASHVSR_INPUT_URL",
    "DEFAULT_FLASHVSR_PRESET",
    "FLASHVSR_MODEL_ID",
    "FlashVSRDemoAdapter",
    "FlashVSRVideoInputProvider",
    "FlashVSRVideoScenario",
    "PreparedFlashVSRVideo",
]
