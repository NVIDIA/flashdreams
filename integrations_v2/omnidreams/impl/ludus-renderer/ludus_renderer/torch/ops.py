# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Backward compatibility layer for ``ludus_renderer.torch.ops`` imports.

New code should import directly from ``ludus_renderer``::

    from ludus_renderer import LudusCudaTimestampedContext, CAMERA_TYPE_REGULAR

This module re-exports the same symbols for backward compatibility::

    from ludus_renderer.torch.ops import CAMERA_TYPE_REGULAR  # Still works
"""

# Re-export everything from _ops
from .._ops import *  # noqa: F401,F403
