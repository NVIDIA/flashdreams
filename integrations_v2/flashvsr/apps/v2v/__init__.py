# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FlashVSR binding for the reusable v2v application."""

from .adapter import (
    create_app,
    create_app_full_attn,
    create_app_sparse_ratio_1_5,
)

__all__ = [
    "create_app",
    "create_app_full_attn",
    "create_app_sparse_ratio_1_5",
]
