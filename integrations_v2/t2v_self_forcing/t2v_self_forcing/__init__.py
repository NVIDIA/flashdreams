# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Self-Forcing text-to-video application for the FlashDreams v2 API."""

from .app import SelfForcingT2VApplication, create_app

__all__ = [
    "SelfForcingT2VApplication",
    "create_app",
]
