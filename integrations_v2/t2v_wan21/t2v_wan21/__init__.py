# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Wan 2.1 text-to-video application for the FlashDreams v2 API."""

from .app import Wan21T2VApplication, create_app

__all__ = [
    "Wan21T2VApplication",
    "create_app",
]
