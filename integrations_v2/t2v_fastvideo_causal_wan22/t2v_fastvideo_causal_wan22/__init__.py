# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FastVideo CausalWan 2.2 text-to-video application for the FlashDreams v2 API."""

from .app import FastvideoCausalWan22T2VApplication, create_app

__all__ = [
    "FastvideoCausalWan22T2VApplication",
    "create_app",
]
