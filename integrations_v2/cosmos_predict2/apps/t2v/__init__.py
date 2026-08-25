# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cosmos Predict2 adapter for the shared FlashDreams T2V application."""

from .adapter import create_app, load_config

__all__ = ["create_app", "load_config"]
